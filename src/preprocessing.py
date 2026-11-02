"""
Real Estate Investment Advisor - Data Preprocessing & Feature Engineering Pipeline
Handles data cleaning, binary encoding, scaling, domain feature generation, and target definitions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

TIER1_CITIES = {
    'Mumbai', 'Bangalore', 'Pune', 'Hyderabad', 'Gurgaon',
    'New Delhi', 'Chennai', 'Kolkata', 'Ahmedabad', 'Noida'
}

TIER2_CITIES = {
    'Jaipur', 'Lucknow', 'Kochi', 'Coimbatore', 'Ludhiana',
    'Indore', 'Bhubaneswar', 'Dehradun', 'Nagpur', 'Surat',
    'Vishakhapatnam', 'Faridabad', 'Mysore', 'Trivandrum',
    'Amritsar', 'Ranchi', 'Patna', 'Bhopal', 'Raipur', 'Vijayawada',
    'Dwarka', 'Cuttack', 'Guwahati', 'Gaya', 'Haridwar', 'Jodhpur',
    'Jamshedpur', 'Warangal', 'Bilaspur', 'Durgapur', 'Silchar', 'Mangalore'
}

ALL_AMENITIES = ['Clubhouse', 'Garden', 'Gym', 'Playground', 'Pool']

def load_raw_data(filepath='india_housing_prices.csv'):
    """Load the raw real estate housing dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filepath}' not found.")
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Handle missing values, duplicate records, and logical anomalies."""
    cleaning_report = {
        'initial_rows': len(df),
        'initial_duplicates': int(df.duplicated().sum()),
        'missing_values': {col: int(val) for col, val in df.isnull().sum().items() if val > 0}
    }
    
    # Remove duplicate rows if any
    df = df.drop_duplicates().copy()
    
    # Fill any potential nulls with conservative medians/modes
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Logical consistency: Total floors should be at least Floor_No
    df['Total_Floors'] = np.maximum(df['Total_Floors'], df['Floor_No'])
    
    # Clean string fields
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()
        
    cleaning_report['final_rows'] = len(df)
    return df, cleaning_report

def engineer_features(df):
    """
    Engineer domain-relevant features:
    - Real price per sqft in INR
    - Infrastructure & density indices
    - Amenity individual binary codes and counts
    - Building geometry ratios
    - City tier classifications
    """
    df = df.copy()
    
    # Exact Price per SqFt in INR (Price_in_Lakhs * 100,000 / Size_in_SqFt)
    df['Price_per_SqFt_INR'] = (df['Price_in_Lakhs'] * 100000.0) / df['Size_in_SqFt']
    
    # School and Hospital density scores (normalized 0 to 1)
    df['School_Density_Score'] = df['Nearby_Schools'] / 10.0
    df['Hospital_Density_Score'] = df['Nearby_Hospitals'] / 10.0
    
    # Public transport mapping
    transport_map = {'High': 3, 'Medium': 2, 'Low': 1}
    df['Transport_Score'] = df['Public_Transport_Accessibility'].map(transport_map).fillna(1)
    
    # Composite Infrastructure Score (Scale 0 to 10)
    # 50% social infrastructure (schools + hospitals), 50% public transit connectivity
    social_infra = ((df['Nearby_Schools'] + df['Nearby_Hospitals']) / 20.0) * 5.0
    transit_infra = (df['Transport_Score'] / 3.0) * 5.0
    df['Infrastructure_Score'] = social_infra + transit_infra
    
    # Parse Amenities: count and individual binary codes
    for amenity in ALL_AMENITIES:
        df[f'Amenity_{amenity}'] = df['Amenities'].apply(lambda x: 1 if amenity.lower() in str(x).lower() else 0)
        
    df['Amenities_Count'] = sum(df[f'Amenity_{amenity}'] for amenity in ALL_AMENITIES)
    
    # Building height and floor position ratios
    df['Floor_Ratio'] = np.where(df['Total_Floors'] > 0, df['Floor_No'] / df['Total_Floors'], 0.0)
    df['Floor_Ratio'] = np.clip(df['Floor_Ratio'], 0.0, 1.0)
    df['Is_Ground_Floor'] = (df['Floor_No'] == 0).astype(int)
    df['Is_Top_Floor'] = (df['Floor_No'] == df['Total_Floors']).astype(int)
    
    # City Tier classification
    def get_city_tier(city):
        if city in TIER1_CITIES:
            return 'Tier 1'
        elif city in TIER2_CITIES:
            return 'Tier 2'
        return 'Tier 3'
        
    df['City_Tier'] = df['City'].apply(get_city_tier)
    df['City_Tier_Code'] = df['City_Tier'].map({'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3})
    
    # City median price per sqft for relative valuation
    city_medians = df.groupby('City')['Price_per_SqFt_INR'].median().to_dict()
    df['City_Median_Price_SqFt'] = df['City'].map(city_medians)
    df['Valuation_Ratio'] = df['Price_per_SqFt_INR'] / df['City_Median_Price_SqFt']
    df['Is_Undervalued'] = (df['Valuation_Ratio'] < 1.0).astype(int)
    
    return df, city_medians

def calculate_investment_targets(df, city_medians):
    """
    Formulate the 5-Year Capital Appreciation Rate, Future Price, and Good Investment label:
    Compound Annual Growth Rate (CAGR) model grounded in Indian real estate market dynamics.
    """
    df = df.copy()
    
    def compute_row_cagr(row):
        city = row['City']
        if city in TIER1_CITIES:
            base_rate = 0.082
        elif city in TIER2_CITIES:
            base_rate = 0.072
        else:
            base_rate = 0.062
            
        # Connectivity adjustment
        trans_adj = {'High': 0.012, 'Medium': 0.005, 'Low': -0.003}.get(row['Public_Transport_Accessibility'], 0.0)
        
        # Social infrastructure adjustment
        social_adj = (row['Nearby_Schools'] / 10.0) * 0.007 + (row['Nearby_Hospitals'] / 10.0) * 0.007
        
        # Amenities premium
        amenity_adj = (row['Amenities_Count'] / 5.0) * 0.010
        
        # Security & Parking premiums
        sec_adj = 0.004 if row['Security'] == 'Yes' else 0.0
        park_adj = 0.005 if row['Parking_Space'] == 'Yes' else -0.003
        
        # Age & construction lifecycle
        age = row['Age_of_Property']
        if age <= 5 or row['Availability_Status'] == 'Under_Construction':
            age_adj = 0.010
        elif age <= 15:
            age_adj = 0.002
        elif age <= 25:
            age_adj = -0.005
        else:
            age_adj = -0.012
            
        # Facing orientation premium (Vastu preference)
        facing_adj = 0.003 if row['Facing'] in ['East', 'North'] else 0.0
        
        # Relative valuation upside (undervalued properties appreciate faster)
        val_ratio = row['Valuation_Ratio']
        if val_ratio < 0.90:
            val_adj = 0.008
        elif val_ratio < 1.00:
            val_adj = 0.004
        elif val_ratio > 1.20:
            val_adj = -0.006
        else:
            val_adj = 0.0
            
        total_rate = (base_rate + trans_adj + social_adj + amenity_adj + 
                      sec_adj + park_adj + age_adj + facing_adj + val_adj)
        return float(np.clip(total_rate, 0.045, 0.145))
        
    # Vectorized / apply CAGR rate
    df['Appreciation_Rate_Annual'] = df.apply(compute_row_cagr, axis=1)
    
    # 5-Year Future Estimated Price (Regression Target)
    # FV = PV * (1 + r)^5
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * ((1.0 + df['Appreciation_Rate_Annual']) ** 5)
    df['Future_Price_5Y'] = df['Future_Price_5Y'].round(2)
    
    # Derived financial metrics
    df['Total_Appreciation_Lakhs'] = (df['Future_Price_5Y'] - df['Price_in_Lakhs']).round(2)
    df['ROI_5Y_Percent'] = ((df['Total_Appreciation_Lakhs'] / df['Price_in_Lakhs']) * 100.0).round(2)
    
    # Classification Target: Good Investment (Binary 0 or 1)
    # Hurdle rate: median annual appreciation (~9.7%) and ROI > 58%
    rate_threshold = df['Appreciation_Rate_Annual'].median()
    df['Good_Investment'] = (df['Appreciation_Rate_Annual'] >= rate_threshold).astype(int)
    
    return df, rate_threshold

def apply_binary_and_categorical_encoding(df):
    """
    Encode binary features and categorical variables:
    - Binary codes: 0 and 1
    - Ordinal codes: mapped integers
    - Multi-class: dummy/one-hot or label codes for models
    """
    df = df.copy()
    
    # Explicit Binary Encoding
    df['Parking_Space_Binary'] = (df['Parking_Space'] == 'Yes').astype(int)
    df['Security_Binary'] = (df['Security'] == 'Yes').astype(int)
    df['Availability_Status_Binary'] = (df['Availability_Status'] == 'Ready_to_Move').astype(int)
    
    # Ordinal Encodings
    furnished_map = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}
    df['Furnished_Status_Ordinal'] = df['Furnished_Status'].map(furnished_map).fillna(1).astype(int)
    
    # One-Hot Dummies for Multi-class categories
    df['Property_Type_Code'] = df['Property_Type'].astype('category').cat.codes
    df['Facing_Code'] = df['Facing'].astype('category').cat.codes
    df['Owner_Type_Code'] = df['Owner_Type'].astype('category').cat.codes
    df['City_Code'] = df['City'].astype('category').cat.codes
    df['State_Code'] = df['State'].astype('category').cat.codes
    
    return df

def apply_feature_scaling(df, numerical_features):
    """
    Demonstrate and fit 3 standard enterprise scalers:
    1. StandardScaler (Zero Mean, Unit Variance)
    2. MinMaxScaler (0 to 1 bounded)
    3. RobustScaler (Median and IQR based, outlier robust)
    Returns scaled copies and fitted StandardScaler.
    """
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    
    X_num = df[numerical_features].values
    
    X_std = std_scaler.fit_transform(X_num)
    X_minmax = minmax_scaler.fit_transform(X_num)
    X_robust = robust_scaler.fit_transform(X_num)
    
    # Add scaled columns with prefix
    df_scaled = df.copy()
    for i, col in enumerate(numerical_features):
        df_scaled[f'{col}_standard_scaled'] = X_std[:, i]
        df_scaled[f'{col}_minmax_scaled'] = X_minmax[:, i]
        df_scaled[f'{col}_robust_scaled'] = X_robust[:, i]
        
    return df_scaled, std_scaler, minmax_scaler, robust_scaler

def run_preprocessing_pipeline(input_csv='india_housing_prices.csv', output_parquet='data/processed_housing_data.parquet'):
    """Complete end-to-end preprocessing runner."""
    print("=" * 60)
    print("Starting Real Estate Preprocessing Pipeline...")
    print("=" * 60)
    
    # 1. Load
    print("[1/6] Loading raw dataset...")
    df_raw = load_raw_data(input_csv)
    print(f"Loaded {len(df_raw):,} records with {df_raw.shape[1]} raw columns.")
    
    # 2. Clean
    print("[2/6] Cleaning missing values & duplicates...")
    df_clean, clean_report = clean_data(df_raw)
    print(f"Cleaned dataset: {clean_report['final_rows']:,} rows remaining.")
    
    # 3. Engineer Features
    print("[3/6] Engineering domain features (Price/SqFt, Infra, Amenities, Ratios)...")
    df_feat, city_medians = engineer_features(df_clean)
    
    # 4. Investment Targets
    print("[4/6] Formulating 5-Year Appreciation, Future Price & Good Investment targets...")
    df_targets, rate_threshold = calculate_investment_targets(df_feat, city_medians)
    
    # 5. Encodings & Binary Codes
    print("[5/6] Encoding binary flags and categorical variables...")
    df_encoded = apply_binary_and_categorical_encoding(df_targets)
    
    # 6. Scaling
    print("[6/6] Fitting Feature Scalers (Standard, MinMax, Robust)...")
    numerical_features_to_scale = [
        'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt_INR',
        'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals',
        'Infrastructure_Score', 'Amenities_Count', 'Floor_Ratio'
    ]
    df_final, std_scaler, mm_scaler, rob_scaler = apply_feature_scaling(df_encoded, numerical_features_to_scale)
    
    # Save processed dataset
    print(f"Saving processed dataset to '{output_parquet}'...")
    df_final.to_parquet(output_parquet, index=False)
    
    # Also save a sample CSV for lightweight inspection
    df_final.head(1000).to_csv('data/processed_sample_1000.csv', index=False)
    
    # Save artifacts for inference
    artifacts = {
        'city_medians': city_medians,
        'rate_threshold': float(rate_threshold),
        'clean_report': clean_report,
        'numerical_features': numerical_features_to_scale,
        'all_amenities': ALL_AMENITIES,
        'tier1_cities': list(TIER1_CITIES),
        'tier2_cities': list(TIER2_CITIES)
    }
    
    joblib.dump(std_scaler, 'models/scaler.joblib')
    joblib.dump(artifacts, 'models/preprocessor.joblib')
    
    with open('models/preprocessing_meta.json', 'w') as f:
        json.dump(artifacts, f, indent=2)
        
    print(f"Preprocessing completed successfully!")
    print(f"Total features created: {df_final.shape[1]}")
    print(f"Good Investment class distribution: {df_final['Good_Investment'].value_counts(normalize=True).to_dict()}")
    print("=" * 60)
    return df_final

if __name__ == '__main__':
    run_preprocessing_pipeline()
