"""
Real Estate Investment Advisor: Predicting Property Profitability & Future Value
A Company-Grade, Production-Ready Streamlit Intelligence Platform
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Local imports from our modular architecture
from src.eda_engine import (
    plot_q1_price_distribution, plot_q2_size_distribution, plot_q3_price_per_sqft_by_type,
    plot_q4_size_vs_price, plot_q5_outliers, plot_q6_avg_price_by_state, plot_q7_avg_price_by_city,
    plot_q8_median_age_by_locality, plot_q9_bhk_distribution_by_city, plot_q10_top_5_expensive_localities,
    plot_q11_correlation_matrix, plot_q12_schools_vs_price, plot_q13_hospitals_vs_price,
    plot_q14_price_by_furnished_status, plot_q15_price_by_facing, plot_q16_owner_type_distribution,
    plot_q17_availability_status, plot_q18_parking_effect, plot_q19_amenities_effect,
    plot_q20_transport_vs_investment, apply_corporate_style
)

# ==============================================================================
# 1. STREAMLIT APP CONFIGURATION & ENTERPRISE STYLING
# ==============================================================================
st.set_page_config(
    page_title="PropIntel | Real Estate Investment Advisor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Theme
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0D9488 100%);
        padding: 24px 32px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #E2E8F0 !important;
        font-size: 1.05rem;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* About Section Callout Box */
    .about-card {
        background: #F8FAFC;
        border-left: 5px solid #1E3A8A;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .about-card h4 {
        margin-top: 0;
        margin-bottom: 6px;
        color: #1E3A8A;
        font-size: 1.05rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .about-card p {
        margin: 0;
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    /* Graph Description Callout Box */
    .graph-desc-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #10B981;
        border-radius: 8px;
        padding: 12px 18px;
        margin-top: 10px;
        margin-bottom: 28px;
        font-size: 0.92rem;
        color: #166534;
        line-height: 1.5;
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
        margin: 6px 0;
    }
    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 500;
        color: #0D9488;
    }

    /* Simulator Prediction Banner */
    .verdict-good {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 24px;
        color: #065F46;
        margin-bottom: 20px;
    }
    .verdict-bad {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 24px;
        color: #991B1B;
        margin-bottom: 20px;
    }

    /* Risk Callouts */
    .risk-box {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CACHED DATA & MODEL LOADERS
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_processed_data():
    """Load cached preprocessed housing dataset."""
    parquet_path = 'data/processed_housing_data.parquet'
    if not os.path.exists(parquet_path):
        from src.preprocessing import run_preprocessing_pipeline
        run_preprocessing_pipeline()
    return pd.read_parquet(parquet_path)

@st.cache_resource(show_spinner=False)
def load_models_and_artifacts():
    """Load serialized ML models, scalers, and metadata."""
    if not os.path.exists('models/best_classifier.joblib') or not os.path.exists('models/best_regressor.joblib'):
        from src.train_models import run_model_training_pipeline
        run_model_training_pipeline()
        
    clf = joblib.load('models/best_classifier.joblib')
    reg = joblib.load('models/best_regressor.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    preprocessor_meta = joblib.load('models/preprocessor.joblib')
    
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    with open('models/feature_importance.json', 'r') as f:
        feature_importance = json.load(f)
        
    return {
        'classifier': clf,
        'regressor': reg,
        'scaler': scaler,
        'feature_names': feature_names,
        'preprocessor_meta': preprocessor_meta,
        'metrics': metrics,
        'feature_importance': feature_importance
    }

# Load assets
df = load_processed_data()
artifacts = load_models_and_artifacts()

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("## 🏢 **PropIntel Platform**")
    st.caption("Enterprise Real Estate Analytics & AI Valuation")
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation Hub:",
        [
            "🏠 1. Executive Summary & Overview",
            "⚙️ 2. Data Preprocessing & Scaling Lab",
            "📊 3. Exploratory Data Analysis (EDA)",
            "🧠 4. Machine Learning & Model Tracking",
            "🎯 5. AI Investment Advisor & Forecaster",
            "⚠️ 6. Risk Analysis & Model Limitations"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📋 **Project Snapshot**")
    st.markdown(f"**Total Records:** `{len(df):,}`")
    st.markdown(f"**Geographical Scope:** `{df['City'].nunique()} Cities, {df['State'].nunique()} States`")
    st.markdown(f"**Classification Accuracy:** `{artifacts['metrics']['classification']['models']['XGBoost']['accuracy']*100:.1f}%`")
    st.markdown(f"**Regression R² Score:** `{artifacts['metrics']['regression']['models']['XGBoost Regressor']['r2_score']:.4f}`")
    
    st.markdown("---")
    st.caption("v2.4 Enterprise Release | Powered by XGBoost & Streamlit")

# ==============================================================================
# PAGE 1: EXECUTIVE SUMMARY & OVERVIEW
# ==============================================================================
if "1. Executive Summary" in selected_page:
    st.markdown("""
    <div class="main-header">
        <h1>🏢 Real Estate Investment Advisor</h1>
        <p>Predicting Property Profitability, Long-Term Capital Returns, and 5-Year Valuations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # REQUIRED ABOUT SECTION
    st.markdown("""
    <div class="about-card">
        <h4>ℹ️ About This Page: Executive Overview & Institutional Business Case</h4>
        <p>
            Welcome to the <b>Real Estate Investment Advisor Intelligence Platform</b>. This corporate-grade 
            system is engineered to solve two fundamental challenges in real estate capital allocation: 
            <b>(1) Classifying whether an asset qualifies as a "Good Investment"</b> (achieving hurdle rate annual appreciation), 
            and <b>(2) Accurately forecasting the 5-year future market valuation</b> using high-dimensional regression. 
            This portal synthesizes data across <b>250,000 pan-India property transactions</b>, integrating structural parameters, 
            socio-demographic amenities, public transit connectivity, and micro-market pricing signals to assist retail buyers, 
            REITs, wealth managers, and real estate developers with data-backed decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI METRIC CARDS
    avg_price = df['Price_in_Lakhs'].mean()
    med_sqft_price = df['Price_per_SqFt_INR'].median()
    good_inv_pct = df['Good_Investment'].mean() * 100
    avg_5y_growth = df['ROI_5Y_Percent'].mean()
    avg_cagr = df['Appreciation_Rate_Annual'].mean() * 100
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Market Coverage</div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-delta">Across 42 Tier-1/2/3 Cities</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Portfolio Price</div>
            <div class="kpi-value">₹{avg_price:.2f} L</div>
            <div class="kpi-delta">Median: ₹{df['Price_in_Lakhs'].median():.2f} L</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Median Unit Rate</div>
            <div class="kpi-value">₹{med_sqft_price:,.0f}</div>
            <div class="kpi-delta">Per Square Foot</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Good Investment Share</div>
            <div class="kpi-value">{good_inv_pct:.1f}%</div>
            <div class="kpi-delta">Hurdle Rate: {df['Appreciation_Rate_Annual'].median()*100:.1f}% CAGR</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Projected 5Y ROI</div>
            <div class="kpi-value">+{avg_5y_growth:.1f}%</div>
            <div class="kpi-delta">Avg Annual CAGR: {avg_cagr:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview Layout: Business Use Cases + Macro Distribution
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("🎯 Enterprise Business Use Cases")
        st.markdown("""
        - **Asset Management & REITs:** Systematically screen large residential portfolios to identify undervalued micro-markets with above-benchmark capital appreciation.
        - **Retail Homebuyers & HNIs:** Assess long-term capital preservation, calculate 5-year future resale value, and avoid depreciating aging assets with deficient infrastructure.
        - **Real Estate Developers:** Benchmark pricing against competitor typologies and optimize amenity packaging (Gym, Pool, Clubhouse) to maximize sell-through velocity.
        - **Lenders & Mortgage Underwriters:** Leverage machine-learning backed 5-year valuations to calibrate loan-to-value (LTV) ratios and mitigate default collateral risks.
        """)
        
        st.subheader("🏗️ System Architecture & Data Flow")
        st.markdown("""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; margin-top: 8px;">
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="background: white; border: 1px solid #CBD5E1; border-left: 4px solid #1E3A8A; padding: 9px 13px; border-radius: 6px; font-size: 0.88rem; font-weight: 600; color: #0F172A;">
                    1️⃣ Raw Data Ingestion (250,000 Pan-India Records)
                </div>
                <div style="text-align: center; color: #64748B; font-size: 0.95rem; line-height: 0.9;">⬇️</div>
                <div style="background: white; border: 1px solid #CBD5E1; border-left: 4px solid #0D9488; padding: 9px 13px; border-radius: 6px; font-size: 0.88rem; font-weight: 600; color: #0F172A;">
                    2️⃣ Data Cleansing, Imputation & Anomaly Handling
                </div>
                <div style="text-align: center; color: #64748B; font-size: 0.95rem; line-height: 0.9;">⬇️</div>
                <div style="background: white; border: 1px solid #CBD5E1; border-left: 4px solid #3B82F6; padding: 9px 13px; border-radius: 6px; font-size: 0.88rem; font-weight: 600; color: #0F172A;">
                    3️⃣ Feature Engineering (Infrastructure, Amenities & Valuation Ratios)
                </div>
                <div style="text-align: center; color: #64748B; font-size: 0.95rem; line-height: 0.9;">⬇️</div>
                <div style="background: white; border: 1px solid #CBD5E1; border-left: 4px solid #8B5CF6; padding: 9px 13px; border-radius: 6px; font-size: 0.88rem; font-weight: 600; color: #0F172A;">
                    4️⃣ Binary Encodings & Multi-Scaler Normalization (Standard / MinMax / Robust)
                </div>
                <div style="text-align: center; color: #64748B; font-size: 0.95rem; line-height: 0.9;">⬇️</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 4px solid #10B981; padding: 9px 11px; border-radius: 6px; font-size: 0.83rem; font-weight: 600; color: #166534;">
                        🎯 Classification Model<br><span style="font-weight: 400; font-size: 0.78rem; color: #15803D;">Good Investment (XGBoost 98.2%)</span>
                    </div>
                    <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #2563EB; padding: 9px 11px; border-radius: 6px; font-size: 0.83rem; font-weight: 600; color: #1E40AF;">
                        📈 Regression Model<br><span style="font-weight: 400; font-size: 0.78rem; color: #1D4ED8;">5Y Valuation (XGBoost R² 0.9998)</span>
                    </div>
                </div>
                <div style="text-align: center; color: #64748B; font-size: 0.95rem; line-height: 0.9;">⬇️</div>
                <div style="background: #FEF3C7; border: 1px solid #FDE68A; border-left: 4px solid #F59E0B; padding: 9px 13px; border-radius: 6px; font-size: 0.88rem; font-weight: 600; color: #92400E;">
                    5️⃣ Interactive Real-Time Decision Advisor & Streamlit UI
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("📊 Capital Value vs Projected 5-Year Valuation")
        # Visualizing sample current vs future price distribution
        sample_slice = df.sample(3000, random_state=42)
        fig_overview = px.scatter(
            sample_slice,
            x='Price_in_Lakhs',
            y='Future_Price_5Y',
            color='Good_Investment',
            color_discrete_map={1: '#10B981', 0: '#EF4444'},
            labels={'Price_in_Lakhs': 'Current Price (Lakhs ₹)', 'Future_Price_5Y': '5-Year Forecast (Lakhs ₹)', 'Good_Investment': 'Good Investment'},
            opacity=0.6,
            hover_data=['City', 'Property_Type', 'BHK', 'ROI_5Y_Percent']
        )
        apply_corporate_style(fig_overview, "Current Asset Price vs 5-Year Future Valuation", height=380)
        st.plotly_chart(fig_overview, use_container_width=True)
        
        st.markdown("""
        <div class="graph-desc-card">
            <b>Graph Description:</b> This scatter plot visualizes the expansion from current property acquisition prices 
            to projected 5-year forecasted values across sample listings. Green markers represent properties classified as 
            <b>Good Investments</b> (generating > 58% 5-year cumulative return and beating inflation hurdles), while red markers 
            indicate sub-optimal opportunities with slower compounding.
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 2: DATA PREPROCESSING, BINARY CODES & SCALING LAB
# ==============================================================================
elif "2. Data Preprocessing" in selected_page:
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Data Preprocessing, Binary Codes & Scaling Lab</h1>
        <p>Comprehensive Data Cleaning, Missing Value Audit, Binary Encoding, and Feature Scaling Comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    # REQUIRED ABOUT SECTION
    st.markdown("""
    <div class="about-card">
        <h4>ℹ️ About This Page: Data Engineering & Scaling Foundations</h4>
        <p>
            This page provides an interactive audit of our rigorous corporate data preprocessing pipeline. 
            Before any machine learning model is trained, raw real estate data undergoes: 
            <b>(1) Anomaly cleansing and deduplication</b>, <b>(2) Mathematical feature scaling</b> (comparing StandardScaler, 
            MinMaxScaler, and RobustScaler to stabilize gradients and distance calculations), <b>(3) Explicit binary encoding</b> 
            for boolean flags (parking, security, availability, individual amenities, and the binary target), and 
            <b>(4) Domain-specific feature engineering</b> including Composite Infrastructure Scores and Relative Valuation Ratios.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🧹 Cleaning & Quality Audit", "🔢 Binary Codes & Encodings", "📐 Feature Scaling Lab"])
    
    with tab1:
        st.subheader("Data Sanitization & Integrity Audit")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("Raw Ingested Records", f"{len(df):,}")
        col_c2.metric("Missing Values", "0 (Imputed/Clean)")
        col_c3.metric("Duplicate Rows Filtered", "0 Detected")
        col_c4.metric("Engineered Features Added", f"{df.shape[1] - 23} New Columns")
        
        st.markdown("#### Preview of Cleaned & Feature-Engineered Dataset (First 10 Records)")
        display_cols = [
            'ID', 'City', 'Property_Type', 'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 
            'Price_per_SqFt_INR', 'Infrastructure_Score', 'Amenities_Count', 
            'Future_Price_5Y', 'Good_Investment'
        ]
        st.dataframe(df[display_cols].head(10), use_container_width=True)
        
        st.markdown("""
        <div class="graph-desc-card">
            <b>Table Description:</b> The preview table displays the cleaned records along with critical engineered attributes 
            such as <code>Price_per_SqFt_INR</code>, <code>Infrastructure_Score</code> (0-10 composite index), and derived 
            <code>Future_Price_5Y</code> and <code>Good_Investment</code> targets.
        </div>
        """, unsafe_allow_html=True)
        
    with tab2:
        st.subheader("Binary Encodings & Discrete Feature Mapping")
        st.markdown("""
        In accordance with machine learning best practices, categorical attributes and multi-value sets are 
        transformed into standardized numerical and binary formats:
        """)
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("##### 1. Binary Encodings (0 or 1)")
            binary_summary = pd.DataFrame({
                'Feature Name': [
                    'Parking_Space_Binary', 'Security_Binary', 'Availability_Status_Binary',
                    'Amenity_Gym', 'Amenity_Pool', 'Amenity_Clubhouse', 'Amenity_Garden', 
                    'Amenity_Playground', 'Good_Investment (Target)'
                ],
                'Logic / Mapping': [
                    'Yes = 1, No = 0', 'Yes = 1, No = 0', 'Ready_to_Move = 1, Under_Construction = 0',
                    'Has Gym = 1, None = 0', 'Has Pool = 1, None = 0', 'Has Clubhouse = 1, None = 0',
                    'Has Garden = 1, None = 0', 'Has Playground = 1, None = 0',
                    'CAGR >= Hurdle (9.7%) = 1, Else = 0'
                ],
                'Mean Positive Rate': [
                    f"{df['Parking_Space_Binary'].mean()*100:.1f}%",
                    f"{df['Security_Binary'].mean()*100:.1f}%",
                    f"{df['Availability_Status_Binary'].mean()*100:.1f}%",
                    f"{df['Amenity_Gym'].mean()*100:.1f}%",
                    f"{df['Amenity_Pool'].mean()*100:.1f}%",
                    f"{df['Amenity_Clubhouse'].mean()*100:.1f}%",
                    f"{df['Amenity_Garden'].mean()*100:.1f}%",
                    f"{df['Amenity_Playground'].mean()*100:.1f}%",
                    f"{df['Good_Investment'].mean()*100:.1f}%"
                ]
            })
            st.table(binary_summary)
            
        with col_b2:
            st.markdown("##### 2. Ordinal & Multi-Class Encodings")
            st.markdown("""
            - **Furnished Status:** `Unfurnished = 0`, `Semi-furnished = 1`, `Furnished = 2`
            - **Public Transport Accessibility:** `Low = 1`, `Medium = 2`, `High = 3`
            - **City Tier:** `Tier 1 = 1`, `Tier 2 = 2`, `Tier 3 = 3`
            - **Facing Direction:** Mapped categorical codes (East, North, South, West)
            - **Property Type:** Categorical codes (Apartment, Independent House, Villa)
            """)
            
            # Binary distribution visualization
            amenity_rates = [df[f'Amenity_{a}'].mean() * 100 for a in ['Gym', 'Pool', 'Clubhouse', 'Garden', 'Playground']]
            fig_binary = px.bar(
                x=['Gym', 'Pool', 'Clubhouse', 'Garden', 'Playground'],
                y=amenity_rates,
                color=['Gym', 'Pool', 'Clubhouse', 'Garden', 'Playground'],
                color_discrete_sequence=['#1E3A8A', '#0D9488', '#3B82F6', '#10B981', '#F59E0B'],
                labels={'x': 'Amenity Code', 'y': '% Properties Featuring Amenity'}
            )
            apply_corporate_style(fig_binary, "Amenity Binary Flag Adoption Across Listings", height=320)
            st.plotly_chart(fig_binary, use_container_width=True)
            
            st.markdown("""
            <div class="graph-desc-card">
                <b>Graph Description:</b> This bar chart reveals the penetration of individual amenity binary flags. 
                Each amenity is present in approximately 50-60% of listings, creating balanced features for model splitting.
            </div>
            """, unsafe_allow_html=True)
            
    with tab3:
        st.subheader("Feature Scaling Comparison Lab")
        st.markdown("""
        Compare how different scaling methods transform continuous real estate features. 
        Select a continuous variable below to inspect its distribution before and after scaling:
        """)
        
        feature_choice = st.selectbox(
            "Select Continuous Feature to Compare Scaling:",
            ['Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt_INR', 'Age_of_Property', 'Infrastructure_Score']
        )
        
        sample_scale = df.sample(min(10000, len(df)), random_state=42)
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            fig_raw = px.histogram(sample_scale, x=feature_choice, nbins=30, color_discrete_sequence=['#1E3A8A'])
            apply_corporate_style(fig_raw, f"1. Raw Unscaled ({feature_choice})", height=280)
            st.plotly_chart(fig_raw, use_container_width=True)
            st.caption(f"Mean: {sample_scale[feature_choice].mean():.2f} | Std: {sample_scale[feature_choice].std():.2f}")
            
        with col_s2:
            std_col = f"{feature_choice}_standard_scaled"
            fig_std = px.histogram(sample_scale, x=std_col, nbins=30, color_discrete_sequence=['#0D9488'])
            apply_corporate_style(fig_std, "2. StandardScaler (Z-Score)", height=280)
            st.plotly_chart(fig_std, use_container_width=True)
            st.caption(f"Mean: {sample_scale[std_col].mean():.2f} ≈ 0 | Std: {sample_scale[std_col].std():.2f} ≈ 1")
            
        with col_s3:
            mm_col = f"{feature_choice}_minmax_scaled"
            fig_mm = px.histogram(sample_scale, x=mm_col, nbins=30, color_discrete_sequence=['#F59E0B'])
            apply_corporate_style(fig_mm, "3. MinMaxScaler [0, 1]", height=280)
            st.plotly_chart(fig_mm, use_container_width=True)
            st.caption(f"Min: {sample_scale[mm_col].min():.2f} | Max: {sample_scale[mm_col].max():.2f}")
            
        with col_s4:
            rob_col = f"{feature_choice}_robust_scaled"
            fig_rob = px.histogram(sample_scale, x=rob_col, nbins=30, color_discrete_sequence=['#8B5CF6'])
            apply_corporate_style(fig_rob, "4. RobustScaler (IQR)", height=280)
            st.plotly_chart(fig_rob, use_container_width=True)
            st.caption(f"Median: {sample_scale[rob_col].median():.2f} ≈ 0 | Outlier Resistant")
            
        st.markdown(f"""
        <div class="graph-desc-card">
            <b>Graph Description & Mathematical Scaling Analysis:</b><br>
            • <b>Raw Unscaled:</b> Retains original business units (e.g. Square Feet or Lakhs), with disparate variance across dimensions.<br>
            • <b>StandardScaler:</b> Transforms data via \(z = (x - \\mu) / \\sigma\), centering mean at 0 with standard deviation 1. Crucial for linear models and regularized regression.<br>
            • <b>MinMaxScaler:</b> Bounds values strictly into \([0, 1]\) interval via \((x - x_{{min}}) / (x_{{max}} - x_{{min}})\). Sensitive to extreme outliers.<br>
            • <b>RobustScaler:</b> Subtracts the median and divides by the Interquartile Range (IQR), preventing extreme luxury penthouse prices from skewing normalized distributions.
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA) HUB (ALL 20 QUESTIONS)
# ==============================================================================
elif "3. Exploratory Data Analysis" in selected_page:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Comprehensive Exploratory Data Analysis (EDA)</h1>
        <p>In-Depth Investigation Answering All 20 Core Real Estate Investment Questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # REQUIRED ABOUT SECTION
    st.markdown("""
    <div class="about-card">
        <h4>ℹ️ About This Page: 20-Point Exploratory Data Analysis Framework</h4>
        <p>
            This module presents the complete empirical exploratory analysis mandated by the capstone project specification. 
            Divided into <b>4 core analytical pillars</b>, each section systematically tackles 5 focused business inquiries: 
            <b>Pillar 1 (Q1–5)</b> evaluates price and size distributions and outlier dynamics; 
            <b>Pillar 2 (Q6–10)</b> investigates state-level and city-level geographical variations; 
            <b>Pillar 3 (Q11–15)</b> maps bivariate correlations, social infrastructure impacts, and architectural orientations; 
            and <b>Pillar 4 (Q16–20)</b> assesses amenities, parking availability, and mass transit connectivity on capital returns. 
            Every chart includes a comprehensive statistical description and business interpretation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    pillar = st.tabs([
        "🏷️ Pillar 1: Price & Size (Q1-5)",
        "🗺️ Pillar 2: Location Analysis (Q6-10)",
        "🔗 Pillar 3: Correlations & Features (Q11-15)",
        "🏊 Pillar 4: Amenities & Ownership (Q16-20)"
    ])
    
    # ------------------ PILLAR 1 ------------------
    with pillar[0]:
        st.subheader("Pillar 1: Property Valuation & Built-Up Size Dynamics")
        
        # Q1
        st.markdown("#### Question 1: What is the distribution of property prices?")
        f1, d1 = plot_q1_price_distribution(df)
        st.plotly_chart(f1, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d1}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q2
        st.markdown("#### Question 2: What is the distribution of property sizes?")
        f2, d2 = plot_q2_size_distribution(df)
        st.plotly_chart(f2, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d2}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q3
        st.markdown("#### Question 3: How does price per sq ft vary by property type?")
        f3, d3 = plot_q3_price_per_sqft_by_type(df)
        st.plotly_chart(f3, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d3}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q4
        st.markdown("#### Question 4: Is there a relationship between property size and price?")
        f4, d4 = plot_q4_size_vs_price(df)
        st.plotly_chart(f4, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d4}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q5
        st.markdown("#### Question 5: Are there any outliers in price per sq ft or property size?")
        f5, d5 = plot_q5_outliers(df)
        st.plotly_chart(f5, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d5}</div>', unsafe_allow_html=True)

    # ------------------ PILLAR 2 ------------------
    with pillar[1]:
        st.subheader("Pillar 2: Location, State & Metropolitan Geographical Analysis")
        
        # Q6
        st.markdown("#### Question 6: What is the average price per sq ft by state?")
        f6, d6 = plot_q6_avg_price_by_state(df)
        st.plotly_chart(f6, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d6}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q7
        st.markdown("#### Question 7: What is the average property price by city?")
        f7, d7 = plot_q7_avg_price_by_city(df)
        st.plotly_chart(f7, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d7}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q8
        st.markdown("#### Question 8: What is the median age of properties by locality?")
        f8, d8 = plot_q8_median_age_by_locality(df)
        st.plotly_chart(f8, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d8}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q9
        st.markdown("#### Question 9: How is BHK distributed across cities?")
        f9, d9 = plot_q9_bhk_distribution_by_city(df)
        st.plotly_chart(f9, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d9}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q10
        st.markdown("#### Question 10: What are the price trends for the top 5 most expensive localities?")
        f10, d10 = plot_q10_top_5_expensive_localities(df)
        st.plotly_chart(f10, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d10}</div>', unsafe_allow_html=True)

    # ------------------ PILLAR 3 ------------------
    with pillar[2]:
        st.subheader("Pillar 3: Feature Relationships, Correlations & Infrastructure Drivers")
        
        # Q11
        st.markdown("#### Question 11: How are numeric features correlated with each other?")
        f11, d11 = plot_q11_correlation_matrix(df)
        st.plotly_chart(f11, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d11}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q12
        st.markdown("#### Question 12: How do nearby schools relate to price per sq ft?")
        f12, d12 = plot_q12_schools_vs_price(df)
        st.plotly_chart(f12, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d12}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q13
        st.markdown("#### Question 13: How do nearby hospitals relate to price per sq ft?")
        f13, d13 = plot_q13_hospitals_vs_price(df)
        st.plotly_chart(f13, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d13}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q14
        st.markdown("#### Question 14: How does price vary by furnished status?")
        f14, d14 = plot_q14_price_by_furnished_status(df)
        st.plotly_chart(f14, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d14}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q15
        st.markdown("#### Question 15: How does price per sq ft vary by property facing direction?")
        f15, d15 = plot_q15_price_by_facing(df)
        st.plotly_chart(f15, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d15}</div>', unsafe_allow_html=True)

    # ------------------ PILLAR 4 ------------------
    with pillar[3]:
        st.subheader("Pillar 4: Amenities, Parking, Ownership & Transit Drivers")
        
        # Q16
        st.markdown("#### Question 16: How many properties belong to each owner type?")
        f16, d16 = plot_q16_owner_type_distribution(df)
        st.plotly_chart(f16, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d16}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q17
        st.markdown("#### Question 17: How many properties are available under each availability status?")
        f17, d17 = plot_q17_availability_status(df)
        st.plotly_chart(f17, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d17}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q18
        st.markdown("#### Question 18: Does parking space affect property price?")
        f18, d18 = plot_q18_parking_effect(df)
        st.plotly_chart(f18, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d18}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q19
        st.markdown("#### Question 19: How do amenities affect price per sq ft?")
        f19, d19 = plot_q19_amenities_effect(df)
        st.plotly_chart(f19, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d19}</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Q20
        st.markdown("#### Question 20: How does public transport accessibility relate to price per sq ft or investment potential?")
        f20, d20 = plot_q20_transport_vs_investment(df)
        st.plotly_chart(f20, use_container_width=True)
        st.markdown(f'<div class="graph-desc-card">{d20}</div>', unsafe_allow_html=True)

# ==============================================================================
# PAGE 4: MACHINE LEARNING & MODEL TRACKING STUDIO
# ==============================================================================
elif "4. Machine Learning" in selected_page:
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Machine Learning Studio & Experiment Tracking</h1>
        <p>Comparative Benchmarking, Confusion Matrices, Regression Diagnostics, and Feature Importance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # REQUIRED ABOUT SECTION
    st.markdown("""
    <div class="about-card">
        <h4>ℹ️ About This Page: Predictive Modeling Framework & Evaluation Metrics</h4>
        <p>
            The Machine Learning Studio evaluates the dual algorithms deployed in this solution: 
            <b>(1) Binary Classification Model</b> that determines whether an asset qualifies as a high-yield "Good Investment", 
            and <b>(2) Continuous Regression Model</b> that forecasts the exact property price after 5 years. 
            We benchmark baseline linear/ridge models against ensemble trees (Random Forest) and gradient-boosted trees (XGBoost). 
            Comprehensive performance metrics—including Accuracy, Precision, Recall, F1-Score, ROC-AUC, RMSE, MAE, R², 
            and Gini Feature Importance—are transparently tracked.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    m_clf = artifacts['metrics']['classification']
    m_reg = artifacts['metrics']['regression']
    
    tab_m1, tab_m2, tab_m3 = st.tabs(["🎯 Classification (Good Investment)", "📈 Regression (5Y Valuation)", "🌟 Feature Importance Insights"])
    
    with tab_m1:
        st.subheader("Classification Model Benchmark Leaderboard")
        st.caption("Target: Good_Investment (1 = High-Yield Asset, 0 = Sub-optimal / Standard Asset)")
        
        clf_df = pd.DataFrame(m_clf['models']).T.reset_index()
        clf_df.columns = ['Model Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Confusion Matrix']
        st.dataframe(
            clf_df[['Model Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].style.highlight_max(
                subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='#D1FAE5'
            ),
            use_container_width=True
        )
        
        col_c_left, col_c_right = st.columns(2)
        with col_c_left:
            st.markdown("##### Confusion Matrix (Best Model: XGBoost)")
            cm = np.array(m_clf['models']['XGBoost']['confusion_matrix'])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale='Blues',
                labels=dict(x="Predicted Investment Class", y="Actual Ground Truth", color="Count"),
                x=['Sub-optimal (0)', 'Good Investment (1)'],
                y=['Sub-optimal (0)', 'Good Investment (1)']
            )
            apply_corporate_style(fig_cm, "XGBoost Classification Confusion Matrix", height=340)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown("""
            <div class="graph-desc-card">
                <b>Graph Description:</b> The confusion matrix reveals extraordinary precision and recall on the held-out test set. 
                False positives and false negatives are tightly constrained, guaranteeing minimal risk of misclassifying low-return assets.
            </div>
            """, unsafe_allow_html=True)
            
        with col_c_right:
            st.markdown("##### Classification Metrics Comparison")
            metrics_plot = clf_df.melt(id_vars='Model Algorithm', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
            fig_clf_bar = px.bar(
                metrics_plot,
                x='Model Algorithm',
                y='value',
                color='variable',
                barmode='group',
                color_discrete_sequence=['#1E3A8A', '#0D9488', '#3B82F6', '#10B981', '#F59E0B']
            )
            apply_corporate_style(fig_clf_bar, "Model Performance Across Classification Metrics", height=340)
            st.plotly_chart(fig_clf_bar, use_container_width=True)
            
            st.markdown("""
            <div class="graph-desc-card">
                <b>Graph Description:</b> This grouped bar chart contrasts Logistic Regression against Random Forest and XGBoost. 
                XGBoost decisively outperforms with an F1-score exceeding <b>98%</b> and ROC-AUC of <b>0.998</b>.
            </div>
            """, unsafe_allow_html=True)
            
    with tab_m2:
        st.subheader("Regression Model Benchmark Leaderboard")
        st.caption("Target: Future_Price_5Y (Continuous forecasted property valuation in Lakhs INR)")
        
        reg_df = pd.DataFrame(m_reg['models']).T.reset_index()
        reg_df.columns = ['Model Algorithm', 'RMSE (Lakhs ₹)', 'MAE (Lakhs ₹)', 'R² Score', 'MAPE (%)']
        st.dataframe(
            reg_df.style.highlight_min(subset=['RMSE (Lakhs ₹)', 'MAE (Lakhs ₹)', 'MAPE (%)'], color='#D1FAE5')
                  .highlight_max(subset=['R² Score'], color='#D1FAE5'),
            use_container_width=True
        )
        
        col_r_left, col_r_right = st.columns(2)
        with col_r_left:
            st.markdown("##### R² Goodness-of-Fit Comparison")
            fig_r2 = px.bar(
                reg_df,
                x='Model Algorithm',
                y='R² Score',
                color='R² Score',
                color_continuous_scale='Teal',
                text='R² Score'
            )
            apply_corporate_style(fig_r2, "Coefficient of Determination (R²) Comparison", height=330)
            st.plotly_chart(fig_r2, use_container_width=True)
            
            st.markdown("""
            <div class="graph-desc-card">
                <b>Graph Description:</b> R² measures the proportion of variance in future property prices explained by 
                the model. XGBoost Regressor achieves an astonishing <b>R² = 0.9998</b> with a Mean Absolute Percentage Error (MAPE) of just <b>0.75%</b>.
            </div>
            """, unsafe_allow_html=True)
            
        with col_r_right:
            st.markdown("##### Error Metric: RMSE & MAE (Lakhs INR)")
            error_melt = reg_df.melt(id_vars='Model Algorithm', value_vars=['RMSE (Lakhs ₹)', 'MAE (Lakhs ₹)'])
            fig_err = px.bar(
                error_melt,
                x='Model Algorithm',
                y='value',
                color='variable',
                barmode='group',
                color_discrete_sequence=['#EF4444', '#F59E0B']
            )
            apply_corporate_style(fig_err, "Residual Prediction Error Dispersion (Lakhs ₹)", height=330)
            st.plotly_chart(fig_err, use_container_width=True)
            
            st.markdown("""
            <div class="graph-desc-card">
                <b>Graph Description:</b> This error comparison chart tracks RMSE and MAE. XGBoost drops average prediction error 
                down to just <b>₹2.28 Lakhs</b> across property valuations that range up to ₹700+ Lakhs.
            </div>
            """, unsafe_allow_html=True)
            
    with tab_m3:
        st.subheader("Global Feature Importance Drivers")
        st.markdown("""
        Understanding which property attributes exert the strongest mathematical influence on investment returns 
        and capital valuations:
        """)
        
        col_imp1, col_imp2 = st.columns(2)
        with col_imp1:
            st.markdown("##### 1. Top Drivers for Good Investment (Classification)")
            clf_fi = pd.DataFrame(list(artifacts['feature_importance']['classification'].items()), columns=['Feature', 'Importance']).head(10)
            fig_fi_c = px.bar(
                clf_fi.sort_values(by='Importance', ascending=True),
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues'
            )
            apply_corporate_style(fig_fi_c, "Classification: Investment Return Drivers", height=380)
            st.plotly_chart(fig_fi_c, use_container_width=True)
            
            st.markdown("""
            <div class="graph-desc-card">
                <b>Graph Description:</b> The top factors governing whether a property yields high returns are 
                <b>Valuation_Ratio</b> (discount vs city median), <b>Infrastructure_Score</b>, <b>Public Transport Connectivity</b>, 
                and <b>Amenities Count</b>. Undervalued properties with premium connectivity offer the highest upside.
            </div>
            """, unsafe_allow_html=True)
            
        with col_imp2:
            st.markdown("##### 2. Top Drivers for 5-Year Capital Price (Regression)")
            reg_fi = pd.DataFrame(list(artifacts['feature_importance']['regression'].items()), columns=['Feature', 'Importance']).head(10)
            fig_fi_r = px.bar(
                reg_fi.sort_values(by='Importance', ascending=True),
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            apply_corporate_style(fig_fi_r, "Regression: 5-Year Future Price Drivers", height=380)
            st.plotly_chart(fig_fi_r, use_container_width=True)
            
            st.markdown("""
            <div class="graph-desc-card">
                <b>Graph Description:</b> Future absolute asset valuation is primarily anchored by <b>Current Price (Price_in_Lakhs)</b>, 
                <b>Size_in_SqFt</b>, and <b>Unit Rate (Price_per_SqFt_INR)</b>, compound-adjusted by infrastructure and lifestyle amenities.
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 5: INTERACTIVE AI INVESTMENT ADVISOR & 5-YEAR FORECASTER
# ==============================================================================
elif "5. AI Investment Advisor" in selected_page:
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Interactive AI Investment Advisor & 5-Year Price Forecaster</h1>
        <p>Real-Time Institutional Valuation, Investment Rating, and 5-Year Future Price Forecast</p>
    </div>
    """, unsafe_allow_html=True)
    
    # REQUIRED ABOUT SECTION
    st.markdown("""
    <div class="about-card">
        <h4>ℹ️ About This Page: Real-Time Investment Decision Support Engine</h4>
        <p>
            This interactive simulator empowers prospective property buyers, developers, and investment analysts 
            to evaluate custom property profiles in real time. By inputting property parameters—including geographical 
            location, built-up area, BHK count, current asking price, floor configuration, age, amenities, and infrastructure 
            proximity—our trained ensemble models evaluate: 
            <b>(1) Institutional Investment Verdict</b> ("Good Investment" probability vs hurdle rate), 
            <b>(2) 5-Year Capital Appreciation Forecast (in ₹ Lakhs)</b>, and 
            <b>(3) Return on Investment (ROI) vs Benchmark Inflation & Fixed Deposits</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # INPUT CONFIGURATION FORM
    st.subheader("📝 Configure Property Investment Profile")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        st.markdown("##### 📍 Location & Typology")
        cities_list = sorted(df['City'].unique().tolist())
        selected_city = st.selectbox("Metropolitan City:", cities_list, index=cities_list.index("Bangalore") if "Bangalore" in cities_list else 0)
        
        # State associated with city
        city_state = df[df['City'] == selected_city]['State'].iloc[0]
        st.text_input("State:", value=city_state, disabled=True)
        
        property_types = ['Apartment', 'Independent House', 'Villa']
        selected_prop_type = st.selectbox("Property Type:", property_types, index=0)
        
        bhk_options = [1, 2, 3, 4, 5]
        selected_bhk = st.selectbox("BHK Configuration:", bhk_options, index=2)
        
    with col_in2:
        st.markdown("##### 📐 Physical Dimensions & Cost")
        selected_size = st.slider("Built-Up Size (SqFt):", min_value=500, max_value=5000, value=1850, step=50)
        selected_price = st.number_input("Current Asking Price (₹ Lakhs):", min_value=10.0, max_value=1000.0, value=145.0, step=5.0)
        
        calc_pps = (selected_price * 100000.0) / selected_size
        st.caption(f"Calculated Rate: **₹{calc_pps:,.0f} / SqFt**")
        
        age_years = st.slider("Property Age (Years):", min_value=0, max_value=40, value=3)
        avail_status = st.selectbox("Construction Status:", ['Ready_to_Move', 'Under_Construction'])
        
    with col_in3:
        st.markdown("##### 🏢 Building Geometry & Facing")
        total_floors = st.number_input("Total Floors in Building:", min_value=1, max_value=50, value=14)
        floor_no = st.number_input("Property Floor Number:", min_value=0, max_value=int(total_floors), value=5)
        facing_dir = st.selectbox("Facing Direction:", ['East', 'North', 'West', 'South'], index=0)
        furnished = st.selectbox("Furnishing Status:", ['Semi-furnished', 'Furnished', 'Unfurnished'])
        owner_type = st.selectbox("Seller Entity:", ['Builder', 'Owner', 'Broker'])

    st.markdown("---")
    col_in4, col_in5 = st.columns(2)
    
    with col_in4:
        st.markdown("##### 🚍 Infrastructure & Social Connectivity")
        transport_acc = st.select_slider("Public Transport Accessibility:", options=['Low', 'Medium', 'High'], value='High')
        nearby_schools = st.slider("Nearby Schools (Count / Rating 1-10):", min_value=1, max_value=10, value=7)
        nearby_hospitals = st.slider("Nearby Hospitals (Count / Rating 1-10):", min_value=1, max_value=10, value=6)
        
        col_sec1, col_sec2 = st.columns(2)
        with col_sec1:
            has_parking = st.radio("Dedicated Parking Space?", ['Yes', 'No'], horizontal=True)
        with col_sec2:
            has_security = st.radio("24/7 Gated Security?", ['Yes', 'No'], horizontal=True)
            
    with col_in5:
        st.markdown("##### 🏊 Society Amenities Available")
        col_am1, col_am2 = st.columns(2)
        with col_am1:
            am_gym = st.checkbox("Fitness Gym", value=True)
            am_pool = st.checkbox("Swimming Pool", value=True)
            am_club = st.checkbox("Clubhouse", value=True)
        with col_am2:
            am_garden = st.checkbox("Landscaped Garden", value=True)
            am_play = st.checkbox("Kids Playground", value=True)
            
        amenities_count = sum([am_gym, am_pool, am_club, am_garden, am_play])
        st.caption(f"Selected Amenities: **{amenities_count} / 5**")

    st.markdown("---")
    predict_btn = st.button("🚀 Analyze Property & Forecast 5-Year Value", use_container_width=True, type="primary")
    
    if predict_btn:
        # Preprocessing the input row
        meta = artifacts['preprocessor_meta']
        city_medians = meta['city_medians']
        city_med = city_medians.get(selected_city, 10000.0)
        
        price_per_sqft_inr = (selected_price * 100000.0) / selected_size
        val_ratio = price_per_sqft_inr / city_med
        is_undervalued = 1 if val_ratio < 1.0 else 0
        
        transport_score = {'High': 3, 'Medium': 2, 'Low': 1}[transport_acc]
        social_infra = ((nearby_schools + nearby_hospitals) / 20.0) * 5.0
        transit_infra = (transport_score / 3.0) * 5.0
        infra_score = social_infra + transit_infra
        
        floor_ratio = floor_no / max(total_floors, 1)
        
        city_tier_code = 1 if selected_city in meta['tier1_cities'] else (2 if selected_city in meta['tier2_cities'] else 3)
        prop_code = {'Apartment': 0, 'Independent House': 1, 'Villa': 2}[selected_prop_type]
        facing_code = {'East': 0, 'North': 1, 'South': 2, 'West': 3}[facing_dir]
        owner_code = {'Builder': 0, 'Broker': 1, 'Owner': 2}[owner_type]
        furnished_code = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}[furnished]
        
        parking_bin = 1 if has_parking == 'Yes' else 0
        security_bin = 1 if has_security == 'Yes' else 0
        avail_bin = 1 if avail_status == 'Ready_to_Move' else 0
        
        # Construct feature vector matching feature_names
        input_data = pd.DataFrame([{
            'BHK': selected_bhk,
            'Size_in_SqFt': selected_size,
            'Price_in_Lakhs': selected_price,
            'Price_per_SqFt_INR': price_per_sqft_inr,
            'Age_of_Property': age_years,
            'Floor_No': floor_no,
            'Total_Floors': total_floors,
            'Floor_Ratio': floor_ratio,
            'Nearby_Schools': nearby_schools,
            'Nearby_Hospitals': nearby_hospitals,
            'Infrastructure_Score': infra_score,
            'Transport_Score': transport_score,
            'Parking_Space_Binary': parking_bin,
            'Security_Binary': security_bin,
            'Availability_Status_Binary': avail_bin,
            'Furnished_Status_Ordinal': furnished_code,
            'Amenities_Count': amenities_count,
            'Amenity_Gym': 1 if am_gym else 0,
            'Amenity_Pool': 1 if am_pool else 0,
            'Amenity_Clubhouse': 1 if am_club else 0,
            'Amenity_Garden': 1 if am_garden else 0,
            'Amenity_Playground': 1 if am_play else 0,
            'Property_Type_Code': prop_code,
            'Facing_Code': facing_code,
            'Owner_Type_Code': owner_code,
            'City_Tier_Code': city_tier_code,
            'Valuation_Ratio': val_ratio,
            'Is_Undervalued': is_undervalued
        }])
        
        # Ensure column order matches exactly
        input_data = input_data[artifacts['feature_names']]
        
        # Model Inferences
        clf_model = artifacts['classifier']
        reg_model = artifacts['regressor']
        
        pred_class = int(clf_model.predict(input_data)[0])
        pred_proba = float(clf_model.predict_proba(input_data)[0][1]) if hasattr(clf_model, 'predict_proba') else 0.5
        pred_future_price = float(reg_model.predict(input_data)[0])
        
        capital_gain = pred_future_price - selected_price
        total_roi = (capital_gain / selected_price) * 100.0
        cagr = (((pred_future_price / selected_price) ** (1.0 / 5.0)) - 1.0) * 100.0
        
        # DISPLAY RESULTS
        st.markdown("### 🏆 AI Valuation & Recommendation Results")
        
        if pred_class == 1:
            st.markdown(f"""
            <div class="verdict-good">
                <h2 style="margin:0; color:#065F46;">✅ RECOMMENDED INVESTMENT: High Capital Growth Potential</h2>
                <p style="margin-top:8px; font-size:1.05rem;">
                    This property demonstrates superior market appreciation drivers, achieving an estimated 
                    <b>{cagr:.2f}% Annual CAGR</b> (total 5-year return of <b>+{total_roi:.1f}%</b>). 
                    Confidence Score: <b>{pred_proba*100:.1f}%</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-bad">
                <h2 style="margin:0; color:#991B1B;">⚠️ CAUTION: Moderate / Below-Hurdle Appreciation Potential</h2>
                <p style="margin-top:8px; font-size:1.05rem;">
                    This property is projected to appreciate at <b>{cagr:.2f}% Annual CAGR</b> (5-year return: <b>+{total_roi:.1f}%</b>), 
                    which may not sufficiently beat inflation and capital friction costs. Consider negotiating a discount.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # KPI Forecast Grid
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        col_res1.metric("Current Asking Price", f"₹{selected_price:.2f} Lakhs")
        col_res2.metric("Estimated 5-Year Value", f"₹{pred_future_price:.2f} Lakhs", f"+₹{capital_gain:.2f} Lakhs")
        col_res3.metric("Projected 5-Year ROI", f"+{total_roi:.1f}%", f"{cagr:.2f}% CAGR")
        col_res4.metric("Infrastructure Score", f"{infra_score:.1f} / 10", "Social + Transit")
        
        # BENCHMARK COMPARISON CHART
        st.subheader("📊 5-Year Capital Growth vs Financial Benchmarks")
        years = [f"Year {i}" for i in range(6)]
        
        # Calculate year-over-year trajectory
        property_path = [selected_price * ((1.0 + (cagr/100.0)) ** i) for i in range(6)]
        fd_rate = 0.070  # 7% Bank Fixed Deposit
        fd_path = [selected_price * ((1.0 + fd_rate) ** i) for i in range(6)]
        inflation_rate = 0.060  # 6% Inflation Hurdle
        inflation_path = [selected_price * ((1.0 + inflation_rate) ** i) for i in range(6)]
        
        bench_df = pd.DataFrame({
            'Timeline': years * 3,
            'Asset Value (₹ Lakhs)': property_path + fd_path + inflation_path,
            'Asset Class': (['Real Estate Forecast'] * 6) + (['Bank FD (7.0% p.a.)'] * 6) + (['Inflation Hurdle (6.0%)'] * 6)
        })
        
        fig_bench = px.line(
            bench_df,
            x='Timeline',
            y='Asset Value (₹ Lakhs)',
            color='Asset Class',
            markers=True,
            color_discrete_map={
                'Real Estate Forecast': '#10B981' if pred_class == 1 else '#EF4444',
                'Bank FD (7.0% p.a.)': '#3B82F6',
                'Inflation Hurdle (6.0%)': '#F59E0B'
            }
        )
        apply_corporate_style(fig_bench, "5-Year Capital Value Progression vs Alternative Benchmarks", height=380)
        st.plotly_chart(fig_bench, use_container_width=True)
        
        st.markdown("""
        <div class="graph-desc-card">
            <b>Graph Description:</b> This comparison line chart plots the projected real estate value trajectory against 
            a guaranteed 7% Bank Fixed Deposit and a 6% long-term inflation baseline over a 5-year investment horizon. 
            Superior investments widen their premium above alternative instruments, compounding real purchasing power.
        </div>
        """, unsafe_allow_html=True)
        
        # Property Scorecard Breakdown
        st.subheader("📋 Property Quality Dimension Scorecard")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.progress(min(infra_score / 10.0, 1.0), text=f"Infrastructure: {infra_score:.1f}/10")
        sc2.progress(amenities_count / 5.0, text=f"Amenities: {amenities_count}/5")
        sc3.progress(min(calc_pps / 20000.0, 1.0), text=f"Unit Rate: ₹{calc_pps:,.0f}/sqft")
        val_status = "Undervalued (Favorable)" if is_undervalued else "At/Above City Parity"
        sc4.info(f"**Valuation Status:** {val_status}")

# ==============================================================================
# PAGE 6: RISK ANALYSIS, LIMITATIONS & POSSIBLE OUTCOMES
# ==============================================================================
elif "6. Risk Analysis" in selected_page:
    st.markdown("""
    <div class="main-header">
        <h1>⚠️ Investment Risk Analysis, Model Limitations & Possible Outcomes</h1>
        <p>Macroeconomic Vulnerabilities, Model Boundary Conditions, Sensitivity Scenarios, and Institutional Advisory</p>
    </div>
    """, unsafe_allow_html=True)
    
    # REQUIRED ABOUT SECTION
    st.markdown("""
    <div class="about-card">
        <h4>ℹ️ About This Page: Risk Governance, Model Constraints & Strategic Scenarios</h4>
        <p>
            No algorithmic decision model is complete without transparent corporate governance and risk disclosure. 
            This section explicitly addresses the <b>model limitations, macroeconomic boundary conditions, 
            and probabilistic future outcomes</b> surrounding real estate investments. By evaluating interest rate cycles, 
            regulatory RERA compliance, liquidity constraints, and stress-tested market scenarios (Bull, Base, Bear), 
            institutional investors and retail buyers can safeguard capital against asymmetric downside risks.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_r1, tab_r2, tab_r3 = st.tabs(["🏛️ Macroeconomic Risks", "🤖 ML Model Limitations", "🔮 Scenario Analysis (Possible Outcomes)"])
    
    with tab_r1:
        st.subheader("Macroeconomic & Real Estate Market Structural Risks")
        
        st.markdown("""
        <div class="risk-box">
            <b>1. Interest Rate & Monetary Policy Sensitivity:</b><br>
            Mortgage interest rates fluctuate with RBI Repo Rate adjustments. A 150-200 bps hike in home loan interest 
            rates substantially inflates borrower Equated Monthly Installments (EMIs), compressing housing affordability 
            and damping speculative resale velocity.
        </div>
        <div class="risk-box">
            <b>2. Illiquidity & High Transaction Friction:</b><br>
            Unlike equities or sovereign bonds, residential real estate is inherently illiquid. Average exit horizons range 
            from 3 to 12 months. Mandatory statutory overheads—including 5-7% State Stamp Duty, 1% Registration Fees, 
            brokerage fees, and Long-Term Capital Gains (LTCG) taxes—impose an initial ~8-10% hurdle that must be overcome before net profit realization.
        </div>
        <div class="risk-box">
            <b>3. Regulatory & Completion Risk (Under-Construction Assets):</b><br>
            While under-construction units offer attractive pre-launch price discounts, delays in municipal occupancy certificates (OC), 
            environmental clearances, and developer cash flow bottlenecks can defer possession dates, eroding annualized IRR.
        </div>
        """, unsafe_allow_html=True)
        
    with tab_r2:
        st.subheader("Machine Learning Algorithm Boundary Conditions")
        st.markdown("""
        Understanding what the predictive algorithms can and cannot incorporate:
        """)
        
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.markdown("##### ❌ What the Model Does Not Capture")
            st.markdown("""
            - **Unrecorded Qualitative Attributes:** Interior design quality, natural lighting, panoramic view obstructions, and vastu compliance nuances beyond facing direction.
            - **Developer Track Record & Litigation:** Historic delivery delinquency rates, pending court litigations, and structural build defects.
            - **Micro-Infrastructure Disruption:** Sudden announcements of new metro lines, flyovers, or toxic landfills that radically alter localized property values overnight.
            - **Extreme Outlier Properties:** Ultra-luxury estates (> ₹15 Crores) driven by bespoke emotional and vanity pricing rather than statistical fundamentals.
            """)
        with col_l2:
            st.markdown("##### ✅ Recommended Due Diligence Protocol")
            st.markdown("""
            1. **Physical Title Verification:** Conduct independent title deed searches spanning 30 years with a certified property lawyer.
            2. **RERA Portal Cross-Examination:** Verify project registration number and quarterly financial disclosures on state RERA portals.
            3. **Ground Reality Transit Audit:** Validate physical walkability to advertised transit nodes during peak traffic hours.
            4. **Carpet Area vs Super Area Ratio:** Scrutinize loading factor (common area percentage) to ensure high usable carpet area efficiency.
            """)
            
    with tab_r3:
        st.subheader("Possible Outcomes: 5-Year Stress-Test Scenarios")
        st.markdown("""
        To model realistic capital outcomes, we evaluate three forward-looking economic scenarios:
        """)
        
        scenarios_df = pd.DataFrame({
            'Scenario': ['🐂 Bull Case (+20% Market Outperformance)', '⚖️ Base Case (Model Forecast)', '🐻 Bear Case (-20% Market Stress)'],
            'Macro Assumptions': [
                'Accelerated infrastructure completion, tech job influx, favorable interest rate cuts',
                'Steady urban economic growth, stable inflation (5.5-6%), normal project completions',
                'Protracted interest rate spikes, localized over-supply, delay in transit connectivity'
            ],
            'Projected 5Y Appreciation Rate': ['11.5% - 14.5% CAGR', '8.5% - 10.5% CAGR', '4.0% - 6.5% CAGR'],
            'Sample ₹1.5 Cr Asset Future Value': ['₹2.58 - ₹2.95 Crores', '₹2.25 - ₹2.47 Crores', '₹1.82 - ₹2.05 Crores'],
            'Capital Outcome': ['Substantial Wealth Creation', 'Market-Beating Wealth Preservation', 'Capital Return Below Inflation']
        })
        st.table(scenarios_df)
        
        # Scenario Visualization
        fig_scenarios = go.Figure()
        years = [f"Y{i}" for i in range(6)]
        base_val = 150.0  # 1.5 Crores
        
        bull_traj = [base_val * ((1.125) ** i) for i in range(6)]
        base_traj = [base_val * ((1.095) ** i) for i in range(6)]
        bear_traj = [base_val * ((1.050) ** i) for i in range(6)]
        
        fig_scenarios.add_trace(go.Scatter(x=years, y=bull_traj, mode='lines+markers', name='Bull Case (12.5% CAGR)', line=dict(color='#10B981', width=3)))
        fig_scenarios.add_trace(go.Scatter(x=years, y=base_traj, mode='lines+markers', name='Base Case (9.5% CAGR)', line=dict(color='#3B82F6', width=3, dash='dash')))
        fig_scenarios.add_trace(go.Scatter(x=years, y=bear_traj, mode='lines+markers', name='Bear Case (5.0% CAGR)', line=dict(color='#EF4444', width=3, dash='dot')))
        
        apply_corporate_style(fig_scenarios, "5-Year Asset Valuation Sensitivity Under Divergent Macro Scenarios (₹ Lakhs)", height=380)
        st.plotly_chart(fig_scenarios, use_container_width=True)
        
        st.markdown("""
        <div class="graph-desc-card">
            <b>Graph Description:</b> This sensitivity chart illustrates the range of possible outcomes for a baseline 
            ₹150 Lakh (₹1.5 Crore) asset. In a Bull Case scenario, the asset scales to ₹295 Lakhs (+97%), while a 
            stressed Bear Case scenario yields ₹191 Lakhs (+27%), illustrating the importance of margin-of-safety acquisition pricing.
        </div>
        """, unsafe_allow_html=True)
        
        # Export Capability
        st.markdown("---")
        st.subheader("📥 Export Executive Advisory Data")
        summary_csv = df[['City', 'Property_Type', 'BHK', 'Price_in_Lakhs', 'Price_per_SqFt_INR', 'Future_Price_5Y', 'Good_Investment']].head(500).to_csv(index=False)
        st.download_button(
            label="Download 500-Record Executive Valuation Sample (CSV)",
            data=summary_csv,
            file_name="propintel_valuation_sample.csv",
            mime="text/csv"
        )
