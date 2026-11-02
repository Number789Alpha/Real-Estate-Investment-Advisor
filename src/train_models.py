"""
Real Estate Investment Advisor - Model Training and Evaluation Pipeline
Trains, evaluates, and serializes classification & regression models for real estate decision support.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

def load_data(parquet_path='data/processed_housing_data.parquet'):
    """Load the preprocessed housing dataset."""
    print(f"Loading preprocessed data from '{parquet_path}'...")
    df = pd.read_parquet(parquet_path)
    return df

def prepare_feature_matrix(df):
    """
    Select high-signal input features for modeling:
    Features include property dimensions, pricing, infrastructure scores,
    amenities binary flags, building geometry, city codes, and property types.
    """
    feature_cols = [
        'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt_INR',
        'Age_of_Property', 'Floor_No', 'Total_Floors', 'Floor_Ratio',
        'Nearby_Schools', 'Nearby_Hospitals', 'Infrastructure_Score',
        'Transport_Score', 'Parking_Space_Binary', 'Security_Binary',
        'Availability_Status_Binary', 'Furnished_Status_Ordinal',
        'Amenities_Count', 'Amenity_Gym', 'Amenity_Pool', 'Amenity_Clubhouse',
        'Amenity_Garden', 'Amenity_Playground', 'Property_Type_Code',
        'Facing_Code', 'Owner_Type_Code', 'City_Tier_Code', 'Valuation_Ratio',
        'Is_Undervalued'
    ]
    
    X = df[feature_cols].copy()
    y_class = df['Good_Investment'].values
    y_reg = df['Future_Price_5Y'].values
    
    return X, y_class, y_reg, feature_cols

def train_and_evaluate_classification(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple classification algorithms and benchmark performance."""
    print("\n" + "=" * 50)
    print("TRAINING CLASSIFICATION MODELS (Good Investment: 0 vs 1)")
    print("=" * 50)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42, eval_metric='logloss', n_jobs=-1)
    }
    
    results = {}
    best_model_name = None
    best_f1 = -1
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name} Classifier...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
        else:
            auc = 0.5
            
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
            'confusion_matrix': cm
        }
        
        print(f"-> {name} | Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            
    print(f"\nBest Classifier: {best_model_name} (F1 Score: {best_f1:.4f})")
    
    # Feature Importances for best tree model
    feat_imp = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx:
            feat_imp[feature_names[idx]] = float(round(importances[idx], 4))
            
    return best_model, best_model_name, results, feat_imp

def train_and_evaluate_regression(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple regression algorithms to forecast 5-Year Property Value."""
    print("\n" + "=" * 50)
    print("TRAINING REGRESSION MODELS (5-Year Estimated Price)")
    print("=" * 50)
    
    models = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1),
        'XGBoost Regressor': XGBRegressor(n_estimators=140, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    }
    
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100.0)
        
        results[name] = {
            'rmse': round(rmse, 2),
            'mae': round(mae, 2),
            'r2_score': round(r2, 4),
            'mape': round(mape, 2)
        }
        
        print(f"-> {name} | RMSE: INR {rmse:.2f}L | MAE: INR {mae:.2f}L | R2: {r2:.4f} | MAPE: {mape:.2f}%")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model
            
    print(f"\nBest Regressor: {best_model_name} (R2 Score: {best_r2:.4f})")
    
    # Feature Importances for best tree model
    feat_imp = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx:
            feat_imp[feature_names[idx]] = float(round(importances[idx], 4))
            
    return best_model, best_model_name, results, feat_imp

def run_model_training_pipeline(sample_size=60000):
    """
    Executes training workflow on a representative stratified dataset
    for optimal speed and enterprise-grade generalization.
    """
    df = load_data()
    
    # Use stratified sample if dataset is huge for rapid training while retaining full distribution
    if len(df) > sample_size:
        print(f"Sampling {sample_size:,} records (stratified by Good_Investment) for responsive training...")
        df_sample, _ = train_test_split(
            df, train_size=sample_size, random_state=42, stratify=df['Good_Investment']
        )
        df_sample = df_sample.reset_index(drop=True)
    else:
        df_sample = df
        
    X, y_class, y_reg, feature_names = prepare_feature_matrix(df_sample)
    
    # Train / Test split (80/20)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
    )
    print(f"Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")
    
    # 1. Train Classification
    best_clf, best_clf_name, clf_metrics, clf_feat_imp = train_and_evaluate_classification(
        X_train, X_test, y_class_train, y_class_test, feature_names
    )
    
    # 2. Train Regression
    best_reg, best_reg_name, reg_metrics, reg_feat_imp = train_and_evaluate_regression(
        X_train, X_test, y_reg_train, y_reg_test, feature_names
    )
    
    # 3. Save artifacts
    print("\nSaving trained models and evaluation reports...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(best_clf, 'models/best_classifier.joblib')
    joblib.dump(best_reg, 'models/best_regressor.joblib')
    joblib.dump(feature_names, 'models/feature_names.joblib')
    
    all_metrics = {
        'classification': {
            'best_model': best_clf_name,
            'models': clf_metrics
        },
        'regression': {
            'best_model': best_reg_name,
            'models': reg_metrics
        }
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
        
    feature_importances = {
        'classification': clf_feat_imp,
        'regression': reg_feat_imp
    }
    with open('models/feature_importance.json', 'w') as f:
        json.dump(feature_importances, f, indent=2)
        
    print("All models and metrics serialized successfully into 'models/' folder.")

if __name__ == '__main__':
    run_model_training_pipeline()
