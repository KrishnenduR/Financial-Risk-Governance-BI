#!/usr/bin/env python3
"""
Quick test for enhanced model training with proper missing value handling.
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def test_enhanced_models():
    """Test enhanced models with proper missing value handling."""
    print("Testing Enhanced Models with Proper Missing Value Handling")
    print("=" * 60)
    
    # Load data
    conn = sqlite3.connect('data/gfd_database.db')
    query = """
    SELECT 
        country_code, year, region, income_group,
        overall_financial_development_index,
        access_institutions_index, access_markets_index,
        depth_institutions_index, depth_markets_index,
        efficiency_institutions_index, efficiency_markets_index,
        stability_institutions_index, stability_markets_index
    FROM financial_development_data 
    WHERE year >= 2000
    ORDER BY country_code, year
    """
    
    data = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(data)} records")
    
    # Basic preprocessing
    data = data.dropna(subset=['overall_financial_development_index'])
    
    # Create features
    feature_cols = [
        'access_institutions_index', 'access_markets_index',
        'depth_institutions_index', 'depth_markets_index', 
        'efficiency_institutions_index', 'efficiency_markets_index',
        'stability_institutions_index', 'stability_markets_index'
    ]
    
    # Add encoded categorical features
    le_region = LabelEncoder()
    le_income = LabelEncoder()
    
    data['region_encoded'] = le_region.fit_transform(data['region'].fillna('Unknown'))
    data['income_encoded'] = le_income.fit_transform(data['income_group'].fillna('Unknown'))
    data['year_normalized'] = (data['year'] - data['year'].min()) / (data['year'].max() - data['year'].min())
    
    feature_cols.extend(['region_encoded', 'income_encoded', 'year_normalized'])
    
    X = data[feature_cols].copy()
    y = data['overall_financial_development_index'].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Create enhanced pipelines with proper missing value handling
    models = {}
    
    # 1. HistGradientBoosting (handles NaNs natively)
    models['hist_gradient_boosting'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Light preprocessing
        ('model', HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=100,
            max_depth=6,
            random_state=42
        ))
    ])
    
    # 2. Enhanced LightGBM with proper preprocessing
    models['lightgbm_enhanced'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('model', lgb.LGBMRegressor(
            objective='regression',
            num_leaves=31,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            verbosity=-1
        ))
    ])
    
    # 3. Enhanced XGBoost with proper preprocessing
    models['xgboost_enhanced'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('model', xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100,
            random_state=42,
            eval_metric='rmse'
        ))
    ])
    
    # 4. Robust Random Forest with complete data imputation
    models['random_forest_robust'] = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler()),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("\\nTesting Models:")
    print("-" * 40)
    
    results = {}
    
    # Train and test models
    for model_name, pipeline in models.items():
        print(f"Training {model_name}...")
        
        try:
            # Fit the pipeline
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=5, scoring='r2', n_jobs=-1
            )
            
            results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'status': 'success'
            }
            
            print(f"✅ {model_name}:")
            print(f"   Test R²: {test_r2:.4f}")
            print(f"   Test RMSE: {test_rmse:.4f}")
            print(f"   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            results[model_name] = {'status': 'failed', 'error': str(e)}
    
    # Test with stress scenarios
    print("\\nTesting Stress Scenarios:")
    print("-" * 40)
    
    stress_scenarios = {
        'economic_shock': {
            'access_institutions_index': -0.2,
            'depth_institutions_index': -0.15,
            'efficiency_institutions_index': -0.1
        },
        'market_volatility': {
            'access_markets_index': -0.3,
            'depth_markets_index': -0.25,
            'efficiency_markets_index': -0.2
        }
    }
    
    for model_name, pipeline in models.items():
        if results[model_name]['status'] != 'success':
            continue
            
        print(f"\\n{model_name} stress testing:")
        baseline_pred = pipeline.predict(X_test)
        baseline_mse = mean_squared_error(y_test, baseline_pred)
        
        for scenario_name, shocks in stress_scenarios.items():
            X_stressed = X_test.copy()
            
            # Apply shocks
            for feature, shock in shocks.items():
                if feature in X_stressed.columns:
                    X_stressed[feature] = X_stressed[feature] * (1 + shock)
            
            # Get stressed predictions
            try:
                stressed_pred = pipeline.predict(X_stressed)
                stressed_mse = mean_squared_error(y_test, stressed_pred)
                mse_change_pct = ((stressed_mse - baseline_mse) / baseline_mse) * 100
                
                print(f"  {scenario_name}: {mse_change_pct:+.1f}% MSE change")
                
            except Exception as e:
                print(f"  {scenario_name}: Error - {e}")
    
    print("\\nSummary:")
    print("-" * 40)
    
    successful_models = {k: v for k, v in results.items() if v['status'] == 'success'}
    if successful_models:
        best_model = max(successful_models.items(), key=lambda x: x[1]['test_r2'])
        print(f"Best model: {best_model[0]} (R² = {best_model[1]['test_r2']:.4f})")
        print(f"Models working: {len(successful_models)}/{len(models)}")
    else:
        print("No models completed successfully")
    
    print("\\n✅ Enhanced model testing completed!")

if __name__ == "__main__":
    test_enhanced_models()