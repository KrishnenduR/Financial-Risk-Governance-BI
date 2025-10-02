#!/usr/bin/env python3
"""
Enhanced Model Training Script

This script provides improved model training with better hyperparameters,
proper missing value handling, and enhanced stress testing for financial
development prediction models.

Features:
- Advanced hyperparameter tuning for LightGBM and XGBoost
- NaN-tolerant models with proper preprocessing pipelines
- Enhanced stress testing scenarios
- Robust cross-validation with proper data handling
- Economic shock resilience training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# Advanced models
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
import optuna

# Data processing
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.gfd_preprocessor import GFDPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """
    Enhanced model trainer with advanced hyperparameter optimization,
    proper missing value handling, and stress testing capabilities.
    """
    
    def __init__(self, data_path: str = "data/gfd_database.db"):
        """Initialize the enhanced trainer."""
        self.data_path = data_path
        self.preprocessor = GFDPreprocessor()
        self.models = {}
        self.pipelines = {}
        self.results = {}
        self.stress_test_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data with proper preprocessing."""
        import sqlite3
        
        # Load data from database
        conn = sqlite3.connect(self.data_path)
        query = """
        SELECT 
            country_code,
            country_name,
            year,
            region,
            income_group,
            overall_financial_development_index,
            access_institutions_index,
            access_markets_index,
            depth_institutions_index,
            depth_markets_index,
            efficiency_institutions_index,
            efficiency_markets_index,
            stability_institutions_index,
            stability_markets_index
        FROM financial_development_data 
        WHERE year >= 2000
        ORDER BY country_code, year
        """
        
        self.data = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(self.data)} records")
        
        # Basic preprocessing
        self.data = self.data.dropna(subset=['overall_financial_development_index'])
        
        # Create feature matrix and target
        feature_cols = [
            'access_institutions_index', 'access_markets_index',
            'depth_institutions_index', 'depth_markets_index', 
            'efficiency_institutions_index', 'efficiency_markets_index',
            'stability_institutions_index', 'stability_markets_index'
        ]
        
        # Add encoded categorical features
        from sklearn.preprocessing import LabelEncoder
        le_region = LabelEncoder()
        le_income = LabelEncoder()
        
        self.data['region_encoded'] = le_region.fit_transform(self.data['region'].fillna('Unknown'))
        self.data['income_encoded'] = le_income.fit_transform(self.data['income_group'].fillna('Unknown'))
        
        # Add temporal features
        self.data['year_normalized'] = (self.data['year'] - self.data['year'].min()) / (self.data['year'].max() - self.data['year'].min())
        
        feature_cols.extend(['region_encoded', 'income_encoded', 'year_normalized'])
        
        self.X = self.data[feature_cols].copy()
        self.y = self.data['overall_financial_development_index'].copy()
        
        # Split data with temporal consideration
        # Use last 20% of years for testing
        years = sorted(self.data['year'].unique())
        test_years = years[-int(len(years) * 0.2):]
        
        train_mask = ~self.data['year'].isin(test_years)
        test_mask = self.data['year'].isin(test_years)
        
        self.X_train = self.X[train_mask]
        self.X_test = self.X[test_mask]
        self.y_train = self.y[train_mask]
        self.y_test = self.y[test_mask]
        
        logger.info(f"Training set: {len(self.X_train)}, Test set: {len(self.X_test)}")
        
    def create_enhanced_pipelines(self):
        """Create enhanced ML pipelines with proper missing value handling."""
        
        # Define preprocessing for different model types
        numeric_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Pipeline for models that need complete data
        complete_data_preprocessor = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', RobustScaler())
        ])
        
        # Pipeline for models that can handle some NaNs
        partial_nan_preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        # Enhanced LightGBM with optimized hyperparameters
        lgb_enhanced = lgb.LGBMRegressor(
            objective='regression',
            num_leaves=64,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            max_depth=8,
            min_data_in_leaf=15,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_estimators=200,
            random_state=42,
            verbosity=-1
        )
        
        # Enhanced XGBoost with better hyperparameters
        xgb_enhanced = xgb.XGBRegressor(
            learning_rate=0.05,
            max_depth=7,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=3,
            random_state=42,
            eval_metric='rmse',
            tree_method='hist'
        )
        
        # Random Forest with enhanced economic shock handling
        rf_robust = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.8,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # HistGradientBoosting (handles NaNs natively)
        hist_gb = HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_iter=150,
            max_depth=8,
            min_samples_leaf=15,
            l2_regularization=0.1,
            random_state=42
        )
        
        # Enhanced Neural Network
        nn_enhanced = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            learning_rate='adaptive',
            learning_rate_init=0.01,
            alpha=0.1,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        )
        
        # Enhanced Elastic Net
        elastic_enhanced = ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=2000,
            random_state=42
        )
        
        # Create pipelines
        self.pipelines = {
            'lightgbm_enhanced': Pipeline([
                ('preprocessor', partial_nan_preprocessor),
                ('model', lgb_enhanced)
            ]),
            'xgboost_enhanced': Pipeline([
                ('preprocessor', partial_nan_preprocessor),
                ('model', xgb_enhanced)
            ]),
            'random_forest_robust': Pipeline([
                ('preprocessor', complete_data_preprocessor),
                ('model', rf_robust)
            ]),
            'hist_gradient_boosting': Pipeline([
                ('preprocessor', SimpleImputer(strategy='median')),  # Minimal preprocessing
                ('model', hist_gb)
            ]),
            'neural_network_enhanced': Pipeline([
                ('preprocessor', complete_data_preprocessor),
                ('model', nn_enhanced)
            ]),
            'elastic_net_enhanced': Pipeline([
                ('preprocessor', complete_data_preprocessor),
                ('model', elastic_enhanced)
            ])
        }
        
        logger.info(f"Created {len(self.pipelines)} enhanced model pipelines")
    
    def hyperparameter_optimization(self, model_name: str, n_trials: int = 50):
        """
        Perform hyperparameter optimization using Optuna.
        
        Args:
            model_name: Name of the model to optimize
            n_trials: Number of optimization trials
        """
        def objective(trial):
            """Objective function for hyperparameter optimization."""
            
            if model_name == 'lightgbm_enhanced':
                params = {
                    'model__num_leaves': trial.suggest_int('model__num_leaves', 10, 100),
                    'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.2),
                    'model__max_depth': trial.suggest_int('model__max_depth', 3, 12),
                    'model__min_data_in_leaf': trial.suggest_int('model__min_data_in_leaf', 5, 50),
                    'model__reg_alpha': trial.suggest_float('model__reg_alpha', 0.0, 1.0),
                    'model__reg_lambda': trial.suggest_float('model__reg_lambda', 0.0, 1.0),
                    'model__n_estimators': trial.suggest_int('model__n_estimators', 50, 300)
                }
                
            elif model_name == 'xgboost_enhanced':
                params = {
                    'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.2),
                    'model__max_depth': trial.suggest_int('model__max_depth', 3, 12),
                    'model__n_estimators': trial.suggest_int('model__n_estimators', 50, 300),
                    'model__subsample': trial.suggest_float('model__subsample', 0.6, 1.0),
                    'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.6, 1.0),
                    'model__reg_alpha': trial.suggest_float('model__reg_alpha', 0.0, 1.0),
                    'model__reg_lambda': trial.suggest_float('model__reg_lambda', 0.0, 1.0)
                }
                
            elif model_name == 'random_forest_robust':
                params = {
                    'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 300),
                    'model__max_depth': trial.suggest_int('model__max_depth', 5, 20),
                    'model__min_samples_split': trial.suggest_int('model__min_samples_split', 5, 20),
                    'model__min_samples_leaf': trial.suggest_int('model__min_samples_leaf', 2, 10),
                    'model__max_features': trial.suggest_float('model__max_features', 0.5, 1.0)
                }
                
            else:
                return float('inf')  # Skip optimization for other models
            
            # Set parameters
            pipeline = self.pipelines[model_name]
            pipeline.set_params(**params)
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=TimeSeriesSplit(n_splits=3),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            return -cv_scores.mean()  # Return positive MSE
        
        # Optimize hyperparameters
        if model_name in ['lightgbm_enhanced', 'xgboost_enhanced', 'random_forest_robust']:
            logger.info(f"Optimizing hyperparameters for {model_name}")
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            # Update pipeline with best parameters
            self.pipelines[model_name].set_params(**study.best_params)
            
            logger.info(f"Best parameters for {model_name}: {study.best_params}")
            logger.info(f"Best score: {study.best_value:.6f}")
    
    def train_models(self, optimize_hyperparameters: bool = True):
        """Train all models with optional hyperparameter optimization."""
        logger.info("Starting enhanced model training")
        
        # Hyperparameter optimization for priority models
        if optimize_hyperparameters:
            priority_models = ['lightgbm_enhanced', 'xgboost_enhanced', 'random_forest_robust']
            for model_name in priority_models:
                self.hyperparameter_optimization(model_name, n_trials=30)
        
        # Train all models
        for model_name, pipeline in self.pipelines.items():
            logger.info(f"Training {model_name}")
            
            try:
                pipeline.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred_train = pipeline.predict(self.X_train)
                y_pred_test = pipeline.predict(self.X_test)
                
                # Calculate metrics
                train_mse = mean_squared_error(self.y_train, y_pred_train)
                test_mse = mean_squared_error(self.y_test, y_pred_test)
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                
                self.results[model_name] = {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': np.sqrt(train_mse),
                    'test_rmse': np.sqrt(test_mse),
                    'test_mae': mean_absolute_error(self.y_test, y_pred_test),
                    'predictions': y_pred_test
                }
                
                logger.info(f"{model_name} - Test R²: {test_r2:.4f}, Test RMSE: {np.sqrt(test_mse):.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
    
    def enhanced_stress_testing(self):
        """Perform enhanced stress testing with economic shock scenarios."""
        logger.info("Performing enhanced stress testing")
        
        # Define enhanced stress scenarios
        stress_scenarios = {
            'economic_shock': {
                'access_institutions_index': -0.3,
                'depth_institutions_index': -0.25,
                'efficiency_institutions_index': -0.2
            },
            'market_volatility': {
                'access_markets_index': -0.4,
                'depth_markets_index': -0.35,
                'efficiency_markets_index': -0.3
            },
            'financial_crisis': {
                'access_institutions_index': -0.4,
                'access_markets_index': -0.5,
                'stability_institutions_index': -0.6,
                'stability_markets_index': -0.7
            },
            'regulatory_tightening': {
                'access_institutions_index': -0.15,
                'access_markets_index': -0.25,
                'efficiency_institutions_index': -0.1,
                'efficiency_markets_index': -0.2
            },
            'pandemic_shock': {
                'access_institutions_index': -0.2,
                'access_markets_index': -0.3,
                'efficiency_institutions_index': -0.25,
                'stability_markets_index': -0.4
            }
        }
        
        baseline_predictions = {}
        
        # Get baseline predictions for each trained model
        for model_name, pipeline in self.pipelines.items():
            if model_name not in self.results or 'error' in self.results[model_name]:
                continue
                
            try:
                baseline_pred = pipeline.predict(self.X_test)
                baseline_mse = mean_squared_error(self.y_test, baseline_pred)
                baseline_predictions[model_name] = baseline_mse
                
                self.stress_test_results[model_name] = {}
                
                # Test each stress scenario
                for scenario_name, shocks in stress_scenarios.items():
                    X_stressed = self.X_test.copy()
                    
                    # Apply shocks
                    for feature, shock in shocks.items():
                        if feature in X_stressed.columns:
                            X_stressed[feature] = X_stressed[feature] * (1 + shock)
                    
                    # Get stressed predictions
                    stressed_pred = pipeline.predict(X_stressed)
                    stressed_mse = mean_squared_error(self.y_test, stressed_pred)
                    
                    # Calculate performance degradation
                    mse_change_pct = ((stressed_mse - baseline_mse) / baseline_mse) * 100
                    
                    self.stress_test_results[model_name][scenario_name] = {
                        'baseline_mse': baseline_mse,
                        'stressed_mse': stressed_mse,
                        'mse_change_pct': mse_change_pct,
                        'resilience_score': max(0, 100 - abs(mse_change_pct))
                    }
                    
                    logger.info(f"{model_name} - {scenario_name}: {mse_change_pct:+.1f}% MSE change")
                    
            except Exception as e:
                logger.error(f"Error in stress testing for {model_name}: {e}")
                self.stress_test_results[model_name] = {'error': str(e)}
    
    def generate_enhanced_report(self) -> str:
        """Generate enhanced training and validation report."""
        report = []
        report.append("=" * 80)
        report.append("ENHANCED MODEL TRAINING AND VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models Trained: {len([r for r in self.results.values() if 'error' not in r])}")
        report.append("")
        
        # Model Performance Summary
        report.append("1. MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        for model_name, results in self.results.items():
            if 'error' not in results:
                report.append(f"{model_name.upper()}:")
                report.append(f"  Test R²: {results['test_r2']:.4f}")
                report.append(f"  Test RMSE: {results['test_rmse']:.4f}")
                report.append(f"  Test MAE: {results['test_mae']:.4f}")
                report.append("")
        
        # Stress Testing Results
        if self.stress_test_results:
            report.append("2. ENHANCED STRESS TESTING RESULTS")
            report.append("-" * 40)
            
            for model_name, stress_results in self.stress_test_results.items():
                if 'error' not in stress_results:
                    report.append(f"{model_name.upper()}:")
                    
                    for scenario, results in stress_results.items():
                        mse_change = results['mse_change_pct']
                        resilience = results['resilience_score']
                        report.append(f"  {scenario}: {mse_change:+.1f}% MSE change (Resilience: {resilience:.1f})")
                    
                    report.append("")
        
        # Recommendations
        report.append("3. RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Find best overall model
        best_model = None
        best_r2 = -1
        for model_name, results in self.results.items():
            if 'error' not in results and results['test_r2'] > best_r2:
                best_r2 = results['test_r2']
                best_model = model_name
        
        if best_model:
            report.append(f"Best Overall Model: {best_model.upper()} (R² = {best_r2:.4f})")
        
        # Find most resilient model
        if self.stress_test_results:
            resilience_scores = {}
            for model_name, stress_results in self.stress_test_results.items():
                if 'error' not in stress_results:
                    avg_resilience = np.mean([r['resilience_score'] for r in stress_results.values()])
                    resilience_scores[model_name] = avg_resilience
            
            if resilience_scores:
                most_resilient = max(resilience_scores.items(), key=lambda x: x[1])
                report.append(f"Most Resilient Model: {most_resilient[0].upper()} (Avg Resilience: {most_resilient[1]:.1f})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\\n".join(report)
    
    def save_enhanced_results(self, output_dir: str = "models/enhanced_results"):
        """Save enhanced training results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_data = {
            'model_performance': self.results,
            'stress_testing': self.stress_test_results,
            'training_config': {
                'data_source': self.data_path,
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'features': list(self.X.columns)
            },
            'timestamp': timestamp
        }
        
        with open(output_path / f"enhanced_results_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str, ensure_ascii=False)
        
        # Save report
        report = self.generate_enhanced_report()
        with open(output_path / f"enhanced_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save trained models
        import joblib
        for model_name, pipeline in self.pipelines.items():
            if model_name in self.results and 'error' not in self.results[model_name]:
                model_path = output_path / f"{model_name}_{timestamp}.joblib"
                joblib.dump(pipeline, model_path)
        
        logger.info(f"Enhanced results saved to {output_path}")
        print(f"Results saved to: {output_path}")


def main():
    """Main function to run enhanced model training."""
    try:
        # Initialize trainer
        trainer = EnhancedModelTrainer()
        
        # Load and prepare data
        print("Loading and preparing data...")
        trainer.load_and_prepare_data()
        
        # Create enhanced pipelines
        print("Creating enhanced model pipelines...")
        trainer.create_enhanced_pipelines()
        
        # Train models with optimization
        print("Training models with hyperparameter optimization...")
        trainer.train_models(optimize_hyperparameters=True)
        
        # Perform enhanced stress testing
        print("Performing enhanced stress testing...")
        trainer.enhanced_stress_testing()
        
        # Generate and print report
        report = trainer.generate_enhanced_report()
        print("\\n" + report)
        
        # Save results
        print("Saving enhanced results...")
        trainer.save_enhanced_results()
        
        print("\\nEnhanced model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Enhanced model training failed: {e}")
        raise


if __name__ == "__main__":
    main()