#!/usr/bin/env python3
"""
Predictive Models for Global Financial Development Analysis

This module implements advanced machine learning models for predicting financial development
indicators, conducting risk assessment, and performing trend analysis using the Global
Financial Development Database.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class GFDPredictiveModels:
    """
    Comprehensive predictive modeling suite for Global Financial Development indicators.
    
    Features:
    - Multiple ML algorithms for different prediction tasks
    - Time series forecasting with LSTM networks
    - Country risk classification models
    - Financial development trend analysis
    - Model ensembling and stacking
    - Automated hyperparameter tuning
    - Model validation and performance tracking
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the predictive modeling suite.
        
        Args:
            data_path: Path to the processed data file or database
        """
        self.data_path = data_path or "data/gfd_database.db"
        self.data = None
        self.models = {}
        self.model_performances = {}
        self.feature_importance = {}
        self.predictions = {}
        
        # Model configurations
        self.model_configs = self._get_model_configurations()
        
        # Load processed data
        self._load_data()
    
    def _get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations for different prediction tasks."""
        return {
            'financial_development_prediction': {
                'algorithms': [
                    'random_forest', 'gradient_boosting', 'xgboost', 
                    'lightgbm', 'neural_network', 'lstm'
                ],
                'target_variable': 'overall_financial_development_index',
                'task_type': 'regression',
                'time_horizon': [1, 3, 5],  # years ahead
                'cross_validation': 'time_series'
            },
            'risk_classification': {
                'algorithms': [
                    'random_forest', 'gradient_boosting', 'xgboost', 'neural_network'
                ],
                'target_variable': 'risk_category',
                'task_type': 'classification',
                'classes': ['Low', 'Medium', 'High', 'Very High'],
                'cross_validation': 'stratified'
            },
            'indicator_forecasting': {
                'algorithms': ['lstm', 'arima', 'prophet'],
                'target_variables': [
                    'access_institutions_index', 'depth_institutions_index',
                    'efficiency_institutions_index', 'stability_institutions_index'
                ],
                'task_type': 'time_series',
                'sequence_length': 10,
                'forecast_horizon': 5
            },
            'volatility_prediction': {
                'algorithms': ['gradient_boosting', 'xgboost', 'neural_network'],
                'target_variable': 'market_volatility_score',
                'task_type': 'regression',
                'feature_engineering': ['volatility_features', 'trend_features'],
                'cross_validation': 'time_series'
            }
        }
    
    def _load_data(self) -> None:
        """Load processed data from database or CSV file."""
        try:
            if str(self.data_path).endswith('.db'):
                # Load from SQLite database
                conn = sqlite3.connect(self.data_path)
                self.data = pd.read_sql_query("""
                    SELECT f.*, c.financial_development_score, c.systemic_risk_score,
                           c.market_volatility_score, c.institutional_quality_score, c.risk_category
                    FROM financial_development_data f
                    LEFT JOIN country_risk_profiles c 
                    ON f.country_code = c.country_code AND f.year = c.year
                    ORDER BY f.country_code, f.year
                """, conn)
                conn.close()
            else:
                # Load from CSV file
                self.data = pd.read_csv(self.data_path)
            
            # Parse raw indicators JSON
            if 'raw_indicators' in self.data.columns:
                raw_indicators = self.data['raw_indicators'].apply(
                    lambda x: json.loads(x) if pd.notna(x) and x != '' else {}
                )
                raw_indicators_df = pd.json_normalize(raw_indicators)
                self.data = pd.concat([self.data, raw_indicators_df], axis=1)
            
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features_and_targets(self, task_config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features and target variables for modeling.
        
        Args:
            task_config: Configuration for the modeling task
            
        Returns:
            Tuple of (features, targets, feature_names)
        """
        # Select feature columns
        feature_cols = []
        
        # Composite indices
        composite_indices = [
            'access_institutions_index', 'access_markets_index',
            'depth_institutions_index', 'depth_markets_index', 
            'efficiency_institutions_index', 'efficiency_markets_index',
            'stability_institutions_index', 'stability_markets_index'
        ]
        
        # Add available composite indices
        for col in composite_indices:
            if col in self.data.columns:
                feature_cols.append(col)
        
        # Add engineered features
        engineered_features = [col for col in self.data.columns 
                              if any(suffix in col for suffix in ['_lag1', '_ma3', '_growth', '_volatility', '_trend', '_relative_to'])]
        feature_cols.extend(engineered_features)
        
        # Add raw indicators
        raw_indicator_cols = [col for col in self.data.columns 
                             if col.startswith(('ai', 'am', 'di', 'dm', 'ei', 'em', 'si', 'sm', 'oi', 'om'))]
        feature_cols.extend(raw_indicator_cols[:50])  # Limit to top 50 to avoid overfitting
        
        # Add temporal features
        if 'year' in self.data.columns:
            feature_cols.append('year')
        
        # Encode categorical variables
        categorical_features = []
        if 'income_group' in self.data.columns:
            le_income = LabelEncoder()
            self.data['income_group_encoded'] = le_income.fit_transform(self.data['income_group'].fillna('Unknown'))
            feature_cols.append('income_group_encoded')
            categorical_features.append('income_group_encoded')
        
        if 'region' in self.data.columns:
            le_region = LabelEncoder()
            self.data['region_encoded'] = le_region.fit_transform(self.data['region'].fillna('Unknown'))
            feature_cols.append('region_encoded')
            categorical_features.append('region_encoded')
        
        # Remove duplicates and ensure all columns exist
        feature_cols = list(set(feature_cols))
        feature_cols = [col for col in feature_cols if col in self.data.columns]
        
        # Prepare features DataFrame
        features = self.data[feature_cols].copy()
        
        # More robust NaN handling and data type conversion
        columns_to_process = list(features.columns)  # Create a copy to avoid iteration issues
        columns_to_drop = []
        
        for col in columns_to_process:
            # Try to convert to numeric first
            if features[col].dtype == 'object':
                # Try to convert to numeric
                features[col] = pd.to_numeric(features[col], errors='coerce')
            
            # Now handle based on actual data type
            if features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # For numeric columns, fill with median
                median_val = features[col].median()
                if pd.isna(median_val):
                    median_val = 0.0  # If all values are NaN, use 0
                features[col] = features[col].fillna(median_val)
            else:
                # For remaining object columns, encode as numeric
                if col in categorical_features:
                    # Already encoded, should be numeric
                    features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
                else:
                    # Mark for dropping non-numeric columns that couldn't be converted
                    logger.warning(f"Marking non-numeric column for removal: {col}")
                    columns_to_drop.append(col)
        
        # Drop columns that couldn't be converted
        if columns_to_drop:
            features = features.drop(columns=columns_to_drop)
            # Update feature_cols list
            feature_cols = [col for col in feature_cols if col not in columns_to_drop]
        
        # Prepare target variable
        target_var = task_config['target_variable']
        if target_var not in self.data.columns:
            raise ValueError(f"Target variable '{target_var}' not found in data")
        
        targets = self.data[target_var].copy()
        
        # Handle missing targets
        valid_indices = targets.notna()
        features = features[valid_indices]
        targets = targets[valid_indices]
        
        logger.info(f"Prepared {len(feature_cols)} features and {len(targets)} samples for {target_var}")
        
        return features, targets, feature_cols
    
    def build_regression_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Build and train regression models."""
        models = {}
        
        # Random Forest
        models['random_forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        
        # Gradient Boosting
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        
        # XGBoost
        models['xgboost'] = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
            objective='reg:squarederror'
        )
        
        # LightGBM
        models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
            verbose=-1
        )
        
        # Neural Network
        models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42,
            early_stopping=True, validation_fraction=0.2
        )
        
        # Elastic Net
        models['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        
        # Train all models
        models_to_train = list(models.items())  # Create a copy to avoid iteration issues
        trained_models = {}
        
        for name, model in models_to_train:
            try:
                logger.info(f"Training {name} model...")
                model.fit(X_train, y_train)
                trained_models[name] = model
                logger.info(f"{name} model trained successfully")
            except Exception as e:
                logger.error(f"Error training {name} model: {e}")
        
        return trained_models
    
    def build_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                        sequence_length: int = 10) -> Sequential:
        """
        Build and train LSTM model for time series prediction.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            sequence_length: Length of input sequences
            
        Returns:
            Trained LSTM model
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(25),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return model
    
    def prepare_time_series_data(self, data: pd.DataFrame, target_col: str, 
                                sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for time series modeling.
        
        Args:
            data: Input data
            target_col: Target column name
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (sequences, targets)
        """
        # Sort by country and year
        data_sorted = data.sort_values(['country_code', 'year'])
        
        sequences = []
        targets = []
        
        # Create sequences for each country
        for country in data_sorted['country_code'].unique():
            country_data = data_sorted[data_sorted['country_code'] == country]
            
            if len(country_data) < sequence_length + 1:
                continue
            
            # Select numeric columns for features
            numeric_cols = country_data.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != target_col and 'year' not in col.lower()]
            
            if len(feature_cols) == 0:
                continue
            
            country_features = country_data[feature_cols].fillna(method='ffill').values
            country_targets = country_data[target_col].fillna(method='ffill').values
            
            # Create sequences
            for i in range(sequence_length, len(country_features)):
                sequences.append(country_features[i-sequence_length:i])
                targets.append(country_targets[i])
        
        return np.array(sequences), np.array(targets)
    
    def evaluate_model_performance(self, model: Any, X_test: pd.DataFrame, 
                                  y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Make predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                # Handle LSTM models
                y_pred = model.predict(X_test).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate MAPE, handling division by zero
            y_test_nonzero = y_test[y_test != 0]
            y_pred_nonzero = y_pred[y_test != 0]
            if len(y_test_nonzero) > 0:
                mape = mean_absolute_percentage_error(y_test_nonzero, y_pred_nonzero)
            else:
                mape = np.inf
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
            
            logger.info(f"{model_name} Performance - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return {'error': str(e)}
    
    def perform_hyperparameter_tuning(self, model_type: str, X_train: pd.DataFrame, 
                                     y_train: pd.Series) -> Any:
        """
        Perform hyperparameter tuning for the specified model type.
        
        Args:
            model_type: Type of model to tune
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best model after tuning
        """
        logger.info(f"Performing hyperparameter tuning for {model_type}")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
            
        elif model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = GradientBoostingRegressor(random_state=42)
            
        else:
            logger.warning(f"Hyperparameter tuning not implemented for {model_type}")
            return None
        
        # Perform grid search
        cv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        
        try:
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning for {model_type}: {e}")
            return None
    
    def create_ensemble_model(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                             y_train: pd.Series) -> Any:
        """
        Create an ensemble model from individual models.
        
        Args:
            models: Dictionary of trained models
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Ensemble model
        """
        from sklearn.ensemble import VotingRegressor
        
        # Select best performing models for ensemble
        voting_models = []
        for name, model in models.items():
            if hasattr(model, 'predict'):  # Skip LSTM models for now
                voting_models.append((name, model))
        
        if len(voting_models) < 2:
            logger.warning("Not enough models for ensemble. Returning None.")
            return None
        
        # Create voting regressor
        ensemble = VotingRegressor(estimators=voting_models)
        ensemble.fit(X_train, y_train)
        
        logger.info(f"Ensemble model created with {len(voting_models)} base models")
        return ensemble
    
    def extract_feature_importance(self, models: Dict[str, Any], feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Extract feature importance from trained models.
        
        Args:
            models: Dictionary of trained models
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_dict = {}
        
        for name, model in models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importances = np.abs(model.coef_)
                else:
                    continue
                
                # Create feature importance dictionary
                feature_importance = dict(zip(feature_names, importances))
                
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
                
                importance_dict[name] = feature_importance
                
                # Log top 10 features
                top_features = list(feature_importance.keys())[:10]
                logger.info(f"{name} top features: {top_features}")
                
            except Exception as e:
                logger.error(f"Error extracting feature importance for {name}: {e}")
        
        return importance_dict
    
    def generate_predictions(self, models: Dict[str, Any], X_future: pd.DataFrame, 
                           prediction_type: str = 'point') -> Dict[str, np.ndarray]:
        """
        Generate predictions using trained models.
        
        Args:
            models: Dictionary of trained models
            X_future: Features for prediction
            prediction_type: Type of prediction ('point', 'interval')
            
        Returns:
            Dictionary of predictions
        """
        predictions = {}
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_future)
                    predictions[name] = pred
                    
                    # Generate prediction intervals for tree-based models
                    if prediction_type == 'interval' and hasattr(model, 'estimators_'):
                        # Calculate prediction intervals from ensemble predictions
                        individual_preds = np.array([est.predict(X_future) for est in model.estimators_])
                        pred_intervals = np.percentile(individual_preds, [5, 95], axis=0)
                        predictions[f'{name}_lower'] = pred_intervals[0]
                        predictions[f'{name}_upper'] = pred_intervals[1]
                        
                logger.info(f"Generated predictions using {name} model")
                
            except Exception as e:
                logger.error(f"Error generating predictions with {name}: {e}")
        
        return predictions
    
    def run_financial_development_prediction(self) -> Dict[str, Any]:
        """
        Run financial development prediction modeling task.
        
        Returns:
            Dictionary containing models, performance metrics, and predictions
        """
        logger.info("Starting financial development prediction modeling")
        
        task_config = self.model_configs['financial_development_prediction']
        
        # Prepare data
        features, targets, feature_names = self.prepare_features_and_targets(task_config)
        
        # Train-test split with time-based splitting
        # Use earlier years for training, later years for testing
        split_year = features['year'].quantile(0.8) if 'year' in features.columns else None
        
        if split_year:
            train_mask = features['year'] <= split_year
            X_train, X_test = features[train_mask], features[~train_mask]
            y_train, y_test = targets[train_mask], targets[~train_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Build and train models
        models = self.build_regression_models(X_train, y_train)
        
        # Hyperparameter tuning for selected models
        tuned_models = {}
        for model_type in ['random_forest', 'xgboost']:
            if model_type in models:
                tuned_model = self.perform_hyperparameter_tuning(model_type, X_train, y_train)
                if tuned_model:
                    tuned_models[f'{model_type}_tuned'] = tuned_model
        
        models.update(tuned_models)
        
        # Create ensemble model
        ensemble_model = self.create_ensemble_model(models, X_train, y_train)
        if ensemble_model:
            models['ensemble'] = ensemble_model
        
        # Evaluate models
        performances = {}
        for name, model in models.items():
            performance = self.evaluate_model_performance(model, X_test, y_test, name)
            performances[name] = performance
        
        # Extract feature importance
        feature_importance = self.extract_feature_importance(models, feature_names)
        
        # Generate predictions
        predictions = self.generate_predictions(models, X_test, prediction_type='interval')
        
        # Store results
        task_results = {
            'models': models,
            'performances': performances,
            'feature_importance': feature_importance,
            'predictions': predictions,
            'feature_names': feature_names,
            'test_data': {'X_test': X_test, 'y_test': y_test}
        }
        
        self.models['financial_development_prediction'] = task_results
        
        logger.info("Financial development prediction modeling completed")
        return task_results
    
    def run_risk_classification(self) -> Dict[str, Any]:
        """
        Run country risk classification modeling task.
        
        Returns:
            Dictionary containing models and performance metrics
        """
        logger.info("Starting risk classification modeling")
        
        task_config = self.model_configs['risk_classification']
        
        # Check if risk categories exist
        if task_config['target_variable'] not in self.data.columns:
            logger.warning("Risk category target variable not found. Skipping risk classification.")
            return {}
        
        # Prepare data
        features, targets, feature_names = self.prepare_features_and_targets(task_config)
        
        # Check if we have any valid targets
        if len(targets) == 0:
            logger.warning("No valid targets found for risk classification. Skipping.")
            return {}
        
        # Encode target classes
        le_risk = LabelEncoder()
        targets_encoded = le_risk.fit_transform(targets)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets_encoded, test_size=0.2, random_state=42, stratify=targets_encoded
        )
        
        # Build classification models
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        import xgboost as xgb
        
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, objective='multi:softprob'),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        # Train models
        for name, model in models.items():
            try:
                logger.info(f"Training {name} classifier...")
                model.fit(X_train, y_train)
                logger.info(f"{name} classifier trained successfully")
            except Exception as e:
                logger.error(f"Error training {name} classifier: {e}")
                del models[name]
        
        # Evaluate models
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
        
        performances = {}
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                performances[name] = {
                    'accuracy': accuracy,
                    'classification_report': classification_report(y_test, y_pred, 
                                                                 target_names=le_risk.classes_),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                
                logger.info(f"{name} Classifier Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name} classifier: {e}")
        
        # Feature importance
        feature_importance = self.extract_feature_importance(models, feature_names)
        
        task_results = {
            'models': models,
            'performances': performances,
            'feature_importance': feature_importance,
            'label_encoder': le_risk,
            'feature_names': feature_names
        }
        
        self.models['risk_classification'] = task_results
        
        logger.info("Risk classification modeling completed")
        return task_results
    
    def run_time_series_forecasting(self) -> Dict[str, Any]:
        """
        Run time series forecasting for financial development indicators.
        
        Returns:
            Dictionary containing models and forecasts
        """
        logger.info("Starting time series forecasting")
        
        task_config = self.model_configs['indicator_forecasting']
        
        results = {}
        
        for target_var in task_config['target_variables']:
            if target_var not in self.data.columns:
                continue
                
            logger.info(f"Forecasting {target_var}")
            
            try:
                # Prepare time series data
                X_seq, y_seq = self.prepare_time_series_data(
                    self.data, target_var, task_config['sequence_length']
                )
                
                if len(X_seq) < 50:  # Need minimum samples for LSTM
                    logger.warning(f"Insufficient data for {target_var} forecasting")
                    continue
                
                # Train-test split for time series
                split_idx = int(0.8 * len(X_seq))
                X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
                y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]
                
                # Build and train LSTM model
                lstm_model = self.build_lstm_model(X_train_seq, y_train_seq, task_config['sequence_length'])
                
                # Evaluate model
                y_pred_seq = lstm_model.predict(X_test_seq).flatten()
                
                performance = {
                    'mse': mean_squared_error(y_test_seq, y_pred_seq),
                    'mae': mean_absolute_error(y_test_seq, y_pred_seq),
                    'r2': r2_score(y_test_seq, y_pred_seq)
                }
                
                logger.info(f"{target_var} LSTM Performance - MSE: {performance['mse']:.4f}, R²: {performance['r2']:.4f}")
                
                results[target_var] = {
                    'model': lstm_model,
                    'performance': performance,
                    'predictions': y_pred_seq,
                    'actual': y_test_seq
                }
                
            except Exception as e:
                logger.error(f"Error in time series forecasting for {target_var}: {e}")
        
        self.models['time_series_forecasting'] = results
        
        logger.info("Time series forecasting completed")
        return results
    
    def save_models_and_results(self, output_dir: str = "models") -> None:
        """
        Save trained models and results to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for task_name, task_results in self.models.items():
            task_dir = output_path / task_name
            task_dir.mkdir(exist_ok=True)
            
            # Save models
            if 'models' in task_results:
                for model_name, model in task_results['models'].items():
                    try:
                        if hasattr(model, 'save'):  # Keras models
                            model.save(task_dir / f"{model_name}_{timestamp}.h5")
                        else:  # Scikit-learn models
                            joblib.dump(model, task_dir / f"{model_name}_{timestamp}.pkl")
                        logger.info(f"Saved {model_name} model for {task_name}")
                    except Exception as e:
                        logger.error(f"Error saving {model_name} model: {e}")
            
            # Save results as JSON
            results_to_save = {}
            for key, value in task_results.items():
                if key != 'models':  # Don't include model objects in JSON
                    if isinstance(value, (dict, list, str, int, float, bool)):
                        results_to_save[key] = value
                    elif isinstance(value, np.ndarray):
                        results_to_save[key] = value.tolist()
                    elif hasattr(value, 'to_dict'):
                        results_to_save[key] = value.to_dict()
            
            with open(task_dir / f"results_{timestamp}.json", 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"Models and results saved to {output_path}")
    
    def generate_model_performance_report(self) -> str:
        """
        Generate comprehensive model performance report.
        
        Returns:
            Formatted performance report
        """
        report = []
        report.append("=" * 80)
        report.append("GLOBAL FINANCIAL DEVELOPMENT - PREDICTIVE MODEL PERFORMANCE REPORT")
        report.append("=" * 80)
        
        for task_name, task_results in self.models.items():
            if not task_results:
                continue
                
            report.append(f"\n{task_name.upper().replace('_', ' ')}")
            report.append("-" * 50)
            
            if 'performances' in task_results:
                performances = task_results['performances']
                
                # Sort models by performance
                if task_name == 'financial_development_prediction':
                    # Sort by R² score (higher is better)
                    sorted_models = sorted(performances.items(), 
                                         key=lambda x: x[1].get('r2', -1), reverse=True)
                    
                    for model_name, metrics in sorted_models:
                        if 'error' not in metrics:
                            report.append(f"  {model_name}:")
                            report.append(f"    RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                            report.append(f"    R²: {metrics.get('r2', 'N/A'):.4f}")
                            report.append(f"    MAE: {metrics.get('mae', 'N/A'):.4f}")
                            report.append(f"    MAPE: {metrics.get('mape', 'N/A'):.4f}")
                
                elif task_name == 'risk_classification':
                    # Sort by accuracy (higher is better)
                    sorted_models = sorted(performances.items(), 
                                         key=lambda x: x[1].get('accuracy', 0), reverse=True)
                    
                    for model_name, metrics in sorted_models:
                        report.append(f"  {model_name}:")
                        report.append(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            
            # Feature importance
            if 'feature_importance' in task_results and task_results['feature_importance']:
                report.append("\n  Top Features:")
                for model_name, importance in task_results['feature_importance'].items():
                    if importance:
                        top_features = list(importance.keys())[:5]
                        report.append(f"    {model_name}: {', '.join(top_features)}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def run_complete_modeling_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete predictive modeling pipeline.
        
        Returns:
            Dictionary containing all modeling results
        """
        logger.info("Starting complete predictive modeling pipeline")
        
        pipeline_results = {}
        
        try:
            # 1. Financial Development Prediction
            fd_results = self.run_financial_development_prediction()
            pipeline_results['financial_development_prediction'] = fd_results
            
            # 2. Risk Classification
            risk_results = self.run_risk_classification()
            pipeline_results['risk_classification'] = risk_results
            
            # 3. Time Series Forecasting
            ts_results = self.run_time_series_forecasting()
            pipeline_results['time_series_forecasting'] = ts_results
            
            # Save models and results
            self.save_models_and_results()
            
            # Generate performance report
            performance_report = self.generate_model_performance_report()
            print(performance_report)
            
            # Save performance report
            with open("models/performance_report.txt", 'w') as f:
                f.write(performance_report)
            
            logger.info("Complete predictive modeling pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in complete modeling pipeline: {e}")
            raise
        
        return pipeline_results


def main():
    """Main execution function for predictive modeling."""
    try:
        # Initialize predictive modeling suite
        gfd_models = GFDPredictiveModels("data/gfd_database.db")
        
        # Run complete modeling pipeline
        results = gfd_models.run_complete_modeling_pipeline()
        
        print("\nPredictive modeling pipeline completed successfully!")
        print(f"Number of modeling tasks completed: {len(results)}")
        
        # Display summary of results
        for task_name, task_results in results.items():
            if task_results and 'performances' in task_results:
                best_model = max(task_results['performances'].items(), 
                               key=lambda x: x[1].get('r2', x[1].get('accuracy', 0)))
                print(f"\n{task_name}: Best model is {best_model[0]}")
        
    except Exception as e:
        logger.error(f"Predictive modeling pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()