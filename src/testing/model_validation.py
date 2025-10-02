#!/usr/bin/env python3
"""
Model Validation and Testing Framework for Global Financial Development Analysis

This module provides comprehensive validation and testing capabilities for predictive models
including statistical validation, cross-validation, backtesting, stress testing, and
automated model performance monitoring.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                            accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report)
from scipy import stats
from scipy.stats import normaltest, jarque_bera, anderson, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ModelValidator:
    """
    Comprehensive model validation and testing framework.
    
    Features:
    - Statistical validation of model assumptions
    - Cross-validation with multiple strategies
    - Backtesting for time series models
    - Stress testing and scenario analysis
    - Model performance monitoring
    - Automated testing suite
    - Performance degradation detection
    """
    
    def __init__(self, data_path: str = "data/gfd_database.db"):
        """
        Initialize the model validator.
        
        Args:
            data_path: Path to the database file
        """
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.validation_results = {}
        
        # Load data and models
        self._load_data()
        self._load_models()
    
    def _load_data(self) -> None:
        """Load validation data from database."""
        try:
            conn = sqlite3.connect(self.data_path)
            
            self.data = pd.read_sql_query("""
                SELECT f.*, c.financial_development_score, c.systemic_risk_score,
                       c.market_volatility_score, c.institutional_quality_score, c.risk_category
                FROM financial_development_data f
                LEFT JOIN country_risk_profiles c 
                ON f.country_code = c.country_code AND f.year = c.year
                ORDER BY f.country_name, f.year
            """, conn)
            
            # Parse raw indicators
            if 'raw_indicators' in self.data.columns:
                raw_indicators = self.data['raw_indicators'].apply(
                    lambda x: json.loads(x) if pd.notna(x) and x != '' else {}
                )
                raw_indicators_df = pd.json_normalize(raw_indicators)
                self.data = pd.concat([self.data, raw_indicators_df], axis=1)
            
            conn.close()
            logger.info(f"Validation data loaded successfully. Shape: {self.data.shape}")
            
        except Exception as e:
            logger.error(f"Error loading validation data: {e}")
            raise
    
    def _load_models(self) -> None:
        """Load trained models for validation."""
        models_dir = Path("models")
        
        # Load financial development prediction models
        fd_models_dir = models_dir / "financial_development_prediction"
        if fd_models_dir.exists():
            for model_file in fd_models_dir.glob("*.pkl"):
                model_name = model_file.stem.split('_')[0]  # Extract model name before timestamp
                try:
                    model = joblib.load(model_file)
                    self.models[model_name] = model
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Could not load model {model_file}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models for validation")
    
    def validate_model_assumptions(self, model_name: str, predictions: np.ndarray, 
                                  actual: np.ndarray) -> Dict[str, Any]:
        """
        Validate statistical assumptions of the model.
        
        Args:
            model_name: Name of the model
            predictions: Model predictions
            actual: Actual values
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Validating assumptions for model: {model_name}")
        
        # Calculate residuals
        residuals = actual - predictions
        
        validation_results = {
            'model_name': model_name,
            'sample_size': len(residuals),
            'residual_statistics': {},
            'normality_tests': {},
            'homoscedasticity_tests': {},
            'autocorrelation_tests': {},
            'outlier_detection': {}
        }
        
        # 1. Residual Statistics
        validation_results['residual_statistics'] = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals))
        }
        
        # 2. Normality Tests
        try:
            # Shapiro-Wilk test (for small samples)
            if len(residuals) <= 5000:
                shapiro_stat, shapiro_p = shapiro(residuals)
                validation_results['normality_tests']['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            
            # Jarque-Bera test
            jb_stat, jb_p = jarque_bera(residuals)
            validation_results['normality_tests']['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'is_normal': jb_p > 0.05
            }
            
            # Anderson-Darling test
            ad_result = anderson(residuals, dist='norm')
            validation_results['normality_tests']['anderson_darling'] = {
                'statistic': float(ad_result.statistic),
                'critical_values': ad_result.critical_values.tolist(),
                'significance_levels': ad_result.significance_level.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Error in normality tests: {e}")
        
        # 3. Homoscedasticity Tests (Breusch-Pagan test approximation)
        try:
            # Simple test: correlation between squared residuals and predictions
            squared_residuals = residuals ** 2
            correlation = np.corrcoef(predictions, squared_residuals)[0, 1]
            validation_results['homoscedasticity_tests']['residual_correlation'] = {
                'correlation': float(correlation),
                'is_homoscedastic': abs(correlation) < 0.1  # Threshold for homoscedasticity
            }
        except Exception as e:
            logger.warning(f"Error in homoscedasticity test: {e}")
        
        # 4. Autocorrelation Tests (Durbin-Watson approximation)
        try:
            # Calculate first-order autocorrelation
            if len(residuals) > 1:
                autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                validation_results['autocorrelation_tests']['first_order'] = {
                    'autocorrelation': float(autocorr),
                    'is_independent': abs(autocorr) < 0.1
                }
        except Exception as e:
            logger.warning(f"Error in autocorrelation test: {e}")
        
        # 5. Outlier Detection
        try:
            # IQR method
            Q1 = np.percentile(residuals, 25)
            Q3 = np.percentile(residuals, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = np.sum((residuals < lower_bound) | (residuals > upper_bound))
            outlier_percentage = (outliers / len(residuals)) * 100
            
            validation_results['outlier_detection'] = {
                'outlier_count': int(outliers),
                'outlier_percentage': float(outlier_percentage),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
        except Exception as e:
            logger.warning(f"Error in outlier detection: {e}")
        
        return validation_results
    
    def cross_validate_models(self, cv_strategy: str = 'time_series', 
                             cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on all loaded models.
        
        Args:
            cv_strategy: Cross-validation strategy ('time_series', 'kfold', 'stratified')
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation results
        """
        logger.info(f"Performing cross-validation with {cv_strategy} strategy")
        
        cv_results = {}
        
        # Prepare data
        features, targets = self._prepare_model_data()
        
        if features is None or targets is None:
            logger.error("Could not prepare data for cross-validation")
            return cv_results
        
        # Select cross-validation strategy
        if cv_strategy == 'time_series':
            cv_splitter = TimeSeriesSplit(n_splits=cv_folds)
        elif cv_strategy == 'kfold':
            cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        elif cv_strategy == 'stratified':
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Validate each model
        for model_name, model in self.models.items():
            logger.info(f"Cross-validating model: {model_name}")
            
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(model, features, targets, 
                                          cv=cv_splitter, 
                                          scoring='neg_mean_squared_error')
                
                # Calculate additional metrics for each fold
                fold_results = []
                for train_idx, test_idx in cv_splitter.split(features, targets):
                    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                    y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]
                    
                    # Fit and predict
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    fold_metrics = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                    }
                    fold_results.append(fold_metrics)
                
                # Aggregate results
                cv_results[model_name] = {
                    'cv_scores': cv_scores.tolist(),
                    'mean_cv_score': float(np.mean(cv_scores)),
                    'std_cv_score': float(np.std(cv_scores)),
                    'fold_metrics': fold_results,
                    'aggregate_metrics': {
                        'mean_mse': float(np.mean([f['mse'] for f in fold_results])),
                        'std_mse': float(np.std([f['mse'] for f in fold_results])),
                        'mean_mae': float(np.mean([f['mae'] for f in fold_results])),
                        'std_mae': float(np.std([f['mae'] for f in fold_results])),
                        'mean_r2': float(np.mean([f['r2'] for f in fold_results])),
                        'std_r2': float(np.std([f['r2'] for f in fold_results]))
                    }
                }
                
                logger.info(f"CV completed for {model_name}. Mean CV Score: {np.mean(cv_scores):.4f}")
                
            except Exception as e:
                logger.error(f"Error in cross-validation for {model_name}: {e}")
                cv_results[model_name] = {'error': str(e)}
        
        return cv_results
    
    def backtest_models(self, start_date: str = '2015-01-01', 
                       test_window: int = 3) -> Dict[str, Any]:
        """
        Perform backtesting for time series models.
        
        Args:
            start_date: Start date for backtesting
            test_window: Size of test window in years
            
        Returns:
            Dictionary containing backtesting results
        """
        logger.info(f"Performing backtesting from {start_date} with {test_window}-year windows")
        
        backtest_results = {}
        
        # Prepare time series data
        if 'year' not in self.data.columns:
            logger.error("Year column not found for backtesting")
            return backtest_results
        
        features, targets = self._prepare_model_data()
        if features is None or targets is None:
            return backtest_results
        
        # Add year column to features for time-based splitting
        if 'year' in self.data.columns:
            features = features.copy()
            features['year'] = self.data['year']
        
        start_year = int(start_date.split('-')[0])
        max_year = features['year'].max()
        
        for model_name, model in self.models.items():
            logger.info(f"Backtesting model: {model_name}")
            
            try:
                backtest_metrics = []
                
                # Perform walk-forward backtesting
                current_year = start_year
                while current_year + test_window <= max_year:
                    # Define training and test periods
                    train_mask = features['year'] < current_year
                    test_mask = ((features['year'] >= current_year) & 
                                (features['year'] < current_year + test_window))
                    
                    if train_mask.sum() < 50 or test_mask.sum() < 10:
                        current_year += 1
                        continue
                    
                    # Split data
                    X_train = features[train_mask].drop('year', axis=1)
                    y_train = targets[train_mask]
                    X_test = features[test_mask].drop('year', axis=1)
                    y_test = targets[test_mask]
                    
                    # Train and predict
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    period_metrics = {
                        'start_year': current_year,
                        'end_year': current_year + test_window - 1,
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'mse': mean_squared_error(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                    }
                    
                    backtest_metrics.append(period_metrics)
                    current_year += 1
                
                # Aggregate backtest results
                if backtest_metrics:
                    backtest_results[model_name] = {
                        'periods': backtest_metrics,
                        'summary': {
                            'total_periods': len(backtest_metrics),
                            'avg_mse': float(np.mean([p['mse'] for p in backtest_metrics])),
                            'std_mse': float(np.std([p['mse'] for p in backtest_metrics])),
                            'avg_mae': float(np.mean([p['mae'] for p in backtest_metrics])),
                            'avg_r2': float(np.mean([p['r2'] for p in backtest_metrics])),
                            'min_r2': float(min([p['r2'] for p in backtest_metrics])),
                            'max_r2': float(max([p['r2'] for p in backtest_metrics]))
                        }
                    }
                    
                    logger.info(f"Backtesting completed for {model_name}. "
                              f"Avg R²: {backtest_results[model_name]['summary']['avg_r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error in backtesting for {model_name}: {e}")
                backtest_results[model_name] = {'error': str(e)}
        
        return backtest_results
    
    def stress_test_models(self, scenarios: Dict[str, Dict] = None) -> Dict[str, Any]:
        """
        Perform stress testing on models with different scenarios.
        
        Args:
            scenarios: Dictionary of stress test scenarios
            
        Returns:
            Dictionary containing stress test results
        """
        logger.info("Performing stress testing on models")
        
        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()
        
        stress_results = {}
        features, targets = self._prepare_model_data()
        
        if features is None or targets is None:
            return stress_results
        
        for model_name, model in self.models.items():
            logger.info(f"Stress testing model: {model_name}")
            
            model_stress_results = {}
            
            # Train model on full dataset
            try:
                model.fit(features, targets)
                baseline_pred = model.predict(features)
                baseline_mse = mean_squared_error(targets, baseline_pred)
                
                for scenario_name, scenario_config in scenarios.items():
                    logger.info(f"Testing scenario: {scenario_name}")
                    
                    # Apply stress scenario
                    stressed_features = self._apply_stress_scenario(features, scenario_config)
                    
                    # Make predictions with stressed data
                    stressed_pred = model.predict(stressed_features)
                    stressed_mse = mean_squared_error(targets, stressed_pred)
                    
                    # Calculate stress impact
                    scenario_results = {
                        'baseline_mse': float(baseline_mse),
                        'stressed_mse': float(stressed_mse),
                        'mse_change': float(stressed_mse - baseline_mse),
                        'mse_change_pct': float(((stressed_mse - baseline_mse) / baseline_mse) * 100),
                        'prediction_change_mean': float(np.mean(stressed_pred - baseline_pred)),
                        'prediction_change_std': float(np.std(stressed_pred - baseline_pred))
                    }
                    
                    model_stress_results[scenario_name] = scenario_results
                
                stress_results[model_name] = model_stress_results
                
            except Exception as e:
                logger.error(f"Error in stress testing for {model_name}: {e}")
                stress_results[model_name] = {'error': str(e)}
        
        return stress_results
    
    def _get_default_stress_scenarios(self) -> Dict[str, Dict]:
        """Get default stress testing scenarios."""
        return {
            'economic_shock': {
                'type': 'multiplicative',
                'features': ['access_institutions_index', 'depth_institutions_index'],
                'factor': 0.8  # 20% decrease
            },
            'market_volatility': {
                'type': 'additive',
                'features': ['efficiency_markets_index', 'stability_markets_index'],
                'value': -10  # Decrease by 10 points
            },
            'financial_crisis': {
                'type': 'multiplicative',
                'features': ['overall_financial_development_index'],
                'factor': 0.7  # 30% decrease
            },
            'regulatory_tightening': {
                'type': 'multiplicative',
                'features': ['access_institutions_index', 'efficiency_institutions_index'],
                'factor': 0.9  # 10% decrease
            }
        }
    
    def _apply_stress_scenario(self, features: pd.DataFrame, 
                              scenario_config: Dict) -> pd.DataFrame:
        """Apply stress scenario to features."""
        stressed_features = features.copy()
        
        scenario_type = scenario_config.get('type', 'multiplicative')
        target_features = scenario_config.get('features', [])
        
        for feature in target_features:
            if feature in stressed_features.columns:
                if scenario_type == 'multiplicative':
                    factor = scenario_config.get('factor', 1.0)
                    stressed_features[feature] *= factor
                elif scenario_type == 'additive':
                    value = scenario_config.get('value', 0.0)
                    stressed_features[feature] += value
        
        return stressed_features
    
    def monitor_model_performance(self) -> Dict[str, Any]:
        """
        Monitor ongoing model performance and detect degradation.
        
        Returns:
            Dictionary containing performance monitoring results
        """
        logger.info("Monitoring model performance")
        
        monitoring_results = {}
        features, targets = self._prepare_model_data()
        
        if features is None or targets is None:
            return monitoring_results
        
        # Get recent data (last 2 years) for performance monitoring
        recent_years = self.data['year'].max() - 1
        recent_mask = self.data['year'] >= recent_years
        
        recent_features = features[recent_mask]
        recent_targets = targets[recent_mask]
        
        # Historical data for baseline comparison
        historical_mask = self.data['year'] < recent_years
        hist_features = features[historical_mask]
        hist_targets = targets[historical_mask]
        
        for model_name, model in self.models.items():
            logger.info(f"Monitoring performance for model: {model_name}")
            
            try:
                # Train on historical data
                model.fit(hist_features, hist_targets)
                
                # Performance on historical data (baseline)
                hist_pred = model.predict(hist_features)
                hist_mse = mean_squared_error(hist_targets, hist_pred)
                hist_r2 = r2_score(hist_targets, hist_pred)
                
                # Performance on recent data
                if len(recent_features) > 0:
                    recent_pred = model.predict(recent_features)
                    recent_mse = mean_squared_error(recent_targets, recent_pred)
                    recent_r2 = r2_score(recent_targets, recent_pred)
                    
                    # Calculate performance degradation
                    mse_degradation = ((recent_mse - hist_mse) / hist_mse) * 100
                    r2_degradation = ((hist_r2 - recent_r2) / hist_r2) * 100
                    
                    # Performance monitoring flags
                    performance_flags = {
                        'significant_mse_increase': mse_degradation > 20,  # MSE increased by >20%
                        'significant_r2_decrease': r2_degradation > 10,   # R² decreased by >10%
                        'requires_retraining': mse_degradation > 30 or r2_degradation > 15
                    }
                    
                    monitoring_results[model_name] = {
                        'historical_performance': {
                            'mse': float(hist_mse),
                            'r2': float(hist_r2)
                        },
                        'recent_performance': {
                            'mse': float(recent_mse),
                            'r2': float(recent_r2)
                        },
                        'performance_changes': {
                            'mse_change_pct': float(mse_degradation),
                            'r2_change_pct': float(r2_degradation)
                        },
                        'flags': performance_flags,
                        'monitoring_date': datetime.now().isoformat()
                    }
                else:
                    monitoring_results[model_name] = {
                        'error': 'Insufficient recent data for monitoring'
                    }
                
            except Exception as e:
                logger.error(f"Error monitoring performance for {model_name}: {e}")
                monitoring_results[model_name] = {'error': str(e)}
        
        return monitoring_results
    
    def _prepare_model_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for model validation."""
        try:
            # Select feature columns (similar to model training)
            feature_cols = []
            
            # Composite indices
            composite_indices = [
                'access_institutions_index', 'access_markets_index',
                'depth_institutions_index', 'depth_markets_index',
                'efficiency_institutions_index', 'efficiency_markets_index',
                'stability_institutions_index', 'stability_markets_index'
            ]
            
            for col in composite_indices:
                if col in self.data.columns:
                    feature_cols.append(col)
            
            # Add some raw indicators
            raw_indicators = [col for col in self.data.columns 
                            if col.startswith(('ai', 'am', 'di', 'dm', 'ei', 'em', 'si', 'sm'))]
            feature_cols.extend(raw_indicators[:20])  # Top 20
            
            # Add year if available
            if 'year' in self.data.columns:
                feature_cols.append('year')
            
            # Remove duplicates and ensure columns exist
            feature_cols = list(set(feature_cols))
            feature_cols = [col for col in feature_cols if col in self.data.columns]
            
            if not feature_cols:
                logger.error("No valid feature columns found")
                return None, None
            
            # Prepare features
            features = self.data[feature_cols].copy()
            features = features.fillna(features.median())
            
            # Target variable
            target_col = 'overall_financial_development_index'
            if target_col not in self.data.columns:
                logger.error("Target variable not found")
                return None, None
            
            targets = self.data[target_col].copy()
            
            # Remove rows with missing targets
            valid_mask = targets.notna()
            features = features[valid_mask]
            targets = targets[valid_mask]
            
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preparing model data: {e}")
            return None, None
    
    def generate_validation_report(self) -> str:
        """
        Generate comprehensive validation report.
        
        Returns:
            Formatted validation report
        """
        logger.info("Generating comprehensive validation report")
        
        report = []
        report.append("=" * 80)
        report.append("MODEL VALIDATION AND TESTING REPORT")
        report.append("=" * 80)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models Validated: {len(self.models)}")
        report.append("")
        
        # Run all validation tests
        try:
            # Cross-validation
            cv_results = self.cross_validate_models()
            
            # Backtesting
            backtest_results = self.backtest_models()
            
            # Stress testing
            stress_results = self.stress_test_models()
            
            # Performance monitoring
            monitoring_results = self.monitor_model_performance()
            
            # Cross-Validation Results
            report.append("1. CROSS-VALIDATION RESULTS")
            report.append("-" * 40)
            for model_name, results in cv_results.items():
                if 'error' not in results:
                    report.append(f"{model_name.upper()}:")
                    report.append(f"  Mean CV Score: {results['mean_cv_score']:.4f}")
                    report.append(f"  Std CV Score: {results['std_cv_score']:.4f}")
                    report.append(f"  Mean R²: {results['aggregate_metrics']['mean_r2']:.4f}")
                    report.append(f"  Mean RMSE: {np.sqrt(results['aggregate_metrics']['mean_mse']):.4f}")
                    report.append("")
            
            # Backtesting Results
            report.append("2. BACKTESTING RESULTS")
            report.append("-" * 40)
            for model_name, results in backtest_results.items():
                if 'error' not in results:
                    summary = results['summary']
                    report.append(f"{model_name.upper()}:")
                    report.append(f"  Periods Tested: {summary['total_periods']}")
                    report.append(f"  Average R²: {summary['avg_r2']:.4f}")
                    report.append(f"  R² Range: {summary['min_r2']:.4f} - {summary['max_r2']:.4f}")
                    report.append(f"  Average MAE: {summary['avg_mae']:.4f}")
                    report.append("")
            
            # Stress Testing Results
            report.append("3. STRESS TESTING RESULTS")
            report.append("-" * 40)
            for model_name, results in stress_results.items():
                if 'error' not in results:
                    report.append(f"{model_name.upper()}:")
                    for scenario, scenario_results in results.items():
                        report.append(f"  {scenario}: {scenario_results['mse_change_pct']:+.1f}% MSE change")
                    report.append("")
            
            # Performance Monitoring
            report.append("4. PERFORMANCE MONITORING")
            report.append("-" * 40)
            for model_name, results in monitoring_results.items():
                if 'error' not in results:
                    report.append(f"{model_name.upper()}:")
                    changes = results['performance_changes']
                    flags = results['flags']
                    report.append(f"  MSE Change: {changes['mse_change_pct']:+.1f}%")
                    report.append(f"  R² Change: {changes['r2_change_pct']:+.1f}%")
                    if any(flags.values()):
                        report.append(f"  ⚠️  Performance warnings detected")
                    else:
                        report.append(f"  ✓ Performance stable")
                    report.append("")
            
            # Summary and Recommendations
            report.append("5. SUMMARY AND RECOMMENDATIONS")
            report.append("-" * 40)
            
            # Find best performing model
            if cv_results:
                best_model = max(cv_results.items(), 
                               key=lambda x: x[1].get('aggregate_metrics', {}).get('mean_r2', -1))
                report.append(f"Best Performing Model: {best_model[0].upper()}")
                report.append(f"  Cross-Validation R²: {best_model[1]['aggregate_metrics']['mean_r2']:.4f}")
            
            # Flag models needing attention
            models_needing_attention = []
            for model_name, results in monitoring_results.items():
                if 'flags' in results and results['flags'].get('requires_retraining', False):
                    models_needing_attention.append(model_name)
            
            if models_needing_attention:
                report.append(f"Models requiring retraining: {', '.join(models_needing_attention)}")
            else:
                report.append("All models performing within acceptable ranges")
            
        except Exception as e:
            report.append(f"Error generating validation report: {e}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def save_validation_results(self, output_dir: str = "validation_results") -> None:
        """Save validation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate and save validation report
        report = self.generate_validation_report()
        with open(output_path / f"validation_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed results as JSON
        validation_data = {
            'cross_validation': self.cross_validate_models(),
            'backtesting': self.backtest_models(),
            'stress_testing': self.stress_test_models(),
            'performance_monitoring': self.monitor_model_performance(),
            'timestamp': timestamp
        }
        
        with open(output_path / f"validation_results_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Validation results saved to {output_path}")


def main():
    """Main function to run model validation."""
    try:
        # Initialize validator
        validator = ModelValidator()
        
        # Run comprehensive validation
        print("Running comprehensive model validation...")
        
        # Generate and print report
        report = validator.generate_validation_report()
        print(report)
        
        # Save results
        validator.save_validation_results()
        
        print("\nModel validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise


if __name__ == "__main__":
    main()