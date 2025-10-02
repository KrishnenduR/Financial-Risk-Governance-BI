#!/usr/bin/env python3
"""
Global Financial Development Database Preprocessor

This module implements comprehensive data preprocessing, cleaning, and feature engineering
for the Global Financial Development Database with advanced missing data imputation,
outlier detection, and feature transformation capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.interpolate import interp1d
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class GFDPreprocessor:
    """
    Advanced preprocessing pipeline for Global Financial Development Database.
    
    Features:
    - Multiple imputation strategies (KNN, Iterative, Time-series interpolation)
    - Outlier detection and treatment
    - Feature engineering and transformation
    - Data quality assessment and improvement
    - Country and income group standardization
    - Temporal consistency checks
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration dictionary for preprocessing parameters
        """
        self.config = config or self._default_config()
        self.data = None
        self.metadata = None
        self.scaler = None
        self.imputer = None
        self.outlier_detector = None
        self.feature_names = None
        self.preprocessing_stats = {}
        
        # Initialize indicator mappings based on 4x2 framework
        self._initialize_indicator_mappings()
        
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration for preprocessing."""
        return {
            'missing_threshold': 0.95,  # Drop columns with >95% missing
            'outlier_contamination': 0.1,  # Isolation Forest contamination rate
            'imputation_method': 'knn',  # 'knn', 'iterative', 'interpolation'
            'scaling_method': 'robust',  # 'standard', 'minmax', 'robust'
            'outlier_treatment': 'cap',  # 'remove', 'cap', 'transform'
            'min_years_per_country': 5,  # Minimum years of data per country
            'temporal_interpolation': True,  # Enable temporal interpolation
            'feature_engineering': True,  # Enable advanced feature engineering
        }
    
    def _initialize_indicator_mappings(self) -> None:
        """Initialize mappings for the 4x2 framework indicators."""
        # Access indicators (ai01-ai36, am01-am03)
        self.access_indicators = {
            'institutions': [f'ai{i:02d}' for i in range(1, 37)],
            'markets': [f'am{i:02d}' for i in range(1, 4)]
        }
        
        # Depth indicators (di01-di14, dm01-dm16)  
        self.depth_indicators = {
            'institutions': [f'di{i:02d}' for i in range(1, 15)],
            'markets': [f'dm{i:02d}' for i in range(1, 17)]
        }
        
        # Efficiency indicators (ei01-ei10, em01)
        self.efficiency_indicators = {
            'institutions': [f'ei{i:02d}' for i in range(1, 11)],
            'markets': ['em01']
        }
        
        # Stability indicators (si01-si07, sm01)
        self.stability_indicators = {
            'institutions': [f'si{i:02d}' for i in range(1, 8)],
            'markets': ['sm01']
        }
        
        # Other indicators (oi01-oi20a, om01-om02)
        self.other_indicators = {
            'institutions': [f'oi{i:02d}' for i in range(1, 21)] + ['oi16a', 'oi20a'],
            'markets': ['om01', 'om02']
        }
        
        # All indicator categories
        self.all_indicators = {
            'access': self.access_indicators,
            'depth': self.depth_indicators,
            'efficiency': self.efficiency_indicators,
            'stability': self.stability_indicators,
            'other': self.other_indicators
        }
    
    def load_data(self, file_path: str, sheet_name: str = 'Data - August 2022') -> None:
        """
        Load the Global Financial Development Database.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to load
        """
        try:
            logger.info(f"Loading data from {file_path}")
            self.data = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Load metadata
            try:
                self.metadata = pd.read_excel(file_path, sheet_name='Metadata')
                logger.info("Metadata loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
            
            # Basic data standardization
            self.data.columns = self.data.columns.str.lower()
            
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def assess_data_quality(self) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.
        
        Returns:
            Dictionary containing data quality metrics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        assessment = {
            'basic_stats': {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024,
            },
            'missing_data': {},
            'data_types': {},
            'temporal_coverage': {},
            'geographical_coverage': {},
            'indicator_completeness': {}
        }
        
        # Missing data analysis
        missing_counts = self.data.isnull().sum()
        missing_pct = (missing_counts / len(self.data)) * 100
        
        assessment['missing_data'] = {
            'total_missing': missing_counts.sum(),
            'missing_percentage': missing_pct.mean(),
            'columns_above_threshold': missing_pct[missing_pct > self.config['missing_threshold'] * 100].to_dict(),
            'most_missing': missing_pct.nlargest(10).to_dict()
        }
        
        # Data types analysis
        assessment['data_types'] = {
            'numeric_columns': len(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.data.select_dtypes(include=['object']).columns),
            'datetime_columns': len(self.data.select_dtypes(include=['datetime']).columns)
        }
        
        # Temporal coverage
        if 'year' in self.data.columns:
            year_stats = self.data['year'].agg(['min', 'max', 'nunique'])
            assessment['temporal_coverage'] = {
                'start_year': int(year_stats['min']),
                'end_year': int(year_stats['max']),
                'total_years': int(year_stats['nunique']),
                'year_completeness': self.data.groupby('year').size().to_dict()
            }
        
        # Geographical coverage
        if 'country' in self.data.columns:
            assessment['geographical_coverage'] = {
                'total_countries': self.data['country'].nunique(),
                'countries_by_region': self.data.groupby('region')['country'].nunique().to_dict() if 'region' in self.data.columns else {},
                'countries_by_income': self.data.groupby('income')['country'].nunique().to_dict() if 'income' in self.data.columns else {}
            }
        
        # Indicator completeness by category
        for category, indicators_dict in self.all_indicators.items():
            category_indicators = []
            for sector, indicators in indicators_dict.items():
                category_indicators.extend(indicators)
            
            available_indicators = [col for col in category_indicators if col in self.data.columns]
            if available_indicators:
                completeness = (1 - self.data[available_indicators].isnull().mean().mean()) * 100
                assessment['indicator_completeness'][category] = {
                    'available_indicators': len(available_indicators),
                    'total_indicators': len(category_indicators),
                    'average_completeness': completeness
                }
        
        self.preprocessing_stats['data_quality'] = assessment
        return assessment
    
    def clean_data(self) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning.
        
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Starting data cleaning process")
        cleaned_data = self.data.copy()
        
        # 1. Remove columns with excessive missing data
        missing_threshold = self.config['missing_threshold']
        missing_pct = cleaned_data.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing data")
            cleaned_data = cleaned_data.drop(columns=cols_to_drop)
        
        # 2. Remove countries with insufficient temporal coverage
        if 'country' in cleaned_data.columns and 'year' in cleaned_data.columns:
            country_years = cleaned_data.groupby('country')['year'].nunique()
            countries_to_keep = country_years[country_years >= self.config['min_years_per_country']].index
            
            before_count = cleaned_data['country'].nunique()
            cleaned_data = cleaned_data[cleaned_data['country'].isin(countries_to_keep)]
            after_count = cleaned_data['country'].nunique()
            
            logger.info(f"Kept {after_count} countries out of {before_count} based on temporal coverage")
        
        # 3. Standardize country and income group names
        cleaned_data = self._standardize_categorical_variables(cleaned_data)
        
        # 4. Handle duplicate records
        duplicates = cleaned_data.duplicated(subset=['country', 'year'] if 'year' in cleaned_data.columns else None)
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate records. Removing duplicates.")
            cleaned_data = cleaned_data[~duplicates]
        
        # 5. Basic outlier detection and flagging
        cleaned_data = self._detect_outliers(cleaned_data)
        
        logger.info(f"Data cleaning completed. Final shape: {cleaned_data.shape}")
        self.preprocessing_stats['cleaning'] = {
            'original_shape': self.data.shape,
            'final_shape': cleaned_data.shape,
            'dropped_columns': len(cols_to_drop),
            'removed_duplicates': duplicates.sum() if duplicates.any() else 0
        }
        
        return cleaned_data
    
    def _standardize_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical variable names and values."""
        # Standardize income group names
        if 'income' in data.columns:
            income_mapping = {
                'High income': 'High Income',
                'Upper middle income': 'Upper Middle Income',
                'Lower middle income': 'Lower Middle Income',
                'Low income': 'Low Income'
            }
            data['income'] = data['income'].replace(income_mapping)
        
        # Standardize region names (if needed)
        if 'region' in data.columns:
            data['region'] = data['region'].str.title()
        
        return data
    
    def _detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers using multiple methods."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'imfn']]
        
        if len(numeric_cols) == 0:
            return data
        
        # Statistical outlier detection (IQR method)
        outlier_flags = pd.DataFrame(index=data.index)
        
        for col in numeric_cols:
            if data[col].notna().sum() < 10:  # Skip columns with too few values
                continue
                
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_flags[f'{col}_outlier'] = outliers
        
        # Add outlier summary
        data['total_outlier_flags'] = outlier_flags.sum(axis=1)
        
        return data
    
    def impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced missing value imputation using multiple strategies.
        
        Args:
            data: DataFrame with missing values
            
        Returns:
            DataFrame with imputed values
        """
        logger.info("Starting missing value imputation")
        imputed_data = data.copy()
        
        # Get numeric columns for imputation
        numeric_cols = imputed_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'imfn', 'total_outlier_flags']]
        
        if len(numeric_cols) == 0:
            return imputed_data
        
        # 1. Temporal interpolation for time series data
        if self.config['temporal_interpolation'] and 'year' in imputed_data.columns:
            imputed_data = self._temporal_interpolation(imputed_data, numeric_cols)
        
        # 2. Cross-sectional imputation
        method = self.config['imputation_method']
        
        if method == 'knn':
            self.imputer = KNNImputer(n_neighbors=5, weights='distance')
        elif method == 'iterative':
            self.imputer = IterativeImputer(random_state=42, max_iter=10)
        else:
            logger.warning(f"Unknown imputation method: {method}. Using KNN.")
            self.imputer = KNNImputer(n_neighbors=5, weights='distance')
        
        # Fit and transform numeric data
        imputed_values = self.imputer.fit_transform(imputed_data[numeric_cols])
        imputed_data[numeric_cols] = imputed_values
        
        # Track imputation statistics
        original_missing = data[numeric_cols].isnull().sum().sum()
        remaining_missing = imputed_data[numeric_cols].isnull().sum().sum()
        
        logger.info(f"Imputation completed. Reduced missing values from {original_missing} to {remaining_missing}")
        
        self.preprocessing_stats['imputation'] = {
            'method': method,
            'original_missing': original_missing,
            'remaining_missing': remaining_missing,
            'imputation_rate': (original_missing - remaining_missing) / original_missing * 100
        }
        
        return imputed_data
    
    def _temporal_interpolation(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Perform temporal interpolation for time series data."""
        logger.info("Performing temporal interpolation")
        
        interpolated_data = data.copy()
        
        # Group by country and interpolate within each country's time series
        for country in data['country'].unique():
            country_mask = data['country'] == country
            country_data = data[country_mask].sort_values('year')
            
            if len(country_data) < 3:  # Need at least 3 points for interpolation
                continue
            
            for col in numeric_cols:
                if country_data[col].notna().sum() >= 2:  # Need at least 2 non-null values
                    # Linear interpolation
                    interpolated_values = country_data[col].interpolate(method='linear')
                    interpolated_data.loc[country_mask, col] = interpolated_values
        
        return interpolated_data
    
    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for financial development indicators.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        if not self.config['feature_engineering']:
            return data
        
        logger.info("Starting feature engineering")
        engineered_data = data.copy()
        
        # 1. Create composite indices for each dimension
        engineered_data = self._create_composite_indices(engineered_data)
        
        # 2. Create temporal features
        if 'year' in engineered_data.columns:
            engineered_data = self._create_temporal_features(engineered_data)
        
        # 3. Create interaction features
        engineered_data = self._create_interaction_features(engineered_data)
        
        # 4. Create relative performance features
        engineered_data = self._create_relative_features(engineered_data)
        
        # 5. Create volatility and trend features
        engineered_data = self._create_volatility_features(engineered_data)
        
        logger.info(f"Feature engineering completed. New shape: {engineered_data.shape}")
        return engineered_data
    
    def _create_composite_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create composite indices for each dimension of the 4x2 framework."""
        for category, indicators_dict in self.all_indicators.items():
            if category == 'other':  # Skip other indicators for composite indices
                continue
                
            for sector, indicators in indicators_dict.items():
                available_indicators = [col for col in indicators if col in data.columns]
                
                if len(available_indicators) >= 2:
                    # Simple average composite index
                    index_name = f'{category}_{sector}_index'
                    data[index_name] = data[available_indicators].mean(axis=1, skipna=True)
                    
                    # Weighted composite index (equal weights for now)
                    weighted_index_name = f'{category}_{sector}_weighted_index'
                    data[weighted_index_name] = data[available_indicators].mean(axis=1, skipna=True)
        
        # Overall financial development index
        composite_cols = [col for col in data.columns if col.endswith('_index')]
        if composite_cols:
            data['overall_financial_development_index'] = data[composite_cols].mean(axis=1, skipna=True)
        
        return data
    
    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features based on year information."""
        # Time-based features
        data['year_normalized'] = (data['year'] - data['year'].min()) / (data['year'].max() - data['year'].min())
        data['decade'] = (data['year'] // 10) * 10
        
        # Create lagged features for key indicators
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        key_indicators = [col for col in numeric_cols if any(col.startswith(prefix) for prefix in ['di', 'dm', 'ai', 'am'])][:10]  # Top 10
        
        for country in data['country'].unique():
            country_mask = data['country'] == country
            country_data = data[country_mask].sort_values('year')
            
            for col in key_indicators:
                if col in country_data.columns:
                    # 1-year lag
                    lag_col = f'{col}_lag1'
                    data.loc[country_mask, lag_col] = country_data[col].shift(1)
                    
                    # 3-year moving average
                    ma_col = f'{col}_ma3'
                    data.loc[country_mask, ma_col] = country_data[col].rolling(window=3, min_periods=2).mean()
                    
                    # Year-over-year growth rate
                    growth_col = f'{col}_growth'
                    data.loc[country_mask, growth_col] = country_data[col].pct_change()
        
        return data
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different dimensions."""
        # Key interaction features
        interactions = [
            ('access_institutions_index', 'depth_institutions_index', 'access_depth_interaction'),
            ('efficiency_institutions_index', 'stability_institutions_index', 'efficiency_stability_interaction'),
            ('depth_institutions_index', 'depth_markets_index', 'institution_market_depth_interaction'),
        ]
        
        for col1, col2, interaction_name in interactions:
            if col1 in data.columns and col2 in data.columns:
                data[interaction_name] = data[col1] * data[col2]
        
        return data
    
    def _create_relative_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features relative to income group and regional averages."""
        grouping_vars = ['income', 'region']
        
        for group_var in grouping_vars:
            if group_var not in data.columns:
                continue
                
            numeric_cols = [col for col in data.columns if col.endswith('_index')][:5]  # Top 5 composite indices
            
            for col in numeric_cols:
                if col in data.columns:
                    # Relative to group average
                    group_means = data.groupby(group_var)[col].transform('mean')
                    data[f'{col}_relative_to_{group_var}'] = data[col] - group_means
                    
                    # Percentile within group
                    data[f'{col}_percentile_in_{group_var}'] = data.groupby(group_var)[col].transform(
                        lambda x: pd.qcut(x.rank(method='first'), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                    )
        
        return data
    
    def _create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility and trend features for key indicators."""
        if 'year' not in data.columns:
            return data
        
        key_indicators = [col for col in data.columns if col.endswith('_index')][:5]
        
        for country in data['country'].unique():
            country_mask = data['country'] == country
            country_data = data[country_mask].sort_values('year')
            
            if len(country_data) < 5:  # Need sufficient data for volatility calculation
                continue
            
            for col in key_indicators:
                if col in country_data.columns and country_data[col].notna().sum() >= 3:
                    # Rolling volatility (3-year window)
                    volatility_col = f'{col}_volatility'
                    data.loc[country_mask, volatility_col] = country_data[col].rolling(
                        window=3, min_periods=2
                    ).std()
                    
                    # Trend indicator (linear trend over 5-year window)
                    trend_col = f'{col}_trend'
                    rolling_trend = country_data[col].rolling(window=5, min_periods=3).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x.dropna()) >= 3 else np.nan
                    )
                    data.loc[country_mask, trend_col] = rolling_trend
        
        return data
    
    def create_robust_ml_pipeline(self, target_col: str, model_type: str = 'default') -> Pipeline:
        """
        Create a robust ML pipeline with proper missing value handling.
        
        Args:
            target_col: Name of the target column
            model_type: Type of model ('default', 'tree_based', 'linear')
            
        Returns:
            Configured sklearn Pipeline
        """
        from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import GridSearchCV
        
        # Define preprocessing steps based on model type
        if model_type == 'tree_based':
            # Tree-based models can handle some missing values
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', LabelEncoder())
            ])
            
        elif model_type == 'linear':
            # Linear models need complete data
            numeric_transformer = Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', LabelEncoder())
            ])
            
        else:  # default - comprehensive imputation
            numeric_transformer = Pipeline(steps=[
                ('simple_imputer', SimpleImputer(strategy='median')),
                ('iterative_imputer', IterativeImputer(random_state=42, max_iter=10)),
                ('scaler', RobustScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', LabelEncoder())
            ])
        
        # Identify numeric and categorical columns
        if self.data is not None:
            numeric_features = self.data.select_dtypes(include=[np.number]).columns.drop([target_col])
            categorical_features = self.data.select_dtypes(include=['object']).columns
        else:
            # Fallback if no data loaded
            numeric_features = []
            categorical_features = []
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Choose model based on type
        if model_type == 'tree_based':
            # Use HistGradientBoostingRegressor which handles NaNs natively
            model = HistGradientBoostingRegressor(
                random_state=42,
                max_iter=100,
                learning_rate=0.1
            )
        elif model_type == 'linear':
            model = LinearRegression()
        else:
            # Default: Random Forest with proper imputation
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        return pipeline
    
    def get_nan_tolerant_models(self) -> Dict[str, Any]:
        """
        Get a dictionary of models that can handle NaN values natively.
        
        Returns:
            Dictionary of model names and configured model instances
        """
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        from lightgbm import LGBMRegressor, LGBMClassifier
        from xgboost import XGBRegressor, XGBClassifier
        import catboost as cb
        
        nan_tolerant_models = {
            'hist_gradient_boosting_regressor': HistGradientBoostingRegressor(
                random_state=42,
                max_iter=100,
                learning_rate=0.1,
                max_depth=6
            ),
            'hist_gradient_boosting_classifier': HistGradientBoostingClassifier(
                random_state=42,
                max_iter=100,
                learning_rate=0.1,
                max_depth=6
            ),
            'lightgbm_regressor': LGBMRegressor(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                verbosity=-1
            ),
            'xgboost_regressor': XGBRegressor(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                eval_metric='rmse'
            ),
            'catboost_regressor': cb.CatBoostRegressor(
                random_state=42,
                iterations=100,
                learning_rate=0.1,
                verbose=False
            )
        }
        
        return nan_tolerant_models
    
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using the specified scaling method.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with scaled features
        """
        logger.info("Starting feature scaling")
        scaled_data = data.copy()
        
        # Get numeric columns to scale (exclude ID columns and categorical encodings)
        numeric_cols = scaled_data.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if not any(col.startswith(prefix) 
                        for prefix in ['year', 'imfn', 'total_outlier_flags'])]
        
        if len(cols_to_scale) == 0:
            return scaled_data
        
        # Initialize scaler
        method = self.config['scaling_method']
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}. Using robust scaler.")
            self.scaler = RobustScaler()
        
        # Handle infinity and extreme values before scaling
        scaled_data[cols_to_scale] = scaled_data[cols_to_scale].replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme values at 99.9th percentile
        for col in cols_to_scale:
            if scaled_data[col].notna().sum() > 0:
                upper_cap = scaled_data[col].quantile(0.999)
                lower_cap = scaled_data[col].quantile(0.001)
                scaled_data[col] = scaled_data[col].clip(lower=lower_cap, upper=upper_cap)
        
        # Fill any remaining NaNs with median
        scaled_data[cols_to_scale] = scaled_data[cols_to_scale].fillna(scaled_data[cols_to_scale].median())
        
        # Fit and transform
        scaled_values = self.scaler.fit_transform(scaled_data[cols_to_scale])
        scaled_data[cols_to_scale] = scaled_values
        
        logger.info(f"Feature scaling completed using {method} method for {len(cols_to_scale)} features")
        
        self.preprocessing_stats['scaling'] = {
            'method': method,
            'features_scaled': len(cols_to_scale),
            'scaler_params': self.scaler.get_params() if hasattr(self.scaler, 'get_params') else {}
        }
        
        return scaled_data
    
    def preprocess_pipeline(self, file_path: str) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Fully preprocessed DataFrame
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Load data
        self.load_data(file_path)
        
        # Assess data quality
        quality_assessment = self.assess_data_quality()
        logger.info(f"Data quality assessment completed. Missing data: {quality_assessment['missing_data']['missing_percentage']:.2f}%")
        
        # Clean data
        cleaned_data = self.clean_data()
        
        # Impute missing values
        imputed_data = self.impute_missing_values(cleaned_data)
        
        # Feature engineering
        engineered_data = self.feature_engineering(imputed_data)
        
        # Scale features
        final_data = self.scale_features(engineered_data)
        
        # Store feature names
        self.feature_names = list(final_data.columns)
        
        logger.info(f"Preprocessing pipeline completed. Final dataset shape: {final_data.shape}")
        
        return final_data
    
    def save_preprocessed_data(self, data: pd.DataFrame, output_path: str) -> None:
        """Save preprocessed data and preprocessing artifacts."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main data
        data.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to {output_path}")
        
        # Save preprocessing artifacts
        artifacts_dir = output_path.parent / "preprocessing_artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, artifacts_dir / "scaler.pkl")
        
        # Save imputer  
        if self.imputer is not None:
            joblib.dump(self.imputer, artifacts_dir / "imputer.pkl")
        
        # Save preprocessing statistics
        import json
        with open(artifacts_dir / "preprocessing_stats.json", 'w') as f:
            json.dump(self.preprocessing_stats, f, indent=2, default=str)
        
        logger.info(f"Preprocessing artifacts saved to {artifacts_dir}")
    
    def generate_preprocessing_report(self) -> str:
        """Generate a comprehensive preprocessing report."""
        if not self.preprocessing_stats:
            return "No preprocessing statistics available."
        
        report = []
        report.append("=" * 80)
        report.append("GLOBAL FINANCIAL DEVELOPMENT DATABASE - PREPROCESSING REPORT")
        report.append("=" * 80)
        
        # Data Quality Assessment
        if 'data_quality' in self.preprocessing_stats:
            quality = self.preprocessing_stats['data_quality']
            report.append("\n1. DATA QUALITY ASSESSMENT")
            report.append("-" * 40)
            report.append(f"Total Records: {quality['basic_stats']['total_rows']:,}")
            report.append(f"Total Features: {quality['basic_stats']['total_columns']}")
            report.append(f"Memory Usage: {quality['basic_stats']['memory_usage_mb']:.2f} MB")
            report.append(f"Overall Missing Data: {quality['missing_data']['missing_percentage']:.2f}%")
            
            if 'temporal_coverage' in quality:
                temp = quality['temporal_coverage']
                report.append(f"Temporal Coverage: {temp['start_year']} - {temp['end_year']} ({temp['total_years']} years)")
            
            if 'geographical_coverage' in quality:
                geo = quality['geographical_coverage']
                report.append(f"Geographical Coverage: {geo['total_countries']} countries")
        
        # Cleaning Statistics
        if 'cleaning' in self.preprocessing_stats:
            cleaning = self.preprocessing_stats['cleaning']
            report.append("\n2. DATA CLEANING")
            report.append("-" * 40)
            report.append(f"Original Shape: {cleaning['original_shape']}")
            report.append(f"Final Shape: {cleaning['final_shape']}")
            report.append(f"Dropped Columns: {cleaning['dropped_columns']}")
            report.append(f"Removed Duplicates: {cleaning['removed_duplicates']}")
        
        # Imputation Statistics
        if 'imputation' in self.preprocessing_stats:
            imputation = self.preprocessing_stats['imputation']
            report.append("\n3. MISSING VALUE IMPUTATION")
            report.append("-" * 40)
            report.append(f"Method: {imputation['method'].upper()}")
            report.append(f"Original Missing Values: {imputation['original_missing']:,}")
            report.append(f"Remaining Missing Values: {imputation['remaining_missing']:,}")
            report.append(f"Imputation Rate: {imputation['imputation_rate']:.2f}%")
        
        # Scaling Statistics
        if 'scaling' in self.preprocessing_stats:
            scaling = self.preprocessing_stats['scaling']
            report.append("\n4. FEATURE SCALING")
            report.append("-" * 40)
            report.append(f"Method: {scaling['method'].upper()}")
            report.append(f"Features Scaled: {scaling['features_scaled']}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def main():
    """Main execution function for preprocessing."""
    # Configuration
    config = {
        'missing_threshold': 0.90,  # More lenient for financial data
        'outlier_contamination': 0.05,
        'imputation_method': 'knn',
        'scaling_method': 'robust',
        'min_years_per_country': 3,
        'temporal_interpolation': True,
        'feature_engineering': True,
    }
    
    # Initialize preprocessor
    preprocessor = GFDPreprocessor(config)
    
    try:
        # Run preprocessing pipeline
        data_path = r"C:\Users\Krishnendu Rarhi\Downloads\Datasets\Global FInancial Dataset\20220909-global-financial-development-database.xlsx"
        processed_data = preprocessor.preprocess_pipeline(data_path)
        
        # Save results
        output_path = Path("data/processed/gfd_preprocessed_data.csv")
        preprocessor.save_preprocessed_data(processed_data, str(output_path))
        
        # Generate report
        report = preprocessor.generate_preprocessing_report()
        print(report)
        
        # Save report
        with open("data/processed/preprocessing_report.txt", 'w') as f:
            f.write(report)
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()