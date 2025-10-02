#!/usr/bin/env python3
"""
Global Financial Development Database Explorer

This module provides comprehensive data exploration capabilities for the
Global Financial Development Database, implementing the 4x2 framework analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class GFDDataExplorer:
    """
    Comprehensive explorer for Global Financial Development Database.
    
    The database follows a 4x2 framework:
    - 4 characteristics: Depth, Access, Efficiency, Stability
    - 2 sectors: Financial Institutions, Financial Markets
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the explorer with the dataset path.
        
        Args:
            data_path: Path to the Global Financial Development Database Excel file
        """
        self.data_path = Path(data_path)
        self.data = None
        self.metadata = None
        self.framework_mapping = {
            'FI': 'Financial Institutions',
            'FM': 'Financial Markets',
            'D': 'Depth',
            'A': 'Access', 
            'E': 'Efficiency',
            'S': 'Stability'
        }
        
    def load_data(self) -> None:
        """Load the Global Financial Development Database from Excel file."""
        try:
            # Load the main data sheet
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_excel(self.data_path, sheet_name='Data - August 2022')
            
            # Load metadata if available
            try:
                self.metadata = pd.read_excel(self.data_path, sheet_name='Metadata')
                logger.info("Metadata loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'data_types': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'null_counts': self.data.isnull().sum().to_dict(),
            'countries': self.data['Country'].nunique() if 'Country' in self.data.columns else 'N/A',
            'years': f"{self.data['Year'].min()} - {self.data['Year'].max()}" if 'Year' in self.data.columns else 'N/A'
        }
        
        return info
        
    def analyze_framework_structure(self) -> Dict[str, Any]:
        """
        Analyze the 4x2 framework structure of the database.
        
        Returns:
            Dictionary containing framework analysis
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Get indicator columns (exclude Country, Year, etc.)
        indicator_cols = [col for col in self.data.columns 
                         if col not in ['Country', 'Country Code', 'Year']]
        
        framework_analysis = {
            'total_indicators': len(indicator_cols),
            'financial_institutions': {},
            'financial_markets': {},
            'characteristics': {'depth': 0, 'access': 0, 'efficiency': 0, 'stability': 0}
        }
        
        # Analyze indicators by framework
        fi_indicators = [col for col in indicator_cols if col.startswith('FI')]
        fm_indicators = [col for col in indicator_cols if col.startswith('FM')]
        
        framework_analysis['financial_institutions']['count'] = len(fi_indicators)
        framework_analysis['financial_markets']['count'] = len(fm_indicators)
        
        # Analyze by characteristics
        for char in ['D', 'A', 'E', 'S']:
            char_indicators = [col for col in indicator_cols if f'.{char}.' in col]
            char_name = {'D': 'depth', 'A': 'access', 'E': 'efficiency', 'S': 'stability'}[char]
            framework_analysis['characteristics'][char_name] = len(char_indicators)
            
        return framework_analysis
        
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        quality_report = {}
        
        # Missing data analysis
        missing_data = self.data.isnull().sum()
        quality_report['missing_data'] = {
            'total_missing': missing_data.sum(),
            'missing_percentage': (missing_data.sum() / self.data.size) * 100,
            'columns_with_missing': missing_data[missing_data > 0].to_dict()
        }
        
        # Data completeness by country
        if 'Country' in self.data.columns:
            country_completeness = self.data.groupby('Country').apply(
                lambda x: (x.notnull().sum().sum() / x.size) * 100
            ).sort_values(ascending=False)
            
            quality_report['country_completeness'] = {
                'top_10_complete': country_completeness.head(10).to_dict(),
                'bottom_10_complete': country_completeness.tail(10).to_dict(),
                'average_completeness': country_completeness.mean()
            }
            
        # Data completeness by year
        if 'Year' in self.data.columns:
            year_completeness = self.data.groupby('Year').apply(
                lambda x: (x.notnull().sum().sum() / x.size) * 100
            )
            
            quality_report['year_completeness'] = {
                'by_year': year_completeness.to_dict(),
                'trend': 'improving' if year_completeness.iloc[-5:].mean() > year_completeness.iloc[:5].mean() else 'declining'
            }
            
        return quality_report
        
    def analyze_temporal_trends(self, indicators: List[str] = None) -> Dict[str, Any]:
        """
        Analyze temporal trends for key indicators.
        
        Args:
            indicators: List of indicators to analyze. If None, analyzes top indicators.
            
        Returns:
            Dictionary containing trend analysis
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if 'Year' not in self.data.columns:
            return {'error': 'Year column not found in data'}
            
        # Select indicators to analyze
        if indicators is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            indicators = [col for col in numeric_cols if col != 'Year'][:10]  # Top 10
            
        trends = {}
        
        for indicator in indicators:
            if indicator not in self.data.columns:
                continue
                
            # Calculate global trend (average across countries)
            yearly_avg = self.data.groupby('Year')[indicator].mean()
            
            # Calculate trend statistics
            trend_data = {
                'start_value': yearly_avg.iloc[0] if len(yearly_avg) > 0 else None,
                'end_value': yearly_avg.iloc[-1] if len(yearly_avg) > 0 else None,
                'growth_rate': None,
                'volatility': yearly_avg.std(),
                'data_points': len(yearly_avg)
            }
            
            if trend_data['start_value'] and trend_data['end_value']:
                years = yearly_avg.index[-1] - yearly_avg.index[0]
                trend_data['growth_rate'] = (
                    (trend_data['end_value'] / trend_data['start_value']) ** (1/years) - 1
                ) * 100 if years > 0 else 0
                
            trends[indicator] = trend_data
            
        return trends
        
    def get_country_rankings(self, indicators: List[str], year: int = 2021) -> Dict[str, Any]:
        """
        Get country rankings for specified indicators.
        
        Args:
            indicators: List of indicators for ranking
            year: Year for ranking (default: 2021)
            
        Returns:
            Dictionary containing rankings
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        year_data = self.data[self.data['Year'] == year] if 'Year' in self.data.columns else self.data
        
        if year_data.empty:
            return {'error': f'No data available for year {year}'}
            
        rankings = {}
        
        for indicator in indicators:
            if indicator not in year_data.columns:
                continue
                
            # Create ranking
            indicator_data = year_data[['Country', indicator]].dropna()
            indicator_data = indicator_data.sort_values(indicator, ascending=False)
            
            rankings[indicator] = {
                'top_10': indicator_data.head(10)[['Country', indicator]].to_dict('records'),
                'bottom_10': indicator_data.tail(10)[['Country', indicator]].to_dict('records'),
                'total_countries': len(indicator_data),
                'global_median': indicator_data[indicator].median(),
                'global_mean': indicator_data[indicator].mean()
            }
            
        return rankings
        
    def create_correlation_matrix(self, indicator_type: str = 'all') -> pd.DataFrame:
        """
        Create correlation matrix for financial indicators.
        
        Args:
            indicator_type: 'FI' for Financial Institutions, 'FM' for Financial Markets, 'all' for both
            
        Returns:
            Correlation matrix DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Select numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Filter by indicator type
        if indicator_type == 'FI':
            cols = [col for col in numeric_cols if col.startswith('FI')]
        elif indicator_type == 'FM':
            cols = [col for col in numeric_cols if col.startswith('FM')]
        else:
            cols = [col for col in numeric_cols if col not in ['Year']]
            
        if len(cols) == 0:
            return pd.DataFrame()
            
        return self.data[cols].corr()
        
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        summary = {
            'descriptive_stats': numeric_data.describe().to_dict(),
            'skewness': numeric_data.skew().to_dict(),
            'kurtosis': numeric_data.kurtosis().to_dict(),
            'data_range': {
                col: {'min': numeric_data[col].min(), 'max': numeric_data[col].max()}
                for col in numeric_data.columns
            }
        }
        
        return summary
        
    def export_processed_data(self, output_path: str, format: str = 'csv') -> None:
        """
        Export processed data to specified format.
        
        Args:
            output_path: Output file path
            format: Export format ('csv', 'parquet', 'json')
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        output_path = Path(output_path)
        
        if format == 'csv':
            self.data.to_csv(output_path, index=False)
        elif format == 'parquet':
            self.data.to_parquet(output_path, index=False)
        elif format == 'json':
            self.data.to_json(output_path, orient='records')
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Data exported to {output_path}")


def main():
    """Main execution function for data exploration."""
    
    # Initialize explorer
    data_path = r"C:\Users\Krishnendu Rarhi\Downloads\Datasets\Global FInancial Dataset\20220909-global-financial-development-database.xlsx"
    explorer = GFDDataExplorer(data_path)
    
    try:
        # Load data
        explorer.load_data()
        
        # Basic information
        basic_info = explorer.get_basic_info()
        print("=== BASIC DATA INFORMATION ===")
        for key, value in basic_info.items():
            print(f"{key}: {value}")
        
        # Framework analysis
        framework = explorer.analyze_framework_structure()
        print("\n=== 4x2 FRAMEWORK ANALYSIS ===")
        print(f"Total Indicators: {framework['total_indicators']}")
        print(f"Financial Institutions: {framework['financial_institutions']['count']}")
        print(f"Financial Markets: {framework['financial_markets']['count']}")
        print("Characteristics:")
        for char, count in framework['characteristics'].items():
            print(f"  {char.title()}: {count}")
        
        # Data quality report
        quality = explorer.get_data_quality_report()
        print("\n=== DATA QUALITY REPORT ===")
        print(f"Total Missing Values: {quality['missing_data']['total_missing']}")
        print(f"Missing Percentage: {quality['missing_data']['missing_percentage']:.2f}%")
        
        # Export processed data
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        explorer.export_processed_data(
            output_dir / "gfd_data_clean.csv", 
            format='csv'
        )
        
        print(f"\nData exploration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during exploration: {e}")
        raise


if __name__ == "__main__":
    main()