#!/usr/bin/env python3
"""
ETL Pipeline for Global Financial Development Database Integration

This module provides a comprehensive ETL pipeline for integrating the Global Financial
Development Database into the existing risk governance platform, with automated
data ingestion, transformation, and loading capabilities.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import json
import os
import schedule
import time
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import yaml

# Import our existing components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.gfd_preprocessor import GFDPreprocessor
from src.utils.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class FinancialDevelopmentData(Base):
    """SQLAlchemy model for financial development data."""
    __tablename__ = 'financial_development_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    country_code = Column(String(3), nullable=False, index=True)
    country_name = Column(String(100), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    region = Column(String(50), nullable=True)
    income_group = Column(String(50), nullable=True)
    
    # 4x2 Framework Composite Indices
    access_institutions_index = Column(Float, nullable=True)
    access_markets_index = Column(Float, nullable=True)
    depth_institutions_index = Column(Float, nullable=True)
    depth_markets_index = Column(Float, nullable=True)
    efficiency_institutions_index = Column(Float, nullable=True)
    efficiency_markets_index = Column(Float, nullable=True)
    stability_institutions_index = Column(Float, nullable=True)
    stability_markets_index = Column(Float, nullable=True)
    
    # Overall financial development index
    overall_financial_development_index = Column(Float, nullable=True)
    
    # Raw indicator data (JSON for flexibility)
    raw_indicators = Column(Text, nullable=True)  # JSON string of all indicators
    
    # Metadata
    data_source = Column(String(50), default='WB_GFD')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CountryRiskProfile(Base):
    """Enhanced country risk profile with financial development integration."""
    __tablename__ = 'country_risk_profiles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    country_code = Column(String(3), nullable=False, index=True)
    country_name = Column(String(100), nullable=False)
    year = Column(Integer, nullable=False, index=True)
    
    # Risk scores (0-100 scale)
    financial_development_score = Column(Float, nullable=True)
    systemic_risk_score = Column(Float, nullable=True)
    market_volatility_score = Column(Float, nullable=True)
    institutional_quality_score = Column(Float, nullable=True)
    
    # Risk categories
    risk_category = Column(String(20), nullable=True)  # 'Low', 'Medium', 'High', 'Very High'
    
    # Metadata
    model_version = Column(String(20), nullable=True)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class GFDETLPipeline:
    """
    Comprehensive ETL Pipeline for Global Financial Development Database.
    
    Features:
    - Automated data ingestion from Excel sources
    - Advanced preprocessing and transformation
    - Database integration with existing risk platform
    - Incremental loading and data versioning
    - Data quality monitoring and alerting
    - Scheduled batch processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ETL pipeline with configuration.
        
        Args:
            config: Configuration dictionary for ETL settings
        """
        self.config = config or self._default_config()
        self.config_manager = ConfigManager()
        self.preprocessor = GFDPreprocessor()
        
        # Database connection
        self.engine = None
        self.session = None
        
        # ETL state tracking
        self.etl_stats = {
            'last_run': None,
            'records_processed': 0,
            'records_inserted': 0,
            'records_updated': 0,
            'processing_time': 0,
            'data_quality_score': 0,
            'errors': []
        }
        
        # Initialize database connection
        self._initialize_database()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default ETL configuration."""
        return {
            'source_file_path': r"C:\Users\Krishnendu Rarhi\Downloads\Datasets\Global FInancial Dataset\20220909-global-financial-development-database.xlsx",
            'staging_directory': Path("data/staging"),
            'processed_directory': Path("data/processed"),
            'backup_directory': Path("data/backups"),
            'batch_size': 1000,
            'max_retries': 3,
            'data_quality_threshold': 0.8,
            'enable_incremental_loading': True,
            'enable_data_versioning': True,
            'schedule_frequency': 'weekly',  # 'daily', 'weekly', 'monthly'
            'notification_emails': [],
            'enable_monitoring': True
        }
    
    def _initialize_database(self) -> None:
        """Initialize database connection and create tables if needed."""
        try:
            # Get database URL from config manager
            db_url = self.config_manager.get_database_url("primary")
            self.engine = create_engine(db_url, echo=False)
            
            # Create session factory
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            
            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Fall back to SQLite for development
            sqlite_path = Path("data/gfd_database.db")
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            Base.metadata.create_all(self.engine)
            
            logger.info("Using SQLite fallback database")
    
    async def extract_data(self) -> pd.DataFrame:
        """
        Extract data from the Global Financial Development Database.
        
        Returns:
            Raw DataFrame from the source
        """
        logger.info("Starting data extraction")
        start_time = time.time()
        
        try:
            # Check if source file exists
            source_path = Path(self.config['source_file_path'])
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            # Load data using our preprocessor
            self.preprocessor.load_data(str(source_path))
            raw_data = self.preprocessor.data.copy()
            
            extraction_time = time.time() - start_time
            logger.info(f"Data extraction completed in {extraction_time:.2f} seconds. Shape: {raw_data.shape}")
            
            self.etl_stats['extraction_time'] = extraction_time
            self.etl_stats['raw_records'] = len(raw_data)
            
            return raw_data
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            self.etl_stats['errors'].append(f"Extraction error: {str(e)}")
            raise
    
    async def transform_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data using comprehensive preprocessing pipeline.
        
        Args:
            raw_data: Raw DataFrame from extraction
            
        Returns:
            Transformed and cleaned DataFrame
        """
        logger.info("Starting data transformation")
        start_time = time.time()
        
        try:
            # Set the data in preprocessor
            self.preprocessor.data = raw_data
            
            # Run transformation pipeline
            quality_assessment = self.preprocessor.assess_data_quality()
            cleaned_data = self.preprocessor.clean_data()
            imputed_data = self.preprocessor.impute_missing_values(cleaned_data)
            engineered_data = self.preprocessor.feature_engineering(imputed_data)
            final_data = self.preprocessor.scale_features(engineered_data)
            
            transformation_time = time.time() - start_time
            logger.info(f"Data transformation completed in {transformation_time:.2f} seconds")
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score(quality_assessment, final_data)
            
            self.etl_stats['transformation_time'] = transformation_time
            self.etl_stats['transformed_records'] = len(final_data)
            self.etl_stats['data_quality_score'] = data_quality_score
            
            # Check if data quality meets threshold
            if data_quality_score < self.config['data_quality_threshold']:
                logger.warning(f"Data quality score {data_quality_score} below threshold {self.config['data_quality_threshold']}")
                self.etl_stats['errors'].append(f"Data quality below threshold: {data_quality_score}")
            
            return final_data
            
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            self.etl_stats['errors'].append(f"Transformation error: {str(e)}")
            raise
    
    def _calculate_data_quality_score(self, quality_assessment: Dict[str, Any], processed_data: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        score = 0.0
        
        # Completeness score (30%)
        completeness = (1 - quality_assessment['missing_data']['missing_percentage'] / 100) * 0.3
        
        # Consistency score (25%) - based on temporal coverage
        if 'temporal_coverage' in quality_assessment:
            temporal_score = min(quality_assessment['temporal_coverage']['total_years'] / 50, 1.0) * 0.25
        else:
            temporal_score = 0.0
        
        # Validity score (25%) - based on successful transformations
        validity_score = min(len(processed_data) / self.etl_stats['raw_records'], 1.0) * 0.25
        
        # Accuracy score (20%) - based on outlier detection
        if 'total_outlier_flags' in processed_data.columns:
            outlier_rate = processed_data['total_outlier_flags'].mean()
            accuracy_score = max(1 - outlier_rate / 10, 0) * 0.2  # Assume 10+ outliers per row is concerning
        else:
            accuracy_score = 0.2
        
        score = completeness + temporal_score + validity_score + accuracy_score
        return min(score, 1.0)
    
    async def load_data(self, transformed_data: pd.DataFrame) -> Dict[str, int]:
        """
        Load transformed data into the database.
        
        Args:
            transformed_data: Preprocessed DataFrame
            
        Returns:
            Dictionary with loading statistics
        """
        logger.info("Starting data loading")
        start_time = time.time()
        
        loading_stats = {'inserted': 0, 'updated': 0, 'errors': 0}
        
        try:
            # Prepare data for loading
            records_to_load = self._prepare_records_for_loading(transformed_data)
            
            # Load in batches
            batch_size = self.config['batch_size']
            total_records = len(records_to_load)
            
            for i in range(0, total_records, batch_size):
                batch = records_to_load[i:i + batch_size]
                batch_stats = await self._load_batch(batch)
                
                loading_stats['inserted'] += batch_stats['inserted']
                loading_stats['updated'] += batch_stats['updated']
                loading_stats['errors'] += batch_stats['errors']
                
                # Progress logging
                progress = min((i + batch_size) / total_records * 100, 100)
                logger.info(f"Loading progress: {progress:.1f}% ({i + len(batch)}/{total_records})")
            
            # Commit transaction
            self.session.commit()
            
            # Update country risk profiles
            await self._update_country_risk_profiles(transformed_data)
            
            loading_time = time.time() - start_time
            logger.info(f"Data loading completed in {loading_time:.2f} seconds")
            logger.info(f"Inserted: {loading_stats['inserted']}, Updated: {loading_stats['updated']}, Errors: {loading_stats['errors']}")
            
            self.etl_stats['loading_time'] = loading_time
            self.etl_stats['records_inserted'] = loading_stats['inserted']
            self.etl_stats['records_updated'] = loading_stats['updated']
            
            return loading_stats
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Data loading failed: {e}")
            self.etl_stats['errors'].append(f"Loading error: {str(e)}")
            raise
    
    def _prepare_records_for_loading(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare DataFrame records for database loading."""
        records = []
        
        # Required columns mapping
        column_mapping = {
            'iso3': 'country_code',
            'country': 'country_name',
            'year': 'year',
            'region': 'region',
            'income': 'income_group'
        }
        
        # Composite index columns
        composite_columns = [
            'access_institutions_index', 'access_markets_index',
            'depth_institutions_index', 'depth_markets_index',
            'efficiency_institutions_index', 'efficiency_markets_index',
            'stability_institutions_index', 'stability_markets_index',
            'overall_financial_development_index'
        ]
        
        for idx, row in data.iterrows():
            record = {}
            
            # Map basic columns
            for old_col, new_col in column_mapping.items():
                if old_col in row:
                    record[new_col] = row[old_col]
            
            # Map composite indices
            for col in composite_columns:
                if col in row:
                    record[col] = row[col] if pd.notna(row[col]) else None
            
            # Create raw indicators JSON
            raw_indicators = {}
            for col in data.columns:
                if col.startswith(('ai', 'am', 'di', 'dm', 'ei', 'em', 'si', 'sm', 'oi', 'om')):
                    if pd.notna(row[col]):
                        raw_indicators[col] = float(row[col])
            
            record['raw_indicators'] = json.dumps(raw_indicators)
            records.append(record)
        
        return records
    
    async def _load_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, int]:
        """Load a batch of records into the database."""
        stats = {'inserted': 0, 'updated': 0, 'errors': 0}
        
        for record in batch:
            try:
                # Check if record exists (based on country_code and year)
                existing = self.session.query(FinancialDevelopmentData).filter_by(
                    country_code=record.get('country_code'),
                    year=record.get('year')
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in record.items():
                        if key not in ['id']:  # Don't update primary key
                            setattr(existing, key, value)
                    existing.updated_at = datetime.utcnow()
                    stats['updated'] += 1
                else:
                    # Insert new record
                    new_record = FinancialDevelopmentData(**record)
                    self.session.add(new_record)
                    stats['inserted'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing record: {e}")
                stats['errors'] += 1
                continue
        
        return stats
    
    async def _update_country_risk_profiles(self, data: pd.DataFrame) -> None:
        """Update country risk profiles based on financial development data."""
        logger.info("Updating country risk profiles")
        
        # Calculate country risk scores
        for country_code in data['iso3'].unique():
            country_data = data[data['iso3'] == country_code]
            
            for year in country_data['year'].unique():
                year_data = country_data[country_data['year'] == year].iloc[0]
                
                # Calculate financial development score
                fd_score = self._calculate_financial_development_score(year_data)
                
                # Calculate other risk scores (placeholder logic)
                systemic_risk = self._calculate_systemic_risk_score(year_data)
                volatility_score = self._calculate_market_volatility_score(year_data)
                institutional_score = self._calculate_institutional_quality_score(year_data)
                
                # Determine risk category
                risk_category = self._determine_risk_category(fd_score, systemic_risk, volatility_score)
                
                # Update or insert risk profile
                existing_profile = self.session.query(CountryRiskProfile).filter_by(
                    country_code=country_code, year=year
                ).first()
                
                if existing_profile:
                    existing_profile.financial_development_score = fd_score
                    existing_profile.systemic_risk_score = systemic_risk
                    existing_profile.market_volatility_score = volatility_score
                    existing_profile.institutional_quality_score = institutional_score
                    existing_profile.risk_category = risk_category
                    existing_profile.updated_at = datetime.utcnow()
                else:
                    new_profile = CountryRiskProfile(
                        country_code=country_code,
                        country_name=year_data.get('country', 'Unknown'),
                        year=year,
                        financial_development_score=fd_score,
                        systemic_risk_score=systemic_risk,
                        market_volatility_score=volatility_score,
                        institutional_quality_score=institutional_score,
                        risk_category=risk_category,
                        model_version='1.0',
                        confidence_score=0.85
                    )
                    self.session.add(new_profile)
    
    def _calculate_financial_development_score(self, row: pd.Series) -> Optional[float]:
        """Calculate composite financial development score."""
        if 'overall_financial_development_index' in row and pd.notna(row['overall_financial_development_index']):
            # Normalize to 0-100 scale
            return min(max(row['overall_financial_development_index'] * 100, 0), 100)
        return None
    
    def _calculate_systemic_risk_score(self, row: pd.Series) -> Optional[float]:
        """Calculate systemic risk score based on stability indicators."""
        stability_cols = ['stability_institutions_index', 'stability_markets_index']
        stability_values = [row[col] for col in stability_cols if col in row and pd.notna(row[col])]
        
        if stability_values:
            # Higher stability = lower systemic risk
            avg_stability = np.mean(stability_values)
            return max(100 - avg_stability * 100, 0)
        return None
    
    def _calculate_market_volatility_score(self, row: pd.Series) -> Optional[float]:
        """Calculate market volatility score."""
        # Placeholder - would use market data in real implementation
        if 'efficiency_markets_index' in row and pd.notna(row['efficiency_markets_index']):
            # Lower efficiency might indicate higher volatility
            return max(100 - row['efficiency_markets_index'] * 100, 0)
        return 50.0  # Default medium volatility
    
    def _calculate_institutional_quality_score(self, row: pd.Series) -> Optional[float]:
        """Calculate institutional quality score."""
        institutional_cols = ['efficiency_institutions_index', 'stability_institutions_index']
        institutional_values = [row[col] for col in institutional_cols if col in row and pd.notna(row[col])]
        
        if institutional_values:
            return np.mean(institutional_values) * 100
        return None
    
    def _determine_risk_category(self, fd_score: Optional[float], systemic_risk: Optional[float], 
                                volatility: Optional[float]) -> str:
        """Determine overall risk category."""
        scores = [score for score in [fd_score, systemic_risk, volatility] if score is not None]
        
        if not scores:
            return 'Unknown'
        
        avg_risk = np.mean([100 - fd_score if fd_score else 50, 
                           systemic_risk if systemic_risk else 50,
                           volatility if volatility else 50])
        
        if avg_risk <= 25:
            return 'Low'
        elif avg_risk <= 50:
            return 'Medium'
        elif avg_risk <= 75:
            return 'High'
        else:
            return 'Very High'
    
    async def run_etl_pipeline(self) -> Dict[str, Any]:
        """
        Run complete ETL pipeline.
        
        Returns:
            ETL execution statistics
        """
        logger.info("Starting ETL pipeline execution")
        pipeline_start_time = time.time()
        
        try:
            # Reset ETL stats
            self.etl_stats = {
                'last_run': datetime.utcnow(),
                'records_processed': 0,
                'records_inserted': 0,
                'records_updated': 0,
                'processing_time': 0,
                'data_quality_score': 0,
                'errors': []
            }
            
            # Execute pipeline steps
            raw_data = await self.extract_data()
            transformed_data = await self.transform_data(raw_data)
            loading_stats = await self.load_data(transformed_data)
            
            # Calculate total processing time
            total_time = time.time() - pipeline_start_time
            self.etl_stats['processing_time'] = total_time
            self.etl_stats['records_processed'] = len(transformed_data)
            
            logger.info(f"ETL pipeline completed successfully in {total_time:.2f} seconds")
            
            # Save ETL statistics
            await self._save_etl_stats()
            
            # Send notifications if configured
            if self.config.get('notification_emails'):
                await self._send_notifications()
            
            return self.etl_stats
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            self.etl_stats['errors'].append(f"Pipeline error: {str(e)}")
            raise
        finally:
            if self.session:
                self.session.close()
    
    async def _save_etl_stats(self) -> None:
        """Save ETL execution statistics."""
        stats_dir = Path("data/etl_logs")
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = stats_dir / f"etl_stats_{timestamp}.json"
        
        async with aiofiles.open(stats_file, 'w') as f:
            await f.write(json.dumps(self.etl_stats, indent=2, default=str))
        
        logger.info(f"ETL statistics saved to {stats_file}")
    
    async def _send_notifications(self) -> None:
        """Send ETL completion notifications."""
        # Placeholder for email notifications
        logger.info("Sending ETL completion notifications")
        
        # In real implementation, would integrate with email service
        # For now, just log the notification
        status = "SUCCESS" if not self.etl_stats['errors'] else "WITH ERRORS"
        message = f"""
        ETL Pipeline Execution Report
        ============================
        Status: {status}
        Processing Time: {self.etl_stats['processing_time']:.2f} seconds
        Records Processed: {self.etl_stats['records_processed']:,}
        Records Inserted: {self.etl_stats['records_inserted']:,}
        Records Updated: {self.etl_stats['records_updated']:,}
        Data Quality Score: {self.etl_stats['data_quality_score']:.2%}
        Errors: {len(self.etl_stats['errors'])}
        """
        
        logger.info(message)
    
    def schedule_etl_pipeline(self) -> None:
        """Schedule ETL pipeline execution."""
        frequency = self.config.get('schedule_frequency', 'weekly')
        
        if frequency == 'daily':
            schedule.every().day.at("02:00").do(lambda: asyncio.run(self.run_etl_pipeline()))
        elif frequency == 'weekly':
            schedule.every().monday.at("02:00").do(lambda: asyncio.run(self.run_etl_pipeline()))
        elif frequency == 'monthly':
            schedule.every().month.do(lambda: asyncio.run(self.run_etl_pipeline()))
        
        logger.info(f"ETL pipeline scheduled to run {frequency}")
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def get_etl_status(self) -> Dict[str, Any]:
        """Get current ETL pipeline status."""
        return {
            'last_run': self.etl_stats.get('last_run'),
            'status': 'Healthy' if len(self.etl_stats.get('errors', [])) == 0 else 'Error',
            'data_quality_score': self.etl_stats.get('data_quality_score', 0),
            'records_in_database': self._count_database_records(),
            'pipeline_config': self.config
        }
    
    def _count_database_records(self) -> int:
        """Count total records in the financial development database."""
        try:
            count = self.session.query(FinancialDevelopmentData).count()
            return count
        except:
            return 0


# API Integration for ETL Management
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

etl_router = APIRouter()
etl_pipeline = None

@etl_router.post("/etl/run")
async def trigger_etl_pipeline(background_tasks: BackgroundTasks):
    """Trigger ETL pipeline execution."""
    global etl_pipeline
    
    if etl_pipeline is None:
        etl_pipeline = GFDETLPipeline()
    
    try:
        # Run ETL in background
        background_tasks.add_task(etl_pipeline.run_etl_pipeline)
        return JSONResponse({
            "status": "ETL pipeline started",
            "message": "Pipeline is running in the background"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start ETL pipeline: {str(e)}")

@etl_router.get("/etl/status")
async def get_etl_status():
    """Get ETL pipeline status."""
    global etl_pipeline
    
    if etl_pipeline is None:
        etl_pipeline = GFDETLPipeline()
    
    return etl_pipeline.get_etl_status()

@etl_router.get("/etl/stats")
async def get_etl_stats():
    """Get ETL execution statistics."""
    global etl_pipeline
    
    if etl_pipeline is None:
        return {"error": "ETL pipeline not initialized"}
    
    return etl_pipeline.etl_stats


async def main():
    """Main execution function for ETL pipeline."""
    # Initialize ETL pipeline
    etl_pipeline = GFDETLPipeline()
    
    try:
        # Run ETL pipeline
        stats = await etl_pipeline.run_etl_pipeline()
        print(f"ETL Pipeline completed successfully!")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        print(f"Records processed: {stats['records_processed']:,}")
        print(f"Data quality score: {stats['data_quality_score']:.2%}")
        
    except Exception as e:
        logger.error(f"ETL pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())