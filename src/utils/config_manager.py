"""
Configuration Management for Financial Risk Governance BI Platform.

This module provides centralized configuration management with support for
YAML files, environment variables, and dynamic configuration updates.
"""

import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration manager with support for YAML files and environment variables.
    
    Features:
    - Load configuration from YAML files
    - Environment variable substitution
    - Nested key access with dot notation
    - Default value support
    - Configuration validation
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, defaults to config/config.yaml
        """
        self._config: Dict[str, Any] = {}
        
        # Determine configuration file path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from the YAML file with environment variable substitution."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Configuration file not found: {self.config_path}")
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                raw_config = yaml.safe_load(file)
            
            # Substitute environment variables
            self._config = self._substitute_env_vars(raw_config)
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration values.
        
        Environment variables should be specified as ${VAR_NAME} or ${VAR_NAME:default_value}
        """
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_var_string(obj)
        else:
            return obj
    
    def _substitute_env_var_string(self, value: str) -> str:
        """Substitute environment variables in a string value."""
        import re
        
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replace_var, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'database.primary.host')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default if key is not found
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config_dict = self._config
        
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]
        
        config_dict[keys[-1]] = value
    
    def get_database_url(self, db_name: str = "primary") -> str:
        """
        Construct database URL from configuration.
        
        Args:
            db_name: Database configuration name (e.g., 'primary', 'warehouse')
            
        Returns:
            Database URL string
        """
        db_config = self.get(f"database.{db_name}")
        if not db_config:
            raise ValueError(f"Database configuration '{db_name}' not found")
        
        db_type = db_config.get("type", "postgresql")
        username = db_config.get("username")
        password = db_config.get("password")
        host = db_config.get("host", "localhost")
        port = db_config.get("port", 5432)
        database = db_config.get("database")
        
        return f"{db_type}://{username}:{password}@{host}:{port}/{database}"
    
    def get_redis_url(self) -> str:
        """
        Construct Redis URL from configuration.
        
        Returns:
            Redis URL string
        """
        redis_config = self.get("redis")
        if not redis_config:
            raise ValueError("Redis configuration not found")
        
        host = redis_config.get("host", "localhost")
        port = redis_config.get("port", 6379)
        db = redis_config.get("db", 0)
        password = redis_config.get("password")
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
    
    def validate_config(self) -> Dict[str, bool]:
        """
        Validate critical configuration sections.
        
        Returns:
            Dictionary with validation results for each section
        """
        validations = {}
        
        # Validate app configuration
        validations['app'] = all([
            self.get("app.name"),
            self.get("app.version"),
            self.get("app.environment") in ["development", "staging", "production"],
        ])
        
        # Validate database configuration
        db_config = self.get("database.primary")
        validations['database'] = all([
            db_config,
            db_config.get("host"),
            db_config.get("database"),
            db_config.get("username"),
        ])
        
        # Validate risk assessment configuration
        validations['risk_assessment'] = all([
            self.get("risk_assessment.credit"),
            self.get("risk_assessment.market"),
            self.get("risk_assessment.operational"),
            self.get("risk_assessment.liquidity"),
        ])
        
        return validations
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        logger.info("Configuration reloaded")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the entire configuration as a dictionary."""
        return self._config.copy()
    
    def __str__(self) -> str:
        """String representation of the configuration manager."""
        return f"ConfigManager(config_path={self.config_path})"
    
    def __repr__(self) -> str:
        """Detailed representation of the configuration manager."""
        return f"ConfigManager(config_path={self.config_path}, keys={list(self._config.keys())})"