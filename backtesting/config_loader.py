"""
Configuration loading and validation for the backtesting framework.
"""
from typing import Dict, Any, Optional, List, Union
import os
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path

from .utils.validation import Validator


class ConfigLoader:
    """
    Loads and validates configuration files for the backtesting framework.
    
    This class handles loading configuration from JSON or YAML files,
    validating the configuration structure, and providing access to
    configuration parameters.
    """
    
    def __init__(self):
        """Initialize the config loader."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = {}
        self.config_path = None
    
    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file (JSON or YAML)
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration file format is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config_path = config_path
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
            
            self.logger.info(f"Loaded configuration from {config_path}")
            
            # Validate the configuration
            self._validate_config()
            
            return self.config
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            raise ValueError(f"Invalid configuration file format: {e}")
    
    def _validate_config(self):
        """
        Validate the configuration structure.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        # Check for required sections
        required_sections = ['general']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate general section
        general = self.config['general']
        if 'name' not in general:
            self.logger.warning("No backtest name specified in configuration")
            general['name'] = f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate date formats if present
        for date_field in ['start_date', 'end_date']:
            if date_field in general:
                try:
                    datetime.fromisoformat(general[date_field])
                except ValueError:
                    raise ValueError(f"Invalid date format for {date_field}: {general[date_field]}")
        
        # Validate data providers if present
        if 'data_providers' in self.config:
            for i, provider in enumerate(self.config['data_providers']):
                if 'type' not in provider:
                    raise ValueError(f"Data provider at index {i} missing 'type' field")
                if 'name' not in provider:
                    raise ValueError(f"Data provider at index {i} missing 'name' field")
        
        # Validate strategies if present
        if 'strategies' in self.config:
            for i, strategy in enumerate(self.config['strategies']):
                if 'name' not in strategy:
                    raise ValueError(f"Strategy at index {i} missing 'name' field")
                if 'class' not in strategy:
                    raise ValueError(f"Strategy at index {i} missing 'class' field")
        
        self.logger.info("Configuration validation successful")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        if not self.config:
            return default
        
        # Handle nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        
        return self.config.get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary containing the section or empty dict if not found
        """
        return self.config.get(section, {})
    
    def save(self, output_path: Optional[str] = None) -> str:
        """
        Save the current configuration to a file.
        
        Args:
            output_path: Path to save the configuration (defaults to original path)
            
        Returns:
            Path where the configuration was saved
            
        Raises:
            ValueError: If no output path is specified and no original path exists
        """
        save_path = output_path or self.config_path
        if not save_path:
            raise ValueError("No output path specified and no original path exists")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        file_ext = os.path.splitext(save_path)[1].lower()
        
        try:
            if file_ext == '.json':
                with open(save_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
            elif file_ext in ['.yaml', '.yml']:
                with open(save_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
            
            self.logger.info(f"Saved configuration to {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def update(self, updates: Dict[str, Any]):
        """
        Update the configuration with new values.
        
        Args:
            updates: Dictionary containing updates to apply
            
        Returns:
            Updated configuration
        """
        def update_nested(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_nested(target[key], value)
                else:
                    target[key] = value
        
        update_nested(self.config, updates)
        self.logger.info("Updated configuration")
        return self.config
    
    @staticmethod
    def create_default_config(output_path: str) -> Dict[str, Any]:
        """
        Create a default configuration file.
        
        Args:
            output_path: Path to save the default configuration
            
        Returns:
            Dictionary containing the default configuration
        """
        default_config = {
            "general": {
                "name": f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "initial_capital": 100000.0,
                "start_date": datetime.now().replace(year=datetime.now().year-1).isoformat(),
                "end_date": datetime.now().isoformat(),
                "output_dir": "results"
            },
            "data_providers": [
                {
                    "name": "csv_provider",
                    "type": "csv",
                    "params": {
                        "directory": "data/csv",
                        "date_format": "%Y-%m-%d"
                    }
                }
            ],
            "data_preprocessing": {
                "fillna_method": "ffill",
                "resample": "1D"
            },
            "strategies": [
                {
                    "name": "moving_average_crossover",
                    "class": "backtesting.strategies.trend_following.MovingAverageCrossover",
                    "params": {
                        "symbols": ["SPY"],
                        "short_window": 50,
                        "long_window": 200
                    }
                }
            ],
            "execution": {
                "slippage_model": {
                    "type": "percentage",
                    "params": {
                        "percentage": 0.001
                    }
                },
                "transaction_cost_model": {
                    "type": "fixed",
                    "params": {
                        "cost": 5.0
                    }
                }
            },
            "risk_management": {
                "position_size_model": {
                    "type": "percentage",
                    "params": {
                        "percentage": 0.02
                    }
                },
                "max_drawdown": 0.25,
                "max_position_size": 0.1
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the default configuration
        file_ext = os.path.splitext(output_path)[1].lower()
        
        if file_ext == '.json':
            with open(output_path, 'w') as f:
                json.dump(default_config, f, indent=4)
        elif file_ext in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        logging.getLogger(__name__).info(f"Created default configuration at {output_path}")
        
        return default_config