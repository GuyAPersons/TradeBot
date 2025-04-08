import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import pandas as pd
from datetime import datetime

class BacktestConfig:
    """
    Configuration manager for backtesting settings.
    
    This class handles loading, validating, and accessing configuration settings
    for backtesting runs. It supports JSON and YAML formats, environment variable
    overrides, and provides default values for missing settings.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file (JSON or YAML)
        """
        self.logger = logging.getLogger(__name__)
        self.config_data = {}
        self.config_path = None
        
        # Default configuration values
        self.default_config = {
            'general': {
                'name': 'backtest',
                'description': 'Backtest configuration',
                'version': '1.0',
                'log_level': 'INFO',
                'random_seed': 42,
                'output_dir': './output'
            },
            'data': {
                'data_dir': './data',
                'start_date': '2010-01-01',
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'timeframe': 'daily',
                'symbols': [],
                'adjust_prices': True,
                'include_dividends': False,
                'cache_data': True,
                'data_source': 'csv'
            },
            'strategy': {
                'name': 'buy_and_hold',
                'parameters': {},
                'position_size': 1.0,
                'max_positions': 10,
                'use_stops': False,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.2
            },
            'execution': {
                'slippage_model': 'fixed',
                'slippage_pct': 0.001,
                'commission_model': 'percentage',
                'commission_value': 0.001,
                'delay_execution': False,
                'execution_delay': 1,
                'allow_partial_fill': True
            },
            'risk_management': {
                'max_drawdown_pct': 0.25,
                'max_risk_per_trade_pct': 0.02,
                'position_sizing_method': 'equal',
                'portfolio_heat': 1.0
            },
            'reporting': {
                'save_trades': True,
                'save_positions': True,
                'save_equity_curve': True,
                'plot_results': True,
                'detailed_metrics': True,
                'benchmark': 'SPY'
            }
        }
        
        # Load configuration if path is provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file (JSON or YAML)
            
        Returns:
            Dictionary containing the configuration data
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            self.logger.error(f"Configuration file not found: {self.config_path}")
            self.config_data = self.default_config.copy()
            return self.config_data
        
        try:
            file_extension = self.config_path.suffix.lower()
            
            if file_extension == '.json':
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
            elif file_extension in ['.yaml', '.yml']:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            else:
                self.logger.error(f"Unsupported configuration file format: {file_extension}")
                loaded_config = {}
            
            # Merge with default config to ensure all required fields exist
            self.config_data = self._merge_configs(self.default_config, loaded_config)
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
            return self.config_data
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.config_data = self.default_config.copy()
            return self.config_data
    
    def _merge_configs(self, default_config: Dict[str, Any], 
                      loaded_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge loaded configuration with default values.
        
        Args:
            default_config: Default configuration dictionary
            loaded_config: Loaded configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = default_config.copy()
        
        for section, values in loaded_config.items():
            if section in result and isinstance(values, dict):
                # If section exists in default and values is a dict, update section
                result[section].update(values)
            else:
                # Otherwise, set the entire section
                result[section] = values
        
        return result
    
    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to the configuration.
        
        Environment variables should be in the format:
        BACKTEST_SECTION_KEY=value
        
        For example:
        BACKTEST_GENERAL_LOG_LEVEL=DEBUG
        """
        prefix = "BACKTEST_"
        
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Remove prefix and split into section and key
                parts = env_var[len(prefix):].lower().split('_', 1)
                
                if len(parts) == 2:
                    section, key = parts
                    
                    # Convert value to appropriate type
                    if value.lower() in ['true', 'yes', '1']:
                        typed_value = True
                    elif value.lower() in ['false', 'no', '0']:
                        typed_value = False
                    elif value.isdigit():
                        typed_value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        typed_value = float(value)
                    else:
                        typed_value = value
                    
                    # Update configuration
                    if section in self.config_data:
                        self.config_data[section][key] = typed_value
                        self.logger.debug(f"Override from environment: {section}.{key} = {typed_value}")
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None, 
                   format: str = 'json') -> None:
        """
        Save the current configuration to a file.
        
        Args:
            output_path: Path to save the configuration file
            format: Output format ('json' or 'yaml')
        """
        if output_path is None:
            if self.config_path is None:
                output_path = Path('./backtest_config.json')
            else:
                output_path = self.config_path
        else:
            output_path = Path(output_path)
        
        try:
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save in the specified format
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(self.config_data, f, indent=4)
            elif format.lower() in ['yaml', 'yml']:
                with open(output_path, 'w') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False)
            else:
                self.logger.error(f"Unsupported output format: {format}")
                return
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if the key doesn't exist
            
        Returns:
            Configuration value or default
        """
        if section in self.config_data and key in self.config_data[section]:
            return self.config_data[section][key]
        return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self.config_data:
            self.config_data[section] = {}
        
        self.config_data[section][key] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Configuration section
            
        Returns:
            Dictionary containing the section data or empty dict if section doesn't exist
        """
        return self.config_data.get(section, {})
    
    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """
        Update an entire configuration section.
        
        Args:
            section: Configuration section
            values: Dictionary of values to update
        """
        if section not in self.config_data:
            self.config_data[section] = {}
        
        self.config_data[section].update(values)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.
        
        Returns:
            Dictionary containing all configuration data
        """
        return self.config_data.copy()
    
    def validate(self) -> List[str]:
        """
        Validate the configuration for required fields and value constraints.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check for required sections
        required_sections = ['general', 'data', 'strategy']
        for section in required_sections:
            if section not in self.config_data:
                errors.append(f"Missing required section: {section}")
        
        # Validate data section
        data = self.get_section('data')
        if data:
            # Validate dates
            try:
                start_date = pd.to_datetime(data.get('start_date'))
                end_date = pd.to_datetime(data.get('end_date'))
                
                if start_date >= end_date:
                    errors.append("start_date must be before end_date")
            except:
                errors.append("Invalid date format in start_date or end_date")
            
            # Validate symbols
            symbols = data.get('symbols', [])
            if not isinstance(symbols, list):
                errors.append("symbols must be a list")
            elif len(symbols) == 0:
                errors.append("At least one symbol must be specified")
        
        # Validate strategy section
        strategy = self.get_section('strategy')
        if strategy:
            if not strategy.get('name'):
                errors.append("Strategy name is required")
        
        # Validate execution section
        execution = self.get_section('execution')
        if execution:
            slippage_pct = execution.get('slippage_pct', 0)
            if not isinstance(slippage_pct, (int, float)) or slippage_pct < 0:
                errors.append("slippage_pct must be a non-negative number")
            
            commission_value = execution.get('commission_value', 0)
            if not isinstance(commission_value, (int, float)) or commission_value < 0:
                errors.append("commission_value must be a non-negative number")
        
        # Validate risk management section
        risk = self.get_section('risk_management')
        if risk:
            max_drawdown = risk.get('max_drawdown_pct', 0)
            if not isinstance(max_drawdown, (int, float)) or max_drawdown <= 0 or max_drawdown > 1:
                errors.append("max_drawdown_pct must be a number between 0 and 1")
            
            max_risk = risk.get('max_risk_per_trade_pct', 0)
            if not isinstance(max_risk, (int, float)) or max_risk <= 0 or max_risk > 1:
                errors.append("max_risk_per_trade_pct must be a number between 0 and 1")
        
        return errors
    
    def create_output_dirs(self) -> None:
        """
        Create output directories specified in the configuration.
        """
        # Create main output directory
        output_dir = Path(self.get('general', 'output_dir', './output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['logs', 'results', 'plots', 'trades', 'metrics']
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        self.logger.info(f"Created output directories in {output_dir}")
    
    def get_strategy_instance(self, *args, **kwargs):
        """
        Create and return a strategy instance based on the configuration.
        
        Returns:
            Strategy instance
        """
        strategy_name = self.get('strategy', 'name')
        strategy_params = self.get('strategy', 'parameters', {})
        
        try:
            # Import the strategy module dynamically
            import importlib
            
            # Try to import from custom strategies first
            try:
                module_path = f"backtesting.strategies.{strategy_name}"
                strategy_module = importlib.import_module(module_path)
            except ImportError:
                # If not found, try to import from built-in strategies
                module_path = f"backtesting.strategies.built_in.{strategy_name}"
                strategy_module = importlib.import_module(module_path)
            
            # Get the strategy class (assume it's the same name as the module but in CamelCase)
            strategy_class_name = ''.join(word.capitalize() for word in strategy_name.split('_'))
            strategy_class = getattr(strategy_module, strategy_class_name)
            
            # Create an instance with parameters from config
            return strategy_class(*args, **kwargs, **strategy_params)
            
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to load strategy '{strategy_name}': {str(e)}")
            raise ValueError(f"Strategy '{strategy_name}' not found or invalid")
