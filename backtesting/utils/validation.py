import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Union, Optional, Tuple, Callable
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Utility class for validating input data for backtesting.
    
    This class provides methods to check data integrity, format, and
    completeness before running backtests.
    """
    
    @staticmethod
    def validate_ohlcv(data: pd.DataFrame, 
                      required_columns: List[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data for backtesting.
        
        Args:
            data: DataFrame containing price data
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if data is empty
        if data is None or data.empty:
            errors.append("Data is empty")
            return False, errors
        
        # Default required columns if not specified
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check for required columns (case-insensitive)
        data_columns_lower = [col.lower() for col in data.columns]
        for col in required_columns:
            if col.lower() not in data_columns_lower:
                errors.append(f"Required column '{col}' not found in data")
        
        # Check if index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("DataFrame index is not a DatetimeIndex")
        
        # Check for duplicate indices
        if data.index.duplicated().any():
            dup_count = data.index.duplicated().sum()
            errors.append(f"DataFrame contains {dup_count} duplicate timestamps")
        
        # Check for missing values in required columns
        for col in required_columns:
            if col.lower() in data_columns_lower:
                col_idx = data_columns_lower.index(col.lower())
                actual_col = data.columns[col_idx]
                
                if data[actual_col].isnull().any():
                    null_count = data[actual_col].isnull().sum()
                    errors.append(f"Column '{actual_col}' contains {null_count} missing values")
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col.lower() in data_columns_lower:
                col_idx = data_columns_lower.index(col.lower())
                actual_col = data.columns[col_idx]
                
                if (data[actual_col] < 0).any():
                    neg_count = (data[actual_col] < 0).sum()
                    errors.append(f"Column '{actual_col}' contains {neg_count} negative values")
        
        # Check for high < low
        if 'high' in data_columns_lower and 'low' in data_columns_lower:
            high_idx = data_columns_lower.index('high')
            low_idx = data_columns_lower.index('low')
            high_col = data.columns[high_idx]
            low_col = data.columns[low_idx]
            
            if (data[high_col] < data[low_col]).any():
                invalid_count = (data[high_col] < data[low_col]).sum()
                errors.append(f"Found {invalid_count} cases where high < low")
        
        # Check for negative volume
        if 'volume' in data_columns_lower:
            vol_idx = data_columns_lower.index('volume')
            vol_col = data.columns[vol_idx]
            
            if (data[vol_col] < 0).any():
                neg_count = (data[vol_col] < 0).sum()
                errors.append(f"Column '{vol_col}' contains {neg_count} negative values")
        
        # Check for sorted index
        if not data.index.is_monotonic_increasing:
            errors.append("DataFrame index is not sorted in ascending order")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_strategy_parameters(parameters: Dict[str, Any], 
                                   parameter_specs: Dict[str, Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate strategy parameters against specifications.
        
        Args:
            parameters: Dictionary of parameter values
            parameter_specs: Dictionary of parameter specifications
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for param_name, param_spec in parameter_specs.items():
            # Check if required parameter is missing
            if param_spec.get('required', False) and param_name not in parameters:
                errors.append(f"Required parameter '{param_name}' is missing")
                continue
            
            # Skip validation if parameter is not provided
            if param_name not in parameters:
                continue
            
            param_value = parameters[param_name]
            param_type = param_spec.get('type')
            
            # Validate parameter type
            if param_type:
                if param_type == 'int':
                    if not isinstance(param_value, int) or isinstance(param_value, bool):
                        errors.append(f"Parameter '{param_name}' must be an integer")
                elif param_type == 'float':
                    if not isinstance(param_value, (int, float)) or isinstance(param_value, bool):
                        errors.append(f"Parameter '{param_name}' must be a number")
                elif param_type == 'bool':
                    if not isinstance(param_value, bool):
                        errors.append(f"Parameter '{param_name}' must be a boolean")
                elif param_type == 'str':
                    if not isinstance(param_value, str):
                        errors.append(f"Parameter '{param_name}' must be a string")
                elif param_type == 'list':
                    if not isinstance(param_value, list):
                        errors.append(f"Parameter '{param_name}' must be a list")
                elif param_type == 'dict':
                    if not isinstance(param_value, dict):
                        errors.append(f"Parameter '{param_name}' must be a dictionary")
            
            # Validate numeric range
            if isinstance(param_value, (int, float)) and not isinstance(param_value, bool):
                if 'min' in param_spec and param_value < param_spec['min']:
                    errors.append(f"Parameter '{param_name}' must be >= {param_spec['min']}")
                
                if 'max' in param_spec and param_value > param_spec['max']:
                    errors.append(f"Parameter '{param_name}' must be <= {param_spec['max']}")
            
            # Validate string pattern
            if isinstance(param_value, str) and 'pattern' in param_spec:
                pattern = param_spec['pattern']
                if not re.match(pattern, param_value):
                    errors.append(f"Parameter '{param_name}' must match pattern {pattern}")
            
            # Validate list/array length
            if isinstance(param_value, (list, tuple, np.ndarray)) and 'length' in param_spec:
                expected_length = param_spec['length']
                if len(param_value) != expected_length:
                    errors.append(f"Parameter '{param_name}' must have length {expected_length}")
            
            # Validate enum values
            if 'choices' in param_spec:
                choices = param_spec['choices']
                if param_value not in choices:
                    errors.append(f"Parameter '{param_name}' must be one of {choices}")
            
            # Validate custom condition
            if 'condition' in param_spec and callable(param_spec['condition']):
                condition_func = param_spec['condition']
                condition_msg = param_spec.get('condition_msg', f"Parameter '{param_name}' failed validation")
                
                if not condition_func(param_value):
                    errors.append(condition_msg)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_date_range(data: pd.DataFrame, start_date: str = None, 
                          end_date: str = None) -> Tuple[bool, List[str]]:
        """
        Validate that the data covers the specified date range.
        
        Args:
            data: DataFrame with DatetimeIndex
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if data is empty
        if data is None or data.empty:
            errors.append("Data is empty")
            return False, errors
        
        # Check if index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("DataFrame index is not a DatetimeIndex")
            return False, errors
        
        # Convert date strings to timestamps
        try:
            if start_date:
                start_ts = pd.Timestamp(start_date)
            else:
                start_ts = data.index[0]
                
            if end_date:
                end_ts = pd.Timestamp(end_date)
            else:
                end_ts = data.index[-1]
        except Exception as e:
            errors.append(f"Invalid date format: {str(e)}")
            return False, errors
        
        # Check if date range is valid
        if start_ts > end_ts:
            errors.append(f"Start date ({start_ts}) is after end date ({end_ts})")
        
        # Check if data covers the date range
        data_start = data.index[0]
        data_end = data.index[-1]
        
        if start_ts < data_start:
            errors.append(f"Requested start date ({start_ts}) is before first data point ({data_start})")
        
        if end_ts > data_end:
            errors.append(f"Requested end date ({end_ts}) is after last data point ({data_end})")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_indicators(data: pd.DataFrame, 
                          required_indicators: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that the data contains required technical indicators.
        
        Args:
            data: DataFrame containing price and indicator data
            required_indicators: List of required indicator column names
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if data is empty
        if data is None or data.empty:
            errors.append("Data is empty")
            return False, errors
        
        # Check for required indicators
        data_columns = set(data.columns)
        missing_indicators = [ind for ind in required_indicators if ind not in data_columns]
        
        if missing_indicators:
            errors.append(f"Missing required indicators: {missing_indicators}")
        
        # Check for NaN values in indicators
        for indicator in required_indicators:
            if indicator in data_columns and data[indicator].isnull().any():
                null_count = data[indicator].isnull().sum()
                errors.append(f"Indicator '{indicator}' contains {null_count} missing values")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_portfolio_weights(weights: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate portfolio weights.
        
        Args:
            weights: Dictionary mapping asset names to weights
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if weights are provided
        if not weights:
            errors.append("No portfolio weights provided")
            return False, errors
        
        # Check if weights are numeric
        non_numeric = [asset for asset, weight in weights.items() 
                      if not isinstance(weight, (int, float))]
        
        if non_numeric:
            errors.append(f"Non-numeric weights for assets: {non_numeric}")
        
        # Check if weights are non-negative
        negative = [asset for asset, weight in weights.items() 
                   if isinstance(weight, (int, float)) and weight < 0]
        
        if negative:
            errors.append(f"Negative weights for assets: {negative}")
        
        # Check if weights sum to approximately 1.0
        weight_sum = sum(weight for weight in weights.values() 
                        if isinstance(weight, (int, float)))
        
        if not np.isclose(weight_sum, 1.0, rtol=1e-5):
            errors.append(f"Weights sum to {weight_sum}, expected 1.0")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_backtest_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate backtest configuration.
        
        Args:
            config: Dictionary containing backtest configuration
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for required sections
        required_sections = ['general', 'data', 'strategy']
        missing_sections = [section for section in required_sections 
                           if section not in config]
        
        if missing_sections:
            errors.append(f"Missing required configuration sections: {missing_sections}")
        
        # Validate data section
        if 'data' in config:
            data_config = config['data']
            
            # Check for required data fields
            required_data_fields = ['symbols', 'start_date', 'end_date']
            missing_fields = [field for field in required_data_fields 
                             if field not in data_config]
            
            if missing_fields:
                errors.append(f"Missing required data fields: {missing_fields}")
            
            # Validate date format
            for date_field in ['start_date', 'end_date']:
                if date_field in data_config:
                    try:
                        pd.Timestamp(data_config[date_field])
                    except:
                        errors.append(f"Invalid date format for {date_field}: {data_config[date_field]}")
            
            # Validate symbols
            if 'symbols' in data_config:
                symbols = data_config['symbols']
                if not isinstance(symbols, list):
                    errors.append("Symbols must be a list")
                elif not symbols:
                    errors.append("At least one symbol must be specified")
        
        # Validate strategy section
        if 'strategy' in config:
            strategy_config = config['strategy']
            
            # Check for required strategy fields
            if 'name' not in strategy_config:
                errors.append("Strategy name is required")
            
            # Validate parameters if present
            if 'parameters' in strategy_config:
                params = strategy_config['parameters']
                if not isinstance(params, dict):
                    errors.append("Strategy parameters must be a dictionary")
        
        # Validate execution section if present
        if 'execution' in config:
            execution_config = config['execution']
            
            # Validate slippage
            if 'slippage_pct' in execution_config:
                slippage = execution_config['slippage_pct']
                if not isinstance(slippage, (int, float)) or slippage < 0:
                    errors.append("Slippage percentage must be a non-negative number")
            
            # Validate commission
            if 'commission_value' in execution_config:
                commission = execution_config['commission_value']
                if not isinstance(commission, (int, float)) or commission < 0:
                    errors.append("Commission value must be a non-negative number")
        
        # Validate risk management section if present
        if 'risk_management' in config:
            risk_config = config['risk_management']
            
            # Validate max drawdown
            if 'max_drawdown_pct' in risk_config:
                max_dd = risk_config['max_drawdown_pct']
                if not isinstance(max_dd, (int, float)) or max_dd <= 0 or max_dd > 1:
                    errors.append("Max drawdown percentage must be a number between 0 and 1")
            
            # Validate max risk per trade
            if 'max_risk_per_trade_pct' in risk_config:
                max_risk = risk_config['max_risk_per_trade_pct']
                if not isinstance(max_risk, (int, float)) or max_risk <= 0 or max_risk > 1:
                    errors.append("Max risk per trade percentage must be a number between 0 and 1")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform a comprehensive data quality check.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            Dictionary with data quality metrics
        """
        if data is None or data.empty:
            return {'valid': False, 'error': 'Data is empty'}
        
        # Initialize results dictionary
        results = {
            'valid': True,
            'row_count': len(data),
            'column_count': len(data.columns),
            'date_range': {
                'start': data.index[0].strftime('%Y-%m-%d'),
                'end': data.index[-1].strftime('%Y-%m-%d'),
                'days': (data.index[-1] - data.index[0]).days
            },
            'missing_values': {},
            'outliers': {},
            'statistics': {},
            'issues': []
        }
        
        # Check for missing values
        missing_values = data.isnull().sum()
        results['missing_values'] = missing_values.to_dict()
        
        if missing_values.sum() > 0:
            results['issues'].append(f"Found {missing_values.sum()} missing values")
        
        # Check for duplicate indices
        duplicate_count = data.index.duplicated().sum()
        if duplicate_count > 0:
            results['issues'].append(f"Found {duplicate_count} duplicate timestamps")
            results['valid'] = False
        
        # Check for sorted index
        if not data.index.is_monotonic_increasing:
            results['issues'].append("DataFrame index is not sorted in ascending order")
            results['valid'] = False
        
        # Calculate basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Basic statistics
            stats = {
                'min': data[col].min(),
                'max': data[col].max(),
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std()
            }
            results['statistics'][col] = stats
            
            # Check for outliers (values more than 3 std devs from mean)
            mean, std = data[col].mean(), data[col].std()
            outliers = data[(data[col] < mean - 3*std) | (data[col] > mean + 3*std)][col]
            
            if not outliers.empty:
                results['outliers'][col] = len(outliers)
                results['issues'].append(f"Found {len(outliers)} outliers in column '{col}'")
        
        # Check for price anomalies if OHLC columns exist
        price_cols = [col for col in data.columns if col.lower() in ['open', 'high', 'low', 'close']]
        
        if len(price_cols) >= 4:
            # Map standard names to actual column names
            col_map = {}
            for std_name in ['open', 'high', 'low', 'close']:
                for col in price_cols:
                    if col.lower() == std_name:
                        col_map[std_name] = col
                        break
            
            # Check for high < low
            if 'high' in col_map and 'low' in col_map:
                invalid_count = (data[col_map['high']] < data[col_map['low']]).sum()
                if invalid_count > 0:
                    results['issues'].append(f"Found {invalid_count} cases where high < low")
                    results['valid'] = False
            
            # Check for negative prices
            for price_type, col in col_map.items():
                neg_count = (data[col] < 0).sum()
                if neg_count > 0:
                    results['issues'].append(f"Found {neg_count} negative values in {price_type} prices")
                    results['valid'] = False
            
            # Check for extreme price changes (daily returns > 50%)
            if 'close' in col_map:
                daily_returns = data[col_map['close']].pct_change()
                extreme_returns = daily_returns[abs(daily_returns) > 0.5]
                
                if not extreme_returns.empty:
                    results['extreme_returns'] = len(extreme_returns)
                    results['issues'].append(f"Found {len(extreme_returns)} days with price changes > 50%")
        
        # Check for volume anomalies if volume column exists
        volume_cols = [col for col in data.columns if col.lower() == 'volume']
        
        if volume_cols:
            vol_col = volume_cols[0]
            
            # Check for negative volume
            neg_count = (data[vol_col] < 0).sum()
            if neg_count > 0:
                results['issues'].append(f"Found {neg_count} negative values in volume")
                results['valid'] = False
            
            # Check for zero volume
            zero_count = (data[vol_col] == 0).sum()
            if zero_count > 0:
                results['zero_volume'] = zero_count
                results['issues'].append(f"Found {zero_count} days with zero volume")
        
        # Check for gaps in time series
        if isinstance(data.index, pd.DatetimeIndex):
            # For daily data
            if data.index.freq is None or 'D' in str(data.index.freq):
                # Create a complete date range
                full_range = pd.date_range(start=data.index[0], end=data.index[-1], freq='B')
                
                # Find missing dates (business days only)
                missing_dates = full_range.difference(data.index)
                
                if len(missing_dates) > 0:
                    results['missing_dates'] = len(missing_dates)
                    results['issues'].append(f"Found {len(missing_dates)} missing business days in the date range")
        
        return results
    
    @staticmethod
    def validate_trade_data(trades: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate trade data for consistency and correctness.
        
        Args:
            trades: DataFrame containing trade data
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if data is empty
        if trades is None or trades.empty:
            errors.append("Trade data is empty")
            return False, errors
        
        # Check for required columns
        required_columns = ['entry_date', 'exit_date', 'symbol', 'direction', 
                           'entry_price', 'exit_price', 'quantity', 'pnl']
        
        missing_columns = [col for col in required_columns if col not in trades.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Skip further validation if required columns are missing
        if missing_columns:
            return False, errors
        
        # Check for invalid dates
        try:
            entry_dates = pd.to_datetime(trades['entry_date'])
            exit_dates = pd.to_datetime(trades['exit_date'])
            
            # Check for entry date after exit date
            invalid_dates = (entry_dates > exit_dates).sum()
            if invalid_dates > 0:
                errors.append(f"Found {invalid_dates} trades with entry date after exit date")
            
        except Exception as e:
            errors.append(f"Invalid date format in trade data: {str(e)}")
        
        # Check for invalid prices
        if (trades['entry_price'] <= 0).any():
            invalid_count = (trades['entry_price'] <= 0).sum()
            errors.append(f"Found {invalid_count} trades with non-positive entry price")
        
        if (trades['exit_price'] <= 0).any():
            invalid_count = (trades['exit_price'] <= 0).sum()
            errors.append(f"Found {invalid_count} trades with non-positive exit price")
        
        # Check for invalid quantities
        if (trades['quantity'] <= 0).any():
            invalid_count = (trades['quantity'] <= 0).sum()
            errors.append(f"Found {invalid_count} trades with non-positive quantity")
        
        # Check for invalid directions
        valid_directions = ['long', 'short']
        invalid_directions = trades[~trades['direction'].isin(valid_directions)]
        
        if not invalid_directions.empty:
            errors.append(f"Found {len(invalid_directions)} trades with invalid direction")
        
        # Validate PnL calculation
        # For long trades: PnL = (exit_price - entry_price) * quantity
        # For short trades: PnL = (entry_price - exit_price) * quantity
        
        long_trades = trades[trades['direction'] == 'long']
        if not long_trades.empty:
            expected_pnl = (long_trades['exit_price'] - long_trades['entry_price']) * long_trades['quantity']
            pnl_diff = (long_trades['pnl'] - expected_pnl).abs()
            
            # Allow for small floating-point differences
            invalid_pnl = (pnl_diff > 0.01).sum()
            if invalid_pnl > 0:
                errors.append(f"Found {invalid_pnl} long trades with incorrect PnL calculation")
        
        short_trades = trades[trades['direction'] == 'short']
        if not short_trades.empty:
            expected_pnl = (short_trades['entry_price'] - short_trades['exit_price']) * short_trades['quantity']
            pnl_diff = (short_trades['pnl'] - expected_pnl).abs()
            
            # Allow for small floating-point differences
            invalid_pnl = (pnl_diff > 0.01).sum()
            if invalid_pnl > 0:
                errors.append(f"Found {invalid_pnl} short trades with incorrect PnL calculation")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_optimization_parameters(param_grid: Dict[str, List[Any]]) -> Tuple[bool, List[str]]:
        """
        Validate parameter grid for optimization.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if param_grid is empty
        if not param_grid:
            errors.append("Parameter grid is empty")
            return False, errors
        
        # Check if all parameters have valid values
        for param_name, param_values in param_grid.items():
            if not isinstance(param_values, (list, tuple, np.ndarray)):
                errors.append(f"Values for parameter '{param_name}' must be a list")
            elif len(param_values) == 0:
                errors.append(f"No values provided for parameter '{param_name}'")
        
        # Calculate total number of combinations
        total_combinations = 1
        for param_values in param_grid.values():
            if isinstance(param_values, (list, tuple, np.ndarray)):
                total_combinations *= len(param_values)
        
        # Warn if too many combinations
        if total_combinations > 10000:
            errors.append(f"Parameter grid has {total_combinations} combinations, which may be excessive")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_market_data_consistency(data_dict: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """
        Validate consistency across multiple market data DataFrames.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if data_dict is empty
        if not data_dict:
            errors.append("No market data provided")
            return False, errors
        
        # Check if all DataFrames have DatetimeIndex
        non_datetime_symbols = [symbol for symbol, df in data_dict.items() 
                               if not isinstance(df.index, pd.DatetimeIndex)]
        
        if non_datetime_symbols:
            errors.append(f"DataFrames for symbols {non_datetime_symbols} do not have DatetimeIndex")
            return False, errors
        
        # Get the first DataFrame as reference
        reference_symbol = next(iter(data_dict.keys()))
        reference_df = data_dict[reference_symbol]
        
        # Check for consistent date ranges
        for symbol, df in data_dict.items():
            if symbol == reference_symbol:
                continue
            
            # Check start date
            if df.index[0] != reference_df.index[0]:
                errors.append(f"Start date for {symbol} ({df.index[0]}) differs from {reference_symbol} ({reference_df.index[0]})")
            
            # Check end date
            if df.index[-1] != reference_df.index[-1]:
                errors.append(f"End date for {symbol} ({df.index[-1]}) differs from {reference_symbol} ({reference_df.index[-1]})")
            
            # Check for missing dates
            if len(df.index) != len(reference_df.index):
                errors.append(f"Number of data points for {symbol} ({len(df.index)}) differs from {reference_symbol} ({len(reference_df.index)})")
            
            # Check for exact index match
            if not df.index.equals(reference_df.index):
                errors.append(f"Date index for {symbol} does not exactly match {reference_symbol}")
        
        # Check for consistent columns
        reference_columns = set(reference_df.columns)
        
        for symbol, df in data_dict.items():
            if symbol == reference_symbol:
                continue
            
            df_columns = set(df.columns)
            
            if df_columns != reference_columns:
                missing = reference_columns - df_columns
                extra = df_columns - reference_columns
                
                if missing:
                    errors.append(f"DataFrame for {symbol} is missing columns: {missing}")
                
                if extra:
                    errors.append(f"DataFrame for {symbol} has extra columns: {extra}")
        
        return len(errors) == 0, errors