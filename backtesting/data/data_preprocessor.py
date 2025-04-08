import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import talib

class DataPreprocessor:
    """
    DataPreprocessor handles data preparation and feature engineering
    for backtesting.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.logger = logging.getLogger(__name__)
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Add technical indicators to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: Dictionary mapping indicator names to parameters
                Example: {'sma': {'window': 20}, 'rsi': {'window': 14}}
                
        Returns:
            DataFrame with added indicators
        """
        result_df = df.copy()
        
        for indicator_name, params in indicators.items():
            try:
                if indicator_name == 'sma':
                    window = params.get('window', 20)
                    result_df[f'sma_{window}'] = talib.SMA(result_df['close'].values, timeperiod=window)
                
                elif indicator_name == 'ema':
                    window = params.get('window', 20)
                    result_df[f'ema_{window}'] = talib.EMA(result_df['close'].values, timeperiod=window)
                
                elif indicator_name == 'rsi':
                    window = params.get('window', 14)
                    result_df[f'rsi_{window}'] = talib.RSI(result_df['close'].values, timeperiod=window)
                
                elif indicator_name == 'macd':
                    fast_period = params.get('fast_period', 12)
                    slow_period = params.get('slow_period', 26)
                    signal_period = params.get('signal_period', 9)
                    macd, macd_signal, macd_hist = talib.MACD(
                        result_df['close'].values,
                        fastperiod=fast_period,
                        slowperiod=slow_period,
                        signalperiod=signal_period
                    )
                    result_df[f'macd_{fast_period}_{slow_period}'] = macd
                    result_df[f'macd_signal_{fast_period}_{slow_period}_{signal_period}'] = macd_signal
                    result_df[f'macd_hist_{fast_period}_{slow_period}_{signal_period}'] = macd_hist
                
                elif indicator_name == 'bollinger_bands':
                    window = params.get('window', 20)
                    num_std = params.get('num_std', 2)
                    upper, middle, lower = talib.BBANDS(
                        result_df['close'].values,
                        timeperiod=window,
                        nbdevup=num_std,
                        nbdevdn=num_std,
                        matype=0
                    )
                    result_df[f'bb_upper_{window}'] = upper
                    result_df[f'bb_middle_{window}'] = middle
                    result_df[f'bb_lower_{window}'] = lower
                
                elif indicator_name == 'atr':
                    window = params.get('window', 14)
                    result_df[f'atr_{window}'] = talib.ATR(
                        result_df['high'].values,
                        result_df['low'].values,
                        result_df['close'].values,
                        timeperiod=window
                    )
                
                elif indicator_name == 'stochastic':
                    k_period = params.get('k_period', 14)
                    d_period = params.get('d_period', 3)
                    slowing = params.get('slowing', 3)
                    k, d = talib.STOCH(
                        result_df['high'].values,
                        result_df['low'].values,
                        result_df['close'].values,
                        fastk_period=k_period,
                        slowk_period=slowing,
                        slowk_matype=0,
                        slowd_period=d_period,
                        slowd_matype=0
                    )
                    result_df[f'stoch_k_{k_period}'] = k
                    result_df[f'stoch_d_{k_period}_{d_period}'] = d
                
                elif indicator_name == 'adx':
                    window = params.get('window', 14)
                    result_df[f'adx_{window}'] = talib.ADX(
                        result_df['high'].values,
                        result_df['low'].values,
                        result_df['close'].values,
                        timeperiod=window
                    )
                
                elif indicator_name == 'obv':
                    result_df['obv'] = talib.OBV(
                        result_df['close'].values,
                        result_df['volume'].values
                    )
                
                else:
                    self.logger.warning(f"Unknown indicator: {indicator_name}")
            
            except Exception as e:
                self.logger.error(f"Error calculating {indicator_name}: {str(e)}")
        
        return result_df
    
    def add_custom_features(
        self,
        df: pd.DataFrame,
        features: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Add custom features to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            features: Dictionary mapping feature names to parameters
                
        Returns:
            DataFrame with added features
        """
        result_df = df.copy()
        
        for feature_name, params in features.items():
            try:
                if feature_name == 'returns':
                    period = params.get('period', 1)
                    result_df[f'return_{period}'] = result_df['close'].pct_change(period)
                
                elif feature_name == 'log_returns':
                    period = params.get('period', 1)
                    result_df[f'log_return_{period}'] = np.log(result_df['close'] / result_df['close'].shift(period))
                
                elif feature_name == 'volatility':
                    window = params.get('window', 20)
                    result_df[f'volatility_{window}'] = result_df['close'].pct_change().rolling(window=window).std()
                
                elif feature_name == 'z_score':
                    window = params.get('window', 20)
                    mean = result_df['close'].rolling(window=window).mean()
                    std = result_df['close'].rolling(window=window).std()
                    result_df[f'z_score_{window}'] = (result_df['close'] - mean) / std
                
                elif feature_name == 'price_channels':
                    window = params.get('window', 20)
                    result_df[f'highest_{window}'] = result_df['high'].rolling(window=window).max()
                    result_df[f'lowest_{window}'] = result_df['low'].rolling(window=window).min()
                
                elif feature_name == 'price_momentum':
                    window = params.get('window', 20)
                    result_df[f'momentum_{window}'] = result_df['close'] / result_df['close'].shift(window) - 1
                
                elif feature_name == 'volume_momentum':
                    window = params.get('window', 20)
                    result_df[f'volume_momentum_{window}'] = result_df['volume'] / result_df['volume'].rolling(window=window).mean()
                
                elif feature_name == 'day_of_week':
                    result_df['day_of_week'] = result_df.index.dayofweek
                
                elif feature_name == 'hour_of_day':
                    result_df['hour_of_day'] = result_df.index.hour
                
                elif feature_name == 'is_month_start':
                    result_df['is_month_start'] = result_df.index.is_month_start.astype(int)
                
                elif feature_name == 'is_month_end':
                    result_df['is_month_end'] = result_df.index.is_month_end.astype(int)
                
                elif feature_name == 'distance_from_mean':
                    window = params.get('window', 20)
                    mean = result_df['close'].rolling(window=window).mean()
                    result_df[f'distance_from_mean_{window}'] = (result_df['close'] - mean) / mean
                
                elif feature_name == 'candle_pattern':
                    # Add various candlestick patterns
                    result_df['doji'] = talib.CDLDOJI(
                        result_df['open'].values,
                        result_df['high'].values,
                        result_df['low'].values,
                        result_df['close'].values
                    )
                    result_df['engulfing'] = talib.CDLENGULFING(
                        result_df['open'].values,
                        result_df['high'].values,
                        result_df['low'].values,
                        result_df['close'].values
                    )
                    result_df['hammer'] = talib.CDLHAMMER(
                        result_df['open'].values,
                        result_df['high'].values,
                        result_df['low'].values,
                        result_df['close'].values
                    )
                
                else:
                    self.logger.warning(f"Unknown feature: {feature_name}")
            
            except Exception as e:
                self.logger.error(f"Error calculating {feature_name}: {str(e)}")
        
        return result_df
    
    def clean_data(self, df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Args:
            df: DataFrame to clean
            method: Method for handling missing values ('ffill', 'bfill', 'drop', 'interpolate')
                
        Returns:
            Cleaned DataFrame
        """
        result_df = df.copy()
        
        # Handle missing values
        if method == 'ffill':
            result_df = result_df.fillna(method='ffill')
        elif method == 'bfill':
            result_df = result_df.fillna(method='bfill')
        elif method == 'drop':
            result_df = result_df.dropna()
        elif method == 'interpolate':
            result_df = result_df.interpolate(method='linear')
        
        # Handle infinite values
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.fillna(method='ffill')
        
        return result_df
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Normalize data in specified columns.
        
        Args:
            df: DataFrame to normalize
            columns: List of columns to normalize
            method: Normalization method ('zscore', 'minmax', 'robust')
                
        Returns:
            DataFrame with normalized columns
        """
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns:
                if method == 'zscore':
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    if std != 0:
                        result_df[f'{col}_norm'] = (result_df[col] - mean) / std
                
                elif method == 'minmax':
                    min_val = result_df[col].min()
                    max_val = result_df[col].max()
                    if max_val > min_val:
                        result_df[f'{col}_norm'] = (result_df[col] - min_val) / (max_val - min_val)
                
                elif method == 'robust':
                    median = result_df[col].median()
                    q1 = result_df[col].quantile(0.25)
                    q3 = result_df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr != 0:
                        result_df[f'{col}_norm'] = (result_df[col] - median) / iqr
            else:
                self.logger.warning(f"Column {col} not found in DataFrame")
        
        return result_df
    
    def resample_data(
        self,
        df: pd.DataFrame,
        timeframe: str,
        agg_dict: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Resample data to a different timeframe.
        
        Args:
            df: DataFrame to resample
            timeframe: Target timeframe (e.g., '1H', '1D', '1W')
            agg_dict: Dictionary mapping columns to aggregation functions
                
        Returns:
            Resampled DataFrame
        """
        # Default aggregation dictionary for OHLCV data
        if agg_dict is None:
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        
        # Filter agg_dict to include only columns present in the DataFrame
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        # Resample the data
        resampled_df = df.resample(timeframe).agg(agg_dict)
        
        return resampled_df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the DataFrame.
        
        Args:
            df: DataFrame with datetime index
                
        Returns:
            DataFrame with added time features
        """
        result_df = df.copy()
        
        # Extract time components
        result_df['hour'] = result_df.index.hour
        result_df['day_of_week'] = result_df.index.dayofweek
        result_df['day_of_month'] = result_df.index.day
        result_df['week_of_year'] = result_df.index.isocalendar().week
        result_df['month'] = result_df.index.month
        result_df['quarter'] = result_df.index.quarter
        result_df['year'] = result_df.index.year
        
        # Binary indicators
        result_df['is_month_start'] = result_df.index.is_month_start.astype(int)
        result_df['is_month_end'] = result_df.index.is_month_end.astype(int)
        result_df['is_quarter_start'] = result_df.index.is_quarter_start.astype(int)
        result_df['is_quarter_end'] = result_df.index.is_quarter_end.astype(int)
        result_df['is_year_start'] = result_df.index.is_year_start.astype(int)
        result_df['is_year_end'] = result_df.index.is_year_end.astype(int)
        
        # Is weekend
        result_df['is_weekend'] = (result_df['day_of_week'] >= 5).astype(int)
        
        return result_df
    
    def add_lagged_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Add lagged values of specified columns.
        
        Args:
            df: DataFrame to process
            columns: List of columns to create lags for
            lags: List of lag periods
                
        Returns:
            DataFrame with added lagged features
        """
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns:
                for lag in lags:
                    result_df[f'{col}_lag_{lag}'] = result_df[col].shift(lag)
            else:
                self.logger.warning(f"Column {col} not found in DataFrame")
        
        return result_df
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int],
        functions: Dict[str, callable]
    ) -> pd.DataFrame:
        """
        Add rolling window calculations for specified columns.
        
        Args:
            df: DataFrame to process
            columns: List of columns to create rolling features for
            windows: List of window sizes
            functions: Dictionary mapping function names to functions
                
        Returns:
            DataFrame with added rolling features
        """
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns:
                for window in windows:
                    for func_name, func in functions.items():
                        result_df[f'{col}_{func_name}_{window}'] = result_df[col].rolling(window=window).apply(func)
            else:
                self.logger.warning(f"Column {col} not found in DataFrame")
        
        return result_df
    
    def add_crossover_signals(
        self,
        df: pd.DataFrame,
        fast_col: str,
        slow_col: str,
        prefix: str = 'crossover'
    ) -> pd.DataFrame:
        """
        Add crossover signals between two indicators.
        
        Args:
            df: DataFrame to process
            fast_col: Name of the faster indicator column
            slow_col: Name of the slower indicator column
            prefix: Prefix for the signal column name
                
        Returns:
            DataFrame with added crossover signals
        """
        result_df = df.copy()
        
        if fast_col in result_df.columns and slow_col in result_df.columns:
            # Create crossover signals
            # 1 for bullish crossover (fast crosses above slow)
            # -1 for bearish crossover (fast crosses below slow)
            # 0 for no crossover
            fast_above_slow = result_df[fast_col] > result_df[slow_col]
            fast_above_slow_prev = fast_above_slow.shift(1)
            
            result_df[f'{prefix}_signal'] = 0
            result_df.loc[(fast_above_slow) & (~fast_above_slow_prev), f'{prefix}_signal'] = 1  # Bullish crossover
            result_df.loc[(~fast_above_slow) & (fast_above_slow_prev), f'{prefix}_signal'] = -1  # Bearish crossover
        else:
            missing_cols = []
            if fast_col not in result_df.columns:
                missing_cols.append(fast_col)
            if slow_col not in result_df.columns:
                missing_cols.append(slow_col)
            self.logger.warning(f"Columns not found in DataFrame: {missing_cols}")
        
        return result_df
    
    def add_volatility_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        windows: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Add volatility-related features.
        
        Args:
            df: DataFrame to process
            price_col: Column name for price data
            windows: List of window sizes for volatility calculation
                
        Returns:
            DataFrame with added volatility features
        """
        result_df = df.copy()
        
        if price_col in result_df.columns:
            # Calculate returns
            returns = result_df[price_col].pct_change()
            
            # Calculate rolling volatility for different windows
            for window in windows:
                result_df[f'volatility_{window}'] = returns.rolling(window=window).std()
                
                # Normalized volatility (current volatility / average volatility)
                result_df[f'volatility_ratio_{window}'] = (
                    result_df[f'volatility_{window}'] / 
                    result_df[f'volatility_{window}'].rolling(window=window*2).mean()
                )
        else:
            self.logger.warning(f"Column {price_col} not found in DataFrame")
        
        return result_df