import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import os

class DataHandler:
    """
    DataHandler is responsible for loading, preprocessing, and providing
    historical market data for backtesting.
    """
    
    def __init__(self, data_provider=None, cache_path: Optional[str] = None):
        """
        Initialize the DataHandler.
        
        Args:
            data_provider: Data provider instance for fetching data
            cache_path: Path to cache directory for storing data
        """
        self.data_provider = data_provider
        self.cache_path = cache_path
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_idx: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory if it doesn't exist
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)
    
    def load_data(
        self,
        instruments: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str,
        include_columns: Optional[List[str]] = None,
        refresh_cache: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple instruments.
        
        Args:
            instruments: List of instrument identifiers
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Timeframe for the data (e.g., '1m', '1h', '1d')
            include_columns: Specific columns to include (None for all)
            refresh_cache: Whether to refresh cached data
            
        Returns:
            Dictionary mapping instrument identifiers to DataFrames
        """
        for instrument in instruments:
            self.load_instrument_data(
                instrument, start_date, end_date, timeframe, 
                include_columns, refresh_cache
            )
        
        return self.data
    
    def load_instrument_data(
        self,
        instrument: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str,
        include_columns: Optional[List[str]] = None,
        refresh_cache: bool = False
    ) -> pd.DataFrame:
        """
        Load historical data for a single instrument.
        
        Args:
            instrument: Instrument identifier
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Timeframe for the data (e.g., '1m', '1h', '1d')
            include_columns: Specific columns to include (None for all)
            refresh_cache: Whether to refresh cached data
            
        Returns:
            DataFrame with historical data
        """
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check if data is already in memory
        if instrument in self.data and not refresh_cache:
            df = self.data[instrument]
            # Check if we have all the required data
            if df.index[0] <= start_date and df.index[-1] >= end_date:
                # Filter to the requested date range
                filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
                if include_columns:
                    filtered_df = filtered_df[include_columns]
                return filtered_df
        
        # Try to load from cache
        cache_file = self._get_cache_filename(instrument, timeframe)
        if cache_file and os.path.exists(cache_file) and not refresh_cache:
            try:
                df = pd.read_parquet(cache_file)
                # Check if cached data covers the requested range
                if df.index[0] <= start_date and df.index[-1] >= end_date:
                    # Filter to the requested date range
                    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
                    if include_columns:
                        filtered_df = filtered_df[include_columns]
                    self.data[instrument] = filtered_df
                    self.current_idx[instrument] = 0
                    return filtered_df
            except Exception as e:
                self.logger.warning(f"Error loading cached data for {instrument}: {str(e)}")
        
        # Fetch data from provider if available
        if self.data_provider:
            try:
                df = self.data_provider.get_historical_data(
                    instrument, start_date, end_date, timeframe
                )
                
                # Ensure the index is datetime and sorted
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'timestamp' in df.columns:
                        df.set_index('timestamp', inplace=True)
                    else:
                        raise ValueError(f"DataFrame for {instrument} has no datetime index or timestamp column")
                
                df = df.sort_index()
                
                # Cache the data if cache path is provided
                if cache_file:
                    df.to_parquet(cache_file)
                
                # Filter columns if specified
                if include_columns:
                    df = df[include_columns]
                
                self.data[instrument] = df
                self.current_idx[instrument] = 0
                return df
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {instrument}: {str(e)}")
                raise
        else:
            raise ValueError("No data provider available and no cached data found")
    
    def _get_cache_filename(self, instrument: str, timeframe: str) -> Optional[str]:
        """Get the cache filename for an instrument and timeframe."""
        if not self.cache_path:
            return None
        
        # Create a safe filename
        safe_instrument = instrument.replace('/', '_').replace('\\', '_')
        filename = f"{safe_instrument}_{timeframe}.parquet"
        return os.path.join(self.cache_path, filename)
    
    def get_latest_bar(self, instrument: str) -> Optional[pd.Series]:
        """
        Get the current bar for an instrument.
        
        Args:
            instrument: Instrument identifier
            
        Returns:
            Series with bar data or None if no more data
        """
        if instrument not in self.data or instrument not in self.current_idx:
            return None
        
        df = self.data[instrument]
        idx = self.current_idx[instrument]
        
        if idx >= len(df):
            return None
        
        return df.iloc[idx]
    
    def get_latest_bars(self, instrument: str, n: int = 1) -> Optional[pd.DataFrame]:
        """
        Get the last n bars for an instrument.
        
        Args:
            instrument: Instrument identifier
            n: Number of bars to return
            
        Returns:
            DataFrame with bar data or None if no more data
        """
        if instrument not in self.data or instrument not in self.current_idx:
            return None
        
        df = self.data[instrument]
        idx = self.current_idx[instrument]
        
        if idx >= len(df):
            return None
        
        # Get up to n bars, but don't go before the start of the data
        start_idx = max(0, idx - n + 1)
        return df.iloc[start_idx:idx+1]
    
    def update_bars(self) -> Dict[str, pd.Series]:
        """
        Update the current bar for all instruments.
        
        Returns:
            Dictionary mapping instrument identifiers to current bars
        """
        current_bars = {}
        
        for instrument in self.data.keys():
            if self.current_idx[instrument] < len(self.data[instrument]):
                current_bars[instrument] = self.data[instrument].iloc[self.current_idx[instrument]]
                self.current_idx[instrument] += 1
        
        return current_bars
    
    def has_more_bars(self, instrument: str = None) -> bool:
        """
        Check if there are more bars available.
        
        Args:
            instrument: Instrument identifier (None to check all)
            
        Returns:
            True if more bars are available, False otherwise
        """
        if instrument:
            if instrument not in self.current_idx:
                return False
            return self.current_idx[instrument] < len(self.data[instrument])
        else:
            # Check if any instrument has more bars
            return any(
                self.current_idx[inst] < len(self.data[inst])
                for inst in self.current_idx.keys()
            )
    
    def reset(self) -> None:
        """Reset the data handler to the beginning of the data."""
        for instrument in self.current_idx.keys():
            self.current_idx[instrument] = 0
    
    def add_technical_indicators(
        self,
        instrument: str,
        indicators: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Add technical indicators to the data.
        
        Args:
            instrument: Instrument identifier
            indicators: Dictionary mapping indicator names to parameters
                Example: {'sma': {'window': 20}, 'rsi': {'window': 14}}
        """
        if instrument not in self.data:
            raise ValueError(f"No data loaded for instrument {instrument}")
        
        df = self.data[instrument]
        
        for indicator_name, params in indicators.items():
            try:
                if indicator_name == 'sma':
                    window = params.get('window', 20)
                    df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                
                elif indicator_name == 'ema':
                    window = params.get('window', 20)
                    df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
                
                elif indicator_name == 'rsi':
                    window = params.get('window', 14)
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=window).mean()
                    avg_loss = loss.rolling(window=window).mean()
                    rs = avg_gain / avg_loss
                    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
                
                elif indicator_name == 'bollinger_bands':
                    window = params.get('window', 20)
                    num_std = params.get('num_std', 2)
                    sma = df['close'].rolling(window=window).mean()
                    std = df['close'].rolling(window=window).std()
                    df[f'bb_upper_{window}'] = sma + (std * num_std)
                    df[f'bb_middle_{window}'] = sma
                    df[f'bb_lower_{window}'] = sma - (std * num_std)
                
                elif indicator_name == 'macd':
                    fast = params.get('fast', 12)
                    slow = params.get('slow', 26)
                    signal = params.get('signal', 9)
                    fast_ema = df['close'].ewm(span=fast, adjust=False).mean()
                    slow_ema = df['close'].ewm(span=slow, adjust=False).mean()
                    df[f'macd_{fast}_{slow}'] = fast_ema - slow_ema
                    df[f'macd_signal_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}'].ewm(span=signal, adjust=False).mean()
                    df[f'macd_hist_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}_{signal}']
                
                else:
                    self.logger.warning(f"Unknown indicator: {indicator_name}")
            
            except Exception as e:
                self.logger.error(f"Error calculating {indicator_name} for {instrument}: {str(e)}")
        
        # Update the stored data
        self.data[instrument] = df
