import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

class DataNormalizer:
    """
    Utility for normalizing data from different sources to a standard format.
    """
    
    @staticmethod
    def normalize_ohlcv(
        df: pd.DataFrame,
        timestamp_column: str,
        open_column: str,
        high_column: str,
        low_column: str,
        close_column: str,
        volume_column: str,
        timestamp_unit: str = 'ms'
    ) -> pd.DataFrame:
        """
        Normalize OHLCV data to standard format.
        
        Args:
            df: DataFrame with raw data
            timestamp_column: Name of timestamp column
            open_column: Name of open price column
            high_column: Name of high price column
            low_column: Name of low price column
            close_column: Name of close price column
            volume_column: Name of volume column
            timestamp_unit: Unit of timestamp ('ms', 's', etc.)
            
        Returns:
            Normalized DataFrame with standard column names
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if pd.api.types.is_numeric_dtype(result[timestamp_column]):
            result['timestamp'] = pd.to_datetime(result[timestamp_column], unit=timestamp_unit)
        else:
            result['timestamp'] = pd.to_datetime(result[timestamp_column])
        
        # Rename columns to standard names
        column_mapping = {
            open_column: 'open',
            high_column: 'high',
            low_column: 'low',
            close_column: 'close',
            volume_column: 'volume'
        }
        
        result = result.rename(columns=column_mapping)
        
        # Select and reorder columns
        standard_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        result = result[standard_columns]
        
        # Set timestamp as index
        result = result.set_index('timestamp')
        
        # Sort by timestamp
        result = result.sort_index()
        
        return result
    
    @staticmethod
    def normalize_ticker(ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize ticker data to standard format.
        
        Args:
            ticker_data: Raw ticker data from provider
            
        Returns:
            Normalized ticker data with standard keys
        """
        # This is a template - actual implementation will depend on the specific providers
        standard_ticker = {
            'symbol': ticker_data.get('symbol', ''),
            'bid': float(ticker_data.get('bid', 0)),
            'ask': float(ticker_data.get('ask', 0)),
            'last': float(ticker_data.get('last', 0)),
            'volume': float(ticker_data.get('volume', 0)),
            'timestamp': pd.to_datetime(ticker_data.get('timestamp', datetime.now()))
        }
        
        return standard_ticker
    
    @staticmethod
    def normalize_orderbook(orderbook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize orderbook data to standard format.
        
        Args:
            orderbook_data: Raw orderbook data from provider
            
        Returns:
            Normalized orderbook data with standard keys
        """
        # This is a template - actual implementation will depend on the specific providers
        standard_orderbook = {
            'symbol': orderbook_data.get('symbol', ''),
            'bids': [[float(price), float(amount)] for price, amount in orderbook_data.get('bids', [])],
            'asks': [[float(price), float(amount)] for price, amount in orderbook_data.get('asks', [])],
            'timestamp': pd.to_datetime(orderbook_data.get('timestamp', datetime.now()))
        }
        
        return standard_orderbook
    
    @staticmethod
    def normalize_trade(trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize trade data to standard format.
        
        Args:
            trade_data: Raw trade data from provider
            
        Returns:
            Normalized trade data with standard keys
        """
        # This is a template - actual implementation will depend on the specific providers
        standard_trade = {
            'symbol': trade_data.get('symbol', ''),
            'id': trade_data.get('id', ''),
            'price': float(trade_data.get('price', 0)),
            'amount': float(trade_data.get('amount', 0)),
            'side': trade_data.get('side', ''),
            'timestamp': pd.to_datetime(trade_data.get('timestamp', datetime.now()))
        }
        
        return standard_trade
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a DataFrame by removing NaN values, duplicates, etc.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with NaN values
        df = df.dropna()
        
        # Remove duplicate indices
        df = df[~df.index.duplicated(keep='first')]
        
        # Remove rows with zero or negative prices
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df = df[df[col] > 0]
        
        return df
