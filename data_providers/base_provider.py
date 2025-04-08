from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime

class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers.
    Defines the interface that all data providers must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data provider with configuration.
        
        Args:
            config: Configuration dictionary with provider-specific settings
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.rate_limiter = None  # Will be initialized by child classes if needed
        self.logger = None  # Will be initialized by child classes
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the data source.
        
        Returns:
            True if disconnection is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_historical_data(
        self,
        instrument: str,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str],
        interval: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical market data for a specific instrument.
        
        Args:
            instrument: Instrument identifier (e.g., 'BTC/USD')
            start_time: Start time for historical data
            end_time: End time for historical data
            interval: Time interval for the data (e.g., '1m', '1h', '1d')
            **kwargs: Additional provider-specific parameters
            
        Returns:
            DataFrame with historical market data
        """
        pass
    
    @abstractmethod
    def get_latest_data(self, instrument: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch the latest market data for a specific instrument.
        
        Args:
            instrument: Instrument identifier (e.g., 'BTC/USD')
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary with latest market data
        """
        pass
    
    @abstractmethod
    def subscribe_to_stream(
        self,
        instruments: List[str],
        callback,
        stream_type: str = 'trade',
        **kwargs
    ) -> Any:
        """
        Subscribe to a real-time data stream.
        
        Args:
            instruments: List of instrument identifiers
            callback: Function to call when new data is received
            stream_type: Type of stream (e.g., 'trade', 'ticker', 'orderbook')
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Stream identifier or subscription object
        """
        pass
    
    @abstractmethod
    def unsubscribe_from_stream(self, stream_id: Any) -> bool:
        """
        Unsubscribe from a real-time data stream.
        
        Args:
            stream_id: Stream identifier returned by subscribe_to_stream
            
        Returns:
            True if unsubscription is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_instruments(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Get a list of available instruments.
        
        Args:
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of instrument dictionaries with metadata
        """
        pass
    
    def normalize_instrument_id(self, instrument: str) -> str:
        """
        Normalize instrument identifier to provider-specific format.
        
        Args:
            instrument: Instrument identifier in standard format
            
        Returns:
            Instrument identifier in provider-specific format
        """
        # Default implementation (override in provider-specific classes)
        return instrument
    
    def normalize_data(self, data: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """
        Normalize data to standard format.
        
        Args:
            data: Raw data from provider
            instrument: Instrument identifier
            
        Returns:
            Normalized data in standard format
        """
        # Default implementation (override in provider-specific classes)
        return data
    
    def check_rate_limit(self) -> bool:
        """
        Check if rate limit is reached.
        
        Returns:
            True if request can proceed, False if rate limit is reached
        """
        if self.rate_limiter:
            return self.rate_limiter.check()
        return True
    
    def handle_error(self, error: Exception, context: str = "") -> None:
        """
        Handle errors in data provider operations.
        
        Args:
            error: Exception that occurred
            context: Context in which the error occurred
        """
        if self.logger:
            self.logger.error(f"Error in {self.name} ({context}): {str(error)}")
        else:
            print(f"Error in {self.name} ({context}): {str(error)}")
