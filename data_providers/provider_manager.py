from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datetime import datetime
import logging
import threading

from .base_provider import BaseDataProvider

class DataProviderManager:
    """
    Manager for multiple data providers.
    Handles provider selection, failover, and data aggregation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data provider manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.providers = {}
        self.default_provider = None
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
    
    def add_provider(self, provider: BaseDataProvider, is_default: bool = False) -> None:
        """
        Add a data provider.
        
        Args:
            provider: Data provider instance
            is_default: Whether this is the default provider
        """
        with self.lock:
            self.providers[provider.name] = provider
            if is_default or self.default_provider is None:
                self.default_provider = provider.name
    
    def remove_provider(self, provider_name: str) -> bool:
        """
        Remove a data provider.
        
        Args:
            provider_name: Name of the provider to remove
            
        Returns:
            True if provider was removed, False otherwise
        """
        with self.lock:
            if provider_name in self.providers:
                # If removing the default provider, set a new default if possible
                if provider_name == self.default_provider and len(self.providers) > 1:
                    self.default_provider = next(name for name in self.providers.keys() if name != provider_name)
                elif provider_name == self.default_provider:
                    self.default_provider = None
                
                del self.providers[provider_name]
                return True
            return False
    
    def get_provider(self, provider_name: Optional[str] = None) -> BaseDataProvider:
        """
        Get a data provider by name.
        
        Args:
            provider_name: Name of the provider (uses default if None)
            
        Returns:
            Data provider instance
            
        Raises:
            ValueError: If provider doesn't exist
        """
        name = provider_name or self.default_provider
        if not name or name not in self.providers:
            raise ValueError(f"Provider '{name}' not found")
        return self.providers[name]
    
    def get_historical_data(
        self,
        instrument: str,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str],
        interval: str,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical market data.
        
        Args:
            instrument: Instrument identifier
            start_time: Start time for historical data
            end_time: End time for historical data
            interval: Time interval for the data
            provider_name: Name of the provider to use (uses default if None)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            DataFrame with historical market data
        """
        provider = self.get_provider(provider_name)
        try:
            return provider.get_historical_data(instrument, start_time, end_time, interval, **kwargs)
        except Exception as e:
            self.logger.error(f"Error fetching historical data from {provider.name}: {str(e)}")
            # Try failover if available and different from the original provider
            if self.config.get('enable_failover', True) and len(self.providers) > 1:
                for name, p in self.providers.items():
                    if name != provider.name:
                        try:
                            self.logger.info(f"Attempting failover to {name} for historical data")
                            return p.get_historical_data(instrument, start_time, end_time, interval, **kwargs)
                        except Exception as e2:
                            self.logger.error(f"Failover to {name} also failed: {str(e2)}")
            
            # Re-raise the original exception if failover is disabled or all failovers failed
            raise
    
    def get_latest_data(
        self,
        instrument: str,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch the latest market data.
        
        Args:
            instrument: Instrument identifier
            provider_name: Name of the provider to use (uses default if None)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary with latest market data
        """
        provider = self.get_provider(provider_name)
        try:
            return provider.get_latest_data(instrument, **kwargs)
        except Exception as e:
            self.logger.error(f"Error fetching latest data from {provider.name}: {str(e)}")
            # Try failover if available
            if self.config.get('enable_failover', True) and len(self.providers) > 1:
                for name, p in self.providers.items():
                    if name != provider.name:
                        try:
                            self.logger.info(f"Attempting failover to {name} for latest data")
                            return p.get_latest_data(instrument, **kwargs)
                        except Exception as e2:
                            self.logger.error(f"Failover to {name} also failed: {str(e2)}")
            
            # Re-raise the original exception if failover is disabled or all failovers failed
            raise
    
    def subscribe_to_stream(
        self,
        instruments: List[str],
        callback,
        stream_type: str = 'trade',
        provider_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Subscribe to a real-time data stream.
        
        Args:
            instruments: List of instrument identifiers
            callback: Function to call when new data is received
            stream_type: Type of stream
            provider_name: Name of the provider to use (uses default if None)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary with subscription details
        """
        provider = self.get_provider(provider_name)
        try:
            stream_id = provider.subscribe_to_stream(instruments, callback, stream_type, **kwargs)
            return {
                'provider': provider.name,
                'stream_id': stream_id,
                'instruments': instruments,
                'stream_type': stream_type
            }
        except Exception as e:
            self.logger.error(f"Error subscribing to stream from {provider.name}: {str(e)}")
            # Try failover if available
            if self.config.get('enable_failover', True) and len(self.providers) > 1:
                for name, p in self.providers.items():
                    if name != provider.name:
                        try:
                            self.logger.info(f"Attempting failover to {name} for stream subscription")
                            stream_id = p.subscribe_to_stream(instruments, callback, stream_type, **kwargs)
                            return {
                                'provider': p.name,
                                'stream_id': stream_id,
                                'instruments': instruments,
                                'stream_type': stream_type
                            }
                        except Exception as e2:
                            self.logger.error(f"Failover to {name} also failed: {str(e2)}")
            
            # Re-raise the original exception if failover is disabled or all failovers failed
            raise
    
    def unsubscribe_from_stream(self, subscription: Dict[str, Any]) -> bool:
        """
        Unsubscribe from a real-time data stream.
        
        Args:
            subscription: Subscription details returned by subscribe_to_stream
            
        Returns:
            True if unsubscription is successful, False otherwise
        """
        provider_name = subscription.get('provider')
        stream_id = subscription.get('stream_id')
        
        if not provider_name or not stream_id:
            self.logger.error("Invalid subscription details")
            return False
        
        try:
            provider = self.get_provider(provider_name)
            return provider.unsubscribe_from_stream(stream_id)
        except Exception as e:
            self.logger.error(f"Error unsubscribing from stream: {str(e)}")
            return False
    
    def get_instruments(
        self,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get a list of available instruments.
        
        Args:
            provider_name: Name of the provider to use (uses default if None)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of instrument dictionaries with metadata
        """
        provider = self.get_provider(provider_name)
        try:
            return provider.get_instruments(**kwargs)
        except Exception as e:
            self.logger.error(f"Error fetching instruments from {provider.name}: {str(e)}")
            # Try failover if available
            if self.config.get('enable_failover', True) and len(self.providers) > 1:
                for name, p in self.providers.items():
                    if name != provider.name:
                        try:
                            self.logger.info(f"Attempting failover to {name} for instruments")
                            return p.get_instruments(**kwargs)
                        except Exception as e2:
                            self.logger.error(f"Failover to {name} also failed: {str(e2)}")
            
            # Re-raise the original exception if failover is disabled or all failovers failed
            raise
    
    def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all data providers.
        
        Returns:
            Dictionary mapping provider names to connection status
        """
        results = {}
        for name, provider in self.providers.items():
            try:
                results[name] = provider.connect()
            except Exception as e:
                self.logger.error(f"Error connecting to {name}: {str(e)}")
                results[name] = False
        return results
    
    def disconnect_all(self) -> Dict[str, bool]:
        """
        Disconnect from all data providers.
        
        Returns:
            Dictionary mapping provider names to disconnection status
        """
        results = {}
        for name, provider in self.providers.items():
            try:
                results[name] = provider.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting from {name}: {str(e)}")
                results[name] = False
        return results