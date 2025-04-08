from typing import Dict, Any, Optional
import logging

from .base_provider import BaseDataProvider
from .provider_manager import DataProviderManager
from .exchanges.binance_provider import BinanceDataProvider
from .exchanges.coinbase_provider import CoinbaseDataProvider

class DataProviderFactory:
    """
    Factory class for creating data provider instances.
    """
    
    @staticmethod
    def create_provider(provider_type: str, config: Dict[str, Any]) -> BaseDataProvider:
        """
        Create a data provider instance.
        
        Args:
            provider_type: Type of provider ('binance', 'coinbase', etc.)
            config: Configuration dictionary
            
        Returns:
            Data provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        logger = logging.getLogger(__name__)
        
        provider_type = provider_type.lower()
        
        if provider_type == 'binance':
            logger.info("Creating Binance data provider")
            return BinanceDataProvider(config)
        elif provider_type == 'coinbase':
            logger.info("Creating Coinbase Pro data provider")
            return CoinbaseDataProvider(config)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def create_manager(config: Dict[str, Any]) -> DataProviderManager:
        """
        Create a data provider manager with configured providers.
        
        Args:
            config: Configuration dictionary with provider configurations
            
        Returns:
            Data provider manager instance
        """
        logger = logging.getLogger(__name__)
        
        manager = DataProviderManager(config)
        
        # Create and add providers from configuration
        providers_config = config.get('providers', {})
        default_provider = config.get('default_provider')
        
        for provider_name, provider_config in providers_config.items():
            provider_type = provider_config.get('type')
            if not provider_type:
                logger.warning(f"Provider type not specified for {provider_name}, skipping")
                continue
            
            try:
                provider = DataProviderFactory.create_provider(provider_type, provider_config)
                is_default = (provider_name == default_provider)
                manager.add_provider(provider, is_default=is_default)
                logger.info(f"Added {provider_type} provider as {provider_name}")
            except Exception as e:
                logger.error(f"Failed to create provider {provider_name}: {str(e)}")
        
        return manager
