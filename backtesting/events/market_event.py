"""
Market event class for new market data.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import pandas as pd

from .event import Event, EventType


class MarketEvent(Event):
    """
    Event for new market data.
    
    This event is triggered when new market data (e.g., bar data, tick data)
    becomes available for processing.
    """
    def __init__(
        self,
        symbols: Union[str, List[str]],
        data: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a market event.
        
        Args:
            symbols: Symbol or list of symbols for which data is available
            data: Market data as a pandas DataFrame
            timestamp: Time when the event occurred (defaults to now)
            metadata: Additional information about the event
        """
        super().__init__(EventType.MARKET, timestamp, metadata)
        
        # Convert single symbol to list
        self.symbols = [symbols] if isinstance(symbols, str) else symbols
        
        # Store market data
        self.data = data
    
    def __str__(self) -> str:
        """String representation of the market event."""
        symbols_str = ", ".join(self.symbols)
        return f"MARKET Event for {symbols_str} at {self.timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert market event to dictionary for serialization."""
        data_dict = super().to_dict()
        data_dict.update({
            'symbols': self.symbols,
            # Convert DataFrame to dict if present
            'data': self.data.to_dict() if self.data is not None else None
        })
        return data_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketEvent':
        """Create market event from dictionary."""
        symbols = data['symbols']
        market_data = pd.DataFrame(data['data']) if data.get('data') is not None else None
        timestamp = datetime.fromisoformat(data['timestamp'])
        metadata = data.get('metadata', {})
        return cls(symbols, market_data, timestamp, metadata)
