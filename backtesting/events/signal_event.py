"""
Signal event class for strategy signals.
"""
from datetime import datetime
from typing import Optional, Dict, Any, Union
from enum import Enum

from .event import Event, EventType


class SignalType(Enum):
    """Types of trading signals."""
    LONG = 'LONG'          # Enter long position
    SHORT = 'SHORT'        # Enter short position
    EXIT_LONG = 'EXIT_LONG'  # Exit long position
    EXIT_SHORT = 'EXIT_SHORT'  # Exit short position
    HOLD = 'HOLD'          # No action


class SignalEvent(Event):
    """
    Event for strategy signals.
    
    This event is generated when a strategy makes a decision to enter or exit
    a position based on market data analysis.
    """
    def __init__(
        self,
        symbol: str,
        signal_type: SignalType,
        strength: float = 1.0,
        price: Optional[float] = None,
        strategy_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a signal event.
        
        Args:
            symbol: Symbol for which the signal is generated
            signal_type: Type of signal (LONG, SHORT, EXIT_LONG, EXIT_SHORT, HOLD)
            strength: Signal strength between 0 and 1 (used for position sizing)
            price: Price at which the signal was generated
            strategy_id: Identifier for the strategy that generated the signal
            timestamp: Time when the event occurred (defaults to now)
            metadata: Additional information about the event
        """
        super().__init__(EventType.SIGNAL, timestamp, metadata)
        
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = max(0.0, min(1.0, strength))  # Clamp between 0 and 1
        self.price = price
        self.strategy_id = strategy_id
    
    def __str__(self) -> str:
        """String representation of the signal event."""
        return (f"SIGNAL Event: {self.signal_type.value} {self.symbol} "
                f"(strength: {self.strength:.2f}) at {self.timestamp}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal event to dictionary for serialization."""
        data_dict = super().to_dict()
        data_dict.update({
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'price': self.price,
            'strategy_id': self.strategy_id
        })
        return data_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalEvent':
        """Create signal event from dictionary."""
        symbol = data['symbol']
        signal_type = SignalType(data['signal_type'])
        strength = data['strength']
        price = data.get('price')
        strategy_id = data.get('strategy_id')
        timestamp = datetime.fromisoformat(data['timestamp'])
        metadata = data.get('metadata', {})
        return cls(symbol, signal_type, strength, price, strategy_id, timestamp, metadata)
