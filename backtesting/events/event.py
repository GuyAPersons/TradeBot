"""
Base event class and event type definitions.
"""
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any


class EventType(Enum):
    """
    Event types in the backtesting system.
    """
    MARKET = 'MARKET'  # New market data available
    SIGNAL = 'SIGNAL'  # Strategy generated a signal
    ORDER = 'ORDER'    # Order to be executed
    FILL = 'FILL'      # Order has been filled


class Event:
    """
    Base event class that all other event types will inherit from.
    """
    def __init__(
        self, 
        event_type: EventType,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an event.
        
        Args:
            event_type: Type of the event
            timestamp: Time when the event occurred (defaults to now)
            metadata: Additional information about the event
        """
        self.event_type = event_type
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        
    def __str__(self) -> str:
        """String representation of the event."""
        return f"{self.event_type.value} Event at {self.timestamp}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        event_type = EventType(data['event_type'])
        timestamp = datetime.fromisoformat(data['timestamp'])
        metadata = data.get('metadata', {})
        return cls(event_type, timestamp, metadata)
