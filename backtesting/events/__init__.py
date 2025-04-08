"""
Events module for the backtesting framework.

This module provides an event-driven architecture for the backtesting system,
allowing components to communicate through events.
"""

from .event import Event, EventType
from .market_event import MarketEvent
from .signal_event import SignalEvent
from .order_event import OrderEvent
from .fill_event import FillEvent

__all__ = [
    'Event', 'EventType',
    'MarketEvent', 'SignalEvent', 'OrderEvent', 'FillEvent'
]
