"""
Order event class for trading orders.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from .event import Event, EventType


class OrderType(Enum):
    """Types of orders."""
    MARKET = 'MARKET'  # Market order
    LIMIT = 'LIMIT'    # Limit order
    STOP = 'STOP'      # Stop order
    STOP_LIMIT = 'STOP_LIMIT'  # Stop-limit order


class OrderDirection(Enum):
    """Order directions."""
    BUY = 'BUY'        # Buy order
    SELL = 'SELL'      # Sell order


class OrderEvent(Event):
    """
    Event for trading orders.
    
    This event is generated when a decision is made to place an order
    in the market, either from a strategy signal or portfolio rebalancing.
    """
    def __init__(
        self,
        symbol: str,
        order_type: OrderType,
        direction: OrderDirection,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        order_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an order event.
        
        Args:
            symbol: Symbol to trade
            order_type: Type of order (MARKET, LIMIT, STOP, STOP_LIMIT)
            direction: Order direction (BUY, SELL)
            quantity: Quantity to trade
            price: Price for limit orders
            stop_price: Price for stop orders
            order_id: Unique identifier for the order
            strategy_id: Identifier for the strategy that generated the order
            timestamp: Time when the event occurred (defaults to now)
            metadata: Additional information about the event
        """
        super().__init__(EventType.ORDER, timestamp, metadata)
        
        self.symbol = symbol
        self.order_type = order_type
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.order_id = order_id
        self.strategy_id = strategy_id
    
    def __str__(self) -> str:
        """String representation of the order event."""
        price_str = f" at {self.price}" if self.price is not None else ""
        return (f"ORDER Event: {self.direction.value} {self.quantity} {self.symbol} "
                f"({self.order_type.value}{price_str}) at {self.timestamp}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order event to dictionary for serialization."""
        data_dict = super().to_dict()
        data_dict.update({
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'direction': self.direction.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'order_id': self.order_id,
            'strategy_id': self.strategy_id
        })
        return data_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderEvent':
        """Create order event from dictionary."""
        symbol = data['symbol']
        order_type = OrderType(data['order_type'])
        direction = OrderDirection(data['direction'])
        quantity = data['quantity']
        price = data.get('price')
        stop_price = data.get('stop_price')
        order_id = data.get('order_id')
        strategy_id = data.get('strategy_id')
        timestamp = datetime.fromisoformat(data['timestamp'])
        metadata = data.get('metadata', {})
        return cls(
            symbol, order_type, direction, quantity, price, 
            stop_price, order_id, strategy_id, timestamp, metadata
        )
