"""
Fill event class for executed orders.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4

from .event import Event, EventType
from .order_event import OrderDirection


class FillEvent(Event):
    """
    Event for order fills.
    
    This event is generated when an order has been executed in the market,
    either fully or partially.
    """
    def __init__(
        self,
        symbol: str,
        direction: OrderDirection,
        quantity: float,
        fill_price: float,
        commission: float = 0.0,
        order_id: Optional[str] = None,
        fill_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a fill event.
        
        Args:
            symbol: Symbol that was traded
            direction: Direction of the fill (BUY, SELL)
            quantity: Quantity that was filled
            fill_price: Price at which the order was filled
            commission: Commission or fees paid for the fill
            order_id: ID of the original order
            fill_id: Unique identifier for the fill
            timestamp: Time when the event occurred (defaults to now)
            metadata: Additional information about the event
        """
        super().__init__(EventType.FILL, timestamp, metadata)
        
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.fill_price = fill_price
        self.commission = commission
        self.order_id = order_id
        self.fill_id = fill_id or str(uuid4())
    
    @property
    def cost(self) -> float:
        """Calculate the total cost of the fill including commission."""
        return self.quantity * self.fill_price + self.commission
    
    def __str__(self) -> str:
        """String representation of the fill event."""
        return (f"FILL Event: {self.direction.value} {self.quantity} {self.symbol} "
                f"at {self.fill_price} (commission: {self.commission}) at {self.timestamp}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fill event to dictionary for serialization."""
        data_dict = super().to_dict()
        data_dict.update({
            'symbol': self.symbol,
            'direction': self.direction.value,
            'quantity': self.quantity,
            'fill_price': self.fill_price,
            'commission': self.commission,
            'order_id': self.order_id,
            'fill_id': self.fill_id
        })
        return data_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FillEvent':
        """Create fill event from dictionary."""
        symbol = data['symbol']
        direction = OrderDirection(data['direction'])
        quantity = data['quantity']
        fill_price = data['fill_price']
        commission = data.get('commission', 0.0)
        order_id = data.get('order_id')
        fill_id = data.get('fill_id')
        timestamp = datetime.fromisoformat(data['timestamp'])
        metadata = data.get('metadata', {})
        return cls(
            symbol, direction, quantity, fill_price, commission,
            order_id, fill_id, timestamp, metadata
        )