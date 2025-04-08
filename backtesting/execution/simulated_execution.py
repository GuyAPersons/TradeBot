"""
Simulated execution handler for backtesting.
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
import logging
import pandas as pd

from ..events.order_event import OrderEvent, OrderType, OrderDirection
from ..events.fill_event import FillEvent
from ..models.slippage import SlippageModel, FixedSlippageModel
from ..models.transaction_costs import TransactionCostModel, FixedCostModel
from .execution_handler import ExecutionHandler


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulated execution handler for backtesting.
    
    This class simulates the execution of orders in a backtesting environment.
    It applies slippage and transaction costs to orders to simulate real-world
    trading conditions.
    """
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        data_handler = None,  # Avoid circular import
        fill_probability: float = 1.0,
        partial_fill_probability: float = 0.0,
        latency_ms: float = 0.0
    ):
        """
        Initialize the simulated execution handler.
        
        Args:
            slippage_model: Model for simulating price slippage
            transaction_cost_model: Model for calculating transaction costs
            data_handler: DataHandler instance for market data
            fill_probability: Probability of an order being filled (0.0-1.0)
            partial_fill_probability: Probability of a partial fill (0.0-1.0)
            latency_ms: Simulated latency in milliseconds
        """
        super().__init__()
        
        # Set default models if none provided
        self.slippage_model = slippage_model or FixedSlippageModel()
        self.transaction_cost_model = transaction_cost_model or FixedCostModel()
        
        self.data_handler = data_handler
        self.fill_probability = max(0.0, min(1.0, fill_probability))
        self.partial_fill_probability = max(0.0, min(1.0, partial_fill_probability))
        self.latency_ms = max(0.0, latency_ms)
        
        # Store open orders
        self.open_orders: Dict[str, OrderEvent] = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute_order(self, order_event: OrderEvent) -> Optional[FillEvent]:
        """
        Execute an order in the simulated environment.
        
        Args:
            order_event: The order to execute
            
        Returns:
            A FillEvent if the order was executed, None otherwise
        """
        # Generate a unique order ID if not provided
        if order_event.order_id is None:
            order_event.order_id = str(uuid.uuid4())
        
        # Store the order
        self.open_orders[order_event.order_id] = order_event
        
        # For market orders, execute immediately
        if order_event.order_type == OrderType.MARKET:
            return self._execute_market_order(order_event)
        
        # For other order types, store them for later execution
        self.logger.info(f"Order {order_event.order_id} added to open orders")
        return None
    
    def _execute_market_order(self, order_event: OrderEvent) -> Optional[FillEvent]:
        """
        Execute a market order.
        
        Args:
            order_event: The market order to execute
            
        Returns:
            A FillEvent if the order was executed, None otherwise
        """
        # Get the current market price
        current_price = self._get_current_price(order_event.symbol, order_event.direction)
        
        if current_price is None:
            self.logger.warning(f"No price available for {order_event.symbol}, order not executed")
            return None
        
        # Apply slippage to the price
        fill_price = self.slippage_model.apply_slippage(
            current_price, 
            order_event.quantity,
            order_event.direction
        )
        
        # Calculate commission
        commission = self.transaction_cost_model.calculate_cost(
            fill_price, 
            order_event.quantity,
            order_event.symbol
        )
        
        # Create and return the fill event
        fill_event = FillEvent(
            symbol=order_event.symbol,
            direction=order_event.direction,
            quantity=order_event.quantity,
            fill_price=fill_price,
            commission=commission,
            order_id=order_event.order_id,
            timestamp=datetime.now()
        )
        
        # Remove the order from open orders
        if order_event.order_id in self.open_orders:
            del self.open_orders[order_event.order_id]
        
        self.logger.info(f"Order executed: {fill_event}")
        return fill_event
    
    def _get_current_price(self, symbol: str, direction: OrderDirection) -> Optional[float]:
        """
        Get the current market price for a symbol.
        
        Args:
            symbol: The symbol to get the price for
            direction: The direction of the order (BUY/SELL)
            
        Returns:
            The current price, or None if not available
        """
        if self.data_handler is None:
            self.logger.warning("No data handler available to get current price")
            return None
        
        try:
            # Get the latest bar for the symbol
            latest_bar = self.data_handler.get_latest_bar(symbol)
            
            if latest_bar is None:
                return None
            
            # Use close price as default
            if 'close' in latest_bar:
                return latest_bar['close']
            
            # If no close price, try to use bid/ask for more realistic pricing
            if direction == OrderDirection.BUY and 'ask' in latest_bar:
                return latest_bar['ask']
            elif direction == OrderDirection.SELL and 'bid' in latest_bar:
                return latest_bar['bid']
            
            # Fall back to other price fields if available
            for field in ['price', 'mid', 'open', 'high', 'low']:
                if field in latest_bar:
                    return latest_bar[field]
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
    
    def get_open_orders(self) -> List[OrderEvent]:
        """
        Get a list of all open orders.
        
        Returns:
            A list of open OrderEvent objects
        """
        return list(self.open_orders.values())
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            True if the order was canceled, False otherwise
        """
        if order_id in self.open_orders:
            del self.open_orders[order_id]
            self.logger.info(f"Order {order_id} canceled")
            return True
        
        self.logger.warning(f"Order {order_id} not found, could not cancel")
        return False
    
    def update(self, market_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> List[FillEvent]:
        """
        Update the execution handler with new market data.
        
        This method checks if any open limit or stop orders should be executed
        based on the new market data.
        
        Args:
            market_data: New market data
            
        Returns:
            A list of FillEvent objects for orders that were executed
        """
        fill_events = []
        
        # Process each open order
        for order_id, order in list(self.open_orders.items()):
            # Skip market orders (they should be executed immediately)
            if order.order_type == OrderType.MARKET:
                continue
            
            # Get the current price
            current_price = self._get_current_price(order.symbol, order.direction)
            
            if current_price is None:
                continue
            
            # Check if limit order should be executed
            if order.order_type == OrderType.LIMIT:
                if (order.direction == OrderDirection.BUY and current_price <= order.price) or \
                   (order.direction == OrderDirection.SELL and current_price >= order.price):
                    # Execute the order
                    fill_event = self._execute_limit_order(order, current_price)
                    if fill_event:
                        fill_events.append(fill_event)
            
            # Check if stop order should be executed
            elif order.order_type == OrderType.STOP:
                if (order.direction == OrderDirection.BUY and current_price >= order.stop_price) or \
                   (order.direction == OrderDirection.SELL and current_price <= order.stop_price):
                    # Execute the order
                    fill_event = self._execute_stop_order(order, current_price)
                    if fill_event:
                        fill_events.append(fill_event)
            
            # Check if stop-limit order should be triggered
            elif order.order_type == OrderType.STOP_LIMIT:
                # First check if stop price is triggered
                if (order.direction == OrderDirection.BUY and current_price >= order.stop_price) or \
                   (order.direction == OrderDirection.SELL and current_price <= order.stop_price):
                    # Then check if limit price is satisfied
                    if (order.direction == OrderDirection.BUY and current_price <= order.price) or \
                       (order.direction == OrderDirection.SELL and current_price >= order.price):
                        # Execute the order
                        fill_event = self._execute_stop_limit_order(order, current_price)
                        if fill_event:
                            fill_events.append(fill_event)
        
        return fill_events
    
    def _execute_limit_order(self, order: OrderEvent, current_price: float) -> Optional[FillEvent]:
        """
        Execute a limit order.
        
        Args:
            order: The limit order to execute
            current_price: The current market price
            
        Returns:
            A FillEvent if the order was executed, None otherwise
        """
        # Use the limit price for the fill
        fill_price = order.price
        
        # Apply slippage (minimal for limit orders)
        fill_price = self.slippage_model.apply_slippage(
            fill_price, 
            order.quantity,
            order.direction,
            slippage_factor=0.1  # Reduced slippage for limit orders
        )
        
        # Calculate commission
        commission = self.transaction_cost_model.calculate_cost(
            fill_price, 
            order.quantity,
            order.symbol
        )
        
        # Create and return the fill event
        fill_event = FillEvent(
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            order_id=order.order_id,
            timestamp=datetime.now()
        )
        
        # Remove the order from open orders
        if order.order_id in self.open_orders:
            del self.open_orders[order.order_id]
        
        self.logger.info(f"Limit order executed: {fill_event}")
        return fill_event
    
    def _execute_stop_order(self, order: OrderEvent, current_price: float) -> Optional[FillEvent]:
        """
        Execute a stop order.
        
        Args:
            order: The stop order to execute
            current_price: The current market price
            
        Returns:
            A FillEvent if the order was executed, None otherwise
        """
        # For stop orders, use the current price (market order once triggered)
        fill_price = current_price
        
        # Apply slippage (higher for stop orders)
        fill_price = self.slippage_model.apply_slippage(
            fill_price, 
            order.quantity,
            order.direction,
            slippage_factor=1.5  # Increased slippage for stop orders
        )
        
        # Calculate commission
        commission = self.transaction_cost_model.calculate_cost(
            fill_price, 
            order.quantity,
            order.symbol
        )
        
        # Create and return the fill event
        fill_event = FillEvent(
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            order_id=order.order_id,
            timestamp=datetime.now()
        )
        
        # Remove the order from open orders
        if order.order_id in self.open_orders:
            del self.open_orders[order.order_id]
        
        self.logger.info(f"Stop order executed: {fill_event}")
        return fill_event
    
    def _execute_stop_limit_order(self, order: OrderEvent, current_price: float) -> Optional[FillEvent]:
        """
        Execute a stop-limit order.
        
        Args:
            order: The stop-limit order to execute
            current_price: The current market price
            
        Returns:
            A FillEvent if the order was executed, None otherwise
        """
        # For stop-limit orders, use the limit price
        fill_price = order.price
        
        # Apply slippage (moderate for stop-limit orders)
        fill_price = self.slippage_model.apply_slippage(
            fill_price, 
            order.quantity,
            order.direction,
            slippage_factor=1.2  # Moderate slippage for stop-limit orders
        )
        
        # Calculate commission
        commission = self.transaction_cost_model.calculate_cost(
            fill_price, 
            order.quantity,
            order.symbol
        )
        
        # Create and return the fill event
        fill_event = FillEvent(
            symbol=order.symbol,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            order_id=order.order_id,
            timestamp=datetime.now()
        )
        
        # Remove the order from open orders
        if order.order_id in self.open_orders:
            del self.open_orders[order.order_id]
        
        self.logger.info(f"Stop-limit order executed: {fill_event}")
        return fill_event
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the execution handler to a dictionary for serialization.
        
        Returns:
            A dictionary representation of the execution handler
        """
        data = super().to_dict()
        data.update({
            'slippage_model': self.slippage_model.to_dict() if hasattr(self.slippage_model, 'to_dict') else None,
            'transaction_cost_model': self.transaction_cost_model.to_dict() if hasattr(self.transaction_cost_model, 'to_dict') else None,
            'fill_probability': self.fill_probability,
            'partial_fill_probability': self.partial_fill_probability,
            'latency_ms': self.latency_ms,
            'open_orders': {order_id: order.to_dict() for order_id, order in self.open_orders.items()}
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulatedExecutionHandler':
        """
        Create a simulated execution handler from a dictionary.
        
        Args:
            data: Dictionary containing execution handler data
            
        Returns:
            A SimulatedExecutionHandler instance
        """
        # Import here to avoid circular imports
        from ..models.slippage import SlippageModel
        from ..models.transaction_costs import TransactionCostModel
        
        # Create slippage model if provided
        slippage_model = None
        if data.get('slippage_model'):
            slippage_model = SlippageModel.from_dict(data['slippage_model'])
        
        # Create transaction cost model if provided
        transaction_cost_model = None
        if data.get('transaction_cost_model'):
            transaction_cost_model = TransactionCostModel.from_dict(data['transaction_cost_model'])
        
        # Create the execution handler
        handler = cls(
            slippage_model=slippage_model,
            transaction_cost_model=transaction_cost_model,
            fill_probability=data.get('fill_probability', 1.0),
            partial_fill_probability=data.get('partial_fill_probability', 0.0),
            latency_ms=data.get('latency_ms', 0.0)
        )
        
        # Restore open orders if provided
        if 'open_orders' in data:
            for order_id, order_data in data['open_orders'].items():
                handler.open_orders[order_id] = OrderEvent.from_dict(order_data)
        
        return handler