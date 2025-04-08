import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime
import logging
from enum import Enum

class OrderType(Enum):
    """Enum for order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Enum for order sides."""
    BUY = "buy"
    SELL = "sell"

class Order:
    """Class representing a trading order."""
    
    def __init__(
        self,
        instrument: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        order_id: Optional[str] = None
    ):
        """
        Initialize an order.
        
        Args:
            instrument: Instrument identifier
            order_type: Type of order (market, limit, etc.)
            side: Order side (buy or sell)
            quantity: Order quantity
            price: Order price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            timestamp: Order timestamp
            order_id: Unique order identifier
        """
        self.instrument = instrument
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.timestamp = timestamp or datetime.now()
        self.order_id = order_id or f"{self.timestamp.timestamp()}_{instrument}"
        
        # Order status
        self.is_filled = False
        self.fill_price = None
        self.fill_timestamp = None
        self.fill_quantity = 0.0
        self.commission = 0.0
        self.slippage = 0.0
        
        # Validate order
        self._validate()
    
    def _validate(self):
        """Validate the order parameters."""
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError(f"Price is required for {self.order_type.value} orders")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"Stop price is required for {self.order_type.value} orders")
        
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
    
    def fill(
        self,
        fill_price: float,
        fill_timestamp: datetime,
        fill_quantity: Optional[float] = None,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> None:
        """
        Mark the order as filled.
        
        Args:
            fill_price: Price at which the order was filled
            fill_timestamp: Timestamp when the order was filled
            fill_quantity: Quantity that was filled (defaults to full order quantity)
            commission: Commission paid for the order
            slippage: Slippage incurred on the order
        """
        self.is_filled = True
        self.fill_price = fill_price
        self.fill_timestamp = fill_timestamp
        self.fill_quantity = fill_quantity if fill_quantity is not None else self.quantity
        self.commission = commission
        self.slippage = slippage
    
    def __str__(self) -> str:
        """String representation of the order."""
        status = "Filled" if self.is_filled else "Open"
        return (
            f"Order {self.order_id}: {status} | "
            f"{self.side.value.upper()} {self.quantity} {self.instrument} @ "
            f"{self.price if self.price else 'MARKET'}"
        )


class Position:
    """Class representing a trading position."""
    
    def __init__(self, instrument: str):
        """
        Initialize a position.
        
        Args:
            instrument: Instrument identifier
        """
        self.instrument = instrument
        self.quantity = 0.0
        self.average_price = 0.0
        self.cost_basis = 0.0
        self.realized_pnl = 0.0
        self.trades = []
    
    def update(self, order: Order) -> float:
        """
        Update the position based on a filled order.
        
        Args:
            order: Filled order
            
        Returns:
            Realized P&L from this order
        """
        if not order.is_filled:
            raise ValueError("Cannot update position with unfilled order")
        
        if order.instrument != self.instrument:
            raise ValueError(f"Order instrument {order.instrument} does not match position instrument {self.instrument}")
        
        # Calculate trade value
        trade_value = order.fill_quantity * order.fill_price
        
        # Update position based on order side
        if order.side == OrderSide.BUY:
            # Calculate new average price and cost basis for buys
            if self.quantity + order.fill_quantity != 0:
                self.average_price = (self.cost_basis + trade_value) / (self.quantity + order.fill_quantity)
            self.cost_basis += trade_value
            self.quantity += order.fill_quantity
            realized_pnl = 0.0
        else:  # SELL
            # Calculate realized P&L for sells
            if self.quantity > 0:
                realized_pnl = (order.fill_price - self.average_price) * min(order.fill_quantity, self.quantity)
            else:
                realized_pnl = 0.0
            
            # Update position
            self.quantity -= order.fill_quantity
            
            # If position is closed or flipped, adjust cost basis
            if self.quantity <= 0:
                remaining_quantity = abs(self.quantity)
                self.cost_basis = remaining_quantity * order.fill_price if remaining_quantity > 0 else 0.0
                self.average_price = order.fill_price if remaining_quantity > 0 else 0.0
        
        # Update realized P&L
        self.realized_pnl += realized_pnl
        
        # Record the trade
        self.trades.append({
            'timestamp': order.fill_timestamp,
            'side': order.side.value,
            'quantity': order.fill_quantity,
            'price': order.fill_price,
            'value': trade_value,
            'commission': order.commission,
            'slippage': order.slippage,
            'realized_pnl': realized_pnl
        })
        
        return realized_pnl
    
    def market_value(self, current_price: float) -> float:
        """
        Calculate the current market value of the position.
        
        Args:
            current_price: Current market price of the instrument
            
        Returns:
            Market value of the position
        """
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate the unrealized P&L of the position.
        
        Args:
            current_price: Current market price of the instrument
            
        Returns:
            Unrealized P&L
        """
        if self.quantity == 0:
            return 0.0
        
        return (current_price - self.average_price) * self.quantity
    
    def total_pnl(self, current_price: float) -> float:
        """
        Calculate the total P&L (realized + unrealized).
        
        Args:
            current_price: Current market price of the instrument
            
        Returns:
            Total P&L
        """
        return self.realized_pnl + self.unrealized_pnl(current_price)
    
    def __str__(self) -> str:
        """String representation of the position."""
        return (
            f"Position: {self.instrument} | "
            f"Quantity: {self.quantity} | "
            f"Avg Price: {self.average_price:.2f} | "
            f"Cost Basis: {self.cost_basis:.2f} | "
            f"Realized P&L: {self.realized_pnl:.2f}"
        )


class Portfolio:
    """
    Portfolio class for tracking positions, cash, and performance.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the portfolio.
        
        Args:
            initial_capital: Initial capital in the portfolio
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.transactions: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.equity_curve = []
        self.drawdowns = []
        self.returns = []
        
        # Current state
        self.current_timestamp = None
        self.logger = logging.getLogger(__name__)
    
    def place_order(self, order: Order) -> str:
        """
        Place a new order.
        
        Args:
            order: Order to place
            
        Returns:
            Order ID
        """
        self.orders.append(order)
        self.logger.info(f"Placed order: {order}")
        return order.order_id
    
    def execute_order(self, order: Order, current_price: float, timestamp: datetime) -> bool:
        """
        Execute an order at the current price.
        
        Args:
            order: Order to execute
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            True if order was executed, False otherwise
        """
        # Skip already filled orders
        if order.is_filled:
            return False
        
        # Determine fill price based on order type
        fill_price = None
        
        if order.order_type == OrderType.MARKET:
            # Market orders are filled at current price
            fill_price = current_price
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders are filled if price is favorable
            if (order.side == OrderSide.BUY and current_price <= order.price) or \
               (order.side == OrderSide.SELL and current_price >= order.price):
                fill_price = order.price
        
        elif order.order_type == OrderType.STOP:
            # Stop orders become market orders when triggered
            if (order.side == OrderSide.BUY and current_price >= order.stop_price) or \
               (order.side == OrderSide.SELL and current_price <= order.stop_price):
                fill_price = current_price
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop-limit orders become limit orders when triggered
            if (order.side == OrderSide.BUY and current_price >= order.stop_price) or \
               (order.side == OrderSide.SELL and current_price <= order.stop_price):
                # Then check if limit price is favorable
                if (order.side == OrderSide.BUY and current_price <= order.price) or \
                   (order.side == OrderSide.SELL and current_price >= order.price):
                    fill_price = order.price
        
        # If no fill price was determined, order wasn't executed
        if fill_price is None:
            return False
        
        # Calculate commission (to be implemented in transaction cost model)
        commission = 0.0
        
        # Calculate slippage (to be implemented in slippage model)
        slippage = 0.0
        
        # Adjust fill price for slippage
        if order.side == OrderSide.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage
        
        # Check if we have enough cash for buy orders
        if order.side == OrderSide.BUY:
            cost = order.quantity * fill_price + commission
            if cost > self.cash:
                self.logger.warning(f"Insufficient cash to execute order {order.order_id}")
                return False
        
        # Fill the order
        order.fill(fill_price, timestamp, commission=commission, slippage=slippage)
        
        # Update position
        self._update_position(order)
        
        # Record transaction
        self._record_transaction(order)
        
        self.logger.info(f"Executed order: {order}")
        return True
    
    def _update_position(self, order: Order) -> None:
        """
        Update the position based on a filled order.
        
        Args:
            order: Filled order
        """
        # Get or create position
        if order.instrument not in self.positions:
            self.positions[order.instrument] = Position(order.instrument)
        
        position = self.positions[order.instrument]
        
        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= order.fill_quantity * order.fill_price + order.commission
        else:
            self.cash += order.fill_quantity * order.fill_price - order.commission
        
        # Update position
        position.update(order)
    
    def _record_transaction(self, order: Order) -> None:
        """
        Record a transaction in the transaction history.
        
        Args:
            order: Filled order
        """
        transaction = {
            'timestamp': order.fill_timestamp,
            'instrument': order.instrument,
            'side': order.side.value,
            'quantity': order.fill_quantity,
            'price': order.fill_price,
            'value': order.fill_quantity * order.fill_price,
            'commission': order.commission,
            'slippage': order.slippage,
            'order_id': order.order_id
        }
        
        self.transactions.append(transaction)
    
    def update(self, timestamp: datetime, prices: Dict[str, float]) -> None:
        """
        Update the portfolio state with current prices.
        
        Args:
            timestamp: Current timestamp
            prices: Dictionary mapping instruments to current prices
        """
        self.current_timestamp = timestamp
        
        # Calculate portfolio value
        portfolio_value = self.cash
        
        for instrument, position in self.positions.items():
            if instrument in prices:
                portfolio_value += position.market_value(prices[instrument])
        
        # Record equity curve point
        self.equity_curve.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'total_value': portfolio_value
        })
        
        # Calculate return
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['total_value']
            current_value = portfolio_value
            period_return = (current_value / prev_value) - 1
            self.returns.append({
                'timestamp': timestamp,
                'return': period_return
            })
        
        # Calculate drawdown
        if len(self.equity_curve) > 0:
            peak = max(point['total_value'] for point in self.equity_curve)
            drawdown = (portfolio_value - peak) / peak
            self.drawdowns.append({
                'timestamp': timestamp,
                'drawdown': drawdown
            })
    
    def get_position(self, instrument: str) -> Optional[Position]:
        """
        Get the current position for an instrument.
        
        Args:
            instrument: Instrument identifier
            
        Returns:
            Position object or None if no position exists
        """
        return self.positions.get(instrument)
    
    def get_position_quantity(self, instrument: str) -> float:
        """
        Get the current position quantity for an instrument.
        
        Args:
            instrument: Instrument identifier
            
        Returns:
            Position quantity (0 if no position exists)
        """
        position = self.get_position(instrument)
        return position.quantity if position else 0.0
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate the current portfolio value.
        
        Args:
            prices: Dictionary mapping instruments to current prices
            
        Returns:
            Total portfolio value
        """
        portfolio_value = self.cash
        
        for instrument, position in self.positions.items():
            if instrument in prices:
                portfolio_value += position.market_value(prices[instrument])
        
        return portfolio_value
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve as a DataFrame.
        
        Returns:
            DataFrame with equity curve data
        """
        return pd.DataFrame(self.equity_curve).set_index('timestamp')
    
    def get_returns(self) -> pd.DataFrame:
        """
        Get the returns as a DataFrame.
        
        Returns:
            DataFrame with return data
        """
        return pd.DataFrame(self.returns).set_index('timestamp')
    
    def get_drawdowns(self) -> pd.DataFrame:
        """
        Get the drawdowns as a DataFrame.
        
        Returns:
            DataFrame with drawdown data
        """
        return pd.DataFrame(self.drawdowns).set_index('timestamp')
    
    def reset(self) -> None:
        """Reset the portfolio to its initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = []
        self.transactions = []
        self.equity_curve = []
        self.drawdowns = []
        self.returns = []
        self.current_timestamp = None