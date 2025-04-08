import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import logging
import time
import uuid
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

from .data.data_handler import DataHandler
from .portfolio import Portfolio, Order, OrderType, OrderSide
from .models.transaction_costs import TransactionCostModel, ZeroCostModel

class BacktestEngine:
    """
    Core backtesting engine for simulating trading strategies.
    """
    
    def __init__(
        self,
        data_handler: DataHandler,
        portfolio: Optional[Portfolio] = None,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        initial_capital: float = 100000.0,
        name: Optional[str] = None
    ):
        """
        Initialize the backtest engine.
        
        Args:
            data_handler: Data handler for historical data
            portfolio: Portfolio instance (created if None)
            transaction_cost_model: Transaction cost model (zero cost if None)
            initial_capital: Initial capital for the portfolio
            name: Name of the backtest
        """
        self.data_handler = data_handler
        self.portfolio = portfolio or Portfolio(initial_capital)
        self.transaction_cost_model = transaction_cost_model or ZeroCostModel()
        self.name = name or f"Backtest_{int(time.time())}"
        
        # Strategy and signal handling
        self.strategy = None
        self.signal_processor = None
        
        # Current state
        self.current_timestamp = None
        self.current_prices = {}
        self.is_running = False
        
        # Performance tracking
        self.performance_metrics = {}
        self.trade_history = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Event handling
        self.on_bar_close_handlers = []
    
    def set_strategy(self, strategy_func: Callable) -> None:
        """
        Set the strategy function.
        
        Args:
            strategy_func: Strategy function that generates signals
        """
        self.strategy = strategy_func
    
    def set_signal_processor(self, signal_processor: Callable) -> None:
        """
        Set the signal processor function.
        
        Args:
            signal_processor: Function that processes signals into orders
        """
        self.signal_processor = signal_processor
    
    def add_on_bar_close_handler(self, handler: Callable) -> None:
        """
        Add a handler function to be called on each bar close.
        
        Args:
            handler: Function to call on bar close
        """
        self.on_bar_close_handlers.append(handler)
    
    def run(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        instruments: List[str],
        timeframe: str,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Args:
            start_date: Start date for the backtest
            end_date: End date for the backtest
            instruments: List of instruments to include
            timeframe: Timeframe for the data
            strategy_params: Parameters to pass to the strategy
            
        Returns:
            Dictionary with backtest results
        """
        if not self.strategy:
            raise ValueError("Strategy function not set")
        
        if not self.signal_processor:
            # Use default signal processor if none is set
            self.signal_processor = self._default_signal_processor
        
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Load data
        self.logger.info(f"Loading data for {len(instruments)} instruments from {start_date} to {end_date}")
        self.data_handler.load_data(instruments, start_date, end_date, timeframe)
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Initialize state
        self.is_running = True
        self.current_prices = {}
        
        # Main backtest loop
        self.logger.info(f"Starting backtest: {self.name}")
        start_time = time.time()
        
        while self.data_handler.has_more_bars() and self.is_running:
            # Get current bars
            current_bars = self.data_handler.update_bars()
            
            if not current_bars:
                continue
            
            # Update current timestamp and prices
            self.current_timestamp = list(current_bars.values())[0].name
            self.current_prices = {symbol: bar['close'] for symbol, bar in current_bars.items()}
            
            # Execute pending orders
            self._execute_pending_orders()
            
            # Generate signals
            signals = self.strategy(
                timestamp=self.current_timestamp,
                prices=self.current_prices,
                portfolio=self.portfolio,
                data_handler=self.data_handler,
                params=strategy_params or {}
            )
            
            # Process signals into orders
            if signals:
                self._process_signals(signals)
            
            # Update portfolio state
            self.portfolio.update(self.current_timestamp, self.current_prices)
            
            # Call bar close handlers
            for handler in self.on_bar_close_handlers:
                handler(
                    timestamp=self.current_timestamp,
                    prices=self.current_prices,
                    portfolio=self.portfolio,
                    data_handler=self.data_handler
                )
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Backtest completed in {elapsed_time:.2f} seconds")
        
        return {
            'name': self.name,
            'start_date': start_date,
            'end_date': end_date,
            'instruments': instruments,
            'timeframe': timeframe,
            'initial_capital': self.portfolio.initial_capital,
            'final_portfolio_value': self.portfolio.get_portfolio_value(self.current_prices),
            'performance_metrics': self.performance_metrics,
            'equity_curve': self.portfolio.get_equity_curve(),
            'returns': self.portfolio.get_returns(),
            'drawdowns': self.portfolio.get_drawdowns(),
            'transactions': pd.DataFrame(self.portfolio.transactions)
        }
    
    def _execute_pending_orders(self) -> None:
        """Execute any pending orders at current prices."""
        for order in self.portfolio.orders[:]:  # Create a copy to avoid modification during iteration
            if not order.is_filled:
                # Get current price for the instrument
                if order.instrument in self.current_prices:
                    current_price = self.current_prices[order.instrument]
                    
                    # Apply transaction costs
                    commission = self.transaction_cost_model.calculate_commission(
                        price=current_price,
                        quantity=order.quantity
                    )
                    
                    slippage = self.transaction_cost_model.calculate_slippage(
                        price=current_price,
                        quantity=order.quantity,
                        is_buy=(order.side == OrderSide.BUY)
                    )
                    
                    # Execute the order
                    executed = self.portfolio.execute_order(
                        order=order,
                        current_price=current_price,
                        timestamp=self.current_timestamp
                    )
                    
                    if executed:
                        # Record the trade
                        self.trade_history.append({
                            'timestamp': self.current_timestamp,
                            'instrument': order.instrument,
                            'side': order.side.value,
                            'quantity': order.quantity,
                            'price': order.fill_price,
                            'commission': commission,
                            'slippage': slippage,
                            'order_id': order.order_id
                        })
    
    def _process_signals(self, signals: List[Dict[str, Any]]) -> None:
        """
        Process trading signals into orders.
        
        Args:
            signals: List of signal dictionaries
        """
        for signal in signals:
            # Process each signal using the signal processor
            orders = self.signal_processor(
                signal=signal,
                portfolio=self.portfolio,
                timestamp=self.current_timestamp,
                prices=self.current_prices
            )
            
            # Place the generated orders
            if orders:
                if isinstance(orders, list):
                    for order in orders:
                        self.portfolio.place_order(order)
                else:
                    self.portfolio.place_order(orders)
    
    def _default_signal_processor(
        self,
        signal: Dict[str, Any],
        portfolio: Portfolio,
        timestamp: datetime,
        prices: Dict[str, float]
    ) -> Optional[Order]:
        """
        Default signal processor that converts signals to market orders.
        
        Args:
            signal: Signal dictionary
            portfolio: Portfolio instance
            timestamp: Current timestamp
            prices: Current prices
            
        Returns:
            Order object or None
        """
        # Check if signal has required fields
        if 'instrument' not in signal or 'action' not in signal or 'quantity' not in signal:
            self.logger.warning(f"Invalid signal format: {signal}")
            return None
        
        instrument = signal['instrument']
        action = signal['action'].lower()
        quantity = float(signal['quantity'])
        
        # Skip if quantity is zero
        if quantity <= 0:
            return None
        
        # Determine order side
        if action in ['buy', 'long']:
            side = OrderSide.BUY
        elif action in ['sell', 'short']:
            side = OrderSide.SELL
        else:
            self.logger.warning(f"Unknown action in signal: {action}")
            return None
        
        # Create market order
        order = Order(
            instrument=instrument,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
            timestamp=timestamp
        )
        
        return order
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics for the backtest."""
        # Get equity curve and returns
        equity_curve = self.portfolio.get_equity_curve()
        returns = self.portfolio.get_returns()
        drawdowns = self.portfolio.get_drawdowns()
        
        if len(equity_curve) == 0 or len(returns) == 0:
            self.logger.warning("Not enough data to calculate performance metrics")
            return
        
        # Total return
        initial_value = self.portfolio.initial_capital
        final_value = equity_curve['total_value'].iloc[-1]
        total_return = (final_value / initial_value) - 1
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0
        
        # Volatility
        daily_returns = returns['return'].resample('D').sum()
        volatility = daily_returns.std() * (252 ** 0.5)  # Annualized
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        max_drawdown = drawdowns['drawdown'].min() if not drawdowns.empty else 0
        
        # Calmar ratio
        if max_drawdown < 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        # Win rate
        trades_df = pd.DataFrame(self.portfolio.transactions)
        if not trades_df.empty and 'value' in trades_df.columns:
            # Group by order_id to get individual trades
            trades = trades_df.groupby('order_id').agg({
                'timestamp': 'first',
                'instrument': 'first',
                'side': 'first',
                'quantity': 'sum',
                'price': 'mean',
                'value': 'sum',
                'commission': 'sum'
            })
            
            # Calculate profit/loss for each trade
            buy_trades = trades[trades['side'] == 'buy']['value'].sum()
            sell_trades = trades[trades['side'] == 'sell']['value'].sum()
            total_pnl = sell_trades - buy_trades - trades['commission'].sum()
            
            # Count winning and losing trades
            winning_trades = len(trades[trades['value'] > 0])
            total_trades = len(trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            total_pnl = 0
            win_rate = 0
            total_trades = 0
        
        # Store metrics
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_pnl': total_pnl
        }
    
    def plot_equity_curve(self, figsize=(12, 6)) -> None:
        """
        Plot the equity curve.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        equity_curve = self.portfolio.get_equity_curve()
        
        if equity_curve.empty:
            self.logger.warning("No equity curve data to plot")
            return
        
        plt.figure(figsize=figsize)
        plt.plot(equity_curve.index, equity_curve['total_value'], label='Portfolio Value')
        plt.title(f'Equity Curve - {self.name}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_drawdowns(self, figsize=(12, 6)) -> None:
        """
        Plot the drawdowns.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        drawdowns = self.portfolio.get_drawdowns()
        
        if drawdowns.empty:
            self.logger.warning("No drawdown data to plot")
            return
        
        plt.figure(figsize=figsize)
        plt.plot(drawdowns.index, drawdowns['drawdown'] * 100)
        plt.title(f'Drawdowns - {self.name}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, figsize=(12, 6)) -> None:
        """
        Plot the distribution of returns.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        returns = self.portfolio.get_returns()
        
        if returns.empty:
            self.logger.warning("No returns data to plot")
            return
        
        plt.figure(figsize=figsize)
        sns.histplot(returns['return'] * 100, kde=True)
        plt.title(f'Returns Distribution - {self.name}')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def print_performance_summary(self) -> None:
        """Print a summary of performance metrics."""
        if not self.performance_metrics:
            self.logger.warning("No performance metrics available")
            return
        
        print(f"\n{'=' * 50}")
        print(f"Performance Summary - {self.name}")
        print(f"{'=' * 50}")
        print(f"Initial Capital: ${self.portfolio.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${self.portfolio.get_portfolio_value(self.current_prices):,.2f}")
        print(f"Total Return: {self.performance_metrics['total_return'] * 100:.2f}%")
        print(f"Annualized Return: {self.performance_metrics['annualized_return'] * 100:.2f}%")
        print(f"Volatility (Annualized): {self.performance_metrics['volatility'] * 100:.2f}%")
        print(f"Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.performance_metrics['max_drawdown'] * 100:.2f}%")
        print(f"Calmar Ratio: {self.performance_metrics['calmar_ratio']:.2f}")
        print(f"Win Rate: {self.performance_metrics['win_rate'] * 100:.2f}%")
        print(f"Total Trades: {self.performance_metrics['total_trades']}")
        print(f"Total P&L: ${self.performance_metrics['total_pnl']:,.2f}")
        print(f"{'=' * 50}\n")