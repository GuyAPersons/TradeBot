"""
Live trading implementation for the backtesting framework.
"""
from typing import Dict, Any, List, Optional, Union, Callable
import logging
import time
import threading
import queue
import pandas as pd
from datetime import datetime, timedelta
import uuid
import os
import json

from .portfolio import Portfolio
from .data.data_handler import DataHandler
from .data_providers.providers_manager import DataProvidersManager
from .events.event import Event, EventType
from .events.market_event import MarketEvent
from .events.signal_event import SignalEvent
from .events.order_event import OrderEvent
from .events.fill_event import FillEvent
from .execution.execution_handler import ExecutionHandler
from .execution.broker_execution import BrokerExecutionHandler
from .strategies.strategy_manager import StrategyManager
from .strategies.base import Strategy
from .models.risk_models import RiskModel
from .utils.config import ConfigManager
from .utils.serialization import Serializer


class LiveTrader:
    """
    Live trading implementation that connects to real market data and executes trades.
    
    This class provides a framework for live trading using the same strategies
    and components as the backtesting system, but with real-time data and
    actual order execution through a broker.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        data_handler: Optional[DataHandler] = None,
        execution_handler: Optional[ExecutionHandler] = None,
        risk_model: Optional[RiskModel] = None,
        name: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the live trader.
        
        Args:
            config_path: Path to configuration file
            data_handler: DataHandler instance for market data
            execution_handler: ExecutionHandler for order execution
            risk_model: Model for risk management
            name: Name for the live trading session
            output_dir: Directory for saving results and logs
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration if provided
        self.config = None
        if config_path:
            self.config = ConfigManager.load_config(config_path)
            self._apply_config()
        
        # Set up components
        self.data_handler = data_handler
        self.data_providers_manager = None
        
        # Initialize data providers if not using external data handler
        if self.data_handler is None and self.config:
            self._initialize_data_providers()
        
        # Set up portfolio
        initial_capital = 100000.0
        if self.config and 'general' in self.config:
            initial_capital = self.config['general'].get('initial_capital', initial_capital)
        
        self.portfolio = Portfolio(initial_capital=initial_capital)
        
        # Set up execution handler
        if execution_handler is None:
            if self.config and 'execution' in self.config:
                # Create broker execution handler from config
                broker_config = self.config['execution'].get('broker', {})
                self.execution_handler = BrokerExecutionHandler(
                    api_key=broker_config.get('api_key'),
                    api_secret=broker_config.get('api_secret'),
                    base_url=broker_config.get('base_url'),
                    account_id=broker_config.get('account_id')
                )
            else:
                # Default to a simulated execution handler for testing
                self.logger.warning("No execution handler provided, using simulated execution")
                from .execution.simulated_execution import SimulatedExecutionHandler
                self.execution_handler = SimulatedExecutionHandler()
        else:
            self.execution_handler = execution_handler
        
        # Set up risk model
        self.risk_model = risk_model
        if self.risk_model is None and self.config and 'risk_management' in self.config:
            self._initialize_risk_model()
        
        # Set up session parameters
        self.name = name or f"LiveTrading_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = output_dir or os.path.join("live_results", self.name)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize strategy manager
        self.strategy_manager = StrategyManager()
        
        # Set up event queues
        self.market_event_queue = queue.Queue()
        self.signal_event_queue = queue.Queue()
        self.order_event_queue = queue.Queue()
        self.fill_event_queue = queue.Queue()
        
        # Trading control flags
        self.running = False
        self.paused = False
        
        # Trading threads
        self.market_thread = None
        self.strategy_thread = None
        self.execution_thread = None
        
        # Trading statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'market_events': 0,
            'signal_events': 0,
            'order_events': 0,
            'fill_events': 0
        }
        
        # Trading log
        self.trade_log = []
        
        # Set up logging to file
        self._setup_file_logging()
        
        self.logger.info(f"Initialized live trader: {self.name}")
    
    def _apply_config(self):
        """Apply configuration settings."""
        if not self.config:
            return
        
        # Apply general settings
        if 'general' in self.config:
            general_config = self.config['general']
            self.name = general_config.get('name', self.name)
            self.output_dir = general_config.get('output_dir', self.output_dir)
        
        # Log configuration
        self.logger.info(f"Loaded configuration for live trading: {self.name}")
    
    def _initialize_data_providers(self):
        """Initialize data providers based on configuration."""
        if not self.config or 'data_providers' not in self.config:
            self.logger.warning("No data providers configured")
            return
        
        # Initialize data providers manager
        self.data_providers_manager = DataProvidersManager()
        
        # Register data providers from configuration
        for provider_config in self.config['data_providers']:
            provider_type = provider_config.get('type')
            provider_name = provider_config.get('name')
            provider_params = provider_config.get('params', {})
            
            if provider_type and provider_name:
                # Create and register the provider
                from .data_providers.base_factory import DataProviderFactory
                provider = DataProviderFactory.create(
                    provider_type, 
                    **provider_params
                )
                self.data_providers_manager.register_provider(provider_name, provider)
                self.logger.info(f"Registered data provider: {provider_name} ({provider_type})")
        
        # Create data handler from providers
        if self.data_providers_manager.has_providers():
            self.data_handler = DataHandler(self.data_providers_manager)
            self.logger.info("Created data handler with registered providers")
    
    def _initialize_risk_model(self):
        """Initialize risk model based on configuration."""
        if not self.config or 'risk_management' not in self.config:
            return
        
        risk_config = self.config['risk_management']
        model_type = risk_config.get('position_size_model', {}).get('type')
        model_params = risk_config.get('position_size_model', {}).get('params', {})
        
        if model_type:
            from .models.risk_models import (
                FixedPositionSizeModel, PercentagePositionSizeModel, 
                KellyPositionSizeModel
            )
            
            if model_type == 'fixed':
                self.risk_model = FixedPositionSizeModel(**model_params)
            elif model_type == 'percentage':
                self.risk_model = PercentagePositionSizeModel(**model_params)
            elif model_type == 'kelly':
                self.risk_model = KellyPositionSizeModel(**model_params)
            else:
                self.logger.warning(f"Unknown risk model type: {model_type}")
                return
            
            self.logger.info(f"Initialized risk model: {model_type}")
    
    def _setup_file_logging(self):
        """Set up logging to file."""
        log_dir = os.path.join(self.output_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"{self.name}.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Set up file logging to {log_file}")
    
    def add_strategy(self, strategy: Union[Strategy, str], **kwargs):
        """
        Add a strategy to the live trader.
        
        Args:
            strategy: Strategy instance or strategy name to load from config
            **kwargs: Additional parameters for the strategy
        """
        if isinstance(strategy, str):
            # Load strategy from configuration
            if not self.config or 'strategies' not in self.config:
                raise ValueError(f"Cannot load strategy '{strategy}': No strategies in config")
            
            # Find strategy in config
            strategy_config = None
            for s_config in self.config['strategies']:
                if s_config.get('name') == strategy:
                    strategy_config = s_config
                    break
            
            if strategy_config is None:
                raise ValueError(f"Strategy '{strategy}' not found in configuration")
            
            # Create strategy instance
            strategy_class = self._get_strategy_class(strategy_config['class'])
            strategy_params = {**strategy_config.get('params', {}), **kwargs}
            strategy_instance = strategy_class(**strategy_params)
            
            self.logger.info(f"Created strategy from config: {strategy}")
            strategy = strategy_instance
        
        # Register strategy with manager
        self.strategy_manager.register_strategy(strategy)
        self.logger.info(f"Added strategy: {strategy.__class__.__name__}")
    
    def _get_strategy_class(self, class_path: str):
        """
        Dynamically import a strategy class.
        
        Args:
            class_path: Full path to the strategy class
            
        Returns:
            Strategy class
        """
        try:
            import importlib
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to import strategy class {class_path}: {e}")
            raise ImportError(f"Could not import strategy class: {class_path}")
    
    def start(self):
        """
        Start the live trading session.
        """
        if self.running:
            self.logger.warning("Live trading already running")
            return
        
        # Check if we have strategies
        if not self.strategy_manager.has_strategies():
            self.logger.error("No strategies added to live trader")
            raise ValueError("No strategies added to live trader")
        
        # Check if we have a data handler
        if self.data_handler is None:
            self.logger.error("No data handler available")
            raise ValueError("No data handler available")
        
        self.logger.info(f"Starting live trading: {self.name}")
        
        # Reset statistics
        self.stats['start_time'] = datetime.now()
        self.stats['market_events'] = 0
        self.stats['signal_events'] = 0
        self.stats['order_events'] = 0
        self.stats['fill_events'] = 0
        
        # Set running flag
        self.running = True
        self.paused = False
        
        # Start threads
        self.market_thread = threading.Thread(target=self._market_data_loop)
        self.strategy_thread = threading.Thread(target=self._strategy_loop)
        self.execution_thread = threading.Thread(target=self._execution_loop)
        
        self.market_thread.daemon = True
        self.strategy_thread.daemon = True
        self.execution_thread.daemon = True
        
        self.market_thread.start()
        self.strategy_thread.start()
        self.execution_thread.start()
        
        self.logger.info("Live trading threads started")
    
    def stop(self):
        """
        Stop the live trading session.
        """
        if not self.running:
            self.logger.warning("Live trading not running")
            return
        
        self.logger.info("Stopping live trading")
        
        # Set flags
        self.running = False
        
        # Wait for threads to finish
        if self.market_thread and self.market_thread.is_alive():
            self.market_thread.join(timeout=5.0)
        
        if self.strategy_thread and self.strategy_thread.is_alive():
            self.strategy_thread.join(timeout=5.0)
        
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5.0)
        
        # Update statistics
        self.stats['end_time'] = datetime.now()
        
        # Save results
        self._save_results()
        
        self.logger.info("Live trading stopped")
    
    def pause(self):
        """
        Pause the live trading session.
        """
        if not self.running:
            self.logger.warning("Live trading not running")
            return
        
        if self.paused:
            self.logger.warning("Live trading already paused")
            return
        
        self.paused = True
        self.logger.info("Live trading paused")
    
    def resume(self):
        """
        Resume the live trading session.
        """
        if not self.running:
            self.logger.warning("Live trading not running")
            return
        
        if not self.paused:
            self.logger.warning("Live trading not paused")
            return
        
        self.paused = False
        self.logger.info("Live trading resumed")
    
    def _market_data_loop(self):
        """
        Main loop for fetching market data.
        """
        self.logger.info("Market data loop started")
        
        # Initialize last update time
        last_update = datetime.now()
        
        # Define update interval (in seconds)
        update_interval = 1.0  # 1 second by default
        if self.config and 'data_providers' in self.config:
            for provider in self.config['data_providers']:
                if 'update_interval' in provider.get('params', {}):
                    update_interval = provider['params']['update_interval']
                    break
        
        while self.running:
            # Skip if paused
            if self.paused:
                time.sleep(0.1)
                continue
            
            # Check if it's time to update
            current_time = datetime.now()
            if (current_time - last_update).total_seconds() < update_interval:
                time.sleep(0.1)
                continue
            
            try:
                # Fetch latest market data
                market_data = self.data_handler.get_latest_data()
                
                if market_data is not None and not market_data.empty:
                    # Create market event
                    market_event = MarketEvent(
                        timestamp=current_time,
                        data=market_data
                    )
                    
                    # Put event in queue
                    self.market_event_queue.put(market_event)
                    
                    # Update statistics
                    self.stats['market_events'] += 1
                    
                    # Log event
                    self.logger.debug(f"Market event created: {len(market_data)} data points")
                
                # Update last update time
                last_update = current_time
                
            except Exception as e:
                self.logger.error(f"Error in market data loop: {e}")
                time.sleep(1.0)  # Wait a bit before retrying
        
        self.logger.info("Market data loop stopped")
    
    def _strategy_loop(self):
        """
        Main loop for strategy processing.
        """
        self.logger.info("Strategy loop started")
        
        while self.running:
            # Skip if paused
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # Get market event from queue with timeout
                try:
                    market_event = self.market_event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process market event with all strategies
                for strategy in self.strategy_manager.get_strategies():
                    # Update strategy with market data
                    strategy.update_data(market_event.data)
                    
                    # Generate signals
                    signals = strategy.generate_signals()
                    
                    # Process signals
                    if signals:
                        for signal in signals:
                            # Apply risk model if available
                            if self.risk_model:
                                signal = self.risk_model.apply(signal, self.portfolio)
                            
                            # Create signal event
                            signal_event = SignalEvent(
                                timestamp=datetime.now(),
                                symbol=signal['symbol'],
                                signal_type=signal['signal_type'],
                                strength=signal.get('strength', 1.0),
                                strategy=strategy.name
                            )
                            
                            # Put event in queue
                            self.signal_event_queue.put(signal_event)
                            
                            # Update statistics
                            self.stats['signal_events'] += 1
                            
                            # Log event
                            self.logger.info(f"Signal generated: {signal_event.symbol} - {signal_event.signal_type}")
                
                # Mark market event as processed
                self.market_event_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in strategy loop: {e}")
                time.sleep(0.1)  # Wait a bit before continuing
        
        self.logger.info("Strategy loop stopped")
    
    def _execution_loop(self):
        """
        Main loop for order execution.
        """
        self.logger.info("Execution loop started")
        
        while self.running:
            # Skip if paused
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # Process signal events
                self._process_signal_events()
                
                # Process order events
                self._process_order_events()
                
                # Process fill events
                self._process_fill_events()
                
                # Small delay to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                time.sleep(0.1)  # Wait a bit before continuing
        
        self.logger.info("Execution loop stopped")
    
    def _process_signal_events(self):
        """Process signal events and generate orders."""
        try:
            # Get signal event from queue with timeout
            try:
                signal_event = self.signal_event_queue.get(timeout=0.01)
            except queue.Empty:
                return
            
            # Convert signal to order based on portfolio state
            order_event = self.portfolio.generate_order(signal_event)
            
            if order_event:
                # Put order event in queue
                self.order_event_queue.put(order_event)
                
                # Update statistics
                self.stats['order_events'] += 1
                
                # Log event
                self.logger.info(f"Order generated: {order_event.symbol} - {order_event.order_type} - {order_event.quantity}")
            
            # Mark signal event as processed
            self.signal_event_queue.task_done()
            
        except Exception as e:
            self.logger.error(f"Error processing signal event: {e}")
    
    def _process_order_events(self):
        """Process order events and execute orders."""
        try:
            # Get order event from queue with timeout
            try:
                order_event = self.order_event_queue.get(timeout=0.01)
            except queue.Empty:
                return
            
            # Execute order
            fill_event = self.execution_handler.execute_order(order_event)
            
            if fill_event:
                # Put fill event in queue
                self.fill_event_queue.put(fill_event)
                
                # Update statistics
                self.stats['fill_events'] += 1
                
                # Log event
                self.logger.info(f"Order executed: {fill_event.symbol} - {fill_event.quantity} @ {fill_event.price}")
            
            # Mark order event as processed
            self.order_event_queue.task_done()
            
        except Exception as e:
            self.logger.error(f"Error processing order event: {e}")
    
    def _process_fill_events(self):
        """Process fill events and update portfolio."""
        try:
            # Get fill event from queue with timeout
            try:
                fill_event = self.fill_event_queue.get(timeout=0.01)
            except queue.Empty:
                return
            
            # Update portfolio
            self.portfolio.update_from_fill(fill_event)
            
            # Record trade in log
            trade_record = {
                'timestamp': fill_event.timestamp,
                'symbol': fill_event.symbol,
                'quantity': fill_event.quantity,
                'price': fill_event.price,
                'commission': fill_event.commission,
                'direction': 'BUY' if fill_event.quantity > 0 else 'SELL',
                'value': abs(fill_event.quantity * fill_event.price),
                'exchange': fill_event.exchange
            }
            self.trade_log.append(trade_record)
            
            # Update trade statistics
            self.stats['total_trades'] += 1
            
            # Calculate PnL if it's a closing trade
            position = self.portfolio.get_position(fill_event.symbol)
            if position and position.quantity == 0:  # Position was closed
                pnl = position.realized_pnl
                self.stats['total_pnl'] += pnl
                
                if pnl > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1
                
                self.logger.info(f"Position closed: {fill_event.symbol} - PnL: {pnl:.2f}")
            
            # Mark fill event as processed
            self.fill_event_queue.task_done()
            
        except Exception as e:
            self.logger.error(f"Error processing fill event: {e}")
    
    def _save_results(self):
        """Save trading results to disk."""
        # Create results directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Save trade log
        if self.trade_log:
            trade_df = pd.DataFrame(self.trade_log)
            trade_path = os.path.join(self.output_dir, 'trade_log.csv')
            trade_df.to_csv(trade_path, index=False)
            self.logger.info(f"Saved trade log to {trade_path}")
        
        # Save portfolio state
        portfolio_state = self.portfolio.get_state()
        portfolio_path = os.path.join(self.output_dir, 'portfolio_state.json')
        with open(portfolio_path, 'w') as f:
            json.dump(portfolio_state, f, indent=4, default=self._json_serialize)
        self.logger.info(f"Saved portfolio state to {portfolio_path}")
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=4, default=self._json_serialize)
        self.logger.info(f"Saved statistics to {stats_path}")
        
        # Save portfolio history
        portfolio_history = self.portfolio.get_history()
        if not portfolio_history.empty:
            history_path = os.path.join(self.output_dir, 'portfolio_history.csv')
            portfolio_history.to_csv(history_path)
            self.logger.info(f"Saved portfolio history to {history_path}")
    
    def _json_serialize(self, obj):
        """Helper method to serialize objects to JSON."""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, (float, int)) and (pd.isna(obj) or pd.isinf(obj)):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the live trader.
        
        Returns:
            Dictionary containing status information
        """
        return {
            'name': self.name,
            'running': self.running,
            'paused': self.paused,
            'start_time': self.stats['start_time'],
            'current_time': datetime.now(),
            'uptime': str(datetime.now() - self.stats['start_time']) if self.stats['start_time'] else None,
            'portfolio_value': self.portfolio.get_value(),
            'cash': self.portfolio.cash,
            'positions': len(self.portfolio.positions),
            'total_trades': self.stats['total_trades'],
            'winning_trades': self.stats['winning_trades'],
            'losing_trades': self.stats['losing_trades'],
            'win_rate': self.stats['winning_trades'] / self.stats['total_trades'] if self.stats['total_trades'] > 0 else 0,
            'total_pnl': self.stats['total_pnl'],
            'market_events': self.stats['market_events'],
            'signal_events': self.stats['signal_events'],
            'order_events': self.stats['order_events'],
            'fill_events': self.stats['fill_events']
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current portfolio.
        
        Returns:
            Dictionary containing portfolio summary
        """
        positions = []
        for symbol, position in self.portfolio.positions.items():
            positions.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'total_pnl': position.total_pnl
            })
        
        return {
            'cash': self.portfolio.cash,
            'equity': self.portfolio.get_value(),
            'positions': positions,
            'timestamp': datetime.now()
        }
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get the trade history.
        
        Returns:
            DataFrame containing trade history
        """
        if not self.trade_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_log)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get the portfolio history.
        
        Returns:
            DataFrame containing portfolio history
        """
        return self.portfolio.get_history()
    
    def load_state(self, state_path: str):
        """
        Load portfolio state from a file.
        
        Args:
            state_path: Path to portfolio state file
        """
        if not os.path.exists(state_path):
            self.logger.error(f"State file not found: {state_path}")
            return False
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            # Restore portfolio state
            self.portfolio.set_state(state)
            
            self.logger.info(f"Loaded portfolio state from {state_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False
    
    def save_state(self, state_path: Optional[str] = None) -> str:
        """
        Save portfolio state to a file.
        
        Args:
            state_path: Path to save state file (optional)
            
        Returns:
            Path where state was saved
        """
        if state_path is None:
            state_path = os.path.join(self.output_dir, 'portfolio_state.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
        
        # Get portfolio state
        state = self.portfolio.get_state()
        
        # Save state to file
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4, default=self._json_serialize)
        
        self.logger.info(f"Saved portfolio state to {state_path}")
        return state_path