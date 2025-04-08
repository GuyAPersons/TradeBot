"""
High-level backtest runner that integrates all components of the backtesting framework.
"""
from typing import Dict, Any, List, Optional, Union, Callable, Type, Set
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import uuid
import os
import json
import importlib
from pathlib import Path

# Core components
from .engine import BacktestEngine
from .portfolio import Portfolio

# Data components
from .data.data_handler import DataHandler
from .data.data_preprocessor import DataPreprocessor
from .data_providers.providers_manager import DataProvidersManager
from .data_providers.base_factory import DataProviderFactory
from .data_providers.utils.data_normalizer import DataNormalizer
from .data_providers.utils.rate_limiter import RateLimiter

# Event system
from .events.event import Event, EventType
from .events.market_event import MarketEvent
from .events.signal_event import SignalEvent, SignalType
from .events.order_event import OrderEvent, OrderType, OrderDirection
from .events.fill_event import FillEvent

# Execution
from .execution.execution_handler import ExecutionHandler
from .execution.simulated_execution import SimulatedExecutionHandler
from .execution.broker_execution import BrokerExecutionHandler

# Models
from .models.transaction_costs import (
    TransactionCostModel, FixedCostModel, 
    PercentageCostModel, TieredCostModel
)
from .models.slippage import (
    SlippageModel, FixedSlippageModel, 
    PercentageSlippageModel, VolumeBasedSlippageModel
)
from .models.risk_models import (
    RiskModel, FixedPositionSizeModel, 
    PercentagePositionSizeModel, KellyPositionSizeModel
)

# Metrics
from .metrics.performance import PerformanceMetrics
from .metrics.risk import RiskMetrics
from .metrics.statistical import StatisticalAnalysis

# Visualization
from .visualization.performance_plots import PerformancePlots
from .visualization.equity_curves import EquityCurvePlotter
from .visualization.drawdown_analysis import DrawdownAnalyzer

# Optimization
from .optimization.parameter_grid import ParameterGridOptimizer
from .optimization.genetic_algorithm import GeneticAlgorithmOptimizer
from .optimization.bayesian_optimization import BayesianOptimizer

# Strategies
from .strategies.strategy_manager import StrategyManager
from .strategies.base import Strategy

# Utilities
from .utils.config import ConfigManager
from .utils.serialization import Serializer
from .utils.validation import Validator


class BacktestRunner:
    """
    High-level backtest runner that coordinates all components.
    
    This class provides a simplified interface for setting up and running
    backtests, handling the coordination between data providers, strategies,
    portfolio management, execution, analysis, and visualization.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        data_handler: Optional[DataHandler] = None,
        initial_capital: float = 100000.0,
        execution_handler: Optional[ExecutionHandler] = None,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        slippage_model: Optional[SlippageModel] = None,
        risk_model: Optional[RiskModel] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        name: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the backtest runner.
        
        Args:
            config_path: Path to configuration file
            data_handler: DataHandler instance for market data
            initial_capital: Initial capital for the portfolio
            execution_handler: ExecutionHandler for order execution
            transaction_cost_model: Model for transaction costs
            slippage_model: Model for price slippage
            risk_model: Model for risk management
            start_date: Start date for the backtest
            end_date: End date for the backtest
            name: Name for the backtest
            output_dir: Directory for saving results
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration if provided
        self.config = None
        if config_path:
            self.config = ConfigManager.load_config(config_path)
            self._apply_config()
        
        # Set up data components
        self.data_handler = data_handler
        self.data_preprocessor = None
        self.data_providers_manager = None
        
        # Initialize data providers if not using external data handler
        if self.data_handler is None and self.config:
            self._initialize_data_providers()
        
        # Set up portfolio
        self.portfolio = Portfolio(
            initial_capital=initial_capital if self.config is None 
            else self.config.get('initial_capital', initial_capital)
        )
        
        # Set up models
        self.transaction_cost_model = transaction_cost_model
        self.slippage_model = slippage_model
        self.risk_model = risk_model
        
        # Create execution handler if not provided
        if execution_handler is None:
            self.execution_handler = SimulatedExecutionHandler(
                slippage_model=self.slippage_model,
                transaction_cost_model=self.transaction_cost_model,
                data_handler=self.data_handler
            )
        else:
            self.execution_handler = execution_handler
        
        # Set up backtest parameters
        self.start_date = start_date
        self.end_date = end_date
        self.name = name or f"Backtest_{uuid.uuid4().hex[:8]}"
        self.output_dir = output_dir or os.path.join("results", self.name)
        
        # Initialize the engine
        self.engine = BacktestEngine(
            data_handler=self.data_handler,
            portfolio=self.portfolio,
            execution_handler=self.execution_handler
        )
        
        # Initialize strategy manager
        self.strategy_manager = StrategyManager()
        
        # Store results
        self.results = None
        self.metrics = {
            'performance': None,
            'risk': None,
            'statistical': None
        }
        
        # Visualization components
        self.visualizers = {
            'performance': None,
            'equity': None,
            'drawdown': None
        }
        
        # Optimization components
        self.optimizers = {
            'grid': None,
            'genetic': None,
            'bayesian': None
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _apply_config(self):
        """Apply configuration settings."""
        if not self.config:
            return
        
        # Apply general settings
        if 'general' in self.config:
            general_config = self.config['general']
            self.name = general_config.get('name', self.name)
            self.output_dir = general_config.get('output_dir', self.output_dir)
            
            # Apply date range if specified
            if 'start_date' in general_config:
                self.start_date = datetime.fromisoformat(general_config['start_date'])
            if 'end_date' in general_config:
                self.end_date = datetime.fromisoformat(general_config['end_date'])
        
        # Log configuration
        self.logger.info(f"Loaded configuration for backtest: {self.name}")
    
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
            
            # Initialize data preprocessor if needed
            if 'data_preprocessing' in self.config:
                self.data_preprocessor = DataPreprocessor(
                    **self.config['data_preprocessing']
                )
                self.logger.info("Initialized data preprocessor")
    
    def add_strategy(self, strategy: Union[Strategy, str], **kwargs):
        """
        Add a strategy to the backtest.
        
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
        
        # Register strategy with manager and engine
        self.strategy_manager.register_strategy(strategy)
        self.engine.add_strategy(strategy)
        self.logger.info(f"Added strategy: {strategy.__class__.__name__}")
    
    def _get_strategy_class(self, class_path: str) -> Type[Strategy]:
        """
        Dynamically import a strategy class.
        
        Args:
            class_path: Full path to the strategy class (e.g., 'backtesting.strategies.trend_following.MovingAverageCrossover')
            
        Returns:
            Strategy class
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to import strategy class {class_path}: {e}")
            raise ImportError(f"Could not import strategy class: {class_path}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Returns:
            Dictionary containing backtest results
        """
        self.logger.info(f"Starting backtest: {self.name}")
        
        # Check if we have strategies
        if not self.strategy_manager.has_strategies():
            self.logger.error("No strategies added to backtest")
            raise ValueError("No strategies added to backtest")
        
        # Preprocess data if needed
        if self.data_preprocessor and self.data_handler:
            self.data_preprocessor.process(self.data_handler)
            self.logger.info("Preprocessed data for backtest")
        
        # Run the backtest
        self.results = self.engine.run(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Save results
        self._save_results()
        
        self.logger.info(f"Backtest completed: {self.name}")
        
        return self.get_results()
    
    def _calculate_metrics(self):
        """Calculate performance, risk, and statistical metrics."""
        if self.results is None:
            self.logger.warning("No results available for metrics calculation")
            return
        
        # Extract portfolio history
        portfolio_history = self.results.get('portfolio_history', pd.DataFrame())
        if portfolio_history.empty:
            self.logger.warning("Empty portfolio history, cannot calculate metrics")
            return
        
        # Calculate performance metrics
        self.metrics['performance'] = PerformanceMetrics(portfolio_history)
        
        # Calculate risk metrics
        self.metrics['risk'] = RiskMetrics(portfolio_history)
        
        # Calculate statistical metrics
        self.metrics['statistical'] = StatisticalAnalysis(portfolio_history)
        
        self.logger.info("Calculated performance, risk, and statistical metrics")
    
    def _generate_visualizations(self):
        """Generate visualizations for backtest results."""
        if self.results is None:
            self.logger.warning("No results available for visualization")
            return
        
        # Extract portfolio history
        portfolio_history = self.results.get('portfolio_history', pd.DataFrame())
        if portfolio_history.empty:
            self.logger.warning("Empty portfolio history, cannot generate visualizations")
            return
        
        # Create visualization components
        self.visualizers['performance'] = PerformancePlots(
            portfolio_history, 
            self.metrics['performance'],
            self.metrics['risk']
        )
        
        self.visualizers['equity'] = EquityCurvePlotter(portfolio_history)
        
        self.visualizers['drawdown'] = DrawdownAnalyzer(portfolio_history)
        
        # Generate and save plots
        output_path = os.path.join(self.output_dir, 'visualizations')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Performance plots
        self.visualizers['performance'].plot_returns_distribution(
            save_path=os.path.join(output_path, 'returns_distribution.png')
        )
        self.visualizers['performance'].plot_rolling_sharpe(
            save_path=os.path.join(output_path, 'rolling_sharpe.png')
        )
        
        # Equity curve
        self.visualizers['equity'].plot_equity_curve(
            save_path=os.path.join(output_path, 'equity_curve.png')
        )
        self.visualizers['equity'].plot_returns(
            save_path=os.path.join(output_path, 'returns.png')
        )
        
        # Drawdown analysis
        self.visualizers['drawdown'].plot_drawdowns(
            save_path=os.path.join(output_path, 'drawdowns.png')
        )
        self.visualizers['drawdown'].plot_underwater(
            save_path=os.path.join(output_path, 'underwater.png')
        )
        
        self.logger.info(f"Generated visualizations in {output_path}")
    
    def _save_results(self):
        """Save backtest results to disk."""
        if self.results is None:
            self.logger.warning("No results available to save")
            return
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Save portfolio history
        portfolio_history = self.results.get('portfolio_history', pd.DataFrame())
        if not portfolio_history.empty:
            portfolio_path = os.path.join(self.output_dir, 'portfolio_history.csv')
            portfolio_history.to_csv(portfolio_path)
            self.logger.info(f"Saved portfolio history to {portfolio_path}")
        
        # Save trade history
        trade_history = self.results.get('trade_history', pd.DataFrame())
        if not trade_history.empty:
            trades_path = os.path.join(self.output_dir, 'trade_history.csv')
            trade_history.to_csv(trades_path)
            self.logger.info(f"Saved trade history to {trades_path}")
        
        # Save metrics
        metrics_data = {}
        
        if self.metrics['performance'] is not None:
            metrics_data['performance'] = self.metrics['performance'].get_metrics()
        
        if self.metrics['risk'] is not None:
            metrics_data['risk'] = self.metrics['risk'].get_metrics()
        
        if self.metrics['statistical'] is not None:
            metrics_data['statistical'] = self.metrics['statistical'].get_metrics()
        
        if metrics_data:
            metrics_path = os.path.join(self.output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=4, default=self._json_serialize)
            self.logger.info(f"Saved metrics to {metrics_path}")
        
        # Save backtest summary
        summary = self.get_summary()
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4, default=self._json_serialize)
        self.logger.info(f"Saved backtest summary to {summary_path}")
        
        # Save full results using serializer
        results_path = os.path.join(self.output_dir, 'full_results.pkl')
        Serializer.save(self.results, results_path)
        self.logger.info(f"Saved full results to {results_path}")
    
    def _json_serialize(self, obj):
        """Helper method to serialize objects to JSON."""
        if isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the backtest results.
        
        Returns:
            Dictionary containing backtest results
        """
        if self.results is None:
            self.logger.warning("No results available")
            return {}
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the backtest results.
        
        Returns:
            Dictionary containing backtest summary
        """
        summary = {
            'name': self.name,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'initial_capital': self.portfolio.initial_capital,
            'strategies': [s.name for s in self.strategy_manager.get_strategies()],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add metrics if available
        if self.metrics['performance'] is not None:
            perf_metrics = self.metrics['performance'].get_metrics()
            summary.update({
                'final_equity': perf_metrics.get('final_equity'),
                'total_return': perf_metrics.get('total_return'),
                'annualized_return': perf_metrics.get('annualized_return'),
                'sharpe_ratio': perf_metrics.get('sharpe_ratio')
            })
        
        if self.metrics['risk'] is not None:
            risk_metrics = self.metrics['risk'].get_metrics()
            summary.update({
                'max_drawdown': risk_metrics.get('max_drawdown'),
                'volatility': risk_metrics.get('volatility'),
                'sortino_ratio': risk_metrics.get('sortino_ratio')
            })
        
        # Add trade statistics if available
        if 'trade_history' in self.results and not self.results['trade_history'].empty:
            trade_df = self.results['trade_history']
            summary.update({
                'total_trades': len(trade_df),
                'winning_trades': len(trade_df[trade_df['pnl'] > 0]),
                'losing_trades': len(trade_df[trade_df['pnl'] <= 0]),
                'win_rate': len(trade_df[trade_df['pnl'] > 0]) / len(trade_df) if len(trade_df) > 0 else 0,
                'avg_profit': trade_df[trade_df['pnl'] > 0]['pnl'].mean() if len(trade_df[trade_df['pnl'] > 0]) > 0 else 0,
                'avg_loss': trade_df[trade_df['pnl'] <= 0]['pnl'].mean() if len(trade_df[trade_df['pnl'] <= 0]) > 0 else 0,
                'profit_factor': (trade_df[trade_df['pnl'] > 0]['pnl'].sum() / 
                                 -trade_df[trade_df['pnl'] < 0]['pnl'].sum() 
                                 if trade_df[trade_df['pnl'] < 0]['pnl'].sum() != 0 else float('inf'))
            })
        
        return summary
    
    def optimize(
        self, 
        strategy_name: str, 
        param_grid: Dict[str, List[Any]], 
        method: str = 'grid',
        metric: str = 'sharpe_ratio',
        maximize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters.
        
        Args:
            strategy_name: Name of the strategy to optimize
            param_grid: Dictionary of parameter names and possible values
            method: Optimization method ('grid', 'genetic', 'bayesian')
            metric: Metric to optimize
            maximize: Whether to maximize (True) or minimize (False) the metric
            **kwargs: Additional parameters for the optimizer
            
        Returns:
            Dictionary containing optimization results
        """
        self.logger.info(f"Starting {method} optimization for {strategy_name}")
        
        # Get the strategy
        strategy = self.strategy_manager.get_strategy(strategy_name)
        if strategy is None:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        # Create optimizer based on method
        optimizer = None
        if method == 'grid':
            optimizer = ParameterGridOptimizer(
                strategy=strategy,
                param_grid=param_grid,
                metric=metric,
                maximize=maximize,
                **kwargs
            )
        elif method == 'genetic':
            optimizer = GeneticAlgorithmOptimizer(
                strategy=strategy,
                param_grid=param_grid,
                metric=metric,
                maximize=maximize,
                **kwargs
            )
        elif method == 'bayesian':
            optimizer = BayesianOptimizer(
                strategy=strategy,
                param_grid=param_grid,
                metric=metric,
                maximize=maximize,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Run optimization
        optimization_results = optimizer.optimize(
            data_handler=self.data_handler,
            portfolio=self.portfolio,
            execution_handler=self.execution_handler,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Save optimization results
        opt_dir = os.path.join(self.output_dir, 'optimization')
        if not os.path.exists(opt_dir):
            os.makedirs(opt_dir)
        
        opt_path = os.path.join(opt_dir, f'{strategy_name}_{method}_optimization.json')
        with open(opt_path, 'w') as f:
            json.dump(optimization_results, f, indent=4, default=self._json_serialize)
        
        self.logger.info(f"Optimization completed. Results saved to {opt_path}")
        
        # Update strategy with best parameters if requested
        if kwargs.get('update_strategy', True):
            best_params = optimization_results.get('best_params', {})
            if best_params:
                for param, value in best_params.items():
                    setattr(strategy, param, value)
                self.logger.info(f"Updated {strategy_name} with best parameters: {best_params}")
        
        return optimization_results
    
    def generate_report(self, template_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive HTML report for the backtest.
        
        Args:
            template_path: Path to a custom HTML template
            
        Returns:
            Path to the generated report
        """
        # This is a placeholder for a more sophisticated reporting system
        # In a real implementation, we would use a templating engine like Jinja2
        
        report_dir = os.path.join(self.output_dir, 'reports')
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        report_path = os.path.join(report_dir, 'backtest_report.html')
        
        # Generate a simple HTML report
        with open(report_path, 'w') as f:
            f.write(f"<html><head><title>Backtest Report: {self.name}</title></head>\n")
            f.write("<body>\n")
            f.write(f"<h1>Backtest Report: {self.name}</h1>\n")
            
            # Add summary section
            f.write("<h2>Summary</h2>\n")
            summary = self.get_summary()
            f.write("<table border='1'>\n")
            for key, value in summary.items():
                f.write(f"<tr><td>{key}</td><td>{value}</td></tr>\n")
            f.write("</table>\n")
            
            # Add performance metrics
            if self.metrics['performance'] is not None:
                f.write("<h2>Performance Metrics</h2>\n")
                perf_metrics = self.metrics['performance'].get_metrics()
                f.write("<table border='1'>\n")
                for key, value in perf_metrics.items():
                    f.write(f"<tr><td>{key}</td><td>{value}</td></tr>\n")
                f.write("</table>\n")
            
            # Add risk metrics
            if self.metrics['risk'] is not None:
                f.write("<h2>Risk Metrics</h2>\n")
                risk_metrics = self.metrics['risk'].get_metrics()
                f.write("<table border='1'>\n")
                for key, value in risk_metrics.items():
                    f.write(f"<tr><td>{key}</td><td>{value}</td></tr>\n")
                f.write("</table>\n")
            
            # Add images
            f.write("<h2>Charts</h2>\n")
            vis_dir = os.path.join(self.output_dir, 'visualizations')
            if os.path.exists(vis_dir):
                for img_file in os.listdir(vis_dir):
                    if img_file.endswith('.png'):
                        img_path = os.path.join('visualizations', img_file)
                        f.write(f"<h3>{img_file.replace('.png', '').replace('_', ' ').title()}</h3>\n")
                        f.write(f"<img src='../{img_path}' width='800'><br>\n")
            
            f.write("</body></html>\n")
        
        self.logger.info(f"Generated HTML report at {report_path}")
        return report_path
    
    def load_results(self, results_path: str) -> Dict[str, Any]:
        """
        Load previously saved backtest results.
        
        Args:
            results_path: Path to saved results file
            
        Returns:
            Dictionary containing backtest results
        """
        self.results = Serializer.load(results_path)
        self.logger.info(f"Loaded results from {results_path}")
        
        # Recalculate metrics
        self._calculate_metrics()
        
        return self.results
    
    def compare_with(self, other_backtest: 'BacktestRunner') -> Dict[str, Any]:
        """
        Compare this backtest with another backtest.
        
        Args:
            other_backtest: Another BacktestRunner instance
            
        Returns:
            Dictionary containing comparison results
        """
        if self.results is None or other_backtest.results is None:
            raise ValueError("Both backtests must have results to compare")
        
        comparison = {
            'backtest1': {
                'name': self.name,
                'summary': self.get_summary()
            },
            'backtest2': {
                'name': other_backtest.name,
                'summary': other_backtest.get_summary()
            },
            'differences': {}
        }
        
        # Compare key metrics
        metrics_to_compare = [
            'total_return', 'annualized_return', 'sharpe_ratio', 
            'max_drawdown', 'volatility', 'sortino_ratio'
        ]
        
        for metric in metrics_to_compare:
            if metric in comparison['backtest1']['summary'] and metric in comparison['backtest2']['summary']:
                val1 = comparison['backtest1']['summary'][metric]
                val2 = comparison['backtest2']['summary'][metric]
                if val1 is not None and val2 is not None:
                    diff = val1 - val2
                    pct_diff = diff / abs(val2) if val2 != 0 else float('inf')
                    comparison['differences'][metric] = {
                        'absolute': diff,
                        'percentage': pct_diff * 100
                    }
        
        return comparison