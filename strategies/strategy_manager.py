import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import time
from datetime import datetime
import importlib
import os
import json

from .base_strategy import BaseStrategy
from .meta_strategy import MetaStrategy

class StrategyManager:
    """
    Strategy Manager that loads, initializes, and coordinates all trading strategies.
    It uses the MetaStrategy to dynamically allocate capital and combine signals.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Strategy Manager.
        
        Args:
            config_path: Path to the strategy configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.strategies = {}
        self.meta_strategy = None
        self.config = {}
        self.performance_tracker = {}
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load strategy configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            self.logger.info(f"Loaded configuration from {config_path}")
            
            # Initialize strategies based on config
            self._initialize_strategies()
            
            # Initialize meta-strategy
            self._initialize_meta_strategy()
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
    
    def _initialize_strategies(self) -> None:
        """Initialize strategy instances based on configuration."""
        strategy_configs = self.config.get("strategies", {})
        
        for strategy_name, strategy_config in strategy_configs.items():
            if not strategy_config.get("enabled", True):
                self.logger.info(f"Strategy {strategy_name} is disabled, skipping")
                continue
                
            try:
                # Get strategy class and parameters
                strategy_class = strategy_config.get("class")
                strategy_params = strategy_config.get("params", {})
                
                if not strategy_class:
                    self.logger.warning(f"No class specified for strategy {strategy_name}, skipping")
                    continue
                
                # Dynamically import and instantiate the strategy
                module_path, class_name = strategy_class.rsplit('.', 1)
                module = importlib.import_module(module_path)
                StrategyClass = getattr(module, class_name)
                
                # Create strategy instance
                strategy = StrategyClass(name=strategy_name, params=strategy_params)
                
                # Add to strategies dictionary
                self.strategies[strategy_name] = strategy
                
                self.logger.info(f"Initialized strategy: {strategy_name}")
                
            except Exception as e:
                self.logger.error(f"Error initializing strategy {strategy_name}: {str(e)}", exc_info=True)
    
    def _initialize_meta_strategy(self) -> None:
        """Initialize the meta-strategy."""
        meta_config = self.config.get("meta_strategy", {})
        
        try:
            # Create meta-strategy instance
            self.meta_strategy = MetaStrategy(
                strategies=self.strategies,
                params=meta_config.get("params", {})
            )
            
            self.logger.info("Initialized meta-strategy")
            
        except Exception as e:
            self.logger.error(f"Error initializing meta-strategy: {str(e)}", exc_info=True)
    
    def execute(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Execute the trading strategy pipeline.
        
        Args:
            data: Dictionary of DataFrames with market data for each instrument
            
        Returns:
            List of trading signals
        """
        if not self.meta_strategy:
            self.logger.warning("Meta-strategy not initialized, cannot execute")
            return []
        
        try:
            # Execute meta-strategy
            signals = self.meta_strategy.execute(data)
            
            # Log execution summary
            self._log_execution_summary(signals)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error executing strategy pipeline: {str(e)}", exc_info=True)
            return []
    
    def _log_execution_summary(self, signals: List[Dict]) -> None:
        """Log a summary of the execution results."""
        if not signals:
            self.logger.info("No signals generated")
            return
            
        # Count signals by type
        signal_counts = {}
        for signal in signals:
            signal_type = signal.get("type", "unknown")
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
        
        # Log summary
        self.logger.info(f"Generated {len(signals)} signals: {signal_counts}")
        
        # Log allocations
        if self.meta_strategy:
            allocations = self.meta_strategy.get_strategy_allocations()
            self.logger.info(f"Current strategy allocations: {allocations}")
    
    def update_performance(self, performance_data: Dict[str, Dict]) -> None:
        """
        Update performance metrics for strategies.
        
        Args:
            performance_data: Dictionary mapping strategy names to performance metrics
        """
        if not self.meta_strategy:
            return
            
        for strategy_name, metrics in performance_data.items():
            if strategy_name in self.strategies:
                self.meta_strategy.update_performance(strategy_name, metrics)
    
    def get_strategy_allocations(self) -> Dict[str, float]:
        """Get current strategy allocations."""
        if self.meta_strategy:
            return self.meta_strategy.get_strategy_allocations()
        return {}
    
    def get_market_conditions(self) -> Dict:
        """Get current market condition analysis."""
        if self.meta_strategy:
            return self.meta_strategy.get_market_conditions()
        return {}
    
    def get_strategy_scores(self) -> Dict[str, float]:
        """Get current strategy suitability scores."""
        if self.meta_strategy:
            return self.meta_strategy.get_strategy_scores()
        return {}
    
    def get_performance_history(self) -> Dict[str, List[Dict]]:
        """Get performance history for all strategies."""
        if self.meta_strategy:
            return self.meta_strategy.get_performance_history()
        return {}
