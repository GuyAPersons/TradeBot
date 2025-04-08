from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, timeframes: List[str], instruments: List[str], params: Dict = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            timeframes: List of timeframes to analyze
            instruments: List of instruments to trade
            params: Strategy-specific parameters
        """
        self.name = name
        self.timeframes = timeframes
        self.instruments = instruments
        self.params = params or {}
        self.logger = logging.getLogger(f"strategy.{name}")
        self.is_active = True
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0
        }
    
    @abstractmethod
    def analyze(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze market data and generate signals.
        
        Args:
            data: Dictionary of DataFrames with market data for each instrument
            
        Returns:
            Dictionary with analysis results and signals
        """
        pass
    
    @abstractmethod
    def generate_signals(self, analysis: Dict) -> List[Dict]:
        """
        Generate trading signals based on analysis.
        
        Args:
            analysis: Analysis results from the analyze method
            
        Returns:
            List of signal dictionaries with action, instrument, etc.
        """
        pass
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float, 
                               stop_loss_pips: float, instrument: str, price: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Percentage of account to risk per trade
            stop_loss_pips: Stop loss distance in pips
            instrument: Trading instrument
            price: Current price
            
        Returns:
            Position size in lots
        """
        risk_amount = account_balance * (risk_per_trade / 100)
        pip_value = self._calculate_pip_value(instrument, price)
        
        if pip_value == 0 or stop_loss_pips == 0:
            return 0
            
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return round(position_size, 2)
    
    def _calculate_pip_value(self, instrument: str, price: float) -> float:
        """Calculate the value of a pip for the given instrument."""
        # This is a simplified calculation - in a real system you'd need to account for
        # different pip definitions across instruments and account currency
        if 'USD' in instrument:
            if instrument.startswith('USD'):
                return 0.0001 / price
            else:
                return 0.0001
        elif 'JPY' in instrument:
            return 0.01
        else:
            return 0.0001
    
    def update_performance(self, trade_result: Dict) -> None:
        """Update strategy performance metrics with new trade result."""
        self.performance_metrics["total_trades"] += 1
        
        if trade_result["profit"] > 0:
            self.performance_metrics["winning_trades"] += 1
        else:
            self.performance_metrics["losing_trades"] += 1
            
        self.performance_metrics["total_profit"] += trade_result["profit"]
        
        # Update win rate
        if self.performance_metrics["total_trades"] > 0:
            self.performance_metrics["win_rate"] = (
                self.performance_metrics["winning_trades"] / self.performance_metrics["total_trades"] * 100
            )
        
        # Other metrics would be calculated periodically rather than per trade
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of strategy performance."""
        return self.performance_metrics
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        # Base implementation just returns True
        # Child classes should override this with specific validation logic
        return True
