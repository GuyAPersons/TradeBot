import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy

class HedgingStrategy(BaseStrategy):
    """
    Strategy for implementing hedging techniques to protect against market risk.
    """
    
    def __init__(self, name: str, timeframes: List[str], instruments: List[str], params: Dict = None):
        """
        Initialize the hedging strategy.
        
        Args:
            name: Strategy name
            timeframes: List of timeframes to analyze
            instruments: List of instruments to trade
            params: Strategy-specific parameters including:
                - correlation_threshold: Correlation threshold for pair selection
                - hedge_ratio: Ratio for hedge position sizing
                - rebalance_frequency: How often to rebalance hedge positions
                - max_hedge_exposure: Maximum exposure for hedge positions
                - hedge_types: Types of hedging to use (e.g., direct, options, futures)
        """
        default_params = {
            "correlation_threshold": -0.7,  # Negative correlation threshold
            "hedge_ratio": 1.0,  # Default 1:1 hedge ratio
            "rebalance_frequency": "daily",  # Options: hourly, daily, weekly
            "max_hedge_exposure": 50.0,  # Maximum hedge exposure as percentage of portfolio
            "hedge_types": ["direct", "futures"],  # Types of hedging to use
            "min_volatility": 1.0,  # Minimum volatility to consider hedging
            "use_options": False,  # Whether to use options for hedging
            "option_delta_target": -0.5,  # Target delta for option hedges
            "use_dynamic_hedge_ratio": True,  # Dynamically adjust hedge ratio
            "use_correlation_matrix": True,  # Use correlation matrix for hedge selection
            "use_var_model": False,  # Use Value at Risk model for hedge sizing
            "confidence_level": 0.95  # Confidence level for VaR calculations
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(name, timeframes, instruments, default_params)
        self.correlation_matrix = pd.DataFrame()
        self.hedge_pairs = []
        self.active_hedges = {}
        self.last_rebalance = datetime.now()
        self.hedge_performance = {
            "total_hedges": 0,
            "active_hedges": 0,
            "total_cost": 0.0,
            "total_savings": 0.0,
            "net_benefit": 0.0
        }
    
    def analyze(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze market data to identify hedging opportunities.
        
        Args:
            data: Dictionary of DataFrames with market data for each instrument
            
        Returns:
            Dictionary with analysis results and potential hedging opportunities
        """
        # Update correlation matrix if using correlation-based hedging
        if self.params["use_correlation_matrix"]:
            self._update_correlation_matrix(data)
        
        # Find potential hedge pairs
        hedge_opportunities = self._find_hedge_opportunities(data)
        
        # Check if rebalancing is needed
        needs_rebalance = self._check_rebalance_needed()
        
        # Calculate portfolio risk metrics
        risk_metrics = self._calculate_risk_metrics(data)
        
        return {
            "hedge_opportunities": hedge_opportunities,
            "needs_rebalance": needs_rebalance,
            "risk_metrics": risk_metrics,
            "correlation_matrix": self.correlation_matrix,
            "timestamp": pd.Timestamp.now()
        }
    
    def _update_correlation_matrix(self, data: Dict[str, pd.DataFrame]) -> None:
        """Update the correlation matrix between instruments."""
        # Extract close prices for all instruments
        prices = {}
        for instrument, df in data.items():
            if df is not None and not df.empty:
                prices[instrument] = df['close']
        
        # Create a DataFrame with all price series
        if prices:
            price_df = pd.DataFrame(prices)
            
            # Calculate returns
            returns_df = price_df.pct_change().dropna()
            
            # Calculate correlation matrix
            self.correlation_matrix = returns_df.corr()
    
    def _find_hedge_opportunities(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Find potential hedging opportunities based on correlation and volatility."""
        opportunities = []
        
        if self.params["use_correlation_matrix"] and not self.correlation_matrix.empty:
            # Find negatively correlated pairs
            for instrument1 in self.correlation_matrix.columns:
                for instrument2 in self.correlation_matrix.columns:
                    if instrument1 != instrument2:
                        correlation = self.correlation_matrix.loc[instrument1, instrument2]
                        
                        # Check if correlation is below threshold (negative correlation)
                        if correlation <= self.params["correlation_threshold"]:
                            # Calculate volatility for both instruments
                            vol1 = self._calculate_volatility(data.get(instrument1))
                            vol2 = self._calculate_volatility(data.get(instrument2))
                            
                            # Only consider if volatility is above minimum threshold
                            if vol1 >= self.params["min_volatility"] and vol2 >= self.params["min_volatility"]:
                                # Calculate optimal hedge ratio based on volatilities
                                if self.params["use_dynamic_hedge_ratio"]:
                                    hedge_ratio = vol1 / vol2
                                else:
                                    hedge_ratio = self.params["hedge_ratio"]
                                
                                opportunities.append({
                                    "primary_instrument": instrument1,
                                    "hedge_instrument": instrument2,
                                    "correlation": correlation,
                                    "primary_volatility": vol1,
                                    "hedge_volatility": vol2,
                                    "hedge_ratio": hedge_ratio,
                                    "timestamp": pd.Timestamp.now()
                                })
        
        # Sort opportunities by correlation (most negative first)
        opportunities.sort(key=lambda x: x["correlation"])
        
        return opportunities
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate annualized volatility for an instrument."""
        if df is None or df.empty or len(df) < 2:
            return 0.0
        
        # Calculate daily returns
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate annualized volatility (standard deviation of returns * sqrt(252))
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)  # Assuming 252 trading days per year
        
        return annualized_vol * 100  # Convert to percentage
    
    def _check_rebalance_needed(self) -> bool:
        """Check if hedge positions need rebalancing based on frequency setting."""
        now = datetime.now()
        
        if self.params["rebalance_frequency"] == "hourly":
            return (now - self.last_rebalance) >= timedelta(hours=1)
        elif self.params["rebalance_frequency"] == "daily":
            return (now - self.last_rebalance) >= timedelta(days=1)
        elif self.params["rebalance_frequency"] == "weekly":
            return (now - self.last_rebalance) >= timedelta(weeks=1)
        else:
            return False
    
    def _calculate_risk_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate portfolio risk metrics for hedging decisions."""
        metrics = {
            "portfolio_volatility": 0.0,
            "var_95": 0.0,
            "var_99": 0.0,
            "expected_shortfall": 0.0,
            "beta": 0.0,
            "downside_deviation": 0.0,
            "max_drawdown": 0.0
        }
        
        # Combine all instrument returns into a portfolio
        returns_series = []
        
        for instrument, df in data.items():
            if df is not None and not df.empty and len(df) > 1:
                returns = df['close'].pct_change().dropna()
                if len(returns) > 0:
                    returns_series.append(returns)
        
        if not returns_series:
            return metrics
        
        # Combine returns (simple equal weighting for now)
        portfolio_returns = pd.concat(returns_series, axis=1).mean(axis=1)
        
        # Calculate portfolio volatility
        metrics["portfolio_volatility"] = portfolio_returns.std() * np.sqrt(252) * 100
        
        # Calculate Value at Risk (VaR)
        if len(portfolio_returns) > 20:  # Need sufficient data points
            metrics["var_95"] = np.percentile(portfolio_returns, 5) * 100  # 95% VaR
            metrics["var_99"] = np.percentile(portfolio_returns, 1) * 100  # 99% VaR
            
            # Expected Shortfall (Conditional VaR)
            es_cutoff = np.percentile(portfolio_returns, 5)
            tail_returns = portfolio_returns[portfolio_returns <= es_cutoff]
            if len(tail_returns) > 0:
                metrics["expected_shortfall"] = tail_returns.mean() * 100
            
            # Downside Deviation (only negative returns)
            negative_returns = portfolio_returns[portfolio_returns < 0]
            if len(negative_returns) > 0:
                metrics["downside_deviation"] = negative_returns.std() * np.sqrt(252) * 100
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1)
            metrics["max_drawdown"] = drawdown.min() * 100
        
        return metrics
    
    def generate_signals(self, analysis: Dict) -> List[Dict]:
        """
        Generate trading signals based on hedging opportunities.
        
        Args:
            analysis: Analysis results from the analyze method
            
        Returns:
            List of signal dictionaries with hedging actions
        """
        signals = []
        
        # Check if rebalancing is needed
        if analysis.get("needs_rebalance", False):
            # Generate rebalance signals for existing hedges
            rebalance_signals = self._generate_rebalance_signals()
            signals.extend(rebalance_signals)
            
            # Update last rebalance timestamp
            self.last_rebalance = datetime.now()
        
        # Process new hedge opportunities
        for opportunity in analysis.get("hedge_opportunities", []):
            # Check if we already have an active hedge for this instrument
            primary_instrument = opportunity["primary_instrument"]
            
            if primary_instrument not in self.active_hedges:
                # Check if adding this hedge would exceed max exposure
                if self._check_hedge_exposure_limit():
                    # Create new hedge signals
                    hedge_signals = self._create_hedge_signals(opportunity)
                    signals.extend(hedge_signals)
                    
                    # Add to active hedges
                    self.active_hedges[primary_instrument] = {
                        "hedge_instrument": opportunity["hedge_instrument"],
                        "hedge_ratio": opportunity["hedge_ratio"],
                        "correlation": opportunity["correlation"],
                        "start_time": datetime.now()
                    }
                    
                    # Update hedge performance tracking
                    self.hedge_performance["total_hedges"] += 1
                    self.hedge_performance["active_hedges"] += 1
        
        return signals
    
    def _generate_rebalance_signals(self) -> List[Dict]:
        """Generate signals to rebalance existing hedge positions."""
        signals = []
        
        for primary_instrument, hedge_info in self.active_hedges.items():
            hedge_instrument = hedge_info["hedge_instrument"]
            
            # Close existing hedge position
            signals.append({
                "strategy": self.name,
                "type": "hedge",
                "subtype": "rebalance",
                "action": "CLOSE",
                "instrument": hedge_instrument,
                "timestamp": pd.Timestamp.now(),
                "metadata": {
                    "primary_instrument": primary_instrument,
                    "hedge_ratio": hedge_info["hedge_ratio"],
                    "correlation": hedge_info["correlation"],
                    "hedge_age_days": (datetime.now() - hedge_info["start_time"]).days
                }
            })
            
            # Open new hedge position with updated ratio if needed
            signals.append({
                "strategy": self.name,
                "type": "hedge",
                "subtype": "rebalance",
                "action": "OPEN",
                "instrument": hedge_instrument,
                "timestamp": pd.Timestamp.now(),
                "metadata": {
                    "primary_instrument": primary_instrument,
                    "hedge_ratio": hedge_info["hedge_ratio"],
                    "correlation": hedge_info["correlation"]
                }
            })
        
        return signals
    
    def _create_hedge_signals(self, opportunity: Dict) -> List[Dict]:
        """Create signals for a new hedge position."""
        signals = []
        
        primary_instrument = opportunity["primary_instrument"]
        hedge_instrument = opportunity["hedge_instrument"]
        hedge_ratio = opportunity["hedge_ratio"]
        
        # Determine hedge type based on parameters
        hedge_types = self.params["hedge_types"]
        
        if "direct" in hedge_types:
            # Direct instrument hedge
            signals.append({
                "strategy": self.name,
                "type": "hedge",
                "subtype": "direct",
                "action": "SELL",  # Assuming negative correlation, so we sell the hedge instrument
                "instrument": hedge_instrument,
                "timestamp": pd.Timestamp.now(),
                "metadata": {
                    "primary_instrument": primary_instrument,
                    "hedge_ratio": hedge_ratio,
                    "correlation": opportunity["correlation"]
                }
            })
        
        if "futures" in hedge_types:
            # Futures hedge
            signals.append({
                "strategy": self.name,
                "type": "hedge",
                "subtype": "futures",
                "action": "SELL",
                "instrument": f"{hedge_instrument}_FUT",  # Futures contract
                "timestamp": pd.Timestamp.now(),
                "metadata": {
                    "primary_instrument": primary_instrument,
                    "hedge_ratio": hedge_ratio,
                    "correlation": opportunity["correlation"]
                }
            })
        
        if self.params["use_options"]:
            # Options hedge
            signals.append({
                "strategy": self.name,
                "type": "hedge",
                "subtype": "options",
                "action": "BUY",
                "instrument": f"{hedge_instrument}_PUT",  # Put option
                "timestamp": pd.Timestamp.now(),
                "metadata": {
                    "primary_instrument": primary_instrument,
                    "hedge_ratio": hedge_ratio,
                    "correlation": opportunity["correlation"],
                    "target_delta": self.params["option_delta_target"]
                }
            })
        
        return signals
    
    def _check_hedge_exposure_limit(self) -> bool:
        """Check if adding another hedge would exceed the maximum hedge exposure."""
        # In a real implementation, this would calculate the actual exposure
        # For now, just check the number of active hedges as a proxy
        current_exposure_pct = len(self.active_hedges) * 10  # Assume each hedge is 10% exposure
        
        return current_exposure_pct < self.params["max_hedge_exposure"]
    
    def close_hedge(self, primary_instrument: str) -> List[Dict]:
        """
        Close an existing hedge position.
        
        Args:
            primary_instrument: The primary instrument being hedged
            
        Returns:
            List of signals to close the hedge
        """
        signals = []
        
        if primary_instrument in self.active_hedges:
            hedge_info = self.active_hedges[primary_instrument]
            hedge_instrument = hedge_info["hedge_instrument"]
            
            # Generate close signal based on hedge type
            for hedge_type in self.params["hedge_types"]:
                if hedge_type == "direct":
                    signals.append({
                        "strategy": self.name,
                        "type": "hedge",
                        "subtype": "direct",
                        "action": "BUY",  # Buy to close the short hedge position
                        "instrument": hedge_instrument,
                        "timestamp": pd.Timestamp.now(),
                        "metadata": {
                            "primary_instrument": primary_instrument,
                            "close_reason": "manual",
                            "hedge_age_days": (datetime.now() - hedge_info["start_time"]).days
                        }
                    })
                
                elif hedge_type == "futures":
                    signals.append({
                        "strategy": self.name,
                        "type": "hedge",
                        "subtype": "futures",
                        "action": "BUY",  # Buy to close the short futures position
                        "instrument": f"{hedge_instrument}_FUT",
                        "timestamp": pd.Timestamp.now(),
                        "metadata": {
                            "primary_instrument": primary_instrument,
                            "close_reason": "manual",
                            "hedge_age_days": (datetime.now() - hedge_info["start_time"]).days
                        }
                    })
                
                elif hedge_type == "options" and self.params["use_options"]:
                    signals.append({
                        "strategy": self.name,
                        "type": "hedge",
                        "subtype": "options",
                        "action": "SELL",  # Sell to close the long put option
                        "instrument": f"{hedge_instrument}_PUT",
                        "timestamp": pd.Timestamp.now(),
                        "metadata": {
                            "primary_instrument": primary_instrument,
                            "close_reason": "manual",
                            "hedge_age_days": (datetime.now() - hedge_info["start_time"]).days
                        }
                    })
            
            # Remove from active hedges
            del self.active_hedges[primary_instrument]
            self.hedge_performance["active_hedges"] -= 1
        
        return signals
    
    def evaluate_hedge_performance(self, primary_returns: pd.Series, hedge_returns: pd.Series) -> Dict:
        """
        Evaluate the performance of a hedge.
        
        Args:
            primary_returns: Returns of the primary instrument
            hedge_returns: Returns of the hedging instrument
            
        Returns:
            Dictionary with hedge performance metrics
        """
        # Calculate combined returns (assuming equal weighting)
        combined_returns = primary_returns + (-1 * hedge_returns)  # Negative for hedge position
        
        # Calculate metrics
        primary_volatility = primary_returns.std() * np.sqrt(252) * 100
        combined_volatility = combined_returns.std() * np.sqrt(252) * 100
        
        # Volatility reduction
        volatility_reduction = primary_volatility - combined_volatility
        volatility_reduction_pct = (volatility_reduction / primary_volatility) * 100 if primary_volatility > 0 else 0
        
        # Correlation
        correlation = primary_returns.corr(hedge_returns)
        
        # Drawdown comparison
        primary_dd = self._calculate_max_drawdown(primary_returns)
        combined_dd = self._calculate_max_drawdown(combined_returns)
        drawdown_reduction = primary_dd - combined_dd
        
        return {
            "volatility_reduction": volatility_reduction,
            "volatility_reduction_pct": volatility_reduction_pct,
            "correlation": correlation,
            "drawdown_reduction": drawdown_reduction,
            "primary_sharpe": self._calculate_sharpe_ratio(primary_returns),
            "combined_sharpe": self._calculate_sharpe_ratio(combined_returns),
            "hedge_cost": hedge_returns.mean() * 252 * 100,  # Annualized cost in percentage
            "net_benefit": volatility_reduction - (hedge_returns.mean() * 252 * 100)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from a returns series."""
        if returns.empty:
            return 0.0
            
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        return drawdown.min() * 100  # Convert to percentage
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from a returns series."""
        if returns.empty or returns.std() == 0:
            return 0.0
            
        excess_returns = returns.mean() * 252 - risk_free_rate
        return excess_returns / (returns.std() * np.sqrt(252))
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        valid = True
        
        # Check correlation threshold
        if self.params["correlation_threshold"] > 0:
            self.logger.warning("correlation_threshold should be negative for effective hedging")
            valid = False
        
        # Check hedge ratio
        if self.params["hedge_ratio"] <= 0:
            self.logger.warning("hedge_ratio must be greater than 0")
            valid = False
        
        # Check max hedge exposure
        if self.params["max_hedge_exposure"] <= 0 or self.params["max_hedge_exposure"] > 100:
            self.logger.warning("max_hedge_exposure must be between 0 and 100")
            valid = False
        
        return valid
    
    def get_hedge_performance(self) -> Dict:
        """Get hedge performance statistics."""
        return self.hedge_performance