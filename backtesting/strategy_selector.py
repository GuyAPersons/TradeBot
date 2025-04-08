"""
Strategy selector module for automatically selecting the best strategy based on market conditions.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from .data.data_handler import DataHandler
from .strategies.base import Strategy
from .strategies.strategy_manager import StrategyManager


class StrategySelector:
    """
    Automatically selects the best strategy based on current market conditions.
    """
    
    def __init__(self, data_handler: DataHandler, strategy_manager: StrategyManager):
        """
        Initialize the strategy selector.
        
        Args:
            data_handler: DataHandler instance for accessing market data
            strategy_manager: StrategyManager instance containing available strategies
        """
        self.data_handler = data_handler
        self.strategy_manager = strategy_manager
        self.logger = logging.getLogger(__name__)
        
        # Market condition indicators
        self.market_indicators = {
            'volatility': self._calculate_volatility,
            'trend_strength': self._calculate_trend_strength,
            'mean_reversion_potential': self._calculate_mean_reversion_potential,
            'market_regime': self._detect_market_regime,
            'liquidity': self._calculate_liquidity,
            'correlation': self._calculate_correlation
        }
        
        # Strategy suitability scores for different market conditions
        # Higher score means the strategy is more suitable for that condition
        self.strategy_suitability = {
            'trend_following': {
                'volatility': 0.3,  # Moderate volatility is good
                'trend_strength': 0.9,  # Strong trends are excellent
                'mean_reversion_potential': -0.5,  # Not good for mean reversion
                'market_regime': {'bull': 0.8, 'bear': 0.6, 'sideways': -0.7},
                'liquidity': 0.4,  # Needs decent liquidity
                'correlation': 0.2  # Low correlation sensitivity
            },
            'mean_reversion': {
                'volatility': 0.6,  # Higher volatility is good
                'trend_strength': -0.7,  # Strong trends are bad
                'mean_reversion_potential': 0.9,  # High potential is excellent
                'market_regime': {'bull': 0.2, 'bear': 0.2, 'sideways': 0.9},
                'liquidity': 0.5,  # Needs good liquidity
                'correlation': 0.4  # Moderate correlation sensitivity
            },
            'arbitrage': {
                'volatility': 0.2,  # Low volatility is fine
                'trend_strength': -0.1,  # Trend doesn't matter much
                'mean_reversion_potential': 0.5,  # Some mean reversion helps
                'market_regime': {'bull': 0.3, 'bear': 0.3, 'sideways': 0.5},
                'liquidity': 0.7,  # Needs high liquidity
                'correlation': 0.8  # High correlation sensitivity
            },
            'market_making': {
                'volatility': -0.6,  # Low volatility is better
                'trend_strength': -0.5,  # No strong trends preferred
                'mean_reversion_potential': 0.3,  # Some mean reversion helps
                'market_regime': {'bull': 0.3, 'bear': 0.3, 'sideways': 0.8},
                'liquidity': 0.9,  # Needs very high liquidity
                'correlation': 0.3  # Moderate correlation sensitivity
            },
            'hedging': {
                'volatility': 0.7,  # Higher volatility is good
                'trend_strength': 0.0,  # Trend doesn't matter
                'mean_reversion_potential': 0.0,  # Mean reversion doesn't matter
                'market_regime': {'bull': 0.3, 'bear': 0.8, 'sideways': 0.4},
                'liquidity': 0.6,  # Needs good liquidity
                'correlation': 0.9  # Very high correlation sensitivity
            },
            'flash': {
                'volatility': 0.5,  # Moderate volatility is good
                'trend_strength': 0.2,  # Some trend helps
                'mean_reversion_potential': 0.4,  # Some mean reversion helps
                'market_regime': {'bull': 0.5, 'bear': 0.5, 'sideways': 0.5},
                'liquidity': 0.9,  # Needs very high liquidity
                'correlation': 0.2  # Low correlation sensitivity
            },
            'bots': {
                'volatility': 0.4,  # Moderate volatility is fine
                'trend_strength': 0.5,  # Moderate trend sensitivity
                'mean_reversion_potential': 0.5,  # Moderate mean reversion sensitivity
                'market_regime': {'bull': 0.6, 'bear': 0.6, 'sideways': 0.6},
                'liquidity': 0.7,  # Needs good liquidity
                'correlation': 0.5  # Moderate correlation sensitivity
            },
            'meta': {
                'volatility': 0.5,  # Adaptable to different volatility
                'trend_strength': 0.5,  # Adaptable to different trends
                'mean_reversion_potential': 0.5,  # Adaptable to mean reversion
                'market_regime': {'bull': 0.7, 'bear': 0.7, 'sideways': 0.7},
                'liquidity': 0.6,  # Needs good liquidity
                'correlation': 0.6  # Moderate correlation sensitivity
            }
        }
    
    def select_best_strategy(self, symbols: List[str], lookback_period: int = 30) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze current market conditions and select the best strategy.
        
        Args:
            symbols: List of symbols to analyze
            lookback_period: Number of days to look back for analysis
            
        Returns:
            Tuple containing the best strategy type and market condition metrics
        """
        self.logger.info(f"Analyzing market conditions for {len(symbols)} symbols with {lookback_period} day lookback")
        
        # Get market data for analysis
        market_data = {}
        for symbol in symbols:
            try:
                data = self.data_handler.get_latest_bars(symbol, lookback_period)
                if data is not None and not data.empty:
                    market_data[symbol] = data
            except Exception as e:
                self.logger.warning(f"Error getting data for {symbol}: {e}")
        
        if not market_data:
            self.logger.error("No market data available for analysis")
            return "trend_following", {}  # Default to trend following if no data
        
        # Calculate market condition indicators
        market_conditions = {}
        for indicator_name, indicator_func in self.market_indicators.items():
            try:
                market_conditions[indicator_name] = indicator_func(market_data)
                self.logger.debug(f"{indicator_name}: {market_conditions[indicator_name]}")
            except Exception as e:
                self.logger.warning(f"Error calculating {indicator_name}: {e}")
                market_conditions[indicator_name] = 0.5  # Neutral value on error
        
        # Score each strategy type based on current market conditions
        strategy_scores = {}
        for strategy_type, suitability in self.strategy_suitability.items():
            score = 0.0
            for indicator, value in market_conditions.items():
                if indicator == 'market_regime':
                    # Handle market regime specially
                    regime = value
                    regime_score = suitability[indicator].get(regime, 0.0)
                    score += regime_score
                else:
                    # For numeric indicators
                    sensitivity = suitability.get(indicator, 0.0)
                    # If sensitivity is negative, a higher indicator value is worse
                    if sensitivity < 0:
                        score += sensitivity * value
                    else:
                        score += sensitivity * value
            
            strategy_scores[strategy_type] = score
            self.logger.debug(f"Strategy {strategy_type} score: {score}")
        
        # Find the strategy with the highest score
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        self.logger.info(f"Selected best strategy: {best_strategy[0]} with score {best_strategy[1]}")
        
        return best_strategy[0], market_conditions
    
    def get_strategy_instance(self, strategy_type: str, **kwargs) -> Optional[Strategy]:
        """
        Get an instance of the selected strategy type.
        
        Args:
            strategy_type: Type of strategy to instantiate
            **kwargs: Additional parameters for the strategy
            
        Returns:
            Strategy instance or None if not available
        """
        try:
            # Get available strategies of this type
            available_strategies = self.strategy_manager.get_strategies_by_type(strategy_type)
            
            if not available_strategies:
                self.logger.warning(f"No strategies available for type {strategy_type}")
                return None
            
            # Select the first available strategy of this type
            strategy_class = available_strategies[0]
            return strategy_class(**kwargs)
            
        except Exception as e:
            self.logger.error(f"Error creating strategy instance for {strategy_type}: {e}")
            return None
    
    def _calculate_volatility(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate average volatility across all symbols.
        
        Args:
            market_data: Dictionary of market data frames by symbol
            
        Returns:
            Normalized volatility score (0-1)
        """
        volatilities = []
        for symbol, data in market_data.items():
            if 'close' in data.columns:
                # Calculate daily returns
                returns = data['close'].pct_change().dropna()
                # Calculate volatility (standard deviation of returns)
                volatility = returns.std()
                volatilities.append(volatility)
        
        if not volatilities:
            return 0.5  # Neutral value if no data
        
        # Average volatility across symbols
        avg_volatility = np.mean(volatilities)
        
        # Normalize to 0-1 range (assuming typical volatility range of 0-0.05)
        normalized_volatility = min(1.0, max(0.0, avg_volatility / 0.05))
        
        return normalized_volatility
    
    def _calculate_trend_strength(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate average trend strength across all symbols.
        
        Args:
            market_data: Dictionary of market data frames by symbol
            
        Returns:
            Normalized trend strength score (0-1)
        """
        trend_strengths = []
        for symbol, data in market_data.items():
            if 'close' in data.columns:
                # Calculate short and long moving averages
                short_ma = data['close'].rolling(window=10).mean()
                long_ma = data['close'].rolling(window=50).mean()
                
                # Calculate trend strength as correlation between price and time
                prices = data['close'].values
                time_index = np.arange(len(prices))
                if len(prices) > 5:  # Need enough data points
                    correlation = np.corrcoef(time_index, prices)[0, 1]
                    trend_strengths.append(abs(correlation))  # Absolute value for strength
        
        if not trend_strengths:
            return 0.5  # Neutral value if no data
        
        # Average trend strength across symbols
        avg_trend_strength = np.mean(trend_strengths)
        
        # Already normalized to 0-1 range
        return avg_trend_strength
    
    def _calculate_mean_reversion_potential(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate mean reversion potential across all symbols.
        
        Args:
            market_data: Dictionary of market data frames by symbol
            
        Returns:
            Normalized mean reversion potential score (0-1)
        """
        mean_reversion_potentials = []
        for symbol, data in market_data.items():
            if 'close' in data.columns:
                # Calculate z-score of price relative to recent history
                price = data['close'].iloc[-1]
                mean_price = data['close'].mean()
                std_price = data['close'].std()
                
                if std_price > 0:
                    z_score = abs((price - mean_price) / std_price)
                    # Higher z-score means more potential for mean reversion
                    mean_reversion_potentials.append(min(z_score / 3.0, 1.0))  # Normalize to 0-1
        
        if not mean_reversion_potentials:
            return 0.5  # Neutral value if no data
        
        # Average mean reversion potential across symbols
        avg_potential = np.mean(mean_reversion_potentials)
        
        return avg_potential
    
    def _detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> str:
        """
        Detect current market regime (bull, bear, sideways).
        
        Args:
            market_data: Dictionary of market data frames by symbol
            
        Returns:
            Market regime as string: 'bull', 'bear', or 'sideways'
        """
        # Count symbols in each regime
        regimes = {'bull': 0, 'bear': 0, 'sideways': 0}
        
        for symbol, data in market_data.items():
            if 'close' in data.columns and len(data) > 20:
                # Calculate short and long term performance
                current_price = data['close'].iloc[-1]
                price_20d_ago = data['close'].iloc[-21] if len(data) > 20 else data['close'].iloc[0]
                price_5d_ago = data['close'].iloc[-6] if len(data) > 5 else data['close'].iloc[0]
                
                # Calculate returns
                return_20d = (current_price / price_20d_ago) - 1
                return_5d = (current_price / price_5d_ago) - 1
                
                # Determine regime
                if return_20d > 0.05:  # 5% up over 20 days
                    regimes['bull'] += 1
                elif return_20d < -0.05:  # 5% down over 20 days
                    regimes['bear'] += 1
                elif abs(return_5d) < 0.02:  # Less than 2% change over 5 days
                    regimes['sideways'] += 1
                else:
                    # If no clear regime, count as sideways
                    regimes['sideways'] += 1
        
        # Return the dominant regime
        if not any(regimes.values()):
            return 'sideways'  # Default to sideways if no data
            
        dominant_regime = max(regimes.items(), key=lambda x: x[1])[0]
        return dominant_regime
    
    def _calculate_liquidity(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate average liquidity across all symbols.
        
        Args:
            market_data: Dictionary of market data frames by symbol
            
        Returns:
            Normalized liquidity score (0-1)
        """
        liquidity_scores = []
        for symbol, data in market_data.items():
            if 'volume' in data.columns and 'close' in data.columns:
                # Calculate average daily dollar volume
                if 'volume' in data.columns and 'close' in data.columns:
                    dollar_volume = (data['volume'] * data['close']).mean()
                    
                    # Normalize dollar volume (assuming $10M is high liquidity)
                    normalized_liquidity = min(1.0, max(0.0, dollar_volume / 10000000.0))
                    liquidity_scores.append(normalized_liquidity)
        
        if not liquidity_scores:
            return 0.5  # Neutral value if no data
        
        # Average liquidity across symbols
        avg_liquidity = np.mean(liquidity_scores)
        
        return avg_liquidity
    
    def _calculate_correlation(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate average correlation between symbols.
        
        Args:
            market_data: Dictionary of market data frames by symbol
            
        Returns:
            Normalized correlation score (0-1)
        """
        # Need at least 2 symbols to calculate correlation
        if len(market_data) < 2:
            return 0.5  # Neutral value if insufficient data
        
        # Extract returns for each symbol
        returns_dict = {}
        for symbol, data in market_data.items():
            if 'close' in data.columns and len(data) > 1:
                returns = data['close'].pct_change().dropna()
                if len(returns) > 0:
                    returns_dict[symbol] = returns
        
        # Need at least 2 symbols with valid returns
        if len(returns_dict) < 2:
            return 0.5  # Neutral value if insufficient data
        
        # Calculate pairwise correlations
        correlations = []
        symbols = list(returns_dict.keys())
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                # Align the two return series
                returns1 = returns_dict[symbols[i]]
                returns2 = returns_dict[symbols[j]]
                
                # Find common date range
                common_index = returns1.index.intersection(returns2.index)
                if len(common_index) > 5:  # Need enough common data points
                    aligned_returns1 = returns1.loc[common_index]
                    aligned_returns2 = returns2.loc[common_index]
                    
                    # Calculate correlation
                    correlation = abs(aligned_returns1.corr(aligned_returns2))
                    correlations.append(correlation)
        
        if not correlations:
            return 0.5  # Neutral value if no correlations could be calculated
        
        # Average absolute correlation across all pairs
        avg_correlation = np.mean(correlations)
        
        return avg_correlation