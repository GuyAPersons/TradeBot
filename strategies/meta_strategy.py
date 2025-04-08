import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import time
from datetime import datetime
import logging
from collections import defaultdict

from .base_strategy import BaseStrategy

class MetaStrategy:
    """
    Meta-strategy layer that coordinates multiple strategies, evaluates market conditions,
    and dynamically allocates capital across strategies based on their suitability.
    """
    
    def __init__(self, strategies: Dict[str, BaseStrategy], params: Dict = None):
        """
        Initialize the meta-strategy.
        
        Args:
            strategies: Dictionary of strategy instances (name -> strategy object)
            params: Configuration parameters including:
                - allocation_method: How to allocate capital ('equal', 'performance', 'adaptive')
                - rebalance_frequency: How often to rebalance allocations (in hours)
                - min_allocation: Minimum allocation for any active strategy
                - max_allocation: Maximum allocation for any strategy
                - lookback_periods: Number of periods to look back for performance evaluation
                - market_regime_indicators: List of indicators to use for regime detection
                - strategy_correlation_threshold: Threshold for strategy correlation
                - confidence_threshold: Minimum confidence to include a strategy
        """
        self.strategies = strategies
        
        # Default parameters
        default_params = {
            "allocation_method": "adaptive",  # 'equal', 'performance', 'adaptive'
            "rebalance_frequency": 24,  # hours
            "min_allocation": 0.05,  # 5% minimum allocation if strategy is used
            "max_allocation": 0.50,  # 50% maximum allocation to any strategy
            "lookback_periods": 30,  # periods to look back for performance
            "market_regime_indicators": ["volatility", "trend_strength", "volume_profile"],
            "strategy_correlation_threshold": 0.7,  # avoid highly correlated strategies
            "confidence_threshold": 0.3,  # minimum confidence to include a strategy
            "enable_strategy_combination": True,  # whether to combine strategy signals
            "combination_method": "weighted",  # 'weighted', 'voting', 'best_only'
            "adaptive_weights": True,  # dynamically adjust weights based on performance
            "max_active_strategies": 3,  # maximum number of strategies to use simultaneously
            "performance_metric": "sharpe",  # 'sharpe', 'sortino', 'profit', 'win_rate'
        }
        
        if params:
            default_params.update(params)
        
        self.params = default_params
        self.logger = logging.getLogger(__name__)
        
        # Initialize state variables
        self.last_rebalance = datetime.now()
        self.current_allocations = self._initialize_allocations()
        self.strategy_performance = {name: {} for name in strategies.keys()}
        self.market_conditions = {}
        self.strategy_scores = {}
        self.strategy_signals = {}
        self.combined_signals = {}
        self.performance_history = defaultdict(list)
        
        # Initialize market regime classifier
        self.market_regime = "unknown"
        
        self.logger.info(f"MetaStrategy initialized with {len(strategies)} strategies")
    
    def _initialize_allocations(self) -> Dict[str, float]:
        """Initialize strategy allocations with equal weights."""
        num_strategies = len(self.strategies)
        if num_strategies == 0:
            return {}
        
        equal_weight = 1.0 / num_strategies
        return {name: equal_weight for name in self.strategies.keys()}
    
    def analyze_market_conditions(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze current market conditions to determine regime and characteristics.
        
        Args:
            data: Dictionary of DataFrames with market data for each instrument
            
        Returns:
            Dictionary with market condition analysis
        """
        market_conditions = {}
        
        # Process each instrument
        for instrument, df in data.items():
            if df is None or df.empty:
                continue
            
            # Calculate volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Determine trend strength
            sma_short = df['close'].rolling(window=20).mean()
            sma_long = df['close'].rolling(window=50).mean()
            trend_strength = (sma_short.iloc[-1] / sma_long.iloc[-1] - 1) * 100
            
            # Calculate volume profile
            avg_volume = df['volume'].mean()
            recent_volume = df['volume'].iloc[-5:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate momentum
            momentum = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1) * 100
            
            # Determine market regime
            regime = self._classify_market_regime(volatility, trend_strength, volume_ratio, momentum)
            
            market_conditions[instrument] = {
                "volatility": volatility,
                "trend_strength": trend_strength,
                "volume_ratio": volume_ratio,
                "momentum": momentum,
                "regime": regime,
                "timestamp": pd.Timestamp.now()
            }
        
        # Aggregate market conditions across instruments
        if market_conditions:
            avg_volatility = np.mean([cond["volatility"] for cond in market_conditions.values()])
            avg_trend_strength = np.mean([cond["trend_strength"] for cond in market_conditions.values()])
            avg_volume_ratio = np.mean([cond["volume_ratio"] for cond in market_conditions.values()])
            
            # Determine overall market regime
            regimes = [cond["regime"] for cond in market_conditions.values()]
            overall_regime = max(set(regimes), key=regimes.count)  # Most common regime
            
            self.market_regime = overall_regime
            self.market_conditions = {
                "overall_regime": overall_regime,
                "avg_volatility": avg_volatility,
                "avg_trend_strength": avg_trend_strength,
                "avg_volume_ratio": avg_volume_ratio,
                "instruments": market_conditions,
                "timestamp": pd.Timestamp.now()
            }
        
        self.logger.info(f"Market regime classified as: {self.market_regime}")
        return self.market_conditions
    
    def _classify_market_regime(self, volatility: float, trend_strength: float, 
                               volume_ratio: float, momentum: float) -> str:
        """
        Classify the market regime based on indicators.
        
        Returns:
            Market regime: 'trending_up', 'trending_down', 'ranging', 'volatile', 'breakout'
        """
        # Thresholds for classification
        high_volatility = 0.25  # 25% annualized volatility
        strong_trend = 5.0      # 5% difference between short and long MA
        high_volume = 1.5       # 50% higher than average volume
        
        # Classification logic
        if abs(trend_strength) > strong_trend:
            if trend_strength > 0:
                if volume_ratio > high_volume:
                    return "strong_uptrend"
                else:
                    return "uptrend"
            else:
                if volume_ratio > high_volume:
                    return "strong_downtrend"
                else:
                    return "downtrend"
        elif volatility > high_volatility:
            if volume_ratio > high_volume:
                return "volatile_breakout"
            else:
                return "volatile"
        else:
            if abs(momentum) < 2.0:
                return "ranging"
            elif momentum > 0:
                return "weak_uptrend"
            else:
                return "weak_downtrend"
    
    def evaluate_strategies(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Evaluate each strategy's suitability for current market conditions.
        
        Args:
            data: Dictionary of DataFrames with market data for each instrument
            
        Returns:
            Dictionary of strategy scores (0-1 scale)
        """
        strategy_scores = {}
        
        # Get current market regime
        if not self.market_conditions:
            self.analyze_market_conditions(data)
        
        regime = self.market_regime
        
        # Strategy-regime suitability matrix (pre-defined)
        suitability_matrix = {
            "trend_following": {
                "strong_uptrend": 0.9, "uptrend": 0.8, "weak_uptrend": 0.6,
                "strong_downtrend": 0.9, "downtrend": 0.8, "weak_downtrend": 0.6,
                "ranging": 0.2, "volatile": 0.3, "volatile_breakout": 0.5
            },
            "mean_reversion": {
                "strong_uptrend": 0.2, "uptrend": 0.3, "weak_uptrend": 0.5,
                "strong_downtrend": 0.2, "downtrend": 0.3, "weak_downtrend": 0.5,
                "ranging": 0.9, "volatile": 0.6, "volatile_breakout": 0.4
            },
            "arbitrage": {
                "strong_uptrend": 0.7, "uptrend": 0.7, "weak_uptrend": 0.7,
                "strong_downtrend": 0.7, "downtrend": 0.7, "weak_downtrend": 0.7,
                "ranging": 0.7, "volatile": 0.8, "volatile_breakout": 0.8
            },
            "flashbots": {
                "strong_uptrend": 0.6, "uptrend": 0.6, "weak_uptrend": 0.6,
                "strong_downtrend": 0.6, "downtrend": 0.6, "weak_downtrend": 0.6,
                "ranging": 0.6, "volatile": 0.8, "volatile_breakout": 0.9
            },
            "market_making": {
                "strong_uptrend": 0.4, "uptrend": 0.5, "weak_uptrend": 0.7,
                "strong_downtrend": 0.4, "downtrend": 0.5, "weak_downtrend": 0.7,
                "ranging": 0.9, "volatile": 0.5, "volatile_breakout": 0.3
            }
        }
        
        # Evaluate each strategy
        for name, strategy in self.strategies.items():
            # Get base score from suitability matrix
            strategy_type = self._get_strategy_type(name, strategy)
            base_score = suitability_matrix.get(strategy_type, {}).get(regime, 0.5)
            
            # Adjust score based on recent performance
            performance_adjustment = self._calculate_performance_adjustment(name)
            
            # Adjust score based on strategy-specific factors
            specific_adjustment = self._calculate_strategy_specific_adjustment(name, strategy, data)
            
            # Combine scores (with weights)
            final_score = (base_score * 0.6) + (performance_adjustment * 0.3) + (specific_adjustment * 0.1)
            
            # Ensure score is in 0-1 range
            final_score = max(0.0, min(1.0, final_score))
            
            strategy_scores[name] = final_score
            
            self.logger.debug(f"Strategy {name} scored {final_score:.4f} for regime {regime}")
        
        self.strategy_scores = strategy_scores
        return strategy_scores
    
    def _get_strategy_type(self, name: str, strategy: BaseStrategy) -> str:
        """Determine the type of strategy based on class name or properties."""
        class_name = strategy.__class__.__name__.lower()
        
        if "trend" in class_name:
            return "trend_following"
        elif "mean" in class_name or "reversion" in class_name:
            return "mean_reversion"
        elif "arbitrage" in class_name:
            return "arbitrage"
        elif "flash" in class_name or "mev" in class_name:
            return "flashbots"
        elif "market" in class_name and "making" in class_name:
            return "market_making"
        else:
            return "unknown"
    
    def _calculate_performance_adjustment(self, strategy_name: str) -> float:
        """Calculate adjustment factor based on recent strategy performance."""
        # Get performance history for this strategy
        history = self.performance_history.get(strategy_name, [])
        
        if not history:
            return 0.0  # No adjustment if no history
        
        # Use recent performance (last N periods)
        lookback = min(self.params["lookback_periods"], len(history))
        recent_performance = history[-lookback:]
        
        # Calculate metrics
        metric = self.params["performance_metric"]
        
        if metric == "sharpe":
            returns = [p.get("return", 0) for p in recent_performance]
            if not returns or all(r == 0 for r in returns):
                return 0.0
                
            mean_return = np.mean(returns)
            std_return = np.std(returns) or 1.0  # Avoid division by zero
            sharpe = mean_return / std_return if std_return > 0 else 0
            
            # Normalize to 0-1 range (assuming sharpe of 3 is excellent)
            normalized = min(1.0, max(0.0, sharpe / 3.0))
            return normalized
            
        elif metric == "win_rate":
            wins = sum(1 for p in recent_performance if p.get("return", 0) > 0)
            win_rate = wins / lookback if lookback > 0 else 0
            return win_rate
            
        elif metric == "profit":
            total_return = sum(p.get("return", 0) for p in recent_performance)
            # Normalize (assuming 10% total return is excellent)
            normalized = min(1.0, max(0.0, total_return / 0.1))
            return normalized
            
        return 0.0
    
    def _calculate_strategy_specific_adjustment(self, name: str, strategy: BaseStrategy, 
                                               data: Dict[str, pd.DataFrame]) -> float:
        """Calculate strategy-specific adjustment factors."""
        strategy_type = self._get_strategy_type(name, strategy)
        
        # Get market conditions
        volatility = self.market_conditions.get("avg_volatility", 0.2)
        trend_strength = self.market_conditions.get("avg_trend_strength", 0)
        volume_ratio = self.market_conditions.get("avg_volume_ratio", 1.0)
        
        # Strategy-specific adjustments
        if strategy_type == "trend_following":
            # Trend strategies do better with stronger trends
            return min(1.0, abs(trend_strength) / 10.0)
            
        elif strategy_type == "mean_reversion":
            # Mean reversion does better with lower trend strength and moderate volatility
            trend_penalty = max(0, 1.0 - abs(trend_strength) / 5.0)
            vol_factor = 1.0 - abs(volatility - 0.15) / 0.15  # Optimal around 15% volatility
            return max(0, min(1.0, (trend_penalty + vol_factor) / 2.0))
            
        elif strategy_type == "arbitrage":
            # Arbitrage does better with higher volatility across markets
            return min(1.0, volatility / 0.2)
            
        elif strategy_type == "flashbots":
            # Flashbots does better with higher volume and volatility
            return min(1.0, (volume_ratio + volatility * 5) / 2.0)
            
        elif strategy_type == "market_making":
            # Market making does better with higher volume and moderate volatility
            vol_factor = 1.0 - abs(volatility - 0.1) / 0.2  # Optimal around 10% volatility
            return min(1.0, (volume_ratio * 0.5 + vol_factor * 0.5))
            
        return 0.0
    
    def allocate_capital(self) -> Dict[str, float]:
        """
        Allocate capital across strategies based on their scores.
        
        Returns:
            Dictionary of strategy allocations (0-1 scale, summing to 1)
        """
        if not self.strategy_scores:
            return self.current_allocations
        
        allocation_method = self.params["allocation_method"]
        
        if allocation_method == "equal":
            # Equal allocation to all strategies
            active_strategies = [name for name, score in self.strategy_scores.items() 
                               if score >= self.params["confidence_threshold"]]
            
            if not active_strategies:
                return {}
                
            equal_weight = 1.0 / len(active_strategies)
            allocations = {name: equal_weight for name in active_strategies}
            
        elif allocation_method == "performance":
            # Allocate based on historical performance
            allocations = {}
            total_score = 0.0
            
            for name, strategy in self.strategies.items():
                score = self.strategy_scores.get(name, 0)
                if score >= self.params["confidence_threshold"]:
                    perf_factor = self._calculate_performance_adjustment(name)
                    # Combine score and performance
                    combined_score = score * 0.5 + perf_factor * 0.5
                    allocations[name] = combined_score
                    total_score += combined_score
            
            # Normalize allocations
            if total_score > 0:
                allocations = {name: score / total_score for name, score in allocations.items()}
            else:
                allocations = {}
                
        elif allocation_method == "adaptive":
            # Adaptive allocation based on scores and correlations
            allocations = {}
            
            # Filter strategies by confidence threshold
            candidates = {name: score for name, score in self.strategy_scores.items() 
                        if score >= self.params["confidence_threshold"]}
            
            if not candidates:
                return {}
            
            # Limit to max active strategies
            max_strategies = self.params["max_active_strategies"]
            if len(candidates) > max_strategies:
                # Keep top strategies by score
                sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                candidates = dict(sorted_candidates[:max_strategies])
            
            # Calculate correlation-adjusted scores
            adjusted_scores = self._calculate_correlation_adjusted_scores(candidates)
            
            # Normalize to get allocations
            total_score = sum(adjusted_scores.values())
            if total_score > 0:
                allocations = {name: score / total_score for name, score in adjusted_scores.items()}
            else:
                # Fallback to equal allocation
                equal_weight = 1.0 / len(candidates)
                allocations = {name: equal_weight for name in candidates}
        
        else:
            self.logger.warning(f"Unknown allocation method: {allocation_method}")
            return self.current_allocations
        
        # Apply min/max constraints
        allocations = self._apply_allocation_constraints(allocations)
        
        self.current_allocations = allocations
        self.logger.info(f"Capital allocated: {allocations}")
        
        return allocations
    
    def _calculate_correlation_adjusted_scores(self, candidates: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate correlation-adjusted scores to avoid allocating to highly correlated strategies.
        
        Args:
            candidates: Dictionary of strategy names and their scores
            
        Returns:
            Dictionary of correlation-adjusted scores
        """
        # If we don't have enough performance history, return original scores
        if not self._has_sufficient_history():
            return candidates
        
        # Calculate correlation matrix between strategies
        correlation_matrix = self._calculate_strategy_correlations()
        
        # Adjust scores based on correlations
        adjusted_scores = {}
        processed = set()
        
        # Sort strategies by score (descending)
        sorted_strategies = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        for name, score in sorted_strategies:
            if name in processed:
                continue
                
            adjusted_scores[name] = score
            processed.add(name)
            
            # Penalize correlated strategies
            for other_name, other_score in sorted_strategies:
                if other_name in processed or other_name == name:
                    continue
                    
                correlation = correlation_matrix.get((name, other_name), 0)
                if abs(correlation) >= self.params["strategy_correlation_threshold"]:
                    # Reduce score based on correlation strength
                    penalty = abs(correlation) * 0.5
                    adjusted_scores[other_name] = other_score * (1 - penalty)
                    processed.add(other_name)
        
        return adjusted_scores
    
    def _has_sufficient_history(self) -> bool:
        """Check if we have sufficient performance history for correlation analysis."""
        min_periods = 10  # Minimum periods needed
        
        for history in self.performance_history.values():
            if len(history) < min_periods:
                return False
                
        return len(self.performance_history) >= 2  # Need at least 2 strategies
    
    def _calculate_strategy_correlations(self) -> Dict[Tuple[str, str], float]:
        """Calculate correlation matrix between strategy returns."""
        correlations = {}
        
        # Extract return series for each strategy
        return_series = {}
        for name, history in self.performance_history.items():
            returns = [p.get("return", 0) for p in history]
            if returns:
                return_series[name] = returns
        
        # Calculate pairwise correlations
        for name1 in return_series:
            for name2 in return_series:
                if name1 >= name2:  # Only calculate once per pair
                    continue
                    
                # Ensure series are of equal length
                min_length = min(len(return_series[name1]), len(return_series[name2]))
                series1 = return_series[name1][-min_length:]
                series2 = return_series[name2][-min_length:]
                
                if min_length >= 5:  # Need at least 5 points for meaningful correlation
                    correlation = np.corrcoef(series1, series2)[0, 1]
                    correlations[(name1, name2)] = correlation
                    correlations[(name2, name1)] = correlation
        
        return correlations
    
    def _apply_allocation_constraints(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum and maximum allocation constraints."""
        min_allocation = self.params["min_allocation"]
        max_allocation = self.params["max_allocation"]
        
        # Apply min/max constraints
        constrained = {}
        for name, allocation in allocations.items():
            if allocation > 0:
                constrained[name] = max(min_allocation, min(max_allocation, allocation))
            else:
                constrained[name] = 0
        
        # Normalize to ensure sum is 1.0
        total = sum(constrained.values())
        if total > 0:
            normalized = {name: alloc / total for name, alloc in constrained.items()}
            return normalized
        
        return constrained
    
    def collect_strategy_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """
        Collect signals from all strategies.
        
        Args:
            data: Dictionary of DataFrames with market data for each instrument
            
        Returns:
            Dictionary of strategy signals
        """
        signals = {}
        
        for name, strategy in self.strategies.items():
            try:
                # Only analyze strategies with non-zero allocation
                if self.current_allocations.get(name, 0) > 0:
                    # Run strategy analysis
                    analysis = strategy.analyze(data)
                    
                    # Generate signals
                    strategy_signals = strategy.generate_signals(analysis)
                    
                    signals[name] = strategy_signals
                    self.logger.debug(f"Strategy {name} generated {len(strategy_signals)} signals")
            except Exception as e:
                self.logger.error(f"Error collecting signals from {name}: {str(e)}", exc_info=True)
                signals[name] = []
        
        self.strategy_signals = signals
        return signals
    
    def combine_signals(self) -> List[Dict]:
        """
        Combine signals from multiple strategies based on allocations.
        
        Returns:
            List of combined trading signals
        """
        if not self.strategy_signals:
            return []
            
        if not self.params["enable_strategy_combination"]:
            # Use signals from the highest-allocated strategy only
            best_strategy = max(self.current_allocations.items(), key=lambda x: x[1])[0]
            return self.strategy_signals.get(best_strategy, [])
        
        combination_method = self.params["combination_method"]
        combined_signals = []
        
        if combination_method == "best_only":
            # Use signals from the highest-allocated strategy only
            best_strategy = max(self.current_allocations.items(), key=lambda x: x[1])[0]
            return self.strategy_signals.get(best_strategy, [])
            
        elif combination_method == "voting":
            # Implement voting-based signal combination
            combined_signals = self._combine_signals_by_voting()
            
        elif combination_method == "weighted":
            # Implement weighted signal combination
            combined_signals = self._combine_signals_by_weighting()
            
        else:
            self.logger.warning(f"Unknown signal combination method: {combination_method}")
            return []
        
        self.combined_signals = combined_signals
        return combined_signals
    
    def _combine_signals_by_voting(self) -> List[Dict]:
        """Combine signals using a voting mechanism."""
        # Group signals by instrument and action
        signal_votes = defaultdict(lambda: defaultdict(int))
        signal_details = defaultdict(lambda: defaultdict(list))
        
        # Count votes for each instrument-action pair
        for strategy_name, signals in self.strategy_signals.items():
            allocation = self.current_allocations.get(strategy_name, 0)
            if allocation <= 0:
                continue
                
            for signal in signals:
                instrument = signal.get("instrument")
                action = signal.get("action")
                
                if not instrument or not action:
                    continue
                
                # Each strategy gets one vote per instrument-action
                signal_votes[instrument][action] += 1
                signal_details[instrument][action].append({
                    "strategy": strategy_name,
                    "allocation": allocation,
                    "signal": signal
                })
        
        # Generate combined signals based on votes
        combined = []
        
        for instrument, actions in signal_votes.items():
            if not actions:
                continue
                
            # Find action with most votes
            best_action = max(actions.items(), key=lambda x: x[1])[0]
            
            # Only use if at least 2 strategies agree (or we only have one strategy)
            if actions[best_action] >= 2 or len(self.strategy_signals) == 1:
                # Get details of the winning signals
                winning_signals = signal_details[instrument][best_action]
                
                # Create a combined signal using the first signal as template
                if winning_signals:
                    template = winning_signals[0]["signal"].copy()
                    template["strategies"] = [s["strategy"] for s in winning_signals]
                    template["vote_count"] = actions[best_action]
                    template["total_allocation"] = sum(s["allocation"] for s in winning_signals)
                    
                    # Average the quantity if present
                    if "quantity" in template:
                        quantities = [s["signal"].get("quantity", 0) for s in winning_signals]
                        template["quantity"] = sum(quantities) / len(quantities)
                    
                    combined.append(template)
        
        return combined
    
    def _combine_signals_by_weighting(self) -> List[Dict]:
        """Combine signals using allocation-weighted averaging."""
        # Group signals by instrument
        instrument_signals = defaultdict(list)
        
        # Collect signals by instrument
        for strategy_name, signals in self.strategy_signals.items():
            allocation = self.current_allocations.get(strategy_name, 0)
            if allocation <= 0:
                continue
                
            for signal in signals:
                instrument = signal.get("instrument")
                if not instrument:
                    continue
                
                instrument_signals[instrument].append({
                    "strategy": strategy_name,
                    "allocation": allocation,
                    "signal": signal
                })
        
        # Generate combined signals for each instrument
        combined = []
        
        for instrument, signals in instrument_signals.items():
            if not signals:
                continue
                
            # Group by action
            action_signals = defaultdict(list)
            for s in signals:
                action = s["signal"].get("action")
                if action:
                    action_signals[action].append(s)
            
            # For each action, create a weighted signal
            for action, action_group in action_signals.items():
                # Skip if only one strategy is signaling this action
                if len(action_group) < 2 and len(self.strategy_signals) > 1:
                    continue
                
                # Calculate total allocation for this action
                total_allocation = sum(s["allocation"] for s in action_group)
                
                # Create a combined signal using the first signal as template
                template = action_group[0]["signal"].copy()
                template["strategies"] = [s["strategy"] for s in action_group]
                template["total_allocation"] = total_allocation
                
                # Weighted average for quantity
                if "quantity" in template:
                    weighted_quantity = sum(s["signal"].get("quantity", 0) * s["allocation"] 
                                          for s in action_group) / total_allocation
                    template["quantity"] = weighted_quantity
                
                # Weighted average for price if it's a limit order
                if template.get("order_type") == "LIMIT" and "price" in template:
                    weighted_price = sum(s["signal"].get("price", 0) * s["allocation"] 
                                       for s in action_group) / total_allocation
                    template["price"] = weighted_price
                
                combined.append(template)
        
        return combined
    
    def update_performance(self, strategy_name: str, performance_data: Dict) -> None:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            performance_data: Dictionary with performance metrics
        """
        if strategy_name not in self.performance_history:
            self.performance_history[strategy_name] = []
        
        # Add timestamp if not present
        if "timestamp" not in performance_data:
            performance_data["timestamp"] = pd.Timestamp.now()
        
        self.performance_history[strategy_name].append(performance_data)
        
        # Trim history if too long
        max_history = 100  # Keep last 100 periods
        if len(self.performance_history[strategy_name]) > max_history:
            self.performance_history[strategy_name] = self.performance_history[strategy_name][-max_history:]
    
    def should_rebalance(self) -> bool:
        """Check if it's time to rebalance strategy allocations."""
        # Get rebalance frequency in hours
        frequency_hours = self.params["rebalance_frequency"]
        
        # Check if enough time has passed since last rebalance
        time_since_rebalance = (datetime.now() - self.last_rebalance).total_seconds() / 3600
        return time_since_rebalance >= frequency_hours
    
    def execute(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Execute the meta-strategy process.
        
        Args:
            data: Dictionary of DataFrames with market data for each instrument
            
        Returns:
            List of combined trading signals
        """
        # Step 1: Analyze market conditions
        self.analyze_market_conditions(data)
        
        # Step 2: Check if rebalancing is needed
        if self.should_rebalance():
            # Step 3: Evaluate strategies
            self.evaluate_strategies(data)
            
            # Step 4: Allocate capital
            self.allocate_capital()
            
            # Update rebalance timestamp
            self.last_rebalance = datetime.now()
        
        # Step 5: Collect signals from strategies
        self.collect_strategy_signals(data)
        
        # Step 6: Combine signals
        combined_signals = self.combine_signals()
        
        return combined_signals
    
    def get_strategy_allocations(self) -> Dict[str, float]:
        """Get current strategy allocations."""
        return self.current_allocations
    
    def get_market_conditions(self) -> Dict:
        """Get current market condition analysis."""
        return self.market_conditions
    
    def get_strategy_scores(self) -> Dict[str, float]:
        """Get current strategy suitability scores."""
        return self.strategy_scores
    
    def get_performance_history(self) -> Dict[str, List[Dict]]:
        """Get performance history for all strategies."""
        return self.performance_history
    
    def get_combined_signals(self) -> List[Dict]:
        """Get the most recently combined signals."""
        return self.combined_signals