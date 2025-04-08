import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy that identifies and trades price deviations from the mean.
    
    This strategy uses statistical measures like z-scores, Bollinger Bands, and RSI
    to identify overbought and oversold conditions for mean reversion trades.
    """
    
    def __init__(self, name: str = "mean_reversion", params: Dict = None):
        """
        Initialize the mean reversion strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters including:
                - lookback_period: Period for calculating mean and standard deviation
                - entry_z_score: Z-score threshold for trade entry
                - exit_z_score: Z-score threshold for trade exit
                - max_holding_period: Maximum number of periods to hold a position
                - stop_loss_std_multiple: Multiple of standard deviation for stop loss
                - risk_per_trade: Percentage of capital to risk per trade
                - max_positions: Maximum number of open positions
        """
        # Default parameters
        default_params = {
            "lookback_period": 20,
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "max_holding_period": 10,
            "stop_loss_std_multiple": 3.0,
            "risk_per_trade": 0.02,  # 2% risk per trade
            "max_positions": 5,
            "use_bollinger_bands": True,
            "bollinger_std": 2.0,
            "use_rsi": True,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "use_volume_filter": True,
            "volume_threshold": 1.0,  # Volume should be at least average
            "use_volatility_filter": True,
            "volatility_threshold": 1.5,  # Max volatility threshold
            "position_sizing_method": "risk_based",  # 'risk_based', 'equal', 'kelly'
            "max_risk_per_asset_class": 0.1,  # 10% max risk per asset class
            "use_correlation_filter": True,  # Filter based on correlation
            "correlation_threshold": 0.7,  # Correlation threshold
            "use_regime_filter": True,  # Filter based on market regime
            "regime_indicators": ["volatility", "trend_strength"]  # Indicators for regime detection
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize state variables
        self.positions = {}  # Current positions {instrument: quantity}
        self.open_trades = {}  # Open trades with metadata
        self.trade_history = []  # History of closed trades
        self.indicators = {}  # Calculated indicators
        self.market_regimes = {}  # Current market regimes
        self.holding_periods = {}  # Track holding periods for positions
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit_per_trade": 0.0,
            "avg_loss_per_trade": 0.0,
            "risk_reward_ratio": 0.0
        }
    
    def analyze(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze market data and calculate indicators.
        
        Args:
            data: Dictionary of DataFrames with market data for each instrument
            
        Returns:
            Dictionary with analysis results
        """
        analysis_results = {}
        
        for instrument, df in data.items():
            if df is None or df.empty:
                continue
            
            # Make a copy to avoid modifying the original data
            analysis_df = df.copy()
            
            # Calculate rolling mean and standard deviation
            analysis_df['rolling_mean'] = analysis_df['close'].rolling(window=self.params["lookback_period"]).mean()
            analysis_df['rolling_std'] = analysis_df['close'].rolling(window=self.params["lookback_period"]).std()
            
            # Calculate z-score
            analysis_df['z_score'] = (analysis_df['close'] - analysis_df['rolling_mean']) / analysis_df['rolling_std']
            
            # Calculate Bollinger Bands if enabled
            if self.params["use_bollinger_bands"]:
                std_multiplier = self.params["bollinger_std"]
                analysis_df['upper_band'] = analysis_df['rolling_mean'] + (analysis_df['rolling_std'] * std_multiplier)
                analysis_df['lower_band'] = analysis_df['rolling_mean'] - (analysis_df['rolling_std'] * std_multiplier)
                analysis_df['percent_b'] = (analysis_df['close'] - analysis_df['lower_band']) / (analysis_df['upper_band'] - analysis_df['lower_band'])
            
            # Calculate RSI if enabled
            if self.params["use_rsi"]:
                delta = analysis_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.params["rsi_period"]).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.params["rsi_period"]).mean()
                
                # Calculate RS and RSI
                rs = gain / loss
                analysis_df['rsi'] = 100 - (100 / (1 + rs))
                
                # Calculate RSI signals
                analysis_df['rsi_overbought'] = analysis_df['rsi'] > self.params["rsi_overbought"]
                analysis_df['rsi_oversold'] = analysis_df['rsi'] < self.params["rsi_oversold"]
            
            # Calculate volatility
            analysis_df['volatility'] = analysis_df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # Calculate volume ratio
            analysis_df['volume_sma'] = analysis_df['volume'].rolling(window=20).mean()
            analysis_df['volume_ratio'] = analysis_df['volume'] / analysis_df['volume_sma']
            
            # Calculate trend strength (using simple moving average)
            analysis_df['sma_20'] = analysis_df['close'].rolling(window=20).mean()
            analysis_df['sma_50'] = analysis_df['close'].rolling(window=50).mean()
            analysis_df['trend_strength'] = abs(analysis_df['sma_20'] / analysis_df['sma_50'] - 1)
            
            # Detect market regime
            analysis_df['market_regime'] = self._detect_market_regime(analysis_df)
            
            # Calculate mean reversion signals
            analysis_df['mean_reversion_signal'] = self._calculate_mean_reversion_signal(analysis_df)
            
            # Store the results
            analysis_results[instrument] = {
                'df': analysis_df,
                'current': {
                    'close': analysis_df['close'].iloc[-1],
                    'rolling_mean': analysis_df['rolling_mean'].iloc[-1],
                    'rolling_std': analysis_df['rolling_std'].iloc[-1],
                    'z_score': analysis_df['z_score'].iloc[-1],
                    'volatility': analysis_df['volatility'].iloc[-1],
                    'volume_ratio': analysis_df['volume_ratio'].iloc[-1],
                    'trend_strength': analysis_df['trend_strength'].iloc[-1],
                    'market_regime': analysis_df['market_regime'].iloc[-1],
                    'mean_reversion_signal': analysis_df['mean_reversion_signal'].iloc[-1]
                }
            }
            
            # Add Bollinger Bands data if enabled
            if self.params["use_bollinger_bands"]:
                analysis_results[instrument]['current'].update({
                    'upper_band': analysis_df['upper_band'].iloc[-1],
                    'lower_band': analysis_df['lower_band'].iloc[-1],
                    'percent_b': analysis_df['percent_b'].iloc[-1]
                })
            
            # Add RSI data if enabled
            if self.params["use_rsi"]:
                analysis_results[instrument]['current'].update({
                    'rsi': analysis_df['rsi'].iloc[-1],
                    'rsi_overbought': analysis_df['rsi_overbought'].iloc[-1],
                    'rsi_oversold': analysis_df['rsi_oversold'].iloc[-1]
                })
            
            # Update market regime
            self.market_regimes[instrument] = analysis_df['market_regime'].iloc[-1]
            
            # Store indicators for this instrument
            self.indicators[instrument] = analysis_results[instrument]['current']
        
        return analysis_results
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime based on indicators.
        
        Returns:
            Series with market regime classifications:
            - 'trending'
            - 'ranging'
            - 'volatile'
            - 'stable'
        """
        regimes = pd.Series(index=df.index, dtype='object')
        
        # Trending: High trend strength
        trending = df['trend_strength'] > 0.05
        
        # Ranging: Low trend strength, low volatility
        ranging = (df['trend_strength'] <= 0.05) & (df['volatility'] < 0.15)
        
        # Volatile: High volatility
        volatile = df['volatility'] >= 0.25
        
        # Stable: Low volatility, moderate trend strength
        stable = (df['volatility'] < 0.15) & (df['trend_strength'] <= 0.03)
        
        # Assign regimes
        regimes[trending] = 'trending'
        regimes[ranging & ~trending] = 'ranging'
        regimes[volatile] = 'volatile'
        regimes[stable & ~(trending | ranging | volatile)] = 'stable'
        
        # Fill any remaining NaN values with 'unknown'
        regimes = regimes.fillna('unknown')
        
        return regimes
    
    def _calculate_mean_reversion_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate mean reversion signal based on multiple indicators.
        
        Returns:
            Series with mean reversion signals: 1 (buy), -1 (sell), 0 (neutral)
        """
        signals = pd.Series(0, index=df.index)
        
        # Z-score signal
        z_score_signal = pd.Series(0, index=df.index)
        z_score_signal[df['z_score'] < -self.params["entry_z_score"]] = 1  # Buy when price is below mean
        z_score_signal[df['z_score'] > self.params["entry_z_score"]] = -1  # Sell when price is above mean
        
        # Bollinger Bands signal if enabled
        bb_signal = pd.Series(0, index=df.index)
        if self.params["use_bollinger_bands"]:
            bb_signal[df['percent_b'] < 0] = 1  # Buy when price is below lower band
            bb_signal[df['percent_b'] > 1] = -1  # Sell when price is above upper band
        
        # RSI signal if enabled
        rsi_signal = pd.Series(0, index=df.index)
        if self.params["use_rsi"]:
            rsi_signal[df['rsi_oversold']] = 1  # Buy when RSI is oversold
            rsi_signal[df['rsi_overbought']] = -1  # Sell when RSI is overbought
        
        # Combine signals with weights
        if self.params["use_bollinger_bands"] and self.params["use_rsi"]:
            signals = (
                z_score_signal * 0.4 +  # 40% weight to z-score
                bb_signal * 0.3 +  # 30% weight to Bollinger Bands
                rsi_signal * 0.3  # 30% weight to RSI
            )
        elif self.params["use_bollinger_bands"]:
            signals = (
                z_score_signal * 0.6 +  # 60% weight to z-score
                bb_signal * 0.4  # 40% weight to Bollinger Bands
            )
        elif self.params["use_rsi"]:
            signals = (
                z_score_signal * 0.6 +  # 60% weight to z-score
                rsi_signal * 0.4  # 40% weight to RSI
            )
        else:
            signals = z_score_signal  # 100% weight to z-score
        
        # Discretize to -1, 0, 1
        final_signals = pd.Series(0, index=df.index)
        final_signals[signals > 0.3] = 1  # Strong buy signal
        final_signals[signals < -0.3] = -1  # Strong sell signal
        
        return final_signals
    
    def generate_signals(self, analysis_results: Dict[str, Any]) -> List[Dict]:
        """
        Generate trading signals based on analysis results.
        
        Args:
            analysis_results: Dictionary with analysis results from analyze()
            
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        for instrument, analysis in analysis_results.items():
            current = analysis['current']
            
            # Skip if we don't have enough data
            if pd.isna(current['rolling_mean']) or pd.isna(current['rolling_std']):
                continue
            
            # Get current position for this instrument
            current_position = self.positions.get(instrument, 0)
            
            # Check if we should apply regime filter
            apply_signal = True
            if self.params["use_regime_filter"]:
                regime = current['market_regime']
                # Only trade in ranging or stable regimes
                if regime not in ['ranging', 'stable']:
                    apply_signal = False
            
            # Check if we should apply volume filter
            if apply_signal and self.params["use_volume_filter"]:
                if current['volume_ratio'] < self.params["volume_threshold"]:
                    apply_signal = False
            
            # Check if we should apply volatility filter
            if apply_signal and self.params["use_volatility_filter"]:
                if current['volatility'] > self.params["volatility_threshold"]:
                    apply_signal = False
            
            # Generate signals based on mean reversion signal
            mean_reversion_signal = current['mean_reversion_signal']
            
            if apply_signal:
                # Long entry signal (buy when price is below mean)
                if mean_reversion_signal == 1 and current_position <= 0:
                    # Check if we can add a new position (max positions constraint)
                    if len(self.positions) < self.params["max_positions"] or current_position < 0:
                        # Calculate position size
                        position_size = self._calculate_position_size(instrument, 'long', current['close'], current['rolling_std'])
                        
                        # Calculate stop loss and take profit
                        stop_loss = current['close'] - (current['rolling_std'] * self.params["stop_loss_std_multiple"])
                        take_profit = current['rolling_mean']  # Target the mean
                        
                        # Generate long entry signal
                        signals.append({
                            'instrument': instrument,
                            'timestamp': pd.Timestamp.now(),
                            'action': 'BUY',
                            'type': 'ENTRY',
                            'price': current['close'],
                            'quantity': position_size,
                            'order_type': 'LIMIT',
                            'reason': f"Mean reversion long entry: z-score={current['z_score']:.2f}, " + 
                                     (f"RSI={current['rsi']:.2f}, " if self.params["use_rsi"] else "") +
                                     (f"percent_b={current['percent_b']:.2f}" if self.params["use_bollinger_bands"] else ""),
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'strategy': self.name
                        })
                        
                        # Track holding period
                        self.holding_periods[instrument] = 0
                
                # Short entry signal (sell when price is above mean)
                elif mean_reversion_signal == -1 and current_position >= 0:
                    # Check if we can add a new position (max positions constraint)
                    if len(self.positions) < self.params["max_positions"] or current_position > 0:
                        # Calculate position size
                        position_size = self._calculate_position_size(instrument, 'short', current['close'], current['rolling_std'])
                        
                        # Calculate stop loss and take profit
                        stop_loss = current['close'] + (current['rolling_std'] * self.params["stop_loss_std_multiple"])
                        take_profit = current['rolling_mean']  # Target the mean
                        
                        # Generate short entry signal
                        signals.append({
                            'instrument': instrument,
                            'timestamp': pd.Timestamp.now(),
                            'action': 'SELL',
                            'type': 'ENTRY',
                            'price': current['close'],
                            'quantity': position_size,
                            'order_type': 'LIMIT',
                            'reason': f"Mean reversion short entry: z-score={current['z_score']:.2f}, " + 
                                     (f"RSI={current['rsi']:.2f}, " if self.params["use_rsi"] else "") +
                                     (f"percent_b={current['percent_b']:.2f}" if self.params["use_bollinger_bands"] else ""),
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'strategy': self.name
                        })
                        
                        # Track holding period
                        self.holding_periods[instrument] = 0
            
            # Exit signals (these apply regardless of regime filters)
            
            # Exit long position
            if current_position > 0:
                # Exit when price reverts to mean (z-score near zero)
                if abs(current['z_score']) < self.params["exit_z_score"]:
                    signals.append({
                        'instrument': instrument,
                        'timestamp': pd.Timestamp.now(),
                        'action': 'SELL',
                        'type': 'EXIT',
                        'price': current['close'],
                        'quantity': abs(current_position),
                        'order_type': 'MARKET',
                        'reason': f"Mean reversion long exit: z-score={current['z_score']:.2f} (mean reversion complete)",
                        'strategy': self.name
                    })
                    
                    # Reset holding period
                    if instrument in self.holding_periods:
                        del self.holding_periods[instrument]
                
                # Exit on max holding period
                elif instrument in self.holding_periods and self.holding_periods[instrument] >= self.params["max_holding_period"]:
                    signals.append({
                        'instrument': instrument,
                        'timestamp': pd.Timestamp.now(),
                        'action': 'SELL',
                        'type': 'EXIT',
                        'price': current['close'],
                        'quantity': abs(current_position),
                        'order_type': 'MARKET',
                        'reason': f"Mean reversion long exit: Max holding period reached ({self.params['max_holding_period']} periods)",
                        'strategy': self.name
                    })
                    
                    # Reset holding period
                    if instrument in self.holding_periods:
                        del self.holding_periods[instrument]
            
            # Exit short position
            elif current_position < 0:
                # Exit when price reverts to mean (z-score near zero)
                if abs(current['z_score']) < self.params["exit_z_score"]:
                    signals.append({
                        'instrument': instrument,
                        'timestamp': pd.Timestamp.now(),
                        'action': 'BUY',
                        'type': 'EXIT',
                        'price': current['close'],
                        'quantity': abs(current_position),
                        'order_type': 'MARKET',
                        'reason': f"Mean reversion short exit: z-score={current['z_score']:.2f} (mean reversion complete)",
                        'strategy': self.name
                    })
                    
                    # Reset holding period
                    if instrument in self.holding_periods:
                        del self.holding_periods[instrument]
                
                # Exit on max holding period
                elif instrument in self.holding_periods and self.holding_periods[instrument] >= self.params["max_holding_period"]:
                    signals.append({
                        'instrument': instrument,
                        'timestamp': pd.Timestamp.now(),
                        'action': 'BUY',
                        'type': 'EXIT',
                        'price': current['close'],
                        'quantity': abs(current_position),
                        'order_type': 'MARKET',
                        'reason': f"Mean reversion short exit: Max holding period reached ({self.params['max_holding_period']} periods)",
                        'strategy': self.name
                    })
                    
                    # Reset holding period
                    if instrument in self.holding_periods:
                        del self.holding_periods[instrument]
            
            # Increment holding period for this instrument if we have a position
            if instrument in self.positions and instrument in self.holding_periods:
                self.holding_periods[instrument] += 1
        
        return signals
    
    def _calculate_position_size(self, instrument: str, direction: str, price: float, std_dev: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            instrument: Instrument to trade
            direction: 'long' or 'short'
            price: Current price
            std_dev: Standard deviation of price
            
        Returns:
            Position size in units of the instrument
        """
        method = self.params["position_sizing_method"]
        
        if method == "risk_based":
            # Risk-based position sizing
            account_balance = 10000.0  # This should come from the portfolio manager
            risk_amount = account_balance * self.params["risk_per_trade"]
            
            # Calculate stop loss distance
            stop_distance = std_dev * self.params["stop_loss_std_multiple"]
            
            # Calculate position size based on risk
            if stop_distance > 0:
                position_size = risk_amount / stop_distance
            else:
                position_size = 0
            
            # Convert to quantity based on price
            if price > 0:
                quantity = position_size / price
            else:
                quantity = 0
            
            return quantity
            
        elif method == "equal":
            # Equal position sizing
            account_balance = 10000.0  # This should come from the portfolio manager
            position_value = account_balance / self.params["max_positions"]
            
            # Convert to quantity based on price
            if price > 0:
                quantity = position_value / price
            else:
                quantity = 0
            
            return quantity
            
        elif method == "kelly":
            # Kelly criterion position sizing
            # This is a simplified implementation
            win_rate = max(0.1, self.performance_metrics.get("win_rate", 0.5))
            avg_win = max(0.01, self.performance_metrics.get("avg_profit_per_trade", 0.02))
            avg_loss = max(0.01, self.performance_metrics.get("avg_loss_per_trade", 0.01))
            
            # Kelly formula: f = (p*b - q) / b where p=win rate, q=loss rate, b=win/loss ratio
            if avg_loss > 0:
                win_loss_ratio = avg_win / avg_loss
                loss_rate = 1 - win_rate
                kelly_fraction = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
                
                # Limit kelly fraction to avoid excessive risk
                kelly_fraction = max(0, min(0.2, kelly_fraction))
                
                account_balance = 10000.0  # This should come from the portfolio manager
                position_value = account_balance * kelly_fraction
                
                # Convert to quantity based on price
                if price > 0:
                    quantity = position_value / price
                else:
                    quantity = 0
                
                return quantity
            else:
                return 0
        
        # Default fallback
        return 1.0
    
    def update_position(self, instrument: str, quantity: float) -> None:
        """
        Update the position for an instrument.
        
        Args:
            instrument: Instrument identifier
            quantity: Quantity to add (positive) or subtract (negative)
        """
        current_position = self.positions.get(instrument, 0)
        new_position = current_position + quantity
        
        if new_position == 0:
            # Position closed, remove from positions
            if instrument in self.positions:
                del self.positions[instrument]
        else:
            # Update position
            self.positions[instrument] = new_position
    
    def update_trade(self, trade_update: Dict) -> None:
        """
        Update trade information.
        
        Args:
            trade_update: Dictionary with trade update information
        """
        instrument = trade_update.get('instrument')
        trade_id = trade_update.get('trade_id')
        status = trade_update.get('status')
        
        if not instrument or not trade_id:
            return
        
        if status == 'FILLED':
            # Update position
            quantity = trade_update.get('quantity', 0)
            if trade_update.get('action') == 'SELL':
                quantity = -quantity
            
            self.update_position(instrument, quantity)
            
            # Update open trades
            if trade_update.get('type') == 'ENTRY':
                self.open_trades[trade_id] = {
                    'instrument': instrument,
                    'entry_price': trade_update.get('price', 0),
                    'quantity': abs(quantity),
                    'direction': 'long' if quantity > 0 else 'short',
                    'entry_time': trade_update.get('timestamp', pd.Timestamp.now()),
                    'stop_loss': trade_update.get('stop_loss'),
                    'take_profit': trade_update.get('take_profit')
                }
            elif trade_update.get('type') == 'EXIT':
                # Find the corresponding entry trade
                entry_trade_id = trade_update.get('entry_trade_id')
                if entry_trade_id and entry_trade_id in self.open_trades:
                    entry_trade = self.open_trades[entry_trade_id]
                    
                    # Calculate profit/loss
                    entry_price = entry_trade.get('entry_price', 0)
                    exit_price = trade_update.get('price', 0)
                    quantity = entry_trade.get('quantity', 0)
                    direction = entry_trade.get('direction', 'long')
                    
                    if direction == 'long':
                        pnl = (exit_price - entry_price) * quantity
                    else:
                        pnl = (entry_price - exit_price) * quantity
                    
                    # Record trade in history
                    self.trade_history.append({
                        'instrument': instrument,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'direction': direction,
                        'entry_time': entry_trade.get('entry_time'),
                        'exit_time': trade_update.get('timestamp', pd.Timestamp.now()),
                        'pnl': pnl,
                        'pnl_percent': pnl / (entry_price * quantity) if entry_price > 0 and quantity > 0 else 0,
                        'reason': trade_update.get('reason', 'Unknown')
                    })
                    
                    # Update performance metrics
                    self._update_performance_metrics(pnl, pnl > 0)
                    
                    # Remove from open trades
                    del self.open_trades[entry_trade_id]
    
    def _update_performance_metrics(self, pnl: float, is_win: bool) -> None:
        """
        Update strategy performance metrics.
        
        Args:
            pnl: Profit/loss amount
            is_win: Whether the trade was a win
        """
        self.performance_metrics["total_trades"] += 1
        
        if is_win:
            self.performance_metrics["winning_trades"] += 1
            self.performance_metrics["total_profit"] += pnl
        else:
            self.performance_metrics["losing_trades"] += 1
            self.performance_metrics["total_loss"] -= pnl  # Convert to positive number
        
        # Calculate win rate
        if self.performance_metrics["total_trades"] > 0:
            self.performance_metrics["win_rate"] = (
                self.performance_metrics["winning_trades"] / self.performance_metrics["total_trades"]
            )
        
        # Calculate profit factor
        if self.performance_metrics["total_loss"] > 0:
            self.performance_metrics["profit_factor"] = (
                self.performance_metrics["total_profit"] / self.performance_metrics["total_loss"]
            )
        
        # Calculate average profit/loss per trade
        if self.performance_metrics["winning_trades"] > 0:
            self.performance_metrics["avg_profit_per_trade"] = (
                self.performance_metrics["total_profit"] / self.performance_metrics["winning_trades"]
            )
        
        if self.performance_metrics["losing_trades"] > 0:
            self.performance_metrics["avg_loss_per_trade"] = (
                self.performance_metrics["total_loss"] / self.performance_metrics["losing_trades"]
            )
        
        # Calculate risk-reward ratio
        if self.performance_metrics["avg_loss_per_trade"] > 0:
            self.performance_metrics["risk_reward_ratio"] = (
                self.performance_metrics["avg_profit_per_trade"] / self.performance_metrics["avg_loss_per_trade"]
            )
        
        # Calculate max drawdown (simplified)
        # For a more accurate calculation, we would need to track equity curve
        cumulative_pnl = self.performance_metrics["total_profit"] - self.performance_metrics["total_loss"]
        if cumulative_pnl < 0 and abs(cumulative_pnl) > self.performance_metrics["max_drawdown"]:
            self.performance_metrics["max_drawdown"] = abs(cumulative_pnl)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_metrics
    
    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions.
        
        Returns:
            Dictionary mapping instruments to position sizes
        """
        return self.positions
    
    def get_open_trades(self) -> Dict[str, Dict]:
        """
        Get open trades.
        
        Returns:
            Dictionary mapping trade IDs to trade details
        """
        return self.open_trades
    
    def get_trade_history(self) -> List[Dict]:
        """
        Get trade history.
        
        Returns:
            List of closed trades
        """
        return self.trade_history
    
    def get_market_regimes(self) -> Dict[str, str]:
        """
        Get current market regimes.
        
        Returns:
            Dictionary mapping instruments to market regime classifications
        """
        return self.market_regimes
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.positions = {}
        self.open_trades = {}
        self.indicators = {}
        self.market_regimes = {}
        self.holding_periods = {}
        # Keep trade history and performance metrics