import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy that identifies and follows market trends.
    
    This strategy uses moving averages, RSI, and other technical indicators
    to identify trends and generate trading signals.
    """
    
    def __init__(self, name: str = "trend_following", params: Dict = None):
        """
        Initialize the trend following strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters including:
                - short_window: Short-term moving average window
                - long_window: Long-term moving average window
                - rsi_period: RSI calculation period
                - rsi_overbought: RSI overbought threshold
                - rsi_oversold: RSI oversold threshold
                - risk_per_trade: Percentage of capital to risk per trade
                - trailing_stop_atr_multiple: Multiple of ATR for trailing stops
                - take_profit_atr_multiple: Multiple of ATR for take profit
                - max_positions: Maximum number of open positions
        """
        # Default parameters
        default_params = {
            "short_window": 20,
            "long_window": 50,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "risk_per_trade": 0.02,  # 2% risk per trade
            "trailing_stop_atr_multiple": 2.0,
            "take_profit_atr_multiple": 3.0,
            "max_positions": 5,
            "atr_period": 14,
            "use_volume_filter": True,
            "volume_threshold": 1.5,  # Volume should be 1.5x average
            "use_volatility_filter": True,
            "volatility_threshold": 1.2,  # Volatility should be 1.2x average
            "trend_confirmation_threshold": 0.5,  # Minimum trend strength
            "position_sizing_method": "risk_based",  # 'risk_based', 'equal', 'kelly'
            "max_risk_per_asset_class": 0.1,  # 10% max risk per asset class
            "use_pyramiding": False,  # Whether to add to winning positions
            "pyramiding_levels": 3,  # Maximum number of entries in same direction
            "use_correlation_filter": True,  # Filter based on correlation
            "correlation_threshold": 0.7,  # Correlation threshold
            "use_regime_filter": True,  # Filter based on market regime
            "regime_indicators": ["atr", "adx", "volatility"]  # Indicators for regime detection
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
            
            # Calculate moving averages
            analysis_df['sma_short'] = analysis_df['close'].rolling(window=self.params["short_window"]).mean()
            analysis_df['sma_long'] = analysis_df['close'].rolling(window=self.params["long_window"]).mean()
            
            # Calculate moving average crossover
            analysis_df['ma_crossover'] = analysis_df['sma_short'] - analysis_df['sma_long']
            analysis_df['ma_crossover_signal'] = np.where(analysis_df['ma_crossover'] > 0, 1, 
                                                         np.where(analysis_df['ma_crossover'] < 0, -1, 0))
            
            # Calculate RSI
            delta = analysis_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.params["rsi_period"]).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.params["rsi_period"]).mean()
            
            # Calculate RS and RSI
            rs = gain / loss
            analysis_df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate RSI signals
            analysis_df['rsi_overbought'] = analysis_df['rsi'] > self.params["rsi_overbought"]
            analysis_df['rsi_oversold'] = analysis_df['rsi'] < self.params["rsi_oversold"]
            
            # Calculate ATR for volatility measurement
            high_low = analysis_df['high'] - analysis_df['low']
            high_close = np.abs(analysis_df['high'] - analysis_df['close'].shift())
            low_close = np.abs(analysis_df['low'] - analysis_df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            analysis_df['atr'] = true_range.rolling(window=self.params["atr_period"]).mean()
            
            # Calculate ADX for trend strength
            plus_dm = np.where((analysis_df['high'] - analysis_df['high'].shift()) > 
                              (analysis_df['low'].shift() - analysis_df['low']),
                              np.maximum(analysis_df['high'] - analysis_df['high'].shift(), 0), 0)
            minus_dm = np.where((analysis_df['low'].shift() - analysis_df['low']) > 
                               (analysis_df['high'] - analysis_df['high'].shift()),
                               np.maximum(analysis_df['low'].shift() - analysis_df['low'], 0), 0)
            
            tr = true_range
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=self.params["atr_period"]).mean() / 
                            analysis_df['atr'])
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=self.params["atr_period"]).mean() / 
                             analysis_df['atr'])
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            analysis_df['adx'] = dx.rolling(window=self.params["atr_period"]).mean()
            
            # Volume analysis
            analysis_df['volume_sma'] = analysis_df['volume'].rolling(window=20).mean()
            analysis_df['volume_ratio'] = analysis_df['volume'] / analysis_df['volume_sma']
            
            # Trend strength calculation
            analysis_df['trend_strength'] = analysis_df['adx'] / 100.0
            
            # Volatility calculation
            analysis_df['volatility'] = analysis_df['atr'] / analysis_df['close']
            
            # Detect market regime
            analysis_df['market_regime'] = self._detect_market_regime(analysis_df)
            
            # Calculate trend signals
            analysis_df['trend_signal'] = self._calculate_trend_signal(analysis_df)
            
            # Store the results
            analysis_results[instrument] = {
                'df': analysis_df,
                'current': {
                    'close': analysis_df['close'].iloc[-1],
                    'sma_short': analysis_df['sma_short'].iloc[-1],
                    'sma_long': analysis_df['sma_long'].iloc[-1],
                    'ma_crossover': analysis_df['ma_crossover'].iloc[-1],
                    'ma_crossover_signal': analysis_df['ma_crossover_signal'].iloc[-1],
                    'rsi': analysis_df['rsi'].iloc[-1],
                    'rsi_overbought': analysis_df['rsi_overbought'].iloc[-1],
                    'rsi_oversold': analysis_df['rsi_oversold'].iloc[-1],
                    'atr': analysis_df['atr'].iloc[-1],
                    'adx': analysis_df['adx'].iloc[-1],
                    'volume_ratio': analysis_df['volume_ratio'].iloc[-1],
                    'trend_strength': analysis_df['trend_strength'].iloc[-1],
                    'volatility': analysis_df['volatility'].iloc[-1],
                    'market_regime': analysis_df['market_regime'].iloc[-1],
                    'trend_signal': analysis_df['trend_signal'].iloc[-1]
                },
                'timestamp': pd.Timestamp.now()
            }
            
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
            - 'strong_uptrend'
            - 'uptrend'
            - 'weak_uptrend'
            - 'ranging'
            - 'weak_downtrend'
            - 'downtrend'
            - 'strong_downtrend'
            - 'volatile'
        """
        regimes = pd.Series(index=df.index, dtype='object')
        
        # Strong uptrend: ADX > 25, positive directional movement, low volatility
        strong_uptrend = (df['adx'] > 25) & (df['ma_crossover'] > 0) & (df['volatility'] < 0.02)
        
        # Uptrend: Positive MA crossover, ADX > 20
        uptrend = (df['ma_crossover'] > 0) & (df['adx'] > 20)
        
        # Weak uptrend: Positive MA crossover but ADX < 20
        weak_uptrend = (df['ma_crossover'] > 0) & (df['adx'] <= 20)
        
        # Strong downtrend: ADX > 25, negative directional movement, low volatility
        strong_downtrend = (df['adx'] > 25) & (df['ma_crossover'] < 0) & (df['volatility'] < 0.02)
        
        # Downtrend: Negative MA crossover, ADX > 20
        downtrend = (df['ma_crossover'] < 0) & (df['adx'] > 20)
        
        # Weak downtrend: Negative MA crossover but ADX < 20
        weak_downtrend = (df['ma_crossover'] < 0) & (df['adx'] <= 20)
        
        # Ranging: ADX < 20, low volatility
        ranging = (df['adx'] < 20) & (df['volatility'] < 0.015)
        
        # Volatile: High volatility regardless of trend
        volatile = df['volatility'] >= 0.025
        
        # Assign regimes
        regimes[strong_uptrend] = 'strong_uptrend'
        regimes[uptrend & ~strong_uptrend] = 'uptrend'
        regimes[weak_uptrend & ~uptrend] = 'weak_uptrend'
        regimes[strong_downtrend] = 'strong_downtrend'
        regimes[downtrend & ~strong_downtrend] = 'downtrend'
        regimes[weak_downtrend & ~downtrend] = 'weak_downtrend'
        regimes[ranging & ~(weak_uptrend | weak_downtrend)] = 'ranging'
        regimes[volatile] = 'volatile'
        
        # Fill any remaining NaN values with 'unknown'
        regimes = regimes.fillna('unknown')
        
        return regimes
    
    def _calculate_trend_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trend signal based on multiple indicators.
        
        Returns:
            Series with trend signals: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        signals = pd.Series(0, index=df.index)
        
        # MA Crossover signal
        ma_signal = df['ma_crossover_signal']
        
        # RSI signal
        rsi_signal = pd.Series(0, index=df.index)
        rsi_signal[df['rsi_oversold']] = 1  # Bullish when oversold
        rsi_signal[df['rsi_overbought']] = -1  # Bearish when overbought
        
        # ADX signal (trend strength)
        adx_signal = pd.Series(0, index=df.index)
        adx_signal[(df['adx'] > 25) & (df['ma_crossover'] > 0)] = 1  # Strong uptrend
        adx_signal[(df['adx'] > 25) & (df['ma_crossover'] < 0)] = -1  # Strong downtrend
        
        # Volume signal
        volume_signal = pd.Series(0, index=df.index)
        volume_signal[df['volume_ratio'] > self.params["volume_threshold"]] = 1  # High volume is bullish for trend continuation
        
        # Combine signals with weights
        signals = (
            ma_signal * 0.4 +  # 40% weight to MA crossover
            rsi_signal * 0.2 +  # 20% weight to RSI
            adx_signal * 0.3 +  # 30% weight to ADX
            volume_signal * 0.1  # 10% weight to volume
        )
        
        # Discretize to -1, 0, 1
        final_signals = pd.Series(0, index=df.index)
        final_signals[signals > self.params["trend_confirmation_threshold"]] = 1
        final_signals[signals < -self.params["trend_confirmation_threshold"]] = -1
        
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
            df = analysis['df']
            
            # Skip if we don't have enough data
            if df.shape[0] < max(self.params["short_window"], self.params["long_window"]):
                continue
            
            # Get current position for this instrument
            current_position = self.positions.get(instrument, 0)
            
            # Check if we should apply regime filter
            apply_signal = True
            if self.params["use_regime_filter"]:
                regime = current['market_regime']
                # Only trade in trending regimes
                if regime not in ['strong_uptrend', 'uptrend', 'strong_downtrend', 'downtrend']:
                    apply_signal = False
            
            # Check if we should apply volume filter
            if apply_signal and self.params["use_volume_filter"]:
                if current['volume_ratio'] < self.params["volume_threshold"]:
                    apply_signal = False
            
            # Check if we should apply volatility filter
            if apply_signal and self.params["use_volatility_filter"]:
                if current['volatility'] > self.params["volatility_threshold"]:
                    apply_signal = False
            
            # Generate signals based on trend signal
            trend_signal = current['trend_signal']
            
            if apply_signal:
                # Long entry signal
                if trend_signal == 1 and current_position <= 0:
                    # Check if we can add a new position (max positions constraint)
                    if len(self.positions) < self.params["max_positions"] or current_position < 0:
                        # Calculate position size
                        position_size = self._calculate_position_size(instrument, 'long', current['close'], current['atr'])
                        
                        # Generate long entry signal
                        signals.append({
                            'instrument': instrument,
                            'timestamp': pd.Timestamp.now(),
                            'action': 'BUY',
                            'type': 'ENTRY',
                            'price': current['close'],
                            'quantity': position_size,
                            'order_type': 'MARKET',
                            'reason': f"Trend following long entry: MA crossover={current['ma_crossover']:.4f}, RSI={current['rsi']:.2f}, ADX={current['adx']:.2f}",
                            'stop_loss': current['close'] - (current['atr'] * self.params["trailing_stop_atr_multiple"]),
                            'take_profit': current['close'] + (current['atr'] * self.params["take_profit_atr_multiple"]),
                            'strategy': self.name
                        })
                
                # Short entry signal
                elif trend_signal == -1 and current_position >= 0:
                    # Check if we can add a new position (max positions constraint)
                    if len(self.positions) < self.params["max_positions"] or current_position > 0:
                        # Calculate position size
                        position_size = self._calculate_position_size(instrument, 'short', current['close'], current['atr'])
                        
                        # Generate short entry signal
                        signals.append({
                            'instrument': instrument,
                            'timestamp': pd.Timestamp.now(),
                            'action': 'SELL',
                            'type': 'ENTRY',
                            'price': current['close'],
                            'quantity': position_size,
                            'order_type': 'MARKET',
                            'reason': f"Trend following short entry: MA crossover={current['ma_crossover']:.4f}, RSI={current['rsi']:.2f}, ADX={current['adx']:.2f}",
                            'stop_loss': current['close'] + (current['atr'] * self.params["trailing_stop_atr_multiple"]),
                            'take_profit': current['close'] - (current['atr'] * self.params["take_profit_atr_multiple"]),
                            'strategy': self.name
                        })
            
            # Exit signals (these apply regardless of regime filters)
            
            # Exit long position
            if current_position > 0:
                # Exit on trend reversal
                if trend_signal == -1:
                    signals.append({
                        'instrument': instrument,
                        'timestamp': pd.Timestamp.now(),
                        'action': 'SELL',
                        'type': 'EXIT',
                        'price': current['close'],
                        'quantity': abs(current_position),
                        'order_type': 'MARKET',
                        'reason': f"Trend following long exit: Trend reversal signal",
                        'strategy': self.name
                    })
                # Exit on stop loss or take profit is handled by the execution engine
            
            # Exit short position
            elif current_position < 0:
                # Exit on trend reversal
                if trend_signal == 1:
                    signals.append({
                        'instrument': instrument,
                        'timestamp': pd.Timestamp.now(),
                        'action': 'BUY',
                        'type': 'EXIT',
                        'price': current['close'],
                        'quantity': abs(current_position),
                        'order_type': 'MARKET',
                        'reason': f"Trend following short exit: Trend reversal signal",
                        'strategy': self.name
                    })
                # Exit on stop loss or take profit is handled by the execution engine
        
        return signals
    
    def _calculate_position_size(self, instrument: str, direction: str, price: float, atr: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            instrument: Instrument to trade
            direction: 'long' or 'short'
            price: Current price
            atr: Average True Range
            
        Returns:
            Position size in units of the instrument
        """
        method = self.params["position_sizing_method"]
        
        if method == "risk_based":
            # Risk-based position sizing
            account_balance = 10000.0  # This should come from the portfolio manager
            risk_amount = account_balance * self.params["risk_per_trade"]
            
            # Calculate stop loss distance
            stop_distance = atr * self.params["trailing_stop_atr_multiple"]
            
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
        # Keep trade history and performance metrics