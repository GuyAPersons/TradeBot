import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy

class MarketMakingStrategy(BaseStrategy):
    """
    Strategy for market making across various exchanges and instruments.
    """
    
    def __init__(self, name: str, timeframes: List[str], instruments: List[str], params: Dict = None):
        """
        Initialize the market making strategy.
        
        Args:
            name: Strategy name
            timeframes: List of timeframes to analyze
            instruments: List of instruments to trade
            params: Strategy-specific parameters including:
                - spread_target: Target spread percentage
                - order_size: Size of orders to place
                - max_position: Maximum position size
                - rebalance_frequency: How often to rebalance inventory
                - risk_factor: Risk factor for position sizing
        """
        default_params = {
            "spread_target": 0.2,  # Target spread percentage
            "order_size": 0.1,  # Base order size
            "max_position": 1.0,  # Maximum position size
            "rebalance_frequency": "hourly",  # Options: hourly, daily
            "risk_factor": 0.5,  # Risk factor for position sizing
            "min_spread": 0.05,  # Minimum spread percentage
            "max_spread": 2.0,  # Maximum spread percentage
            "inventory_target": 0.5,  # Target inventory level (0.5 = 50%)
            "use_dynamic_spreads": True,  # Adjust spreads based on volatility
            "use_inventory_skewing": True,  # Skew quotes based on inventory
            "volatility_lookback": 24,  # Hours to look back for volatility calculation
            "max_orders_per_side": 3,  # Maximum number of orders per side
            "order_spacing": 0.1,  # Spacing between orders as percentage
            "cancel_threshold": 0.5,  # Cancel orders when price moves by this percentage
            "min_order_age": 30,  # Minimum age of orders before cancellation (seconds)
            "use_iceberg_orders": False,  # Use iceberg orders to hide true size
            "iceberg_display_size": 0.2,  # Display size for iceberg orders
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(name, timeframes, instruments, default_params)
        self.active_orders = {}
        self.positions = {}
        self.last_rebalance = datetime.now()
        self.market_stats = {}
        self.performance_metrics = {
            "total_trades": 0,
            "profitable_trades": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
            "total_pnl": 0.0,
            "inventory_cost": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0
        }
        
    def analyze(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze market data to determine optimal market making parameters.
        
        Args:
            data: Dictionary of DataFrames with market data for each instrument
            
        Returns:
            Dictionary with analysis results and market making parameters
        """
        analysis_results = {}
        
        for instrument, df in data.items():
            if df is None or df.empty:
                continue
                
            # Calculate market statistics
            stats = self._calculate_market_stats(instrument, df)
            self.market_stats[instrument] = stats
            
            # Determine optimal spread based on volatility
            optimal_spread = self._calculate_optimal_spread(stats)
            
            # Adjust for inventory if enabled
            if self.params["use_inventory_skewing"]:
                inventory_level = self._get_inventory_level(instrument)
                spread_adjustment = self._calculate_inventory_adjustment(inventory_level)
                bid_adjustment = spread_adjustment
                ask_adjustment = -spread_adjustment
            else:
                bid_adjustment = 0
                ask_adjustment = 0
            
            # Calculate order sizes based on risk
            base_order_size = self.params["order_size"]
            risk_adjusted_size = self._calculate_risk_adjusted_size(instrument, stats)
            
            # Determine order placement levels
            bid_levels = self._calculate_order_levels("bid", stats["mid_price"], optimal_spread, bid_adjustment)
            ask_levels = self._calculate_order_levels("ask", stats["mid_price"], optimal_spread, ask_adjustment)
            
            # Check if rebalancing is needed
            needs_rebalance = self._check_rebalance_needed(instrument)
            
            analysis_results[instrument] = {
                "timestamp": pd.Timestamp.now(),
                "market_stats": stats,
                "optimal_spread": optimal_spread,
                "bid_levels": bid_levels,
                "ask_levels": ask_levels,
                "base_order_size": base_order_size,
                "risk_adjusted_size": risk_adjusted_size,
                "inventory_level": self._get_inventory_level(instrument),
                "needs_rebalance": needs_rebalance
            }
        
        return {
            "market_making_params": analysis_results,
            "global_stats": self._calculate_global_stats(data)
        }
    
    def _calculate_market_stats(self, instrument: str, df: pd.DataFrame) -> Dict:
        """Calculate market statistics for an instrument."""
        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(f"Missing required columns for {instrument}")
            return {}
        
        # Get the most recent data
        recent_data = df.iloc[-self.params["volatility_lookback"]:]
        
        # Calculate basic statistics
        last_price = df["close"].iloc[-1]
        mid_price = (df["high"].iloc[-1] + df["low"].iloc[-1]) / 2
        
        # Calculate volatility (annualized)
        returns = df["close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized percentage
        
        # Calculate recent volume statistics
        avg_volume = recent_data["volume"].mean()
        volume_std = recent_data["volume"].std()
        
        # Calculate bid-ask spread if available
        if "ask" in df.columns and "bid" in df.columns:
            last_spread = (df["ask"].iloc[-1] - df["bid"].iloc[-1]) / mid_price * 100
        else:
            # Estimate spread from high-low
            last_spread = (df["high"].iloc[-1] - df["low"].iloc[-1]) / mid_price * 100 * 0.1  # Rough estimate
        
        # Calculate price momentum
        momentum = self._calculate_momentum(df)
        
        return {
            "last_price": last_price,
            "mid_price": mid_price,
            "volatility": volatility,
            "avg_volume": avg_volume,
            "volume_std": volume_std,
            "last_spread": last_spread,
            "momentum": momentum,
            "timestamp": pd.Timestamp.now()
        }
    
    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate price momentum indicator."""
        # Simple momentum calculation using recent price changes
        short_ma = df["close"].rolling(window=5).mean().iloc[-1]
        long_ma = df["close"].rolling(window=20).mean().iloc[-1]
        
        # Normalize momentum between -1 and 1
        if long_ma == 0:
            return 0
            
        momentum = (short_ma / long_ma - 1)
        return max(min(momentum, 1), -1)
    
    def _calculate_optimal_spread(self, stats: Dict) -> float:
        """Calculate optimal spread based on market statistics."""
        if not stats:
            return self.params["spread_target"]
        
        base_spread = self.params["spread_target"]
        
        # Adjust spread based on volatility if enabled
        if self.params["use_dynamic_spreads"]:
            # Higher volatility = wider spread
            volatility_factor = stats["volatility"] / 20  # Normalize volatility
            volatility_adjustment = base_spread * volatility_factor
            
            # Adjust for momentum - widen spreads when momentum is strong in either direction
            momentum_factor = abs(stats["momentum"])
            momentum_adjustment = base_spread * momentum_factor * 0.5
            
            # Combine adjustments
            optimal_spread = base_spread + volatility_adjustment + momentum_adjustment
        else:
            optimal_spread = base_spread
        
        # Ensure spread is within allowed range
        optimal_spread = max(self.params["min_spread"], min(self.params["max_spread"], optimal_spread))
        
        return optimal_spread
    
    def _get_inventory_level(self, instrument: str) -> float:
        """
        Get current inventory level as a percentage of max position.
        
        Returns value between 0 and 1, where 0.5 is neutral (50% of max).
        """
        current_position = self.positions.get(instrument, 0)
        max_position = self.params["max_position"]
        
        if max_position == 0:
            return 0.5  # Neutral if max position is 0
        
        # Normalize to 0-1 range where 0.5 is neutral
        normalized = (current_position / max_position + 1) / 2
        return max(0, min(1, normalized))
    
    def _calculate_inventory_adjustment(self, inventory_level: float) -> float:
        """
        Calculate spread adjustment based on inventory level.
        
        Returns a positive value to widen spread on one side and narrow on the other.
        """
        # Convert inventory level to a -1 to 1 scale where 0 is neutral
        inventory_skew = (inventory_level - 0.5) * 2
        
        # Calculate adjustment - more aggressive as we move away from neutral
        adjustment = inventory_skew * self.params["spread_target"] * 0.5
        
        return adjustment
    
    def _calculate_risk_adjusted_size(self, instrument: str, stats: Dict) -> float:
        """Calculate risk-adjusted order size based on market conditions."""
        base_size = self.params["order_size"]
        risk_factor = self.params["risk_factor"]
        
        # Adjust size based on volatility - lower size for higher volatility
        if stats and "volatility" in stats:
            volatility_adjustment = 1 - (stats["volatility"] / 100) * risk_factor
            volatility_adjustment = max(0.2, min(1.5, volatility_adjustment))
        else:
            volatility_adjustment = 1
        
        # Adjust size based on available balance (would be implemented in a real system)
        balance_adjustment = 1.0
        
        # Adjust size based on current inventory
        inventory_level = self._get_inventory_level(instrument)
        inventory_distance = abs(inventory_level - 0.5) * 2  # 0 at neutral, 1 at extremes
        inventory_adjustment = 1 - inventory_distance * 0.5
        
        # Combine adjustments
        adjusted_size = base_size * volatility_adjustment * balance_adjustment * inventory_adjustment
        
        return adjusted_size
    
    def _calculate_order_levels(self, side: str, mid_price: float, spread: float, adjustment: float) -> List[Dict]:
        """
        Calculate price levels for order placement.
        
        Args:
            side: 'bid' or 'ask'
            mid_price: Current mid price
            spread: Target spread percentage
            adjustment: Adjustment to spread based on inventory
            
        Returns:
            List of dictionaries with price and size for each order level
        """
        levels = []
        max_orders = self.params["max_orders_per_side"]
        spacing = self.params["order_spacing"]
        
        # Calculate half spread (adjusted)
        half_spread_pct = (spread / 2 + adjustment) / 100
        
        for i in range(max_orders):
            # Calculate price level with increasing distance from mid
            level_spread = half_spread_pct * (1 + i * spacing)
            
            if side == "bid":
                price = mid_price * (1 - level_spread)
                # Size increases as we move away from mid (more aggressive at better prices)
                size_multiplier = 1 - (i * 0.2)
            else:  # ask
                price = mid_price * (1 + level_spread)
                # Size increases as we move away from mid (more aggressive at better prices)
                size_multiplier = 1 - (i * 0.2)
            
            # Ensure size multiplier is positive
            size_multiplier = max(0.2, size_multiplier)
            
            levels.append({
                "price": price,
                "size_multiplier": size_multiplier,
                "level": i + 1
            })
        
        return levels
    
    def _check_rebalance_needed(self, instrument: str) -> bool:
        """Check if inventory rebalancing is needed."""
        # Get rebalance frequency in hours
        if self.params["rebalance_frequency"] == "hourly":
            frequency_hours = 1
        elif self.params["rebalance_frequency"] == "daily":
            frequency_hours = 24
        else:
            frequency_hours = 1  # Default to hourly
        
        # Check if enough time has passed since last rebalance
        time_since_rebalance = datetime.now() - self.last_rebalance
        if time_since_rebalance > timedelta(hours=frequency_hours):
            return True
        
        # Check if inventory is significantly away from target
        inventory_level = self._get_inventory_level(instrument)
        target = self.params["inventory_target"]
        
        # If inventory is more than 20% away from target, rebalance
        if abs(inventory_level - target) > 0.2:
            return True
        
        return False
    
    def _calculate_global_stats(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate global market statistics across all instruments."""
        # Average volatility across instruments
        volatilities = []
        for instrument, stats in self.market_stats.items():
            if stats and "volatility" in stats:
                volatilities.append(stats["volatility"])
        
        avg_volatility = np.mean(volatilities) if volatilities else 0
        
        # Market correlation matrix
        returns_dict = {}
        for instrument, df in data.items():
            if df is not None and not df.empty and len(df) > 1:
                returns_dict[instrument] = df["close"].pct_change().dropna()
        
        correlation_matrix = pd.DataFrame(returns_dict).corr() if returns_dict else pd.DataFrame()
        
        # Overall market direction
        market_momentum = np.mean([stats.get("momentum", 0) for stats in self.market_stats.values()])
        
        return {
            "avg_volatility": avg_volatility,
            "correlation_matrix": correlation_matrix,
            "market_momentum": market_momentum,
            "timestamp": pd.Timestamp.now()
        }
    
    def generate_signals(self, analysis: Dict) -> List[Dict]:
        """
        Generate trading signals for market making.
        
        Args:
            analysis: Analysis results from the analyze method
            
        Returns:
            List of signal dictionaries with order placement/cancellation details
        """
        signals = []
        market_making_params = analysis.get("market_making_params", {})
        
        for instrument, params in market_making_params.items():
            # Check if we need to cancel existing orders
            cancel_signals = self._generate_cancel_signals(instrument, params)
            signals.extend(cancel_signals)
            
            # Generate new order signals
            order_signals = self._generate_order_signals(instrument, params)
            signals.extend(order_signals)
            
            # Check if rebalancing is needed
            if params.get("needs_rebalance", False):
                rebalance_signals = self._generate_rebalance_signals(instrument, params)
                signals.extend(rebalance_signals)
                self.last_rebalance = datetime.now()
        
        return signals
    
    def _generate_cancel_signals(self, instrument: str, params: Dict) -> List[Dict]:
        """Generate signals to cancel existing orders."""
        signals = []
        
        # Get current active orders for this instrument
        active_orders = self.active_orders.get(instrument, [])
        if not active_orders:
            return signals
        
        current_mid = params["market_stats"]["mid_price"]
        cancel_threshold = self.params["cancel_threshold"] / 100  # Convert to decimal
        min_order_age = self.params["min_order_age"]  # In seconds
        
        for order in active_orders[:]:  # Use a copy to avoid modification during iteration
            # Check if price has moved significantly
            price_change_pct = abs(order["price"] - current_mid) / current_mid
            
            # Check order age
            order_age = (datetime.now() - order["timestamp"]).total_seconds()
            
            # Cancel if price moved significantly or order is old
            if price_change_pct > cancel_threshold or order_age > min_order_age:
                signals.append({
                    "strategy": self.name,
                    "type": "market_making",
                    "subtype": "cancel",
                    "action": "CANCEL",
                    "instrument": instrument,
                    "order_id": order["order_id"],
                    "timestamp": pd.Timestamp.now(),
                    "metadata": {
                        "reason": "price_change" if price_change_pct > cancel_threshold else "age",
                        "price_change_pct": price_change_pct * 100,
                        "order_age": order_age
                    }
                })
                
                # Remove from active orders
                self.active_orders[instrument].remove(order)
        
        return signals
    
    def _generate_order_signals(self, instrument: str, params: Dict) -> List[Dict]:
        """Generate signals for new market making orders."""
        signals = []
        
        # Get bid and ask levels
        bid_levels = params.get("bid_levels", [])
        ask_levels = params.get("ask_levels", [])
        
        # Get base order size
        base_size = params.get("risk_adjusted_size", self.params["order_size"])
        
        # Generate bid orders
        for level in bid_levels:
            price = level["price"]
            size = base_size * level["size_multiplier"]
            
            # Check if we already have an active order near this price
            existing_order = self._find_existing_order(instrument, "bid", price)
            
            if not existing_order:
                # Create new bid order
                order_id = f"mm_bid_{instrument}_{int(time.time())}_{level['level']}"
                
                signals.append({
                    "strategy": self.name,
                    "type": "market_making",
                    "subtype": "bid",
                    "action": "BUY",
                    "instrument": instrument,
                    "price": price,
                    "quantity": size,
                    "order_type": "LIMIT",
                    "timestamp": pd.Timestamp.now(),
                    "order_id": order_id,
                    "metadata": {
                        "level": level["level"],
                        "is_iceberg": self.params["use_iceberg_orders"],
                        "display_size": size * self.params["iceberg_display_size"] if self.params["use_iceberg_orders"] else size
                    }
                })
                
                # Add to active orders
                if instrument not in self.active_orders:
                    self.active_orders[instrument] = []
                
                self.active_orders[instrument].append({
                    "order_id": order_id,
                    "side": "bid",
                    "price": price,
                    "size": size,
                    "level": level["level"],
                    "timestamp": datetime.now()
                })
        
        # Generate ask orders
        for level in ask_levels:
            price = level["price"]
            size = base_size * level["size_multiplier"]
            
            # Check if we already have an active order near this price
            existing_order = self._find_existing_order(instrument, "ask", price)
            
            if not existing_order:
                # Create new ask order
                order_id = f"mm_ask_{instrument}_{int(time.time())}_{level['level']}"
                
                signals.append({
                    "strategy": self.name,
                    "type": "market_making",
                    "subtype": "ask",
                    "action": "SELL",
                    "instrument": instrument,
                    "price": price,
                    "quantity": size,
                    "order_type": "LIMIT",
                    "timestamp": pd.Timestamp.now(),
                    "order_id": order_id,
                    "metadata": {
                        "level": level["level"],
                        "is_iceberg": self.params["use_iceberg_orders"],
                        "display_size": size * self.params["iceberg_display_size"] if self.params["use_iceberg_orders"] else size
                    }
                })
                
                # Add to active orders
                if instrument not in self.active_orders:
                    self.active_orders[instrument] = []
                
                self.active_orders[instrument].append({
                    "order_id": order_id,
                    "side": "ask",
                    "price": price,
                    "size": size,
                    "level": level["level"],
                    "timestamp": datetime.now()
                })
        
        return signals
    
    def _find_existing_order(self, instrument: str, side: str, price: float) -> Optional[Dict]:
        """Find if there's an existing order near the given price."""
        if instrument not in self.active_orders:
            return None
        
        # Define price tolerance (0.1% of price)
        tolerance = price * 0.001
        
        for order in self.active_orders[instrument]:
            if order["side"] == side and abs(order["price"] - price) < tolerance:
                return order
        
        return None
    
    def _generate_rebalance_signals(self, instrument: str, params: Dict) -> List[Dict]:
        """Generate signals to rebalance inventory."""
        signals = []
        
        # Get current inventory level and target
        current_level = self._get_inventory_level(instrument)
        target_level = self.params["inventory_target"]
        
        # Calculate how far we are from target (as a percentage of max position)
        deviation = (current_level - target_level) * self.params["max_position"]
        
        # If deviation is significant, create a rebalancing order
        if abs(deviation) > 0.1 * self.params["max_position"]:
            # Determine direction and size
            if deviation > 0:  # We have too much inventory
                action = "SELL"
                size = min(abs(deviation), self.positions.get(instrument, 0))
            else:  # We have too little inventory
                action = "BUY"
                size = abs(deviation)
            
            # Only proceed if size is meaningful
            if size > 0.01:
                # Get current mid price
                mid_price = params["market_stats"]["mid_price"]
                
                # Create rebalance order (market order to ensure execution)
                order_id = f"mm_rebalance_{instrument}_{int(time.time())}"
                
                signals.append({
                    "strategy": self.name,
                    "type": "market_making",
                    "subtype": "rebalance",
                    "action": action,
                    "instrument": instrument,
                    "price": mid_price,  # Reference price, will be executed as market
                    "quantity": size,
                    "order_type": "MARKET",
                    "timestamp": pd.Timestamp.now(),
                    "order_id": order_id,
                    "metadata": {
                        "current_level": current_level,
                        "target_level": target_level,
                        "deviation": deviation,
                        "reason": "inventory_rebalance"
                    }
                })
        
        return signals
    
    def update_positions(self, fill_event: Dict) -> None:
        """
        Update positions based on fill events.
        
        Args:
            fill_event: Dictionary with fill details
        """
        instrument = fill_event.get("instrument")
        if not instrument:
            return
        
        action = fill_event.get("action")
        quantity = fill_event.get("quantity", 0)
        price = fill_event.get("price", 0)
        
        # Update position
        if instrument not in self.positions:
            self.positions[instrument] = 0
        
        if action == "BUY":
            self.positions[instrument] += quantity
        elif action == "SELL":
            self.positions[instrument] -= quantity
        
        # Update performance metrics
        self.performance_metrics["total_trades"] += 1
        self.performance_metrics["total_volume"] += quantity * price
        
        # Calculate fees (simplified)
        fee = quantity * price * 0.001  # Assume 0.1% fee
        self.performance_metrics["total_fees"] += fee
        
        # Update PnL (simplified)
        if fill_event.get("realized_pnl") is not None:
            realized_pnl = fill_event["realized_pnl"]
            self.performance_metrics["realized_pnl"] += realized_pnl
            
            if realized_pnl > 0:
                self.performance_metrics["profitable_trades"] += 1
        
        # Log the position update
        self.logger.info(f"Updated position for {instrument}: {self.positions[instrument]}")
        self.logger.info(f"Performance metrics: {self.performance_metrics}")
    
    def update_market_data(self, market_update: Dict) -> None:
        """
        Update internal market data based on market updates.
        
        Args:
            market_update: Dictionary with market update details
        """
        instrument = market_update.get("instrument")
        if not instrument:
            return
        
        # Update market stats if available
        if "mid_price" in market_update:
            if instrument not in self.market_stats:
                self.market_stats[instrument] = {}
            
            self.market_stats[instrument]["mid_price"] = market_update["mid_price"]
        
        # Update unrealized PnL
        if instrument in self.positions and self.positions[instrument] != 0:
            position = self.positions[instrument]
            mid_price = self.market_stats.get(instrument, {}).get("mid_price")
            
            if mid_price and "avg_entry_price" in self.market_stats.get(instrument, {}):
                avg_entry = self.market_stats[instrument]["avg_entry_price"]
                unrealized_pnl = position * (mid_price - avg_entry)
                self.performance_metrics["unrealized_pnl"] = unrealized_pnl
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        # Calculate additional metrics
        metrics = self.performance_metrics.copy()
        
        # Calculate win rate
        if metrics["total_trades"] > 0:
            metrics["win_rate"] = metrics["profitable_trades"] / metrics["total_trades"]
        else:
            metrics["win_rate"] = 0
        
        # Calculate total PnL
        metrics["total_pnl"] = metrics["realized_pnl"] + metrics["unrealized_pnl"]
        
        # Calculate PnL net of fees
        metrics["net_pnl"] = metrics["total_pnl"] - metrics["total_fees"]
        
        # Calculate inventory value
        inventory_value = 0
        for instrument, position in self.positions.items():
            mid_price = self.market_stats.get(instrument, {}).get("mid_price", 0)
            inventory_value += position * mid_price
        
        metrics["inventory_value"] = inventory_value
        
        return metrics
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_trades": 0,
            "profitable_trades": 0,
            "total_volume": 0.0,
            "total_fees": 0.0,
            "total_pnl": 0.0,
            "inventory_cost": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0
        }
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        valid = True
        
        # Check spread parameters
        if self.params["min_spread"] >= self.params["max_spread"]:
            self.logger.warning("min_spread must be less than max_spread")
            valid = False
        
        if self.params["spread_target"] < self.params["min_spread"] or self.params["spread_target"] > self.params["max_spread"]:
            self.logger.warning("spread_target must be between min_spread and max_spread")
            valid = False
        
        # Check order parameters
        if self.params["order_size"] <= 0:
            self.logger.warning("order_size must be greater than 0")
            valid = False
        
        if self.params["max_position"] <= 0:
            self.logger.warning("max_position must be greater than 0")
            valid = False
        
        # Check inventory target
        if self.params["inventory_target"] < 0 or self.params["inventory_target"] > 1:
            self.logger.warning("inventory_target must be between 0 and 1")
            valid = False
        
        # Check order spacing
        if self.params["order_spacing"] <= 0:
            self.logger.warning("order_spacing must be greater than 0")
            valid = False
        
        return valid