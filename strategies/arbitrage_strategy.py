import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import asyncio
from decimal import Decimal

from .base_strategy import BaseStrategy

class ArbitrageStrategy(BaseStrategy):
    """
    Strategy for identifying and executing arbitrage opportunities across exchanges or markets.
    """
    
    def __init__(self, name: str, timeframes: List[str], instruments: List[str], params: Dict = None):
        """
        Initialize the arbitrage strategy.
        
        Args:
            name: Strategy name
            timeframes: List of timeframes to analyze
            instruments: List of instruments to trade
            params: Strategy-specific parameters including:
                - min_profit_threshold: Minimum profit percentage to execute arbitrage
                - max_execution_time: Maximum time (ms) allowed for execution
                - exchanges: List of exchanges to monitor
                - use_triangular: Whether to use triangular arbitrage
                - use_cross_exchange: Whether to use cross-exchange arbitrage
        """
        default_params = {
            "min_profit_threshold": 0.5,  # Minimum profit percentage
            "max_execution_time": 2000,   # Maximum execution time in milliseconds
            "exchanges": ["binance", "coinbase", "kraken"],
            "use_triangular": True,
            "use_cross_exchange": True,
            "max_slippage": 0.2,          # Maximum allowed slippage percentage
            "min_volume": 1000,           # Minimum volume in USD
            "gas_price_threshold": 50,    # Maximum gas price in Gwei for DeFi arbitrage
            "check_interval": 1.0         # Check interval in seconds
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(name, timeframes, instruments, default_params)
        self.opportunities = []
        self.last_prices = {}
        self.execution_stats = {
            "total_opportunities": 0,
            "executed_opportunities": 0,
            "failed_executions": 0,
            "total_profit": 0.0,
            "average_execution_time": 0.0
        }
    
    def analyze(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """
        Analyze market data across exchanges to identify arbitrage opportunities.
        
        Args:
            data: Dictionary of dictionaries with market data for each instrument on each exchange
            
        Returns:
            Dictionary with analysis results and potential arbitrage opportunities
        """
        opportunities = []
        
        # Update last known prices
        for exchange, exchange_data in data.items():
            for instrument, df in exchange_data.items():
                if df is not None and not df.empty:
                    self.last_prices[(exchange, instrument)] = df['close'].iloc[-1]
        
        # Find cross-exchange arbitrage opportunities
        if self.params["use_cross_exchange"]:
            cross_exchange_opps = self._find_cross_exchange_opportunities()
            opportunities.extend(cross_exchange_opps)
        
        # Find triangular arbitrage opportunities
        if self.params["use_triangular"]:
            triangular_opps = self._find_triangular_opportunities(data)
            opportunities.extend(triangular_opps)
        
        # Filter opportunities by profit threshold
        filtered_opps = [
            opp for opp in opportunities 
            if opp["expected_profit_pct"] >= self.params["min_profit_threshold"]
        ]
        
        self.opportunities = filtered_opps
        self.execution_stats["total_opportunities"] += len(filtered_opps)
        
        return {
            "opportunities": filtered_opps,
            "timestamp": pd.Timestamp.now(),
            "market_state": self._get_market_state(data)
        }
    
    def _find_cross_exchange_opportunities(self) -> List[Dict]:
        """Find arbitrage opportunities across different exchanges."""
        opportunities = []
        
        # Get all instruments that are available on multiple exchanges
        instruments = {}
        for (exchange, instrument), price in self.last_prices.items():
            if instrument not in instruments:
                instruments[instrument] = []
            instruments[instrument].append((exchange, price))
        
        # Look for price differences
        for instrument, exchange_prices in instruments.items():
            if len(exchange_prices) < 2:
                continue
                
            # Find lowest ask and highest bid
            lowest_ask = min(exchange_prices, key=lambda x: x[1])
            highest_bid = max(exchange_prices, key=lambda x: x[1])
            
            # Calculate potential profit
            price_diff = highest_bid[1] - lowest_ask[1]
            profit_pct = (price_diff / lowest_ask[1]) * 100
            
            # Account for fees
            estimated_fees_pct = 0.2  # Estimated fees percentage (0.2%)
            net_profit_pct = profit_pct - estimated_fees_pct
            
            if net_profit_pct > 0:
                opportunities.append({
                    "type": "cross_exchange",
                    "instrument": instrument,
                    "buy_exchange": lowest_ask[0],
                    "buy_price": lowest_ask[1],
                    "sell_exchange": highest_bid[0],
                    "sell_price": highest_bid[1],
                    "price_difference": price_diff,
                    "expected_profit_pct": net_profit_pct,
                    "timestamp": pd.Timestamp.now()
                })
        
        return opportunities
    
    def _find_triangular_opportunities(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> List[Dict]:
        """Find triangular arbitrage opportunities within a single exchange."""
        opportunities = []
        
        # Define common triangular paths to check
        # For example: BTC -> ETH -> USDT -> BTC
        triangular_paths = [
            ("BTC", "ETH", "USDT"),
            ("ETH", "BTC", "USDT"),
            ("BTC", "XRP", "USDT"),
            ("ETH", "XRP", "USDT")
        ]
        
        for exchange, exchange_data in data.items():
            for path in triangular_paths:
                # Check if we have all required pairs
                pairs = [
                    f"{path[0]}{path[1]}",
                    f"{path[1]}{path[2]}",
                    f"{path[2]}{path[0]}"
                ]
                
                if not all(pair in exchange_data for pair in pairs):
                    continue
                
                # Get latest prices
                price_1 = exchange_data[pairs[0]]['close'].iloc[-1]
                price_2 = exchange_data[pairs[1]]['close'].iloc[-1]
                price_3 = exchange_data[pairs[2]]['close'].iloc[-1]
                
                # Calculate potential profit
                # Start with 1 unit of first currency
                step1 = 1 / price_1  # Convert to second currency
                step2 = step1 / price_2  # Convert to third currency
                step3 = step2 / price_3  # Convert back to first currency
                
                profit_pct = (step3 - 1) * 100
                
                # Account for fees (0.1% per trade)
                net_profit_pct = profit_pct - (0.1 * 3)
                
                if net_profit_pct > 0:
                    opportunities.append({
                        "type": "triangular",
                        "exchange": exchange,
                        "path": path,
                        "pairs": pairs,
                        "prices": [price_1, price_2, price_3],
                        "expected_profit_pct": net_profit_pct,
                        "timestamp": pd.Timestamp.now()
                    })
        
        return opportunities
    
    def _get_market_state(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
        """Get current market state summary."""
        state = {
            "exchange_count": len(data),
            "instrument_count": sum(len(exchange_data) for exchange_data in data.values()),
            "price_volatility": {},
            "spread_metrics": {}
        }
        
        # Calculate volatility for each instrument
        for exchange, exchange_data in data.items():
            for instrument, df in exchange_data.items():
                if df is not None and len(df) > 1:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std() * 100  # Convert to percentage
                        state["price_volatility"][(exchange, instrument)] = volatility
        
        return state
    
    def generate_signals(self, analysis: Dict) -> List[Dict]:
        """
        Generate trading signals based on arbitrage opportunities.
        
        Args:
            analysis: Analysis results from the analyze method
            
        Returns:
            List of signal dictionaries with arbitrage execution details
        """
        signals = []
        
        for opportunity in analysis.get("opportunities", []):
            if opportunity["type"] == "cross_exchange":
                signals.append({
                    "strategy": self.name,
                    "type": "arbitrage",
                    "subtype": "cross_exchange",
                    "action": "BUY",
                    "instrument": opportunity["instrument"],
                    "exchange": opportunity["buy_exchange"],
                    "price": opportunity["buy_price"],
                    "timestamp": pd.Timestamp.now(),
                    "metadata": {
                        "sell_exchange": opportunity["sell_exchange"],
                        "sell_price": opportunity["sell_price"],
                        "expected_profit_pct": opportunity["expected_profit_pct"]
                    }
                })
                
                signals.append({
                    "strategy": self.name,
                    "type": "arbitrage",
                    "subtype": "cross_exchange",
                    "action": "SELL",
                    "instrument": opportunity["instrument"],
                    "exchange": opportunity["sell_exchange"],
                    "price": opportunity["sell_price"],
                    "timestamp": pd.Timestamp.now(),
                    "metadata": {
                        "buy_exchange": opportunity["buy_exchange"],
                        "buy_price": opportunity["buy_price"],
                        "expected_profit_pct": opportunity["expected_profit_pct"]
                    }
                })
            
            elif opportunity["type"] == "triangular":
                # For triangular arbitrage, we need three signals
                path = opportunity["path"]
                pairs = opportunity["pairs"]
                prices = opportunity["prices"]
                
                # First trade
                signals.append({
                    "strategy": self.name,
                    "type": "arbitrage",
                    "subtype": "triangular",
                    "action": "BUY",
                    "instrument": pairs[0],
                    "exchange": opportunity["exchange"],
                    "price": prices[0],
                    "timestamp": pd.Timestamp.now(),
                    "metadata": {
                        "step": 1,
                        "path": path,
                        "expected_profit_pct": opportunity["expected_profit_pct"]
                    }
                })
                
                # Second trade
                signals.append({
                    "strategy": self.name,
                    "type": "arbitrage",
                    "subtype": "triangular",
                    "action": "BUY",
                    "instrument": pairs[1],
                    "exchange": opportunity["exchange"],
                    "price": prices[1],
                    "timestamp": pd.Timestamp.now(),
                    "metadata": {
                        "step": 2,
                        "path": path,
                        "expected_profit_pct": opportunity["expected_profit_pct"]
                    }
                })
                
                # Third trade
                signals.append({
                    "strategy": self.name,
                    "type": "arbitrage",
                    "subtype": "triangular",
                    "action": "BUY",
                    "instrument": pairs[2],
                    "exchange": opportunity["exchange"],
                    "price": prices[2],
                    "timestamp": pd.Timestamp.now(),
                    "metadata": {
                        "step": 3,
                        "path": path,
                        "expected_profit_pct": opportunity["expected_profit_pct"]
                    }
                })
        
        return signals
    
    async def execute_arbitrage(self, opportunity: Dict, wallet, exchange_clients: Dict) -> Dict:
        """
        Execute an arbitrage opportunity.
        
        Args:
            opportunity: Arbitrage opportunity details
            wallet: Wallet instance for crypto transactions
            exchange_clients: Dictionary of exchange API clients
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        result = {
            "success": False,
            "profit": 0.0,
            "execution_time": 0.0,
            "errors": []
        }
        
        try:
            if opportunity["type"] == "cross_exchange":
                # Get exchange clients
                buy_exchange = exchange_clients.get(opportunity["buy_exchange"])
                sell_exchange = exchange_clients.get(opportunity["sell_exchange"])
                
                if not buy_exchange or not sell_exchange:
                    result["errors"].append("Exchange client not available")
                    return result
                
                # Execute buy order
                buy_order = await buy_exchange.create_market_buy_order(
                    opportunity["instrument"],
                    self.params.get("order_size", 1.0)
                )
                
                if not buy_order or buy_order.get("status") != "filled":
                    result["errors"].append("Buy order failed")
                    return result
                
                # Execute sell order
                sell_order = await sell_exchange.create_market_sell_order(
                    opportunity["instrument"],
                    buy_order["filled"]
                )
                
                if not sell_order or sell_order.get("status") != "filled":
                    result["errors"].append("Sell order failed")
                    # Attempt to sell back on original exchange to minimize loss
                    await buy_exchange.create_market_sell_order(
                        opportunity["instrument"],
                        buy_order["filled"]
                    )
                    return result
                
                # Calculate actual profit
                buy_cost = buy_order["cost"]
                sell_proceeds = sell_order["cost"]
                actual_profit = sell_proceeds - buy_cost
                actual_profit_pct = (actual_profit / buy_cost) * 100
                
                result["success"] = True
                result["profit"] = actual_profit
                result["profit_pct"] = actual_profit_pct
                
            elif opportunity["type"] == "triangular":
                exchange = exchange_clients.get(opportunity["exchange"])
                
                if not exchange:
                    result["errors"].append("Exchange client not available")
                    return result
                
                # Execute the three trades in sequence
                pairs = opportunity["pairs"]
                
                # First trade
                order1 = await exchange.create_market_buy_order(
                    pairs[0],
                    self.params.get("order_size", 1.0)
                )
                
                if not order1 or order1.get("status") != "filled":
                    result["errors"].append("First trade failed")
                    return result
                
                # Second trade
                order2 = await exchange.create_market_buy_order(
                    pairs[1],
                    order1["filled"]
                )
                
                if not order2 or order2.get("status") != "filled":
                    result["errors"].append("Second trade failed")
                    # Try to reverse the first trade to minimize loss
                    await exchange.create_market_sell_order(
                        pairs[0],
                        order1["filled"]
                    )
                    return result
                
                # Third trade
                order3 = await exchange.create_market_buy_order(
                    pairs[2],
                    order2["filled"]
                )
                
                if not order3 or order3.get("status") != "filled":
                    result["errors"].append("Third trade failed")
                    # Try to reverse the second trade to minimize loss
                    await exchange.create_market_sell_order(
                        pairs[1],
                        order2["filled"]
                    )
                    return result
                
                # Calculate actual profit
                initial_amount = order1["cost"]
                final_amount = order3["amount"]
                actual_profit = final_amount - initial_amount
                actual_profit_pct = (actual_profit / initial_amount) * 100
                
                result["success"] = True
                result["profit"] = actual_profit
                result["profit_pct"] = actual_profit_pct
                
        except Exception as e:
            result["errors"].append(f"Execution error: {str(e)}")
            self.logger.error(f"Arbitrage execution error: {str(e)}", exc_info=True)
        
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        result["execution_time"] = execution_time
        
        # Update execution stats
        self.execution_stats["executed_opportunities"] += 1
        if result["success"]:
            self.execution_stats["total_profit"] += result.get("profit", 0)
        else:
            self.execution_stats["failed_executions"] += 1
        
        # Update average execution time
        total_executions = self.execution_stats["executed_opportunities"]
        current_avg = self.execution_stats["average_execution_time"]
        self.execution_stats["average_execution_time"] = (
            (current_avg * (total_executions - 1) + execution_time) / total_executions
        )
        
        return result
    
    async def execute_flashbots_arbitrage(self, opportunity: Dict, wallet, provider) -> Dict:
        """
        Execute an arbitrage opportunity using Flashbots to prevent front-running.
        
        Args:
            opportunity: Arbitrage opportunity details
            wallet: Wallet instance for crypto transactions
            provider: Web3 provider with Flashbots support
            
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        result = {
            "success": False,
            "profit": 0.0,
            "execution_time": 0.0,
            "errors": []
        }
        
        try:
            # Check current gas price
            gas_price = await provider.eth.gas_price
            gas_price_gwei = gas_price / 1e9
            
            if gas_price_gwei > self.params["gas_price_threshold"]:
                result["errors"].append(f"Gas price too high: {gas_price_gwei} Gwei")
                return result
            
            # Prepare the bundle of transactions for atomic execution
            if opportunity["type"] == "dex_arbitrage":
                # Create a flashbots bundle for DEX arbitrage
                bundle = self._create_dex_arbitrage_bundle(opportunity, wallet)
                
                # Simulate the bundle to check profitability
                simulation = await provider.flashbots.simulate(bundle)
                
                if not simulation["success"]:
                    result["errors"].append(f"Bundle simulation failed: {simulation['error']}")
                    return result
                
                # Check if the simulation shows profit
                profit_wei = simulation["profit"]
                if profit_wei <= 0:
                    result["errors"].append("No profit in simulation")
                    return result
                
                # Send the bundle
                bundle_submission = await provider.flashbots.send_bundle(bundle)
                
                # Wait for bundle to be included
                inclusion_result = await bundle_submission.wait_for_inclusion(2)  # Wait for 2 blocks max
                
                if not inclusion_result["success"]:
                    result["errors"].append("Bundle not included in blocks")
                    return result
                
                # Calculate actual profit
                profit_eth = profit_wei / 1e18
                result["success"] = True
                result["profit"] = profit_eth
                result["tx_hash"] = inclusion_result["tx_hash"]
                
        except Exception as e:
            result["errors"].append(f"Flashbots execution error: {str(e)}")
            self.logger.error(f"Flashbots arbitrage execution error: {str(e)}", exc_info=True)
        
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        result["execution_time"] = execution_time
        
        # Update execution stats
        self.execution_stats["executed_opportunities"] += 1
        if result["success"]:
            self.execution_stats["total_profit"] += result.get("profit", 0)
        else:
            self.execution_stats["failed_executions"] += 1
        
        return result
    
    def _create_dex_arbitrage_bundle(self, opportunity: Dict, wallet) -> List:
        """
        Create a bundle of transactions for DEX arbitrage.
        
        Args:
            opportunity: Arbitrage opportunity details
            wallet: Wallet instance for signing transactions
            
        Returns:
            List of signed transactions for the bundle
        """
        # This is a simplified implementation
        # In a real system, you would create actual transactions based on the opportunity
        
        # Example: Create a flash loan transaction followed by DEX swaps
        transactions = []
        
        # Add flash loan transaction
        # transactions.append(...)
        
        # Add DEX swap transactions
        # transactions.append(...)
        
        # Add repayment transaction
        # transactions.append(...)
        
        return transactions
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        valid = True
        
        # Check min profit threshold
        if self.params["min_profit_threshold"] <= 0:
            self.logger.warning("min_profit_threshold must be greater than 0")
            valid = False
        
        # Check max execution time
        if self.params["max_execution_time"] <= 0:
            self.logger.warning("max_execution_time must be greater than 0")
            valid = False
        
        # Check exchanges
        if not self.params["exchanges"] or len(self.params["exchanges"]) < 2:
            self.logger.warning("At least two exchanges are required for cross-exchange arbitrage")
            if self.params["use_cross_exchange"]:
                valid = False
        
        return valid
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics."""
        return self.execution_stats
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.execution_stats = {
            "total_opportunities": 0,
            "executed_opportunities": 0,
            "failed_executions": 0,
            "total_profit": 0.0,
            "average_execution_time": 0.0
        }