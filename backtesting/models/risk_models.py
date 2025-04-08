from typing import Dict, List, Union, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from abc import ABC, abstractmethod

class RiskModel(ABC):
    """
    Abstract base class for risk models.
    """
    
    def __init__(self):
        """
        Initialize the risk model.
        """
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def calculate_position_size(self, **kwargs) -> float:
        """
        Calculate position size based on risk parameters.
        
        Returns:
            Position size
        """
        pass
    
    @abstractmethod
    def calculate_portfolio_risk(self, **kwargs) -> float:
        """
        Calculate overall portfolio risk.
        
        Returns:
            Portfolio risk measure
        """
        pass


class FixedFractionalRiskModel(RiskModel):
    """
    Fixed fractional risk model that risks a fixed percentage of portfolio on each trade.
    """
    
    def __init__(self, risk_pct: float = 0.02, max_position_pct: float = 0.1):
        """
        Initialize the fixed fractional risk model.
        
        Args:
            risk_pct: Percentage of portfolio to risk per trade (default: 2%)
            max_position_pct: Maximum position size as percentage of portfolio (default: 10%)
        """
        super().__init__()
        self.risk_pct = risk_pct
        self.max_position_pct = max_position_pct
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        risk_pct: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Calculate position size based on fixed fractional risk.
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price
            stop_loss_price: Stop loss price (optional)
            risk_pct: Risk percentage override (optional)
            
        Returns:
            Position size in units
        """
        # Use provided risk percentage or default
        risk_percentage = risk_pct if risk_pct is not None else self.risk_pct
        
        # Calculate risk amount
        risk_amount = portfolio_value * risk_percentage
        
        # Calculate position size
        if stop_loss_price and stop_loss_price > 0 and entry_price > 0:
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss_price)
            
            if risk_per_unit > 0:
                # Calculate position size based on risk per unit
                position_size = risk_amount / risk_per_unit
            else:
                # Fallback if stop loss is at entry price
                position_size = 0
        else:
            # If no stop loss, use a default risk per unit (e.g., 1% of entry price)
            risk_per_unit = entry_price * 0.01
            position_size = risk_amount / risk_per_unit
        
        # Apply maximum position size constraint
        max_position_size = portfolio_value * self.max_position_pct / entry_price
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        stop_losses: Dict[str, float],
        portfolio_value: float,
        **kwargs
    ) -> float:
        """
        Calculate overall portfolio risk.
        
        Args:
            positions: Dictionary of positions (instrument -> quantity)
            prices: Dictionary of current prices (instrument -> price)
            stop_losses: Dictionary of stop loss prices (instrument -> stop price)
            portfolio_value: Current portfolio value
            
        Returns:
            Portfolio risk as percentage of portfolio
        """
        total_risk_amount = 0.0
        
        for instrument, position in positions.items():
            if position == 0 or instrument not in prices:
                continue
            
            price = prices[instrument]
            stop_loss = stop_losses.get(instrument)
            
            if stop_loss and price > 0:
                # Calculate risk for this position
                risk_per_unit = abs(price - stop_loss)
                position_risk = abs(position) * risk_per_unit
                total_risk_amount += position_risk
        
        # Calculate risk as percentage of portfolio
        if portfolio_value > 0:
            portfolio_risk = total_risk_amount / portfolio_value
        else:
            portfolio_risk = 0.0
        
        return portfolio_risk


class KellyRiskModel(RiskModel):
    """
    Kelly criterion risk model for position sizing.
    """
    
    def __init__(
        self,
        win_rate: float = 0.5,
        win_loss_ratio: float = 2.0,
        fraction: float = 0.5,  # Half-Kelly for more conservative sizing
        max_position_pct: float = 0.2
    ):
        """
        Initialize the Kelly risk model.
        
        Args:
            win_rate: Historical win rate (default: 0.5)
            win_loss_ratio: Ratio of average win to average loss (default: 2.0)
            fraction: Fraction of full Kelly to use (default: 0.5)
            max_position_pct: Maximum position size as percentage of portfolio (default: 20%)
        """
        super().__init__()
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self.fraction = fraction
        self.max_position_pct = max_position_pct
        
        # Trade history for adaptive Kelly
        self.trade_history = []
    
    def update_parameters(self, trades: List[Dict[str, Any]]) -> None:
        """
        Update model parameters based on trade history.
        
        Args:
            trades: List of trade dictionaries with 'pnl' field
        """
        if not trades:
            return
        
        # Add trades to history
        self.trade_history.extend(trades)
        
        # Calculate win rate
        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        if self.trade_history:
            self.win_rate = len(winning_trades) / len(self.trade_history)
        
        # Calculate win/loss ratio
        if winning_trades and len(self.trade_history) > len(winning_trades):
            avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades)
            losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
            if losing_trades:
                avg_loss = abs(sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades))
                if avg_loss > 0:
                    self.win_loss_ratio = avg_win / avg_loss
        
        self.logger.info(f"Updated Kelly parameters: win_rate={self.win_rate:.2f}, win_loss_ratio={self.win_loss_ratio:.2f}")
    
    def calculate_kelly_fraction(self) -> float:
        """
        Calculate the Kelly fraction.
        
        Returns:
            Kelly fraction
        """
        # Kelly formula: f* = (p * b - q) / b
        # where p = win rate, q = 1 - p, b = win/loss ratio
        p = self.win_rate
        q = 1 - p
        b = self.win_loss_ratio
        
        if b <= 0:
            return 0
        
        kelly = (p * b - q) / b
        
        # Apply fraction of Kelly and ensure it's not negative
        kelly = max(0, kelly * self.fraction)
        
        return kelly
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        target_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Calculate position size based on Kelly criterion.
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price
            target_price: Target price (optional)
            stop_loss_price: Stop loss price (optional)
            
        Returns:
            Position size in units
        """
        # Calculate Kelly fraction
        kelly = self.calculate_kelly_fraction()
        
        # If target and stop loss are provided, recalculate win/loss ratio for this trade
        if target_price and stop_loss_price and entry_price > 0:
            potential_gain = abs(target_price - entry_price)
            potential_loss = abs(entry_price - stop_loss_price)
            
            if potential_loss > 0:
                trade_win_loss_ratio = potential_gain / potential_loss
                
                # Recalculate Kelly for this specific trade
                p = self.win_rate
                q = 1 - p
                b = trade_win_loss_ratio
                
                if b > 0:
                    kelly = (p * b - q) / b
                    kelly = max(0, kelly * self.fraction)
        
        # Calculate position size
        position_value = portfolio_value * kelly
        
        if entry_price > 0:
            position_size = position_value / entry_price
        else:
            position_size = 0
        
        # Apply maximum position size constraint
        max_position_size = portfolio_value * self.max_position_pct / entry_price
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        stop_losses: Dict[str, float],
        portfolio_value: float,
        **kwargs
    ) -> float:
        """
        Calculate overall portfolio risk.
        
        Args:
            positions: Dictionary of positions (instrument -> quantity)
            prices: Dictionary of current prices (instrument -> price)
            stop_losses: Dictionary of stop loss prices (instrument -> stop price)
            portfolio_value: Current portfolio value
            
        Returns:
            Portfolio risk as percentage of portfolio
        """
        total_risk_amount = 0.0
        
        for instrument, position in positions.items():
            if position == 0 or instrument not in prices:
                continue
            
            price = prices[instrument]
            stop_loss = stop_losses.get(instrument)
            
            if stop_loss and price > 0:
                # Calculate risk for this position
                risk_per_unit = abs(price - stop_loss)
                position_risk = abs(position) * risk_per_unit
                total_risk_amount += position_risk
        
        # Calculate risk as percentage of portfolio
        if portfolio_value > 0:
            portfolio_risk = total_risk_amount / portfolio_value
        else:
            portfolio_risk = 0.0
        
        return portfolio_risk


class VolatilityRiskModel(RiskModel):
    """
    Volatility-based risk model that adjusts position size based on market volatility.
    """
    
    def __init__(
        self,
        risk_pct: float = 0.02,
        max_position_pct: float = 0.1,
        volatility_lookback: int = 20,
        volatility_scaling: bool = True,
        target_volatility: float = 0.01  # 1% daily volatility target
    ):
        """
        Initialize the volatility risk model.
        
        Args:
            risk_pct: Percentage of portfolio to risk per trade (default: 2%)
            max_position_pct: Maximum position size as percentage of portfolio (default: 10%)
            volatility_lookback: Number of periods to calculate volatility (default: 20)
            volatility_scaling: Whether to scale position size by volatility (default: True)
            target_volatility: Target daily volatility (default: 1%)
        """
        super().__init__()
        self.risk_pct = risk_pct
        self.max_position_pct = max_position_pct
        self.volatility_lookback = volatility_lookback
        self.volatility_scaling = volatility_scaling
        self.target_volatility = target_volatility
        
        # Store historical volatility
        self.volatility_history = {}
    
    def update_volatility(self, instrument: str, prices: pd.Series) -> float:
        """
        Update volatility for an instrument.
        
        Args:
            instrument: Instrument identifier
            prices: Price series
            
        Returns:
            Current volatility
        """
        if len(prices) < self.volatility_lookback:
            # Not enough data, use a default volatility
            volatility = 0.01  # 1% daily volatility as default
        else:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.tail(self.volatility_lookback).std()
        
        # Store volatility
        self.volatility_history[instrument] = volatility
        
        return volatility
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        instrument: str,
        prices: Optional[pd.Series] = None,
        stop_loss_price: Optional[float] = None,
        volatility: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Calculate position size based on volatility.
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price
            instrument: Instrument identifier
            prices: Historical price series (optional)
            stop_loss_price: Stop loss price (optional)
            volatility: Volatility override (optional)
            
        Returns:
            Position size in units
        """
        # Get or calculate volatility
        if volatility is not None:
            current_volatility = volatility
        elif instrument in self.volatility_history:
            current_volatility = self.volatility_history[instrument]
        elif prices is not None:
            current_volatility = self.update_volatility(instrument, prices)
        else:
            current_volatility = 0.01  # Default volatility
        
        # Calculate risk amount
        risk_amount = portfolio_value * self.risk_pct
        
        # Calculate position size
        if self.volatility_scaling and current_volatility > 0:
            # Scale position size inversely with volatility
            volatility_scalar = self.target_volatility / current_volatility
            risk_amount *= volatility_scalar
        
        if stop_loss_price and stop_loss_price > 0 and entry_price > 0:
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss_price)
            
            if risk_per_unit > 0:
                # Calculate position size based on risk per unit
                position_size = risk_amount / risk_per_unit
            else:
                # Fallback if stop loss is at entry price
                position_size = 0
        else:
            # If no stop loss, use volatility to determine risk per unit
            risk_per_unit = entry_price * current_volatility * 2  # 2 standard deviations
            position_size = risk_amount / risk_per_unit
        
        # Apply maximum position size constraint
        max_position_size = portfolio_value * self.max_position_pct / entry_price
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        historical_prices: Dict[str, pd.Series],
        portfolio_value: float,
        correlation_matrix: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> float:
        """
        Calculate overall portfolio risk considering volatility and correlations.
        
        Args:
            positions: Dictionary of positions (instrument -> quantity)
            prices: Dictionary of current prices (instrument -> price)
            historical_prices: Dictionary of historical price series
            portfolio_value: Current portfolio value
            correlation_matrix: Correlation matrix of instruments (optional)
            
        Returns:
            Portfolio risk as percentage of portfolio
        """
        # If no positions, return zero risk
        if not positions or not prices or portfolio_value <= 0:
            return 0.0
        
        # Calculate position values and weights
        position_values = {}
        position_weights = {}
        
        for instrument, position in positions.items():
            if position == 0 or instrument not in prices:
                continue
            
            price = prices[instrument]
            position_value = abs(position * price)
            position_values[instrument] = position_value
            position_weights[instrument] = position_value / portfolio_value
        
        # If only one position or no correlation matrix, use simple sum of individual risks
        if len(position_values) <= 1 or correlation_matrix is None:
            total_risk = 0.0
            
            for instrument, weight in position_weights.items():
                # Get or calculate volatility
                if instrument in self.volatility_history:
                    volatility = self.volatility_history[instrument]
                elif instrument in historical_prices:
                    volatility = self.update_volatility(instrument, historical_prices[instrument])
                else:
                    volatility = 0.01  # Default volatility
                
                # Add weighted volatility to total risk
                total_risk += weight * volatility
            
            return total_risk
        
        # Calculate portfolio risk using correlation matrix
        instruments = list(position_weights.keys())
        weights = np.array([position_weights[inst] for inst in instruments])
        
        # Create volatility vector
        volatilities = np.array([
            self.volatility_history.get(inst, 0.01) for inst in instruments
        ])
        
        # Extract correlation submatrix for current positions
        corr_submatrix = correlation_matrix.loc[instruments, instruments].values
        
        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * corr_submatrix
        
        # Calculate portfolio variance
        portfolio_variance = weights.T @ cov_matrix @ weights
        
        # Calculate portfolio volatility (risk)
        portfolio_risk = np.sqrt(portfolio_variance)
        
        return portfolio_risk


class RiskParityModel(RiskModel):
    """
    Risk parity model that allocates risk equally across assets.
    """
    
    def __init__(
        self,
        target_portfolio_risk: float = 0.1,  # 10% annualized volatility target
        max_position_pct: float = 0.25,
        volatility_lookback: int = 60,
        rebalance_threshold: float = 0.1  # Rebalance when risk contribution deviates by 10%
    ):
        """
        Initialize the risk parity model.
        
        Args:
            target_portfolio_risk: Target annualized portfolio risk (default: 10%)
            max_position_pct: Maximum position size as percentage of portfolio (default: 25%)
            volatility_lookback: Number of periods to calculate volatility (default: 60)
            rebalance_threshold: Threshold for rebalancing (default: 10%)
        """
        super().__init__()
        self.target_portfolio_risk = target_portfolio_risk
        self.max_position_pct = max_position_pct
        self.volatility_lookback = volatility_lookback
        self.rebalance_threshold = rebalance_threshold
        
        # Store volatilities and correlations
        self.volatilities = {}
        self.correlation_matrix = None
        self.target_weights = {}
    
    def update_volatilities(self, historical_prices: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Update volatilities for all instruments.
        
        Args:
            historical_prices: Dictionary of historical price series
            
        Returns:
            Dictionary of volatilities
        """
        for instrument, prices in historical_prices.items():
            if len(prices) >= self.volatility_lookback:
                # Calculate returns
                returns = prices.pct_change().dropna()
                
                # Calculate volatility (annualized)
                daily_volatility = returns.tail(self.volatility_lookback).std()
                annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days
                
                self.volatilities[instrument] = annualized_volatility
        
        return self.volatilities
    
    def update_correlation_matrix(self, historical_prices: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Update correlation matrix for all instruments.
        
        Args:
            historical_prices: Dictionary of historical price series
            
        Returns:
            Correlation matrix
        """
        # Create returns DataFrame
        returns_data = {}
        
        for instrument, prices in historical_prices.items():
            if len(prices) >= self.volatility_lookback:
                returns = prices.pct_change().dropna()
                returns_data[instrument] = returns.tail(self.volatility_lookback)
        
        if returns_data:
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate correlation matrix
            self.correlation_matrix = returns_df.corr()
        
        return self.correlation_matrix
    
    def calculate_risk_parity_weights(
        self,
        instruments: List[str],
        correlation_matrix: Optional[pd.DataFrame] = None,
        volatilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights.
        
        Args:
            instruments: List of instruments
            correlation_matrix: Correlation matrix (optional)
            volatilities: Dictionary of volatilities (optional)
            
        Returns:
            Dictionary of target weights
        """
        if not instruments:
            return {}
        
        # Use provided values or stored values
        corr_matrix = correlation_matrix if correlation_matrix is not None else self.correlation_matrix
        vols = volatilities if volatilities is not None else self.volatilities
        
        # Filter for instruments with volatility data
        valid_instruments = [inst for inst in instruments if inst in vols]
        
        if not valid_instruments:
            # Equal weights if no volatility data
            equal_weight = 1.0 / len(instruments)
            return {inst: equal_weight for inst in instruments}
        
        if len(valid_instruments) == 1:
            # Single instrument case
            return {valid_instruments[0]: 1.0}
        
        # Extract correlation submatrix
        if corr_matrix is not None and all(inst in corr_matrix.index for inst in valid_instruments):
            sub_corr = corr_matrix.loc[valid_instruments, valid_instruments].values
        else:
            # Use identity matrix if correlation data is missing
            sub_corr = np.eye(len(valid_instruments))
        
        # Create volatility vector
        vol_vector = np.array([vols[inst] for inst in valid_instruments])
        
        # Create covariance matrix
        cov_matrix = np.outer(vol_vector, vol_vector) * sub_corr
        
        # Simple risk parity solution using inverse volatility
        inv_vol = 1.0 / vol_vector
        weights = inv_vol / np.sum(inv_vol)
        
        # Create weights dictionary
        target_weights = {inst: weight for inst, weight in zip(valid_instruments, weights)}
        
        # Normalize weights to sum to 1
        total_weight = sum(target_weights.values())
        if total_weight > 0:
            target_weights = {inst: w / total_weight for inst, w in target_weights.items()}
        
        # Store target weights
        self.target_weights = target_weights
        
        return target_weights
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        instrument: str,
        instruments: List[str],
        historical_prices: Optional[Dict[str, pd.Series]] = None,
        **kwargs
    ) -> float:
        """
        Calculate position size based on risk parity.
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price
            instrument: Instrument identifier
            instruments: List of all instruments in portfolio
            historical_prices: Dictionary of historical price series (optional)
            
        Returns:
            Position size in units
        """
        # Update volatilities and correlation matrix if historical prices provided
        if historical_prices:
            self.update_volatilities(historical_prices)
            self.update_correlation_matrix(historical_prices)
        
        # Calculate target weights
        if not self.target_weights or set(instruments) != set(self.target_weights.keys()):
            self.calculate_risk_parity_weights(instruments)
        
        # Get target weight for instrument
        target_weight = self.target_weights.get(instrument, 0.0)
        
        # Calculate position value
        position_value = portfolio_value * target_weight
        
        # Calculate position size
        if entry_price > 0:
            position_size = position_value / entry_price
        else:
            position_size = 0
        
        # Apply maximum position size constraint
        max_position_size = portfolio_value * self.max_position_pct / entry_price
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def calculate_portfolio_risk(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        historical_prices: Dict[str, pd.Series],
        portfolio_value: float,
        **kwargs
    ) -> float:
        """
        Calculate overall portfolio risk.
        
        Args:
            positions: Dictionary of positions (instrument -> quantity)
            prices: Dictionary of current prices (instrument -> price)
            historical_prices: Dictionary of historical price series
            portfolio_value: Current portfolio value
            
        Returns:
            Portfolio risk as percentage of portfolio
        """
        # If no positions, return zero risk
        if not positions or not prices or portfolio_value <= 0:
            return 0.0
        
        # Update volatilities and correlation matrix
        if historical_prices:
            self.update_volatilities(historical_prices)
            self.update_correlation_matrix(historical_prices)
        
        # Calculate position values and weights
        position_values = {}
        position_weights = {}
        
        for instrument, position in positions.items():
            if position == 0 or instrument not in prices:
                continue
            
            price = prices[instrument]
            position_value = abs(position * price)
            position_values[instrument] = position_value
            position_weights[instrument] = position_value / portfolio_value
        
        # If no valid positions, return zero risk
        if not position_weights:
            return 0.0
        
        # If only one position, return its volatility
        if len(position_weights) == 1:
            instrument = list(position_weights.keys())[0]
            return self.volatilities.get(instrument, 0.1)  # Default to 10% if no volatility data
        
        # Calculate portfolio risk using weights and covariance matrix
        instruments = list(position_weights.keys())
        weights = np.array([position_weights[inst] for inst in instruments])
        
        # Create volatility vector
        volatilities = np.array([
            self.volatilities.get(inst, 0.1) for inst in instruments
        ])
        
        # Extract correlation submatrix
        if self.correlation_matrix is not None and all(inst in self.correlation_matrix.index for inst in instruments):
            corr_submatrix = self.correlation_matrix.loc[instruments, instruments].values
        else:
            # Use identity matrix if correlation data is missing
            corr_submatrix = np.eye(len(instruments))
        
        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * corr_submatrix
        
        # Calculate portfolio variance
        portfolio_variance = weights.T @ cov_matrix @ weights
        
        # Calculate portfolio volatility (risk)
        portfolio_risk = np.sqrt(portfolio_variance)
        
        return portfolio_risk
    
    def check_rebalance_needed(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ) -> bool:
        """
        Check if portfolio rebalancing is needed.
        
        Args:
            positions: Dictionary of positions (instrument -> quantity)
            prices: Dictionary of current prices (instrument -> price)
            portfolio_value: Current portfolio value
            
        Returns:
            True if rebalancing is needed, False otherwise
        """
        if not self.target_weights or not positions or not prices or portfolio_value <= 0:
            return False
        
        # Calculate current weights
        current_weights = {}
        
        for instrument, position in positions.items():
            if position == 0 or instrument not in prices:
                continue
            
            price = prices[instrument]
            position_value = position * price
            current_weights[instrument] = position_value / portfolio_value
        
        # Check deviation from target weights
        for instrument, target_weight in self.target_weights.items():
            current_weight = current_weights.get(instrument, 0.0)
            
            # Calculate relative deviation
            if target_weight > 0:
                deviation = abs(current_weight - target_weight) / target_weight
                
                if deviation > self.rebalance_threshold:
                    return True
        
        return False