from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class SlippageModel(ABC):
    """
    Abstract base class for slippage models.
    """
    
    @abstractmethod
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        **kwargs
    ) -> float:
        """
        Calculate the slippage for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            is_buy: Whether the trade is a buy (True) or sell (False)
            **kwargs: Additional parameters
            
        Returns:
            Slippage amount (positive for buys, negative for sells)
        """
        pass


class NoSlippage(SlippageModel):
    """
    Slippage model with no slippage.
    """
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        **kwargs
    ) -> float:
        """
        Calculate the slippage for a trade (always zero).
        
        Args:
            price: Trade price
            quantity: Trade quantity
            is_buy: Whether the trade is a buy (True) or sell (False)
            **kwargs: Additional parameters
            
        Returns:
            Zero slippage
        """
        return 0.0


class FixedSlippage(SlippageModel):
    """
    Slippage model with fixed percentage slippage.
    """
    
    def __init__(self, slippage_pct: float = 0.0):
        """
        Initialize the fixed slippage model.
        
        Args:
            slippage_pct: Slippage percentage (e.g., 0.1 for 0.1%)
        """
        self.slippage_pct = slippage_pct / 100.0  # Convert to decimal
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        **kwargs
    ) -> float:
        """
        Calculate the slippage for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            is_buy: Whether the trade is a buy (True) or sell (False)
            **kwargs: Additional parameters
            
        Returns:
            Slippage amount (positive for buys, negative for sells)
        """
        slippage_amount = price * self.slippage_pct
        
        # Slippage increases price for buys, decreases for sells
        return slippage_amount if is_buy else -slippage_amount


class VolumeBasedSlippage(SlippageModel):
    """
    Slippage model with volume-based slippage.
    """
    
    def __init__(
        self,
        base_slippage_pct: float = 0.0,
        volume_impact_factor: float = 0.1
    ):
        """
        Initialize the volume-based slippage model.
        
        Args:
            base_slippage_pct: Base slippage percentage (e.g., 0.1 for 0.1%)
            volume_impact_factor: Factor for volume impact on slippage
        """
        self.base_slippage_pct = base_slippage_pct / 100.0  # Convert to decimal
        self.volume_impact_factor = volume_impact_factor
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        volume: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Calculate the slippage for a trade based on volume.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            is_buy: Whether the trade is a buy (True) or sell (False)
            volume: Trading volume (if available)
            **kwargs: Additional parameters
            
        Returns:
            Slippage amount (positive for buys, negative for sells)
        """
        # Base slippage
        slippage_pct = self.base_slippage_pct
        
        # Adjust for volume if available
        if volume is not None and volume > 0:
            # Calculate trade's percentage of volume
            volume_pct = (quantity * price) / volume
            
            # Increase slippage based on volume impact
            slippage_pct += volume_pct * self.volume_impact_factor
        
        slippage_amount = price * slippage_pct
        
        # Slippage increases price for buys, decreases for sells
        return slippage_amount if is_buy else -slippage_amount


class VolatilityBasedSlippage(SlippageModel):
    """
    Slippage model with volatility-based slippage.
    """
    
    def __init__(
        self,
        base_slippage_pct: float = 0.0,
        volatility_factor: float = 1.0
    ):
        """
        Initialize the volatility-based slippage model.
        
        Args:
            base_slippage_pct: Base slippage percentage (e.g., 0.1 for 0.1%)
            volatility_factor: Factor for volatility impact on slippage
        """
        self.base_slippage_pct = base_slippage_pct / 100.0  # Convert to decimal
        self.volatility_factor = volatility_factor
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        volatility: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Calculate the slippage for a trade based on volatility.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            is_buy: Whether the trade is a buy (True) or sell (False)
            volatility: Price volatility (if available)
            **kwargs: Additional parameters
            
        Returns:
            Slippage amount (positive for buys, negative for sells)
        """
        # Base slippage
        slippage_pct = self.base_slippage_pct
        
        # Adjust for volatility if available
        if volatility is not None and volatility > 0:
            # Increase slippage based on volatility
            slippage_pct += volatility * self.volatility_factor
        
        slippage_amount = price * slippage_pct
        
        # Slippage increases price for buys, decreases for sells
        return slippage_amount if is_buy else -slippage_amount


class RandomSlippage(SlippageModel):
    """
    Slippage model with random slippage within a range.
    """
    
    def __init__(
        self,
        min_slippage_pct: float = 0.0,
        max_slippage_pct: float = 0.1
    ):
        """
        Initialize the random slippage model.
        
        Args:
            min_slippage_pct: Minimum slippage percentage (e.g., 0.0 for 0.0%)
            max_slippage_pct: Maximum slippage percentage (e.g., 0.1 for 0.1%)
        """
        self.min_slippage_pct = min_slippage_pct / 100.0  # Convert to decimal
        self.max_slippage_pct = max_slippage_pct / 100.0  # Convert to decimal
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        **kwargs
    ) -> float:
        """
        Calculate random slippage for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            is_buy: Whether the trade is a buy (True) or sell (False)
            **kwargs: Additional parameters
            
        Returns:
            Slippage amount (positive for buys, negative for sells)
        """
        # Generate random slippage percentage within range
        slippage_pct = np.random.uniform(self.min_slippage_pct, self.max_slippage_pct)
        
        slippage_amount = price * slippage_pct
        
        # Slippage increases price for buys, decreases for sells
        return slippage_amount if is_buy else -slippage_amount


class BidAskSpreadSlippage(SlippageModel):
    """
    Slippage model based on bid-ask spread.
    """
    
    def __init__(
        self,
        default_spread_pct: float = 0.1,
        additional_slippage_pct: float = 0.0
    ):
        """
        Initialize the bid-ask spread slippage model.
        
        Args:
            default_spread_pct: Default bid-ask spread percentage (e.g., 0.1 for 0.1%)
            additional_slippage_pct: Additional slippage percentage beyond spread
        """
        self.default_spread_pct = default_spread_pct / 100.0  # Convert to decimal
        self.additional_slippage_pct = additional_slippage_pct / 100.0  # Convert to decimal
    
    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        **kwargs
    ) -> float:
        """
        Calculate slippage based on bid-ask spread.
        
        Args:
            price: Trade price (mid price)
            quantity: Trade quantity
            is_buy: Whether the trade is a buy (True) or sell (False)
            bid_price: Current bid price (if available)
            ask_price: Current ask price (if available)
            **kwargs: Additional parameters
            
        Returns:
            Slippage amount (positive for buys, negative for sells)
        """
        # If bid and ask prices are provided, use them to calculate spread
        if bid_price is not None and ask_price is not None and bid_price < ask_price:
            spread_pct = (ask_price - bid_price) / price
        else:
            # Otherwise use default spread
            spread_pct = self.default_spread_pct
        
        # Calculate slippage based on half the spread plus additional slippage
        slippage_pct = (spread_pct / 2) + self.additional_slippage_pct
        
        slippage_amount = price * slippage_pct
        
        # Slippage increases price for buys, decreases for sells
        return slippage_amount if is_buy else -slippage_amount