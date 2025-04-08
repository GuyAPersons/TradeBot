from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod

class TransactionCostModel(ABC):
    """
    Abstract base class for transaction cost models.
    """
    
    @abstractmethod
    def calculate_commission(
        self,
        price: float,
        quantity: float,
        **kwargs
    ) -> float:
        """
        Calculate the commission for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
            
        Returns:
            Commission amount
        """
        pass
    
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


class ZeroCostModel(TransactionCostModel):
    """
    Transaction cost model with zero costs.
    """
    
    def calculate_commission(
        self,
        price: float,
        quantity: float,
        **kwargs
    ) -> float:
        """
        Calculate the commission for a trade (always zero).
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
            
        Returns:
            Zero commission
        """
        return 0.0
    
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


class FixedCostModel(TransactionCostModel):
    """
    Transaction cost model with fixed commission per trade.
    """
    
    def __init__(self, commission: float = 0.0):
        """
        Initialize the fixed cost model.
        
        Args:
            commission: Fixed commission per trade
        """
        self.commission = commission
    
    def calculate_commission(
        self,
        price: float,
        quantity: float,
        **kwargs
    ) -> float:
        """
        Calculate the commission for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
            
        Returns:
            Fixed commission amount
        """
        return self.commission
    
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


class PercentageCostModel(TransactionCostModel):
    """
    Transaction cost model with percentage-based commission.
    """
    
    def __init__(
        self,
        commission_pct: float = 0.0,
        min_commission: float = 0.0,
        max_commission: Optional[float] = None
    ):
        """
        Initialize the percentage cost model.
        
        Args:
            commission_pct: Commission percentage (e.g., 0.1 for 0.1%)
            min_commission: Minimum commission amount
            max_commission: Maximum commission amount (None for no maximum)
        """
        self.commission_pct = commission_pct / 100.0  # Convert to decimal
        self.min_commission = min_commission
        self.max_commission = max_commission
    
    def calculate_commission(
        self,
        price: float,
        quantity: float,
        **kwargs
    ) -> float:
        """
        Calculate the commission for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
            
        Returns:
            Commission amount
        """
        trade_value = price * quantity
        commission = trade_value * self.commission_pct
        
        # Apply minimum commission
        commission = max(commission, self.min_commission)
        
        # Apply maximum commission if specified
        if self.max_commission is not None:
            commission = min(commission, self.max_commission)
        
        return commission
    
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


class FixedSlippageModel(TransactionCostModel):
    """
    Transaction cost model with fixed slippage and commission.
    """
    
    def __init__(
        self,
        slippage_pct: float = 0.0,
        commission_model: Optional[TransactionCostModel] = None
    ):
        """
        Initialize the fixed slippage model.
        
        Args:
            slippage_pct: Slippage percentage (e.g., 0.1 for 0.1%)
            commission_model: Commission model to use (None for zero commission)
        """
        self.slippage_pct = slippage_pct / 100.0  # Convert to decimal
        self.commission_model = commission_model or ZeroCostModel()
    
    def calculate_commission(
        self,
        price: float,
        quantity: float,
        **kwargs
    ) -> float:
        """
        Calculate the commission for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
            
        Returns:
            Commission amount from the underlying commission model
        """
        return self.commission_model.calculate_commission(price, quantity, **kwargs)
    
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


class VariableSlippageModel(TransactionCostModel):
    """
    Transaction cost model with volume-based slippage.
    """
    
    def __init__(
        self,
        base_slippage_pct: float = 0.0,
        volume_impact_factor: float = 0.1,
        commission_model: Optional[TransactionCostModel] = None
    ):
        """
        Initialize the variable slippage model.
        
        Args:
            base_slippage_pct: Base slippage percentage (e.g., 0.1 for 0.1%)
            volume_impact_factor: Factor for volume impact on slippage
            commission_model: Commission model to use (None for zero commission)
        """
        self.base_slippage_pct = base_slippage_pct / 100.0  # Convert to decimal
        self.volume_impact_factor = volume_impact_factor
        self.commission_model = commission_model or ZeroCostModel()
    
    def calculate_commission(
        self,
        price: float,
        quantity: float,
        **kwargs
    ) -> float:
        """
        Calculate the commission for a trade.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            **kwargs: Additional parameters
            
        Returns:
            Commission amount from the underlying commission model
        """
        return self.commission_model.calculate_commission(price, quantity, **kwargs)
    
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