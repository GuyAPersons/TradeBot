"""
Base execution handler class.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

from ..events.order_event import OrderEvent
from ..events.fill_event import FillEvent


class ExecutionHandler(ABC):
    """
    Abstract base class for all execution handlers.
    
    The ExecutionHandler abstract class handles the interaction between
    the backtesting system and the execution venue. It receives OrderEvent
    objects and returns FillEvent objects that contain information about
    the executed order.
    """
    
    def __init__(self):
        """Initialize the execution handler."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute_order(self, order_event: OrderEvent) -> Optional[FillEvent]:
        """
        Execute an order in the market.
        
        This method takes an OrderEvent and returns a FillEvent
        containing information about the execution.
        
        Args:
            order_event: The order to execute
            
        Returns:
            A FillEvent if the order was executed, None otherwise
        """
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[OrderEvent]:
        """
        Get a list of all open orders.
        
        Returns:
            A list of open OrderEvent objects
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            True if the order was canceled, False otherwise
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the execution handler to a dictionary for serialization.
        
        Returns:
            A dictionary representation of the execution handler
        """
        return {
            'type': self.__class__.__name__
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionHandler':
        """
        Create an execution handler from a dictionary.
        
        Args:
            data: Dictionary containing execution handler data
            
        Returns:
            An ExecutionHandler instance
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement from_dict")
