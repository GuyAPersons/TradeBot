"""
Execution module for the backtesting framework.

This module provides handlers for executing orders in both simulated
backtesting environments and potentially with live brokers.
"""

from .execution_handler import ExecutionHandler
from .simulated_execution import SimulatedExecutionHandler
from .broker_execution import BrokerExecutionHandler

__all__ = [
    'ExecutionHandler',
    'SimulatedExecutionHandler',
    'BrokerExecutionHandler'
]
