"""
Utilities module for the backtesting framework.

This module provides utility functions for configuration, serialization,
and validation of backtest inputs and results.
"""

from .config import Config
from .serialization import Serializer
from .validation import DataValidator

__all__ = ['Config', 'Serializer', 'DataValidator']
