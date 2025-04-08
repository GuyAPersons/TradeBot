"""
Data handling module for the backtesting framework.

This module provides functionality for loading, preprocessing, and managing
historical market data for backtesting.
"""

from .data_handler import DataHandler
from .data_preprocessor import DataPreprocessor

__all__ = ['DataHandler', 'DataPreprocessor']
