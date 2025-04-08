"""
Metrics module for the backtesting framework.

This module provides tools for calculating performance metrics, risk metrics,
and statistical analysis of backtest results.
"""

from .performance import PerformanceMetrics
from .risk import RiskMetrics
from .statistical import StatisticalAnalysis

__all__ = ['PerformanceMetrics', 'RiskMetrics', 'StatisticalAnalysis']
