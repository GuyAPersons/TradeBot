"""
Visualization module for the backtesting framework.

This module provides tools for visualizing backtest results, equity curves,
drawdowns, and performance metrics.
"""

from .performance_plots import PerformancePlotter
from .equity_curves import EquityCurvePlotter
from .drawdown_analysis import DrawdownAnalyzer

__all__ = ['PerformancePlotter', 'EquityCurvePlotter', 'DrawdownAnalyzer']
