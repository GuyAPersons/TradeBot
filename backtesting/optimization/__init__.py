"""
Optimization module for the backtesting framework.

This module provides tools for optimizing strategy parameters using various
techniques such as grid search, genetic algorithms, and Bayesian optimization.
"""

from .parameter_grid import ParameterGrid
from .genetic_algorithm import GeneticOptimizer
from .bayesian_optimization import BayesianOptimizer

__all__ = ['ParameterGrid', 'GeneticOptimizer', 'BayesianOptimizer']
