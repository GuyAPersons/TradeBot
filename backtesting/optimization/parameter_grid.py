from typing import Dict, List, Any, Tuple, Callable, Optional, Union, Iterator
import itertools
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
import json
import os
from datetime import datetime

class ParameterGrid:
    """
    Grid search for parameter optimization.
    """
    
    def __init__(self, param_grid: Dict[str, List[Any]], strategy_class: Any = None):
        """
        Initialize parameter grid.
        
        Args:
            param_grid: Dictionary of parameters to grid search over
            strategy_class: Strategy class to optimize (optional)
        """
        self.param_grid = param_grid
        self.strategy_class = strategy_class
        self.logger = logging.getLogger(__name__)
        self.results = []
        self.best_params = None
        self.best_metric = None
        self.best_result = None
    
    def generate_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations.
        
        Returns:
            List of parameter dictionaries
        """
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)
        
        return combinations
    
    def evaluate_params(self, params: Dict[str, Any], data: Any, metric_func: Callable, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a single parameter combination.
        
        Args:
            params: Parameter dictionary
            data: Data for backtesting
            metric_func: Function to calculate evaluation metric
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Dictionary with parameters and results
        """
        try:
            start_time = time.time()
            
            # Initialize strategy with parameters
            if self.strategy_class is not None:
                strategy_params = {**kwargs, **params}
                strategy = self.strategy_class(**strategy_params)
                
                # Run backtest
                strategy.backtest(data)
                
                # Calculate metric
                metric_value = metric_func(strategy)
            else:
                # If no strategy class is provided, assume metric_func takes params directly
                metric_value = metric_func(params, data, **kwargs)
            
            execution_time = time.time() - start_time
            
            result = {
                'params': params,
                'metric': metric_value,
                'execution_time': execution_time
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error evaluating parameters {params}: {str(e)}")
            return {
                'params': params,
                'metric': float('-inf') if kwargs.get('maximize', True) else float('inf'),
                'error': str(e),
                'execution_time': 0
            }
    
    def optimize(self, data: Any, metric_func: Callable, maximize: bool = True, 
                n_jobs: int = 1, verbose: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Run grid search optimization.
        
        Args:
            data: Data for backtesting
            metric_func: Function to calculate evaluation metric
            maximize: Whether to maximize (True) or minimize (False) the metric
            n_jobs: Number of parallel jobs (default: 1)
            verbose: Whether to print progress (default: True)
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Dictionary with best parameters and results
        """
        # Generate parameter combinations
        combinations = self.generate_combinations()
        total_combinations = len(combinations)
        
        if verbose:
            self.logger.info(f"Running grid search with {total_combinations} parameter combinations")
        
        # Run grid search
        self.results = []
        
        if n_jobs > 1:
            # Parallel execution
            evaluate_func = partial(self.evaluate_params, data=data, metric_func=metric_func, maximize=maximize, **kwargs)
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(evaluate_func, params) for params in combinations]
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    self.results.append(result)
                    
                    if verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                        self.logger.info(f"Progress: {i + 1}/{total_combinations} combinations evaluated")
        else:
            # Sequential execution
            for i, params in enumerate(combinations):
                result = self.evaluate_params(params, data, metric_func, maximize=maximize, **kwargs)
                self.results.append(result)
                
                if verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                    self.logger.info(f"Progress: {i + 1}/{total_combinations} combinations evaluated")
        
        # Find best parameters
        if maximize:
            best_result = max(self.results, key=lambda x: x['metric'])
        else:
            best_result = min(self.results, key=lambda x: x['metric'])
        
        self.best_params = best_result['params']
        self.best_metric = best_result['metric']
        self.best_result = best_result
        
        if verbose:
            self.logger.info(f"Best parameters: {self.best_params}")
            self.logger.info(f"Best metric value: {self.best_metric}")
        
        return best_result
    
    def get_results_df(self) -> pd.DataFrame:
        """
        Get results as a DataFrame.
        
        Returns:
            DataFrame of results
        """
        if not self.results:
            return pd.DataFrame()
        
        # Extract parameters and flatten them
        flat_results = []
        
        for result in self.results:
            flat_result = {}
            
            # Add parameters
            for param_name, param_value in result['params'].items():
                flat_result[f"param_{param_name}"] = param_value
            
            # Add metric and execution time
            flat_result['metric'] = result['metric']
            flat_result['execution_time'] = result['execution_time']
            
            # Add error if present
            if 'error' in result:
                flat_result['error'] = result['error']
            
            flat_results.append(flat_result)
        
        return pd.DataFrame(flat_results)
    
    def save_results(self, filepath: str) -> None:
        """
        Save results to a file.
        
        Args:
            filepath: Path to save results
        """
        results_df = self.get_results_df()
        
        # Determine file format
        if filepath.endswith('.csv'):
            results_df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            results_df.to_json(filepath, orient='records')
        elif filepath.endswith('.xlsx'):
            results_df.to_excel(filepath, index=False)
        else:
            # Default to CSV
            results_df.to_csv(filepath, index=False)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> pd.DataFrame:
        """
        Load results from a file.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            DataFrame of results
        """
        # Determine file format
        if filepath.endswith('.csv'):
            results_df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            results_df = pd.read_json(filepath, orient='records')
        elif filepath.endswith('.xlsx'):
            results_df = pd.read_excel(filepath)
        else:
            # Default to CSV
            results_df = pd.read_csv(filepath)
        
        # Convert DataFrame back to results list
        self.results = []
        
        for _, row in results_df.iterrows():
            # Extract parameters
            params = {}
            for col in results_df.columns:
                if col.startswith('param_'):
                    param_name = col[6:]  # Remove 'param_' prefix
                    params[param_name] = row[col]
            
            result = {
                'params': params,
                'metric': row['metric'],
                'execution_time': row['execution_time']
            }
            
            if 'error' in row:
                result['error'] = row['error']
            
            self.results.append(result)
        
        # Find best parameters
        if self.results:
            best_result = max(self.results, key=lambda x: x['metric'])
            self.best_params = best_result['params']
            self.best_metric = best_result['metric']
            self.best_result = best_result
        
        self.logger.info(f"Results loaded from {filepath}")
        return results_df
    
    def plot_results(self, param_names: Optional[List[str]] = None, 
                    metric_name: str = 'Metric', figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot grid search results.
        
        Args:
            param_names: List of parameter names to plot (default: all)
            metric_name: Name of the metric for plot labels
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.error("Matplotlib and/or Seaborn not available for plotting")
            return
        
        if not self.results:
            self.logger.warning("No results to plot")
            return
        
        results_df = self.get_results_df()
        
        # Get parameter columns
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        
        if param_names is not None:
            param_cols = [f"param_{name}" for name in param_names if f"param_{name}" in param_cols]
        
        if not param_cols:
            self.logger.warning("No parameter columns found for plotting")
            return
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot based on number of parameters
        if len(param_cols) == 1:
            # One parameter: line plot
            param_name = param_cols[0][6:]  # Remove 'param_' prefix
            plt.figure(figsize=figsize)
            sns.lineplot(x=param_cols[0], y='metric', data=results_df)
            plt.title(f"{metric_name} vs {param_name}")
            plt.xlabel(param_name)
            plt.ylabel(metric_name)
            plt.grid(True)
            
            # Mark best parameter
            if self.best_params is not None:
                best_param_value = self.best_params[param_name]
                plt.axvline(x=best_param_value, color='r', linestyle='--', 
                           label=f"Best: {param_name}={best_param_value}, {metric_name}={self.best_metric:.4f}")
                plt.legend()
            
        elif len(param_cols) == 2:
            # Two parameters: heatmap
            param1 = param_cols[0][6:]  # Remove 'param_' prefix
            param2 = param_cols[1][6:]  # Remove 'param_' prefix
            
            # Create pivot table
            pivot_table = results_df.pivot_table(
                values='metric', 
                index=param_cols[0], 
                columns=param_cols[1]
            )
            
            # Plot heatmap
            plt.figure(figsize=figsize)
            sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.4f')
            plt.title(f"{metric_name} for different {param1} and {param2} values")
            plt.xlabel(param2)
            plt.ylabel(param1)
            
            # Mark best parameters
            if self.best_params is not None:
                best_param1_value = self.best_params[param1]
                best_param2_value = self.best_params[param2]
                plt.text(0.5, 0.01, 
                        f"Best: {param1}={best_param1_value}, {param2}={best_param2_value}, {metric_name}={self.best_metric:.4f}",
                        horizontalalignment='center', verticalalignment='bottom', 
                        transform=plt.gca().transAxes, fontsize=12, color='red')
            
        else:
            # More than two parameters: pairplot
            # Create a copy of results_df with parameter names without 'param_' prefix
            plot_df = results_df.copy()
            for col in param_cols:
                plot_df[col[6:]] = plot_df[col]  # Create column without 'param_' prefix
            
            param_names_no_prefix = [col[6:] for col in param_cols]
            
            # Create pairplot
            sns.pairplot(plot_df, vars=param_names_no_prefix, hue='metric', palette='viridis', 
                         diag_kind='kde', plot_kws={'alpha': 0.6})
            plt.suptitle(f"Pairplot of Parameters vs {metric_name}", y=1.02)
        
        plt.tight_layout()
        plt.show()
    
    def get_top_n_params(self, n: int = 5, maximize: bool = True) -> List[Dict[str, Any]]:
        """
        Get top N parameter combinations.
        
        Args:
            n: Number of top combinations to return
            maximize: Whether to maximize (True) or minimize (False) the metric
            
        Returns:
            List of top parameter combinations
        """
        if not self.results:
            return []
        
        # Sort results
        if maximize:
            sorted_results = sorted(self.results, key=lambda x: x['metric'], reverse=True)
        else:
            sorted_results = sorted(self.results, key=lambda x: x['metric'])
        
        # Return top N
        return sorted_results[:n]
    
    def parameter_importance(self) -> pd.DataFrame:
        """
        Calculate parameter importance.
        
        Returns:
            DataFrame with parameter importance
        """
        if not self.results:
            return pd.DataFrame()
        
        results_df = self.get_results_df()
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        
        if not param_cols:
            return pd.DataFrame()
        
        # Calculate importance for each parameter
        importance = {}
        
        for param_col in param_cols:
            param_name = param_col[6:]  # Remove 'param_' prefix
            
            # Group by parameter value and calculate mean metric
            grouped = results_df.groupby(param_col)['metric'].mean()
            
            # Calculate range of mean metrics
            metric_range = grouped.max() - grouped.min()
            
            # Calculate standard deviation of mean metrics
            metric_std = grouped.std()
            
            importance[param_name] = {
                'range': metric_range,
                'std': metric_std,
                'unique_values': len(grouped),
                'normalized_importance': 0.0  # Will be calculated after all parameters are processed
            }
        
        # Calculate normalized importance
        total_range = sum(item['range'] for item in importance.values())
        if total_range > 0:
            for param_name in importance:
                importance[param_name]['normalized_importance'] = importance[param_name]['range'] / total_range
        
        # Convert to DataFrame
        importance_df = pd.DataFrame.from_dict(importance, orient='index')
        
        # Sort by normalized importance
        importance_df = importance_df.sort_values('normalized_importance', ascending=False)
        
        return importance_df
    
    def cross_validate(self, data_splits: List[Tuple[Any, Any]], metric_func: Callable, 
                      maximize: bool = True, n_jobs: int = 1, verbose: bool = True, 
                      **kwargs) -> Dict[str, Any]:
        """
        Perform cross-validation for parameter optimization.
        
        Args:
            data_splits: List of (train_data, test_data) tuples
            metric_func: Function to calculate evaluation metric
            maximize: Whether to maximize (True) or minimize (False) the metric
            n_jobs: Number of parallel jobs (default: 1)
            verbose: Whether to print progress (default: True)
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Dictionary with cross-validation results
        """
        if not data_splits:
            self.logger.error("No data splits provided for cross-validation")
            return {}
        
        # Generate parameter combinations
        combinations = self.generate_combinations()
        total_combinations = len(combinations)
        
        if verbose:
            self.logger.info(f"Running cross-validation with {total_combinations} parameter combinations and {len(data_splits)} data splits")
        
        # Initialize results
        cv_results = {
            'params': [],
            'mean_metric': [],
            'std_metric': [],
            'split_metrics': [],
            'execution_time': []
        }
        
        # Run cross-validation for each parameter combination
        for i, params in enumerate(combinations):
            start_time = time.time()
            split_metrics = []
            
            for j, (train_data, test_data) in enumerate(data_splits):
                try:
                    # Initialize strategy with parameters
                    if self.strategy_class is not None:
                        strategy_params = {**kwargs, **params}
                        strategy = self.strategy_class(**strategy_params)
                        
                        # Train on training data
                        strategy.backtest(train_data)
                        
                        # Evaluate on test data
                        metric_value = metric_func(strategy, test_data)
                    else:
                        # If no strategy class is provided, assume metric_func takes params directly
                        metric_value = metric_func(params, train_data, test_data, **kwargs)
                    
                    split_metrics.append(metric_value)
                    
                    if verbose and total_combinations <= 10:
                        self.logger.info(f"Params {i+1}/{total_combinations}, Split {j+1}/{len(data_splits)}: Metric = {metric_value:.4f}")
                
                except Exception as e:
                    self.logger.error(f"Error in cross-validation for parameters {params}, split {j+1}: {str(e)}")
                    split_metrics.append(float('-inf') if maximize else float('inf'))
            
            # Calculate mean and std of metrics across splits
            mean_metric = np.mean(split_metrics)
            std_metric = np.std(split_metrics)
            execution_time = time.time() - start_time
            
            # Store results
            cv_results['params'].append(params)
            cv_results['mean_metric'].append(mean_metric)
            cv_results['std_metric'].append(std_metric)
            cv_results['split_metrics'].append(split_metrics)
            cv_results['execution_time'].append(execution_time)
            
            if verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                self.logger.info(f"Progress: {i + 1}/{total_combinations} combinations evaluated")
        
        # Find best parameters
        if maximize:
            best_idx = np.argmax(cv_results['mean_metric'])
        else:
            best_idx = np.argmin(cv_results['mean_metric'])
        
        best_params = cv_results['params'][best_idx]
        best_mean_metric = cv_results['mean_metric'][best_idx]
        best_std_metric = cv_results['std_metric'][best_idx]
        
        # Store best parameters
        self.best_params = best_params
        self.best_metric = best_mean_metric
        
        if verbose:
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Best mean metric: {best_mean_metric:.4f} ± {best_std_metric:.4f}")
        
        # Create summary
        cv_summary = {
            'best_params': best_params,
            'best_mean_metric': best_mean_metric,
            'best_std_metric': best_std_metric,
            'all_results': cv_results
        }
        
        return cv_summary
    
    def random_search(self, data: Any, metric_func: Callable, n_iterations: int = 100,
                     maximize: bool = True, n_jobs: int = 1, verbose: bool = True,
                     random_state: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform random search for parameter optimization.
        
        Args:
            data: Data for backtesting
            metric_func: Function to calculate evaluation metric
            n_iterations: Number of random parameter combinations to try
            maximize: Whether to maximize (True) or minimize (False) the metric
            n_jobs: Number of parallel jobs (default: 1)
            verbose: Whether to print progress (default: True)
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Dictionary with best parameters and results
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate all parameter combinations
        all_combinations = self.generate_combinations()
        total_combinations = len(all_combinations)
        
        # If n_iterations is greater than total combinations, use all combinations
        if n_iterations >= total_combinations:
            if verbose:
                self.logger.info(f"n_iterations ({n_iterations}) >= total combinations ({total_combinations}), using all combinations")
            return self.optimize(data, metric_func, maximize, n_jobs, verbose, **kwargs)
        
        # Sample random combinations
        random_indices = np.random.choice(total_combinations, size=n_iterations, replace=False)
        random_combinations = [all_combinations[i] for i in random_indices]
        
        if verbose:
            self.logger.info(f"Running random search with {n_iterations} parameter combinations")
        
        # Run random search
        self.results = []
        
        if n_jobs > 1:
            # Parallel execution
            evaluate_func = partial(self.evaluate_params, data=data, metric_func=metric_func, maximize=maximize, **kwargs)
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(evaluate_func, params) for params in random_combinations]
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    self.results.append(result)
                    
                    if verbose and (i + 1) % max(1, n_iterations // 10) == 0:
                        self.logger.info(f"Progress: {i + 1}/{n_iterations} combinations evaluated")
        else:
            # Sequential execution
            for i, params in enumerate(random_combinations):
                result = self.evaluate_params(params, data, metric_func, maximize=maximize, **kwargs)
                self.results.append(result)
                
                if verbose and (i + 1) % max(1, n_iterations // 10) == 0:
                    self.logger.info(f"Progress: {i + 1}/{n_iterations} combinations evaluated")
        
        # Find best parameters
        if maximize:
            best_result = max(self.results, key=lambda x: x['metric'])
        else:
            best_result = min(self.results, key=lambda x: x['metric'])
        
        self.best_params = best_result['params']
        self.best_metric = best_result['metric']
        self.best_result = best_result
        
        if verbose:
            self.logger.info(f"Best parameters: {self.best_params}")
            self.logger.info(f"Best metric value: {self.best_metric}")
        
        return best_result
    
    def parameter_sensitivity(self, base_params: Dict[str, Any], data: Any, metric_func: Callable,
                             n_points: int = 10, maximize: bool = True, verbose: bool = True,
                             **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Analyze parameter sensitivity around a base parameter set.
        
        Args:
            base_params: Base parameter set
            data: Data for backtesting
            metric_func: Function to calculate evaluation metric
            n_points: Number of points to evaluate for each parameter
            maximize: Whether to maximize (True) or minimize (False) the metric
            verbose: Whether to print progress (default: True)
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Dictionary with parameter sensitivity results
        """
        sensitivity_results = {}
        
        for param_name, param_values in self.param_grid.items():
            if verbose:
                self.logger.info(f"Analyzing sensitivity for parameter: {param_name}")
            
            # Create parameter combinations by varying only this parameter
            param_combinations = []
            metrics = []
            
            for value in param_values:
                # Create a copy of base parameters and update the current parameter
                params = base_params.copy()
                params[param_name] = value
                
                # Evaluate parameters
                result = self.evaluate_params(params, data, metric_func, maximize=maximize, **kwargs)
                
                param_combinations.append(params)
                metrics.append(result['metric'])
            
            # Create sensitivity DataFrame
            sensitivity_df = pd.DataFrame({
                'parameter_value': param_values,
                'metric': metrics
            })
            
            # Calculate normalized sensitivity
            if len(metrics) > 1:
                metric_range = max(metrics) - min(metrics)
                if metric_range > 0:
                    sensitivity_df['normalized_sensitivity'] = (sensitivity_df['metric'] - min(metrics)) / metric_range
                else:
                    sensitivity_df['normalized_sensitivity'] = 0
            else:
                sensitivity_df['normalized_sensitivity'] = 0
            
            sensitivity_results[param_name] = sensitivity_df
        
        return sensitivity_results