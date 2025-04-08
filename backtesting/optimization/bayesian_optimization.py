from typing import Dict, List, Any, Tuple, Callable, Optional, Union
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class BayesianOptimization:
    """
    Bayesian optimization for parameter tuning.
    """
    
    def __init__(self, param_space: Dict[str, Tuple[float, float]], strategy_class: Any = None):
        """
        Initialize Bayesian optimization.
        
        Args:
            param_space: Dictionary of parameter spaces (parameter_name -> (min_value, max_value))
            strategy_class: Strategy class to optimize (optional)
        """
        self.param_space = param_space
        self.strategy_class = strategy_class
        self.logger = logging.getLogger(__name__)
        self.results = []
        self.best_params = None
        self.best_metric = None
        self.best_result = None
        
        # Check if scikit-optimize is available
        try:
            import skopt
            self.skopt_available = True
        except ImportError:
            self.logger.warning("scikit-optimize not available. Please install it with 'pip install scikit-optimize'")
            self.skopt_available = False
    
    def _convert_params_to_skopt_space(self) -> List[Any]:
        """
        Convert parameter space to scikit-optimize space.
        
        Returns:
            List of parameter spaces for scikit-optimize
        """
        if not self.skopt_available:
            self.logger.error("scikit-optimize not available")
            return []
        
        import skopt.space as skspace
        
        space = []
        
        for param_name, param_range in self.param_space.items():
            min_val, max_val = param_range
            
            # Check if parameter is integer
            if isinstance(min_val, int) and isinstance(max_val, int):
                space.append(skspace.Integer(min_val, max_val, name=param_name))
            else:
                space.append(skspace.Real(min_val, max_val, name=param_name))
        
        return space
    
    def _evaluate_params(self, params: List[float], data: Any, metric_func: Callable, 
                        param_names: List[str], maximize: bool = True, **kwargs) -> float:
        """
        Evaluate a set of parameters.
        
        Args:
            params: List of parameter values
            data: Data for backtesting
            metric_func: Function to calculate evaluation metric
            param_names: List of parameter names
            maximize: Whether to maximize (True) or minimize (False) the metric
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Metric value (negated if maximizing)
        """
        # Convert params list to dictionary
        param_dict = dict(zip(param_names, params))
        
        try:
            start_time = time.time()
            
            # Initialize strategy with parameters
            if self.strategy_class is not None:
                strategy_params = {**kwargs, **param_dict}
                strategy = self.strategy_class(**strategy_params)
                
                # Run backtest
                strategy.backtest(data)
                
                # Calculate metric
                metric_value = metric_func(strategy)
            else:
                # If no strategy class is provided, assume metric_func takes params directly
                metric_value = metric_func(param_dict, data, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Store result
            result = {
                'params': param_dict,
                'metric': metric_value,
                'execution_time': execution_time
            }
            
            self.results.append(result)
            
            # Update best result
            if self.best_metric is None or (maximize and metric_value > self.best_metric) or (not maximize and metric_value < self.best_metric):
                self.best_params = param_dict
                self.best_metric = metric_value
                self.best_result = result
            
            # Return negated metric if maximizing (scikit-optimize minimizes by default)
            return -metric_value if maximize else metric_value
        
        except Exception as e:
            self.logger.error(f"Error evaluating parameters {param_dict}: {str(e)}")
            
            # Store error result
            result = {
                'params': param_dict,
                'metric': float('-inf') if maximize else float('inf'),
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            
            self.results.append(result)
            
            # Return a very bad value
            return float('inf') if maximize else float('-inf')
    
    def optimize(self, data: Any, metric_func: Callable, maximize: bool = True, 
                n_calls: int = 50, n_initial_points: int = 10, 
                acq_func: str = 'gp_hedge', random_state: Optional[int] = None,
                verbose: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Args:
            data: Data for backtesting
            metric_func: Function to calculate evaluation metric
            maximize: Whether to maximize (True) or minimize (False) the metric
            n_calls: Number of function evaluations (default: 50)
            n_initial_points: Number of initial random points (default: 10)
            acq_func: Acquisition function ('gp_hedge', 'EI', 'PI', 'LCB')
            random_state: Random state for reproducibility
            verbose: Whether to print progress (default: True)
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Dictionary with best parameters and results
        """
        if not self.skopt_available:
            self.logger.error("scikit-optimize not available. Please install it with 'pip install scikit-optimize'")
            return {}
        
        import skopt
        
        # Convert parameter space to scikit-optimize space
        space = self._convert_params_to_skopt_space()
        param_names = list(self.param_space.keys())
        
        if verbose:
            self.logger.info(f"Running Bayesian optimization with {n_calls} evaluations")
            self.logger.info(f"Parameter space: {self.param_space}")
        
        # Define objective function
        def objective(params):
            return self._evaluate_params(params, data, metric_func, param_names, maximize, **kwargs)
        
        # Run optimization
        start_time = time.time()
        
        result = skopt.gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            random_state=random_state,
            verbose=verbose
        )
        
        total_time = time.time() - start_time
        
        if verbose:
            self.logger.info(f"Bayesian optimization completed in {total_time:.2f} seconds")
            self.logger.info(f"Best parameters: {self.best_params}")
            self.logger.info(f"Best metric value: {self.best_metric}")
        
        # Store optimization result
        self.skopt_result = result
        
        return self.best_result
    
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
    
    def plot_convergence(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot convergence of the optimization.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.skopt_available:
            self.logger.error("scikit-optimize not available")
            return
        
        try:
            import matplotlib.pyplot as plt
            from skopt.plots import plot_convergence
        except ImportError:
            self.logger.error("Matplotlib or scikit-optimize plotting not available")
            return
        
        if not hasattr(self, 'skopt_result'):
            self.logger.warning("No optimization result to plot")
            return
        
        plt.figure(figsize=figsize)
        plot_convergence(self.skopt_result)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_evaluations(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot parameter evaluations.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.skopt_available:
            self.logger.error("scikit-optimize not available")
            return
        
        try:
            import matplotlib.pyplot as plt
            from skopt.plots import plot_evaluations
        except ImportError:
            self.logger.error("Matplotlib or scikit-optimize plotting not available")
            return
        
        if not hasattr(self, 'skopt_result'):
            self.logger.warning("No optimization result to plot")
            return
        
        plt.figure(figsize=figsize)
        plot_evaluations(self.skopt_result)
        plt.tight_layout()
        plt.show()
    
    def plot_objective(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot objective function.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.skopt_available:
            self.logger.error("scikit-optimize not available")
            return
        
        try:
            import matplotlib.pyplot as plt
            from skopt.plots import plot_objective
        except ImportError:
            self.logger.error("Matplotlib or scikit-optimize plotting not available")
            return
        
        if not hasattr(self, 'skopt_result'):
            self.logger.warning("No optimization result to plot")
            return
        
        plt.figure(figsize=figsize)
        plot_objective(self.skopt_result)
        plt.tight_layout()
        plt.show()
    
    def plot_regret(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot optimization regret.
        
        Args:
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
        
        # Get results as DataFrame
        results_df = self.get_results_df()
        
        # Calculate cumulative best metric
        if 'metric' in results_df.columns:
            cumulative_best = results_df['metric'].cummax() if results_df['metric'].max() > 0 else results_df['metric'].cummin()
            regret = cumulative_best.max() - cumulative_best
            
            # Plot regret
            plt.figure(figsize=figsize)
            plt.plot(range(len(regret)), regret)
            plt.title('Optimization Regret')
            plt.xlabel('Iteration')
            plt.ylabel('Regret')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            self.logger.warning("No metric column found in results")
    
    def plot_parameter_importance(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot parameter importance.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.skopt_available:
            self.logger.error("scikit-optimize not available")
            return
        
        try:
            import matplotlib.pyplot as plt
            from skopt.plots import plot_objective
        except ImportError:
            self.logger.error("Matplotlib or scikit-optimize plotting not available")
            return
        
        if not hasattr(self, 'skopt_result'):
            self.logger.warning("No optimization result to plot")
            return
        
        # Get parameter names
        param_names = list(self.param_space.keys())
        
        # Calculate parameter importance using scikit-optimize's feature importances
        if hasattr(self.skopt_result, 'models') and self.skopt_result.models:
            model = self.skopt_result.models[-1]  # Get the last model
            
            if hasattr(model, 'kernel_') and hasattr(model.kernel_, 'k2'):
                # Extract length scales from the kernel
                length_scales = model.kernel_.k2.length_scale
                
                if not isinstance(length_scales, np.ndarray):
                    length_scales = np.array([length_scales])
                
                # Inverse of length scale is proportional to importance
                importance = 1 / length_scales
                
                # Normalize importance
                importance = importance / importance.sum()
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'parameter': param_names[:len(importance)],
                    'importance': importance
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                # Plot importance
                plt.figure(figsize=figsize)
                sns.barplot(x='parameter', y='importance', data=importance_df)
                plt.title('Parameter Importance')
                plt.xlabel('Parameter')
                plt.ylabel('Importance')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                self.logger.warning("Kernel length scales not available for importance calculation")
        else:
            self.logger.warning("Models not available for importance calculation")
    
    def get_parameter_importance(self) -> pd.DataFrame:
        """
        Calculate parameter importance.
        
        Returns:
            DataFrame with parameter importance
        """
        if not self.skopt_available:
            self.logger.error("scikit-optimize not available")
            return pd.DataFrame()
        
        if not hasattr(self, 'skopt_result'):
            self.logger.warning("No optimization result to calculate importance")
            return pd.DataFrame()
        
        # Get parameter names
        param_names = list(self.param_space.keys())
        
        # Calculate parameter importance using scikit-optimize's feature importances
        if hasattr(self.skopt_result, 'models') and self.skopt_result.models:
            model = self.skopt_result.models[-1]  # Get the last model
            
            if hasattr(model, 'kernel_') and hasattr(model.kernel_, 'k2'):
                # Extract length scales from the kernel
                length_scales = model.kernel_.k2.length_scale
                
                if not isinstance(length_scales, np.ndarray):
                    length_scales = np.array([length_scales])
                
                # Inverse of length scale is proportional to importance
                importance = 1 / length_scales
                
                # Normalize importance
                importance = importance / importance.sum()
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'parameter': param_names[:len(importance)],
                    'importance': importance,
                    'length_scale': length_scales
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                return importance_df
            else:
                self.logger.warning("Kernel length scales not available for importance calculation")
        else:
            self.logger.warning("Models not available for importance calculation")
        
        return pd.DataFrame()