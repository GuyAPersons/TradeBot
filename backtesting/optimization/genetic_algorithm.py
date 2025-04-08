from typing import Dict, List, Any, Tuple, Callable, Optional, Union
import numpy as np
import pandas as pd
import logging
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

class GeneticAlgorithm:
    """
    Genetic algorithm for parameter optimization.
    """
    
    def __init__(self, param_space: Dict[str, List[Any]], strategy_class: Any = None):
        """
        Initialize genetic algorithm optimizer.
        
        Args:
            param_space: Dictionary of parameter spaces to search
            strategy_class: Strategy class to optimize (optional)
        """
        self.param_space = param_space
        self.strategy_class = strategy_class
        self.logger = logging.getLogger(__name__)
        self.results = []
        self.best_params = None
        self.best_metric = None
        self.best_individual = None
        self.population_history = []
        self.metric_history = []
    
    def _initialize_population(self, population_size: int, random_state: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Initialize random population.
        
        Args:
            population_size: Size of the population
            random_state: Random state for reproducibility
            
        Returns:
            List of parameter dictionaries (individuals)
        """
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        population = []
        
        for _ in range(population_size):
            individual = {}
            
            for param_name, param_values in self.param_space.items():
                # Randomly select a value from the parameter space
                individual[param_name] = random.choice(param_values)
            
            population.append(individual)
        
        return population
    
    def _evaluate_individual(self, individual: Dict[str, Any], data: Any, 
                            metric_func: Callable, **kwargs) -> Dict[str, Any]:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Parameter dictionary
            data: Data for backtesting
            metric_func: Function to calculate evaluation metric
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Dictionary with individual and fitness
        """
        try:
            start_time = time.time()
            
            # Initialize strategy with parameters
            if self.strategy_class is not None:
                strategy_params = {**kwargs, **individual}
                strategy = self.strategy_class(**strategy_params)
                
                # Run backtest
                strategy.backtest(data)
                
                # Calculate metric
                fitness = metric_func(strategy)
            else:
                # If no strategy class is provided, assume metric_func takes params directly
                fitness = metric_func(individual, data, **kwargs)
            
            execution_time = time.time() - start_time
            
            result = {
                'individual': individual,
                'fitness': fitness,
                'execution_time': execution_time
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error evaluating individual {individual}: {str(e)}")
            return {
                'individual': individual,
                'fitness': float('-inf') if kwargs.get('maximize', True) else float('inf'),
                'error': str(e),
                'execution_time': 0
            }
    
    def _select_parents(self, population: List[Dict[str, Any]], fitnesses: List[float], 
                       num_parents: int, maximize: bool = True) -> List[Dict[str, Any]]:
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            population: List of individuals
            fitnesses: List of fitness values
            num_parents: Number of parents to select
            maximize: Whether to maximize (True) or minimize (False) the fitness
            
        Returns:
            List of selected parents
        """
        parents = []
        tournament_size = max(2, len(population) // 5)  # 20% of population size, at least 2
        
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            
            # Select the best individual from the tournament
            if maximize:
                winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            else:
                winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
            
            parents.append(population[winner_idx])
        
        return parents
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], 
                  crossover_prob: float = 0.7) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            crossover_prob: Probability of crossover
            
        Returns:
            Tuple of two offspring
        """
        if random.random() > crossover_prob:
            # No crossover, return copies of parents
            return parent1.copy(), parent2.copy()
        
        # Perform uniform crossover
        child1 = {}
        child2 = {}
        
        for param_name in parent1.keys():
            if random.random() < 0.5:
                # Swap parameter values
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
            else:
                # Keep parameter values
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], mutation_prob: float = 0.1) -> Dict[str, Any]:
        """
        Perform mutation on an individual.
        
        Args:
            individual: Individual to mutate
            mutation_prob: Probability of mutation for each parameter
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        for param_name, param_values in self.param_space.items():
            # Mutate with probability mutation_prob
            if random.random() < mutation_prob:
                # Select a random value from parameter space, different from current value
                current_value = mutated[param_name]
                possible_values = [v for v in param_values if v != current_value]
                
                if possible_values:
                    mutated[param_name] = random.choice(possible_values)
        
        return mutated
    
    def optimize(self, data: Any, metric_func: Callable, maximize: bool = True, 
                population_size: int = 50, num_generations: int = 20, 
                crossover_prob: float = 0.7, mutation_prob: float = 0.1,
                elitism: int = 2, n_jobs: int = 1, verbose: bool = True,
                random_state: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.
        
        Args:
            data: Data for backtesting
            metric_func: Function to calculate evaluation metric
            maximize: Whether to maximize (True) or minimize (False) the metric
            population_size: Size of the population (default: 50)
            num_generations: Number of generations (default: 20)
            crossover_prob: Probability of crossover (default: 0.7)
            mutation_prob: Probability of mutation (default: 0.1)
            elitism: Number of best individuals to keep (default: 2)
            n_jobs: Number of parallel jobs (default: 1)
            verbose: Whether to print progress (default: True)
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Dictionary with best parameters and results
        """
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Initialize population
        population = self._initialize_population(population_size, random_state)
        
        # Initialize history
        self.population_history = [population.copy()]
        self.metric_history = []
        
        if verbose:
            self.logger.info(f"Running genetic algorithm with population size {population_size} for {num_generations} generations")
        
        # Run genetic algorithm
        for generation in range(num_generations):
            start_time = time.time()
            
            # Evaluate population
            evaluation_results = []
            
            if n_jobs > 1:
                # Parallel evaluation
                evaluate_func = partial(self._evaluate_individual, data=data, metric_func=metric_func, maximize=maximize, **kwargs)
                
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [executor.submit(evaluate_func, individual) for individual in population]
                    
                    for future in as_completed(futures):
                        result = future.result()
                        evaluation_results.append(result)
            else:
                # Sequential evaluation
                for individual in population:
                    result = self._evaluate_individual(individual, data, metric_func, maximize=maximize, **kwargs)
                    evaluation_results.append(result)
            
            # Extract fitness values
            fitnesses = [result['fitness'] for result in evaluation_results]
            
            # Store results
            self.results.extend(evaluation_results)
            
            # Find best individual in current generation
            if maximize:
                best_idx = np.argmax(fitnesses)
            else:
                best_idx = np.argmin(fitnesses)
            
            best_individual = population[best_idx]
            best_fitness = fitnesses[best_idx]
            
            # Update best overall individual
            if self.best_metric is None or (maximize and best_fitness > self.best_metric) or (not maximize and best_fitness < self.best_metric):
                self.best_params = best_individual
                self.best_metric = best_fitness
                self.best_individual = evaluation_results[best_idx]
            
            # Store generation metrics
            generation_metrics = {
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses),
                'execution_time': time.time() - start_time
            }
            
            self.metric_history.append(generation_metrics)
            
            if verbose:
                self.logger.info(f"Generation {generation + 1}/{num_generations}: Best fitness = {best_fitness:.4f}, Avg fitness = {np.mean(fitnesses):.4f}")
            
            # Check if this is the last generation
            if generation == num_generations - 1:
                break
            
            # Select elite individuals
            if elitism > 0:
                elite_indices = np.argsort(fitnesses)
                if maximize:
                    elite_indices = elite_indices[::-1]  # Reverse for maximization
                
                elite_indices = elite_indices[:elitism]
                elites = [population[i] for i in elite_indices]
            else:
                elites = []
            
            # Select parents
            num_parents = (population_size - len(elites)) // 2
            parents = self._select_parents(population, fitnesses, num_parents, maximize)
            
            # Create new population through crossover and mutation
            new_population = elites.copy()
            
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    # Perform crossover
                    child1, child2 = self._crossover(parents[i], parents[i + 1], crossover_prob)
                    
                    # Perform mutation
                    child1 = self._mutate(child1, mutation_prob)
                    child2 = self._mutate(child2, mutation_prob)
                    
                    new_population.extend([child1, child2])
                else:
                    # Odd number of parents, just mutate the last one
                    child = self._mutate(parents[i], mutation_prob)
                    new_population.append(child)
            
            # Ensure population size remains constant
            if len(new_population) < population_size:
                # Add random individuals if needed
                additional = self._initialize_population(population_size - len(new_population), random_state)
                new_population.extend(additional)
            elif len(new_population) > population_size:
                # Truncate if too many individuals
                new_population = new_population[:population_size]
            
            # Update population
            population = new_population
            
            # Store population history
            self.population_history.append(population.copy())
        
        if verbose:
            self.logger.info(f"Genetic algorithm completed")
            self.logger.info(f"Best parameters: {self.best_params}")
            self.logger.info(f"Best metric value: {self.best_metric}")
        
        return self.best_individual
    
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
            for param_name, param_value in result['individual'].items():
                flat_result[f"param_{param_name}"] = param_value
            
            # Add fitness and execution time
            flat_result['fitness'] = result['fitness']
            flat_result['execution_time'] = result['execution_time']
            
            # Add error if present
            if 'error' in result:
                flat_result['error'] = result['error']
            
            flat_results.append(flat_result)
        
        return pd.DataFrame(flat_results)
    
    def get_generation_metrics_df(self) -> pd.DataFrame:
        """
        Get generation metrics as a DataFrame.
        
        Returns:
            DataFrame of generation metrics
        """
        if not self.metric_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metric_history)
    
    def plot_fitness_evolution(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot the evolution of fitness over generations.
        
        Args:
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.error("Matplotlib and/or Seaborn not available for plotting")
            return
        
        if not self.metric_history:
            self.logger.warning("No metric history to plot")
            return
        
        metrics_df = self.get_generation_metrics_df()
        
        plt.figure(figsize=figsize)
        
        # Plot best and average fitness
        plt.plot(metrics_df['generation'], metrics_df['best_fitness'], 'b-', label='Best Fitness')
        plt.plot(metrics_df['generation'], metrics_df['avg_fitness'], 'r-', label='Average Fitness')
        
        # Add error bands for average fitness
        plt.fill_between(
            metrics_df['generation'],
            metrics_df['avg_fitness'] - metrics_df['std_fitness'],
            metrics_df['avg_fitness'] + metrics_df['std_fitness'],
            alpha=0.2, color='r'
        )
        
        plt.title('Fitness Evolution Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_parameter_distribution(self, generation: int = -1, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot the distribution of parameter values in a specific generation.
        
        Args:
            generation: Generation to plot (-1 for last generation)
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.error("Matplotlib and/or Seaborn not available for plotting")
            return
        
        if not self.population_history:
            self.logger.warning("No population history to plot")
            return
        
        # Get population for the specified generation
        if generation < 0:
            generation = len(self.population_history) + generation
        
        if generation < 0 or generation >= len(self.population_history):
            self.logger.error(f"Invalid generation: {generation}")
            return
        
        population = self.population_history[generation]
        
        # Convert population to DataFrame
        population_data = []
        for individual in population:
            population_data.append(individual)
        
        population_df = pd.DataFrame(population_data)
        
        # Create figure
        num_params = len(self.param_space)
        fig, axes = plt.subplots(nrows=num_params, figsize=figsize)
        
        if num_params == 1:
            axes = [axes]
        
        # Plot distribution for each parameter
        for i, (param_name, param_values) in enumerate(self.param_space.items()):
            ax = axes[i]
            
            # Check parameter type and plot accordingly
            if all(isinstance(val, (int, float)) for val in param_values):
                # Numeric parameter: histogram
                sns.histplot(population_df[param_name], ax=ax, kde=True)
            else:
                # Categorical parameter: count plot
                sns.countplot(x=param_name, data=population_df, ax=ax)
            
            ax.set_title(f'Distribution of {param_name} (Generation {generation})')
            ax.grid(True)
            
            # Mark best parameter value
            if self.best_params is not None:
                best_value = self.best_params[param_name]
                ax.axvline(x=best_value, color='r', linestyle='--', 
                          label=f'Best: {best_value}')
                ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_parameter_evolution(self, param_name: str, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot the evolution of a parameter over generations.
        
        Args:
            param_name: Name of the parameter to plot
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.error("Matplotlib and/or Seaborn not available for plotting")
            return
        
        if not self.population_history:
            self.logger.warning("No population history to plot")
            return
        
        if param_name not in self.param_space:
            self.logger.error(f"Invalid parameter name: {param_name}")
            return
        
        # Extract parameter values for each generation
        param_values = []
        
        for generation, population in enumerate(self.population_history):
            generation_values = [individual[param_name] for individual in population]
            
            # Create DataFrame for this generation
            gen_df = pd.DataFrame({
                'generation': generation,
                'value': generation_values
            })
            
            param_values.append(gen_df)
        
        # Combine all generations
        param_df = pd.concat(param_values, ignore_index=True)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Check parameter type and plot accordingly
        if all(isinstance(val, (int, float)) for val in self.param_space[param_name]):
            # Numeric parameter: violin plot
            sns.violinplot(x='generation', y='value', data=param_df)
            
            # Add scatter plot for individual points
            sns.stripplot(x='generation', y='value', data=param_df, 
                         size=4, color='black', alpha=0.3)
        else:
            # Categorical parameter: stacked bar plot
            pivot_df = pd.crosstab(param_df['generation'], param_df['value'], normalize='index')
            pivot_df.plot(kind='bar', stacked=True, ax=plt.gca())
        
        plt.title(f'Evolution of {param_name} Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Parameter Value')
        plt.grid(True)
        
        # Mark best parameter value
        if self.best_params is not None:
            best_value = self.best_params[param_name]
            plt.axhline(y=best_value, color='r', linestyle='--', 
                      label=f'Best: {best_value}')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
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
    
    def save_generation_metrics(self, filepath: str) -> None:
        """
        Save generation metrics to a file.
        
        Args:
            filepath: Path to save generation metrics
        """
        metrics_df = self.get_generation_metrics_df()
        
        # Determine file format
        if filepath.endswith('.csv'):
            metrics_df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            metrics_df.to_json(filepath, orient='records')
        elif filepath.endswith('.xlsx'):
            metrics_df.to_excel(filepath, index=False)
        else:
            # Default to CSV
            metrics_df.to_csv(filepath, index=False)
        
        self.logger.info(f"Generation metrics saved to {filepath}")
    
    def get_diversity_metrics(self) -> pd.DataFrame:
        """
        Calculate population diversity metrics for each generation.
        
        Returns:
            DataFrame with diversity metrics
        """
        if not self.population_history:
            return pd.DataFrame()
        
        diversity_metrics = []
        
        for generation, population in enumerate(self.population_history):
            generation_metrics = {'generation': generation}
            
            # Calculate diversity for each parameter
            for param_name in self.param_space:
                param_values = [individual[param_name] for individual in population]
                
                # Count unique values
                unique_values = len(set(param_values))
                
                # Calculate entropy
                value_counts = pd.Series(param_values).value_counts(normalize=True)
                entropy = -sum(p * np.log2(p) for p in value_counts)
                
                generation_metrics[f'{param_name}_unique'] = unique_values
                generation_metrics[f'{param_name}_entropy'] = entropy
            
            diversity_metrics.append(generation_metrics)
        
        return pd.DataFrame(diversity_metrics)
    
    def plot_diversity_metrics(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot diversity metrics over generations.
        
        Args:
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.error("Matplotlib and/or Seaborn not available for plotting")
            return
        
        diversity_df = self.get_diversity_metrics()
        
        if diversity_df.empty:
            self.logger.warning("No diversity metrics to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(nrows=2, figsize=figsize)
        
        # Plot unique values
        unique_cols = [col for col in diversity_df.columns if col.endswith('_unique')]
        for col in unique_cols:
            param_name = col.replace('_unique', '')
            axes[0].plot(diversity_df['generation'], diversity_df[col], label=param_name)
        
        axes[0].set_title('Number of Unique Parameter Values')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Unique Values')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot entropy
        entropy_cols = [col for col in diversity_df.columns if col.endswith('_entropy')]
        for col in entropy_cols:
            param_name = col.replace('_entropy', '')
            axes[1].plot(diversity_df['generation'], diversity_df[col], label=param_name)
        
        axes[1].set_title('Parameter Entropy (Diversity)')
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Entropy')
        axes[1].grid(True)
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """
        Calculate convergence metrics.
        
        Returns:
            Dictionary with convergence metrics
        """
        if not self.metric_history:
            return {}
        
        metrics_df = self.get_generation_metrics_df()
        
        # Calculate convergence metrics
        convergence_metrics = {}
        
        # Generation of best fitness
        best_gen = metrics_df['best_fitness'].idxmax()
        convergence_metrics['best_generation'] = best_gen
        
        # Convergence speed (generations to reach 90% of best fitness)
        best_fitness = metrics_df['best_fitness'].max()
        threshold = 0.9 * best_fitness
        gens_to_threshold = metrics_df[metrics_df['best_fitness'] >= threshold].iloc[0]['generation']
        convergence_metrics['generations_to_90pct'] = gens_to_threshold
        
        # Fitness improvement rate
        if len(metrics_df) > 1:
            first_fitness = metrics_df.iloc[0]['best_fitness']
            last_fitness = metrics_df.iloc[-1]['best_fitness']
            improvement_rate = (last_fitness - first_fitness) / len(metrics_df)
            convergence_metrics['improvement_rate'] = improvement_rate
        
        # Fitness plateau detection
        plateau_threshold = 0.001  # 0.1% improvement
        plateau_window = min(5, len(metrics_df))
        
        for i in range(len(metrics_df) - plateau_window + 1):
            window = metrics_df.iloc[i:i+plateau_window]
            improvement = (window['best_fitness'].max() - window['best_fitness'].min()) / window['best_fitness'].min()
            
            if improvement < plateau_threshold:
                convergence_metrics['plateau_generation'] = i
                break
        
        return convergence_metrics