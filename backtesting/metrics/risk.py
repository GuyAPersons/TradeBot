from typing import Dict, List, Union, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class RiskMetrics:
    """
    Risk metrics calculation for backtesting.
    """
    
    def __init__(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None):
        """
        Initialize risk metrics calculator.
        
        Args:
            returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns (optional)
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.logger = logging.getLogger(__name__)
    
    def calculate_drawdown(self) -> Tuple[pd.Series, pd.Series, float, int]:
        """
        Calculate drawdown statistics.
        
        Returns:
            Tuple of (drawdown series, drawdown duration series, maximum drawdown, maximum drawdown duration)
        """
        # Calculate cumulative returns
        cum_returns = (1 + self.returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cum_returns / running_max) - 1
        
        # Calculate drawdown duration
        drawdown_duration = pd.Series(0, index=self.returns.index)
        
        # Initialize variables for tracking drawdown periods
        in_drawdown = False
        drawdown_start = None
        
        for date, value in drawdown.items():
            if value < 0 and not in_drawdown:
                # Start of a new drawdown period
                in_drawdown = True
                drawdown_start = date
            elif value == 0 and in_drawdown:
                # End of drawdown period
                in_drawdown = False
                drawdown_start = None
            
            if in_drawdown and drawdown_start is not None:
                # Calculate duration of current drawdown
                drawdown_duration[date] = (date - drawdown_start).days
        
        # Calculate maximum drawdown and its duration
        max_drawdown = drawdown.min()
        max_drawdown_duration = drawdown_duration.max()
        
        return drawdown, drawdown_duration, max_drawdown, max_drawdown_duration
    
    def calculate_volatility(self, annualize: bool = True, trading_days: int = 252) -> float:
        """
        Calculate return volatility.
        
        Args:
            annualize: Whether to annualize the volatility (default: True)
            trading_days: Number of trading days per year (default: 252)
            
        Returns:
            Volatility
        """
        if len(self.returns) < 2:
            return 0.0
        
        volatility = self.returns.std()
        
        if annualize:
            volatility *= np.sqrt(trading_days)
        
        return volatility
    
    def calculate_var(self, confidence: float = 0.95, method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            confidence: Confidence level (default: 0.95)
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Value at Risk
        """
        if len(self.returns) < 10:
            self.logger.warning("Not enough data to calculate VaR")
            return 0.0
        
        if method == 'historical':
            # Historical VaR
            var = -np.percentile(self.returns, 100 * (1 - confidence))
        
        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            z_score = abs(np.percentile(np.random.normal(0, 1, 10000), 100 * (1 - confidence)))
            var = self.returns.mean() - z_score * self.returns.std()
            var = -var  # Convert to positive value
        
        elif method == 'monte_carlo':
            # Monte Carlo VaR
            mean = self.returns.mean()
            std = self.returns.std()
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean, std, 10000)
            
            # Calculate VaR
            var = -np.percentile(simulated_returns, 100 * (1 - confidence))
        
        else:
            self.logger.error(f"Unknown VaR method: {method}")
            var = 0.0
        
        return var
    
    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Conditional Value at Risk
        """
        if len(self.returns) < 10:
            self.logger.warning("Not enough data to calculate CVaR")
            return 0.0
        
        # Calculate VaR
        var = self.calculate_var(confidence, 'historical')
        
        # Calculate CVaR (average of returns below VaR)
        returns_below_var = self.returns[self.returns < -var]
        
        if len(returns_below_var) == 0:
            return var
        
        cvar = -returns_below_var.mean()
        
        return cvar
    
    def calculate_downside_deviation(self, mar: float = 0.0) -> float:
        """
        Calculate downside deviation.
        
        Args:
            mar: Minimum acceptable return (default: 0.0)
            
        Returns:
            Downside deviation
        """
        if len(self.returns) < 2:
            return 0.0
        
        # Calculate downside returns (returns below MAR)
        downside_returns = self.returns[self.returns < mar]
        
        if len(downside_returns) == 0:
            return 0.0
        
        # Calculate squared deviations from MAR
        squared_deviations = (downside_returns - mar) ** 2
        
        # Calculate downside deviation
        downside_deviation = np.sqrt(squared_deviations.mean())
        
        return downside_deviation
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0, mar: float = 0.0, annualize: bool = True, trading_days: int = 252) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.0)
            mar: Minimum acceptable return (default: 0.0)
            annualize: Whether to annualize the ratio (default: True)
            trading_days: Number of trading days per year (default: 252)
            
        Returns:
            Sortino ratio
        """
        if len(self.returns) < 2:
            return 0.0
        
        # Calculate excess returns
        excess_returns = self.returns.mean() - risk_free_rate
        
        # Calculate downside deviation
        downside_dev = self.calculate_downside_deviation(mar)
        
        if downside_dev == 0:
            return 0.0
        
        # Calculate Sortino ratio
        sortino = excess_returns / downside_dev
        
        if annualize:
            sortino *= np.sqrt(trading_days)
        
        return sortino
    
    def calculate_calmar_ratio(self, annualize: bool = True, trading_days: int = 252) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            annualize: Whether to annualize the returns (default: True)
            trading_days: Number of trading days per year (default: 252)
            
        Returns:
            Calmar ratio
        """
        if len(self.returns) < 10:
            self.logger.warning("Not enough data to calculate Calmar ratio")
            return 0.0
        
        # Calculate annualized return
        total_return = (1 + self.returns).prod() - 1
        days = len(self.returns)
        years = days / trading_days
        
        if years <= 0:
            return 0.0
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate maximum drawdown
        _, _, max_drawdown, _ = self.calculate_drawdown()
        
        if max_drawdown == 0:
            return 0.0
        
        # Calculate Calmar ratio
        calmar = annualized_return / abs(max_drawdown)
        
        return calmar
    
    def calculate_omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.
        
        Args:
            threshold: Return threshold (default: 0.0)
            
        Returns:
            Omega ratio
        """
        if len(self.returns) < 2:
            return 0.0
        
        # Calculate returns above and below threshold
        returns_above = self.returns[self.returns > threshold]
        returns_below = self.returns[self.returns < threshold]
        
        if len(returns_below) == 0:
            return float('inf')  # No downside
        
        # Calculate upside and downside
        upside = (returns_above - threshold).sum()
        downside = (threshold - returns_below).sum()
        
        if downside == 0:
            return float('inf')
        
        # Calculate Omega ratio
        omega = upside / downside
        
        return omega
    
    def calculate_beta(self) -> float:
        """
        Calculate beta relative to benchmark.
        
        Returns:
            Beta
        """
        if self.benchmark_returns is None or len(self.returns) < 10:
            self.logger.warning("Benchmark returns not provided or not enough data")
            return 0.0
        
        # Align returns with benchmark
        aligned_returns = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        
        if aligned_returns.empty or aligned_returns.shape[1] != 2:
            return 0.0
        
        # Extract aligned series
        strategy_returns = aligned_returns.iloc[:, 0]
        benchmark_returns = aligned_returns.iloc[:, 1]
        
        # Calculate covariance and variance
        covariance = strategy_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        
        if benchmark_variance == 0:
            return 0.0
        
        # Calculate beta
        beta = covariance / benchmark_variance
        
        return beta
    
    def calculate_alpha(self, risk_free_rate: float = 0.0, annualize: bool = True, trading_days: int = 252) -> float:
        """
        Calculate Jensen's alpha.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.0)
            annualize: Whether to annualize the alpha (default: True)
            trading_days: Number of trading days per year (default: 252)
            
        Returns:
            Alpha
        """
        if self.benchmark_returns is None or len(self.returns) < 10:
            self.logger.warning("Benchmark returns not provided or not enough data")
            return 0.0
        
        # Calculate beta
        beta = self.calculate_beta()
        
        # Align returns with benchmark
        aligned_returns = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        
        if aligned_returns.empty or aligned_returns.shape[1] != 2:
            return 0.0
        
        # Extract aligned series
        strategy_returns = aligned_returns.iloc[:, 0]
        benchmark_returns = aligned_returns.iloc[:, 1]
        
        # Calculate average returns
        avg_strategy_return = strategy_returns.mean()
        avg_benchmark_return = benchmark_returns.mean()
        
        # Calculate alpha
        alpha = avg_strategy_return - (risk_free_rate + beta * (avg_benchmark_return - risk_free_rate))
        
        if annualize:
            alpha *= trading_days
        
        return alpha
    
    def calculate_information_ratio(self) -> float:
        """
        Calculate Information Ratio.
        
        Returns:
            Information Ratio
        """
        if self.benchmark_returns is None or len(self.returns) < 10:
            self.logger.warning("Benchmark returns not provided or not enough data")
            return 0.0
        
        # Align returns with benchmark
        aligned_returns = pd.concat([self.returns, self.benchmark_returns], axis=1).dropna()
        
        if aligned_returns.empty or aligned_returns.shape[1] != 2:
            return 0.0
        
        # Extract aligned series
        strategy_returns = aligned_returns.iloc[:, 0]
        benchmark_returns = aligned_returns.iloc[:, 1]
        
        # Calculate tracking error
        tracking_error = (strategy_returns - benchmark_returns).std()
        
        if tracking_error == 0:
            return 0.0
        
        # Calculate information ratio
        information_ratio = (strategy_returns.mean() - benchmark_returns.mean()) / tracking_error
        
        return information_ratio
    
    def calculate_treynor_ratio(self, risk_free_rate: float = 0.0, annualize: bool = True, trading_days: int = 252) -> float:
        """
        Calculate Treynor Ratio.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.0)
            annualize: Whether to annualize the ratio (default: True)
            trading_days: Number of trading days per year (default: 252)
            
        Returns:
            Treynor Ratio
        """
        if self.benchmark_returns is None or len(self.returns) < 10:
            self.logger.warning("Benchmark returns not provided or not enough data")
            return 0.0
        
        # Calculate beta
        beta = self.calculate_beta()
        
        if beta == 0:
            return 0.0
        
        # Calculate excess return
        excess_return = self.returns.mean() - risk_free_rate
        
        # Calculate Treynor ratio
        treynor = excess_return / beta
        
        if annualize:
            treynor *= trading_days
        
        return treynor
    
    def calculate_tail_ratio(self, percentile: float = 0.05) -> float:
        """
        Calculate tail ratio.
        
        Args:
            percentile: Percentile for tail calculation (default: 0.05)
            
        Returns:
            Tail ratio
        """
        if len(self.returns) < 10:
            self.logger.warning("Not enough data to calculate tail ratio")
            return 0.0
        
        # Calculate right and left tails
        right_tail = np.percentile(self.returns, 100 * (1 - percentile))
        left_tail = np.percentile(self.returns, 100 * percentile)
        
        if left_tail == 0:
            return 0.0
        
        # Calculate tail ratio
        tail_ratio = abs(right_tail / left_tail)
        
        return tail_ratio
    
    def calculate_all_metrics(self, risk_free_rate: float = 0.0, annualize: bool = True, trading_days: int = 252) -> Dict[str, float]:
        """
        Calculate all risk metrics.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.0)
            annualize: Whether to annualize metrics (default: True)
            trading_days: Number of trading days per year (default: 252)
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        # Drawdown metrics
        drawdown, drawdown_duration, max_drawdown, max_drawdown_duration = self.calculate_drawdown()
        metrics['max_drawdown'] = max_drawdown
        metrics['max_drawdown_duration'] = max_drawdown_duration
        
        # Volatility
        metrics['volatility'] = self.calculate_volatility(annualize, trading_days)
        
        # VaR and CVaR
        metrics['var_95'] = self.calculate_var(0.95, 'historical')
        metrics['cvar_95'] = self.calculate_cvar(0.95)
        
        # Downside deviation
        metrics['downside_deviation'] = self.calculate_downside_deviation()
        
        # Sortino ratio
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(risk_free_rate, 0.0, annualize, trading_days)
        
        # Calmar ratio
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(annualize, trading_days)
        
        # Omega ratio
        metrics['omega_ratio'] = self.calculate_omega_ratio()
        
        # Tail ratio
        metrics['tail_ratio'] = self.calculate_tail_ratio()
        
        # Benchmark-related metrics (if benchmark provided)
        if self.benchmark_returns is not None:
            metrics['beta'] = self.calculate_beta()
            metrics['alpha'] = self.calculate_alpha(risk_free_rate, annualize, trading_days)
            metrics['information_ratio'] = self.calculate_information_ratio()
            metrics['treynor_ratio'] = self.calculate_treynor_ratio(risk_free_rate, annualize, trading_days)
        
        return metrics


class RiskAnalyzer:
    """
    Risk analyzer for portfolio and strategy analysis.
    """
    
    def __init__(self, returns: pd.Series, positions: Optional[pd.DataFrame] = None, prices: Optional[pd.DataFrame] = None):
        """
        Initialize risk analyzer.
        
        Args:
            returns: Series of strategy returns
            positions: DataFrame of positions (optional)
            prices: DataFrame of prices (optional)
        """
        self.returns = returns
        self.positions = positions
        self.prices = prices
        self.logger = logging.getLogger(__name__)
        
        # Initialize risk metrics calculator
        self.risk_metrics = RiskMetrics(returns)
    
    def set_benchmark(self, benchmark_returns: pd.Series) -> None:
        """
        Set benchmark returns for comparison.
        
        Args:
            benchmark_returns: Series of benchmark returns
        """
        self.risk_metrics.benchmark_returns = benchmark_returns
    
    def analyze_returns_distribution(self) -> Dict[str, float]:
        """
        Analyze returns distribution.
        
        Returns:
            Dictionary of distribution statistics
        """
        if len(self.returns) < 10:
            self.logger.warning("Not enough data to analyze returns distribution")
            return {}
        
        stats = {}
        
        # Basic statistics
        stats['mean'] = self.returns.mean()
        stats['median'] = self.returns.median()
        stats['std'] = self.returns.std()
        stats['min'] = self.returns.min()
        stats['max'] = self.returns.max()
        
        # Skewness and kurtosis
        stats['skewness'] = self.returns.skew()
        stats['kurtosis'] = self.returns.kurtosis()
        
        # Percentiles
        for p in [1, 5, 10, 25, 75, 90, 95, 99]:
            stats[f'percentile_{p}'] = np.percentile(self.returns, p)
        
        # Jarque-Bera test for normality
        try:
            from scipy import stats as scipy_stats
            jb_stat, jb_pvalue = scipy_stats.jarque_bera(self.returns)
            stats['jarque_bera_stat'] = jb_stat
            stats['jarque_bera_pvalue'] = jb_pvalue
            stats['is_normal'] = jb_pvalue > 0.05  # At 5% significance level
        except ImportError:
            self.logger.warning("SciPy not available for Jarque-Bera test")
        
        return stats
    
    def analyze_drawdowns(self, top_n: int = 5) -> pd.DataFrame:
        """
        Analyze top drawdowns.
        
        Args:
            top_n: Number of top drawdowns to analyze (default: 5)
            
        Returns:
            DataFrame of top drawdowns
        """
        drawdown, drawdown_duration, _, _ = self.risk_metrics.calculate_drawdown()
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, value in drawdown.items():
            if value < 0 and not in_drawdown:
                # Start of a new drawdown period
                in_drawdown = True
                start_date = date
            elif value == 0 and in_drawdown:
                # End of drawdown period
                if start_date is not None:
                    end_date = date
                    max_drawdown = drawdown[start_date:end_date].min()
                    recovery_time = (end_date - start_date).days
                    
                    drawdown_periods.append({
                        'start_date': start_date,
                        'end_date': end_date,
                        'max_drawdown': max_drawdown,
                        'recovery_time': recovery_time
                    })
                
                in_drawdown = False
                start_date = None
        
        # Handle ongoing drawdown
        if in_drawdown and start_date is not None:
            end_date = drawdown.index[-1]
            max_drawdown = drawdown[start_date:end_date].min()
            recovery_time = (end_date - start_date).days
            
            drawdown_periods.append({
                'start_date': start_date,
                'end_date': end_date,
                'max_drawdown': max_drawdown,
                'recovery_time': recovery_time,
                'ongoing': True
            })
        
        # Convert to DataFrame and sort by max drawdown
        if drawdown_periods:
            df_drawdowns = pd.DataFrame(drawdown_periods)
            df_drawdowns = df_drawdowns.sort_values('max_drawdown').head(top_n)
            return df_drawdowns
        else:
            return pd.DataFrame()
    
    def analyze_rolling_risk(self, window: int = 60) -> pd.DataFrame:
        """
        Calculate rolling risk metrics.
        
        Args:
            window: Rolling window size (default: 60)
            
        Returns:
            DataFrame of rolling risk metrics
        """
        if len(self.returns) < window:
            self.logger.warning(f"Not enough data for rolling window of {window}")
            return pd.DataFrame()
        
        # Initialize DataFrame for rolling metrics
        rolling_metrics = pd.DataFrame(index=self.returns.index[window-1:])
        
        # Calculate rolling volatility
        rolling_metrics['volatility'] = self.returns.rolling(window=window).std() * np.sqrt(252)
        
        # Calculate rolling drawdown
        cum_returns = (1 + self.returns).cumprod()
        rolling_max = cum_returns.rolling(window=window).max()
        rolling_metrics['drawdown'] = (cum_returns / rolling_max - 1).iloc[window-1:]
        
        # Calculate rolling VaR
        rolling_metrics['var_95'] = self.returns.rolling(window=window).quantile(0.05) * -1
        
        # Calculate rolling Sortino ratio
        def rolling_sortino(x):
            downside_returns = x[x < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0
            return x.mean() / downside_returns.std() * np.sqrt(252)
        
        rolling_metrics['sortino_ratio'] = self.returns.rolling(window=window).apply(rolling_sortino, raw=True)
        
        # Calculate rolling Sharpe ratio
        rolling_metrics['sharpe_ratio'] = (
            self.returns.rolling(window=window).mean() / 
            self.returns.rolling(window=window).std() * 
            np.sqrt(252)
        )
        
        return rolling_metrics
    
    def analyze_stress_periods(self, stress_periods: Dict[str, Tuple[datetime, datetime]]) -> pd.DataFrame:
        """
        Analyze performance during stress periods.
        
        Args:
            stress_periods: Dictionary of stress periods (name -> (start_date, end_date))
            
        Returns:
            DataFrame of stress period performance
        """
        stress_results = []
        
        for name, (start_date, end_date) in stress_periods.items():
            # Filter returns for stress period
            period_returns = self.returns[(self.returns.index >= start_date) & (self.returns.index <= end_date)]
            
            if len(period_returns) == 0:
                continue
            
            # Calculate metrics for stress period
            total_return = (1 + period_returns).prod() - 1
            volatility = period_returns.std() * np.sqrt(252)
            max_drawdown = (period_returns.cumsum() - period_returns.cumsum().cummax()).min()
            
            stress_results.append({
                'period': name,
                'start_date': start_date,
                'end_date': end_date,
                'total_return': total_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': total_return / volatility if volatility > 0 else 0
            })
        
        if stress_results:
            return pd.DataFrame(stress_results)
        else:
            return pd.DataFrame()
    
    def analyze_concentration_risk(self) -> Dict[str, Any]:
        """
        Analyze portfolio concentration risk.
        
        Returns:
            Dictionary of concentration risk metrics
        """
        if self.positions is None or self.prices is None:
            self.logger.warning("Positions or prices data not provided")
            return {}
        
        # Calculate position values
        position_values = pd.DataFrame()
        
        for column in self.positions.columns:
            if column in self.prices.columns:
                position_values[column] = self.positions[column] * self.prices[column]
        
        if position_values.empty:
            return {}
        
        # Calculate portfolio value
        portfolio_value = position_values.sum(axis=1)
        
        # Calculate position weights
        weights = position_values.div(portfolio_value, axis=0)
        
        # Calculate concentration metrics
        concentration = {}
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = (weights ** 2).sum(axis=1)
        concentration['hhi'] = hhi.mean()
        
        # Gini coefficient
        def gini(weights_row):
            weights_sorted = np.sort(weights_row.dropna().values)
            n = len(weights_sorted)
            if n <= 1:
                return 0
            cumulative_weights = np.cumsum(weights_sorted)
            return 1 - 2 * np.sum(cumulative_weights) / (n * np.sum(weights_sorted)) + 1 / n
        
        gini_values = weights.apply(gini, axis=1)
        concentration['gini'] = gini_values.mean()
        
        # Top holdings concentration
        for n in [1, 3, 5, 10]:
            if weights.shape[1] >= n:
                top_n_concentration = weights.apply(lambda x: x.nlargest(n).sum(), axis=1)
                concentration[f'top_{n}_concentration'] = top_n_concentration.mean()
        
        return concentration
    
    def analyze_tail_risk(self, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """
        Analyze tail risk using different methods.
        
        Args:
            confidence_levels: List of confidence levels (default: [0.95, 0.99])
            
        Returns:
            Dictionary of tail risk metrics
        """
        tail_risk = {}
        
        for confidence in confidence_levels:
            # Historical VaR
            tail_risk[f'historical_var_{int(confidence*100)}'] = self.risk_metrics.calculate_var(confidence, 'historical')
            
            # Parametric VaR
            tail_risk[f'parametric_var_{int(confidence*100)}'] = self.risk_metrics.calculate_var(confidence, 'parametric')
            
            # CVaR (Expected Shortfall)
            tail_risk[f'cvar_{int(confidence*100)}'] = self.risk_metrics.calculate_cvar(confidence)
        
        # Calculate tail risk ratios
        tail_risk['tail_ratio'] = self.risk_metrics.calculate_tail_ratio()
        
        return tail_risk
    
    def get_all_risk_metrics(self, risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Get all risk metrics.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.0)
            
        Returns:
            Dictionary of all risk metrics
        """
        all_metrics = {}
        
        # Standard risk metrics
        all_metrics.update(self.risk_metrics.calculate_all_metrics(risk_free_rate))
        
        # Returns distribution analysis
        all_metrics['distribution'] = self.analyze_returns_distribution()
        
        # Tail risk analysis
        all_metrics['tail_risk'] = self.analyze_tail_risk()
        
        # Concentration risk (if positions and prices provided)
        if self.positions is not None and self.prices is not None:
            all_metrics['concentration'] = self.analyze_concentration_risk()
        
        return all_metrics