from typing import Dict, List, Union, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings

class StatisticalAnalysis:
    """
    Statistical analysis tools for backtesting.
    """
    
    def __init__(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None):
        """
        Initialize statistical analysis.
        
        Args:
            returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns (optional)
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.logger = logging.getLogger(__name__)
        
        # Suppress warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def basic_statistics(self) -> Dict[str, float]:
        """
        Calculate basic statistics of returns.
        
        Returns:
            Dictionary of basic statistics
        """
        stats = {}
        
        # Return statistics
        stats['mean'] = self.returns.mean()
        stats['median'] = self.returns.median()
        stats['std'] = self.returns.std()
        stats['min'] = self.returns.min()
        stats['max'] = self.returns.max()
        stats['skew'] = self.returns.skew()
        stats['kurtosis'] = self.returns.kurtosis()
        
        # Annualized statistics (assuming daily returns)
        stats['annualized_return'] = (1 + self.returns.mean()) ** 252 - 1
        stats['annualized_volatility'] = self.returns.std() * np.sqrt(252)
        
        # Positive/negative days
        stats['positive_days'] = (self.returns > 0).sum() / len(self.returns)
        stats['negative_days'] = (self.returns < 0).sum() / len(self.returns)
        
        # Winning/losing streaks
        pos_streak = self.returns > 0
        neg_streak = self.returns < 0
        
        pos_streaks = []
        neg_streaks = []
        
        current_pos_streak = 0
        current_neg_streak = 0
        
        for is_pos, is_neg in zip(pos_streak, neg_streak):
            if is_pos:
                current_pos_streak += 1
                if current_neg_streak > 0:
                    neg_streaks.append(current_neg_streak)
                    current_neg_streak = 0
            elif is_neg:
                current_neg_streak += 1
                if current_pos_streak > 0:
                    pos_streaks.append(current_pos_streak)
                    current_pos_streak = 0
            else:  # Zero return
                if current_pos_streak > 0:
                    pos_streaks.append(current_pos_streak)
                    current_pos_streak = 0
                if current_neg_streak > 0:
                    neg_streaks.append(current_neg_streak)
                    current_neg_streak = 0
        
        # Add any remaining streaks
        if current_pos_streak > 0:
            pos_streaks.append(current_pos_streak)
        if current_neg_streak > 0:
            neg_streaks.append(current_neg_streak)
        
        stats['max_winning_streak'] = max(pos_streaks) if pos_streaks else 0
        stats['max_losing_streak'] = max(neg_streaks) if neg_streaks else 0
        stats['avg_winning_streak'] = np.mean(pos_streaks) if pos_streaks else 0
        stats['avg_losing_streak'] = np.mean(neg_streaks) if neg_streaks else 0
        
        return stats
    
    def normality_test(self) -> Dict[str, float]:
        """
        Test for normality of returns.
        
        Returns:
            Dictionary of normality test results
        """
        if len(self.returns) < 8:
            self.logger.warning("Not enough data for normality tests")
            return {}
        
        results = {}
        
        try:
            from scipy import stats as scipy_stats
            
            # Jarque-Bera test
            jb_stat, jb_pvalue = scipy_stats.jarque_bera(self.returns)
            results['jarque_bera_stat'] = jb_stat
            results['jarque_bera_pvalue'] = jb_pvalue
            results['jarque_bera_normal'] = jb_pvalue > 0.05
            
            # Shapiro-Wilk test
            if len(self.returns) < 5000:  # Shapiro-Wilk is only valid for n < 5000
                sw_stat, sw_pvalue = scipy_stats.shapiro(self.returns)
                results['shapiro_wilk_stat'] = sw_stat
                results['shapiro_wilk_pvalue'] = sw_pvalue
                results['shapiro_wilk_normal'] = sw_pvalue > 0.05
            
            # D'Agostino-Pearson test
            dp_stat, dp_pvalue = scipy_stats.normaltest(self.returns)
            results['dagostino_pearson_stat'] = dp_stat
            results['dagostino_pearson_pvalue'] = dp_pvalue
            results['dagostino_pearson_normal'] = dp_pvalue > 0.05
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = scipy_stats.kstest(
                self.returns, 
                'norm', 
                args=(self.returns.mean(), self.returns.std())
            )
            results['kolmogorov_smirnov_stat'] = ks_stat
            results['kolmogorov_smirnov_pvalue'] = ks_pvalue
            results['kolmogorov_smirnov_normal'] = ks_pvalue > 0.05
            
            # Overall normality assessment
            results['is_normal'] = (
                results.get('jarque_bera_normal', False) and
                results.get('shapiro_wilk_normal', False) and
                results.get('dagostino_pearson_normal', False) and
                results.get('kolmogorov_smirnov_normal', False)
            )
            
        except ImportError:
            self.logger.warning("SciPy not available for normality tests")
            results['error'] = "SciPy not available"
        
        return results
    
    def autocorrelation_analysis(self, lags: int = 10) -> Dict[str, Any]:
        """
        Analyze autocorrelation of returns.
        
        Args:
            lags: Number of lags to analyze (default: 10)
            
        Returns:
            Dictionary of autocorrelation analysis results
        """
        if len(self.returns) <= lags:
            self.logger.warning(f"Not enough data for {lags} lags")
            return {}
        
        results = {}
        
        # Calculate autocorrelation for each lag
        autocorr = {}
        for lag in range(1, lags + 1):
            autocorr[lag] = self.returns.autocorr(lag)
        
        results['autocorrelation'] = autocorr
        
        # Calculate Ljung-Box test for autocorrelation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            lb_stat, lb_pvalue = acorr_ljungbox(self.returns, lags=[lags])
            results['ljung_box_stat'] = lb_stat[0]
            results['ljung_box_pvalue'] = lb_pvalue[0]
            results['has_autocorrelation'] = lb_pvalue[0] < 0.05
            
        except ImportError:
            self.logger.warning("statsmodels not available for Ljung-Box test")
            results['error'] = "statsmodels not available"
        
        # Calculate runs test for randomness
        try:
            from statsmodels.stats.diagnostic import runs_test
            
            # Convert returns to binary sequence (1 for positive, 0 for negative)
            binary_returns = (self.returns > 0).astype(int)
            
            runs_stat, runs_pvalue = runs_test(binary_returns)
            results['runs_test_stat'] = runs_stat
            results['runs_test_pvalue'] = runs_pvalue
            results['is_random'] = runs_pvalue > 0.05
            
        except ImportError:
            self.logger.warning("statsmodels not available for runs test")
        
        return results
    
    def stationarity_test(self) -> Dict[str, Any]:
        """
        Test for stationarity of returns.
        
        Returns:
            Dictionary of stationarity test results
        """
        if len(self.returns) < 20:
            self.logger.warning("Not enough data for stationarity tests")
            return {}
        
        results = {}
        
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(self.returns.dropna())
            results['adf_stat'] = adf_result[0]
            results['adf_pvalue'] = adf_result[1]
            results['adf_stationary'] = adf_result[1] < 0.05
            
            # KPSS test
            kpss_result = kpss(self.returns.dropna())
            results['kpss_stat'] = kpss_result[0]
            results['kpss_pvalue'] = kpss_result[1]
            results['kpss_stationary'] = kpss_result[1] > 0.05
            
            # Overall stationarity assessment
            results['is_stationary'] = results['adf_stationary'] and results['kpss_stationary']
            
        except ImportError:
            self.logger.warning("statsmodels not available for stationarity tests")
            results['error'] = "statsmodels not available"
        
        return results
    
    def seasonality_analysis(self) -> Dict[str, Any]:
        """
        Analyze seasonality in returns.
        
        Returns:
            Dictionary of seasonality analysis results
        """
        if len(self.returns) < 30:
            self.logger.warning("Not enough data for seasonality analysis")
            return {}
        
        results = {}
        
        # Day of week analysis
        if isinstance(self.returns.index, pd.DatetimeIndex):
            day_of_week = self.returns.groupby(self.returns.index.dayofweek).mean()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_of_week.index = [day_names[i] for i in day_of_week.index if i < len(day_names)]
            results['day_of_week'] = day_of_week.to_dict()
            
            # Month analysis
            month_returns = self.returns.groupby(self.returns.index.month).mean()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_returns.index = [month_names[i-1] for i in month_returns.index if 0 < i <= 12]
            results['month'] = month_returns.to_dict()
            
            # Quarter analysis
            quarter_returns = self.returns.groupby(self.returns.index.quarter).mean()
            quarter_returns.index = [f'Q{i}' for i in quarter_returns.index]
            results['quarter'] = quarter_returns.to_dict()
            
            # Year analysis
            year_returns = self.returns.groupby(self.returns.index.year).mean()
            results['year'] = year_returns.to_dict()
        
        # Test for seasonality significance
        try:
            from scipy import stats as scipy_stats
            
            # Test if day of week returns are significantly different
            if 'day_of_week' in results and len(set(self.returns.index.dayofweek)) > 1:
                day_groups = [self.returns[self.returns.index.dayofweek == i] for i in range(5)]
                day_groups = [g for g in day_groups if len(g) > 0]
                
                if len(day_groups) > 1:
                    f_stat, p_value = scipy_stats.f_oneway(*day_groups)
                    results['day_of_week_f_stat'] = f_stat
                    results['day_of_week_p_value'] = p_value
                    results['day_of_week_significant'] = p_value < 0.05
            
            # Test if month returns are significantly different
            if 'month' in results and len(set(self.returns.index.month)) > 1:
                month_groups = [self.returns[self.returns.index.month == i] for i in range(1, 13)]
                month_groups = [g for g in month_groups if len(g) > 0]
                
                if len(month_groups) > 1:
                    f_stat, p_value = scipy_stats.f_oneway(*month_groups)
                    results['month_f_stat'] = f_stat
                    results['month_p_value'] = p_value
                    results['month_significant'] = p_value < 0.05
            
        except ImportError:
            self.logger.warning("SciPy not available for seasonality significance tests")
        
        return results
    
    def correlation_analysis(self, other_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Analyze correlation with other return series.
        
        Args:
            other_returns: Dictionary of other return series (name -> returns)
            
        Returns:
            Correlation matrix
        """
        if not other_returns:
            return pd.DataFrame()
        
        # Create DataFrame with all return series
        all_returns = {'Strategy': self.returns}
        
        if self.benchmark_returns is not None:
            all_returns['Benchmark'] = self.benchmark_returns
        
        all_returns.update(other_returns)
        
        # Create DataFrame and calculate correlation
        returns_df = pd.DataFrame(all_returns)
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def regression_analysis(self, factors: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Perform regression analysis against factors.
        
        Args:
            factors: Dictionary of factor returns (factor_name -> returns)
            
        Returns:
            Dictionary of regression results
        """
        if not factors:
            return {}
        
        results = {}
        
        try:
            import statsmodels.api as sm
            
            # Prepare data
            Y = self.returns
            
            # Create DataFrame with all factors
            X_data = pd.DataFrame(factors)
            
            # Align data
            aligned_data = pd.concat([Y, X_data], axis=1).dropna()
            
            if len(aligned_data) < len(factors) + 2:
                self.logger.warning("Not enough data for regression analysis")
                return {}
            
            Y = aligned_data.iloc[:, 0]
            X = aligned_data.iloc[:, 1:]
            
            # Add constant
            X = sm.add_constant(X)
            
            # Fit model
            model = sm.OLS(Y, X).fit()
            
            # Extract results
            results['summary'] = model.summary().as_text()
            results['r_squared'] = model.rsquared
            results['adj_r_squared'] = model.rsquared_adj
            results['f_stat'] = model.fvalue
            results['f_pvalue'] = model.f_pvalue
            results['aic'] = model.aic
            results['bic'] = model.bic
            
            # Coefficients
            results['coefficients'] = {}
            for i, name in enumerate(['const'] + list(factors.keys())):
                results['coefficients'][name] = {
                    'value': model.params[i],
                    'std_err': model.bse[i],
                    't_stat': model.tvalues[i],
                    'p_value': model.pvalues[i],
                    'significant': model.pvalues[i] < 0.05
                }
            
            # Residual analysis
            results['residuals'] = {
                'mean': model.resid.mean(),
                'std': model.resid.std(),
                'skew': model.resid.skew(),
                'kurtosis': model.resid.kurtosis()
            }
            
            # Test for heteroskedasticity
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                
                bp_test = het_breuschpagan(model.resid, model.model.exog)
                results['heteroskedasticity'] = {
                    'bp_stat': bp_test[0],
                    'bp_pvalue': bp_test[1],
                    'has_heteroskedasticity': bp_test[1] < 0.05
                }
            except:
                self.logger.warning("Error in heteroskedasticity test")
            
            # Test for autocorrelation in residuals
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                
                lb_test = acorr_ljungbox(model.resid, lags=[10])
                results['residual_autocorrelation'] = {
                    'lb_stat': lb_test[0][0],
                    'lb_pvalue': lb_test[1][0],
                    'has_autocorrelation': lb_test[1][0] < 0.05
                }
            except:
                self.logger.warning("Error in residual autocorrelation test")
            
        except ImportError:
            self.logger.warning("statsmodels not available for regression analysis")
            results['error'] = "statsmodels not available"
        
        return results
    
    def factor_analysis(self, factors: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Perform factor analysis.
        
        Args:
            factors: Dictionary of factor returns (factor_name -> returns)
            
        Returns:
            Dictionary of factor analysis results
        """
        if not factors:
            return {}
        
        results = {}
        
        # Regression analysis
        regression_results = self.regression_analysis(factors)
        
        if 'coefficients' in regression_results:
            results['factor_exposures'] = {
                name: coef['value'] 
                for name, coef in regression_results['coefficients'].items()
                if name != 'const'
            }
            
            results['factor_significance'] = {
                name: coef['significant'] 
                for name, coef in regression_results['coefficients'].items()
                if name != 'const'
            }
            
            results['alpha'] = regression_results['coefficients'].get('const', {}).get('value', 0)
            results['alpha_significant'] = regression_results['coefficients'].get('const', {}).get('significant', False)
            results['r_squared'] = regression_results.get('r_squared', 0)
        
        # Calculate factor contribution
        if 'factor_exposures' in results:
            # Align data
            factor_data = pd.DataFrame(factors)
            aligned_data = pd.concat([self.returns, factor_data], axis=1).dropna()
            
            if not aligned_data.empty:
                strategy_returns = aligned_data.iloc[:, 0]
                factor_returns = aligned_data.iloc[:, 1:]
                
                # Calculate factor contribution
                factor_contribution = {}
                
                for factor_name in factors.keys():
                    if factor_name in results['factor_exposures']:
                        exposure = results['factor_exposures'][factor_name]
                        factor_return = factor_returns[factor_name].mean()
                        contribution = exposure * factor_return
                        factor_contribution[factor_name] = contribution
                
                results['factor_contribution'] = factor_contribution
                
                # Calculate explained vs. unexplained return
                total_return = strategy_returns.mean()
                explained_return = sum(factor_contribution.values())
                unexplained_return = total_return - explained_return
                
                results['total_return'] = total_return
                results['explained_return'] = explained_return
                results['unexplained_return'] = unexplained_return
                results['explained_pct'] = explained_return / total_return if total_return != 0 else 0
        
        return results
    
    def regime_analysis(self, regime_indicator: pd.Series, regime_names: Optional[Dict[Any, str]] = None) -> Dict[str, Any]:
        """
        Analyze performance across different regimes.
        
        Args:
            regime_indicator: Series indicating the regime for each date
            regime_names: Dictionary mapping regime values to names (optional)
            
        Returns:
            Dictionary of regime analysis results
        """
        if regime_indicator is None or len(regime_indicator) == 0:
            return {}
        
        results = {}
        
        # Align data
        aligned_data = pd.concat([self.returns, regime_indicator], axis=1).dropna()
        
        if aligned_data.empty:
            return {}
        
        strategy_returns = aligned_data.iloc[:, 0]
        regimes = aligned_data.iloc[:, 1]
        
        # Get unique regimes
        unique_regimes = regimes.unique()
        
        # Create regime names if not provided
        if regime_names is None:
            regime_names = {regime: f"Regime {i+1}" for i, regime in enumerate(unique_regimes)}
        
        # Analyze performance in each regime
        regime_performance = {}
        
        for regime in unique_regimes:
            regime_name = regime_names.get(regime, str(regime))
            regime_returns = strategy_returns[regimes == regime]
            
            if len(regime_returns) > 0:
                regime_performance[regime_name] = {
                    'count': len(regime_returns),
                    'mean_return': regime_returns.mean(),
                    'std_dev': regime_returns.std(),
                    'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'cumulative_return': (1 + regime_returns).prod() - 1,
                    'win_rate': (regime_returns > 0).mean(),
                    'max_return': regime_returns.max(),
                    'min_return': regime_returns.min()
                }
        
        results['regime_performance'] = regime_performance
        
        # Calculate regime transition matrix
        if len(unique_regimes) > 1:
            transitions = {}
            
            for regime in unique_regimes:
                regime_name = regime_names.get(regime, str(regime))
                transitions[regime_name] = {}
                
                # Find indices where this regime occurs
                regime_indices = regimes[regimes == regime].index
                
                for next_regime in unique_regimes:
                    next_regime_name = regime_names.get(next_regime, str(next_regime))
                    
                    # Count transitions to next regime
                    transition_count = 0
                    
                    for idx in regime_indices:
                        if idx in regimes.index:
                            idx_pos = regimes.index.get_loc(idx)
                            if idx_pos + 1 < len(regimes.index):
                                next_idx = regimes.index[idx_pos + 1]
                                if next_idx in regimes.index and regimes.loc[next_idx] == next_regime:
                                    transition_count += 1
                    
                    # Calculate transition probability
                    if len(regime_indices) > 0:
                        transitions[regime_name][next_regime_name] = transition_count / len(regime_indices)
                    else:
                        transitions[regime_name][next_regime_name] = 0
            
            results['regime_transitions'] = transitions
        
        return results
    
    def get_all_statistics(self, factors: Optional[Dict[str, pd.Series]] = None) -> Dict[str, Any]:
        """
        Get all statistical analysis results.
        
        Args:
            factors: Dictionary of factor returns for factor analysis (optional)
            
        Returns:
            Dictionary of all statistical analysis results
        """
        all_stats = {}
        
        # Basic statistics
        all_stats['basic'] = self.basic_statistics()
        
        # Normality test
        all_stats['normality'] = self.normality_test()
        
        # Autocorrelation analysis
        all_stats['autocorrelation'] = self.autocorrelation_analysis()
        
        # Stationarity test
        all_stats['stationarity'] = self.stationarity_test()
        
        # Seasonality analysis
        all_stats['seasonality'] = self.seasonality_analysis()
        
        # Factor analysis (if factors provided)
        if factors is not None:
            all_stats['factor_analysis'] = self.factor_analysis(factors)
        
        return all_stats