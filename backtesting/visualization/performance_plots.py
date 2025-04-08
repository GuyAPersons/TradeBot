from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import calendar
import math

class PerformancePlots:
    """
    Comprehensive performance visualization tools for trading strategies.
    """
    
    def __init__(self, returns: pd.Series = None, equity_curve: pd.Series = None, 
                benchmark_returns: pd.Series = None, trades: pd.DataFrame = None):
        """
        Initialize performance plots.
        
        Args:
            returns: Series of strategy returns (optional)
            equity_curve: Series of equity curve values (optional)
            benchmark_returns: Series of benchmark returns (optional)
            trades: DataFrame of trade information (optional)
        """
        self.returns = returns
        self.equity_curve = equity_curve
        self.benchmark_returns = benchmark_returns
        self.trades = trades
        self.logger = logging.getLogger(__name__)
        
        # Set default style
        self.set_style()
    
    def set_style(self, style: str = 'whitegrid', context: str = 'notebook', 
                 palette: str = 'deep', font_scale: float = 1.2):
        """
        Set the visualization style.
        
        Args:
            style: Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
            context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
            palette: Color palette
            font_scale: Font scale factor
        """
        try:
            sns.set_style(style)
            sns.set_context(context, font_scale=font_scale)
            sns.set_palette(palette)
        except Exception as e:
            self.logger.error(f"Error setting style: {str(e)}")
    
    def _format_pct(self, x: float, pos: int = None) -> str:
        """Format as percentage."""
        return f'{100 * x:.1f}%'
    
    def _format_currency(self, x: float, pos: int = None, currency: str = '$') -> str:
        """Format as currency."""
        if abs(x) >= 1e6:
            return f'{currency}{x/1e6:.1f}M'
        elif abs(x) >= 1e3:
            return f'{currency}{x/1e3:.1f}K'
        else:
            return f'{currency}{x:.2f}'
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (12, 6), 
                         title: str = 'Equity Curve', 
                         log_scale: bool = False,
                         include_benchmark: bool = True,
                         include_drawdowns: bool = True,
                         currency: str = '$') -> None:
        """
        Plot equity curve.
        
        Args:
            figsize: Figure size (width, height)
            title: Plot title
            log_scale: Whether to use log scale for y-axis
            include_benchmark: Whether to include benchmark
            include_drawdowns: Whether to highlight drawdowns
            currency: Currency symbol for formatting
        """
        if self.equity_curve is None:
            self.logger.error("Equity curve data not provided")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot equity curve
        ax.plot(self.equity_curve.index, self.equity_curve.values, 
               label='Strategy', linewidth=2, color='#1f77b4')
        
        # Plot benchmark if available
        if include_benchmark and self.benchmark_returns is not None:
            # Convert benchmark returns to equity curve
            benchmark_equity = (1 + self.benchmark_returns).cumprod()
            
            # Scale benchmark to start at the same value as strategy
            benchmark_equity = benchmark_equity * self.equity_curve.iloc[0] / benchmark_equity.iloc[0]
            
            ax.plot(benchmark_equity.index, benchmark_equity.values, 
                   label='Benchmark', linewidth=1.5, linestyle='--', color='#ff7f0e')
        
        # Highlight drawdowns if requested
        if include_drawdowns:
            # Calculate running maximum
            running_max = self.equity_curve.cummax()
            
            # Calculate drawdowns
            drawdowns = (self.equity_curve / running_max - 1)
            
            # Find drawdown periods (drawdown > 5%)
            threshold = -0.05  # 5% drawdown
            is_drawdown = drawdowns < threshold
            
            # Find start and end of drawdown periods
            drawdown_starts = is_drawdown[is_drawdown].index
            
            if not drawdown_starts.empty:
                # Group consecutive dates
                groups = []
                current_group = [drawdown_starts[0]]
                
                for i in range(1, len(drawdown_starts)):
                    if (drawdown_starts[i] - drawdown_starts[i-1]).days <= 7:  # Within a week
                        current_group.append(drawdown_starts[i])
                    else:
                        groups.append(current_group)
                        current_group = [drawdown_starts[i]]
                
                if current_group:
                    groups.append(current_group)
                
                # Highlight drawdown periods
                for group in groups:
                    start_date = group[0]
                    end_date = group[-1]
                    
                    # Find drawdown depth
                    depth = drawdowns.loc[start_date:end_date].min()
                    
                    ax.axvspan(start_date, end_date, alpha=0.2, color='red')
                    
                    # Add annotation for significant drawdowns
                    if depth < -0.1:  # More than 10% drawdown
                        mid_date = start_date + (end_date - start_date) / 2
                        ax.annotate(f'{depth:.1%}', 
                                   xy=(mid_date, self.equity_curve.loc[mid_date]),
                                   xytext=(0, -30),
                                   textcoords='offset points',
                                   arrowprops=dict(arrowstyle='->', color='black'),
                                   ha='center')
        
        # Set log scale if requested
        if log_scale:
            ax.set_yscale('log')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: self._format_currency(x, pos, currency)))
        
        # Set labels and title
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'Equity ({currency})', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, figsize: Tuple[int, int] = (12, 6),
                                 bins: int = 50,
                                 include_benchmark: bool = True,
                                 include_normal: bool = True) -> None:
        """
        Plot distribution of returns.
        
        Args:
            figsize: Figure size (width, height)
            bins: Number of histogram bins
            include_benchmark: Whether to include benchmark
            include_normal: Whether to include normal distribution
        """
        if self.returns is None:
            self.logger.error("Returns data not provided")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot strategy returns distribution
        sns.histplot(self.returns, bins=bins, kde=True, 
                    label='Strategy Returns', color='#1f77b4', alpha=0.7, ax=ax)
        
        # Plot benchmark returns distribution if available
        if include_benchmark and self.benchmark_returns is not None:
            sns.histplot(self.benchmark_returns, bins=bins, kde=True, 
                        label='Benchmark Returns', color='#ff7f0e', alpha=0.5, ax=ax)
        
        # Plot normal distribution if requested
        if include_normal:
            x = np.linspace(self.returns.min(), self.returns.max(), 1000)
            mean = self.returns.mean()
            std = self.returns.std()
            
            normal_dist = np.exp(-(x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
            normal_dist = normal_dist * (self.returns.count() * (self.returns.max() - self.returns.min()) / bins)
            
            ax.plot(x, normal_dist, label='Normal Distribution', 
                   linestyle='--', color='green', linewidth=2)
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(FuncFormatter(self._format_pct))
        
        # Set labels and title
        ax.set_title('Returns Distribution', fontsize=14)
        ax.set_xlabel('Return', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add vertical lines for mean and standard deviations
        mean = self.returns.mean()
        std = self.returns.std()
        
        ax.axvline(x=mean, color='red', linestyle='--', alpha=0.8, 
                  label=f'Mean: {mean:.2%}')
        
        ax.axvline(x=mean + std, color='purple', linestyle=':', alpha=0.6, 
                  label=f'+1 Std Dev: {(mean + std):.2%}')
        
        ax.axvline(x=mean - std, color='purple', linestyle=':', alpha=0.6, 
                  label=f'-1 Std Dev: {(mean - std):.2%}')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_statistics(self, window: int = 60, 
                               figsize: Tuple[int, int] = (15, 10),
                               include_benchmark: bool = True) -> None:
        """
        Plot rolling statistics.
        
        Args:
            window: Rolling window size
            figsize: Figure size (width, height)
            include_benchmark: Whether to include benchmark
        """
        if self.returns is None:
            self.logger.error("Returns data not provided")
            return
        
        # Calculate rolling statistics
        rolling_mean = self.returns.rolling(window=window).mean()
        rolling_std = self.returns.rolling(window=window).std()
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)  # Annualized
        
        if include_benchmark and self.benchmark_returns is not None:
            bench_rolling_mean = self.benchmark_returns.rolling(window=window).mean()
            bench_rolling_std = self.benchmark_returns.rolling(window=window).std()
            bench_rolling_sharpe = bench_rolling_mean / bench_rolling_std * np.sqrt(252)
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot rolling mean
        axes[0].plot(rolling_mean.index, rolling_mean.values, 
                    label=f'Strategy ({window}-day Rolling Mean)', 
                    color='#1f77b4', linewidth=2)
        
        if include_benchmark and self.benchmark_returns is not None:
            axes[0].plot(bench_rolling_mean.index, bench_rolling_mean.values, 
                        label=f'Benchmark ({window}-day Rolling Mean)', 
                        color='#ff7f0e', linewidth=1.5, linestyle='--')
        
        axes[0].set_title(f'{window}-day Rolling Statistics', fontsize=14)
        axes[0].set_ylabel('Mean Return', fontsize=12)
        axes[0].yaxis.set_major_formatter(FuncFormatter(self._format_pct))
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        
        # Plot rolling standard deviation
        axes[1].plot(rolling_std.index, rolling_std.values, 
                    label=f'Strategy ({window}-day Rolling Std Dev)', 
                    color='#1f77b4', linewidth=2)
        
        if include_benchmark and self.benchmark_returns is not None:
            axes[1].plot(bench_rolling_std.index, bench_rolling_std.values, 
                        label=f'Benchmark ({window}-day Rolling Std Dev)', 
                        color='#ff7f0e', linewidth=1.5, linestyle='--')
        
        axes[1].set_ylabel('Std Dev', fontsize=12)
        axes[1].yaxis.set_major_formatter(FuncFormatter(self._format_pct))
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        
        # Plot rolling Sharpe ratio
        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, 
                    label=f'Strategy ({window}-day Rolling Sharpe)', 
                    color='#1f77b4', linewidth=2)
        
        if include_benchmark and self.benchmark_returns is not None:
            axes[2].plot(bench_rolling_sharpe.index, bench_rolling_sharpe.values, 
                        label=f'Benchmark ({window}-day Rolling Sharpe)', 
                        color='#ff7f0e', linewidth=1.5, linestyle='--')
        
        axes[2].set_ylabel('Sharpe Ratio', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=10)
        axes[2].set_xlabel('Date', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self, top_n: int = 5, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot drawdown chart.
        
        Args:
            top_n: Number of worst drawdowns to highlight
            figsize: Figure size (width, height)
        """
        if self.equity_curve is None:
            self.logger.error("Equity curve data not provided")
            return
        
        # Calculate running maximum
        running_max = self.equity_curve.cummax()
        
        # Calculate drawdowns
        drawdowns = (self.equity_curve / running_max - 1) * 100  # Convert to percentage
        
        # Find top N drawdowns
        drawdown_periods = []
        remaining_drawdowns = drawdowns.copy()
        
        for i in range(top_n):
            if remaining_drawdowns.min() >= 0:
                break
                
            # Find the worst drawdown
            worst_idx = remaining_drawdowns.idxmin()
            worst_drawdown = remaining_drawdowns.loc[worst_idx]
            
            # Find the start of this drawdown (last peak)
            peak_idx = remaining_drawdowns[:worst_idx].iloc[::-1].idxmax()
            
            # Find the end of this drawdown (recovery)
            try:
                recovery_idx = remaining_drawdowns[worst_idx:].loc[remaining_drawdowns[worst_idx:] >= 0].index[0]
            except IndexError:
                # No recovery yet
                recovery_idx = remaining_drawdowns.index[-1]
            
            # Store drawdown period
            drawdown_periods.append({
                'start': peak_idx,
                'bottom': worst_idx,
                'end': recovery_idx,
                'depth': worst_drawdown
            })
            
            # Remove this drawdown period from consideration
            mask = (remaining_drawdowns.index >= peak_idx) & (remaining_drawdowns.index <= recovery_idx)
            remaining_drawdowns.loc[mask] = 0
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(self.equity_curve.index, self.equity_curve.values, 
               label='Equity Curve', linewidth=2, color='#1f77b4')
        
        # Highlight drawdown periods
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(drawdown_periods)))
        
        for i, period in enumerate(drawdown_periods):
            # Highlight drawdown period
            ax1.axvspan(period['start'], period['end'], alpha=0.2, color=colors[i])
            
            # Mark peak and bottom
            ax1.scatter(period['start'], self.equity_curve.loc[period['start']], 
                       color=colors[i], s=100, marker='^', zorder=5)
            
            ax1.scatter(period['bottom'], self.equity_curve.loc[period['bottom']], 
                       color=colors[i], s=100, marker='v', zorder=5)
            
            # Add annotation
            mid_date = period['start'] + (period['bottom'] - period['start']) / 2
            ax1.annotate(f"#{i+1}: {period['depth']:.1f}%", 
                        xy=(mid_date, self.equity_curve.loc[mid_date]),
                        xytext=(0, 30),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color=colors[i]),
                        ha='center', color=colors[i], fontweight='bold')
        
        ax1.set_title('Equity Curve with Top Drawdowns', fontsize=14)
        ax1.set_ylabel('Equity', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdowns
        ax2.fill_between(drawdowns.index, drawdowns.values, 0, 
                        color='red', alpha=0.5, label='Drawdowns')
        
        # Add horizontal lines at 5%, 10%, 20% drawdowns
        for level in [-5, -10, -20]:
            ax2.axhline(y=level, color='gray', linestyle='--', alpha=0.7)
            ax2.text(drawdowns.index[0], level, f'{level}%', va='center', ha='left', fontsize=8)
        
        ax2.set_title('Drawdowns', fontsize=14)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Set y-axis limits for drawdowns
        min_drawdown = min(drawdowns.min(), -25)  # At least -25%
        ax2.set_ylim([min_drawdown, 5])
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns_heatmap(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot monthly returns heatmap.
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.returns is None:
            self.logger.error("Returns data not provided")
            return
        
        # Resample returns to monthly
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table with years as rows and months as columns
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        # Replace column numbers with month names
        month_names = {i: calendar.month_abbr[i] for i in range(1, 13)}
        monthly_pivot.columns = [month_names[col] for col in monthly_pivot.columns]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create heatmap
        ax = sns.heatmap(monthly_pivot * 100, annot=True, fmt=".1f", 
                        cmap="RdYlGn", center=0, linewidths=1, 
                        cbar_kws={'label': 'Monthly Return (%)'})
        
        # Set labels and title
        ax.set_title('Monthly Returns (%)', fontsize=14)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_annual_returns(self, figsize: Tuple[int, int] = (12, 6),
                           include_benchmark: bool = True) -> None:
        """
        Plot annual returns.
        
        Args:
            figsize: Figure size (width, height)
            include_benchmark: Whether to include benchmark
        """
        if self.returns is None:
            self.logger.error("Returns data not provided")
            return
        
        # Calculate annual returns
        annual_returns = self.returns.resample('A').apply(lambda x: (1 + x).prod() - 1)
        annual_returns.index = annual_returns.index.year
        
        if include_benchmark and self.benchmark_returns is not None:
            benchmark_annual = self.benchmark_returns.resample('A').apply(lambda x: (1 + x).prod() - 1)
            benchmark_annual.index = benchmark_annual.index.year
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot annual returns
        width = 0.35
        x = np.arange(len(annual_returns))
        
        ax.bar(x - width/2 if include_benchmark and self.benchmark_returns is not None else x, 
              annual_returns * 100, width, label='Strategy', color='#1f77b4')
        
        if include_benchmark and self.benchmark_returns is not None:
            # Align benchmark years with strategy years
            aligned_benchmark = pd.Series(index=annual_returns.index)
            for year in annual_returns.index:
                if year in benchmark_annual.index:
                    aligned_benchmark[year] = benchmark_annual[year]
            
            ax.bar(x + width/2, aligned_benchmark * 100, width, label='Benchmark', color='#ff7f0e')
        
        # Set labels and title
        ax.set_title('Annual Returns', fontsize=14)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        
        # Set x-axis ticks
        ax.set_xticks(x)
        ax.set_xticklabels(annual_returns.index)
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Add value labels on bars
        for i, v in enumerate(annual_returns):
            ax.text(i - width/2 if include_benchmark and self.benchmark_returns is not None else i, 
                   v * 100 + (5 if v >= 0 else -10), 
                   f'{v:.1%}', ha='center', fontsize=10)
        
        if include_benchmark and self.benchmark_returns is not None:
            for i, v in enumerate(aligned_benchmark.dropna()):
                ax.text(i + width/2, 
                       v * 100 + (5 if v >= 0 else -10), 
                       f'{v:.1%}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_return_quantiles(self, figsize: Tuple[int, int] = (12, 6),
                             include_benchmark: bool = True) -> None:
        """
        Plot return quantiles.
        
        Args:
            figsize: Figure size (width, height)
            include_benchmark: Whether to include benchmark
        """
        if self.returns is None:
            self.logger.error("Returns data not provided")
            return
        
        # Calculate return quantiles for different periods
        periods = {
            'Daily': self.returns,
            'Weekly': self.returns.resample('W').apply(lambda x: (1 + x).prod() - 1),
            'Monthly': self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
            'Quarterly': self.returns.resample('Q').apply(lambda x: (1 + x).prod() - 1),
            'Annual': self.returns.resample('A').apply(lambda x: (1 + x).prod() - 1)
        }
        
        if include_benchmark and self.benchmark_returns is not None:
            benchmark_periods = {
                'Daily': self.benchmark_returns,
                'Weekly': self.benchmark_returns.resample('W').apply(lambda x: (1 + x).prod() - 1),
                'Monthly': self.benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
                'Quarterly': self.benchmark_returns.resample('Q').apply(lambda x: (1 + x).prod() - 1),
                'Annual': self.benchmark_returns.resample('A').apply(lambda x: (1 + x).prod() - 1)
            }
        
        # Calculate quantiles
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up positions for bars
        x = np.arange(len(periods))
        width = 0.35
        
        # Plot strategy quantiles
        strategy_data = []
        for period_name, period_returns in periods.items():
            period_quantiles = period_returns.quantile(quantiles)
            strategy_data.append(period_quantiles)
        
        strategy_df = pd.DataFrame(strategy_data, index=periods.keys())
        
        # Plot benchmark quantiles if available
        if include_benchmark and self.benchmark_returns is not None:
            benchmark_data = []
            for period_name, period_returns in benchmark_periods.items():
                period_quantiles = period_returns.quantile(quantiles)
                benchmark_data.append(period_quantiles)
            
            benchmark_df = pd.DataFrame(benchmark_data, index=benchmark_periods.keys())
        
        # Plot boxplots
        positions = np.arange(len(periods))
        
        # Create box plot data
        box_data = []
        labels = []
        
        for i, period_name in enumerate(periods.keys()):
            box_data.append(periods[period_name])
            labels.append(f'Strategy\n{period_name}')
            
            if include_benchmark and self.benchmark_returns is not None:
                box_data.append(benchmark_periods[period_name])
                labels.append(f'Benchmark\n{period_name}')
        
        # Plot boxplots
        bp = ax.boxplot(box_data, patch_artist=True, 
                       positions=np.arange(len(box_data)), 
                       widths=0.6)
        
        # Customize boxplot colors
        for i, box in enumerate(bp['boxes']):
            if i % 2 == 0:
                box.set(facecolor='#1f77b4', alpha=0.7)
            else:
                box.set(facecolor='#ff7f0e', alpha=0.7)
        
        # Set labels and title
        ax.set_title('Return Quantiles by Time Period', fontsize=14)
        ax.set_ylabel('Return', fontsize=12)
        
        # Set x-axis ticks
        ax.set_xticks(np.arange(len(box_data)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(self._format_pct))
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_beta(self, window: int = 60, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot rolling beta to benchmark.
        
        Args:
            window: Rolling window size
            figsize: Figure size (width, height)
        """
        if self.returns is None or self.benchmark_returns is None:
            self.logger.error("Returns and benchmark data required")
            return
        
        # Align returns and benchmark
        aligned_returns = pd.DataFrame({
            'strategy': self.returns,
            'benchmark': self.benchmark_returns
        }).dropna()
        
        # Calculate rolling beta
        rolling_cov = aligned_returns['strategy'].rolling(window=window).cov(aligned_returns['benchmark'])
        rolling_var = aligned_returns['benchmark'].rolling(window=window).var()
        rolling_beta = rolling_cov / rolling_var
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot rolling beta
        ax.plot(rolling_beta.index, rolling_beta.values, 
               label=f'{window}-day Rolling Beta', 
               color='#1f77b4', linewidth=2)
        
        # Add horizontal line at beta = 1
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, 
                  label='Beta = 1 (Market)')
        
        # Add horizontal line at beta = 0
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.7,
                  label='Beta = 0 (Market Neutral)')
        
        # Set labels and title
        ax.set_title(f'{window}-day Rolling Beta to Benchmark', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Beta', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_underwater(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot underwater chart (drawdowns over time).
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.equity_curve is None:
            self.logger.error("Equity curve data not provided")
            return
        
        # Calculate running maximum
        running_max = self.equity_curve.cummax()
        
        # Calculate drawdowns
        drawdowns = (self.equity_curve / running_max - 1) * 100  # Convert to percentage
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot underwater chart
        ax.fill_between(drawdowns.index, drawdowns.values, 0, 
                       color='red', alpha=0.5, label='Drawdown')
        
        # Add horizontal lines at 5%, 10%, 20% drawdowns
        for level in [-5, -10, -20]:
            ax.axhline(y=level, color='gray', linestyle='--', alpha=0.7)
            ax.text(drawdowns.index[0], level, f'{level}%', va='center', ha='left', fontsize=8)
        
        # Set labels and title
        ax.set_title('Underwater Chart (Drawdowns)', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        
        # Set y-axis limits
        min_drawdown = min(drawdowns.min(), -25)  # At least -25%
        ax.set_ylim([min_drawdown, 5])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot trade analysis charts.
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.trades is None or len(self.trades) == 0:
            self.logger.error("Trade data not provided or empty")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Profit/Loss Distribution
        sns.histplot(self.trades['pnl'], bins=50, kde=True, ax=axes[0, 0], color='#1f77b4')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].axvline(x=self.trades['pnl'].mean(), color='green', linestyle='-', alpha=0.7,
                          label=f'Mean: {self.trades["pnl"].mean():.2f}')
        axes[0, 0].set_title('Trade P&L Distribution', fontsize=12)
        axes[0, 0].set_xlabel('Profit/Loss', fontsize=10)
        axes[0, 0].set_ylabel('Frequency', fontsize=10)
        axes[0, 0].legend()
        
        # 2. Win/Loss by Trade Duration
        if 'duration' in self.trades.columns:
            # Create scatter plot of P&L vs duration
            scatter = axes[0, 1].scatter(self.trades['duration'], self.trades['pnl'], 
                                        c=self.trades['pnl'] > 0, cmap='RdYlGn', alpha=0.7)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0, 1].set_title('P&L vs Trade Duration', fontsize=12)
            axes[0, 1].set_xlabel('Duration (days)', fontsize=10)
            axes[0, 1].set_ylabel('Profit/Loss', fontsize=10)
            
            # Add legend
            legend1 = axes[0, 1].legend(*scatter.legend_elements(),
                                       loc="upper right", title="Outcome")
            axes[0, 1].add_artist(legend1)
            
            # Add trend line
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(self.trades['duration'], self.trades['pnl'])
                x = np.array([self.trades['duration'].min(), self.trades['duration'].max()])
                axes[0, 1].plot(x, intercept + slope * x, 'r', 
                               label=f'Trend: y={slope:.4f}x+{intercept:.2f}')
                axes[0, 1].legend()
            except ImportError:
                pass
        else:
            axes[0, 1].text(0.5, 0.5, 'Duration data not available', 
                           ha='center', va='center', fontsize=12)
        
        # 3. Cumulative P&L
        cumulative_pnl = self.trades['pnl'].cumsum()
        axes[1, 0].plot(range(len(cumulative_pnl)), cumulative_pnl, 
                       color='#1f77b4', linewidth=2)
        axes[1, 0].set_title('Cumulative P&L', fontsize=12)
        axes[1, 0].set_xlabel('Trade Number', fontsize=10)
        axes[1, 0].set_ylabel('Cumulative P&L', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Win/Loss Ratio by Month/Year
        if 'entry_time' in self.trades.columns:
            # Add month and year columns
            trades_with_time = self.trades.copy()
            trades_with_time['month'] = trades_with_time['entry_time'].dt.month
            trades_with_time['year'] = trades_with_time['entry_time'].dt.year
            trades_with_time['month_year'] = trades_with_time['entry_time'].dt.strftime('%Y-%m')
            
            # Calculate win ratio by month-year
            win_ratio = trades_with_time.groupby('month_year').apply(
                lambda x: (x['pnl'] > 0).mean() if len(x) > 0 else 0
            ).reset_index()
            
            # Sort by month-year
            win_ratio = win_ratio.sort_values('month_year')
            
            # Plot win ratio
            axes[1, 1].bar(win_ratio['month_year'], win_ratio[0] * 100, color='#1f77b4')
            axes[1, 1].axhline(y=50, color='red', linestyle='--', alpha=0.7, 
                              label='50% Win Rate')
            axes[1, 1].set_title('Win Rate by Month', fontsize=12)
            axes[1, 1].set_xlabel('Month', fontsize=10)
            axes[1, 1].set_ylabel('Win Rate (%)', fontsize=10)
            axes[1, 1].set_ylim([0, 100])
            axes[1, 1].legend()
            
            # Rotate x-axis labels
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=90)
        else:
            axes[1, 1].text(0.5, 0.5, 'Entry time data not available', 
                           ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_size_analysis(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot trade size analysis.
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.trades is None or len(self.trades) == 0:
            self.logger.error("Trade data not provided or empty")
            return
        
        if 'size' not in self.trades.columns:
            self.logger.error("Trade size data not available")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Trade Size Distribution
        sns.histplot(self.trades['size'], bins=30, kde=True, ax=axes[0, 0], color='#1f77b4')
        axes[0, 0].axvline(x=self.trades['size'].mean(), color='red', linestyle='--', alpha=0.7,
                          label=f'Mean: {self.trades["size"].mean():.2f}')
        axes[0, 0].set_title('Trade Size Distribution', fontsize=12)
        axes[0, 0].set_xlabel('Trade Size', fontsize=10)
        axes[0, 0].set_ylabel('Frequency', fontsize=10)
        axes[0, 0].legend()
        
        # 2. P&L vs Trade Size
        scatter = axes[0, 1].scatter(self.trades['size'], self.trades['pnl'], 
                                    c=self.trades['pnl'] > 0, cmap='RdYlGn', alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].set_title('P&L vs Trade Size', fontsize=12)
        axes[0, 1].set_xlabel('Trade Size', fontsize=10)
        axes[0, 1].set_ylabel('Profit/Loss', fontsize=10)
        
        # Add legend
        legend1 = axes[0, 1].legend(*scatter.legend_elements(),
                                   loc="upper right", title="Outcome")
        axes[0, 1].add_artist(legend1)
        
        # Add trend line
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(self.trades['size'], self.trades['pnl'])
            x = np.array([self.trades['size'].min(), self.trades['size'].max()])
            axes[0, 1].plot(x, intercept + slope * x, 'r', 
                           label=f'Trend: y={slope:.4f}x+{intercept:.2f}')
            axes[0, 1].legend()
        except ImportError:
            pass
        
        # 3. Trade Size Over Time
        if 'entry_time' in self.trades.columns:
            axes[1, 0].plot(self.trades['entry_time'], self.trades['size'], 
                           color='#1f77b4', linewidth=1, marker='o', markersize=3)
            axes[1, 0].set_title('Trade Size Over Time', fontsize=12)
            axes[1, 0].set_xlabel('Date', fontsize=10)
            axes[1, 0].set_ylabel('Trade Size', fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # Add rolling average
            window = min(30, len(self.trades) // 5)  # Reasonable window size
            if window > 1:
                rolling_size = self.trades.sort_values('entry_time')['size'].rolling(window=window).mean()
                axes[1, 0].plot(self.trades.sort_values('entry_time')['entry_time'], rolling_size, 
                               color='red', linewidth=2, label=f'{window}-trade Moving Avg')
                axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'Entry time data not available', 
                           ha='center', va='center', fontsize=12)
        
        # 4. Win Rate by Size Quantile
        # Create size quantiles
        self.trades['size_quantile'] = pd.qcut(self.trades['size'], 5, labels=False)
        
        # Calculate win rate by size quantile
        win_rate_by_size = self.trades.groupby('size_quantile').apply(
            lambda x: (x['pnl'] > 0).mean() * 100
        ).reset_index()
        
        # Calculate average size by quantile for labels
        avg_size_by_quantile = self.trades.groupby('size_quantile')['size'].mean().reset_index()
        
        # Create labels
        size_labels = [f'Q{q+1}\n(Avg: {avg:.1f})' for q, avg in 
                      zip(avg_size_by_quantile['size_quantile'], avg_size_by_quantile['size'])]
        
        # Plot win rate by size quantile
        axes[1, 1].bar(win_rate_by_size['size_quantile'], win_rate_by_size[0], color='#1f77b4')
        axes[1, 1].axhline(y=50, color='red', linestyle='--', alpha=0.7, 
                          label='50% Win Rate')
        axes[1, 1].set_title('Win Rate by Size Quantile', fontsize=12)
        axes[1, 1].set_xlabel('Size Quantile', fontsize=10)
        axes[1, 1].set_ylabel('Win Rate (%)', fontsize=10)
        axes[1, 1].set_xticks(win_rate_by_size['size_quantile'])
        axes[1, 1].set_xticklabels(size_labels)
        axes[1, 1].set_ylim([0, 100])
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_summary(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot comprehensive performance summary.
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.returns is None:
            self.logger.error("Returns data not provided")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # Define grid layout
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # 1. Equity Curve (top row, spans all columns)
        ax_equity = fig.add_subplot(gs[0, :])
        if self.equity_curve is not None:
            ax_equity.plot(self.equity_curve.index, self.equity_curve.values, 
                          label='Strategy', linewidth=2, color='#1f77b4')
            
            if self.benchmark_returns is not None:
                # Convert benchmark returns to equity curve
                benchmark_equity = (1 + self.benchmark_returns).cumprod()
                
                # Scale benchmark to start at the same value as strategy
                benchmark_equity = benchmark_equity * self.equity_curve.iloc[0] / benchmark_equity.iloc[0]
                
                ax_equity.plot(benchmark_equity.index, benchmark_equity.values, 
                              label='Benchmark', linewidth=1.5, linestyle='--', color='#ff7f0e')
            
            ax_equity.set_title('Equity Curve', fontsize=12)
            ax_equity.set_ylabel('Equity', fontsize=10)
            ax_equity.grid(True, alpha=0.3)
            ax_equity.legend(fontsize=10)
        else:
            # If equity curve not available, plot cumulative returns
            cumulative_returns = (1 + self.returns).cumprod()
            ax_equity.plot(cumulative_returns.index, cumulative_returns.values, 
                          label='Strategy', linewidth=2, color='#1f77b4')
            
            if self.benchmark_returns is not None:
                cumulative_benchmark = (1 + self.benchmark_returns).cumprod()
                ax_equity.plot(cumulative_benchmark.index, cumulative_benchmark.values, 
                              label='Benchmark', linewidth=1.5, linestyle='--', color='#ff7f0e')
            
            ax_equity.set_title('Cumulative Returns', fontsize=12)
            ax_equity.set_ylabel('Cumulative Return', fontsize=10)
            ax_equity.grid(True, alpha=0.3)
            ax_equity.legend(fontsize=10)
        
        # 2. Drawdowns (middle row, first column)
        ax_drawdown = fig.add_subplot(gs[1, 0])
        if self.equity_curve is not None:
            # Calculate running maximum
            running_max = self.equity_curve.cummax()
            
            # Calculate drawdowns
            drawdowns = (self.equity_curve / running_max - 1) * 100  # Convert to percentage
        else:
            # Calculate from returns
            cumulative_returns = (1 + self.returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns / running_max - 1) * 100
        
        ax_drawdown.fill_between(drawdowns.index, drawdowns.values, 0, 
                                color='red', alpha=0.5)
        
        # Add horizontal lines at 5%, 10%, 20% drawdowns
        for level in [-5, -10, -20]:
            ax_drawdown.axhline(y=level, color='gray', linestyle='--', alpha=0.7)
        
        ax_drawdown.set_title('Drawdowns', fontsize=12)
        ax_drawdown.set_ylabel('Drawdown (%)', fontsize=10)
        
        # Set y-axis limits
        min_drawdown = min(drawdowns.min(), -25)  # At least -25%
        ax_drawdown.set_ylim([min_drawdown, 5])
        
        ax_drawdown.grid(True, alpha=0.3)
        
        # 3. Monthly Returns Heatmap (middle row, spans second and third columns)
        ax_monthly = fig.add_subplot(gs[1, 1:])
        
        # Resample returns to monthly
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table with years as rows and months as columns
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        # Replace column numbers with month names
        month_names = {i: calendar.month_abbr[i] for i in range(1, 13)}
        monthly_pivot.columns = [month_names[col] for col in monthly_pivot.columns]
        
        # Create heatmap
        sns.heatmap(monthly_pivot * 100, annot=True, fmt=".1f", 
                   cmap="RdYlGn", center=0, linewidths=1, ax=ax_monthly,
                   cbar_kws={'label': 'Monthly Return (%)'})
        
        ax_monthly.set_title('Monthly Returns (%)', fontsize=12)
        ax_monthly.set_ylabel('Year', fontsize=10)
        
        # 4. Rolling Statistics (bottom row, first column)
        ax_rolling = fig.add_subplot(gs[2, 0])
        
        # Calculate rolling statistics
        window = min(60, len(self.returns) // 5)  # Reasonable window size
        if window > 1:
            rolling_mean = self.returns.rolling(window=window).mean() * 252  # Annualized
            rolling_std = self.returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            rolling_sharpe = rolling_mean / rolling_std
            
            ax_rolling.plot(rolling_sharpe.index, rolling_sharpe.values, 
                           label=f'{window}-day Rolling Sharpe', 
                           color='#1f77b4', linewidth=2)
            
            # Add horizontal line at Sharpe = 1
            ax_rolling.axhline(y=1, color='red', linestyle='--', alpha=0.7)
            
            ax_rolling.set_title(f'{window}-day Rolling Sharpe Ratio', fontsize=12)
            ax_rolling.set_ylabel('Sharpe Ratio', fontsize=10)
            ax_rolling.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax_rolling.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax_rolling.text(0.5, 0.5, 'Insufficient data for rolling statistics', 
                           ha='center', va='center', fontsize=10)
        
        # 5. Return Distribution (bottom row, second column)
        ax_dist = fig.add_subplot(gs[2, 1])
        
        sns.histplot(self.returns * 100, bins=50, kde=True, ax=ax_dist, color='#1f77b4')
        
        # Add vertical line at zero
        ax_dist.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add vertical lines for mean and standard deviations
        mean = self.returns.mean() * 100
        std = self.returns.std() * 100
        
        ax_dist.axvline(x=mean, color='red', linestyle='--', alpha=0.8, 
                       label=f'Mean: {mean:.2f}%')
        
        ax_dist.set_title('Daily Returns Distribution', fontsize=12)
        ax_dist.set_xlabel('Return (%)', fontsize=10)
        ax_dist.set_ylabel('Frequency', fontsize=10)
        ax_dist.legend(fontsize=8)
        
        # 6. Performance Metrics (bottom row, third column)
        ax_metrics = fig.add_subplot(gs[2, 2])
        
        # Calculate key metrics
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) if self.equity_curve is not None else (1 + self.returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(self.returns)) - 1
        volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        if self.equity_curve is not None:
            running_max = self.equity_curve.cummax()
            drawdowns = self.equity_curve / running_max - 1
        else:
            cumulative_returns = (1 + self.returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdowns = cumulative_returns / running_max - 1
        
        max_drawdown = drawdowns.min()
        
        # Calculate win rate
        win_rate = (self.returns > 0).mean() * 100
        
        # Calculate Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Calculate Sortino ratio
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
        
        # Create metrics table
        metrics = {
            'Metric': ['Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio', 
                      'Max Drawdown', 'Win Rate', 'Calmar Ratio', 'Sortino Ratio'],
            'Value': [f'{total_return:.2%}', f'{annual_return:.2%}', f'{volatility:.2%}', 
                     f'{sharpe_ratio:.2f}', f'{max_drawdown:.2%}', f'{win_rate:.1f}%',
                     f'{calmar_ratio:.2f}', f'{sortino_ratio:.2f}']
        }
        
        # Hide axes
        ax_metrics.axis('off')
        
        # Create table
        table = ax_metrics.table(cellText=list(zip(metrics['Metric'], metrics['Value'])),
                                colLabels=['Metric', 'Value'],
                                loc='center',
                                cellLoc='left',
                                colWidths=[0.6, 0.4])
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Set title
        ax_metrics.set_title('Performance Metrics', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_analysis(self, other_returns: Dict[str, pd.Series], 
                                 figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot correlation analysis with other return series.
        
        Args:
            other_returns: Dictionary of other return series {name: returns}
            figsize: Figure size (width, height)
        """
        if self.returns is None:
            self.logger.error("Returns data not provided")
            return
        
        if not other_returns:
            self.logger.error("No other return series provided")
            return
        
        # Create combined DataFrame
        combined_returns = pd.DataFrame({'Strategy': self.returns})
        
        # Add benchmark if available
        if self.benchmark_returns is not None:
            combined_returns['Benchmark'] = self.benchmark_returns
        
        # Add other return series
        for name, returns in other_returns.items():
            combined_returns[name] = returns
        
        # Drop rows with NaN values
        combined_returns = combined_returns.dropna()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Correlation Matrix (top left)
        correlation_matrix = combined_returns.corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", 
                   cmap="coolwarm", center=0, linewidths=1, ax=axes[0, 0],
                   cbar_kws={'label': 'Correlation'})
        
        axes[0, 0].set_title('Correlation Matrix', fontsize=12)
        
        # 2. Rolling Correlation with Benchmark (top right)
        if self.benchmark_returns is not None:
            window = min(60, len(combined_returns) // 5)  # Reasonable window size
            if window > 1:
                rolling_corr = combined_returns['Strategy'].rolling(window=window).corr(combined_returns['Benchmark'])
                
                axes[0, 1].plot(rolling_corr.index, rolling_corr.values, 
                               label=f'{window}-day Rolling Correlation', 
                               color='#1f77b4', linewidth=2)
                
                # Add horizontal line at correlation = 0
                axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Add horizontal line at correlation = 1
                axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7)
                
                # Add horizontal line at correlation = -1
                axes[0, 1].axhline(y=-1, color='green', linestyle='--', alpha=0.7)
                
                axes[0, 1].set_title(f'{window}-day Rolling Correlation with Benchmark', fontsize=12)
                axes[0, 1].set_ylabel('Correlation', fontsize=10)
                axes[0, 1].grid(True, alpha=0.3)
                
                # Set y-axis limits
                axes[0, 1].set_ylim([-1.1, 1.1])
                
                # Rotate x-axis labels
                plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'Insufficient data for rolling correlation', 
                               ha='center', va='center', fontsize=10)
        else:
            axes[0, 1].text(0.5, 0.5, 'Benchmark data not available', 
                           ha='center', va='center', fontsize=10)
        
        # 3. Scatter Plot with Benchmark (bottom left)
        if self.benchmark_returns is not None:
            axes[1, 0].scatter(combined_returns['Benchmark'] * 100, 
                              combined_returns['Strategy'] * 100, 
                              alpha=0.5, color='#1f77b4')
            
            # Add diagonal line (y=x)
            min_val = min(combined_returns['Benchmark'].min(), combined_returns['Strategy'].min()) * 100
            max_val = max(combined_returns['Benchmark'].max(), combined_returns['Strategy'].max()) * 100
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 
                           linestyle='--', color='red', alpha=0.7)
            
            # Add horizontal and vertical lines at zero
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add trend line
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    combined_returns['Benchmark'] * 100, combined_returns['Strategy'] * 100)
                
                x = np.array([min_val, max_val])
                axes[1, 0].plot(x, intercept + slope * x, 'g', 
                               label=f'Slope: {slope:.2f}, R²: {r_value**2:.2f}')
                
                # Add beta annotation
                axes[1, 0].text(0.05, 0.95, f'Beta: {slope:.2f}', 
                               transform=axes[1, 0].transAxes,
                               fontsize=10, verticalalignment='top')
            except ImportError:
                pass
            
            axes[1, 0].set_title('Strategy vs Benchmark Returns', fontsize=12)
            axes[1, 0].set_xlabel('Benchmark Return (%)', fontsize=10)
            axes[1, 0].set_ylabel('Strategy Return (%)', fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend(fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'Benchmark data not available', 
                           ha='center', va='center', fontsize=10)
        
        # 4. Quadrant Analysis (bottom right)
        if self.benchmark_returns is not None:
            # Calculate quadrant statistics
            q1 = ((combined_returns['Strategy'] > 0) & (combined_returns['Benchmark'] > 0)).mean() * 100  # Both positive
            q2 = ((combined_returns['Strategy'] > 0) & (combined_returns['Benchmark'] < 0)).mean() * 100  # Strategy positive, Benchmark negative
            q3 = ((combined_returns['Strategy'] < 0) & (combined_returns['Benchmark'] < 0)).mean() * 100  # Both negative
            q4 = ((combined_returns['Strategy'] < 0) & (combined_returns['Benchmark'] > 0)).mean() * 100  # Strategy negative, Benchmark positive
            
            # Create quadrant plot
            axes[1, 1].axis('equal')
            axes[1, 1].axis([-1, 1, -1, 1])
            
            # Draw quadrant lines
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Add quadrant labels
            axes[1, 1].text(0.5, 0.5, f'Q1: {q1:.1f}%\nBoth Positive', 
                           ha='center', va='center', fontsize=10)
            
            axes[1, 1].text(-0.5, 0.5, f'Q2: {q2:.1f}%\nStrategy +\nBenchmark -', 
                           ha='center', va='center', fontsize=10)
            
            axes[1, 1].text(-0.5, -0.5, f'Q3: {q3:.1f}%\nBoth Negative', 
                           ha='center', va='center', fontsize=10)
            
            axes[1, 1].text(0.5, -0.5, f'Q4: {q4:.1f}%\nStrategy -\nBenchmark +', 
                           ha='center', va='center', fontsize=10)
            
            # Add arrows to indicate axes
            axes[1, 1].arrow(0, 0, 0.9, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
            axes[1, 1].arrow(0, 0, 0, 0.9, head_width=0.05, head_length=0.05, fc='black', ec='black')
            
            # Add axis labels
            axes[1, 1].text(0.95, 0, 'Benchmark +', ha='center', va='center', fontsize=8)
            axes[1, 1].text(0, 0.95, 'Strategy +', ha='center', va='center', fontsize=8)
            
            axes[1, 1].set_title('Quadrant Analysis', fontsize=12)
            
            # Remove ticks
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
        else:
            axes[1, 1].text(0.5, 0.5, 'Benchmark data not available', 
                           ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()