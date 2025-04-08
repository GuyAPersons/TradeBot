import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import calendar
from scipy import stats

class EquityCurvePlotter:
    """
    Class for creating and visualizing equity curves from backtest results.
    
    This class provides methods to plot equity curves, drawdowns, underwater plots,
    and other visualizations related to portfolio equity over time.
    """
    
    def __init__(self, equity_data: Optional[pd.DataFrame] = None):
        """
        Initialize the equity curve plotter.
        
        Args:
            equity_data: DataFrame containing equity curves with DatetimeIndex
        """
        self.equity_data = equity_data
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set default style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def load_equity_data(self, data: Union[pd.DataFrame, pd.Series, str, Path]) -> None:
        """
        Load equity data from various sources.
        
        Args:
            data: Equity data as DataFrame, Series, or file path
        """
        if isinstance(data, pd.DataFrame):
            self.equity_data = data
        elif isinstance(data, pd.Series):
            self.equity_data = pd.DataFrame(data)
        elif isinstance(data, (str, Path)):
            try:
                # Try to load from file
                path = Path(data)
                if path.suffix.lower() == '.csv':
                    self.equity_data = pd.read_csv(path, index_col=0, parse_dates=True)
                elif path.suffix.lower() in ['.pkl', '.pickle']:
                    self.equity_data = pd.read_pickle(path)
                else:
                    self.logger.error(f"Unsupported file format: {path.suffix}")
                    return
            except Exception as e:
                self.logger.error(f"Failed to load equity data: {str(e)}")
                return
        else:
            self.logger.error(f"Unsupported data type: {type(data)}")
            return
        
        # Validate the data
        if not isinstance(self.equity_data.index, pd.DatetimeIndex):
            try:
                self.equity_data.index = pd.to_datetime(self.equity_data.index)
            except:
                self.logger.error("Failed to convert index to DatetimeIndex")
                self.equity_data = None
                return
        
        self.logger.info(f"Loaded equity data with shape {self.equity_data.shape}")
    
    def calculate_drawdowns(self) -> pd.DataFrame:
        """
        Calculate drawdowns for each equity curve.
        
        Returns:
            DataFrame containing drawdown series for each equity curve
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return pd.DataFrame()
        
        drawdowns = pd.DataFrame(index=self.equity_data.index)
        
        for column in self.equity_data.columns:
            equity = self.equity_data[column]
            # Calculate running maximum
            running_max = equity.cummax()
            # Calculate drawdown percentage
            drawdown = (equity / running_max - 1) * 100
            drawdowns[f"{column}_drawdown"] = drawdown
        
        return drawdowns
    
    def plot_equity_curve(self, 
                         figsize: Tuple[int, int] = (12, 6),
                         title: str = "Equity Curve",
                         log_scale: bool = False,
                         include_drawdown: bool = True) -> None:
        """
        Plot equity curve with optional drawdown panel.
        
        Args:
            figsize: Figure size (width, height)
            title: Plot title
            log_scale: Whether to use logarithmic scale for equity
            include_drawdown: Whether to include drawdown panel
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return
        
        # Create figure
        if include_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]},
                                          sharex=True)
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot equity curves
        for column in self.equity_data.columns:
            ax1.plot(self.equity_data.index, self.equity_data[column], 
                    label=column, linewidth=2)
        
        # Set log scale if requested
        if log_scale:
            ax1.set_yscale('log')
        
        # Format y-axis as currency
        def currency_formatter(x, pos):
            return f"${x:,.0f}"
        
        ax1.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        
        # Add labels and title
        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel('Portfolio Value', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Add drawdown panel if requested
        if include_drawdown:
            drawdowns = self.calculate_drawdowns()
            
            for column in drawdowns.columns:
                ax2.fill_between(drawdowns.index, 0, drawdowns[column], 
                                alpha=0.3, label=column.replace('_drawdown', ''))
            
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Set y-axis limits for drawdown
            ax2.set_ylim(drawdowns.min().min() * 1.1, 5)
        else:
            ax1.set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_underwater_chart(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot underwater chart showing drawdowns over time.
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return
        
        drawdowns = self.calculate_drawdowns()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for column in drawdowns.columns:
            strategy_name = column.replace('_drawdown', '')
            ax.fill_between(drawdowns.index, 0, drawdowns[column], 
                           alpha=0.7, label=strategy_name)
        
        # Add horizontal lines at common drawdown levels
        levels = [-5, -10, -20, -30, -50]
        for level in levels:
            ax.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
            ax.text(drawdowns.index[0], level, f"{level}%", va='center', ha='left')
        
        # Add labels and title
        ax.set_title('Underwater Chart (Drawdowns)', fontsize=14)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        
        # Set y-axis limits
        min_dd = drawdowns.min().min()
        ax.set_ylim(min_dd * 1.1, 5)
        
        # Add legend
        ax.legend(loc='lower right')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot distribution of drawdowns.
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return
        
        drawdowns = self.calculate_drawdowns()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot histograms of drawdowns
        for column in drawdowns.columns:
            strategy_name = column.replace('_drawdown', '')
            sns.histplot(drawdowns[column], kde=True, 
                        label=strategy_name, ax=axes[0], alpha=0.5)
        
        # Add labels and title for histogram
        axes[0].set_title('Drawdown Distribution', fontsize=14)
        axes[0].set_xlabel('Drawdown (%)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend()
        
        # Plot boxplot of drawdowns
        drawdown_data = pd.DataFrame()
        for column in drawdowns.columns:
            strategy_name = column.replace('_drawdown', '')
            drawdown_data[strategy_name] = drawdowns[column]
        
        sns.boxplot(data=drawdown_data, ax=axes[1])
        
        # Add labels and title for boxplot
        axes[1].set_title('Drawdown Boxplot', fontsize=14)
        axes[1].set_xlabel('Strategy', fontsize=12)
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_returns(self, 
                            window: int = 252, 
                            figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot rolling returns over specified window.
        
        Args:
            window: Rolling window size in days
            figsize: Figure size (width, height)
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return
        
        # Calculate daily returns
        daily_returns = self.equity_data.pct_change().dropna()
        
        # Calculate rolling returns (annualized)
        rolling_returns = pd.DataFrame(index=daily_returns.index)
        
        for column in daily_returns.columns:
            # Calculate rolling return (annualized)
            rolling_ret = (1 + daily_returns[column]).rolling(window).apply(
                lambda x: (x.prod() - 1) * (252 / window), raw=True
            )
            rolling_returns[f"{column}_rolling"] = rolling_ret * 100
        
        # Plot rolling returns
        fig, ax = plt.subplots(figsize=figsize)
        
        for column in rolling_returns.columns:
            strategy_name = column.replace('_rolling', '')
            ax.plot(rolling_returns.index, rolling_returns[column], 
                   label=strategy_name, linewidth=2)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels and title
        ax.set_title(f'{window}-Day Rolling Returns (Annualized)', fontsize=14)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns_heatmap(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot monthly returns heatmap.
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return
        
        # Calculate daily returns
        daily_returns = self.equity_data.pct_change().dropna()
        
        # Create figure with subplots for each strategy
        fig, axes = plt.subplots(len(daily_returns.columns), 1, 
                                figsize=figsize, squeeze=False)
        
        for i, column in enumerate(daily_returns.columns):
            # Calculate monthly returns
            monthly_returns = daily_returns[column].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Create pivot table: rows=years, columns=months
            monthly_pivot = pd.pivot_table(
                monthly_returns.reset_index(),
                values=column,
                index=monthly_returns.index.year,
                columns=monthly_returns.index.month,
                aggfunc='first'
            )
            
            # Replace month numbers with month names
            month_names = {i: calendar.month_abbr[i] for i in range(1, 13)}
            monthly_pivot.columns = [month_names[col] for col in monthly_pivot.columns]
            
            # Plot heatmap
            sns.heatmap(
                monthly_pivot * 100,  # Convert to percentage
                annot=True,
                fmt=".1f",
                cmap="RdYlGn",
                center=0,
                linewidths=1,
                ax=axes[i, 0],
                cbar_kws={'label': 'Monthly Return (%)'}
            )
            
            axes[i, 0].set_title(f'{column} - Monthly Returns (%)', fontsize=12)
            axes[i, 0].set_ylabel('Year', fontsize=10)
            
            # If last subplot, add x-label
            if i == len(daily_returns.columns) - 1:
                axes[i, 0].set_xlabel('Month', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_return_quantiles(self, 
                             periods: List[int] = [1, 5, 21, 63, 126, 252],
                             figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot return quantiles for different time periods.
        
        Args:
            periods: List of periods (in days) to analyze
            figsize: Figure size (width, height)
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return
        
        # Calculate daily returns
        daily_returns = self.equity_data.pct_change().dropna()
        
        # Create figure
        fig, axes = plt.subplots(len(daily_returns.columns), 1, figsize=figsize, squeeze=False)
        
        for i, column in enumerate(daily_returns.columns):
            # Calculate returns for different periods
            period_returns = {}
            for period in periods:
                if period == 1:
                    # Daily returns
                    period_returns[f"{period}D"] = daily_returns[column]
                else:
                    # Rolling returns for longer periods
                    period_returns[f"{period}D"] = daily_returns[column].rolling(period).apply(
                        lambda x: (1 + x).prod() - 1, raw=True
                    ).dropna()
            
            # Calculate quantiles
            quantiles = {}
            for period_name, returns in period_returns.items():
                quantiles[period_name] = [
                    returns.quantile(0.05),  # 5th percentile
                    returns.quantile(0.25),  # 25th percentile
                    returns.quantile(0.5),   # Median
                    returns.quantile(0.75),  # 75th percentile
                    returns.quantile(0.95)   # 95th percentile
                ]
            
            # Convert to DataFrame for plotting
            quantile_df = pd.DataFrame(quantiles).T * 100  # Convert to percentage
            quantile_df.columns = ['5%', '25%', '50%', '75%', '95%']
            
            # Plot
            quantile_df.plot(kind='bar', ax=axes[i, 0], width=0.8)
            
            # Add horizontal line at 0
            axes[i, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add labels and title
            axes[i, 0].set_title(f'{column} - Return Quantiles by Time Period', fontsize=12)
            axes[i, 0].set_ylabel('Return (%)', fontsize=10)
            
            # If last subplot, add x-label
            if i == len(daily_returns.columns) - 1:
                axes[i, 0].set_xlabel('Time Period', fontsize=10)
            
            # Add grid
            axes[i, 0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for j, p in enumerate(quantile_df.columns):
                for k, v in enumerate(quantile_df[p]):
                    axes[i, 0].text(k, v + (0.5 if v >= 0 else -1.5), 
                                   f'{v:.1f}%', ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_calendar_returns(self, figsize: Tuple[int, int] = (14, 8)) -> None:
        """
        Plot calendar returns (monthly and yearly).
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return
        
        # Convert equity curves to returns
        returns_data = self.equity_data.pct_change().dropna()
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(returns_data.columns), 2, figsize=figsize)
        
        # Handle single curve case
        if len(returns_data.columns) == 1:
            axes = np.array([axes])
        
        for i, column in enumerate(returns_data.columns):
            # Calculate monthly returns
            monthly_returns = returns_data[column].resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns.index = monthly_returns.index.to_period('M')
            
            # Create pivot table for monthly returns
            monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
            
            # Replace column numbers with month names
            import calendar
            month_names = {i: calendar.month_abbr[i] for i in range(1, 13)}
            monthly_pivot.columns = [month_names[col] for col in monthly_pivot.columns]
            
            # Create heatmap for monthly returns
            sns.heatmap(monthly_pivot * 100, annot=True, fmt=".1f",
                        cmap="RdYlGn", center=0, linewidths=1, ax=axes[i, 0],
                       cbar_kws={'label': 'Monthly Return (%)'})
            
            axes[i, 0].set_title(f'{column} - Monthly Returns (%)', fontsize=12)
            axes[i, 0].set_ylabel('Year', fontsize=10)
            
            # Calculate yearly returns
            yearly_returns = returns_data[column].resample('Y').apply(lambda x: (1 + x).prod() - 1)
            yearly_returns.index = yearly_returns.index.year
            
            # Create bar chart for yearly returns
            bars = axes[i, 1].bar(yearly_returns.index, yearly_returns.values * 100)
            
            # Color bars based on return (green for positive, red for negative)
            for j, bar in enumerate(bars):
                bar.set_color('green' if yearly_returns.values[j] >= 0 else 'red')
                
                # Add value labels
                height = bar.get_height()
                axes[i, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom')
            
            axes[i, 1].set_title(f'{column} - Yearly Returns (%)', fontsize=12)
            axes[i, 1].set_ylabel('Return (%)', fontsize=10)
            axes[i, 1].set_xlabel('Year', fontsize=10)
            
            # Add horizontal line at 0%
            axes[i, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add grid
            axes[i, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_return_distribution_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot and compare return distributions.
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return
        
        # Convert equity curves to returns
        returns_data = self.equity_data.pct_change().dropna()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot histograms
        for column in returns_data.columns:
            sns.histplot(returns_data[column] * 100, kde=True, 
                         label=column, ax=axes[0], alpha=0.5)
        
        # Add vertical line at 0
        axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Set labels and title for histogram
        axes[0].set_title('Return Distribution Comparison', fontsize=14)
        axes[0].set_xlabel('Daily Return (%)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend(fontsize=10)
        
        # Plot QQ plot if we have two return series
        if len(returns_data.columns) == 2:
            from scipy import stats
            
            # Get the two return series
            series1 = returns_data[returns_data.columns[0]]
            series2 = returns_data[returns_data.columns[1]]
            
            # Create QQ plot
            stats.probplot(series1, dist="norm", plot=axes[1])
            axes[1].set_title(f'QQ Plot - {returns_data.columns[0]}', fontsize=14)
            axes[1].set_xlabel('Theoretical Quantiles', fontsize=12)
            axes[1].set_ylabel('Sample Quantiles', fontsize=12)
        else:
            # If we have more or less than 2 series, show boxplots instead
            sns.boxplot(data=returns_data * 100, ax=axes[1])
            
            # Add points for mean values
            means = returns_data.mean() * 100
            for i, mean in enumerate(means):
                axes[1].scatter(i, mean, marker='o', color='red', s=50, zorder=10)
                axes[1].text(i, mean, f'  Mean: {mean:.2f}%', ha='left', va='center', fontsize=8)
            
            # Set labels and title for boxplot
            axes[1].set_title('Return Distribution Boxplots', fontsize=14)
            axes[1].set_xlabel('Strategy', fontsize=12)
            axes[1].set_ylabel('Daily Return (%)', fontsize=12)
            
            # Add horizontal line at 0
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cumulative_returns_comparison(self, benchmark_data: pd.Series = None,
                                           log_scale: bool = False,
                                          figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot cumulative returns comparison.
        
        Args:
            benchmark_data: Optional benchmark return series
            log_scale: Whether to use log scale for y-axis
            figsize: Figure size (width, height)
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cumulative returns for each equity curve
        for column in self.equity_data.columns:
            # Normalize to start at 1
            normalized = self.equity_data[column] / self.equity_data[column].iloc[0]
            ax.plot(normalized.index, normalized.values,
                    label=column, linewidth=2)
        
        # Add benchmark if provided
        if benchmark_data is not None:
            # Align benchmark with equity data
            common_index = self.equity_data.index.intersection(benchmark_data.index)
            if len(common_index) > 0:
                aligned_benchmark = benchmark_data.loc[common_index]
                
                # Normalize to start at 1
                normalized_benchmark = aligned_benchmark / aligned_benchmark.iloc[0]
                ax.plot(normalized_benchmark.index, normalized_benchmark.values,
                        label='Benchmark', linewidth=2, linestyle='--', color='black')
        
        # Set log scale if requested
        if log_scale:
            ax.set_yscale('log')
        
        # Set labels and title
        ax.set_title('Cumulative Return Comparison', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Growth of $1', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=10)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()