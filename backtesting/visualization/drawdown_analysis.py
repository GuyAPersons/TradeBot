import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
from matplotlib.ticker import FuncFormatter
import logging
from datetime import datetime, timedelta

class DrawdownAnalysis:
    """
    Specialized tools for drawdown analysis and visualization.
    """
    
    def __init__(self, equity_data: Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]] = None):
        """
        Initialize drawdown analysis tools.
        
        Args:
            equity_data: Equity curve data. Can be:
                - pd.Series: Single equity curve
                - pd.DataFrame: Multiple equity curves as columns
                - Dict[str, pd.Series]: Dictionary of named equity curves
        """
        self.logger = logging.getLogger(__name__)
        
        if equity_data is None:
            self.equity_data = None
        elif isinstance(equity_data, pd.Series):
            self.equity_data = pd.DataFrame({equity_data.name or 'Strategy': equity_data})
        elif isinstance(equity_data, pd.DataFrame):
            self.equity_data = equity_data
        elif isinstance(equity_data, dict):
            self.equity_data = pd.DataFrame(equity_data)
        else:
            self.logger.error(f"Unsupported equity data type: {type(equity_data)}")
            self.equity_data = None
        
        # Calculate drawdowns if equity data is provided
        if self.equity_data is not None:
            self.calculate_drawdowns()
        
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
    
    def calculate_drawdowns(self) -> None:
        """
        Calculate drawdown series and drawdown information for all equity curves.
        """
        if self.equity_data is None or self.equity_data.empty:
            self.logger.error("No equity data available")
            return
        
        # Initialize containers
        self.drawdown_series = pd.DataFrame(index=self.equity_data.index)
        self.drawdown_info = {}
        
        for column in self.equity_data.columns:
            equity_curve = self.equity_data[column].dropna()
            
            # Calculate running maximum
            running_max = equity_curve.cummax()
            
            # Calculate drawdowns as percentage
            drawdowns = (equity_curve / running_max - 1) * 100
            self.drawdown_series[column] = drawdowns
            
            # Identify drawdown periods
            is_drawdown = drawdowns < 0
            
            # Find start of drawdown periods
            starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
            start_dates = drawdowns.index[starts]
            
            # Find end of drawdown periods (recovery)
            ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)
            end_dates = drawdowns.index[ends]
            
            # If we're currently in a drawdown, add the last date
            if is_drawdown.iloc[-1]:
                end_dates = end_dates.append(pd.Index([drawdowns.index[-1]]))
            
            # Compile drawdown information
            drawdown_periods = []
            
            for i, start_date in enumerate(start_dates):
                if i >= len(end_dates):
                    break
                    
                end_date = end_dates[i]
                
                # Get drawdown values in this period
                period_drawdowns = drawdowns.loc[start_date:end_date]
                
                # Find the lowest point (maximum drawdown in this period)
                max_dd_date = period_drawdowns.idxmin()
                max_dd = period_drawdowns.loc[max_dd_date]
                
                # Calculate recovery time (in trading days)
                recovery_time = len(period_drawdowns.loc[max_dd_date:end_date])
                
                # Calculate drawdown duration (in trading days)
                duration = len(period_drawdowns)
                
                # Get the peak value before drawdown
                peak_value = running_max.loc[start_date]
                
                # Get the value at the bottom of the drawdown
                bottom_value = equity_curve.loc[max_dd_date]
                
                # Calculate the monetary loss
                monetary_loss = bottom_value - peak_value
                
                drawdown_periods.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'max_dd_date': max_dd_date,
                    'max_drawdown': max_dd,
                    'duration': duration,
                    'recovery_time': recovery_time,
                    'peak_value': peak_value,
                    'bottom_value': bottom_value,
                    'monetary_loss': monetary_loss
                })
            
            # Sort drawdown periods by maximum drawdown (worst first)
            drawdown_periods.sort(key=lambda x: x['max_drawdown'])
            
            self.drawdown_info[column] = drawdown_periods
    
    def get_worst_drawdowns(self, n: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Get information about the worst drawdowns for each equity curve.
        
        Args:
            n: Number of worst drawdowns to return
            
        Returns:
            Dictionary of DataFrames with drawdown information
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return {}
        
        result = {}
        
        for column, drawdowns in self.drawdown_info.items():
            # Take the n worst drawdowns
            worst_n = drawdowns[:n]
            
            # Convert to DataFrame
            df = pd.DataFrame(worst_n)
            
            # Format percentage columns
            if not df.empty:
                df['max_drawdown'] = df['max_drawdown'].round(2)
            
            result[column] = df
        
        return result
    
    def plot_underwater_chart(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot underwater chart (drawdowns over time).
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_series') or self.drawdown_series.empty:
            self.logger.error("Drawdown series not available")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drawdowns for each equity curve
        for column in self.drawdown_series.columns:
            ax.fill_between(self.drawdown_series.index, 0, self.drawdown_series[column], 
                           label=column, alpha=0.7)
        
        # Add horizontal lines at 5%, 10%, 20% drawdowns
        for level in [-5, -10, -20]:
            ax.axhline(y=level, color='gray', linestyle='--', alpha=0.7)
            ax.text(self.drawdown_series.index[0], level, f'{level}%', va='center', ha='left', fontsize=8)
        
        # Set labels and title
        ax.set_title('Underwater Chart (Drawdowns)', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        
        # Set y-axis limits
        min_drawdown = min(self.drawdown_series.min().min(), -25)  # At least -25%
        ax.set_ylim([min_drawdown, 5])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_distribution(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot drawdown distribution analysis.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Prepare data for plotting
        all_data = {
            'Strategy': [],
            'Max Drawdown (%)': [],
            'Duration (days)': [],
            'Recovery Time (days)': []
        }
        
        for strategy, drawdowns in self.drawdown_info.items():
            for dd in drawdowns:
                all_data['Strategy'].append(strategy)
                all_data['Max Drawdown (%)'].append(dd['max_drawdown'])
                all_data['Duration (days)'].append(dd['duration'])
                all_data['Recovery Time (days)'].append(dd['recovery_time'])
        
        # Convert to DataFrame
        dd_df = pd.DataFrame(all_data)
        
        # 1. Drawdown Magnitude Distribution
        sns.histplot(data=dd_df, x='Max Drawdown (%)', hue='Strategy', 
                    kde=True, ax=axes[0], alpha=0.6)
        axes[0].set_title('Drawdown Magnitude Distribution', fontsize=12)
        axes[0].set_xlabel('Drawdown (%)', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)
        
        # 2. Drawdown Duration Distribution
        sns.histplot(data=dd_df, x='Duration (days)', hue='Strategy', 
                    kde=True, ax=axes[1], alpha=0.6)
        axes[1].set_title('Drawdown Duration Distribution', fontsize=12)
        axes[1].set_xlabel('Duration (trading days)', fontsize=10)
        axes[1].set_ylabel('Frequency', fontsize=10)
        
        # 3. Recovery Time Distribution
        sns.histplot(data=dd_df, x='Recovery Time (days)', hue='Strategy', 
                    kde=True, ax=axes[2], alpha=0.6)
        axes[2].set_title('Recovery Time Distribution', fontsize=12)
        axes[2].set_xlabel('Recovery Time (trading days)', fontsize=10)
        axes[2].set_ylabel('Frequency', fontsize=10)
        
        # 4. Scatter plot of Drawdown vs Duration
        sns.scatterplot(data=dd_df, x='Max Drawdown (%)', y='Duration (days)', 
                       hue='Strategy', size='Recovery Time (days)', 
                       sizes=(20, 200), alpha=0.7, ax=axes[3])
        axes[3].set_title('Drawdown vs Duration', fontsize=12)
        axes[3].set_xlabel('Drawdown (%)', fontsize=10)
        axes[3].set_ylabel('Duration (trading days)', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def plot_worst_drawdowns(self, n: int = 5, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot the worst drawdowns for each equity curve.
        
        Args:
            n: Number of worst drawdowns to plot
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return
        
        # Create figure with subplots (one row per strategy)
        fig, axes = plt.subplots(len(self.drawdown_info), 1, figsize=figsize)
        
        # Handle single strategy case
        if len(self.drawdown_info) == 1:
            axes = [axes]
        
        # Plot worst drawdowns for each strategy
        for i, (strategy, drawdowns) in enumerate(self.drawdown_info.items()):
            ax = axes[i]
            
            # Get equity curve
            equity_curve = self.equity_data[strategy]
            
            # Plot equity curve
            ax.plot(equity_curve.index, equity_curve.values, 
                   label=strategy, linewidth=2, color='#1f77b4')
            
            # Highlight worst drawdowns
            colors = plt.cm.Reds(np.linspace(0.3, 0.8, min(n, len(drawdowns))))
            
            for j, dd in enumerate(drawdowns[:n]):
                # Highlight drawdown period
                ax.axvspan(dd['start_date'], dd['end_date'], alpha=0.2, color=colors[j])
                
                # Mark peak and bottom
                ax.scatter(dd['start_date'], dd['peak_value'], 
                          color=colors[j], s=100, marker='^', zorder=5)
                
                ax.scatter(dd['max_dd_date'], dd['bottom_value'], 
                          color=colors[j], s=100, marker='v', zorder=5)
                
                # Add annotation
                mid_date = dd['start_date'] + (dd['max_dd_date'] - dd['start_date']) / 2
                mid_value = (dd['peak_value'] + dd['bottom_value']) / 2
                
                ax.annotate(f"#{j+1}: {dd['max_drawdown']:.1f}%", 
                           xy=(mid_date, mid_value),
                           xytext=(0, 30),
                           textcoords='offset points',
                           arrowprops=dict(arrowstyle='->', color=colors[j]),
                           ha='center', color=colors[j], fontweight='bold')
            
            ax.set_title(f'{strategy} with Top {min(n, len(drawdowns))} Drawdowns', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            formatter = FuncFormatter(lambda x, pos: f'${x:,.0f}')
            ax.yaxis.set_major_formatter(formatter)
            
            # Rotate x-axis labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_recovery_analysis(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot drawdown recovery analysis.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for plotting
        strategies = []
        avg_recovery_times = []
        max_drawdowns = []
        total_drawdowns = []
        
        for strategy, drawdowns in self.drawdown_info.items():
            if not drawdowns:
                continue
                
            strategies.append(strategy)
            
            # Calculate average recovery time
            recovery_times = [dd['recovery_time'] for dd in drawdowns]
            avg_recovery_times.append(np.mean(recovery_times))
            
            # Get maximum drawdown
            max_drawdowns.append(min([dd['max_drawdown'] for dd in drawdowns]))
            
            # Count total drawdowns
            total_drawdowns.append(len(drawdowns))
        
        # Create scatter plot
        scatter = ax.scatter(max_drawdowns, avg_recovery_times, 
                            s=[t * 20 for t in total_drawdowns],  # Size based on total drawdowns
                            c=range(len(strategies)),  # Color based on strategy index
                            alpha=0.7, cmap='viridis')
        
        # Add strategy labels
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, 
                       (max_drawdowns[i], avg_recovery_times[i]),
                       xytext=(5, 5),
                       textcoords='offset points')
        
        # Add legend for bubble size
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, 
                                                num=4, func=lambda s: s/20)
        legend1 = ax.legend(handles, labels, loc="upper left", title="Total Drawdowns")
        ax.add_artist(legend1)
        
        # Set labels and title
        ax.set_title('Drawdown Recovery Analysis', fontsize=14)
        ax.set_xlabel('Maximum Drawdown (%)', fontsize=12)
        ax.set_ylabel('Average Recovery Time (trading days)', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Invert x-axis (so worse drawdowns are on the right)
        ax.invert_xaxis()
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_timeline(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot drawdown timeline showing when drawdowns occurred across strategies.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for timeline
        strategies = list(self.drawdown_info.keys())
        
        # Plot timeline for each strategy
        for i, strategy in enumerate(strategies):
            drawdowns = self.drawdown_info[strategy]
            
            for dd in drawdowns:
                # Calculate color based on drawdown severity
                # Deeper drawdowns are darker red
                severity = min(1.0, abs(dd['max_drawdown']) / 20)  # Normalize to [0,1]
                color = plt.cm.Reds(0.3 + 0.7 * severity)
                
                # Plot horizontal line for drawdown period
                ax.plot([dd['start_date'], dd['end_date']], [i, i], 
                       linewidth=6, solid_capstyle='butt',
                       color=color, alpha=0.7)
                
                # Mark the bottom of the drawdown
                ax.scatter(dd['max_dd_date'], i, color='black', s=30, zorder=5)
        
        # Set y-ticks to strategy names
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)
        
        # Set labels and title
        ax.set_title('Drawdown Timeline', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_drawdown_stats(self) -> pd.DataFrame:
        """
        Calculate and return drawdown statistics.
        
        Returns:
            DataFrame with drawdown statistics
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return pd.DataFrame()
        
        stats = []
        
        for strategy, drawdowns in self.drawdown_info.items():
            if not drawdowns:
                continue
                
            # Extract drawdown values and durations
            dd_values = [dd['max_drawdown'] for dd in drawdowns]
            dd_durations = [dd['duration'] for dd in drawdowns]
            recovery_times = [dd['recovery_time'] for dd in drawdowns]
            
            # Calculate statistics
            stats.append({
                'Strategy': strategy,
                'Max Drawdown (%)': min(dd_values),
                'Avg Drawdown (%)': np.mean(dd_values),
                'Median Drawdown (%)': np.median(dd_values),
                'Total Drawdowns': len(drawdowns),
                'Avg Duration (days)': np.mean(dd_durations),
                'Max Duration (days)': max(dd_durations),
                'Avg Recovery (days)': np.mean(recovery_times),
                'Max Recovery (days)': max(recovery_times),
                'Time in Drawdown (%)': sum(dd_durations) / len(self.equity_data) * 100
            })
        
        return pd.DataFrame(stats)
    
    def plot_drawdown_stats(self, figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Plot drawdown statistics comparison.
        
        Args:
            figsize: Figure size (width, height)
        """
        # Calculate drawdown statistics
        stats_df = self.calculate_drawdown_stats()
        
        if stats_df.empty:
            self.logger.error("No drawdown statistics available")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # 1. Maximum Drawdown
        sns.barplot(x='Strategy', y='Max Drawdown (%)', data=stats_df, ax=axes[0], palette='Blues_r')
        axes[0].set_title('Maximum Drawdown', fontsize=12)
        axes[0].set_ylabel('Drawdown (%)', fontsize=10)
        axes[0].set_xlabel('')
        
        # Add value labels
        for p in axes[0].patches:
            axes[0].annotate(f'{p.get_height():.1f}%', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        # 2. Average Drawdown Duration
        sns.barplot(x='Strategy', y='Avg Duration (days)', data=stats_df, ax=axes[1], palette='Greens')
        axes[1].set_title('Average Drawdown Duration', fontsize=12)
        axes[1].set_ylabel('Trading Days', fontsize=10)
        axes[1].set_xlabel('')
        
        # Add value labels
        for p in axes[1].patches:
            axes[1].annotate(f'{p.get_height():.1f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        # 3. Average Recovery Time
        sns.barplot(x='Strategy', y='Avg Recovery (days)', data=stats_df, ax=axes[2], palette='Oranges')
        axes[2].set_title('Average Recovery Time', fontsize=12)
        axes[2].set_ylabel('Trading Days', fontsize=10)
        axes[2].set_xlabel('')
        
        # Add value labels
        for p in axes[2].patches:
            axes[2].annotate(f'{p.get_height():.1f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        # 4. Time in Drawdown
        sns.barplot(x='Strategy', y='Time in Drawdown (%)', data=stats_df, ax=axes[3], palette='Reds')
        axes[3].set_title('Time Spent in Drawdown', fontsize=12)
        axes[3].set_ylabel('Percentage of Time', fontsize=10)
        axes[3].set_xlabel('')
        
        # Add value labels
        for p in axes[3].patches:
            axes[3].annotate(f'{p.get_height():.1f}%', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return stats_df
    
    def plot_drawdown_heatmap(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot drawdown heatmap showing when drawdowns occurred over time.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_series') or self.drawdown_series.empty:
            self.logger.error("Drawdown series not available")
            return
        
        # Resample drawdowns to monthly frequency for better visualization
        monthly_drawdowns = self.drawdown_series.resample('M').min()
        
        # Create pivot table with years as rows and months as columns
        pivot_data = {}
        
        for column in monthly_drawdowns.columns:
            series = monthly_drawdowns[column]
            
            # Create pivot table
            pivot = series.unstack(level=0)
            pivot.index = pivot.index.year
            pivot.columns = pivot.columns.month
            
            # Replace month numbers with month names
            import calendar
            pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]
            
            pivot_data[column] = pivot
        
        # Create figure with subplots (one per strategy)
        fig, axes = plt.subplots(len(pivot_data), 1, figsize=figsize)
        
        # Handle single strategy case
        if len(pivot_data) == 1:
            axes = [axes]
        
        # Plot heatmap for each strategy
        for i, (strategy, pivot) in enumerate(pivot_data.items()):
            # Create heatmap
            sns.heatmap(pivot, cmap="RdYlGn", center=0, linewidths=1, 
                       ax=axes[i], cbar_kws={'label': 'Drawdown (%)'})
            
            axes[i].set_title(f'{strategy} - Monthly Maximum Drawdowns', fontsize=12)
            axes[i].set_ylabel('Year', fontsize=10)
            
            # Only show x-label for the bottom subplot
            if i == len(pivot_data) - 1:
                axes[i].set_xlabel('Month', fontsize=10)
            else:
                axes[i].set_xlabel('')
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_recovery_scatter(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot scatter of drawdown magnitude vs recovery time.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return
        
        # Prepare data for plotting
        all_data = {
            'Strategy': [],
            'Max Drawdown (%)': [],
            'Recovery Time (days)': [],
            'Duration (days)': []
        }
        
        for strategy, drawdowns in self.drawdown_info.items():
            for dd in drawdowns:
                all_data['Strategy'].append(strategy)
                all_data['Max Drawdown (%)'].append(dd['max_drawdown'])
                all_data['Recovery Time (days)'].append(dd['recovery_time'])
                all_data['Duration (days)'].append(dd['duration'])
        
        # Convert to DataFrame
        dd_df = pd.DataFrame(all_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot
        sns.scatterplot(data=dd_df, x='Max Drawdown (%)', y='Recovery Time (days)', 
                       hue='Strategy', size='Duration (days)', 
                       sizes=(20, 200), alpha=0.7, ax=ax)
        
        # Add regression line for each strategy
        strategies = dd_df['Strategy'].unique()
        
        for strategy in strategies:
            strategy_data = dd_df[dd_df['Strategy'] == strategy]
            
            # Only add regression line if we have enough data points
            if len(strategy_data) >= 3:
                sns.regplot(x='Max Drawdown (%)', y='Recovery Time (days)', 
                           data=strategy_data, scatter=False, 
                           ax=ax, line_kws={"linestyle": "--"})
        
        # Set labels and title
        ax.set_title('Drawdown Magnitude vs Recovery Time', fontsize=14)
        ax.set_xlabel('Drawdown (%)', fontsize=12)
        ax.set_ylabel('Recovery Time (trading days)', fontsize=12)
        
        # Invert x-axis (so worse drawdowns are on the right)
        ax.invert_xaxis()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_duration_histogram(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot histogram of drawdown durations.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return
        
        # Prepare data for plotting
        all_data = {
            'Strategy': [],
            'Duration (days)': []
        }
        
        for strategy, drawdowns in self.drawdown_info.items():
            for dd in drawdowns:
                all_data['Strategy'].append(strategy)
                all_data['Duration (days)'].append(dd['duration'])
        
        # Convert to DataFrame
        dd_df = pd.DataFrame(all_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create histogram
        sns.histplot(data=dd_df, x='Duration (days)', hue='Strategy', 
                    kde=True, element='step', ax=ax)
        
        # Set labels and title
        ax.set_title('Drawdown Duration Distribution', fontsize=14)
        ax.set_xlabel('Duration (trading days)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_calendar(self, figsize: Tuple[int, int] = (14, 8)) -> None:
        """
        Plot calendar heatmap of drawdowns.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_series') or self.drawdown_series.empty:
            self.logger.error("Drawdown series not available")
            return
        
        # Create figure with subplots (one per strategy)
        fig, axes = plt.subplots(len(self.drawdown_series.columns), 1, figsize=figsize)
        
        # Handle single strategy case
        if len(self.drawdown_series.columns) == 1:
            axes = [axes]
        
        # Plot calendar heatmap for each strategy
        for i, column in enumerate(self.drawdown_series.columns):
            drawdowns = self.drawdown_series[column]
            
            # Create a pivot table with year and day of year
            drawdowns_pivot = drawdowns.copy()
            drawdowns_pivot.index = pd.MultiIndex.from_arrays([
                drawdowns_pivot.index.year,
                drawdowns_pivot.index.dayofyear
            ])
            
            # Unstack to create a matrix with years as rows and days as columns
            pivot = drawdowns_pivot.unstack(level=0)
            
            # Create heatmap
            sns.heatmap(pivot, cmap="RdYlGn", center=0, linewidths=0.5, 
                       ax=axes[i], cbar_kws={'label': 'Drawdown (%)'})
            
            axes[i].set_title(f'{column} - Daily Drawdowns', fontsize=12)
            axes[i].set_ylabel('Day of Year', fontsize=10)
            
            # Only show x-label for the bottom subplot
            if i == len(self.drawdown_series.columns) - 1:
                axes[i].set_xlabel('Year', fontsize=10)
            else:
                axes[i].set_xlabel('')
        
        plt.tight_layout()
        plt.show()
        
    def plot_comparative_drawdowns(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot comparative analysis of drawdowns across strategies.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return
        
        # Calculate drawdown statistics
        stats_df = self.calculate_drawdown_stats()
        
        if stats_df.empty:
            self.logger.error("No drawdown statistics available")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # 1. Scatter plot of Max Drawdown vs Time in Drawdown
        sns.scatterplot(x='Max Drawdown (%)', y='Time in Drawdown (%)', 
                       data=stats_df, s=100, ax=axes[0])
        
        # Add strategy labels
        for i, row in stats_df.iterrows():
            axes[0].annotate(row['Strategy'], 
                           (row['Max Drawdown (%)'], row['Time in Drawdown (%)']),
                           xytext=(5, 5),
                           textcoords='offset points')
        
        axes[0].set_title('Max Drawdown vs Time in Drawdown', fontsize=12)
        axes[0].set_xlabel('Max Drawdown (%)', fontsize=10)
        axes[0].set_ylabel('Time in Drawdown (%)', fontsize=10)
        axes[0].invert_xaxis()  # Worse drawdowns on the right
        axes[0].grid(True, alpha=0.3)
        
        # 2. Scatter plot of Max Drawdown vs Avg Recovery
        sns.scatterplot(x='Max Drawdown (%)', y='Avg Recovery (days)', 
                       data=stats_df, s=100, ax=axes[1])
        
        # Add strategy labels
        for i, row in stats_df.iterrows():
            axes[1].annotate(row['Strategy'], 
                           (row['Max Drawdown (%)'], row['Avg Recovery (days)']),
                           xytext=(5, 5),
                           textcoords='offset points')
        
        axes[1].set_title('Max Drawdown vs Avg Recovery Time', fontsize=12)
        axes[1].set_xlabel('Max Drawdown (%)', fontsize=10)
        axes[1].set_ylabel('Avg Recovery (days)', fontsize=10)
        axes[1].invert_xaxis()  # Worse drawdowns on the right
        axes[1].grid(True, alpha=0.3)
        
        # 3. Scatter plot of Avg Drawdown vs Total Drawdowns
        sns.scatterplot(x='Avg Drawdown (%)', y='Total Drawdowns', 
                       data=stats_df, s=100, ax=axes[2])
        
        # Add strategy labels
        for i, row in stats_df.iterrows():
            axes[2].annotate(row['Strategy'], 
                           (row['Avg Drawdown (%)'], row['Total Drawdowns']),
                           xytext=(5, 5),
                           textcoords='offset points')
        
        axes[2].set_title('Avg Drawdown vs Total Drawdowns', fontsize=12)
        axes[2].set_xlabel('Avg Drawdown (%)', fontsize=10)
        axes[2].set_ylabel('Total Drawdowns', fontsize=10)
        axes[2].invert_xaxis()  # Worse drawdowns on the right
        axes[2].grid(True, alpha=0.3)
        
        # 4. Scatter plot of Avg Duration vs Max Duration
        sns.scatterplot(x='Avg Duration (days)', y='Max Duration (days)', 
                       data=stats_df, s=100, ax=axes[3])
        
        # Add strategy labels
        for i, row in stats_df.iterrows():
            axes[3].annotate(row['Strategy'], 
                           (row['Avg Duration (days)'], row['Max Duration (days)']),
                           xytext=(5, 5),
                           textcoords='offset points')
        
        axes[3].set_title('Avg Duration vs Max Duration', fontsize=12)
        axes[3].set_xlabel('Avg Duration (days)', fontsize=10)
        axes[3].set_ylabel('Max Duration (days)', fontsize=10)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown_severity_analysis(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot analysis of drawdown severity.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return
        
        # Prepare data for plotting
        all_data = {
            'Strategy': [],
            'Drawdown (%)': [],
            'Category': []
        }
        
        # Define drawdown severity categories
        categories = [
            ('Mild', 0, 5),
            ('Moderate', 5, 10),
            ('Severe', 10, 20),
            ('Extreme', 20, float('inf'))
        ]
        
        for strategy, drawdowns in self.drawdown_info.items():
            for dd in drawdowns:
                dd_value = abs(dd['max_drawdown'])
                
                # Determine category
                for cat_name, cat_min, cat_max in categories:
                    if cat_min <= dd_value < cat_max:
                        category = cat_name
                        break
                else:
                    category = 'Unknown'
                
                all_data['Strategy'].append(strategy)
                all_data['Drawdown (%)'].append(dd_value)
                all_data['Category'].append(category)
        
        # Convert to DataFrame
        dd_df = pd.DataFrame(all_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Count of drawdowns by severity category
        category_counts = dd_df.groupby(['Strategy', 'Category']).size().unstack().fillna(0)
        
        # Ensure all categories are present
        for cat_name, _, _ in categories:
            if cat_name not in category_counts.columns:
                category_counts[cat_name] = 0
        
        # Sort columns by severity
        category_order = [cat[0] for cat in categories]
        category_counts = category_counts[category_order]
        
        # Plot stacked bar chart
        category_counts.plot(kind='bar', stacked=True, ax=axes[0], 
                           colormap='RdYlGn_r')
        
        axes[0].set_title('Drawdown Counts by Severity', fontsize=12)
        axes[0].set_xlabel('Strategy', fontsize=10)
        axes[0].set_ylabel('Count', fontsize=10)
        axes[0].legend(title='Severity')
        
        # 2. Distribution of drawdown magnitudes
        sns.boxplot(x='Strategy', y='Drawdown (%)', data=dd_df, ax=axes[1])
        
        # Add individual points
        sns.stripplot(x='Strategy', y='Drawdown (%)', data=dd_df, 
                     ax=axes[1], color='black', alpha=0.5, size=4)
        
        axes[1].set_title('Drawdown Magnitude Distribution', fontsize=12)
        axes[1].set_xlabel('Strategy', fontsize=10)
        axes[1].set_ylabel('Drawdown (%)', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Return the DataFrame for further analysis
        return dd_df
    
    def plot_drawdown_clustering(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot clustering of drawdowns by duration and magnitude.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not hasattr(self, 'drawdown_info') or not self.drawdown_info:
            self.logger.error("Drawdown information not available")
            return
        
        # Prepare data for clustering
        all_data = {
            'Strategy': [],
            'Max Drawdown (%)': [],
            'Duration (days)': [],
            'Recovery Time (days)': []
        }
        
        for strategy, drawdowns in self.drawdown_info.items():
            for dd in drawdowns:
                all_data['Strategy'].append(strategy)
                all_data['Max Drawdown (%)'].append(abs(dd['max_drawdown']))
                all_data['Duration (days)'].append(dd['duration'])
                all_data['Recovery Time (days)'].append(dd['recovery_time'])
        
        # Convert to DataFrame
        dd_df = pd.DataFrame(all_data)
        
        # Check if we have enough data for clustering
        if len(dd_df) < 5:
            self.logger.warning("Not enough drawdown data for clustering analysis")
            return
        
        try:
            # Standardize the data for clustering
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            
            # Select features for clustering
            features = ['Max Drawdown (%)', 'Duration (days)', 'Recovery Time (days)']
            X = dd_df[features].copy()
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters using elbow method
            from sklearn.cluster import KMeans
            
            inertia = []
            k_range = range(1, min(6, len(dd_df)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Flatten axes for easier iteration
            axes = axes.flatten()
            
            # 1. Elbow method plot
            axes[0].plot(k_range, inertia, 'o-')
            axes[0].set_title('Elbow Method for Optimal k', fontsize=12)
            axes[0].set_xlabel('Number of Clusters (k)', fontsize=10)
            axes[0].set_ylabel('Inertia', fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # Choose number of clusters
            # Simple heuristic: if we have few points, use fewer clusters
            if len(dd_df) < 10:
                n_clusters = 2
            else:
                n_clusters = 3
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            dd_df['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Get cluster centers
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            
            # 2. Scatter plot of Max Drawdown vs Duration
            sns.scatterplot(x='Max Drawdown (%)', y='Duration (days)', 
                           hue='Cluster', style='Strategy', s=100,
                           data=dd_df, ax=axes[1])
            
            # Add cluster centers
            axes[1].scatter(centers[:, 0], centers[:, 1], 
                           marker='X', s=200, c='black', 
                           label='Cluster Centers')
            
            axes[1].set_title('Drawdown Clusters: Magnitude vs Duration', fontsize=12)
            axes[1].set_xlabel('Max Drawdown (%)', fontsize=10)
            axes[1].set_ylabel('Duration (days)', fontsize=10)
            axes[1].grid(True, alpha=0.3)
            
            # 3. Scatter plot of Max Drawdown vs Recovery Time
            sns.scatterplot(x='Max Drawdown (%)', y='Recovery Time (days)', 
                           hue='Cluster', style='Strategy', s=100,
                           data=dd_df, ax=axes[2])
            
            # Add cluster centers
            axes[2].scatter(centers[:, 0], centers[:, 2], 
                           marker='X', s=200, c='black')
            
            axes[2].set_title('Drawdown Clusters: Magnitude vs Recovery', fontsize=12)
            axes[2].set_xlabel('Max Drawdown (%)', fontsize=10)
            axes[2].set_ylabel('Recovery Time (days)', fontsize=10)
            axes[2].grid(True, alpha=0.3)
            
            # 4. Scatter plot of Duration vs Recovery Time
            sns.scatterplot(x='Duration (days)', y='Recovery Time (days)', 
                           hue='Cluster', style='Strategy', s=100,
                           data=dd_df, ax=axes[3])
            
            # Add cluster centers
            axes[3].scatter(centers[:, 1], centers[:, 2], 
                           marker='X', s=200, c='black')
            
            axes[3].set_title('Drawdown Clusters: Duration vs Recovery', fontsize=12)
            axes[3].set_xlabel('Duration (days)', fontsize=10)
            axes[3].set_ylabel('Recovery Time (days)', fontsize=10)
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Return the DataFrame with cluster assignments
            return dd_df
            
        except Exception as e:
            self.logger.error(f"Error in drawdown clustering: {str(e)}")
            return dd_df
    
    def set_equity_data(self, equity_data: Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]]) -> None:
        """
        Set or update the equity data.
        
        Args:
            equity_data: Equity curve data. Can be:
                - pd.Series: Single equity curve
                - pd.DataFrame: Multiple equity curves as columns
                - Dict[str, pd.Series]: Dictionary of named equity curves
        """
        if isinstance(equity_data, pd.Series):
            self.equity_data = pd.DataFrame({equity_data.name or 'Strategy': equity_data})
        elif isinstance(equity_data, pd.DataFrame):
            self.equity_data = equity_data
        elif isinstance(equity_data, dict):
            self.equity_data = pd.DataFrame(equity_data)
        else:
            self.logger.error(f"Unsupported equity data type: {type(equity_data)}")
            return
        
        # Recalculate drawdowns
        self.calculate_drawdowns()