import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import empyrical

class PerformanceAnalyzer:
    """
    PerformanceAnalyzer calculates and visualizes trading strategy performance metrics.
    """
    
    def __init__(self, equity_curve: pd.DataFrame, returns: pd.DataFrame, trades: pd.DataFrame):
        """
        Initialize the performance analyzer.
        
        Args:
            equity_curve: DataFrame with equity curve data
            returns: DataFrame with returns data
            trades: DataFrame with trade data
        """
        self.equity_curve = equity_curve
        self.returns = returns
        self.trades = trades
        
        # Calculate daily returns if not already daily
        if isinstance(returns.index, pd.DatetimeIndex):
            self.daily_returns = returns.resample('D').sum()
        else:
            self.daily_returns = returns
    
    def calculate_metrics(self, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            risk_free_rate: Annual risk-free rate (decimal)
            
        Returns:
            Dictionary of performance metrics
        """
        daily_return_values = self.daily_returns['return'].values
        
        # Basic metrics
        total_return = self.equity_curve['total_value'].iloc[-1] / self.equity_curve['total_value'].iloc[0] - 1
        
        # Annualized return
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        
        # Volatility
        daily_volatility = np.std(daily_return_values)
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # Sharpe ratio
        daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_returns = daily_return_values - daily_risk_free
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = daily_return_values[daily_return_values < 0]
        downside_deviation = np.std(downside_returns)
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + daily_return_values) - 1
        max_return = np.maximum.accumulate(cumulative_returns)
        drawdowns = (1 + cumulative_returns) / (1 + max_return) - 1
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Win rate
        if not self.trades.empty:
            winning_trades = len(self.trades[self.trades['pnl'] > 0])
            total_trades = len(self.trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Average win/loss
            avg_win = self.trades[self.trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = self.trades[self.trades['pnl'] < 0]['pnl'].mean() if total_trades - winning_trades > 0 else 0
            
            # Profit factor
            gross_profit = self.trades[self.trades['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(self.trades[self.trades['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Use empyrical for additional metrics
        try:
            alpha = empyrical.alpha(daily_return_values, daily_risk_free, 0.0)  # Assuming market return of 0
            beta = empyrical.beta(daily_return_values, daily_risk_free)
            omega_ratio = empyrical.omega_ratio(daily_return_values)
        except:
            alpha = 0
            beta = 0
            omega_ratio = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'alpha': alpha,
            'beta': beta,
            'omega_ratio': omega_ratio
        }
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot the equity curve.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        plt.figure(figsize=figsize)
        plt.plot(self.equity_curve.index, self.equity_curve['total_value'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_drawdowns(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot the drawdowns.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        # Calculate drawdowns
        equity = self.equity_curve['total_value']
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        
        plt.figure(figsize=figsize)
        plt.plot(drawdown.index, drawdown * 100)
        plt.title('Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot monthly returns as a heatmap.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        if isinstance(self.returns.index, pd.DatetimeIndex):
            # Resample to monthly returns
            monthly_returns = self.returns['return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Create a pivot table with years as rows and months as columns
            monthly_returns_table = pd.DataFrame({
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month,
                'return': monthly_returns.values
            })
            
            pivot_table = monthly_returns_table.pivot(
                index='year', columns='month', values='return'
            )
            
            # Plot heatmap
            plt.figure(figsize=figsize)
            sns.heatmap(
                pivot_table * 100,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                linewidths=1,
                cbar_kws={'label': 'Monthly Return (%)'}
            )
            plt.title('Monthly Returns (%)')
            plt.xlabel('Month')
            plt.ylabel('Year')
            
            # Set month names as x-axis labels
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.xticks(np.arange(12) + 0.5, month_names)
            
            plt.tight_layout()
            plt.show()
        else:
            print("Returns index is not a DatetimeIndex. Cannot create monthly returns heatmap.")
    
    def plot_return_distribution(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot the distribution of returns.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        plt.figure(figsize=figsize)
        sns.histplot(self.daily_returns['return'] * 100, kde=True)
        plt.title('Daily Returns Distribution')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.show()
    
    def plot_rolling_sharpe(self, window: int = 60, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot rolling Sharpe ratio.
        
        Args:
            window: Rolling window size in days
            figsize: Figure size as (width, height) tuple
        """
        # Calculate rolling Sharpe ratio
        rolling_return = self.daily_returns['return'].rolling(window=window).mean()
        rolling_vol = self.daily_returns['return'].rolling(window=window).std()
        rolling_sharpe = rolling_return / rolling_vol * np.sqrt(252)
        
        plt.figure(figsize=figsize)
        plt.plot(rolling_sharpe.index, rolling_sharpe)
        plt.title(f'Rolling {window}-Day Sharpe Ratio')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.show()
    
    def plot_trade_analysis(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot trade analysis charts.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        if self.trades.empty:
            print("No trade data available for analysis.")
            return
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Trade P&L distribution
        sns.histplot(self.trades['pnl'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Trade P&L Distribution')
        axes[0, 0].set_xlabel('P&L')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(x=0, color='r', linestyle='--')
        
        # 2. Cumulative P&L
        cumulative_pnl = self.trades['pnl'].cumsum()
        axes[0, 1].plot(cumulative_pnl.index, cumulative_pnl)
        axes[0, 1].set_title('Cumulative P&L')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Cumulative P&L')
        axes[0, 1].grid(True)
        
        # 3. Win/Loss by instrument
        if 'instrument' in self.trades.columns:
            instrument_performance = self.trades.groupby('instrument')['pnl'].agg(['sum', 'count'])
            instrument_performance['avg_pnl'] = instrument_performance['sum'] / instrument_performance['count']
            instrument_performance = instrument_performance.sort_values('sum', ascending=False)
            
            instrument_performance['sum'].plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('P&L by Instrument')
            axes[1, 0].set_xlabel('Instrument')
            axes[1, 0].set_ylabel('Total P&L')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Trade duration analysis
        if 'duration' in self.trades.columns:
            sns.boxplot(x='side', y='duration', data=self.trades, ax=axes[1, 1])
            axes[1, 1].set_title('Trade Duration by Side')
            axes[1, 1].set_xlabel('Side')
            axes[1, 1].set_ylabel('Duration (days)')
        
        plt.tight_layout()
        plt.show()
    
    def print_performance_summary(self, risk_free_rate: float = 0.0) -> None:
        """
        Print a summary of performance metrics.
        
        Args:
            risk_free_rate: Annual risk-free rate (decimal)
        """
        metrics = self.calculate_metrics(risk_free_rate)
        
        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
        print(f"Annualized Return: {metrics['annualized_return'] * 100:.2f}%")
        print(f"Annualized Volatility: {metrics['annualized_volatility'] * 100:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate'] * 100:.2f}%")
        print(f"Average Win: {metrics['avg_win']:.2f}")
        print(f"Average Loss: {metrics['avg_loss']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Alpha: {metrics['alpha']:.4f}")
        print(f"Beta: {metrics['beta']:.4f}")
        print(f"Omega Ratio: {metrics['omega_ratio']:.2f}")
        
        if not self.trades.empty:
            print("\nTRADE STATISTICS")
            print(f"Total Trades: {len(self.trades)}")
            print(f"Winning Trades: {len(self.trades[self.trades['pnl'] > 0])}")
            print(f"Losing Trades: {len(self.trades[self.trades['pnl'] < 0])}")
            
            if 'duration' in self.trades.columns:
                avg_duration = self.trades['duration'].mean()
                print(f"Average Trade Duration: {avg_duration:.2f} days")
        
        print("=" * 50)
    
    def generate_report(self, filename: str = 'performance_report.html', risk_free_rate: float = 0.0) -> None:
        """
        Generate an HTML performance report.
        
        Args:
            filename: Output HTML filename
            risk_free_rate: Annual risk-free rate (decimal)
        """
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import plotly.figure_factory as ff
        from jinja2 import Template
        
        metrics = self.calculate_metrics(risk_free_rate)
        
        # Create equity curve figure
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            x=self.equity_curve.index,
            y=self.equity_curve['total_value'],
            mode='lines',
            name='Portfolio Value'
        ))
        equity_fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            template='plotly_white'
        )
        
        # Create drawdown figure
        equity = self.equity_curve['total_value']
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        
        drawdown_fig = go.Figure()
        drawdown_fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red')
        ))
        drawdown_fig.update_layout(
            title='Drawdowns',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white'
        )
        
        # Create returns distribution figure
        returns_fig = px.histogram(
            self.daily_returns['return'] * 100,
            nbins=50,
            marginal='box',
            title='Daily Returns Distribution',
            labels={'value': 'Daily Return (%)'},
            template='plotly_white'
        )
        returns_fig.add_vline(x=0, line_dash='dash', line_color='red')
        
        # Create monthly returns heatmap
        if isinstance(self.returns.index, pd.DatetimeIndex):
            # Resample to monthly returns
            monthly_returns = self.returns['return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Create a pivot table with years as rows and months as columns
            monthly_returns_table = pd.DataFrame({
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month,
                'return': monthly_returns.values
            })
            
            pivot_table = monthly_returns_table.pivot(
                index='year', columns='month', values='return'
            )
            
            # Convert to percentage
            pivot_table = pivot_table * 100
            
            # Create heatmap
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            monthly_fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=month_names,
                y=pivot_table.index,
                colorscale='RdYlGn',
                zmid=0,
                text=[[f'{val:.2f}%' for val in row] for row in pivot_table.values],
                texttemplate='%{text}',
                colorbar=dict(title='Return (%)')
            ))
            
            monthly_fig.update_layout(
                title='Monthly Returns (%)',
                xaxis_title='Month',
                yaxis_title='Year',
                template='plotly_white'
            )
        else:
            monthly_fig = None
        
        # Create trade analysis figures if trade data is available
        if not self.trades.empty:
            # Trade P&L distribution
            trade_pnl_fig = px.histogram(
                self.trades,
                x='pnl',
                nbins=50,
                marginal='box',
                title='Trade P&L Distribution',
                template='plotly_white'
            )
            trade_pnl_fig.add_vline(x=0, line_dash='dash', line_color='red')
            
            # Cumulative P&L
            cumulative_pnl = self.trades['pnl'].cumsum()
            cum_pnl_fig = go.Figure()
            cum_pnl_fig.add_trace(go.Scatter(
                x=list(range(len(cumulative_pnl))),
                y=cumulative_pnl.values,
                mode='lines',
                name='Cumulative P&L'
            ))
            cum_pnl_fig.update_layout(
                title='Cumulative P&L',
                xaxis_title='Trade Number',
                yaxis_title='Cumulative P&L',
                template='plotly_white'
            )
            
            # P&L by instrument
            if 'instrument' in self.trades.columns:
                instrument_performance = self.trades.groupby('instrument')['pnl'].agg(['sum', 'count'])
                instrument_performance['avg_pnl'] = instrument_performance['sum'] / instrument_performance['count']
                instrument_performance = instrument_performance.sort_values('sum', ascending=False)
                
                instrument_fig = px.bar(
                    x=instrument_performance.index,
                    y=instrument_performance['sum'],
                    title='P&L by Instrument',
                    labels={'x': 'Instrument', 'y': 'Total P&L'},
                    template='plotly_white'
                )
            else:
                instrument_fig = None
            
            # Trade duration analysis
            if 'duration' in self.trades.columns and 'side' in self.trades.columns:
                duration_fig = px.box(
                    self.trades,
                    x='side',
                    y='duration',
                    title='Trade Duration by Side',
                    labels={'side': 'Side', 'duration': 'Duration (days)'},
                    template='plotly_white'
                )
            else:
                duration_fig = None
        else:
            trade_pnl_fig = None
            cum_pnl_fig = None
            instrument_fig = None
            duration_fig = None
        
        # Create HTML template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Performance Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1, h2 {
                    color: #333;
                }
                .metrics-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                .metrics-table th, .metrics-table td {
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }
                .metrics-table th {
                    background-color: #f2f2f2;
                }
                .plot-container {
                    margin: 20px 0;
                }
                .row {
                    display: flex;
                    flex-wrap: wrap;
                    margin: 0 -10px;
                }
                .col {
                    flex: 1;
                    padding: 0 10px;
                    min-width: 300px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Trading Strategy Performance Report</h1>
                <p>Generated on {{ generation_date }}</p>
                
                <h2>Performance Metrics</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Return</td>
                        <td>{{ metrics.total_return|round(4) * 100 }}%</td>
                    </tr>
                    <tr>
                        <td>Annualized Return</td>
                        <td>{{ metrics.annualized_return|round(4) * 100 }}%</td>
                    </tr>
                    <tr>
                        <td>Annualized Volatility</td>
                        <td>{{ metrics.annualized_volatility|round(4) * 100 }}%</td>
                    </tr>
                    <tr>
                        <td>Sharpe Ratio</td>
                        <td>{{ metrics.sharpe_ratio|round(2) }}</td>
                    </tr>
                    <tr>
                        <td>Sortino Ratio</td>
                        <td>{{ metrics.sortino_ratio|round(2) }}</td>
                    </tr>
                    <tr>
                        <td>Maximum Drawdown</td>
                        <td>{{ metrics.max_drawdown|round(4) * 100 }}%</td>
                    </tr>
                    <tr>
                        <td>Calmar Ratio</td>
                        <td>{{ metrics.calmar_ratio|round(2) }}</td>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td>{{ metrics.win_rate|round(4) * 100 }}%</td>
                    </tr>
                    <tr>
                        <td>Profit Factor</td>
                        <td>{{ metrics.profit_factor|round(2) }}</td>
                    </tr>
                    <tr>
                        <td>Alpha</td>
                        <td>{{ metrics.alpha|round(4) }}</td>
                    </tr>
                    <tr>
                        <td>Beta</td>
                        <td>{{ metrics.beta|round(4) }}</td>
                    </tr>
                    <tr>
                        <td>Omega Ratio</td>
                        <td>{{ metrics.omega_ratio|round(2) }}</td>
                    </tr>
                </table>
                
                <h2>Equity and Drawdowns</h2>
                <div class="row">
                    <div class="col">
                        <div class="plot-container" id="equity-plot"></div>
                    </div>
                    <div class="col">
                        <div class="plot-container" id="drawdown-plot"></div>
                    </div>
                </div>
                
                <h2>Returns Analysis</h2>
                <div class="row">
                    <div class="col">
                        <div class="plot-container" id="returns-dist-plot"></div>
                    </div>
                    <div class="col">
                        {% if monthly_fig %}
                        <div class="plot-container" id="monthly-returns-plot"></div>
                        {% endif %}
                    </div>
                </div>
                
                {% if not trades.empty %}
                <h2>Trade Analysis</h2>
                <div class="row">
                    <div class="col">
                        <div class="plot-container" id="trade-pnl-plot"></div>
                    </div>
                    <div class="col">
                        <div class="plot-container" id="cum-pnl-plot"></div>
                    </div>
                </div>
                
                <div class="row">
                    {% if instrument_fig %}
                    <div class="col">
                        <div class="plot-container" id="instrument-plot"></div>
                    </div>
                    {% endif %}
                    
                    {% if duration_fig %}
                    <div class="col">
                        <div class="plot-container" id="duration-plot"></div>
                    </div>
                    {% endif %}
                </div>
                {% endif %}
            </div>
            
            <script>
                // Equity curve plot
                var equityData = {{ equity_json }};
                Plotly.newPlot('equity-plot', equityData.data, equityData.layout);
                
                // Drawdown plot
                var drawdownData = {{ drawdown_json }};
                Plotly.newPlot('drawdown-plot', drawdownData.data, drawdownData.layout);
                
                // Returns distribution plot
                var returnsData = {{ returns_json }};
                Plotly.newPlot('returns-dist-plot', returnsData.data, returnsData.layout);
                
                {% if monthly_fig %}
                // Monthly returns plot
                var monthlyData = {{ monthly_json }};
                Plotly.newPlot('monthly-returns-plot', monthlyData.data, monthlyData.layout);
                {% endif %}
                
                {% if not trades.empty %}
                // Trade P&L distribution plot
                var tradePnlData = {{ trade_pnl_json }};
                Plotly.newPlot('trade-pnl-plot', tradePnlData.data, tradePnlData.layout);
                
                // Cumulative P&L plot
                var cumPnlData = {{ cum_pnl_json }};
                Plotly.newPlot('cum-pnl-plot', cumPnlData.data, cumPnlData.layout);
                
                {% if instrument_fig %}
                // Instrument P&L plot
                var instrumentData = {{ instrument_json }};
                Plotly.newPlot('instrument-plot', instrumentData.data, instrumentData.layout);
                {% endif %}
                
                {% if duration_fig %}
                // Duration plot
                var durationData = {{ duration_json }};
                Plotly.newPlot('duration-plot', durationData.data, durationData.layout);
                {% endif %}
                {% endif %}
            </script>
        </body>
        </html>
        """
        
        # Render template with data
        template = Template(template_str)
        
        html_content = template.render(
            generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            metrics=metrics,
            trades=self.trades,
            equity_json=equity_fig.to_json(),
            drawdown_json=drawdown_fig.to_json(),
            returns_json=returns_fig.to_json(),
            monthly_json=monthly_fig.to_json() if monthly_fig else None,
            trade_pnl_json=trade_pnl_fig.to_json() if trade_pnl_fig else None,
            cum_pnl_json=cum_pnl_fig.to_json() if cum_pnl_fig else None,
            instrument_json=instrument_fig.to_json() if instrument_fig else None,
            duration_json=duration_fig.to_json() if duration_fig else None
        )
        
        # Write to file
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"Performance report generated: {filename}")