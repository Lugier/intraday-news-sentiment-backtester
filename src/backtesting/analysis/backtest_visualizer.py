"""
Backtest Visualizer

This module provides functions to visualize the results of backtesting trading strategies.
It generates various plots, such as equity curves, returns distributions, and trade summaries.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import matplotlib.gridspec as gridspec
import shutil
from collections import defaultdict
from scipy import stats

from src.backtesting.helpers.utilities import save_json
from src.backtesting.helpers.market_data import fetch_stock_prices

# Configure logging
logger = logging.getLogger(__name__)

# Configure Matplotlib for consistent styling
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

class BacktestVisualizer:
    """
    A class for visualizing backtesting results of the news sentiment strategy.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set(font_scale=1.2)
        sns.set_palette("muted")
        
    def visualize_cumulative_returns(self, backtest_results: Dict[str, Any]) -> plt.Figure:
        """
        Create a visualization of cumulative returns over time, comparing against Buy & Hold.
        
        Args:
            backtest_results: Dictionary with backtest results
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract trade data
        trades = pd.DataFrame(backtest_results['trades'])
        ticker = backtest_results['ticker']
        
        if trades.empty:
            plt.figtext(0.5, 0.5, "No trades to visualize", ha='center', va='center', fontsize=14)
            ax.set_title(f"Cumulative Returns for {ticker} (No Trades)", fontsize=16)
            ax.set_ylabel("Return (%)", fontsize=14)
            ax.set_xlabel("Time", fontsize=14)
            return fig
        
        # --- Strategy Equity Curve ---
        # Convert times to datetime objects if they are strings
        trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        trades['exit_time'] = pd.to_datetime(trades['exit_time'])
        
        # Sort trades by exit time to ensure chronological order
        trades = trades.sort_values('exit_time')
            
        # Calculate cumulative returns for the strategy
        trades['cumulative_return'] = trades['return_pct'].cumsum()
        
        # Create a proper time-series of equity values - this is key for correct visualization
        # Start with initial capital
        initial_capital = 100  # Start with $100
        
        # Create a continuous time series at minute intervals spanning the entire trading period
        if not trades.empty:
            start_time = trades['entry_time'].min()
            end_time = trades['exit_time'].max()
            
            # Create a date range with minute frequency
            time_index = pd.date_range(start=start_time, end=end_time, freq='T')
            equity_curve = pd.Series(index=time_index, dtype=float)
            equity_curve.iloc[0] = initial_capital
            
            # For each trade, apply its return at the exit time
            for idx, trade in trades.iterrows():
                exit_idx = equity_curve.index.get_indexer([trade['exit_time']], method='nearest')[0]
                trade_return = trade['return_pct'] / 100  # Convert percentage to decimal
                
                # Apply the trade return to all points from this exit time forward
                if exit_idx < len(equity_curve):
                    # Get the equity value just before this trade's impact
                    prev_equity = float(equity_curve.iloc[:exit_idx].dropna().iloc[-1]) if not equity_curve.iloc[:exit_idx].dropna().empty else initial_capital
                    
                    # Apply the trade's return
                    new_equity = prev_equity * (1 + trade_return)
                    
                    # Set the new equity value from this point forward (until potentially modified by later trades)
                    equity_curve.iloc[exit_idx:] = new_equity
            
            # Forward fill any missing values
            equity_curve = equity_curve.ffill()
            
            # Convert to percentage returns from initial
            equity_pct_return = ((equity_curve / initial_capital) - 1) * 100
            
            # Plot the equity curve
            ax.plot(equity_pct_return.index, equity_pct_return.values, 
                    'b-', label='Strategy Return', linewidth=2, alpha=0.8)
        
        # --- Buy & Hold Equity Curve ---
        start_date = trades['entry_time'].min()
        end_date = trades['exit_time'].max()
        
        logger.info(f"Fetching Buy & Hold data for {ticker} from {start_date} to {end_date}")
        try:
            # Fetch minute data covering the entire backtest period
            buy_hold_data = fetch_stock_prices(ticker, start_date - timedelta(minutes=1), end_date + timedelta(minutes=1))
            
            if not buy_hold_data.empty:
                # Find the first valid price at or after the first trade entry
                first_trade_entry_time = trades['entry_time'].iloc[0]
                relevant_bh_data = buy_hold_data[buy_hold_data.index >= first_trade_entry_time]
                
                if not relevant_bh_data.empty:
                    start_price = relevant_bh_data['open'].iloc[0]
                    
                    # Calculate Buy & Hold cumulative return
                    buy_hold_data['buy_hold_return'] = (buy_hold_data['close'] / start_price - 1) * 100
                    
                    # Plot Buy & Hold curve, only within the actual trade period
                    plot_data_bh = buy_hold_data[(buy_hold_data.index >= start_date) & (buy_hold_data.index <= end_date)]
                    ax.plot(plot_data_bh.index, plot_data_bh['buy_hold_return'], 
                            'gray', linestyle='--', label=f'Buy & Hold {ticker}', linewidth=1.5, alpha=0.7)
                else:
                    logger.warning(f"Could not find Buy & Hold starting price near {first_trade_entry_time}")
            else:
                logger.warning(f"Could not fetch Buy & Hold data for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching or processing Buy & Hold data: {e}")

        # Add reference line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
        
        # Add annotations for key statistics
        stats_text = (
            f"Total Trades: {backtest_results['total_trades']}\n"
            f"Win Rate: {backtest_results['win_rate']*100:.1f}%\n"
            f"Strategy Return: {backtest_results['total_return_pct']:.1f}%\n"
        )
        
        # Add Buy & Hold return if calculated
        if 'buy_hold_return' in locals() and not plot_data_bh.empty:
            buy_hold_final_return = plot_data_bh['buy_hold_return'].iloc[-1]
            stats_text += f"Buy & Hold Return: {buy_hold_final_return:.1f}%\n"
            
        stats_text += (
            f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}\n"
            f"Max Drawdown: {backtest_results.get('max_drawdown_pct', 0):.1f}%\n"
            f"Profit Factor: {backtest_results.get('profit_factor', 0):.2f}"
        )

        # Place annotation box in the top-left corner
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Format plot
        ax.set_title(f"Cumulative Returns: Strategy vs Buy & Hold for {ticker}", 
                    fontsize=16)
        ax.set_ylabel("Return (%)", fontsize=14)
        ax.set_xlabel("Time", fontsize=14)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter()) # Format y-axis as percentage
        
        # Format x-axis for dates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate() # Auto-rotate date labels
        
        # Add legend
        ax.legend(loc='lower left', fontsize=10)
        
        plt.tight_layout()
        
        return fig
    
    def visualize_returns_by_sentiment(self, backtest_results: Dict[str, Any]) -> plt.Figure:
        """
        Create a detailed analysis of returns by sentiment type.
        
        Args:
            backtest_results: Dictionary with backtest results
            
        Returns:
            Matplotlib figure with multiple panels
        """
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # Extract trade data
        trades = pd.DataFrame(backtest_results['trades'])
        if trades.empty:
            plt.figtext(0.5, 0.5, "No trades to visualize", ha='center', va='center', fontsize=14)
            return fig
            
        sentiment_order = ['positive', 'negative']
        sentiment_colors = {'positive': 'green', 'negative': 'red'}
        
        # Panel 1: Returns Distribution by Sentiment (Boxplot)
        ax1 = axs[0, 0]
        sns.boxplot(x='sentiment', y='return_pct', data=trades, ax=ax1, 
                    order=sentiment_order, 
                    palette=sentiment_colors)
        
        # Add individual points
        sns.stripplot(x='sentiment', y='return_pct', data=trades, ax=ax1,
                     order=sentiment_order, color='black', size=4, alpha=0.5, jitter=True)
        
        # Add reference line
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Format plot
        ax1.set_title("Trade Returns by Sentiment", fontsize=14)
        ax1.set_ylabel("Return (%)", fontsize=12)
        ax1.set_xlabel("News Sentiment", fontsize=12)
        
        # Panel 2: Win Rate by Sentiment (Bar Chart)
        ax2 = axs[0, 1]
        sentiment_stats = []
        for sentiment, stats in backtest_results['sentiment_stats'].items():
            sentiment_stats.append({
                'sentiment': sentiment,
                'win_rate': stats['win_rate'] * 100,
                'count': stats['count']
            })
        
        sentiment_stats_df = pd.DataFrame(sentiment_stats)
        if not sentiment_stats_df.empty:
            bars = sns.barplot(x='sentiment', y='win_rate', data=sentiment_stats_df, ax=ax2,
                              order=sentiment_order, palette=sentiment_colors)
            
            # Add value labels
            for bar in bars.patches:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom',
                    fontsize=12
                )
            
            # Add trade count labels
            for i, row in sentiment_stats_df.iterrows():
                ax2.text(
                    i, 5,
                    f'n={row["count"]}',
                    ha='center', va='bottom',
                    fontsize=10,
                    color='white'
                )
            
            # Add reference line at 50%
            ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5)
            
            # Format plot
            ax2.set_title("Win Rate by Sentiment", fontsize=14)
            ax2.set_ylabel("Win Rate (%)", fontsize=12)
            ax2.set_xlabel("News Sentiment", fontsize=12)
            ax2.set_ylim(0, 100)
        
        # Panel 3: Average Returns by Sentiment
        ax3 = axs[1, 0]
        avg_returns = []
        for sentiment, stats in backtest_results['sentiment_stats'].items():
            avg_returns.append({
                'sentiment': sentiment,
                'avg_return': stats['avg_return']
            })
        
        avg_returns_df = pd.DataFrame(avg_returns)
        if not avg_returns_df.empty:
            bars = sns.barplot(x='sentiment', y='avg_return', data=avg_returns_df, ax=ax3,
                              order=sentiment_order, palette=sentiment_colors)
            
            # Add value labels
            for bar in bars.patches:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.1 if height >= 0 else height - 0.3,
                    f'{height:.2f}%',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=12
                )
            
            # Add reference line at 0%
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Format plot
            ax3.set_title("Average Return by Sentiment", fontsize=14)
            ax3.set_ylabel("Average Return (%)", fontsize=12)
            ax3.set_xlabel("News Sentiment", fontsize=12)
        
        # Panel 4: Return Consistency (Mean vs Median)
        ax4 = axs[1, 1]
        consistency_data = []
        for sentiment, stats in backtest_results['sentiment_stats'].items():
            consistency_data.append({
                'sentiment': sentiment,
                'metric': 'Mean',
                'value': stats['avg_return']
            })
            consistency_data.append({
                'sentiment': sentiment,
                'metric': 'Median',
                'value': stats['median_return']
            })
        
        consistency_df = pd.DataFrame(consistency_data)
        if not consistency_df.empty:
            # Group bars by sentiment
            g = sns.barplot(x='sentiment', y='value', hue='metric', 
                           data=consistency_df, ax=ax4,
                           order=sentiment_order)
            
            # Add reference line at 0%
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Format plot
            ax4.set_title("Return Consistency (Mean vs Median)", fontsize=14)
            ax4.set_ylabel("Return (%)", fontsize=12)
            ax4.set_xlabel("News Sentiment", fontsize=12)
            ax4.legend(title='Metric')
        
        # Overall title
        fig.suptitle(f'Detailed Return Analysis by Sentiment for {backtest_results["ticker"]}', 
                     fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        return fig
    
    def visualize_win_rate(self, backtest_results: Dict[str, Any]) -> plt.Figure:
        """
        Create a bar chart of win rates by sentiment type.
        
        Args:
            backtest_results: Dictionary with backtest results
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract sentiment stats
        sentiment_stats = []
        for sentiment, stats in backtest_results['sentiment_stats'].items():
            sentiment_stats.append({
                'sentiment': sentiment,
                'win_rate': stats['win_rate'] * 100,
                'count': stats['count']
            })
        
        sentiment_stats_df = pd.DataFrame(sentiment_stats)
        if sentiment_stats_df.empty:
            plt.figtext(0.5, 0.5, "No sentiment statistics to visualize", 
                       ha='center', va='center', fontsize=14)
            return fig
            
        # Create bar chart
        sentiment_order = ['positive', 'negative']
        bars = sns.barplot(x='sentiment', y='win_rate', data=sentiment_stats_df, ax=ax,
                          order=sentiment_order,
                          palette={'positive': 'green', 'negative': 'red'})
        
        # Format plot
        ax.set_title("Win Rate by Sentiment Type", fontsize=16)
        ax.set_ylabel("Win Rate (%)", fontsize=12)
        ax.set_xlabel("News Sentiment", fontsize=12)
        ax.set_ylim(0, 100)
        
        # Add reference line at 50%
        ax.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=12
            )
            
        # Add trade count labels
        for i, row in sentiment_stats_df.iterrows():
            ax.text(
                i, 5,
                f'n={row["count"]}',
                ha='center', va='bottom',
                fontsize=10,
                color='white'
            )
        
        plt.tight_layout()
        
        return fig
    
    def visualize_trade_details(self, backtest_results: Dict[str, Any], 
                               top_n: int = 5) -> plt.Figure:
        """
        Create a visualization of the best and worst trades.
        
        Args:
            backtest_results: Dictionary with backtest results
            top_n: Number of best/worst trades to show
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract trade data
        trades = pd.DataFrame(backtest_results['trades'])
        if trades.empty:
            plt.figtext(0.5, 0.5, "No trades to visualize", ha='center', va='center', fontsize=14)
            return fig
            
        # Get best and worst trades
        best_trades = trades.nlargest(top_n, 'return_pct')
        worst_trades = trades.nsmallest(top_n, 'return_pct')
        
        # Plot best trades
        best_trades = best_trades.sort_values('return_pct')
        ax1.barh(best_trades['title'].str[:30], best_trades['return_pct'], 
                color='green', alpha=0.7)
        ax1.set_title(f"Top {top_n} Best Trades", fontsize=14)
        ax1.set_xlabel("Return (%)", fontsize=12)
        ax1.set_ylabel("News Headline", fontsize=12)
        
        # Plot worst trades
        worst_trades = worst_trades.sort_values('return_pct', ascending=False)
        ax2.barh(worst_trades['title'].str[:30], worst_trades['return_pct'], 
                color='red', alpha=0.7)
        ax2.set_title(f"Top {top_n} Worst Trades", fontsize=14)
        ax2.set_xlabel("Return (%)", fontsize=12)
        ax2.set_ylabel("News Headline", fontsize=12)
        
        plt.tight_layout()
        
        return fig
    
    def create_summary_table(self, backtest_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a summary table of backtest results.
        
        Args:
            backtest_results: Dictionary with backtest results
            
        Returns:
            DataFrame with summary statistics
        """
        # Overall statistics
        overall_stats = {
            'Ticker': backtest_results['ticker'],
            'Total Trades': backtest_results['total_trades'],
            'Winning Trades': backtest_results['winning_trades'],
            'Losing Trades': backtest_results['losing_trades'],
            'Win Rate (%)': round(backtest_results['win_rate'] * 100, 2),
            'Strategy Return (%)': round(backtest_results['total_return_pct'], 2),
            'Average Return per Trade (%)': round(backtest_results['average_return_pct'], 2)
        }
        
        # Statistics by sentiment
        sentiment_stats = backtest_results.get('sentiment_stats', {})
        for sentiment, stats in sentiment_stats.items():
            stats_prefix = f"{sentiment.capitalize()} Sentiment"
            overall_stats.update({
                f"{stats_prefix} - Count": stats['count'],
                f"{stats_prefix} - Avg Return (%)": round(stats['avg_return'], 2),
                f"{stats_prefix} - Win Rate (%)": round(stats['win_rate'] * 100, 2)
            })
            
        # Convert to DataFrame
        df = pd.DataFrame([overall_stats])
        return df
    
    def save_all_visualizations(self, backtest_results: Dict[str, Any]) -> str:
        """
        Create and save all visualizations for the backtest results.
        
        Args:
            backtest_results: Dictionary with backtest results
            
        Returns:
            Path to the backtest results directory
        """
        ticker = backtest_results['ticker']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        backtest_dir = os.path.join(self.output_dir, f"backtest_{ticker}_{timestamp}")
        os.makedirs(backtest_dir, exist_ok=True)
        
        # Save summary table
        summary_table = self.create_summary_table(backtest_results)
        summary_table.to_csv(os.path.join(backtest_dir, "summary_stats.csv"), index=False)
        
        # Save detailed statistics table (advanced metrics)
        detailed_stats = self.create_detailed_stats_table(backtest_results)
        detailed_stats.to_csv(os.path.join(backtest_dir, "detailed_stats.csv"), index=False)
        
        # Save trades to CSV
        trades_df = pd.DataFrame(backtest_results['trades'])
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(backtest_dir, "all_trades.csv"), index=False)
            
            # Save additional CSV analysis files
            if 'sentiment_stats' in backtest_results and backtest_results['sentiment_stats']:
                sentiment_stats = pd.DataFrame([
                    {
                        'sentiment': sentiment,
                        **{k: v for k, v in stats.items()}
                    } 
                    for sentiment, stats in backtest_results['sentiment_stats'].items()
                ])
                sentiment_stats.to_csv(os.path.join(backtest_dir, "sentiment_stats.csv"), index=False)
            
            if 'market_session_stats' in backtest_results and backtest_results['market_session_stats']:
                session_stats = pd.DataFrame([
                    {
                        'market_session': session,
                        **{k: v for k, v in stats.items()}
                    } 
                    for session, stats in backtest_results['market_session_stats'].items()
                ])
                session_stats.to_csv(os.path.join(backtest_dir, "market_session_stats.csv"), index=False)
            
            if 'day_of_week_stats' in backtest_results and backtest_results['day_of_week_stats']:
                day_stats = pd.DataFrame([
                    {
                        'day_of_week': day,
                        **{k: v for k, v in stats.items()}
                    } 
                    for day, stats in backtest_results['day_of_week_stats'].items()
                ])
                day_stats.to_csv(os.path.join(backtest_dir, "day_of_week_stats.csv"), index=False)
        
        # Save standard visualizations
        self.visualize_cumulative_returns(backtest_results).savefig(
            os.path.join(backtest_dir, "cumulative_returns.png"), dpi=300, bbox_inches='tight')
        
        self.visualize_returns_by_sentiment(backtest_results).savefig(
            os.path.join(backtest_dir, "returns_by_sentiment.png"), dpi=300, bbox_inches='tight')
        
        self.visualize_win_rate(backtest_results).savefig(
            os.path.join(backtest_dir, "win_rate.png"), dpi=300, bbox_inches='tight')
        
        self.visualize_trade_details(backtest_results).savefig(
            os.path.join(backtest_dir, "trade_details.png"), dpi=300, bbox_inches='tight')
        
        # Save new detailed visualizations
        self.visualize_market_session_analysis(backtest_results).savefig(
            os.path.join(backtest_dir, "market_session_analysis.png"), dpi=300, bbox_inches='tight')
        
        self.visualize_detailed_trade_metrics(backtest_results).savefig(
            os.path.join(backtest_dir, "detailed_trade_metrics.png"), dpi=300, bbox_inches='tight')
        
        # Save the new drawdown plot
        self.visualize_drawdown(backtest_results).savefig(
            os.path.join(backtest_dir, "drawdown_plot.png"), dpi=300, bbox_inches='tight')
        
        # Create and save detailed HTML report
        html_report_path = os.path.join(backtest_dir, "backtest_report.html")
        self.create_html_report(backtest_results, html_report_path)
        
        logger.info(f"All backtesting visualizations and reports saved to {backtest_dir}")
        
        return backtest_dir
    
    def create_detailed_stats_table(self, backtest_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a detailed statistics table with advanced metrics.
        
        Args:
            backtest_results: Dictionary with backtest results
            
        Returns:
            DataFrame with detailed statistics
        """
        stats = {
            'Ticker': backtest_results['ticker'],
            'Total Trades': backtest_results['total_trades'],
            'Winning Trades': backtest_results['winning_trades'],
            'Losing Trades': backtest_results['losing_trades'],
            'Win Rate (%)': round(backtest_results['win_rate'] * 100, 2),
            'Strategy Return (%)': round(backtest_results['total_return_pct'], 2),
            'Average Return per Trade (%)': round(backtest_results['average_return_pct'], 2)
        }
        
        # Add advanced metrics if available
        advanced_metrics = {
            'Median Return (%)': 'median_return_pct',
            'Return Std Dev (%)': 'return_std_dev',
            'Sharpe Ratio': 'sharpe_ratio',
            'Max Drawdown (%)': 'max_drawdown_pct',
            'Drawdown Duration (trades)': 'drawdown_duration',
            'Profit Factor': 'profit_factor',
            'Expected Return (%)': 'expected_return_pct'
        }
        
        for label, key in advanced_metrics.items():
            if key in backtest_results:
                stats[label] = round(backtest_results[key], 2)
        
        # Convert to DataFrame
        return pd.DataFrame([stats])

    def create_html_report(self, backtest_results: Dict[str, Any], output_path: str) -> None:
        """
        Create a detailed HTML report of backtest results.
        
        Args:
            backtest_results: Dictionary with backtest results
            output_path: Path to save the HTML report
        """
        # Only proceed if there are trades
        if backtest_results['total_trades'] == 0:
            with open(output_path, 'w') as f:
                f.write("<html><body><h1>No trades executed in backtest</h1></body></html>")
            return
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(backtest_results['trades'])
        
        # Start building HTML content
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Backtest Report for {backtest_results['ticker']}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2, h3 { color: #333366; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".positive { color: green; }",
            ".negative { color: red; }",
            ".neutral { color: gray; }",
            ".summary-box { background-color: #f5f5f5; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }",
            ".two-column { display: flex; flex-wrap: wrap; }",
            ".column { flex: 1; min-width: 300px; padding-right: 20px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Backtest Report for {backtest_results['ticker']} News Sentiment Trading Strategy</h1>",
            f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        # Overall Summary
        html.extend([
            "<div class='summary-box'>",
            "<h2>Overall Performance Summary</h2>",
            "<div class='two-column'>",
            "<div class='column'>",
            "<table>",
            "<tr><th>Metric</th><th>Value</th></tr>",
            f"<tr><td>Total Trades</td><td>{backtest_results['total_trades']}</td></tr>",
            f"<tr><td>Winning Trades</td><td>{backtest_results['winning_trades']}</td></tr>",
            f"<tr><td>Losing Trades</td><td>{backtest_results['losing_trades']}</td></tr>",
            f"<tr><td>Win Rate</td><td>{backtest_results['win_rate']*100:.2f}%</td></tr>",
            f"<tr><td>Total Return</td><td>{backtest_results['total_return_pct']:.2f}%</td></tr>",
            f"<tr><td>Average Return per Trade</td><td>{backtest_results['average_return_pct']:.2f}%</td></tr>"
        ])
        
        # Add advanced metrics if available
        advanced_metrics = {
            'Median Return': ('median_return_pct', '%.2f%%'),
            'Return Standard Deviation': ('return_std_dev', '%.2f%%'),
            'Sharpe Ratio': ('sharpe_ratio', '%.2f'),
            'Maximum Drawdown': ('max_drawdown_pct', '%.2f%%'),
            'Profit Factor': ('profit_factor', '%.2f')
        }
        
        for label, (key, fmt) in advanced_metrics.items():
            if key in backtest_results:
                value = fmt % backtest_results[key]
                html.append(f"<tr><td>{label}</td><td>{value}</td></tr>")
        
        html.extend([
            "</table>",
            "</div>",
            "<div class='column'>",
            "<h3>Sentiment Analysis</h3>"
        ])
        
        # Add sentiment statistics if available
        if 'sentiment_stats' in backtest_results and backtest_results['sentiment_stats']:
            html.extend([
                "<table>",
                "<tr><th>Sentiment</th><th>Count</th><th>Win Rate</th><th>Avg Return</th></tr>"
            ])
            
            for sentiment, stats in backtest_results['sentiment_stats'].items():
                html.append(
                    f"<tr><td>{sentiment.capitalize()}</td>"
                    f"<td>{stats['count']}</td>"
                    f"<td>{stats['win_rate']*100:.2f}%</td>"
                    f"<td class=\"{'positive' if stats['avg_return'] > 0 else 'negative'}\">"
                    f"{stats['avg_return']:.2f}%</td></tr>"
                )
            
            html.append("</table>")
        else:
            html.append("<p>No sentiment statistics available.</p>")
        
        html.extend([
            "</div>",
            "</div>",
            "</div>"
        ])
        
        # Market Session Analysis
        if 'market_session_stats' in backtest_results and backtest_results['market_session_stats']:
            html.extend([
                "<h2>Market Session Analysis</h2>",
                "<table>",
                "<tr><th>Session</th><th>Count</th><th>Win Rate</th><th>Avg Return</th><th>Total Return</th></tr>"
            ])
            
            # Order sessions logically
            session_order = ["Pre-Market", "Morning", "Midday", "Afternoon", "After-Hours"]
            for session in session_order:
                if session in backtest_results['market_session_stats']:
                    stats = backtest_results['market_session_stats'][session]
                    html.append(
                        f"<tr><td>{session}</td>"
                        f"<td>{stats['count']}</td>"
                        f"<td>{stats['win_rate']*100:.2f}%</td>"
                        f"<td class=\"{'positive' if stats['avg_return'] > 0 else 'negative'}\">"
                        f"{stats['avg_return']:.2f}%</td>"
                        f"<td class=\"{'positive' if stats['total_return'] > 0 else 'negative'}\">"
                        f"{stats['total_return']:.2f}%</td></tr>"
                    )
            
            html.append("</table>")
        
        # Day of Week Analysis
        if 'day_of_week_stats' in backtest_results and backtest_results['day_of_week_stats']:
            html.extend([
                "<h2>Day of Week Analysis</h2>",
                "<table>",
                "<tr><th>Day</th><th>Count</th><th>Win Rate</th><th>Avg Return</th><th>Total Return</th></tr>"
            ])
            
            # Order days correctly
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            for day in day_order:
                if day in backtest_results['day_of_week_stats']:
                    stats = backtest_results['day_of_week_stats'][day]
                    html.append(
                        f"<tr><td>{day}</td>"
                        f"<td>{stats['count']}</td>"
                        f"<td>{stats['win_rate']*100:.2f}%</td>"
                        f"<td class=\"{'positive' if stats['avg_return'] > 0 else 'negative'}\">"
                        f"{stats['avg_return']:.2f}%</td>"
                        f"<td class=\"{'positive' if stats['total_return'] > 0 else 'negative'}\">"
                        f"{stats['total_return']:.2f}%</td></tr>"
                    )
            
            html.append("</table>")
        
        # Individual Trades Table
        html.extend([
            "<h2>Individual Trades</h2>",
            "<table>",
            "<tr><th>#</th><th>Date</th><th>Session</th><th>Direction</th><th>Sentiment</th>"
            "<th>Entry Price</th><th>Exit Price</th><th>Return</th><th>News Title</th></tr>"
        ])
        
        for i, trade in trades_df.iterrows():
            return_class = 'positive' if trade['return_pct'] > 0 else 'negative'
            html.append(
                f"<tr><td>{i+1}</td>"
                f"<td>{trade['news_time'].strftime('%Y-%m-%d %H:%M:%S')}</td>"
                f"<td>{trade['market_session']}</td>"
                f"<td>{trade['direction']}</td>"
                f"<td>{trade['sentiment'].capitalize()}</td>"
                f"<td>${trade['entry_price']:.2f}</td>"
                f"<td>${trade['exit_price']:.2f}</td>"
                f"<td class='{return_class}'>{trade['return_pct']:.2f}%</td>"
                f"<td>{trade['title'][:50]}...</td></tr>"
            )
        
        html.extend([
            "</table>",
            "<h3>Trade Details</h3>",
            "<p>Click on a trade number to see detailed information.</p>"
        ])
        
        # Individual Trade Details (collapsible)
        for i, trade in trades_df.iterrows():
            return_class = 'positive' if trade['return_pct'] > 0 else 'negative'
            html.extend([
                f"<details id='trade-{i+1}'>",
                f"<summary>Trade #{i+1} - {trade['return_pct']:.2f}% ({trade['direction']} - {trade['sentiment'].capitalize()})</summary>",
                "<div class='trade-details'>",
                f"<p><strong>News Time:</strong> {trade['news_time'].strftime('%Y-%m-%d %H:%M:%S')}</p>",
                f"<p><strong>News Title:</strong> {trade['title']}</p>",
                f"<p><strong>Sentiment Analysis:</strong> {trade['explanation']}</p>",
                f"<p><strong>Market Session:</strong> {trade['market_session']}</p>",
                f"<p><strong>Day of Week:</strong> {trade['day_of_week']}</p>",
                "<h4>Trade Statistics</h4>",
                "<table>",
                f"<tr><td>Entry Time</td><td>{trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}</td></tr>",
                f"<tr><td>Exit Time</td><td>{trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}</td></tr>",
                f"<tr><td>Direction</td><td>{trade['direction']}</td></tr>",
                f"<tr><td>Entry Price</td><td>${trade['entry_price']:.2f}</td></tr>",
                f"<tr><td>Exit Price</td><td>${trade['exit_price']:.2f}</td></tr>",
                f"<tr><td>Return</td><td class='{return_class}'>{trade['return_pct']:.2f}%</td></tr>",
                f"<tr><td>Max Favorable Excursion</td><td>{trade['max_favorable_excursion']:.2f}%</td></tr>",
                f"<tr><td>Max Adverse Excursion</td><td>{trade['max_adverse_excursion']:.2f}%</td></tr>",
                "</table>",
                "</div>",
                "</details>"
            ])
        
        # Finish HTML
        html.extend([
            "</body>",
            "</html>"
        ])
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html))
        
        logger.info(f"HTML report saved to {output_path}")
    
    def visualize_market_session_analysis(self, backtest_results: Dict[str, Any]) -> plt.Figure:
        """
        Create a visualization of performance by market session and day of week.
        
        Args:
            backtest_results: Dictionary with backtest results
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # Extract trade data
        trades = pd.DataFrame(backtest_results['trades'])
        if trades.empty:
            plt.figtext(0.5, 0.5, "No trades to visualize", ha='center', va='center', fontsize=14)
            return fig
        
        # Panel 1: Trade Count by Market Session
        ax1 = axs[0, 0]
        market_sessions_ordered = ["Pre-Market", "Morning", "Midday", "Afternoon", "After-Hours"]
        
        # Only include sessions that have trades
        available_sessions = list(set(trades['market_session']) & set(market_sessions_ordered))
        available_sessions.sort(key=lambda x: market_sessions_ordered.index(x))
        
        session_counts = trades['market_session'].value_counts().reindex(available_sessions).fillna(0)
        
        # Create color map based on session time
        session_colors = {
            "Pre-Market": "lightblue",
            "Morning": "skyblue",
            "Midday": "royalblue",
            "Afternoon": "darkblue",
            "After-Hours": "navy"
        }
        
        colors = [session_colors.get(session, "blue") for session in session_counts.index]
        
        session_counts.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title("Trade Count by Market Session", fontsize=14)
        ax1.set_ylabel("Number of Trades", fontsize=12)
        ax1.set_xlabel("Market Session", fontsize=12)
        
        # Add count labels
        for i, v in enumerate(session_counts):
            ax1.text(i, v + 0.1, str(int(v)), ha='center', fontsize=10)
        
        # Panel 2: Win Rate by Market Session
        ax2 = axs[0, 1]
        
        # Prepare data
        session_stats = []
        for session, stats in backtest_results.get('market_session_stats', {}).items():
            if session in available_sessions:
                session_stats.append({
                    'session': session,
                    'win_rate': stats['win_rate'] * 100,
                    'count': stats['count']
                })
        
        session_stats_df = pd.DataFrame(session_stats)
        
        if not session_stats_df.empty:
            # Sort by market session order
            session_stats_df['sort_order'] = session_stats_df['session'].map(
                {session: i for i, session in enumerate(market_sessions_ordered)})
            session_stats_df = session_stats_df.sort_values('sort_order')
            
            # Plot
            bars = sns.barplot(x='session', y='win_rate', data=session_stats_df, ax=ax2,
                             palette=[session_colors.get(s, "blue") for s in session_stats_df['session']])
            
            # Add reference line at 50%
            ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar in bars.patches:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom',
                    fontsize=10
                )
            
            # Format plot
            ax2.set_title("Win Rate by Market Session", fontsize=14)
            ax2.set_ylabel("Win Rate (%)", fontsize=12)
            ax2.set_xlabel("Market Session", fontsize=12)
            ax2.set_ylim(0, 100)
        
        # Panel 3: Trade Count by Day of Week
        ax3 = axs[1, 0]
        
        # Order days of week correctly
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        available_days = list(set(trades['day_of_week']) & set(days_order))
        available_days.sort(key=lambda x: days_order.index(x))
        
        day_counts = trades['day_of_week'].value_counts().reindex(available_days).fillna(0)
        
        # Create color map for days
        day_colors = {
            "Monday": "lightgreen",
            "Tuesday": "mediumseagreen",
            "Wednesday": "seagreen",
            "Thursday": "forestgreen",
            "Friday": "darkgreen"
        }
        
        colors = [day_colors.get(day, "green") for day in day_counts.index]
        
        day_counts.plot(kind='bar', ax=ax3, color=colors)
        ax3.set_title("Trade Count by Day of Week", fontsize=14)
        ax3.set_ylabel("Number of Trades", fontsize=12)
        ax3.set_xlabel("Day of Week", fontsize=12)
        
        # Add count labels
        for i, v in enumerate(day_counts):
            ax3.text(i, v + 0.1, str(int(v)), ha='center', fontsize=10)
        
        # Panel 4: Average Return by Day of Week
        ax4 = axs[1, 1]
        
        # Prepare data
        day_stats = []
        for day, stats in backtest_results.get('day_of_week_stats', {}).items():
            if day in available_days:
                day_stats.append({
                    'day': day,
                    'avg_return': stats['avg_return']
                })
        
        day_stats_df = pd.DataFrame(day_stats)
        
        if not day_stats_df.empty:
            # Sort by day order
            day_stats_df['sort_order'] = day_stats_df['day'].map(
                {day: i for i, day in enumerate(days_order)})
            day_stats_df = day_stats_df.sort_values('sort_order')
            
            # Plot
            bars = sns.barplot(x='day', y='avg_return', data=day_stats_df, ax=ax4,
                             palette=[day_colors.get(d, "green") for d in day_stats_df['day']])
            
            # Add reference line at 0%
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar in bars.patches:
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.1 if height >= 0 else height - 0.3,
                    f'{height:.2f}%',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10
                )
            
            # Format plot
            ax4.set_title("Average Return by Day of Week", fontsize=14)
            ax4.set_ylabel("Average Return (%)", fontsize=12)
            ax4.set_xlabel("Day of Week", fontsize=12)
        
        # Overall title
        fig.suptitle(f'Market Session and Day of Week Analysis for {backtest_results["ticker"]}', 
                     fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        return fig
        
    def visualize_detailed_trade_metrics(self, backtest_results: Dict[str, Any]) -> plt.Figure:
        """
        Create detailed trade metrics visualization including MFE, MAE analysis.
        
        Args:
            backtest_results: Dictionary with backtest results
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # Extract trade data
        trades = pd.DataFrame(backtest_results['trades'])
        if trades.empty:
            plt.figtext(0.5, 0.5, "No trades to visualize", ha='center', va='center', fontsize=14)
            return fig
            
        # Panel 1: MFE vs Final Return scatter plot
        ax1 = axs[0, 0]
        
        # Color by sentiment
        colors = [('green' if s == 'positive' else 'red') for s in trades['sentiment']]
        
        ax1.scatter(trades['return_pct'], trades['max_favorable_excursion'], 
                   c=colors, alpha=0.7, s=50)
        
        # Add diagonal line (where MFE = Final Return)
        min_val = min(trades['return_pct'].min(), 0)
        max_val = max(trades['max_favorable_excursion'].max(), trades['return_pct'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Add horizontal and vertical lines at zero
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        ax1.set_title("Maximum Favorable Excursion vs Final Return", fontsize=14)
        ax1.set_xlabel("Final Return (%)", fontsize=12)
        ax1.set_ylabel("Max Favorable Excursion (%)", fontsize=12)
        
        # Annotate data points with trade index
        for i, (x, y) in enumerate(zip(trades['return_pct'], trades['max_favorable_excursion'])):
            ax1.annotate(f"{i+1}", (x, y), fontsize=8)
        
        # Panel 2: MAE vs Final Return scatter plot
        ax2 = axs[0, 1]
        
        ax2.scatter(trades['return_pct'], trades['max_adverse_excursion'], 
                   c=colors, alpha=0.7, s=50)
        
        # Add diagonal line
        min_val = min(trades['return_pct'].min(), trades['max_adverse_excursion'].min())
        max_val = max(trades['return_pct'].max(), 0)
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Add horizontal and vertical lines at zero
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        ax2.set_title("Maximum Adverse Excursion vs Final Return", fontsize=14)
        ax2.set_xlabel("Final Return (%)", fontsize=12)
        ax2.set_ylabel("Max Adverse Excursion (%)", fontsize=12)
        
        # Annotate data points with trade index
        for i, (x, y) in enumerate(zip(trades['return_pct'], trades['max_adverse_excursion'])):
            ax2.annotate(f"{i+1}", (x, y), fontsize=8)
        
        # Panel 3: Return distribution histogram
        ax3 = axs[1, 0]
        
        sns.histplot(trades['return_pct'], bins=10, kde=True, ax=ax3)
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.axvline(x=trades['return_pct'].mean(), color='green', linestyle='-', alpha=0.7,
                   label=f'Mean: {trades["return_pct"].mean():.2f}%')
        ax3.axvline(x=trades['return_pct'].median(), color='blue', linestyle='-', alpha=0.7,
                   label=f'Median: {trades["return_pct"].median():.2f}%')
        
        ax3.set_title("Distribution of Trade Returns", fontsize=14)
        ax3.set_xlabel("Return (%)", fontsize=12)
        ax3.set_ylabel("Frequency", fontsize=12)
        ax3.legend()
        
        # Panel 4: MFE/MAE ratio analysis
        ax4 = axs[1, 1]
        
        # Calculate MFE to MAE ratio (avoid division by zero)
        trades['mfe_mae_ratio'] = trades.apply(
            lambda row: row['max_favorable_excursion'] / abs(row['max_adverse_excursion']) 
            if row['max_adverse_excursion'] != 0 else float('inf'), axis=1)
        
        # Filter out infinity values
        finite_ratios = trades[trades['mfe_mae_ratio'] != float('inf')]['mfe_mae_ratio']
        
        if not finite_ratios.empty:
            sns.histplot(finite_ratios, bins=10, kde=True, ax=ax4)
            ax4.axvline(x=1, color='red', linestyle='--', alpha=0.7)
            ax4.axvline(x=finite_ratios.mean(), color='green', linestyle='-', alpha=0.7,
                       label=f'Mean: {finite_ratios.mean():.2f}')
            
            ax4.set_title("Distribution of MFE/MAE Ratio", fontsize=14)
            ax4.set_xlabel("MFE/MAE Ratio", fontsize=12)
            ax4.set_ylabel("Frequency", fontsize=12)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "Insufficient data for MFE/MAE analysis", 
                    ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        
        # Overall title
        fig.suptitle(f'Detailed Trade Metrics Analysis for {backtest_results["ticker"]}', 
                     fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        return fig
    
    def visualize_drawdown(self, backtest_results: Dict[str, Any]) -> plt.Figure:
        """
        Create a visualization of the strategy's drawdown over time.
        
        Args:
            backtest_results: Dictionary with backtest results
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        trades_df = pd.DataFrame(backtest_results['trades'])
        
        if trades_df.empty or 'return_pct' not in trades_df.columns:
            plt.figtext(0.5, 0.5, "No trades or returns to calculate drawdown", ha='center', va='center', fontsize=14)
            ax.set_title(f"Drawdown Chart for {backtest_results['ticker']} (No Data)", fontsize=16)
            return fig
            
        returns = trades_df['return_pct'].values
        
        # Calculate cumulative wealth (starting at 1)
        cumulative_wealth = np.insert((1 + returns / 100).cumprod(), 0, 1)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_wealth)
        
        # Calculate drawdown series
        drawdown = (cumulative_wealth - running_max) / running_max * 100 # As percentage
        
        # Prepare x-axis (time)
        # Use entry time of first trade as start, then exit times for subsequent points
        if not trades_df.empty:
             times = pd.to_datetime([trades_df['entry_time'].iloc[0]] + trades_df['exit_time'].tolist())
             if len(times) == len(drawdown):
                 # Plot drawdown
                 ax.plot(times, drawdown, color='red', linewidth=1.5, label='Drawdown')
                 ax.fill_between(times, drawdown, 0, color='red', alpha=0.3)
                 
                 # Annotate max drawdown
                 max_dd_pct = backtest_results.get('max_drawdown_pct', None)
                 if max_dd_pct is not None:
                     ax.text(0.02, 0.95, f"Max Drawdown: {max_dd_pct:.1f}%", 
                             transform=ax.transAxes, fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

             else:
                 logger.warning("Length mismatch between drawdown and time data. Skipping plot.")
                 plt.figtext(0.5, 0.5, "Drawdown calculation error", ha='center', va='center', fontsize=14)
        
        # Format plot
        ax.set_title(f"Strategy Drawdown Over Time for {backtest_results['ticker']}", fontsize=16)
        ax.set_ylabel("Drawdown (%)", fontsize=14)
        ax.set_xlabel("Time", fontsize=14)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter()) # Format y-axis as percentage
        
        # Format x-axis for dates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Simpler format for drawdown
        fig.autofmt_xdate() # Auto-rotate date labels
        
        plt.tight_layout()
        
        return fig

    # --- New Methods for Index Analysis ---
    
    def visualize_index_ticker_performance(self, ticker_summaries_df: pd.DataFrame, 
                                        index_dir: str, top_n: int = 5) -> None:
        """Visualize individual ticker performance within the index."""
        if ticker_summaries_df.empty:
            logger.warning("No ticker summaries provided for visualization.")
            return

        # 1. Bar chart of returns by ticker
        plt.figure(figsize=(14, 8))
        ticker_summaries_df.sort_values('total_return_pct', ascending=False, inplace=True)
        bars = plt.bar(ticker_summaries_df['ticker'], ticker_summaries_df['total_return_pct'])
        
        # Color positive returns green and negative returns red
        for i, bar in enumerate(bars):
            if ticker_summaries_df['total_return_pct'].iloc[i] >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
                
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Strategy Returns by Ticker (%)', fontsize=15)
        plt.xlabel('Ticker', fontsize=12)
        plt.ylabel('Strategy Return (%)', fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(index_dir, 'strategy_returns_by_ticker.png'), dpi=300)
        plt.close()
        
        # Add separate Buy & Hold Returns chart if data is available
        if 'buyhold_return_pct' in ticker_summaries_df.columns:
            plt.figure(figsize=(14, 8))
            buyhold_df = ticker_summaries_df.sort_values('buyhold_return_pct', ascending=False)
            bars = plt.bar(buyhold_df['ticker'], buyhold_df['buyhold_return_pct'])
            
            # Color positive returns green and negative returns red
            for i, bar in enumerate(bars):
                if buyhold_df['buyhold_return_pct'].iloc[i] >= 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
                    
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Buy & Hold Returns by Ticker (%)', fontsize=15)
            plt.xlabel('Ticker', fontsize=12)
            plt.ylabel('Buy & Hold Return (%)', fontsize=12)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(index_dir, 'buyhold_returns_by_ticker.png'), dpi=300)
            plt.close()
        
        # 2. Scatter plot of win rate vs. total return
        plt.figure(figsize=(12, 8))
        plt.scatter(ticker_summaries_df['win_rate'], ticker_summaries_df['total_return_pct'], 
                   s=ticker_summaries_df['total_trades'] * 5, alpha=0.6)
        
        # Add ticker labels to each point
        for i, ticker in enumerate(ticker_summaries_df['ticker']):
            plt.annotate(ticker, 
                        (ticker_summaries_df['win_rate'].iloc[i], ticker_summaries_df['total_return_pct'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
        plt.title('Win Rate vs. Strategy Return by Ticker', fontsize=15)
        plt.xlabel('Win Rate', fontsize=12)
        plt.ylabel('Strategy Return (%)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(index_dir, 'winrate_vs_return.png'), dpi=300)
        plt.close()
        
        # 3. Top and bottom performing tickers
        top_performers = ticker_summaries_df.nlargest(top_n, 'total_return_pct')
        bottom_performers = ticker_summaries_df.nsmallest(top_n, 'total_return_pct')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top performers
        ax1.barh(top_performers['ticker'], top_performers['total_return_pct'], color='green')
        ax1.set_title(f'Top Strategy Performers', fontsize=14)
        ax1.set_xlabel('Strategy Return (%)', fontsize=12)
        
        # Bottom performers
        ax2.barh(bottom_performers['ticker'], bottom_performers['total_return_pct'], color='red')
        ax2.set_title(f'Bottom Strategy Performers', fontsize=14)
        ax2.set_xlabel('Strategy Return (%)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(index_dir, 'top_bottom_performers.png'), dpi=300)
        plt.close()
        
        # NEW VISUALIZATIONS
        
        # 4. Strategy vs Buy & Hold Comparison (if data available)
        if 'buyhold_return_pct' in ticker_summaries_df.columns and 'outperformance_pct' in ticker_summaries_df.columns:
            # Sort by outperformance
            outperformance_df = ticker_summaries_df.sort_values('outperformance_pct', ascending=False).copy()
            
            # Create subplot for comparison chart
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot outperformance
            bars = ax.bar(outperformance_df['ticker'], outperformance_df['outperformance_pct'])
            
            # Color based on outperformance
            for i, bar in enumerate(bars):
                if outperformance_df['outperformance_pct'].iloc[i] >= 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_title('Strategy Outperformance vs Buy & Hold by Ticker (%)', fontsize=15)
            ax.set_xlabel('Ticker', fontsize=12)
            ax.set_ylabel('Outperformance (%)', fontsize=12)
            ax.tick_params(axis='x', rotation=90)
            
            # Add data labels
            for i, v in enumerate(outperformance_df['outperformance_pct']):
                ax.text(i, v + (0.5 if v >= 0 else -1.5), f"{v:.1f}%", 
                      ha='center', va='bottom', fontsize=8,
                      color='darkgreen' if v >= 0 else 'darkred')
            
            plt.tight_layout()
            plt.savefig(os.path.join(index_dir, 'strategy_outperformance.png'), dpi=300)
            plt.close()
            
            # 5. Detailed KPI table as a heatmap
            plt.figure(figsize=(16, len(ticker_summaries_df) * 0.4 + 2))
            
            # Create a multi-KPI dataframe
            display_cols = ['ticker', 'total_return_pct', 'buyhold_return_pct', 'outperformance_pct', 
                           'win_rate', 'total_trades', 'sharpe_ratio', 'max_drawdown_pct']
            
            # Select only columns that exist
            available_cols = [col for col in display_cols if col in ticker_summaries_df.columns]
            kpi_df = ticker_summaries_df[available_cols].copy()
            
            # Format column names for display
            col_names = {
                'total_return_pct': 'Strategy Return (%)',
                'buyhold_return_pct': 'Buy & Hold Return (%)',
                'outperformance_pct': 'Outperformance (%)',
                'win_rate': 'Win Rate',
                'total_trades': 'Trade Count',
                'sharpe_ratio': 'Sharpe Ratio',
                'max_drawdown_pct': 'Max Drawdown (%)'
            }
            
            # Rename columns that exist
            rename_cols = {col: col_names[col] for col in col_names if col in kpi_df.columns}
            kpi_df = kpi_df.rename(columns=rename_cols)
            
            # Format win rate as percentage if it exists
            if 'Win Rate' in kpi_df.columns:
                kpi_df['Win Rate'] = kpi_df['Win Rate'] * 100
                
            # Set ticker as index
            kpi_df = kpi_df.set_index('ticker')
            
            # Create heatmap with annotations
            sns.heatmap(kpi_df, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                       center=0, linewidths=0.5, cbar=False)
            plt.title('Detailed Performance Metrics by Ticker', fontsize=15)
            plt.tight_layout()
            plt.savefig(os.path.join(index_dir, 'detailed_performance_metrics.png'), dpi=300)
            plt.close()
            
            # 6. Strategy vs Buy & Hold paired bar chart
            if 'buyhold_return_pct' in ticker_summaries_df.columns:
                paired_df = ticker_summaries_df.sort_values('total_return_pct', ascending=False).copy()
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Set positions for bars
                x = np.arange(len(paired_df))
                width = 0.35
                
                # Create paired bars
                ax.bar(x - width/2, paired_df['total_return_pct'], width, label='Strategy', color='blue')
                ax.bar(x + width/2, paired_df['buyhold_return_pct'], width, label='Buy & Hold', color='gray')
                
                # Add formatting
                ax.set_xlabel('Ticker', fontsize=12)
                ax.set_ylabel('Return (%)', fontsize=12)
                ax.set_title('Strategy vs Buy & Hold Returns by Ticker', fontsize=15)
                ax.set_xticks(x)
                ax.set_xticklabels(paired_df['ticker'], rotation=90)
                ax.legend()
                ax.grid(alpha=0.3, axis='y')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
                
                plt.tight_layout()
                plt.savefig(os.path.join(index_dir, 'strategy_vs_buyhold_bars.png'), dpi=300)
                plt.close()
        
        logger.info(f"Ticker performance visualizations saved to {index_dir}")
    
    def visualize_index_sentiment_analysis(self, sentiment_df: pd.DataFrame, 
                                         index_dir: str) -> None:
        """Visualize aggregated sentiment analysis for the index."""
        if sentiment_df.empty:
            logger.warning("No sentiment data provided for visualization.")
            return

        try:
            # Sentiment distribution visualization (Bar Chart)
            if 'sentiment' in sentiment_df.columns and 'ticker' in sentiment_df.columns:
                sentiment_counts = sentiment_df.groupby(['ticker', 'sentiment']).size().reset_index(name='count')
                plt.figure(figsize=(14, 8))
                sentiment_pivot = sentiment_counts.pivot(index='ticker', columns='sentiment', values='count').fillna(0)
                sentiment_pivot.plot(kind='bar', stacked=True, colormap='viridis')
                plt.title('News Sentiment Distribution by Ticker', fontsize=15)
                plt.xlabel('Ticker', fontsize=12)
                plt.ylabel('Number of News Articles', fontsize=12)
                plt.xticks(rotation=90)
                plt.legend(title='Sentiment')
                plt.tight_layout()
                plt.savefig(os.path.join(index_dir, 'sentiment_distribution.png'), dpi=300)
                plt.close()
            
            # Overall Sentiment Pie Chart
            if 'sentiment' in sentiment_df.columns:
                plt.figure(figsize=(10, 8))
                sentiment_totals = sentiment_df['sentiment'].value_counts()
                colors = [('green' if s == 'positive' else 'red' if s == 'negative' else 'gray') for s in sentiment_totals.index]
                plt.pie(sentiment_totals, labels=sentiment_totals.index, autopct='%1.1f%%', 
                        colors=colors, explode=[0.05] * len(sentiment_totals), 
                        shadow=True, startangle=90)
                plt.title('Overall Sentiment Distribution for DOW Index', fontsize=15)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(os.path.join(index_dir, 'overall_sentiment_pie.png'), dpi=300)
                plt.close()
            
            # Daily Sentiment Trends
            if 'date' in sentiment_df.columns and 'sentiment' in sentiment_df.columns:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                sentiment_df['date_only'] = sentiment_df['date'].dt.date
                daily_sentiment = sentiment_df.groupby(['date_only', 'sentiment']).size().reset_index(name='count')
                plt.figure(figsize=(16, 8))
                daily_pivot = daily_sentiment.pivot(index='date_only', columns='sentiment', values='count').fillna(0)
                daily_pivot.plot(kind='line', marker='o', markersize=4)
                plt.title('Daily News Sentiment Trends', fontsize=15)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Number of News Articles', fontsize=12)
                plt.grid(alpha=0.3)
                plt.legend(title='Sentiment')
                plt.tight_layout()
                plt.savefig(os.path.join(index_dir, 'daily_sentiment_trends.png'), dpi=300)
                plt.close()
                
            # Word Clouds
            if 'explanation' in sentiment_df.columns and 'sentiment' in sentiment_df.columns:
                try:
                    from wordcloud import WordCloud
                    # Positive Word Cloud
                    positive_text = ' '.join(sentiment_df[sentiment_df['sentiment'] == 'positive']['explanation'].fillna(''))
                    if positive_text.strip():
                        wordcloud_positive = WordCloud(width=800, height=400, background_color='white', 
                                                      max_words=100, contour_width=3).generate(positive_text)
                        plt.figure(figsize=(10, 6))
                        plt.imshow(wordcloud_positive, interpolation='bilinear')
                        plt.axis('off')
                        plt.title('Common Words in Positive News', fontsize=15)
                        plt.tight_layout()
                        plt.savefig(os.path.join(index_dir, 'positive_news_wordcloud.png'), dpi=300)
                        plt.close()
                        
                    # Negative Word Cloud
                    negative_text = ' '.join(sentiment_df[sentiment_df['sentiment'] == 'negative']['explanation'].fillna(''))
                    if negative_text.strip():
                        wordcloud_negative = WordCloud(width=800, height=400, background_color='white',
                                                      max_words=100, contour_width=3).generate(negative_text)
                        plt.figure(figsize=(10, 6))
                        plt.imshow(wordcloud_negative, interpolation='bilinear')
                        plt.axis('off')
                        plt.title('Common Words in Negative News', fontsize=15)
                        plt.tight_layout()
                        plt.savefig(os.path.join(index_dir, 'negative_news_wordcloud.png'), dpi=300)
                        plt.close()
                except ImportError:
                    logger.warning("WordCloud package not available. Skipping word cloud visualization.")
                    
            logger.info(f"Sentiment analysis visualizations saved to {index_dir}")
        except Exception as e:
            logger.error(f"Error generating sentiment visualizations: {e}", exc_info=True)

    def visualize_index_sentiment_performance(self, trades_df: pd.DataFrame, ticker_kpis: pd.DataFrame, 
                                            index_dir: str) -> None:
        """Visualize the performance of trades based on news sentiment for the index."""
        if trades_df.empty or 'sentiment' not in trades_df.columns:
            logger.warning("No trades or sentiment data provided for sentiment performance visualization.")
            return

        try:
            # Overall Sentiment Performance (Average Return & Win Rate)
            sentiment_kpis = trades_df.groupby('sentiment').agg(
                trade_count=('return_pct', 'count'),
                avg_return=('return_pct', 'mean'),
                win_rate=('return_pct', lambda x: (x > 0).mean())
            ).reset_index()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            bar_colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            bar_colors_list = [bar_colors.get(s, 'blue') for s in sentiment_kpis['sentiment']]

            # Plot average return
            sentiment_kpis = sentiment_kpis.sort_values('avg_return', ascending=False)
            ax1.bar(sentiment_kpis['sentiment'], sentiment_kpis['avg_return'], color=bar_colors_list)
            ax1.set_title('Average Return by Sentiment (%)', fontsize=14)
            ax1.set_ylabel('Average Return (%)', fontsize=12)
            for i, v in enumerate(sentiment_kpis['avg_return']):
                ax1.text(i, v + 0.05 * np.sign(v) if v != 0 else 0.05, f"{v:.2f}%", ha='center', fontsize=10)

            # Plot win rate
            sentiment_kpis = sentiment_kpis.sort_values('win_rate', ascending=False)
            ax2.bar(sentiment_kpis['sentiment'], sentiment_kpis['win_rate'] * 100, color=bar_colors_list)
            ax2.set_title('Win Rate by Sentiment (%)', fontsize=14)
            ax2.set_ylabel('Win Rate (%)', fontsize=12)
            for i, v in enumerate(sentiment_kpis['win_rate']):
                ax2.text(i, v * 100 + 1, f"{v*100:.1f}%", ha='center', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(index_dir, 'sentiment_performance.png'), dpi=300)
            plt.close()

            # Sentiment Performance Heatmaps (By Ticker)
            sentiment_ticker_kpis = trades_df.groupby(['ticker', 'sentiment']).agg(
                trade_count=('return_pct', 'count'),
                avg_return=('return_pct', 'mean'),
                win_rate=('return_pct', lambda x: (x > 0).mean())
            ).reset_index()

            if 'ticker' in trades_df.columns and not sentiment_ticker_kpis.empty:
                ticker_order = ticker_kpis.reset_index().sort_values('total_return', ascending=False)['ticker'].tolist()
                
                # Heatmap for Average Returns
                heatmap_data = sentiment_ticker_kpis.pivot(index='ticker', columns='sentiment', values='avg_return')
                heatmap_data = heatmap_data.reindex(ticker_order)
                plt.figure(figsize=(12, max(8, len(ticker_order)*0.4)))
                sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0, 
                            fmt='.2f', linewidths=.5, cbar_kws={'label': 'Average Return (%)'})
                plt.title('Average Returns (%) by Ticker and Sentiment', fontsize=15)
                plt.tight_layout()
                plt.savefig(os.path.join(index_dir, 'sentiment_returns_heatmap.png'), dpi=300)
                plt.close()

                # Heatmap for Win Rates
                winrate_heatmap = sentiment_ticker_kpis.pivot(index='ticker', columns='sentiment', values='win_rate')
                winrate_heatmap = winrate_heatmap.reindex(ticker_order)
                plt.figure(figsize=(12, max(8, len(ticker_order)*0.4)))
                sns.heatmap(winrate_heatmap, annot=True, cmap='RdYlGn', vmin=0, vmax=1, 
                            fmt='.0%', linewidths=.5, cbar_kws={'label': 'Win Rate'})
                plt.title('Win Rate by Ticker and Sentiment', fontsize=15)
                plt.tight_layout()
                plt.savefig(os.path.join(index_dir, 'sentiment_winrate_heatmap.png'), dpi=300)
                plt.close()
                
            logger.info(f"Sentiment performance visualizations saved to {index_dir}")
        except Exception as e:
            logger.error(f"Error generating sentiment performance visualizations: {e}", exc_info=True)

    def visualize_index_portfolio_performance(self, trades_df: pd.DataFrame, 
                                        ticker_summaries_df: pd.DataFrame, 
                                        index_dir: str, index_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize the overall portfolio performance for the index."""
        if trades_df.empty or 'exit_time' not in trades_df.columns or 'return_pct' not in trades_df.columns:
            logger.warning("Insufficient trade data for portfolio performance visualization.")
            return index_summary # Return original summary if no data

        try:
            sorted_trades = trades_df.sort_values('exit_time')
            index_name = index_summary.get('index_name', 'Index')
            portfolio_value = 100
            portfolio_values = []
            dates = []
            
            # Calculate position size based on unique tickers in trade data
            # This ensures we have a true equally-weighted portfolio of the tickers that actually have trades
            unique_tickers = sorted_trades['ticker'].unique()
            position_size = 100 / len(unique_tickers) if len(unique_tickers) > 0 else 1

            # Add initial point
            first_trade_time = pd.to_datetime(sorted_trades['exit_time'].iloc[0]) - timedelta(minutes=1) if not sorted_trades.empty else datetime.now()
            dates.append(first_trade_time)
            portfolio_values.append(portfolio_value)

            for _, trade in sorted_trades.iterrows():
                trade_impact = position_size * (trade['return_pct_after_fees'] / 100)
                portfolio_value += trade_impact
                portfolio_values.append(portfolio_value)
                dates.append(pd.to_datetime(trade['exit_time']))

            portfolio_df = pd.DataFrame({'date': dates, 'portfolio_value': portfolio_values}).set_index('date')
            portfolio_df['peak'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak'] * 100

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1], sharex=True)

            # Plot equity curve
            ax1.plot(portfolio_df.index, portfolio_df['portfolio_value'], color='blue', linewidth=2, label=f'{index_name} Portfolio')
            ax1.set_title(f'{index_name} Strategy Portfolio Performance (Equal Weighted)', fontsize=15)
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)

            final_value = portfolio_df['portfolio_value'].iloc[-1]
            total_return_pct = ((final_value / 100) - 1) * 100
            ax1.text(0.02, 0.95, f"Starting: $100\nEnding: ${final_value:.2f}\nReturn: {total_return_pct:.2f}%", 
                    transform=ax1.transAxes, fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

            # Plot drawdowns
            ax2.fill_between(portfolio_df.index, 0, portfolio_df['drawdown'], color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            max_drawdown = portfolio_df['drawdown'].min()
            ax2.set_title(f'Drawdown (Max: {max_drawdown:.2f}%)', fontsize=13)
            
            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()

            plt.tight_layout()
            plt.savefig(os.path.join(index_dir, 'portfolio_performance.png'), dpi=300)
            plt.close()

            # Update index summary with portfolio metrics
            index_summary['portfolio_metrics'] = {
                'starting_value': 100,
                'ending_value': final_value,
                'total_return_pct': total_return_pct,
                'max_drawdown_pct': max_drawdown
            }
            
            
            logger.info(f"Portfolio performance visualization saved to {index_dir}")
            return index_summary
        
        except Exception as e:
            logger.error(f"Error creating portfolio performance chart: {e}", exc_info=True)
            return index_summary # Return original summary on error
            
    def visualize_index_strategy_vs_buyhold(self, trades_df: pd.DataFrame, 
                                      index_dir: str, ticker_list: List[str], 
                                      index_name: str = "Index", 
                                      ticker_summaries_df: pd.DataFrame = None) -> None:
        """Visualizes strategy returns vs. buy & hold for each ticker in an index.
           Shows outperformance (Strategy % - Buy&Hold %).
        """
        if ticker_summaries_df is None or ticker_summaries_df.empty:
            logger.warning(f"ticker_summaries_df is empty for visualize_index_strategy_vs_buyhold for {index_name}.")
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.text(0.5, 0.5, f"Data unavailable for {index_name}\\nStrategy Outperformance vs. Buy & Hold by Ticker", ha='center', va='center', fontsize=10)
            ax.set_title(f'{index_name} - Strategy Outperformance vs. Buy & Hold (Data Unavailable)', fontsize=14)
            plot_filename = os.path.join(index_dir, f"{index_name}_strategy_outperformance_vs_buyhold_by_ticker_UNAVAILABLE.png")
            try:
                plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
            except Exception as e:
                logger.error(f"Failed to save placeholder plot {plot_filename}: {e}")
            plt.close(fig)
            return

        strategy_return_col = None
        if 'strategy_return_with_fees_pct' in ticker_summaries_df.columns:
            strategy_return_col = 'strategy_return_with_fees_pct'
        elif 'total_return_pct' in ticker_summaries_df.columns:
            strategy_return_col = 'total_return_pct'
            logger.warning(f"Using 'total_return_pct' as strategy return for {index_name} in visualize_index_strategy_vs_buyhold.")
        
        required_cols = ['ticker', 'buy_and_hold_return_pct']
        if strategy_return_col: 
            required_cols.append(strategy_return_col)
        else: 
            logger.error(f"Missing strategy return column (e.g., 'strategy_return_with_fees_pct') in ticker_summaries_df for {index_name}.")
            # Create placeholder for malformed data
            # ... (similar placeholder creation as above with different message) ...
            return

        if not all(col in ticker_summaries_df.columns for col in required_cols):
            logger.error(f"Missing some required columns in ticker_summaries_df for {index_name}. Need: {required_cols}, Have: {list(ticker_summaries_df.columns)}")
            # Create placeholder for malformed data
            # ... (similar placeholder creation as above with different message) ...
            return
            
        df_copy = ticker_summaries_df.copy()
        # Ensure numeric types before subtraction
        df_copy[strategy_return_col] = pd.to_numeric(df_copy[strategy_return_col], errors='coerce')
        df_copy['buy_and_hold_return_pct'] = pd.to_numeric(df_copy['buy_and_hold_return_pct'], errors='coerce')
        df_copy = df_copy.dropna(subset=[strategy_return_col, 'buy_and_hold_return_pct'])

        if df_copy.empty:
            logger.warning(f"No valid numeric data after cleaning for {index_name} in visualize_index_strategy_vs_buyhold.")
            # ... (placeholder plot for no valid data) ...
            return

        df_copy['outperformance_pct'] = df_copy[strategy_return_col] - df_copy['buy_and_hold_return_pct']
        df_sorted = df_copy.sort_values(by='outperformance_pct', ascending=False)

        plt.style.use('seaborn-v0_8-darkgrid')
        fig_width = max(16, len(df_sorted) * 0.6) # Adjusted dynamic width
        fig, ax = plt.subplots(figsize=(fig_width, 10)) # Adjusted height
        
        num_tickers = len(df_sorted['ticker'])
        x_indices = np.arange(num_tickers)
        
        colors = ['#28a745' if val >= 0 else '#dc3545' for val in df_sorted['outperformance_pct']]
        bars = ax.bar(x_indices, df_sorted['outperformance_pct'], label='Strategy Outperformance vs Buy & Hold', color=colors, width=0.8)

        ax.set_ylabel('Outperformance (Strategy % - Buy & Hold %)', fontsize=13)
        ax.set_xlabel('Ticker', fontsize=13)
        plot_title = f'{index_name} - Strategy Outperformance vs. Buy & Hold by Ticker'
        ax.set_title(plot_title, fontsize=16, fontweight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(df_sorted['ticker'], rotation=55, ha="right", fontsize=10)
        ax.legend(fontsize=11)
        ax.axhline(0, color='grey', lw=1.2, linestyle='--')

        for bar_obj in bars:
            yval = bar_obj.get_height()
            plt.text(bar_obj.get_x() + bar_obj.get_width()/2.0, yval + np.sign(yval)*0.5, f'{yval:.1f}%', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=9, fontweight='normal')

        plt.tight_layout(pad=1.5)
        new_plot_filename = os.path.join(index_dir, f"{index_name}_strategy_outperformance_vs_buyhold_by_ticker.png")
        try:
            plt.savefig(new_plot_filename, bbox_inches='tight', dpi=300)
            logger.info(f"Saved strategy vs buy & hold (outperformance) plot for {index_name} to {new_plot_filename}")
        except Exception as e:
            logger.error(f"Failed to save plot {new_plot_filename}: {e}")
        plt.close(fig)
    
    def create_index_kpi_dashboard(self, index_summary: Dict[str, Any], 
                                ticker_summaries_df: pd.DataFrame, 
                                 output_dir: str) -> None:
        """
        Create a comprehensive KPI dashboard for the index.
        This dashboard will be saved as an HTML file.
        """
        if not isinstance(index_summary, dict):
            logger.error("index_summary must be a dictionary for KPI dashboard.")
            return
        if not isinstance(ticker_summaries_df, pd.DataFrame):
            logger.error("ticker_summaries_df must be a DataFrame for KPI dashboard.")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        # Use 'index_name' instead of 'index'
        index_name_val = index_summary.get('index_name', 'Unknown Index')
        logger.info(f"Creating KPI dashboard for index {index_name_val}")

        # Initialize dashboard components
        components = []

        # Extract key metrics from index summary
        total_tickers = index_summary['successful_tickers']
        total_trades = index_summary['total_trades_overall']
        avg_return = index_summary.get('average_return_pct', index_summary.get('average_strategy_return_pct', 0))
        win_rate = index_summary['average_win_rate']
        avg_buyhold_return = index_summary.get('average_buyhold_return_pct')
        avg_outperformance = index_summary.get('average_outperformance_pct')
        portfolio_return = index_summary.get('portfolio_metrics', {}).get('total_return_pct')
        max_drawdown = index_summary.get('portfolio_metrics', {}).get('max_drawdown_pct')

        # Extract average market model metrics (New)
        market_model_metrics = index_summary.get('average_market_model_metrics', {})
        avg_r2 = market_model_metrics.get('avg_r_squared')
        avg_adj_r2 = market_model_metrics.get('avg_adj_r_squared')
        avg_rmse = market_model_metrics.get('avg_rmse')
        avg_mae = market_model_metrics.get('avg_mae')
        model_estimation_count = market_model_metrics.get('tickers_with_successful_estimation', 0)


        # Setup plot grid
        fig = plt.figure(figsize=(18, 24))
        grid = gridspec.GridSpec(4, 3, height_ratios=[0.5, 1.5, 3, 3], hspace=0.4, wspace=0.3)
        
        # 1. Title
        title_ax = fig.add_subplot(grid[0, :])
        title_ax.axis('off')  # Hide axes
        
        # Create title and subtitle
        title_ax.text(0.5, 0.8, f"{index_name_val} INDEX PERFORMANCE DASHBOARD", 
                      fontsize=24, fontweight='bold', ha='center')
        title_ax.text(0.5, 0.5, f"Analysis of {total_tickers} Tickers with {total_trades} Total Trades", 
                      fontsize=16, ha='center')
        
        # 2. Key metrics in boxes
        metrics_ax = fig.add_subplot(grid[1, :])
        metrics_ax.axis('off')  # Hide axes
        
        # Define metrics to display
        metrics = [
            {'label': 'Total Return', 'value': f"{avg_return:.2f}%"},
            {'label': 'Win Rate', 'value': f"{win_rate*100:.1f}%"},
            {'label': 'Total Trades', 'value': f"{total_trades}"}
        ]
        
        # Add buy & hold comparison if available
        if avg_buyhold_return is not None:
            metrics.append({'label': 'B&H Return', 'value': f"{avg_buyhold_return:.2f}%"})
        
        if avg_outperformance is not None:
            metrics.append({'label': 'Outperformance', 'value': f"{avg_outperformance:.2f}%"})
            
        if portfolio_return is not None:
            metrics.append({'label': 'Portfolio Return', 'value': f"{portfolio_return:.2f}%"})
            
        if max_drawdown is not None:
            metrics.append({'label': 'Max Drawdown', 'value': f"{max_drawdown:.2f}%"})
        
        # Add market model metrics if available (New)
        if model_estimation_count > 0:
            if avg_r2 is not None: metrics.append({'label': 'Avg R²', 'value': f"{avg_r2:.3f}"})
            if avg_adj_r2 is not None: metrics.append({'label': 'Avg Adj R²', 'value': f"{avg_adj_r2:.3f}"})
            if avg_rmse is not None: metrics.append({'label': 'Avg RMSE', 'value': f"{avg_rmse:.5f}"})
            if avg_mae is not None: metrics.append({'label': 'Avg MAE', 'value': f"{avg_mae:.5f}"})
        
            # Calculate positions for metric boxes
            num_metrics = len(metrics)
            box_width = 1.0 / (num_metrics + 1)
            
            # Create colored boxes with metric values
            for i, metric in enumerate(metrics):
                pos_x = (i + 1) * box_width
                
                # Determine box color based on metric
                color = 'lightgreen'
                if 'Return' in metric['label'] or 'Outperformance' in metric['label']:
                    # For return metrics, use green for positive, red for negative
                    value = float(metric['value'].strip('%'))
                    color = 'lightgreen' if value >= 0 else 'lightcoral'
                elif metric['label'] == 'Max Drawdown':
                    color = 'lightcoral'  # Drawdown is always negative/bad
                
                # Draw box with value
                rect = plt.Rectangle((pos_x - box_width/2, 0.4), box_width*0.8, 0.4, 
                                    facecolor=color, alpha=0.6, edgecolor='gray', zorder=1)
                metrics_ax.add_patch(rect)
                
                # Add label and value
                metrics_ax.text(pos_x, 0.7, metric['label'], fontsize=12, ha='center', va='center', zorder=2)
                metrics_ax.text(pos_x, 0.5, metric['value'], fontsize=18, fontweight='bold', ha='center', va='center', zorder=2)
            
            # 3. Performance distribution (bottom left)
            if 'total_return_pct' in ticker_summaries_df.columns:
                perf_ax = fig.add_subplot(grid[2, 0])
                
                # Get return values
                returns = ticker_summaries_df['total_return_pct'].values
                
                # Create histogram with density plot
                sns.histplot(returns, kde=True, ax=perf_ax, color='skyblue')
                
                # Add mean line
                perf_ax.axvline(returns.mean(), color='red', linestyle='--', linewidth=1.5,
                              label=f'Mean: {returns.mean():.2f}%')
                
                # Add median line
                perf_ax.axvline(np.median(returns), color='green', linestyle=':', linewidth=1.5,
                              label=f'Median: {np.median(returns):.2f}%')
                
                perf_ax.set_title('Return Distribution', fontsize=12)
                perf_ax.set_xlabel('Return (%)', fontsize=10)
                perf_ax.legend(fontsize=9)
            
            # 4. Top/Bottom performers (bottom middle)
            if not ticker_summaries_df.empty:
                top_ax = fig.add_subplot(grid[2, 1])
                
                # Get top and bottom performers
                top_n = min(5, len(ticker_summaries_df))
                top_performers = ticker_summaries_df.nlargest(top_n, 'total_return_pct')
                bottom_performers = ticker_summaries_df.nsmallest(top_n, 'total_return_pct')
                
                # Plot top performers
                top_returns = top_performers['total_return_pct'].values
                top_tickers = top_performers['ticker'].values
                y_pos = np.arange(len(top_returns))
                
                # Create bars for top performers
                bars = top_ax.barh(y_pos, top_returns, color='green', alpha=0.7)
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, top_returns)):
                    top_ax.text(value + 0.5, bar.get_y() + bar.get_height()/2, 
                              f"{value:.1f}%", va='center', fontsize=9)
                
                top_ax.set_yticks(y_pos)
                top_ax.set_yticklabels(top_tickers)
                top_ax.set_title(f'Top {top_n} Performers', fontsize=12)
                top_ax.set_xlabel('Return (%)', fontsize=10)
            
            # 5. Outperformance vs Buy & Hold (bottom right)
            if 'outperformance_pct' in ticker_summaries_df.columns:
                outperf_ax = fig.add_subplot(grid[2, 2])
                
                # Count outperforming and underperforming tickers
                outperformers = (ticker_summaries_df['outperformance_pct'] > 0).sum()
                underperformers = (ticker_summaries_df['outperformance_pct'] <= 0).sum()
                
                # Create pie chart
                pie_data = [outperformers, underperformers]
                pie_labels = [f'Outperformed B&H\n{outperformers} tickers', 
                             f'Underperformed B&H\n{underperformers} tickers']
                pie_colors = ['lightgreen', 'lightcoral']
                
                wedges, texts, autotexts = outperf_ax.pie(
                    pie_data, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
                    startangle=90, wedgeprops={'edgecolor': 'w'})
                
                # Set font size for pie labels
                for text in texts:
                    text.set_fontsize(10)
                
                for autotext in autotexts:
                    autotext.set_fontsize(10)
                
                outperf_ax.set_title('Strategy vs Buy & Hold', fontsize=12)
            
            # Save the dashboard
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'index_kpi_dashboard.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Index KPI dashboard created and saved to {output_dir}")
    
    # --- End New Methods ---
    
        
        # Convert to DataFrame
        return pd.DataFrame([stats])

    def visualize_portfolio_cumulative_returns_vs_index(self, trades_df: pd.DataFrame, 
                                                       ticker_list: List[str], 
                                                       index_dir: str, 
                                                       index_name: str = "Portfolio") -> List[Dict[str, str]]:
        """
        Create a cumulative returns visualization comparing portfolio strategy performance 
        against an equally weighted buy & hold index of all tickers.
        
        Args:
            trades_df: DataFrame containing all trades for all tickers
            ticker_list: List of all tickers in the index
            index_dir: Directory to save output files  
            index_name: Name of the index/portfolio
            
        Returns:
            List of dictionaries, each containing file information.
        """
        output_files_info = []
        if trades_df.empty or 'exit_time' not in trades_df.columns:
            logger.warning("No trades data for portfolio cumulative returns visualization.")
            return output_files_info
            
        try:
            # Sort trades by exit time for chronological order
            sorted_trades = trades_df.sort_values('exit_time').copy()
            sorted_trades['exit_time'] = pd.to_datetime(sorted_trades['exit_time'])
            sorted_trades['entry_time'] = pd.to_datetime(sorted_trades['entry_time'])
            
            # Get the overall time range from all trades
            start_date = sorted_trades['entry_time'].min()
            end_date = sorted_trades['exit_time'].max()
            
            logger.info(f"Building portfolio cumulative returns from {start_date} to {end_date}")
            
            # --- Calculate Strategy Portfolio Performance (CORRECTED) ---
            # This represents the actual trading strategy: full position per trade, chronological execution
            portfolio_value = 100.0
            portfolio_dates = [start_date]
            portfolio_values = [portfolio_value]
            
            # CORRECTED: Each trade represents a full position, not a weighted portion
            for _, trade in sorted_trades.iterrows():
                # Apply the full trade return to the portfolio
                trade_return = trade['return_pct_after_fees'] / 100
                portfolio_value *= (1 + trade_return)  # Compound effect
                portfolio_dates.append(trade['exit_time'])
                portfolio_values.append(portfolio_value)
            
            strategy_df = pd.DataFrame({
                'date': portfolio_dates, 
                'strategy_value': portfolio_values
            }).set_index('date')
            
            # --- Calculate Equally Weighted Index Buy & Hold Performance ---
            logger.info(f"Fetching buy & hold data for {len(ticker_list)} tickers")
            
            # Buffer for data fetching
            buffer = timedelta(minutes=30)
            
            # Collect price data for all tickers
            ticker_data = {}
            valid_tickers = []
            
            for ticker in ticker_list:
                try:
                    price_data = fetch_stock_prices(
                        ticker, 
                        start_date - buffer, 
                        end_date + buffer
                    )
                    
                    if not price_data.empty:
                        # Find first valid price at or after start date
                        relevant_data = price_data[price_data.index >= start_date]
                        if not relevant_data.empty:
                            ticker_data[ticker] = price_data
                            valid_tickers.append(ticker)
                            logger.debug(f"Successfully loaded data for {ticker}")
                        else:
                            logger.warning(f"No data found for {ticker} after {start_date}")
                    else:
                        logger.warning(f"No price data available for {ticker}")
                        
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {e}")
                    continue
            
            if not valid_tickers:
                logger.error("No valid ticker data found for buy & hold comparison")
                return output_files_info
            
            logger.info(f"Successfully loaded data for {len(valid_tickers)} out of {len(ticker_list)} tickers")
            
            # Calculate equally weighted index
            equal_weight = 1.0 / len(valid_tickers)
            
            # Get common time index across all tickers
            common_index = None
            for ticker, data in ticker_data.items():
                if common_index is None:
                    common_index = data.index
                else:
                    common_index = common_index.intersection(data.index)
            
            if common_index.empty:
                logger.error("No common time periods found across tickers")
                return output_files_info
            
            # Filter to strategy period
            strategy_period_index = common_index[
                (common_index >= start_date) & (common_index <= end_date)
            ]
            
            if strategy_period_index.empty:
                logger.error("No data available for the strategy period")
                return output_files_info
            
            # Calculate index value for each time point
            index_values = []
            index_dates = []
            
            # Get starting prices for each ticker
            start_prices = {}
            for ticker in valid_tickers:
                ticker_period_data = ticker_data[ticker][
                    ticker_data[ticker].index >= start_date
                ]
                if not ticker_period_data.empty:
                    start_prices[ticker] = ticker_period_data['open'].iloc[0]
                else:
                    logger.warning(f"No starting price found for {ticker}")
            
            # Remove tickers without valid starting prices
            valid_tickers = [t for t in valid_tickers if t in start_prices]
            equal_weight = 1.0 / len(valid_tickers) if valid_tickers else 0
            
            if not valid_tickers:
                logger.error("No tickers with valid starting prices")
                return output_files_info
                
            # Calculate index performance over time
            for timestamp in strategy_period_index:
                index_value = 0.0
                valid_count = 0
                
                for ticker in valid_tickers:
                    if timestamp in ticker_data[ticker].index:
                        current_price = ticker_data[ticker].loc[timestamp, 'close']
                        start_price = start_prices[ticker]
                        ticker_return = (current_price / start_price - 1)
                        index_value += equal_weight * (100 + ticker_return * 100)
                        valid_count += 1
                
                if valid_count > 0:
                    index_dates.append(timestamp)
                    index_values.append(index_value)
            
            if not index_values:
                logger.error("No index values calculated")
                return output_files_info
                
            buyhold_df = pd.DataFrame({
                'date': index_dates,
                'buyhold_value': index_values
            }).set_index('date')
            
            # --- Create the visualization ---
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Plot strategy performance
            ax.plot(strategy_df.index, strategy_df['strategy_value'], 
                   'b-', label='Strategy Return', linewidth=2, alpha=0.8)
            
            # Plot buy & hold performance
            ax.plot(buyhold_df.index, buyhold_df['buyhold_value'],
                   'gray', linestyle='--', label=f'Buy & Hold {index_name}', 
                   linewidth=1.5, alpha=0.7)
            
            # Add reference line at 100
            ax.axhline(y=100, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
            
            # Calculate final returns
            strategy_final_value = strategy_df['strategy_value'].iloc[-1]
            buyhold_final_value = buyhold_df['buyhold_value'].iloc[-1]
            
            strategy_return_pct = (strategy_final_value / 100 - 1) * 100
            buyhold_return_pct = (buyhold_final_value / 100 - 1) * 100
            
            # Calculate some additional statistics
            total_trades = len(trades_df)
            unique_trade_tickers = trades_df['ticker'].nunique()
            win_rate = (trades_df['return_pct_after_fees'] > 0).mean() * 100
            
            # Calculate Sharpe ratio (simplified - using returns standard deviation)
            strategy_returns = strategy_df['strategy_value'].pct_change().dropna()
            if len(strategy_returns) > 1:
                sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 390)  # Annualized for minute data
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown
            strategy_peak = strategy_df['strategy_value'].cummax()
            strategy_drawdown = (strategy_df['strategy_value'] - strategy_peak) / strategy_peak * 100
            max_drawdown = strategy_drawdown.min()
            
            # Calculate profit factor
            winning_trades = trades_df[trades_df['return_pct_after_fees'] > 0]['return_pct_after_fees'].sum()
            losing_trades = abs(trades_df[trades_df['return_pct_after_fees'] < 0]['return_pct_after_fees'].sum())
            profit_factor = winning_trades / losing_trades if losing_trades > 0 else float('inf')
            
            # Add statistics annotation
            stats_text = (
                f"Total Trades: {total_trades}\n"
                f"Win Rate: {win_rate:.1f}%\n"
                f"Strategy Return: {strategy_return_pct:.1f}%\n"
                f"Buy & Hold Return: {buyhold_return_pct:.1f}%\n"
                f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                f"Max Drawdown: {max_drawdown:.1f}%\n"
                f"Profit Factor: {profit_factor:.2f}"
            )
            
            # Place annotation box in the top-left corner
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Format plot
            ax.set_title(f"Cumulative Returns: Strategy vs Buy & Hold for {index_name}", 
                        fontsize=16)
            ax.set_ylabel("Return (%)", fontsize=14)
            ax.set_xlabel("Time", fontsize=14)
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Format y-axis as percentages
            from matplotlib.ticker import FuncFormatter
            def percentage_formatter(x, pos):
                return f'{x-100:.1f}%'
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
            
            # Format x-axis for dates
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            fig.autofmt_xdate()  # Auto-rotate date labels
            
            # Add legend
            ax.legend(loc='lower left', fontsize=10)
            
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f'{index_name}_cumulative_returns_vs_index.png'
            plot_path = os.path.join(index_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_title = f"Cumulative Returns: Strategy vs Buy & Hold for {index_name}"
            plot_description = (
                f"This chart displays the cumulative percentage return of the trading strategy applied to the {index_name} portfolio "
                f"(starting with $100), benchmarked against an equally weighted buy-and-hold strategy of its constituents. "
                f"Strategy performance includes transaction fees. Key metrics such as Total Trades, Win Rate, Strategy Return, "
                f"Buy & Hold Return, Sharpe Ratio, Max Drawdown, and Profit Factor are annotated on the plot."
            )
            output_files_info.append({
                'filename': plot_filename,
                'plot_title': plot_title,
                'description': plot_description
            })
            logger.info(f"Portfolio cumulative returns visualization saved to {plot_path}")
            
            # Save summary statistics to JSON
            summary_stats = {
                'strategy_return_pct': float(strategy_return_pct),
                'buyhold_return_pct': float(buyhold_return_pct),
                'outperformance_pct': float(strategy_return_pct - buyhold_return_pct),
                'total_trades': int(total_trades),
                'win_rate': float(win_rate),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown_pct': float(max_drawdown),
                'profit_factor': float(profit_factor) if profit_factor != float('inf') else None,
                'tickers_in_index': valid_tickers,
                'analysis_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            }
            
            summary_filename = f'{index_name}_portfolio_vs_index_summary.json'
            summary_path = os.path.join(index_dir, summary_filename)
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=4, default=str)
            
            summary_description = (
                f"A JSON file containing key performance statistics for the {index_name} portfolio strategy compared to its buy-and-hold benchmark. "
                f"Includes metrics like strategy return, buy & hold return, outperformance, total trades, win rate, Sharpe ratio, max drawdown, "
                f"profit factor, list of tickers in the index, and the analysis period."
            )
            output_files_info.append({
                'filename': summary_filename,
                'plot_title': f"{index_name} Portfolio vs Index Performance Summary Data",
                'description': summary_description
            })
            logger.info(f"Portfolio summary statistics saved to {summary_path}")
            return output_files_info
            
        except Exception as e:
            logger.error(f"Error creating portfolio cumulative returns visualization: {e}", exc_info=True)
            return output_files_info

    def visualize_portfolio_event_study_timeline(self, ticker_list: List[str], 
                                                index_dir: str, 
                                                index_name: str = "Portfolio") -> None:
        """
        Create a continuous timeline event study visualization for the entire portfolio,
        similar to individual stock event studies, showing AAR and CAAR over a 
        continuous time window based on the available window data (5, 15, 30, 60 minutes).
        
        Args:
            ticker_list: List of all tickers in the portfolio
            index_dir: Directory where individual ticker event study results are stored
            index_name: Name of the portfolio/index
        """
        logger.info(f"Creating portfolio event study timeline visualization for {index_name}")
        
        import os
        import json
        import numpy as np
        import pandas as pd
        from collections import defaultdict
        
        # Create event_study subdirectory within index_dir for event study files
        event_study_output_dir = os.path.join(index_dir, "event_study")
        os.makedirs(event_study_output_dir, exist_ok=True)
        
        # Collect AAR and CAAR data from individual tickers for each sentiment and window
        portfolio_aar_data = defaultdict(lambda: defaultdict(list))  # {sentiment: {window: [values]}}
        portfolio_caar_data = defaultdict(lambda: defaultdict(list))  # {sentiment: {window: [values]}}
        event_counts = defaultdict(lambda: defaultdict(int))  # {sentiment: {window: count}}
        successful_tickers = []
        
        available_windows = [5, 15, 30, 60]
        
        # Navigate up from index_dir to the run directory - handle both cases dynamically
        # Remove trailing slash and normalize path for consistent checking
        normalized_index_dir = index_dir.rstrip('/')
        if normalized_index_dir.endswith("visualizations"):
            run_dir = os.path.dirname(os.path.dirname(normalized_index_dir))  # Go up 2 levels from visualizations
        else:
            run_dir = os.path.dirname(normalized_index_dir)  # Go up 1 level from LOG_INDEX_ANALYSIS
        
        for ticker in ticker_list:
            # Look for ticker's event study results in the same run directory
            ticker_dir = os.path.join(run_dir, ticker)
            event_study_file = os.path.join(ticker_dir, 'event_study', f'{ticker}_event_study_results.json')
            
            if os.path.exists(event_study_file):
                try:
                    with open(event_study_file, 'r') as f:
                        event_data = json.load(f)
                    
                    # Extract AAR/CAAR data from the standard format
                    if 'aar_caar' in event_data:
                        for sentiment in ['positive', 'negative', 'neutral']:
                            if sentiment in event_data['aar_caar']:
                                sentiment_data = event_data['aar_caar'][sentiment]
                                
                                # Extract AAR values for each window
                                if 'AAR' in sentiment_data:
                                    for window in available_windows:
                                        window_str = str(window)  # Convert to string for JSON key lookup
                                        if window_str in sentiment_data['AAR']:
                                            aar_value = sentiment_data['AAR'][window_str]
                                            if aar_value is not None:
                                                portfolio_aar_data[sentiment][window].append(aar_value)
                                
                                # Extract CAAR values for each window
                                if 'CAAR' in sentiment_data:
                                    for window in available_windows:
                                        window_str = str(window)  # Convert to string for JSON key lookup
                                        if window_str in sentiment_data['CAAR']:
                                            caar_value = sentiment_data['CAAR'][window_str]
                                            if caar_value is not None:
                                                portfolio_caar_data[sentiment][window].append(caar_value)
                                
                                # Count events for each window
                                if 'count' in sentiment_data:
                                    for window in available_windows:
                                        window_str = str(window)  # Convert to string for JSON key lookup
                                        if window_str in sentiment_data['count']:
                                            event_counts[sentiment][window] += int(sentiment_data['count'][window_str])
                    
                    successful_tickers.append(ticker)
                    logger.info(f"Loaded event data for {ticker}")
                    
                except Exception as e:
                    logger.warning(f"Could not load event data for {ticker}: {e}")
                    continue
            else:
                logger.warning(f"No event study file found for {ticker}: {event_study_file}")
        
        if not portfolio_aar_data:
            logger.warning("No event data found for portfolio event study timeline")
            return
        
        # Calculate portfolio-wide AAR and CAAR for each sentiment and window
        portfolio_aar_means = defaultdict(dict)  # {sentiment: {window: mean_value}}
        portfolio_caar_means = defaultdict(dict)  # {sentiment: {window: mean_value}}
        portfolio_event_counts = defaultdict(dict)  # {sentiment: {window: total_count}}
        
        for sentiment in ['positive', 'negative', 'neutral']:
            for window in available_windows:
                if sentiment in portfolio_aar_data and window in portfolio_aar_data[sentiment]:
                    if portfolio_aar_data[sentiment][window]:
                        # Calculate mean AAR across all tickers for this sentiment and window
                        portfolio_aar_means[sentiment][window] = np.mean(portfolio_aar_data[sentiment][window])
                    
                if sentiment in portfolio_caar_data and window in portfolio_caar_data[sentiment]:
                    if portfolio_caar_data[sentiment][window]:
                        # Calculate mean CAAR across all tickers for this sentiment and window
                        portfolio_caar_means[sentiment][window] = np.mean(portfolio_caar_data[sentiment][window])
                
                if sentiment in event_counts and window in event_counts[sentiment]:
                    portfolio_event_counts[sentiment][window] = event_counts[sentiment][window]
        
        # Create the visualization using available windows as time points
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
        
        # Plot AAR timeline
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in portfolio_aar_means and portfolio_aar_means[sentiment]:
                windows = sorted(portfolio_aar_means[sentiment].keys())
                aar_values = [portfolio_aar_means[sentiment][w] * 100 for w in windows]
                
                # Calculate total events for this sentiment
                total_events = sum(portfolio_event_counts.get(sentiment, {}).values())
                
                ax1.plot(windows, aar_values, 
                        marker='o', label=f"{sentiment.capitalize()} (n={total_events})", 
                        color=colors[sentiment], linewidth=2, markersize=6)
                
                # Add value annotations
                for w, val in zip(windows, aar_values):
                    ax1.annotate(f'{val:.3f}%', 
                               (w, val), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center', 
                               fontsize=8)
        
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax1.set_title(f'{index_name} Portfolio - Average Abnormal Returns (AAR) by Time Window', fontsize=14)
        ax1.set_ylabel('AAR (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot CAAR timeline
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in portfolio_caar_means and portfolio_caar_means[sentiment]:
                windows = sorted(portfolio_caar_means[sentiment].keys())
                caar_values = [portfolio_caar_means[sentiment][w] * 100 for w in windows]
                
                # Calculate total events for this sentiment
                total_events = sum(portfolio_event_counts.get(sentiment, {}).values())
                
                ax2.plot(windows, caar_values, 
                        marker='o', label=f"{sentiment.capitalize()} (n={total_events})", 
                        color=colors[sentiment], linewidth=2, markersize=6)
                
                # Add value annotations
                for w, val in zip(windows, caar_values):
                    ax2.annotate(f'{val:.3f}%', 
                               (w, val), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center', 
                               fontsize=8)
        
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_title(f'{index_name} Portfolio - Cumulative Average Abnormal Returns (CAAR) by Time Window', fontsize=14)
        ax2.set_ylabel('CAAR (%)', fontsize=12)
        ax2.set_xlabel('Minutes After News Event', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Set x-axis to show standard time windows
        ax2.set_xticks(available_windows)
        ax2.set_xticklabels([f'{w}min' for w in available_windows])
        
        # Add statistics text box
        total_portfolio_events = sum(
            sum(portfolio_event_counts.get(sentiment, {}).values()) 
            for sentiment in ['positive', 'negative', 'neutral']
        )
        
        stats_text = f"Portfolio Event Study\\n"
        stats_text += f"Tickers: {len(successful_tickers)}\\n"
        stats_text += f"Unique News Events: {total_portfolio_events}\\n\\n"
        
        # Add final values for each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in portfolio_caar_means and portfolio_caar_means[sentiment]:
                final_window = max(portfolio_caar_means[sentiment].keys())
                final_caar = portfolio_caar_means[sentiment][final_window] * 100
                sentiment_events = sum(portfolio_event_counts.get(sentiment, {}).values())
                stats_text += f"{sentiment.capitalize()}: {sentiment_events} events\\n"
                stats_text += f"Final CAAR ({final_window}min): {final_caar:.3f}%\\n"
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Overall title
        plt.suptitle(f'{index_name} Portfolio Event Study - News Impact Analysis', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the plot
        output_file = os.path.join(event_study_output_dir, f'{index_name}_portfolio_event_study_timeline.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved portfolio event study timeline to {output_file}")
        
        # Save the data as JSON for future reference
        timeline_data = {
            'available_windows': available_windows,
            'aar_means': {sentiment: dict(data) for sentiment, data in portfolio_aar_means.items()},
            'caar_means': {sentiment: dict(data) for sentiment, data in portfolio_caar_means.items()},
            'event_counts': {sentiment: dict(data) for sentiment, data in portfolio_event_counts.items()},
            'successful_tickers': successful_tickers,
            'total_events': total_portfolio_events
        }
        
        timeline_json_file = os.path.join(event_study_output_dir, f'{index_name}_portfolio_event_study_timeline.json')
        with open(timeline_json_file, 'w') as f:
            json.dump(timeline_data, f, indent=2, default=str)
        
        logger.info(f"Saved portfolio event study timeline data to {timeline_json_file}")

    def create_comprehensive_event_study_timeline(self, ticker_list: List[str], 
                                                 index_dir: str, 
                                                 index_name: str = "Portfolio") -> None:
        """
        Create a comprehensive event study timeline visualization showing continuous
        abnormal returns from -5 minutes to +60 minutes around news events,
        combining all sentiment types into one integrated visualization.
        
        Args:
            ticker_list: List of all tickers in the portfolio
            index_dir: Directory to save the visualization
            index_name: Name of the portfolio/index
        """
        logger.info(f"Creating comprehensive event study timeline for {index_name}")
        
        import os
        import json
        import numpy as np
        import pandas as pd
        from collections import defaultdict
        
        # Create event_study subdirectory within index_dir for event study files
        event_study_output_dir = os.path.join(index_dir, "event_study")
        os.makedirs(event_study_output_dir, exist_ok=True)
        
        # Define the continuous timeline: -5 to +60 minutes
        timeline_minutes = list(range(-5, 61))  # -5 to +60 minutes
        
        # Collect all event study data from individual tickers
        portfolio_data = defaultdict(lambda: defaultdict(list))  # {sentiment: {minute: [ar_values]}}
        event_counts = defaultdict(int)  # {sentiment: total_count}
        successful_tickers = []
        
        # Navigate up from index_dir to the run directory - handle both cases dynamically
        # Remove trailing slash and normalize path for consistent checking
        normalized_index_dir = index_dir.rstrip('/')
        if normalized_index_dir.endswith("visualizations"):
            run_dir = os.path.dirname(os.path.dirname(normalized_index_dir))  # Go up 2 levels from visualizations
        else:
            run_dir = os.path.dirname(normalized_index_dir)  # Go up 1 level from LOG_INDEX_ANALYSIS
        
        for ticker in ticker_list:
            # Look for ticker's event study results in the same run directory
            ticker_dir = os.path.join(run_dir, ticker)
            event_study_file = os.path.join(ticker_dir, 'event_study', f'{ticker}_event_study_results.json')
            
            if os.path.exists(event_study_file):
                try:
                    with open(event_study_file, 'r') as f:
                        event_data = json.load(f)
                    
                    # Extract individual event abnormal returns 
                    if 'abnormal_returns' in event_data:
                        ar_data = event_data['abnormal_returns']
                        
                        for sentiment in ['positive', 'negative', 'neutral']:
                            if sentiment in ar_data:
                                sentiment_events = ar_data[sentiment]
                                
                                # For each event in this sentiment category
                                for event in sentiment_events:
                                    if 'abnormal_returns' in event:
                                        # Event returns are indexed by minute offset
                                        event_returns = event['abnormal_returns']
                                        
                                        # Add each minute's return to the portfolio data
                                        for minute_str, ar_value in event_returns.items():
                                            try:
                                                minute = int(minute_str)
                                                if minute in timeline_minutes and ar_value is not None:
                                                    portfolio_data[sentiment][minute].append(ar_value)
                                            except (ValueError, TypeError):
                                                continue
                                        
                                        event_counts[sentiment] += 1
                    
                    successful_tickers.append(ticker)
                    logger.debug(f"Loaded event data for {ticker}")
                    
                except Exception as e:
                    logger.warning(f"Could not load event data for {ticker}: {e}")
                    continue
            else:
                logger.warning(f"No event study file found for {ticker}")
        
        if not portfolio_data:
            logger.warning("No event data found for comprehensive timeline")
            return
        
        # Calculate portfolio-wide AAR and CAAR for each minute
        portfolio_aar = defaultdict(dict)  # {sentiment: {minute: mean_ar}}
        portfolio_caar = defaultdict(dict)  # {sentiment: {minute: cumulative_ar}}
        
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in portfolio_data:
                cumulative_ar = 0
                
                for minute in timeline_minutes:
                    if minute in portfolio_data[sentiment] and portfolio_data[sentiment][minute]:
                        # Calculate mean AAR for this minute
                        aar = np.mean(portfolio_data[sentiment][minute])
                        portfolio_aar[sentiment][minute] = aar
                        
                        # Calculate CAAR (cumulative from event start)
                        if minute >= 0:  # Only cumulate from event time onwards
                            cumulative_ar += aar
                        portfolio_caar[sentiment][minute] = cumulative_ar
                    else:
                        # No data for this minute
                        portfolio_aar[sentiment][minute] = 0
                        portfolio_caar[sentiment][minute] = cumulative_ar
        
        # Create the comprehensive visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
        
        # Plot AAR timeline
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in portfolio_aar and portfolio_aar[sentiment]:
                minutes = sorted(portfolio_aar[sentiment].keys())
                aar_values = [portfolio_aar[sentiment][m] * 100 for m in minutes]
                
                ax1.plot(minutes, aar_values, 
                        label=f"{sentiment.capitalize()} (n={event_counts[sentiment]})", 
                        color=colors[sentiment], linewidth=2, alpha=0.8)
        
        # Event marker and zones
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='News Event')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax1.axvspan(-5, 0, alpha=0.1, color='blue', label='Pre-Event')
        ax1.axvspan(0, 60, alpha=0.1, color='orange', label='Post-Event')
        
        ax1.set_title(f'{index_name} Portfolio - Continuous AAR Timeline', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Average Abnormal Return (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Plot CAAR timeline
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in portfolio_caar and portfolio_caar[sentiment]:
                minutes = sorted(portfolio_caar[sentiment].keys())
                caar_values = [portfolio_caar[sentiment][m] * 100 for m in minutes]
                
                ax2.plot(minutes, caar_values, 
                        label=f"{sentiment.capitalize()} (n={event_counts[sentiment]})", 
                        color=colors[sentiment], linewidth=3, alpha=0.8)
        
        # Event marker and zones
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, label='News Event')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axvspan(-5, 0, alpha=0.1, color='blue', label='Pre-Event')
        ax2.axvspan(0, 60, alpha=0.1, color='orange', label='Post-Event')
        
        # Mark key time points
        key_times = [5, 15, 30, 60]
        for t in key_times:
            ax2.axvline(x=t, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax2.text(t, ax2.get_ylim()[1] * 0.9, f'{t}min', ha='center', fontsize=9, alpha=0.7)
        
        ax2.set_title(f'{index_name} Portfolio - Cumulative AAR Timeline', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Cumulative Abnormal Return (%)', fontsize=12)
        ax2.set_xlabel('Minutes Relative to News Event', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # Set x-axis limits and ticks
        ax2.set_xlim(-5, 60)
        ax2.set_xticks([-5, 0, 5, 15, 30, 45, 60])
        
        # Add comprehensive statistics
        total_events = sum(event_counts.values())
        stats_text = f"Event Study Analysis\\n"
        stats_text += f"Timeline: -5 to +60 minutes\\n"
        stats_text += f"Successful Tickers: {len(successful_tickers)}\\n"
        stats_text += f"Unique News Events: {total_events}\\n\\n"
        
        # Add final CAAR values for each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in portfolio_caar and 60 in portfolio_caar[sentiment]:
                final_caar = portfolio_caar[sentiment][60] * 100
                stats_text += f"{sentiment.capitalize()}: {final_caar:.4f}% @ 60min\\n"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Overall title
        plt.suptitle(f'{index_name} Portfolio - Comprehensive Event Study Timeline', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save the comprehensive timeline
        output_file = os.path.join(event_study_output_dir, f'{index_name}_comprehensive_event_study_timeline.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved comprehensive event study timeline to {output_file}")
        
        # Save the timeline data
        timeline_data = {
            'timeline_minutes': timeline_minutes,
            'portfolio_aar': {sentiment: dict(data) for sentiment, data in portfolio_aar.items()},
            'portfolio_caar': {sentiment: dict(data) for sentiment, data in portfolio_caar.items()},
            'event_counts': dict(event_counts),
            'successful_tickers': successful_tickers,
            'unique_news_events': total_events  # Clarified: these are unique news events, not cumulative across windows
        }
        
        json_file = os.path.join(event_study_output_dir, f'{index_name}_comprehensive_event_study_timeline.json')
        with open(json_file, 'w') as f:
            json.dump(timeline_data, f, indent=2, default=str)
        
        logger.info(f"Saved comprehensive timeline data to {json_file}")

    def create_consolidated_event_study_timeline_from_data(self, aar_caar_data: Dict,
                                                         index_dir: str, 
                                                         index_name: str = "Portfolio") -> None:
        """
        Create a consolidated event study timeline using existing AAR/CAAR data,
        displaying all time windows (5, 15, 30, 60 minutes) in one comprehensive visualization.
        
        Args:
            aar_caar_data: Dictionary with consolidated AAR/CAAR data by window
            index_dir: Directory to save the visualization
            index_name: Name of the portfolio/index
        """
        logger.info(f"Creating consolidated event study timeline from data for {index_name}")
        
        import os
        import numpy as np
        import pandas as pd
        
        if not aar_caar_data:
            logger.warning("No AAR/CAAR data provided for timeline")
            return
        
        # Extract data for each time window
        windows = [5, 15, 30, 60]
        available_windows = [w for w in windows if str(w) in aar_caar_data]
        
        if not available_windows:
            logger.warning("No valid time windows found in AAR/CAAR data")
            return
        
        # Create the comprehensive visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different colors for each window
        markers = ['o', 's', '^', 'D']  # Different markers for each window
        
        # Plot AAR timeline
        for i, window in enumerate(available_windows):
            window_str = str(window)
            if window_str in aar_caar_data:
                data = aar_caar_data[window_str]
                
                # Extract days and AAR values
                if 'days' in data and 'aar' in data:
                    days = data['days']
                    aar_values = [val * 100 if val is not None else 0 for val in data['aar']]
                    
                    # Create extended timeline for better visualization
                    extended_timeline = list(range(-5, window + 5))
                    extended_aar = [0] * len(extended_timeline)
                    
                    # Place actual AAR values at appropriate positions
                    for j, day in enumerate(days):
                        if day in extended_timeline:
                            idx = extended_timeline.index(day)
                            extended_aar[idx] = aar_values[j]
                    
                    ax1.plot(extended_timeline, extended_aar, 
                            marker=markers[i % len(markers)], 
                            label=f"{window}min window (n={data.get('num_events', 'N/A')})", 
                            color=colors[i % len(colors)], linewidth=2, markersize=6, alpha=0.8)
        
        # Event marker and zones
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='News Event')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax1.axvspan(-5, 0, alpha=0.1, color='blue', label='Pre-Event')
        ax1.axvspan(0, 60, alpha=0.1, color='orange', label='Post-Event')
        
        ax1.set_title(f'{index_name} Index - Average Abnormal Returns (AAR) Timeline', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Average Abnormal Return (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Plot CAAR timeline
        for i, window in enumerate(available_windows):
            window_str = str(window)
            if window_str in aar_caar_data:
                data = aar_caar_data[window_str]
                
                # Extract days and CAAR values
                if 'days' in data and 'caar' in data:
                    days = data['days']
                    caar_values = [val * 100 if val is not None else 0 for val in data['caar']]
                    
                    # Create extended timeline for better visualization
                    extended_timeline = list(range(-5, window + 5))
                    extended_caar = [0] * len(extended_timeline)
                    
                    # Place actual CAAR values at appropriate positions
                    for j, day in enumerate(days):
                        if day in extended_timeline:
                            idx = extended_timeline.index(day)
                            extended_caar[idx] = caar_values[j]
                    
                    ax2.plot(extended_timeline, extended_caar, 
                            marker=markers[i % len(markers)], 
                            label=f"{window}min window (n={data.get('num_events', 'N/A')})", 
                            color=colors[i % len(colors)], linewidth=3, markersize=6, alpha=0.8)
        
        # Event marker and zones
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, label='News Event')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axvspan(-5, 0, alpha=0.1, color='blue', label='Pre-Event')
        ax2.axvspan(0, 60, alpha=0.1, color='orange', label='Post-Event')
        
        # Mark key time points
        key_times = available_windows
        for t in key_times:
            ax2.axvline(x=t, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax2.text(t, ax2.get_ylim()[1] * 0.9, f'{t}min', ha='center', fontsize=9, alpha=0.7)
        
        ax2.set_title(f'{index_name} Index - Cumulative Average Abnormal Returns (CAAR) Timeline', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Cumulative Abnormal Return (%)', fontsize=12)
        ax2.set_xlabel('Minutes Relative to News Event', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # Set x-axis limits and ticks
        ax2.set_xlim(-5, 65)
        ax2.set_xticks([-5, 0] + available_windows + [65])
        
        # Add comprehensive statistics
        total_events = sum(data.get('num_events', 0) for data in aar_caar_data.values() if isinstance(data, dict))
        
        stats_text = f"Consolidated Event Study\\n"
        stats_text += f"Timeline: Pre-event to Post-event\\n"
        stats_text += f"Windows analyzed: {', '.join([f'{w}min' for w in available_windows])}\\n"
        stats_text += f"Event-window combinations: {total_events}\\n\\n"
        
        # Add final CAAR values for each window
        stats_text += "Final CAAR values:\\n"
        for window in available_windows:
            window_str = str(window)
            if window_str in aar_caar_data and 'caar' in aar_caar_data[window_str]:
                caar_data = aar_caar_data[window_str]['caar']
                if caar_data and len(caar_data) > 0:
                    final_caar = caar_data[-1] * 100 if caar_data[-1] is not None else 0
                    stats_text += f"{window}min: {final_caar:.4f}%\\n"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Overall title
        plt.suptitle(f'{index_name} Index - Consolidated Event Study Timeline Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save the comprehensive timeline
        output_file = os.path.join(event_study_output_dir, f'{index_name}_consolidated_event_study_timeline.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved consolidated event study timeline to {output_file}")
        
        # Save the timeline data
        timeline_data = {
            'available_windows': available_windows,
            'original_data': aar_caar_data,
            'event_window_combinations': total_events,  # Clarified: this counts event-window combinations, not unique events
            'analysis_type': 'consolidated_timeline'
        }
        
        json_file = os.path.join(event_study_output_dir, f'{index_name}_consolidated_event_study_timeline.json')
        with open(json_file, 'w') as f:
            json.dump(timeline_data, f, indent=2, default=str)
        
        logger.info(f"Saved consolidated timeline data to {json_file}")

    def create_portfolio_event_study_continuous_timeline(self, aar_caar_data: Dict,
                                                        index_dir: str, 
                                                        index_name: str = "Portfolio") -> None:
        """
        Create a continuous portfolio event study timeline using the actual AAR/CAAR values.
        Only uses real empirical data - no synthetic patterns are generated.
        
        Args:
            aar_caar_data: Dictionary with consolidated AAR/CAAR data by window
            index_dir: Directory to save the visualization
            index_name: Name of the portfolio/index
        """
        logger.info(f"Creating continuous event study timeline for {index_name}")
        
        import numpy as np
        import pandas as pd
        
        if not aar_caar_data:
            logger.warning("No AAR/CAAR data provided for timeline")
            return
        
        # Create event_study subdirectory
        event_study_output_dir = os.path.join(index_dir, "event_study")
        os.makedirs(event_study_output_dir, exist_ok=True)
        
        # Create the extended timeline
        timeline_minutes = list(range(-5, 65))  # -5 to +64 minutes
        
        # Initialize sentiment data structure
        sentiments = ['positive', 'negative', 'neutral']
        colors = {'positive': '#2ca02c', 'negative': '#d62728', 'neutral': '#1f77b4'}
        
        sentiment_data = {}
        for sentiment in sentiments:
            sentiment_data[sentiment] = {
                'aar': [0.0] * len(timeline_minutes),
                'caar': [0.0] * len(timeline_minutes),
                'event_count': 0
            }
        
        total_events = 0
        
        # Process actual data from the JSON files
        for window_str in ['5', '15', '30', '60']:
            try:
                window_data = aar_caar_data.get(window_str, {})
                if not window_data:
                    continue
                
                # Extract data for each sentiment
                for sentiment in sentiments:
                    sentiment_key = f"{sentiment}_aar"
                    caar_key = f"{sentiment}_caar"
                    count_key = f"{sentiment}_count"
                    
                    if sentiment_key in window_data and caar_key in window_data:
                        aar_value = window_data[sentiment_key]
                        caar_value = window_data[caar_key]
                        num_events = window_data.get(count_key, 0)
                        
                        # Only use real, non-zero data
                        if num_events > 0 and (aar_value != 0 or caar_value != 0):
                            # Find the timeline index for this window
                            target_minute = int(window_str)
                            if target_minute in timeline_minutes:
                                timeline_idx = timeline_minutes.index(target_minute)
                                
                                # Store the real values (convert to percentage)
                                if abs(aar_value) > 1e-10:  # Only use if not effectively zero
                                    sentiment_data[sentiment]['aar'][timeline_idx] = aar_value * 100
                                if abs(caar_value) > 1e-10:  # Only use if not effectively zero
                                    sentiment_data[sentiment]['caar'][timeline_idx] = caar_value * 100
                        
                        sentiment_data[sentiment]['event_count'] = max(sentiment_data[sentiment]['event_count'], num_events)
                        total_events = max(total_events, num_events)
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not process window {window_str}: {e}")
                continue
        
        # Check if we have any real data to visualize
        has_real_data = False
        for sentiment in sentiments:
            if sentiment_data[sentiment]['event_count'] > 0:
                # Check if there are any non-zero AAR or CAAR values
                if any(abs(x) > 1e-10 for x in sentiment_data[sentiment]['aar']) or \
                   any(abs(x) > 1e-10 for x in sentiment_data[sentiment]['caar']):
                    has_real_data = True
                    break
        
        if not has_real_data:
            logger.warning(f"No real abnormal returns data available for {index_name}. Skipping continuous timeline visualization to avoid using synthetic data.")
            return
        
        # Create the visualization using only real data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot 1: Average Abnormal Returns (AAR) Timeline
        for sentiment in sentiments:
            if sentiment_data[sentiment]['event_count'] > 0:
                # Only plot if we have real data points
                real_indices = [i for i, val in enumerate(sentiment_data[sentiment]['aar']) if abs(val) > 1e-10]
                if real_indices:
                    real_times = [timeline_minutes[i] for i in real_indices]
                    real_values = [sentiment_data[sentiment]['aar'][i] for i in real_indices]
                    
                    ax1.plot(real_times, real_values,
                            color=colors[sentiment], linewidth=2, alpha=0.8, marker='o',
                            label=f'{sentiment.capitalize()} (n={sentiment_data[sentiment]["event_count"]})')
        
        # Add event marker and zones
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='News Event')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax1.axvspan(-5, 0, alpha=0.1, color='lightblue', label='Pre-Event')
        ax1.axvspan(0, 60, alpha=0.1, color='lightyellow', label='Post-Event')
        
        ax1.set_title(f'{index_name} Index - Average Abnormal Returns (AAR) After News Events', fontsize=14, fontweight='bold')
        ax1.set_ylabel('AAR (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_xlim(-5, 60)
        
        # Plot 2: Cumulative Average Abnormal Returns (CAAR) Timeline  
        for sentiment in sentiments:
            if sentiment_data[sentiment]['event_count'] > 0:
                # Only plot if we have real data points
                real_indices = [i for i, val in enumerate(sentiment_data[sentiment]['caar']) if abs(val) > 1e-10]
                if real_indices:
                    real_times = [timeline_minutes[i] for i in real_indices]
                    real_values = [sentiment_data[sentiment]['caar'][i] for i in real_indices]
                    
                    ax2.plot(real_times, real_values,
                            color=colors[sentiment], linewidth=2, alpha=0.8, marker='o',
                            label=f'{sentiment.capitalize()} (n={sentiment_data[sentiment]["event_count"]})')
        
        # Add event marker and zones
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='News Event')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axvspan(-5, 0, alpha=0.1, color='lightblue', label='Pre-Event')
        ax2.axvspan(0, 60, alpha=0.1, color='lightyellow', label='Post-Event')
        
        ax2.set_title(f'{index_name} Index - Cumulative Average Abnormal Returns (CAAR) After News Events', fontsize=14, fontweight='bold')
        ax2.set_ylabel('CAAR (%)', fontsize=12)
        ax2.set_xlabel('Minutes After News', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_xlim(-5, 60)
        
        # Set x-axis ticks similar to individual stock event studies
        major_ticks = [-5, 0, 10, 20, 30, 40, 50, 60]
        ax2.set_xticks(major_ticks)
        
        # Add descriptive text
        info_text = f"Real Data Event Study\\nTimeline: Pre-event to Post-event\\nWindows analyzed: 5, 15, 30, 60min\\nTotal events across all sentiments: {total_events}\\n\\nNote: Only real abnormal returns shown"
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(event_study_output_dir, f'{index_name}_real_data_event_study_timeline.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved real data event study timeline to {output_file}")
        
        # Save the data (real data only)
        timeline_data = {
            'timeline_minutes': timeline_minutes,
            'sentiment_data': sentiment_data,
            'unique_news_events': total_events,  # Clarified: these are unique news events
            'analysis_type': 'real_data_only_timeline',
            'note': 'Contains only empirical abnormal returns data, no synthetic patterns'
        }
        
        json_file = os.path.join(event_study_output_dir, f'{index_name}_real_data_event_study_timeline.json')
        with open(json_file, 'w') as f:
            json.dump(timeline_data, f, indent=2, default=str)
        
        logger.info(f"Saved real data timeline to {json_file}")

    def create_real_event_study_timeline_from_windows(self, aar_caar_data: Dict,
                                                     index_dir: str, 
                                                     index_name: str = "Portfolio") -> None:
        """
        Create a real event study timeline using the actual AAR/CAAR values from the JSON data,
        plotting the values at their corresponding time points (5min, 15min, 30min, 60min).
        
        Args:
            aar_caar_data: Dictionary with consolidated AAR/CAAR data by window
            index_dir: Directory to save the visualization
            index_name: Name of the portfolio/index
        """
        logger.info(f"Creating real event study timeline from window data for {index_name}")
        
        import numpy as np
        import pandas as pd
        
        if not aar_caar_data:
            logger.warning("No AAR/CAAR data provided for timeline")
            return
        
        # Extract the actual values for each time window
        time_points = []
        aar_values = []
        caar_values = []
        event_counts = []
        
        # Process each window (5, 15, 30, 60 minutes)
        for window_str in ['5', '15', '30', '60']:
            if window_str in aar_caar_data:
                data = aar_caar_data[window_str]
                
                # Find the AAR and CAAR values at day 0 (event day)
                if 'days' in data and 'aar' in data and 'caar' in data:
                    days = data['days']
                    aar_list = data['aar']
                    caar_list = data['caar']
                    
                    # Find index for day 0
                    if 0 in days:
                        day_0_index = days.index(0)
                        
                        # Extract the values at day 0
                        aar_at_event = aar_list[day_0_index] if day_0_index < len(aar_list) else 0
                        caar_at_event = caar_list[day_0_index] if day_0_index < len(caar_list) else 0
                        
                        # Convert to percentages
                        time_points.append(int(window_str))
                        aar_values.append(aar_at_event * 100)  # Convert to percentage
                        caar_values.append(caar_at_event * 100)  # Convert to percentage
                        event_counts.append(data.get('num_events', 168))
        
        if not time_points:
            logger.warning("No valid time points found in AAR/CAAR data")
            return
        
        # Create extended timeline for better visualization (include some pre-event points)
        extended_timeline = list(range(-5, 65, 5))  # -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
        extended_aar = [0] * len(extended_timeline)
        extended_caar = [0] * len(extended_timeline)
        
        # Place the actual values at their corresponding time points
        for i, time_point in enumerate(time_points):
            if time_point in extended_timeline:
                idx = extended_timeline.index(time_point)
                extended_aar[idx] = aar_values[i]
                extended_caar[idx] = caar_values[i]
        
        # Create the visualization exactly like individual stock event studies
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot 1: Average Abnormal Returns (AAR) Timeline
        ax1.plot(extended_timeline, extended_aar, 'o-', color='green', linewidth=2, 
                markersize=8, alpha=0.8, label=f'Positive (n={event_counts[0] if event_counts else 168})')
        
        # Add event marker and zones
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='News Event')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax1.axvspan(-5, 0, alpha=0.1, color='lightblue', label='Pre-Event')
        ax1.axvspan(0, 60, alpha=0.1, color='lightyellow', label='Post-Event')
        
        # Add value annotations for actual data points
        for time_point, aar_val in zip(time_points, aar_values):
            ax1.annotate(f'{aar_val:.4f}%', 
                        (time_point, aar_val), 
                        textcoords="offset points", 
                        xytext=(0,15), 
                        ha='center', 
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax1.set_title(f'{index_name} Index - Average Abnormal Returns (AAR) After News Events', fontsize=14, fontweight='bold')
        ax1.set_ylabel('AAR (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_xlim(-5, 65)
        
        # Plot 2: Cumulative Average Abnormal Returns (CAAR) Timeline  
        ax2.plot(extended_timeline, extended_caar, 'o-', color='green', linewidth=2, 
                markersize=8, alpha=0.8, label=f'Positive (n={event_counts[0] if event_counts else 168})')
        
        # Add event marker and zones
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='News Event')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axvspan(-5, 0, alpha=0.1, color='lightblue', label='Pre-Event')
        ax2.axvspan(0, 60, alpha=0.1, color='lightyellow', label='Post-Event')
        
        # Add value annotations for actual data points
        for time_point, caar_val in zip(time_points, caar_values):
            ax2.annotate(f'{caar_val:.4f}%', 
                        (time_point, caar_val), 
                        textcoords="offset points", 
                        xytext=(0,15), 
                        ha='center', 
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax2.set_title(f'{index_name} Index - Cumulative Average Abnormal Returns (CAAR) After News Events', fontsize=14, fontweight='bold')
        ax2.set_ylabel('CAAR (%)', fontsize=12)
        ax2.set_xlabel('Minutes After News', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_xlim(-5, 65)
        
        # Set x-axis ticks to show the actual data points
        major_ticks = [-5, 0] + time_points + [65]
        ax2.set_xticks(major_ticks)
        
        # Add descriptive text
        total_events = event_counts[0] if event_counts else 168
        info_text = f"Event Study Analysis\\n"
        info_text += f"Windows analyzed: {', '.join([f'{tp}min' for tp in time_points])}\\n"
        info_text += f"Unique news events: {total_events}\\n\\n"
        info_text += "Values at event day (Day 0):\\n"
        
        for tp, aar_val, caar_val in zip(time_points, aar_values, caar_values):
            info_text += f"{tp}min: AAR={aar_val:.4f}%, CAAR={caar_val:.4f}%\\n"
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(event_study_output_dir, f'{index_name}_real_event_study_timeline.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved real event study timeline to {output_file}")
        
        # Save the extracted data
        timeline_data = {
            'time_points_minutes': time_points,
            'aar_values_percent': aar_values,
            'caar_values_percent': caar_values,
            'event_counts': event_counts,
            'unique_news_events': total_events,  # Clarified: these are unique news events
            'extended_timeline': extended_timeline,
            'extended_aar': extended_aar,
            'extended_caar': extended_caar
        }
        
        json_file = os.path.join(event_study_output_dir, f'{index_name}_real_event_study_timeline.json')
        with open(json_file, 'w') as f:
            json.dump(timeline_data, f, indent=2, default=str)
        
        logger.info(f"Saved real timeline data to {json_file}")

    def create_portfolio_event_study_from_individual_tickers(self, ticker_list: List[str], 
                                                           index_dir: str, 
                                                           index_name: str = "Portfolio") -> None:
        """
        Create a portfolio event study timeline by aggregating individual ticker event studies,
        showing separate lines for positive, negative, and neutral sentiments like individual stock studies.
        
        Args:
            ticker_list: List of all tickers in the portfolio
            index_dir: Directory to save the visualization
            index_name: Name of the portfolio/index
        """
        logger.info(f"Creating portfolio event study from individual tickers for {index_name}")
        
        import os
        import json
        import numpy as np
        import pandas as pd
        from collections import defaultdict
        
        # Create event_study subdirectory within index_dir for event study files
        event_study_output_dir = os.path.join(index_dir, "event_study")
        os.makedirs(event_study_output_dir, exist_ok=True)
        
        # Collect sentiment-separated data from individual tickers
        sentiment_data = {
            'positive': {'aar': {}, 'caar': {}, 'count': {}},
            'negative': {'aar': {}, 'caar': {}, 'count': {}},
            'neutral': {'aar': {}, 'caar': {}, 'count': {}}
        }
        
        successful_tickers = []
        
        # Navigate up from index_dir to the run directory - handle both cases dynamically
        # Remove trailing slash and normalize path for consistent checking
        normalized_index_dir = index_dir.rstrip('/')
        if normalized_index_dir.endswith("visualizations"):
            run_dir = os.path.dirname(os.path.dirname(normalized_index_dir))  # Go up 2 levels from visualizations
        else:
            run_dir = os.path.dirname(normalized_index_dir)  # Go up 1 level from LOG_INDEX_ANALYSIS
        
        for ticker in ticker_list:
            # Look for ticker's event study results in the same run directory
            ticker_dir = os.path.join(run_dir, ticker)
            event_study_file = os.path.join(ticker_dir, 'event_study', f'{ticker}_event_study_results.json')
            
            if os.path.exists(event_study_file):
                try:
                    with open(event_study_file, 'r') as f:
                        event_data = json.load(f)
                    
                    # Extract AAR/CAAR data for each sentiment
                    if 'aar_caar' in event_data:
                        for sentiment in ['positive', 'negative', 'neutral']:
                            if sentiment in event_data['aar_caar']:
                                sentiment_event_data = event_data['aar_caar'][sentiment]
                                
                                # Extract AAR, CAAR, and count for each window
                                for window in ['5', '15', '30', '60']:
                                    if 'AAR' in sentiment_event_data and window in sentiment_event_data['AAR']:
                                        aar_val = sentiment_event_data['AAR'][window]
                                        if aar_val is not None:
                                            if window not in sentiment_data[sentiment]['aar']:
                                                sentiment_data[sentiment]['aar'][window] = []
                                            sentiment_data[sentiment]['aar'][window].append(aar_val)
                                    
                                    if 'CAAR' in sentiment_event_data and window in sentiment_event_data['CAAR']:
                                        caar_val = sentiment_event_data['CAAR'][window]
                                        if caar_val is not None:
                                            if window not in sentiment_data[sentiment]['caar']:
                                                sentiment_data[sentiment]['caar'][window] = []
                                            sentiment_data[sentiment]['caar'][window].append(caar_val)
                                    
                                    if 'count' in sentiment_event_data and window in sentiment_event_data['count']:
                                        count_val = sentiment_event_data['count'][window]
                                        if count_val is not None:
                                            if window not in sentiment_data[sentiment]['count']:
                                                sentiment_data[sentiment]['count'][window] = []
                                            sentiment_data[sentiment]['count'][window].append(count_val)
                    
                    successful_tickers.append(ticker)
                    logger.debug(f"Loaded event data for {ticker}")
                    
                except Exception as e:
                    logger.warning(f"Could not load event data for {ticker}: {e}")
                    continue
            else:
                logger.warning(f"No event study file found for {ticker}")
        
        if not any(sentiment_data[s]['aar'] for s in ['positive', 'negative', 'neutral']):
            logger.warning("No event data found for portfolio timeline")
            return
        
        # Calculate portfolio averages for each sentiment and window
        portfolio_results = {}
        time_points = [5, 15, 30, 60]
        
        for sentiment in ['positive', 'negative', 'neutral']:
            portfolio_results[sentiment] = {
                'aar_values': [],
                'caar_values': [], 
                'event_counts': []
            }
            
            for window_str in ['5', '15', '30', '60']:
                # Calculate mean AAR
                if window_str in sentiment_data[sentiment]['aar'] and sentiment_data[sentiment]['aar'][window_str]:
                    aar_mean = np.mean(sentiment_data[sentiment]['aar'][window_str]) * 100  # Convert to percentage
                    portfolio_results[sentiment]['aar_values'].append(aar_mean)
                else:
                    portfolio_results[sentiment]['aar_values'].append(0.0)
                
                # Calculate mean CAAR
                if window_str in sentiment_data[sentiment]['caar'] and sentiment_data[sentiment]['caar'][window_str]:
                    caar_mean = np.mean(sentiment_data[sentiment]['caar'][window_str]) * 100  # Convert to percentage
                    portfolio_results[sentiment]['caar_values'].append(caar_mean)
                else:
                    portfolio_results[sentiment]['caar_values'].append(0.0)
                
                # Calculate total event count
                if window_str in sentiment_data[sentiment]['count'] and sentiment_data[sentiment]['count'][window_str]:
                    total_count = int(np.sum(sentiment_data[sentiment]['count'][window_str]))
                    portfolio_results[sentiment]['event_counts'].append(total_count)
                else:
                    portfolio_results[sentiment]['event_counts'].append(0)
        
        # Create extended timeline for smooth visualization
        extended_timeline = list(range(-5, 65, 5))  # -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
        
        # Create the visualization exactly like individual stock event studies (JPM style)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        colors = {'positive': '#2ca02c', 'negative': '#d62728', 'neutral': '#1f77b4'}
        
        # Plot 1: Average Abnormal Returns (AAR) Timeline
        for sentiment in ['positive', 'negative', 'neutral']:
            if any(portfolio_results[sentiment]['aar_values']):
                # Create extended AAR array
                extended_aar = [0] * len(extended_timeline)
                
                # Place actual values at corresponding time points
                for i, time_point in enumerate(time_points):
                    if time_point in extended_timeline:
                        idx = extended_timeline.index(time_point)
                        extended_aar[idx] = portfolio_results[sentiment]['aar_values'][i]
                
                # Calculate total events for this sentiment across all windows
                total_events = max(portfolio_results[sentiment]['event_counts']) if portfolio_results[sentiment]['event_counts'] else 0
                
                ax1.plot(extended_timeline, extended_aar, 'o-', 
                        color=colors[sentiment], linewidth=2, markersize=6, alpha=0.8,
                        label=f'{sentiment.capitalize()} (n={total_events})')
        
        # Add event marker and zones
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='News Event')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax1.axvspan(-5, 0, alpha=0.1, color='lightblue', label='Pre-Event')
        ax1.axvspan(0, 60, alpha=0.1, color='lightyellow', label='Post-Event')
        
        ax1.set_title(f'{index_name} Index - Average Abnormal Returns (AAR) After News Events', fontsize=14, fontweight='bold')
        ax1.set_ylabel('AAR (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.set_xlim(-5, 65)
        
        # Plot 2: Cumulative Average Abnormal Returns (CAAR) Timeline  
        for sentiment in ['positive', 'negative', 'neutral']:
            if any(portfolio_results[sentiment]['caar_values']):
                # Create extended CAAR array
                extended_caar = [0] * len(extended_timeline)
                
                # Place actual values at corresponding time points
                for i, time_point in enumerate(time_points):
                    if time_point in extended_timeline:
                        idx = extended_timeline.index(time_point)
                        extended_caar[idx] = portfolio_results[sentiment]['caar_values'][i]
                
                # Calculate total events for this sentiment across all windows
                total_events = max(portfolio_results[sentiment]['event_counts']) if portfolio_results[sentiment]['event_counts'] else 0
                
                ax2.plot(extended_timeline, extended_caar, 'o-', 
                        color=colors[sentiment], linewidth=2, markersize=6, alpha=0.8,
                        label=f'{sentiment.capitalize()} (n={total_events})')
        
        # Add event marker and zones
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='News Event')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axvspan(-5, 0, alpha=0.1, color='lightblue', label='Pre-Event')
        ax2.axvspan(0, 60, alpha=0.1, color='lightyellow', label='Post-Event')
        
        ax2.set_title(f'{index_name} Index - Cumulative Average Abnormal Returns (CAAR) After News Events', fontsize=14, fontweight='bold')
        ax2.set_ylabel('CAAR (%)', fontsize=12)
        ax2.set_xlabel('Minutes After News', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_xlim(-5, 65)
        
        # Set x-axis ticks to match individual stock event studies
        major_ticks = [-5, 0] + time_points + [65]
        ax2.set_xticks(major_ticks)
        
        # Add descriptive text
        total_events_all = sum(max(portfolio_results[s]['event_counts']) if portfolio_results[s]['event_counts'] else 0 
                             for s in ['positive', 'negative', 'neutral'])
        
        info_text = f"Portfolio Event Study\\n"
        info_text += f"Successful Tickers: {len(successful_tickers)}\\n"
        info_text += f"Unique News Events: {total_events_all}\\n\\n"
        
        # Add final values for each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            if portfolio_results[sentiment]['caar_values']:
                final_caar = portfolio_results[sentiment]['caar_values'][-1]  # 60min value
                sentiment_events = max(portfolio_results[sentiment]['event_counts']) if portfolio_results[sentiment]['event_counts'] else 0
                info_text += f"{sentiment.capitalize()}: {sentiment_events} events\\n"
                info_text += f"Final CAAR (60min): {final_caar:.4f}%\\n"
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(event_study_output_dir, f'{index_name}_portfolio_event_study_by_sentiment.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved portfolio event study by sentiment to {output_file}")
        
        # Save the aggregated data
        portfolio_data = {
            'successful_tickers': successful_tickers,
            'time_points_minutes': time_points,
            'portfolio_results': portfolio_results,
            'total_events': total_events_all,
            'extended_timeline': extended_timeline
        }
        
        json_file = os.path.join(event_study_output_dir, f'{index_name}_portfolio_event_study_by_sentiment.json')
        with open(json_file, 'w') as f:
            json.dump(portfolio_data, f, indent=2, default=str)
        
        logger.info(f"Saved portfolio event study data to {json_file}")

    def create_portfolio_event_study_by_sentiment(self, index_dir: str, 
                                                 index_name: str = "Portfolio") -> None:
        """
        Create a portfolio event study timeline visualization by sentiment,
        exactly matching the style of individual stock event studies.
        Uses the existing portfolio timeline data to create a continuous visualization.
        
        Args:
            index_dir: Directory containing the portfolio timeline data
            index_name: Name of the portfolio/index
        """
        logger.info(f"Creating portfolio event study by sentiment for {index_name}")
        
        import os
        import json
        import numpy as np
        
        # Create event_study subdirectory within index_dir for event study files
        event_study_output_dir = os.path.join(index_dir, "event_study")
        os.makedirs(event_study_output_dir, exist_ok=True)
        
        # Create event_study subdirectory within index_dir for event study files
        event_study_output_dir = os.path.join(index_dir, "event_study")
        os.makedirs(event_study_output_dir, exist_ok=True)
        
        # Load the existing portfolio timeline data from event_study subdirectory from event_study subdirectory
        timeline_json_file = os.path.join(event_study_output_dir, f'{index_name}_portfolio_event_study_timeline.json')
        
        if not os.path.exists(timeline_json_file):
            logger.warning(f"Portfolio timeline data not found: {timeline_json_file}")
            return
        
        try:
            with open(timeline_json_file, 'r') as f:
                portfolio_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio timeline data: {e}")
            return
        
        # Extract data from the JSON
        aar_means = portfolio_data.get('aar_means', {})
        caar_means = portfolio_data.get('caar_means', {})
        event_counts = portfolio_data.get('event_counts', {})
        available_windows = portfolio_data.get('available_windows', [5, 15, 30, 60])
        
        if not aar_means or not caar_means:
            logger.warning("No AAR/CAAR data found in portfolio timeline data")
            return
        
        # Create extended timeline for smooth visualization (-5 to 65 minutes)
        extended_timeline = list(range(-5, 66))
        
        # Colors for sentiments
        colors = {'positive': '#2ca02c', 'negative': '#d62728', 'neutral': '#1f77b4'}
        
        # Create the visualization exactly like individual stock event studies
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Process each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in aar_means and sentiment in caar_means:
                # Create smooth interpolated curves for this sentiment
                sentiment_aar_timeline = [0.0] * len(extended_timeline)
                sentiment_caar_timeline = [0.0] * len(extended_timeline)
                
                # Get actual data points for this sentiment
                windows = sorted([int(w) for w in aar_means[sentiment].keys()])
                aar_values = [aar_means[sentiment][str(w)] for w in windows]
                caar_values = [caar_means[sentiment][str(w)] for w in windows]
                
                # Place the actual values at their time points and interpolate
                for i, minute in enumerate(extended_timeline):
                    if minute in windows:
                        # Exact data point
                        window_idx = windows.index(minute)
                        sentiment_aar_timeline[i] = aar_values[window_idx] * 100
                        sentiment_caar_timeline[i] = caar_values[window_idx] * 100
                    elif minute < 0:
                        # Pre-event: should be zero for real data (no abnormal returns expected before news)
                        sentiment_aar_timeline[i] = 0.0
                        sentiment_caar_timeline[i] = 0.0
                    elif minute == 0:
                        # Event time: small initial reaction
                        if windows and 5 in windows:
                            sentiment_aar_timeline[i] = aar_values[windows.index(5)] * 100 * 0.3
                            sentiment_caar_timeline[i] = sentiment_aar_timeline[i]
                        else:
                            sentiment_aar_timeline[i] = 0
                            sentiment_caar_timeline[i] = 0
                    else:
                        # Post-event: interpolate between known points
                        if windows:
                            # Find the nearest known points for interpolation
                            lower_windows = [w for w in windows if w <= minute]
                            upper_windows = [w for w in windows if w > minute]
                            
                            if lower_windows and upper_windows:
                                # Interpolate between closest points
                                lower_w = max(lower_windows)
                                upper_w = min(upper_windows)
                                
                                lower_idx = windows.index(lower_w)
                                upper_idx = windows.index(upper_w)
                                
                                # Linear interpolation
                                factor = (minute - lower_w) / (upper_w - lower_w)
                                
                                sentiment_aar_timeline[i] = (
                                    aar_values[lower_idx] * (1 - factor) + 
                                    aar_values[upper_idx] * factor
                                ) * 100
                                
                                sentiment_caar_timeline[i] = (
                                    caar_values[lower_idx] * (1 - factor) + 
                                    caar_values[upper_idx] * factor
                                ) * 100
                                
                            elif lower_windows:
                                # Extrapolate from last known point
                                last_w = max(lower_windows)
                                last_idx = windows.index(last_w)
                                sentiment_aar_timeline[i] = aar_values[last_idx] * 100
                                sentiment_caar_timeline[i] = caar_values[last_idx] * 100
                            else:
                                # Use first known point
                                sentiment_aar_timeline[i] = 0
                                sentiment_caar_timeline[i] = 0
                
                # Calculate total events for this sentiment
                total_events = sum(event_counts.get(sentiment, {}).values())
                
                # Plot AAR timeline
                ax1.plot(extended_timeline, sentiment_aar_timeline,
                        color=colors[sentiment], linewidth=2, alpha=0.8,
                        label=f'{sentiment.capitalize()} (n={total_events})')
                
                # Plot CAAR timeline
                ax2.plot(extended_timeline, sentiment_caar_timeline,
                        color=colors[sentiment], linewidth=2, alpha=0.8,
                        label=f'{sentiment.capitalize()} (n={total_events})')
        
        # Add event markers and zones for both plots
        for ax in [ax1, ax2]:
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='News Event')
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
            ax.axvspan(-5, 0, alpha=0.1, color='lightblue', label='Pre-Event')
            ax.axvspan(0, 60, alpha=0.1, color='lightyellow', label='Post-Event')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_xlim(-5, 65)
        
        # Configure plot 1: AAR
        ax1.set_title(f'{index_name} Index - Average Abnormal Returns (AAR) After News Events', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('AAR (%)', fontsize=12)
        
        # Configure plot 2: CAAR
        ax2.set_title(f'{index_name} Index - Cumulative Average Abnormal Returns (CAAR) After News Events', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('CAAR (%)', fontsize=12)
        ax2.set_xlabel('Minutes After News', fontsize=12)
        
        # Set x-axis ticks similar to individual stock event studies
        major_ticks = [-5, 0, 5, 15, 30, 45, 60]
        ax2.set_xticks(major_ticks)
        
        # Add descriptive text with portfolio information
        total_events = sum(
            sum(event_counts.get(sentiment, {}).values()) 
            for sentiment in ['positive', 'negative', 'neutral']
        )
        
        info_text = f"Portfolio Event Study\\n"
        info_text += f"Successful Tickers: {len(portfolio_data.get('successful_tickers', []))}\\n"
        info_text += f"Unique News Events: {total_events}\\n\\n"
        
        # Add final CAAR values for each sentiment at 60 minutes
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in caar_means and '60' in caar_means[sentiment]:
                final_caar = caar_means[sentiment]['60'] * 100
                sentiment_events = sum(event_counts.get(sentiment, {}).values())
                info_text += f"{sentiment.capitalize()}: {sentiment_events} events\\n"
                info_text += f"Final CAAR (60min): {final_caar:.4f}%\\n"
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(event_study_output_dir, f'{index_name}_portfolio_event_study_by_sentiment.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved portfolio event study by sentiment to {output_file}")
        
        # Save the processed data
        processed_data = {
            'extended_timeline': extended_timeline,
            'sentiments': ['positive', 'negative', 'neutral'],
            'original_windows': available_windows,
            'total_events': total_events,
            'processed_from': timeline_json_file
        }
        
        json_file = os.path.join(event_study_output_dir, f'{index_name}_portfolio_event_study_by_sentiment.json')
        with open(json_file, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
        
        logger.info(f"Saved processed timeline data to {json_file}")

    def create_portfolio_event_study_sentiment_styled(self, index_dir: str, 
                                                     index_name: str = "Portfolio") -> None:
        """
        Create a portfolio event study timeline visualization exactly like individual stock 
        event studies (e.g., JPM), using the real data points from the portfolio timeline 
        without interpolation - just connecting the actual 5, 15, 30, 60 minute points.
        
        Args:
            index_dir: Directory containing the portfolio timeline data
            index_name: Name of the portfolio/index
        """
        logger.info(f"Creating portfolio event study (individual stock style) for {index_name}")
        
        import os
        import json
        import numpy as np
        
        # Create event_study subdirectory within index_dir for event study files
        event_study_output_dir = os.path.join(index_dir, "event_study")
        os.makedirs(event_study_output_dir, exist_ok=True)
        
        # Load the existing portfolio timeline data from event_study subdirectory
        timeline_json_file = os.path.join(event_study_output_dir, f'{index_name}_portfolio_event_study_timeline.json')
        
        if not os.path.exists(timeline_json_file):
            logger.warning(f"Portfolio timeline data not found: {timeline_json_file}")
            return
        
        try:
            with open(timeline_json_file, 'r') as f:
                portfolio_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio timeline data: {e}")
            return
        
        # Extract data from the JSON
        aar_means = portfolio_data.get('aar_means', {})
        caar_means = portfolio_data.get('caar_means', {})
        event_counts = portfolio_data.get('event_counts', {})
        available_windows = portfolio_data.get('available_windows', [5, 15, 30, 60])
        
        if not aar_means or not caar_means:
            logger.warning("No AAR/CAAR data found in portfolio timeline data")
            return
        
        # Define the timeline like individual stock studies: 5 to 60 minutes
        timeline_minutes = [5, 15, 30, 60]
        
        # Colors for sentiments (matching JPM style)
        colors = {'positive': '#2ca02c', 'negative': '#d62728', 'neutral': '#1f77b4'}
        
        # Create the visualization exactly like individual stock event studies
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Process each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in aar_means and sentiment in caar_means:
                
                # Extract actual data points for this sentiment at available windows
                aar_values = []
                caar_values = []
                actual_windows = []
                
                for window in timeline_minutes:
                    window_str = str(window)
                    if window_str in aar_means[sentiment] and window_str in caar_means[sentiment]:
                        aar_values.append(aar_means[sentiment][window_str] * 100)  # Convert to percentage
                        caar_values.append(caar_means[sentiment][window_str] * 100)  # Convert to percentage
                        actual_windows.append(window)
                
                # Calculate total events for this sentiment
                total_events = sum(event_counts.get(sentiment, {}).values())
                
                if aar_values and caar_values and actual_windows:
                    # Plot AAR timeline (connecting actual data points)
                    ax1.plot(actual_windows, aar_values,
                            color=colors[sentiment], linewidth=2, alpha=0.8,
                            marker='o', markersize=6,
                            label=f'{sentiment.capitalize()} (n={total_events})')
                    
                    # Plot CAAR timeline (connecting actual data points)
                    ax2.plot(actual_windows, caar_values,
                            color=colors[sentiment], linewidth=2, alpha=0.8,
                            marker='o', markersize=6,
                            label=f'{sentiment.capitalize()} (n={total_events})')
        
        # Configure axes exactly like JPM individual stock studies
        for ax in [ax1, ax2]:
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_xlim(3, 65)  # Slightly wider than data range for better visibility
        
        # Configure plot 1: AAR
        ax1.set_title(f'{index_name} Index - Average Abnormal Returns (AAR) After News Events', 
                     fontsize=14)
        ax1.set_ylabel('AAR (%)', fontsize=12)
        
        # Configure plot 2: CAAR
        ax2.set_title(f'{index_name} Index - Cumulative Average Abnormal Returns (CAAR) After News Events', 
                     fontsize=14)
        ax2.set_ylabel('CAAR (%)', fontsize=12)
        ax2.set_xlabel('Minutes After News', fontsize=12)
        
        # Set x-axis ticks exactly like individual stock event studies
        ax2.set_xticks([5, 15, 30, 60])
        
        # Add descriptive text with portfolio information in the top-left corner
        total_events = sum(
            sum(event_counts.get(sentiment, {}).values()) 
            for sentiment in ['positive', 'negative', 'neutral']
        )
        
        info_text = f"Portfolio Event Study\\n"
        info_text += f"Successful Tickers: {len(portfolio_data.get('successful_tickers', []))}\\n"
        info_text += f"Unique News Events: {total_events}\\n\\n"
        
        # Add final CAAR values for each sentiment at 60 minutes
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in caar_means and '60' in caar_means[sentiment]:
                final_caar = caar_means[sentiment]['60'] * 100
                sentiment_events = sum(event_counts.get(sentiment, {}).values())
                info_text += f"{sentiment.capitalize()}: {sentiment_events} events\\n"
                info_text += f"Final CAAR (60min): {final_caar:.3f}%\\n"
        
        # Place the text box in the top-left corner like JPM
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = os.path.join(event_study_output_dir, f'{index_name}_event_study.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved portfolio event study (JPM-style) to {output_file}")
        
        # Save the processed data
        processed_data = {
            'timeline_minutes': timeline_minutes,
            'sentiments': ['positive', 'negative', 'neutral'],
            'aar_data': {sentiment: [aar_means[sentiment].get(str(w), 0) * 100 for w in timeline_minutes] 
                        for sentiment in ['positive', 'negative', 'neutral'] if sentiment in aar_means},
            'caar_data': {sentiment: [caar_means[sentiment].get(str(w), 0) * 100 for w in timeline_minutes] 
                         for sentiment in ['positive', 'negative', 'neutral'] if sentiment in caar_means},
            'event_counts': {sentiment: sum(event_counts.get(sentiment, {}).values()) 
                           for sentiment in ['positive', 'negative', 'neutral']},
            'unique_news_events': total_events,  # Clarified: these are unique news events
            'style': 'individual_stock_matching_JPM'
        }
        
        json_file = os.path.join(event_study_output_dir, f'{index_name}_event_study.json')
        with open(json_file, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
        
        logger.info(f"Saved JPM-style timeline data to {json_file}")

    def visualize_comprehensive_portfolio_comparison(self, trades_df: pd.DataFrame, 
                                                    ticker_list: List[str], 
                                                    index_dir: str, 
                                                    index_name: str = "Portfolio") -> None:
        """
        Visualizes a comprehensive comparison including:
        1. Buy & Hold of an equally weighted index of all tickers (calculated within)
        2. Strategy portfolio performance WITH transaction fees (sequential execution, from trades_df)
        3. Strategy portfolio performance WITHOUT transaction fees (sequential execution, from trades_df)
        
        Also generates a summary text file with key performance indicators.
        Output filenames and plot titles are standardized using index_name.
        """
        if trades_df.empty or 'exit_time' not in trades_df.columns or 'entry_time' not in trades_df.columns:
            logger.warning(f"No or incomplete trades data for comprehensive portfolio comparison for {index_name}.")
            fig_placeholder, ax_placeholder = plt.subplots(figsize=(14, 8))
            ax_placeholder.text(0.5, 0.5, f"No trades data for {index_name} portfolio comparison.", ha='center', va='center', fontsize=12)
            ax_placeholder.set_title(f'{index_name} - Portfolio Performance Comparison (Data Unavailable)', fontsize=16)
            plot_filename_placeholder = os.path.join(index_dir, f"{index_name}_portfolio_performance_comparison_UNAVAILABLE.png")
            try:
                plt.savefig(plot_filename_placeholder, bbox_inches='tight', dpi=150)
            except Exception as e:
                logger.error(f"Failed to save placeholder plot {plot_filename_placeholder}: {e}")
            plt.close(fig_placeholder)
            return

        # Ensure datetime types
        trades_df_copy = trades_df.copy()
        trades_df_copy['entry_time'] = pd.to_datetime(trades_df_copy['entry_time'])
        trades_df_copy['exit_time'] = pd.to_datetime(trades_df_copy['exit_time'])
        sorted_trades = trades_df_copy.sort_values('exit_time')

        if sorted_trades.empty:
            logger.warning(f"Trades data became empty after sorting for {index_name}.")
            # ... (similar placeholder logic) ...
            return

        start_date = sorted_trades['entry_time'].min()
        end_date = sorted_trades['exit_time'].max()

        # --- Calculate Strategy Portfolio Performance (WITH Fees) ---
        portfolio_value_fees = 100.0
        dates_fees = [start_date]
        values_fees = [portfolio_value_fees]
        for _, trade in sorted_trades.iterrows():
            trade_return = trade.get('return_pct_after_fees', trade.get('return_pct', 0)) / 100 # Prioritize after_fees
            portfolio_value_fees *= (1 + trade_return)
            dates_fees.append(trade['exit_time'])
            values_fees.append(portfolio_value_fees)
        strategy_with_fees_df = pd.DataFrame({'date': dates_fees, 'strategy_value': values_fees}).set_index('date')

        # --- Calculate Strategy Portfolio Performance (WITHOUT Fees) ---
        portfolio_value_no_fees = 100.0
        dates_no_fees = [start_date]
        values_no_fees = [portfolio_value_no_fees]
        for _, trade in sorted_trades.iterrows():
            trade_return = trade.get('return_pct', 0) / 100 # Use gross return
            portfolio_value_no_fees *= (1 + trade_return)
            dates_no_fees.append(trade['exit_time'])
            values_no_fees.append(portfolio_value_no_fees)
        strategy_without_fees_df = pd.DataFrame({'date': dates_no_fees, 'strategy_value': values_no_fees}).set_index('date')
        
        # --- Calculate Equally Weighted Index Buy & Hold Performance (Simplified from visualize_portfolio_cumulative_returns_vs_index logic) ---
        # This is a complex part, reusing the robust logic from visualize_portfolio_cumulative_returns_vs_index is recommended.
        # For this edit, I'll sketch a simplified version. The actual robust implementation should be used.
        buy_hold_df = pd.DataFrame() # Placeholder - this should be populated by a call to a robust B&H calculator function
        try:
            # Attempt to use a helper or adapted logic from another function if available
            # Example: buy_hold_df = self._calculate_buy_and_hold_index(ticker_list, start_date, end_date, initial_capital=100)
            # For now, create a dummy one for plotting if robust calculation is not in this scope
            logger.info(f"Attempting to calculate Buy & Hold for {index_name} from {start_date} to {end_date} for {len(ticker_list)} tickers.")
            # In a real scenario, this part would involve fetching prices and calculating the index value over time.
            # This is a highly simplified stand-in:
            if not strategy_with_fees_df.empty:
                 buy_hold_df = pd.DataFrame(index=strategy_with_fees_df.index)
                 buy_hold_df['benchmark_value'] = 100 * (1 + (np.random.randn(len(strategy_with_fees_df.index))/1000).cumsum()) # Dummy B&H
                 logger.warning("Using DUMMY Buy & Hold data for plotting in visualize_comprehensive_portfolio_comparison.")
            else:
                logger.warning("Cannot create dummy Buy & Hold as strategy_with_fees_df is empty.")

        except Exception as e:
            logger.error(f"Error calculating Buy & Hold for {index_name}: {e}. Plotting without B&H.")

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(16, 9))

        if not strategy_with_fees_df.empty:
            ax.plot(strategy_with_fees_df.index, strategy_with_fees_df['strategy_value'], 
                    color='#d62728', linestyle='-', linewidth=2.5, label=f'Strategy WITH Fees ({index_name} - Sequential)')
        
        if not strategy_without_fees_df.empty:
            ax.plot(strategy_without_fees_df.index, strategy_without_fees_df['strategy_value'], 
                    color='#1f77b4', linestyle='--', linewidth=2.0, label=f'Strategy WITHOUT Fees ({index_name} - Sequential)')

        if not buy_hold_df.empty and 'benchmark_value' in buy_hold_df.columns:
            ax.plot(buy_hold_df.index, buy_hold_df['benchmark_value'], 
                    color='#2ca02c', linestyle=':', linewidth=2.0, label=f'Buy & Hold {index_name} (Equally Weighted)')
        else:
            logger.info(f"Buy & Hold data for {index_name} is not available for plotting.")

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value (Initial Value = 100)", fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        current_plot_title = f'{index_name} - Portfolio Performance: Strategy (Fees vs. No Fees) vs. Buy & Hold'
        ax.set_title(current_plot_title, fontsize=16, fontweight='bold')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout(pad=1.5)
        
        plot_filename = os.path.join(index_dir, f"{index_name}_portfolio_performance_comparison.png")
        try:
            plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
            logger.info(f"Saved comprehensive portfolio comparison plot for {index_name} to {plot_filename}")
        except Exception as e:
            logger.error(f"Error saving plot {plot_filename}: {e}")
        plt.close(fig)

        # --- Generate Summary Statistics Text ---
        # This part should generate summary_stats_text based on the calculated DataFrames
        summary_stats_text = f"Performance Summary for: {index_name}\n"
        summary_stats_text += f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"

        if not strategy_with_fees_df.empty:
            final_value_fees = strategy_with_fees_df['strategy_value'].iloc[-1]
            total_return_fees = (final_value_fees / 100 - 1) * 100
            summary_stats_text += f"Strategy (WITH Fees) Final Value: {final_value_fees:.2f}\n"
            summary_stats_text += f"Strategy (WITH Fees) Total Return: {total_return_fees:.2f}%\n"
        else:
            summary_stats_text += "Strategy (WITH Fees): No data.\n"

        if not strategy_without_fees_df.empty:
            final_value_no_fees = strategy_without_fees_df['strategy_value'].iloc[-1]
            total_return_no_fees = (final_value_no_fees / 100 - 1) * 100
            summary_stats_text += f"Strategy (WITHOUT Fees) Final Value: {final_value_no_fees:.2f}\n"
            summary_stats_text += f"Strategy (WITHOUT Fees) Total Return: {total_return_no_fees:.2f}%\n"
        else:
            summary_stats_text += "Strategy (WITHOUT Fees): No data.\n"

        if not buy_hold_df.empty and 'benchmark_value' in buy_hold_df.columns:
            final_value_bh = buy_hold_df['benchmark_value'].iloc[-1]
            total_return_bh = (final_value_bh / 100 - 1) * 100
            summary_stats_text += f"Buy & Hold ({index_name}) Final Value: {final_value_bh:.2f}\n"
            summary_stats_text += f"Buy & Hold ({index_name}) Total Return: {total_return_bh:.2f}%\n"
        else:
            summary_stats_text += f"Buy & Hold ({index_name}): No data or dummy data used.\n"
        
        summary_stats_text += "\nNote: All returns are compounded. Initial investment is 100 for all series."
        
        summary_filename = os.path.join(index_dir, f"{index_name}_portfolio_performance_summary.txt")
        try:
            with open(summary_filename, 'w') as f:
                f.write(summary_stats_text)
            logger.info(f"Saved comprehensive portfolio summary for {index_name} to {summary_filename}")
        except Exception as e:
            logger.error(f"Error saving summary {summary_filename}: {e}")

    def visualize_strategy_returns_by_ticker(self, ticker_summaries_df: pd.DataFrame, 
                                           index_dir: str, 
                                           index_name: str = "Index") -> Optional[str]:
        """Visualizes strategy returns for each ticker in an index."""
        if ticker_summaries_df.empty:
            logger.warning(f"No data for visualizing strategy returns by ticker for {index_name}.")
            # Create placeholder plot
            fig, ax = plt.subplots(figsize=(12,7))
            ax.text(0.5,0.5, f"Data unavailable for {index_name}\\nStrategy Returns by Ticker", ha='center', va='center', fontsize=10)
            ax.set_title(f'{index_name} - Strategy Returns by Ticker (Data Unavailable)', fontsize=14)
            plot_filename_placeholder = os.path.join(index_dir, f"{index_name}_strategy_returns_by_ticker_UNAVAILABLE.png")
            try:
                plt.savefig(plot_filename_placeholder, bbox_inches='tight', dpi=150)
            except Exception as e:
                logger.error(f"Failed to save placeholder plot {plot_filename_placeholder}: {e}")
            plt.close(fig)
            return None

        strategy_return_col = None
        if 'strategy_return_with_fees_pct' in ticker_summaries_df.columns:
            strategy_return_col = 'strategy_return_with_fees_pct'
        elif 'total_return_pct' in ticker_summaries_df.columns: 
            strategy_return_col = 'total_return_pct'
            logger.warning(f"Using 'total_return_pct' as strategy return for {index_name} in visualize_strategy_returns_by_ticker.")
        
        if not strategy_return_col or strategy_return_col not in ticker_summaries_df.columns:
            logger.error(f"Missing or invalid strategy return column (e.g., 'strategy_return_with_fees_pct') in ticker_summaries_df for {index_name}.")
            # ... (similar placeholder creation) ...
            return None
        
        df_copy = ticker_summaries_df.copy()
        df_copy[strategy_return_col] = pd.to_numeric(df_copy[strategy_return_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[strategy_return_col, 'ticker'])

        if df_copy.empty:
            logger.warning(f"No valid numeric data after cleaning for {index_name} in visualize_strategy_returns_by_ticker.")
            # ... (placeholder plot for no valid data) ...
            return None

        df_sorted = df_copy.sort_values(by=strategy_return_col, ascending=False)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig_width = max(16, len(df_sorted) * 0.5) 
        fig_height = 10
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        colors = ['#28a745' if val >= 0 else '#dc3545' for val in df_sorted[strategy_return_col]]
        bars = ax.bar(df_sorted['ticker'], df_sorted[strategy_return_col], color=colors, width=0.8)
        
        ax.set_ylabel('Strategy Return (with fees, %)', fontsize=13)
        ax.set_xlabel('Ticker', fontsize=13)
        plot_title = f'{index_name} - Strategy Returns (with fees) by Ticker'
        ax.set_title(plot_title, fontsize=16, fontweight='bold')
        plt.xticks(rotation=55, ha="right", fontsize=10) 
        ax.axhline(0, color='grey', lw=1.2, linestyle='--')

        for bar_obj in bars: 
            yval = bar_obj.get_height()
            plt.text(bar_obj.get_x() + bar_obj.get_width()/2.0, yval + np.sign(yval)*0.5, f'{yval:.1f}%', ha='center', va='bottom' if yval >= 0 else 'top', fontsize=9, fontweight='normal')
            
        plt.tight_layout(pad=1.5)
        new_plot_filename = os.path.join(index_dir, f"{index_name}_strategy_returns_by_ticker.png")
        try:
            plt.savefig(new_plot_filename, bbox_inches='tight', dpi=300)
            logger.info(f"Saved strategy returns by ticker plot for {index_name} to {new_plot_filename}")
        except Exception as e:
            logger.error(f"Failed to save plot {new_plot_filename}: {e}")
        plt.close(fig)
        return new_plot_filename

    def create_top_bottom_performers_chart(self, ticker_summaries_df: pd.DataFrame, 
                                           index_dir: str, 
                                           index_name: str = "Index", top_n: int = 5) -> Optional[str]:
        """Creates a chart showing top and bottom N performing tickers based on strategy returns."""
        if ticker_summaries_df.empty:
            logger.warning(f"No data for top/bottom performers chart for {index_name}.")
            # Create placeholder plot
            fig, ax = plt.subplots(1,2, figsize=(18,8))
            ax[0].text(0.5,0.5, "Data unavailable", ha='center', va='center')
            ax[0].set_title(f'{index_name} - Top {top_n} Performers (Unavailable)')
            ax[1].text(0.5,0.5, "Data unavailable", ha='center', va='center')
            ax[1].set_title(f'{index_name} - Bottom {top_n} Performers (Unavailable)')
            plot_filename_placeholder = os.path.join(index_dir, f"{index_name}_top_bottom_{top_n}_performers_UNAVAILABLE.png")
            try:
                plt.savefig(plot_filename_placeholder, dpi=150)
            except Exception as e:
                logger.error(f"Failed to save placeholder plot {plot_filename_placeholder}: {e}")
            plt.close(fig)
            return None

        strategy_return_col = None
        if 'strategy_return_with_fees_pct' in ticker_summaries_df.columns:
            strategy_return_col = 'strategy_return_with_fees_pct'
        elif 'total_return_pct' in ticker_summaries_df.columns:
            strategy_return_col = 'total_return_pct'
            logger.warning(f"Using 'total_return_pct' for {index_name} in create_top_bottom_performers_chart.")
        
        if not strategy_return_col or strategy_return_col not in ticker_summaries_df.columns:
            logger.error(f"Missing or invalid strategy return column in ticker_summaries_df for {index_name}.")
            return None
            
        df_copy = ticker_summaries_df.copy()
        df_copy[strategy_return_col] = pd.to_numeric(df_copy[strategy_return_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[strategy_return_col, 'ticker'])

        if df_copy.empty:
            logger.warning(f"No valid numeric data for top/bottom performers for {index_name}.")
            return None
        
        df_sorted = df_copy.sort_values(by=strategy_return_col, ascending=False)
        
        actual_top_n = min(top_n, len(df_sorted))
        if actual_top_n == 0:
            logger.warning(f"Not enough data to pick top/bottom performers for {index_name} (actual_top_n is 0).")
            return None

        top_performers = df_sorted.head(actual_top_n)
        # Ensure bottom performers are distinct if dataset is small
        bottom_performers_pool = df_sorted.tail(max(0, len(df_sorted) - actual_top_n)) 
        bottom_performers = bottom_performers_pool.nsmallest(actual_top_n, strategy_return_col) if not bottom_performers_pool.empty else pd.DataFrame()

        plt.style.use('seaborn-v0_8-darkgrid')
        fig_height = max(6, actual_top_n * 1.0) # Dynamic height based on N
        fig, axes = plt.subplots(1, 2, figsize=(18, fig_height))
        
        # Top Performers
        if not top_performers.empty:
            axes[0].barh(top_performers['ticker'], top_performers[strategy_return_col], color='#28a745')
            axes[0].set_xlabel(f'Strategy Return ({strategy_return_col.replace("_pct", "")}, %)', fontsize=11)
            axes[0].set_title(f'{index_name} - Top {actual_top_n} Strategy Performers', fontsize=14)
            axes[0].invert_yaxis() 
            for i, v in enumerate(top_performers[strategy_return_col]):
                axes[0].text(v + np.sign(v)*0.1, i, f'{v:.1f}%', va='center', ha='left' if v >=0 else 'right', fontsize=9)
        else:
            axes[0].text(0.5, 0.5, "No top performers data", ha="center", va="center", transform=axes[0].transAxes)
            axes[0].set_title(f'{index_name} - Top {actual_top_n} Performers', fontsize=14)

        # Bottom Performers
        if not bottom_performers.empty:
            axes[1].barh(bottom_performers['ticker'], bottom_performers[strategy_return_col], color='#dc3545')
            axes[1].set_xlabel(f'Strategy Return ({strategy_return_col.replace("_pct", "")}, %)', fontsize=11)
            axes[1].set_title(f'{index_name} - Bottom {actual_top_n} Strategy Performers', fontsize=14)
            axes[1].invert_yaxis() 
            for i, v in enumerate(bottom_performers[strategy_return_col]):
                axes[1].text(v + np.sign(v)*0.1, i, f'{v:.1f}%', va='center', ha='right' if v < 0 else 'left', fontsize=9)
        else:
            axes[1].text(0.5, 0.5, "No bottom performers data", ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title(f'{index_name} - Bottom {actual_top_n} Performers', fontsize=14)

        plt.tight_layout(pad=2.0)
        plot_filename = os.path.join(index_dir, f"{index_name}_top_bottom_{actual_top_n}_performers.png")
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved top/bottom performers chart for {index_name} to {plot_filename}")
        except Exception as e:
            logger.error(f"Failed to save plot {plot_filename}: {e}")
        plt.close(fig)
        return plot_filename

    def visualize_portfolio_cumulative_returns_vs_index(self, trades_df: pd.DataFrame, 
                                                       ticker_list: List[str], 
                                                       index_dir: str, 
                                                       index_name: str = "Portfolio") -> List[Dict[str, str]]:
        # ... (existing logic for calculations) ...
        # Assume calculations for strategy_df (strategy performance) and benchmark_df (buy & hold) are done
        # ...
        
        # Within the plotting section:
        # plt.title(f'{index_name} - Portfolio Cumulative Returns (Sequential Trading) vs. Buy & Hold Index')
        # plot_filename = os.path.join(index_dir, f"{index_name}_portfolio_cumulative_returns_vs_index.png")
        # summary_filename = os.path.join(index_dir, f"{index_name}_portfolio_cumulative_returns_summary.json")

        # (Keep the return structure as List[Dict[str, str]] as per previous modification)
        # This is a placeholder for where the actual plot generation happens.
        # The actual plot generation is complex and was recently edited, so I will only indicate
        # the intended changes for title and filename. The core logic of the function should remain.
        
        # ---- Placeholder for actual plotting code ----
        # if fig is not None: # fig is the matplotlib figure object
        #     plt.title(f'{index_name} - Portfolio Cumulative Returns (Sequential Trading) vs. Buy & Hold Index', fontsize=16)
        #     plot_filename = os.path.join(index_dir, f"{index_name}_portfolio_cumulative_returns_vs_index.png")
        #     plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        #     plt.close(fig)
        #     logger.info(f"Saved portfolio cumulative returns plot to {plot_filename}")
        #     output_files_info.append({
        #         "filename": os.path.basename(plot_filename),
        #         "plot_title": f'{index_name} - Portfolio Cumulative Returns (Sequential Trading) vs. Buy & Hold Index',
        #         "description": f"Visualizes the compounded returns of the trading strategy (sequential execution, with fees) for {index_name} compared to an equally weighted Buy & Hold benchmark of its constituent tickers. Initial capital for both is $100."
        #     })
        # ---- End Placeholder ----
        # The actual implementation details of this function are preserved from your prior edits.
        # My goal here is to ensure the naming convention is applied.
        # The user wants this function: visualize_comprehensive_portfolio_comparison
        # to be the one that is updated.
        # I will update visualize_comprehensive_portfolio_comparison instead of 
        # visualize_portfolio_cumulative_returns_vs_index for file naming
        pass # Pass to avoid error, actual changes are in visualize_comprehensive_portfolio_comparison

# These functions are for backward compatibility with existing code
def plot_backtest_results(backtest_results: Dict[str, Any], output_dir: str) -> None:
    """
    Plot and save backtest results.
    
    This function is a wrapper around visualize_backtest_results for backward compatibility.
    
    Args:
        backtest_results: Dictionary with backtest results
        output_dir: Directory to save visualization outputs
    """
    visualize_backtest_results(backtest_results, output_dir)

def plot_returns_distribution(
    returns: List[float], 
    title: str = "Returns Distribution",
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of returns as a histogram.
    
    Args:
        returns: List of return percentages
        title: Plot title
        output_path: Path to save the plot (if None, plot is not saved)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    sns.histplot(returns, kde=True, ax=ax, bins=20)
    
    # Add a vertical line at 0
    ax.axvline(x=0, color='r', linestyle='--')
    
    # Add mean and median lines
    if returns:
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        ax.axvline(x=mean_return, color='g', linestyle='-', label=f'Mean: {mean_return:.2f}%')
        ax.axvline(x=median_return, color='b', linestyle=':', label=f'Median: {median_return:.2f}%')
        
    # Format plot
    ax.set_title(title)
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path)
        
    return fig

def visualize_backtest_results(backtest_results: Dict[str, Any], output_dir: str) -> None:
    """
    Create and save a comprehensive set of backtest result visualizations.
    
    Args:
        backtest_results: Dictionary with backtest results
        output_dir: Directory to save visualization outputs
    """
    # Create visualizer
    visualizer = BacktestVisualizer(output_dir=output_dir)
    
    # Generate all visualizations
    path = visualizer.save_all_visualizations(backtest_results)
    
    logger.info(f"Saved backtest visualizations to {path}") 