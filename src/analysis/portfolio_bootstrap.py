"""
Portfolio Bootstrap Analysis Module

Advanced statistical testing for portfolio-wide trading strategy performance
including Sharpe ratio bootstrap, drawdown analysis, and consolidated P-values.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioBootstrapAnalyzer:
    """
    Comprehensive portfolio-wide bootstrap analysis for trading strategies.
    """
    
    def __init__(self, num_simulations: int = 10000, significance_level: float = 0.05):
        self.num_simulations = num_simulations
        self.significance_level = significance_level
        
    def load_consolidated_trades(self, trades_file: str) -> pd.DataFrame:
        """Load consolidated trades from CSV file."""
        logger.info(f"Loading consolidated trades from {trades_file}")
        try:
            df = pd.read_csv(trades_file)
            logger.info(f"Loaded {len(df)} trades from consolidated file")
            return df
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            raise
    
    def calculate_portfolio_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive portfolio performance metrics."""
        returns = df['return_pct_after_fees'].dropna().values
        
        # Basic metrics
        total_return = np.sum(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Risk-free rate set to 0% for short-term trading strategies
        risk_free_rate = 0.0
        
        # Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        
        # Calculate drawdown series
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else std_return
        sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_trades': len(returns),
            'total_return': total_return,
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'profit_factor': np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.sum(returns < 0) < 0 else np.inf
        }
    
    def bootstrap_portfolio_returns(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Bootstrap test for portfolio returns with comprehensive metrics.
        """
        logger.info(f"Running portfolio bootstrap with {self.num_simulations} simulations")
        
        # Observed metrics
        observed_mean = np.mean(returns)
        observed_sharpe = self._calculate_sharpe(returns)
        observed_sortino = self._calculate_sortino(returns)
        observed_max_dd = self._calculate_max_drawdown(returns)
        
        # Bootstrap simulations
        bootstrap_means = []
        bootstrap_sharpes = []
        bootstrap_sortinos = []
        bootstrap_max_dds = []
        
        for i in range(self.num_simulations):
            if i % 1000 == 0:
                logger.info(f"Bootstrap simulation {i}/{self.num_simulations}")
            
            # Sample with replacement
            bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
            
            bootstrap_means.append(np.mean(bootstrap_sample))
            bootstrap_sharpes.append(self._calculate_sharpe(bootstrap_sample))
            bootstrap_sortinos.append(self._calculate_sortino(bootstrap_sample))
            bootstrap_max_dds.append(self._calculate_max_drawdown(bootstrap_sample))
        
        # Convert to arrays for easier calculation
        bootstrap_means = np.array(bootstrap_means)
        bootstrap_sharpes = np.array(bootstrap_sharpes)
        bootstrap_sortinos = np.array(bootstrap_sortinos)
        bootstrap_max_dds = np.array(bootstrap_max_dds)
        
        # Calculate p-values (two-tailed tests)
        mean_p_value = 2 * min(
            np.mean(bootstrap_means <= observed_mean),
            np.mean(bootstrap_means >= observed_mean)
        )
        
        sharpe_p_value = np.mean(bootstrap_sharpes <= 0) if observed_sharpe > 0 else 1.0
        sortino_p_value = np.mean(bootstrap_sortinos <= 0) if observed_sortino > 0 else 1.0
        
        # For max drawdown, test if significantly different from random
        dd_p_value = np.mean(bootstrap_max_dds >= observed_max_dd)
        
        # Confidence intervals
        results = {
            'mean_return': {
                'observed': observed_mean,
                'p_value': mean_p_value,
                'ci_lower': np.percentile(bootstrap_means, 2.5),
                'ci_upper': np.percentile(bootstrap_means, 97.5),
                'significant': mean_p_value < self.significance_level
            },
            'sharpe_ratio': {
                'observed': observed_sharpe,
                'p_value': sharpe_p_value,
                'ci_lower': np.percentile(bootstrap_sharpes, 2.5),
                'ci_upper': np.percentile(bootstrap_sharpes, 97.5),
                'significant': sharpe_p_value < self.significance_level
            },
            'sortino_ratio': {
                'observed': observed_sortino,
                'p_value': sortino_p_value,
                'ci_lower': np.percentile(bootstrap_sortinos, 2.5),
                'ci_upper': np.percentile(bootstrap_sortinos, 97.5),
                'significant': sortino_p_value < self.significance_level
            },
            'max_drawdown': {
                'observed': observed_max_dd,
                'p_value': dd_p_value,
                'ci_lower': np.percentile(bootstrap_max_dds, 2.5),
                'ci_upper': np.percentile(bootstrap_max_dds, 97.5),
                'significant': dd_p_value < self.significance_level
            }
        }
        
        return results
    
    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio. Using 0% risk-free rate for short-term trading returns."""
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        return (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
    
    def _calculate_sortino(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio. Using 0% risk-free rate for short-term trading returns."""
        mean_return = np.mean(returns)
        downside_returns = returns[returns < risk_free_rate]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else np.std(returns, ddof=1)
        return (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        return np.min(drawdowns)
    
    def analyze_drawdowns(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive drawdown analysis.
        """
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        
        # Find drawdown periods
        is_drawdown = drawdowns < 0
        drawdown_starts = []
        drawdown_ends = []
        drawdown_depths = []
        drawdown_durations = []
        
        in_drawdown = False
        start_idx = 0
        
        for i, is_dd in enumerate(is_drawdown):
            if is_dd and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                start_idx = i
            elif not is_dd and in_drawdown:
                # End of drawdown
                in_drawdown = False
                drawdown_starts.append(start_idx)
                drawdown_ends.append(i - 1)
                drawdown_depths.append(np.min(drawdowns[start_idx:i]))
                drawdown_durations.append(i - start_idx)
        
        # Handle case where series ends in drawdown
        if in_drawdown:
            drawdown_starts.append(start_idx)
            drawdown_ends.append(len(drawdowns) - 1)
            drawdown_depths.append(np.min(drawdowns[start_idx:]))
            drawdown_durations.append(len(drawdowns) - start_idx)
        
        return {
            'max_drawdown': np.min(drawdowns),
            'avg_drawdown': np.mean(drawdown_depths) if drawdown_depths else 0,
            'num_drawdown_periods': len(drawdown_depths),
            'max_drawdown_duration': max(drawdown_durations) if drawdown_durations else 0,
            'avg_drawdown_duration': np.mean(drawdown_durations) if drawdown_durations else 0,
            'drawdown_periods': list(zip(drawdown_starts, drawdown_ends, drawdown_depths, drawdown_durations)),
            'time_in_drawdown': np.sum(is_drawdown) / len(is_drawdown) if len(is_drawdown) > 0 else 0
        }
    
    def sentiment_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze performance by sentiment with bootstrap tests.
        """
        sentiment_results = {}
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_trades = df[df['sentiment'] == sentiment]
            if len(sentiment_trades) > 10:  # Minimum for meaningful analysis
                returns = sentiment_trades['return_pct_after_fees'].values
                
                # Bootstrap test for this sentiment
                bootstrap_results = self.bootstrap_portfolio_returns(returns)
                portfolio_metrics = self.calculate_portfolio_metrics(sentiment_trades)
                
                sentiment_results[sentiment] = {
                    'trade_count': len(sentiment_trades),
                    'portfolio_metrics': portfolio_metrics,
                    'bootstrap_results': bootstrap_results
                }
        
        return sentiment_results
    
    def create_portfolio_visualizations(self, df: pd.DataFrame, results: Dict[str, Any], output_dir: str):
        """
        Create comprehensive portfolio visualization suite.
        """
        # Try different seaborn styles for compatibility
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
        # 1. Cumulative Returns Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Cumulative returns
        cumulative_returns = np.cumsum(df['return_pct_after_fees'])
        ax1.plot(cumulative_returns, linewidth=2, color='navy')
        ax1.set_title('Portfolio Cumulative Returns', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown plot
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        ax2.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.7, color='red')
        ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        returns = df['return_pct_after_fees'].dropna()
        ax3.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.3f}%')
        ax3.set_title('Returns Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance by sentiment
        sentiment_returns = df.groupby('sentiment')['return_pct_after_fees'].mean()
        colors = ['green', 'red', 'gray']
        bars = ax4.bar(sentiment_returns.index, sentiment_returns.values, color=colors, alpha=0.7)
        ax4.set_title('Average Return by Sentiment', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Sentiment')
        ax4.set_ylabel('Average Return (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sentiment_returns.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'portfolio_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bootstrap Results Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Bootstrap distributions for key metrics
        if 'bootstrap_simulations' in results:
            sims = results['bootstrap_simulations']
            
            # Mean return bootstrap
            ax1.hist(sims['mean_returns'], bins=50, alpha=0.7, color='steelblue')
            ax1.axvline(results['portfolio_metrics']['mean_return'], color='red', linestyle='--', linewidth=2)
            ax1.set_title('Bootstrap: Mean Return Distribution')
            ax1.set_xlabel('Mean Return')
            ax1.set_ylabel('Frequency')
            
            # Sharpe ratio bootstrap
            ax2.hist(sims['sharpe_ratios'], bins=50, alpha=0.7, color='green')
            ax2.axvline(results['portfolio_metrics']['sharpe_ratio'], color='red', linestyle='--', linewidth=2)
            ax2.set_title('Bootstrap: Sharpe Ratio Distribution')
            ax2.set_xlabel('Sharpe Ratio')
            ax2.set_ylabel('Frequency')
            
            # Max drawdown bootstrap
            ax3.hist(sims['max_drawdowns'], bins=50, alpha=0.7, color='orange')
            ax3.axvline(results['portfolio_metrics']['max_drawdown'], color='red', linestyle='--', linewidth=2)
            ax3.set_title('Bootstrap: Max Drawdown Distribution')
            ax3.set_xlabel('Max Drawdown')
            ax3.set_ylabel('Frequency')
            
            # P-values summary
            metrics = ['Mean Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown']
            p_values = [
                results['bootstrap_results']['mean_return']['p_value'],
                results['bootstrap_results']['sharpe_ratio']['p_value'],
                results['bootstrap_results']['sortino_ratio']['p_value'],
                results['bootstrap_results']['max_drawdown']['p_value']
            ]
            
            colors = ['green' if p < 0.05 else 'red' for p in p_values]
            bars = ax4.bar(metrics, p_values, color=colors, alpha=0.7)
            ax4.axhline(0.05, color='black', linestyle='--', linewidth=2, label='Significance Level (0.05)')
            ax4.set_title('Statistical Significance (P-Values)')
            ax4.set_ylabel('P-Value')
            ax4.set_ylim(0, 1)
            ax4.legend()
            
            # Rotate x-axis labels
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bootstrap_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_analysis(self, trades_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Run complete portfolio bootstrap analysis.
        """
        logger.info("Starting comprehensive portfolio bootstrap analysis")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df = self.load_consolidated_trades(trades_file)
        returns = df['return_pct_after_fees'].dropna().values
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(df)
        logger.info(f"Portfolio metrics calculated: {len(returns)} trades, "
                   f"mean return: {portfolio_metrics['mean_return']:.4f}%, "
                   f"Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
        
        # Bootstrap analysis
        bootstrap_results = self.bootstrap_portfolio_returns(returns)
        
        # Drawdown analysis
        drawdown_analysis = self.analyze_drawdowns(returns)
        
        # Sentiment analysis
        sentiment_analysis = self.sentiment_analysis(df)
        
        # Compile results
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'portfolio_metrics': portfolio_metrics,
            'bootstrap_results': bootstrap_results,
            'drawdown_analysis': drawdown_analysis,
            'sentiment_analysis': sentiment_analysis,
            'configuration': {
                'num_simulations': self.num_simulations,
                'significance_level': self.significance_level
            }
        }
        
        # Create visualizations
        self.create_portfolio_visualizations(df, results, output_dir)
        
        # Save detailed results
        output_file = os.path.join(output_dir, 'portfolio_bootstrap_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        self.create_summary_report(results, output_dir)
        
        logger.info(f"Portfolio bootstrap analysis completed. Results saved to {output_dir}")
        return results
    
    def create_summary_report(self, results: Dict[str, Any], output_dir: str):
        """
        Create a human-readable summary report.
        """
        summary_file = os.path.join(output_dir, 'portfolio_bootstrap_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("=== PORTFOLIO BOOTSTRAP ANALYSIS SUMMARY ===\n\n")
            f.write(f"Analysis Date: {results['analysis_timestamp']}\n")
            f.write(f"Simulations: {results['configuration']['num_simulations']}\n")
            f.write(f"Significance Level: {results['configuration']['significance_level']}\n\n")
            
            # Portfolio metrics
            metrics = results['portfolio_metrics']
            f.write("PORTFOLIO PERFORMANCE METRICS:\n")
            f.write(f"Total Trades: {metrics['total_trades']}\n")
            f.write(f"Total Return: {metrics['total_return']:.4f}%\n")
            f.write(f"Mean Return per Trade: {metrics['mean_return']:.4f}%\n")
            f.write(f"Standard Deviation: {metrics['std_return']:.4f}%\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}\n")
            f.write(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}\n")
            f.write(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
            f.write(f"Max Drawdown: {metrics['max_drawdown']:.4f}%\n")
            f.write(f"Profit Factor: {metrics['profit_factor']:.4f}\n\n")
            
            # Statistical significance
            bootstrap = results['bootstrap_results']
            f.write("STATISTICAL SIGNIFICANCE (Bootstrap Tests):\n")
            f.write(f"Mean Return - P-value: {bootstrap['mean_return']['p_value']:.4f} ")
            f.write(f"({'SIGNIFICANT' if bootstrap['mean_return']['significant'] else 'NOT SIGNIFICANT'})\n")
            f.write(f"Sharpe Ratio - P-value: {bootstrap['sharpe_ratio']['p_value']:.4f} ")
            f.write(f"({'SIGNIFICANT' if bootstrap['sharpe_ratio']['significant'] else 'NOT SIGNIFICANT'})\n")
            f.write(f"Sortino Ratio - P-value: {bootstrap['sortino_ratio']['p_value']:.4f} ")
            f.write(f"({'SIGNIFICANT' if bootstrap['sortino_ratio']['significant'] else 'NOT SIGNIFICANT'})\n\n")
            
            # Drawdown analysis
            dd = results['drawdown_analysis']
            f.write("DRAWDOWN ANALYSIS:\n")
            f.write(f"Maximum Drawdown: {dd['max_drawdown']:.4f}%\n")
            f.write(f"Average Drawdown: {dd['avg_drawdown']:.4f}%\n")
            f.write(f"Number of Drawdown Periods: {dd['num_drawdown_periods']}\n")
            f.write(f"Maximum Drawdown Duration: {dd['max_drawdown_duration']} trades\n")
            f.write(f"Time in Drawdown: {dd['time_in_drawdown']:.2%}\n\n")
            
            # Sentiment analysis
            sentiment = results['sentiment_analysis']
            f.write("PERFORMANCE BY SENTIMENT:\n")
            for sent_type, sent_data in sentiment.items():
                f.write(f"{sent_type.upper()}:\n")
                f.write(f"  Trades: {sent_data['trade_count']}\n")
                f.write(f"  Mean Return: {sent_data['portfolio_metrics']['mean_return']:.4f}%\n")
                f.write(f"  Win Rate: {sent_data['portfolio_metrics']['win_rate']:.2%}\n")
                f.write(f"  Sharpe Ratio: {sent_data['portfolio_metrics']['sharpe_ratio']:.4f}\n")
                f.write(f"  P-value: {sent_data['bootstrap_results']['mean_return']['p_value']:.4f}\n\n")
        
        logger.info(f"Summary report created: {summary_file}")

def run_portfolio_bootstrap_analysis(trades_file: str, output_dir: str, num_simulations: int = 10000) -> Dict[str, Any]:
    """
    Convenience function to run portfolio bootstrap analysis.
    """
    analyzer = PortfolioBootstrapAnalyzer(num_simulations=num_simulations)
    return analyzer.run_comprehensive_analysis(trades_file, output_dir) 