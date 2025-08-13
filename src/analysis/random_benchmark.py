"""
Random Trading Benchmark Analysis
Compares sentiment-based strategy against random trading to validate strategy effectiveness
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import random
from scipy import stats
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RandomTradingBenchmark:
    """
    Implements random trading benchmark to validate sentiment strategy performance
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        self.random_results = []
        
    def generate_random_trades(self, 
                             sentiment_trades: List[Dict[str, Any]], 
                             price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate random trades with same frequency and timing as sentiment trades
        but with random directions
        """
        random_trades = []
        
        for original_trade in sentiment_trades:
            # Keep same entry time and exit time, but randomize direction
            random_direction = random.choice(['long', 'short'])
            
            # Calculate random trade return based on actual price movements
            entry_time = pd.to_datetime(original_trade['entry_time'])
            exit_time = pd.to_datetime(original_trade['exit_time'])
            
            # Get actual price data for these times
            entry_price_data = price_data[price_data.index <= entry_time]
            exit_price_data = price_data[price_data.index <= exit_time]
            
            if entry_price_data.empty or exit_price_data.empty:
                continue
                
            entry_price = entry_price_data.iloc[-1]['close']
            exit_price = exit_price_data.iloc[-1]['close']
            
            # Calculate return based on random direction
            if random_direction == 'long':
                return_pct = (exit_price / entry_price - 1) * 100
            else:  # short
                return_pct = (entry_price / exit_price - 1) * 100
                
            random_trade = {
                'entry_time': original_trade['entry_time'],
                'exit_time': original_trade['exit_time'],
                'direction': random_direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': return_pct,
                'trade_duration_min': original_trade.get('trade_duration_min', 0),
                'ticker': original_trade.get('ticker', ''),
                'exit_reason': 'random_benchmark'
            }
            
            random_trades.append(random_trade)
            
        return random_trades
    
    def run_random_simulations(self, 
                              sentiment_trades: List[Dict[str, Any]], 
                              price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run multiple random trading simulations
        """
        simulation_results = []
        
        for sim in range(self.n_simulations):
            # Set seed for reproducibility while maintaining randomness
            random.seed(sim)
            np.random.seed(sim)
            
            random_trades = self.generate_random_trades(sentiment_trades, price_data)
            
            if not random_trades:
                continue
                
            # Calculate simulation metrics
            total_return = sum(trade['return_pct'] for trade in random_trades)
            returns = [trade['return_pct'] for trade in random_trades]
            
            win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
            avg_return = np.mean(returns) if returns else 0
            std_return = np.std(returns) if returns else 0
            sharpe_ratio = avg_return / std_return if std_return != 0 else 0
            
            max_drawdown = self._calculate_max_drawdown(returns)
            
            simulation_results.append({
                'simulation': sim,
                'total_return_pct': total_return,
                'avg_return_pct': avg_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'total_trades': len(random_trades),
                'std_return': std_return
            })
        
        return self._analyze_simulation_results(simulation_results)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown) if len(drawdown) > 0 else 0
    
    def _analyze_simulation_results(self, simulation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results from all random simulations"""
        if not simulation_results:
            return {}
            
        df = pd.DataFrame(simulation_results)
        
        return {
            'n_simulations': len(simulation_results),
            'random_benchmark_stats': {
                'mean_total_return': float(df['total_return_pct'].mean()),
                'std_total_return': float(df['total_return_pct'].std()),
                'median_total_return': float(df['total_return_pct'].median()),
                'percentile_25': float(df['total_return_pct'].quantile(0.25)),
                'percentile_75': float(df['total_return_pct'].quantile(0.75)),
                'min_return': float(df['total_return_pct'].min()),
                'max_return': float(df['total_return_pct'].max()),
                'mean_win_rate': float(df['win_rate'].mean()),
                'mean_sharpe_ratio': float(df['sharpe_ratio'].mean()),
                'mean_max_drawdown': float(df['max_drawdown_pct'].mean())
            },
            'simulation_details': simulation_results
        }
    
    def compare_strategy_vs_random(self, 
                                   strategy_results: Dict[str, Any], 
                                   random_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Statistical comparison between sentiment strategy and random trading
        """
        if not random_results or 'simulation_details' not in random_results:
            return {}
            
        strategy_return = strategy_results.get('total_return_pct', 0)
        random_returns = [sim['total_return_pct'] for sim in random_results['simulation_details']]
        
        # Calculate percentile rank of strategy performance
        percentile_rank = stats.percentileofscore(random_returns, strategy_return)
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_1samp(random_returns, strategy_return)
        
        # Count how many random simulations outperformed strategy
        outperforming_simulations = len([r for r in random_returns if r > strategy_return])
        outperforming_percentage = (outperforming_simulations / len(random_returns)) * 100
        
        return {
            'strategy_vs_random_comparison': {
                'strategy_return_pct': float(strategy_return),
                'random_mean_return_pct': float(np.mean(random_returns)),
                'random_std_return_pct': float(np.std(random_returns)),
                'strategy_percentile_rank': float(percentile_rank),
                'outperforming_simulations': int(outperforming_simulations),
                'outperforming_percentage': float(outperforming_percentage),
                'statistical_significance': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_at_5pct': bool(p_value < 0.05),
                    'significant_at_1pct': bool(p_value < 0.01)
                },
                'interpretation': self._interpret_results(percentile_rank, p_value)
            }
        }
    
    def _interpret_results(self, percentile_rank: float, p_value: float) -> str:
        """Provide interpretation of random benchmark results"""
        if percentile_rank >= 95:
            performance = "exceptional (top 5%)"
        elif percentile_rank >= 90:
            performance = "very strong (top 10%)"
        elif percentile_rank >= 75:
            performance = "good (top 25%)"
        elif percentile_rank >= 50:
            performance = "above average"
        else:
            performance = "below average"
            
        # Clarify what statistical significance means in this context
        if p_value < 0.05:
            significance_text = f"the difference between strategy returns and random trading mean is statistically significant (p={p_value:.4f})"
        else:
            significance_text = f"the difference between strategy returns and random trading mean is not statistically significant (p={p_value:.4f})"
        
        return f"Strategy performance is {performance} compared to random trading. Additionally, {significance_text}."

def run_random_benchmark_analysis(ticker: str, 
                                sentiment_trades: List[Dict[str, Any]], 
                                strategy_results: Dict[str, Any],
                                price_data: pd.DataFrame,
                                output_dir: str) -> Dict[str, Any]:
    """
    Run complete random trading benchmark analysis
    """
    logger.info(f"Running random trading benchmark for {ticker}...")
    
    benchmark = RandomTradingBenchmark(n_simulations=1000)
    
    # Run random simulations
    random_results = benchmark.run_random_simulations(sentiment_trades, price_data)
    
    # Compare strategy vs random
    comparison_results = benchmark.compare_strategy_vs_random(strategy_results, random_results)
    
    # Combine results
    complete_results = {
        **random_results,
        **comparison_results
    }
    
    # Save results
    import json
    import os
    
    output_file = os.path.join(output_dir, f"{ticker}_random_benchmark.json")
    with open(output_file, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    # Create summary CSV
    if 'strategy_vs_random_comparison' in complete_results:
        comparison = complete_results['strategy_vs_random_comparison']
        
        csv_data = [
            ["Metric", "Value"],
            ["Strategy Return (%)", f"{comparison['strategy_return_pct']:.2f}"],
            ["Random Mean Return (%)", f"{comparison['random_mean_return_pct']:.2f}"],
            ["Strategy Percentile Rank", f"{comparison['strategy_percentile_rank']:.1f}%"],
            ["Outperformed by Random (%)", f"{comparison['outperforming_percentage']:.1f}"],
            ["Statistical Significance (p-value)", f"{comparison['statistical_significance']['p_value']:.4f}"],
            ["Significant at 5%", comparison['statistical_significance']['significant_at_5pct']],
            ["Interpretation", comparison['interpretation']]
        ]
        
        import csv
        csv_file = os.path.join(output_dir, f"{ticker}_random_benchmark_summary.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
    
    logger.info(f"Random benchmark analysis completed for {ticker}")
    return complete_results 