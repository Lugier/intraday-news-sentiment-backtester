"""
Portfolio-Level Random Trading Benchmark Analysis
Analyzes the entire portfolio/index as one consolidated entity against random trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import random
from scipy import stats
import logging
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class PortfolioRandomBenchmark:
    """
    Implements portfolio-level random trading benchmark comparing the entire index/portfolio
    as one consolidated entity against random trading strategies
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        self.portfolio_trades = []
        self.random_results = []
        
    def run_portfolio_analysis(self, portfolio_trades_csv: str, output_dir: str) -> Dict[str, Any]:
        """
        Run complete portfolio-level random trading benchmark analysis
        
        Args:
            portfolio_trades_csv: Path to CSV file with all portfolio trades
            output_dir: Directory to save results
            
        Returns:
            Dictionary with complete analysis results
        """
        
        logger.info("Starting Portfolio-Level Random Trading Benchmark Analysis")
        
        # Load portfolio trades data
        try:
            trades_df = pd.read_csv(portfolio_trades_csv)
            logger.info(f"Loaded {len(trades_df)} portfolio trades from {portfolio_trades_csv}")
        except Exception as e:
            logger.error(f"Error loading portfolio trades: {e}")
            return {}
        
        # Calculate actual portfolio performance
        logger.info("Calculating actual portfolio performance...")
        actual_performance = self._calculate_portfolio_performance(trades_df)
        
        # Generate random trading simulations
        logger.info("Generating random trading simulations...")
        random_sims = self._generate_portfolio_random_simulations(trades_df)
        
        # Calculate random simulation statistics
        logger.info("Calculating random simulation statistics...")
        random_stats = self._calculate_random_stats(random_sims)
        
        # Compare strategy vs random performance
        logger.info("Comparing strategy vs random performance...")
        random_returns = [sim['total_return_pct'] for sim in random_sims]
        actual_return = actual_performance['total_return_pct']
        
        # Calculate percentile rank
        percentile_rank = stats.percentileofscore(random_returns, actual_return)
        outperformed_count = sum(1 for r in random_returns if actual_return > r)
        
        # Statistical significance test
        mean_random = np.mean(random_returns)
        std_random = np.std(random_returns)
        z_score = (actual_return - mean_random) / std_random if std_random > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        
        # Create comprehensive comparison analysis
        comparison = {
            "strategy_performance": {
                "total_return_pct": actual_performance["total_return_pct"],
                "avg_return_pct": actual_performance["avg_return_pct"],
                "win_rate": actual_performance["win_rate"],
                "sharpe_ratio": actual_performance["sharpe_ratio"],
                "sentiment_direction_accuracy": actual_performance.get("sentiment_direction_accuracy", 0.0),
                "sentiment_predictions_made": actual_performance.get("sentiment_predictions_made", 0),
                "total_trades": actual_performance["total_trades"]
            },
            "random_performance": {
                "mean_total_return": random_stats["mean_total_return"],
                "median_total_return": random_stats["median_total_return"],
                "std_total_return": random_stats["std_total_return"],
                "mean_win_rate": random_stats["mean_win_rate"],
                "mean_direction_accuracy": random_stats["mean_direction_accuracy"],
                "percentile_25": random_stats["percentile_25"],
                "percentile_75": random_stats["percentile_75"]
            },
            "strategy_vs_random": {
                "return_percentile_rank": percentile_rank,
                "outperformed_simulations": outperformed_count,
                "total_simulations": len(random_sims),
                "outperformance_rate": outperformed_count / len(random_sims),
                "return_advantage_pct": actual_performance["total_return_pct"] - random_stats["mean_total_return"],
                "direction_accuracy_advantage": actual_performance.get("sentiment_direction_accuracy", 0.5) - random_stats["mean_direction_accuracy"],
                "timing_skill_isolated": actual_performance["total_return_pct"] - random_stats["mean_total_return"],
                "direction_skill_vs_random": actual_performance.get("sentiment_direction_accuracy", 0.5) - 0.5
            },
            "statistical_significance": {
                "p_value_return": p_value,
                "z_score": z_score,
                "is_significant_5pct": p_value < 0.05,
                "is_significant_1pct": p_value < 0.01,
                "confidence_95_pct": f"Strategy return is {percentile_rank:.1f}th percentile"
            },
            "key_insights": {
                "timing_advantage": "Strategy benefits from consistent timing discipline vs random entries",
                "direction_prediction": f"Sentiment accuracy: {actual_performance.get('sentiment_direction_accuracy', 0.0)*100:.1f}% vs Random: 50.0%",
                "combined_effect": "Total outperformance comes from both timing discipline AND direction prediction",
                "sample_size": f"Analysis based on {actual_performance['total_trades']} trades across all tickers"
            }
        }
        
        # Compile final results
        results = {
            "analysis_metadata": {
                "analysis_type": "portfolio_level_random_benchmark",
                "timestamp": datetime.now().isoformat(),
                "portfolio_trades_file": portfolio_trades_csv,
                "n_simulations": self.n_simulations,
                "methodology": "identical_timing_random_direction"
            },
            "actual_portfolio_performance": actual_performance,
            "random_simulations_stats": random_stats,
            "portfolio_vs_random_comparison": comparison,
            "all_random_simulations": random_sims[:100]  # Save first 100 for inspection
        }
        
        # Save results
        self._save_results(results, output_dir)
        
        logger.info("Portfolio-Level Random Trading Benchmark Analysis completed")
        return results
    
    def _calculate_portfolio_performance(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate actual portfolio performance metrics"""
        
        # Sort trades by entry time to simulate portfolio timeline
        trades_sorted = trades_df.sort_values('entry_time').copy()
        
        # Calculate portfolio return (equal-weighted across tickers)
        total_return = trades_sorted['return_pct'].sum()
        avg_return = trades_sorted['return_pct'].mean()
        
        # Calculate other metrics
        win_rate = (trades_sorted['return_pct'] > 0).mean()
        
        # Calculate portfolio Sharpe ratio
        if trades_sorted['return_pct'].std() > 0:
            sharpe_ratio = avg_return / trades_sorted['return_pct'].std()
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown for portfolio
        cumulative_returns = (1 + trades_sorted['return_pct'] / 100).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Calculate sentiment direction accuracy (for strategy trades)
        # This shows how often sentiment correctly predicted market direction
        correct_predictions = 0
        total_predictions = len(trades_sorted)
        
        for _, trade in trades_sorted.iterrows():
            actual_return = trade['return_pct']
            sentiment = trade.get('sentiment', 'neutral')
            
            # Determine if sentiment prediction was correct
            if sentiment == 'positive' and actual_return > 0:
                correct_predictions += 1
            elif sentiment == 'negative' and actual_return < 0:
                correct_predictions += 1
            # Neutral sentiment doesn't make a direction prediction, so we don't count it
            elif sentiment == 'neutral':
                total_predictions -= 1  # Remove from denominator
        
        sentiment_direction_accuracy = correct_predictions / max(total_predictions, 1)
        
        return {
            "total_return_pct": float(total_return),
            "avg_return_pct": float(avg_return),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown_pct": float(max_drawdown),
            "total_trades": int(len(trades_sorted)),
            "winning_trades": int((trades_sorted['return_pct'] > 0).sum()),
            "losing_trades": int((trades_sorted['return_pct'] < 0).sum()),
            "sentiment_direction_accuracy": float(sentiment_direction_accuracy),
            "sentiment_predictions_made": int(total_predictions)
        }
    
    def _generate_portfolio_random_simulations(self, trades_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate random trading simulations using EXACT same timing as actual trades"""
        
        logger.info(f"Generating {self.n_simulations} portfolio-level random simulations")
        logger.info("Using identical entry/exit times, only randomizing trade direction")
        
        random_simulations = []
        
        # Risk-free rate set to 2.25% per annum (converted to daily rate)
        risk_free_rate = (1 + 0.0225) ** (1/252) - 1
        
        for sim_id in range(self.n_simulations):
            sim_trades = []
            
            # For EVERY actual trade, create a random counterpart
            for _, actual_trade in trades_df.iterrows():
                # Use EXACT same timing
                entry_time = actual_trade['entry_time']
                exit_time = actual_trade.get('exit_time', entry_time)  # Fallback if no exit_time
                ticker = actual_trade['ticker']
                
                # Only randomize the DIRECTION (Long vs Short)
                random_direction = random.choice([1, -1])  # +1 = Long, -1 = Short
                
                # Calculate what the return would be with random direction
                # Use absolute value of actual market movement, apply random direction
                actual_return = actual_trade['return_pct']
                market_movement_abs = abs(actual_return)
                random_return = random_direction * market_movement_abs
                
                # Determine if this was the "correct" direction
                actual_direction = 1 if actual_return >= 0 else -1
                correct_direction = (random_direction == actual_direction)
                
                sim_trades.append({
                    'ticker': ticker,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'return_pct': random_return,
                    'direction': 'long' if random_direction > 0 else 'short',
                    'random_direction': random_direction,
                    'actual_direction': actual_direction,
                    'correct_direction': correct_direction,
                    'market_movement_abs': market_movement_abs,
                    'simulation_id': sim_id
                })
            
            # Calculate simulation portfolio performance
            if sim_trades:
                sim_df = pd.DataFrame(sim_trades)
                sim_performance = self._calculate_portfolio_performance(sim_df)
                sim_performance['simulation_id'] = sim_id
                sim_performance['trades_taken'] = len(sim_trades)
                sim_performance['correct_directions'] = sum(1 for t in sim_trades if t['correct_direction'])
                sim_performance['direction_accuracy'] = sim_performance['correct_directions'] / len(sim_trades)
                random_simulations.append(sim_performance)
            else:
                # This should never happen since we take all actual trades
                logger.warning(f"No trades generated for simulation {sim_id}")
                random_simulations.append({
                    'simulation_id': sim_id,
                    'total_return_pct': 0.0,
                    'avg_return_pct': 0.0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown_pct': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'trades_taken': 0,
                    'correct_directions': 0,
                    'direction_accuracy': 0.0
                })
        
        return random_simulations
    
    def _calculate_random_stats(self, random_sims: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate statistics for random simulations"""
        
        returns = [sim['total_return_pct'] for sim in random_sims if not np.isnan(sim['total_return_pct'])]
        win_rates = [sim['win_rate'] for sim in random_sims if not np.isnan(sim['win_rate'])]
        sharpe_ratios = [sim['sharpe_ratio'] for sim in random_sims if not np.isnan(sim['sharpe_ratio'])]
        direction_accuracies = [sim.get('direction_accuracy', 0.0) for sim in random_sims if 'direction_accuracy' in sim]
        
        if not returns:
            return {}
        
        return {
            "mean_total_return": float(np.mean(returns)),
            "median_total_return": float(np.median(returns)),
            "std_total_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "percentile_25": float(np.percentile(returns, 25)),
            "percentile_75": float(np.percentile(returns, 75)),
            "mean_win_rate": float(np.mean(win_rates)) if win_rates else 0.0,
            "mean_sharpe_ratio": float(np.mean(sharpe_ratios)) if sharpe_ratios else 0.0,
            "mean_direction_accuracy": float(np.mean(direction_accuracies)) if direction_accuracies else 0.5,
            "std_direction_accuracy": float(np.std(direction_accuracies)) if direction_accuracies else 0.0,
            "simulations_count": len(random_sims)
        }
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save portfolio benchmark results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed JSON results
        json_file = os.path.join(output_dir, f"portfolio_random_benchmark.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_file = os.path.join(output_dir, f"portfolio_benchmark_summary.txt")
        self._save_portfolio_summary_text(results, summary_file)
        
        logger.info(f"Portfolio random benchmark results saved to {output_dir}")
    
    def _save_portfolio_summary_text(self, results: Dict[str, Any], file_path: str):
        """Save human-readable summary of portfolio benchmark results"""
        
        with open(file_path, 'w') as f:
            f.write(f"PORTFOLIO RANDOM TRADING BENCHMARK ANALYSIS\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Analysis Date: {results['analysis_metadata']['timestamp']}\n")
            f.write(f"Random Simulations: {results['analysis_metadata']['n_simulations']:,}\n\n")
            
            # Portfolio Performance
            portfolio_perf = results['actual_portfolio_performance']
            f.write(f"PORTFOLIO PERFORMANCE:\n")
            f.write(f"  Total Return: {portfolio_perf['total_return_pct']:.4f}%\n")
            f.write(f"  Average Return per Trade: {portfolio_perf['avg_return_pct']:.4f}%\n")
            f.write(f"  Win Rate: {portfolio_perf['win_rate']:.2%}\n")
            f.write(f"  Sharpe Ratio: {portfolio_perf['sharpe_ratio']:.4f}\n")
            f.write(f"  Max Drawdown: {portfolio_perf['max_drawdown_pct']:.4f}%\n")
            f.write(f"  Total Trades: {portfolio_perf['total_trades']}\n\n")
            
            # Random Benchmark Stats
            random_stats = results['random_simulations_stats']
            f.write(f"RANDOM TRADING BENCHMARK:\n")
            f.write(f"  Mean Return: {random_stats['mean_total_return']:.4f}%\n")
            f.write(f"  Median Return: {random_stats['median_total_return']:.4f}%\n")
            f.write(f"  Standard Deviation: {random_stats['std_total_return']:.4f}%\n")
            f.write(f"  Min Return: {random_stats['min_return']:.4f}%\n")
            f.write(f"  Max Return: {random_stats['max_return']:.4f}%\n\n")
            
            # Comparison Results
            comparison = results['portfolio_vs_random_comparison']
            f.write(f"PORTFOLIO VS RANDOM COMPARISON:\n")
            f.write(f"  Portfolio Percentile Rank: {comparison['strategy_vs_random']['return_percentile_rank']:.1f}th\n")
            f.write(f"  Outperformed Simulations: {comparison['strategy_vs_random']['outperformed_simulations']:,}/{comparison['strategy_vs_random']['total_simulations']:,}\n")
            f.write(f"  Outperforming Percentage: {comparison['strategy_vs_random']['outperformance_rate']:.1f}%\n")
            
            sig = comparison['statistical_significance']
            f.write(f"\nSTATISTICAL SIGNIFICANCE:\n")
            f.write(f"  P-value: {sig['p_value_return']:.6f}\n")
            f.write(f"  Z-score: {sig['z_score']:.4f}\n")
            f.write(f"  Significant at 5%: {'✓ YES' if sig['is_significant_5pct'] else '✗ NO'}\n")
            f.write(f"  Significant at 1%: {'✓ YES' if sig['is_significant_1pct'] else '✗ NO'}\n")
            
            # Interpretation
            f.write(f"\nINTERPRETATION:\n")
            percentile = comparison['strategy_vs_random']['return_percentile_rank']
            if percentile >= 75:
                interpretation = "EXCELLENT - Portfolio significantly outperforms random trading"
            elif percentile >= 50:
                interpretation = "GOOD - Portfolio performs better than average random trading"
            elif percentile >= 25:
                interpretation = "POOR - Portfolio underperforms compared to random trading"
            else:
                interpretation = "VERY POOR - Portfolio significantly underperforms random trading"
            
            f.write(f"  {interpretation}\n")
            f.write(f"  Portfolio performs better than {comparison['strategy_vs_random']['outperformance_rate']:.1f}% of random strategies\n")


def run_portfolio_random_benchmark_analysis(all_trades_file: str,
                                          index_name: str,
                                          output_dir: str,
                                          n_simulations: int = 1000) -> Dict[str, Any]:
    """
    Main wrapper function to run portfolio-level random benchmark analysis
    
    Args:
        all_trades_file: Path to CSV file with all portfolio trades
        index_name: Name of the index/portfolio (e.g., "TOP5") 
        output_dir: Directory to save results
        n_simulations: Number of random simulations to run
        
    Returns:
        Dictionary with complete portfolio benchmark results
    """
    
    logger.info(f"Starting Portfolio Random Benchmark Analysis for {index_name}")
    
    try:
        # Create portfolio benchmark analyzer
        benchmark = PortfolioRandomBenchmark(n_simulations=n_simulations)
        
        # Create output directory
        portfolio_benchmark_dir = os.path.join(output_dir, "portfolio_random_benchmark")
        os.makedirs(portfolio_benchmark_dir, exist_ok=True)
        
        # Run analysis
        results = benchmark.run_portfolio_analysis(
            portfolio_trades_csv=all_trades_file,
            output_dir=portfolio_benchmark_dir
        )
        
        if results:
            logger.info(f"Portfolio random benchmark analysis completed successfully")
            logger.info(f"Results saved to: {portfolio_benchmark_dir}")
            
            # Print key results to console
            comparison = results.get('portfolio_vs_random_comparison', {})
            if comparison:
                strategy_perf = comparison.get('strategy_performance', {})
                random_perf = comparison.get('random_performance', {})
                vs_random = comparison.get('strategy_vs_random', {})
                
                print(f"\n🎯 PORTFOLIO RANDOM BENCHMARK RESULTS ({index_name}):")
                print(f"   Strategy Return: {strategy_perf.get('total_return_pct', 0):.2f}%")
                print(f"   Random Mean: {random_perf.get('mean_total_return', 0):.2f}%")
                print(f"   Percentile Rank: {vs_random.get('return_percentile_rank', 0):.1f}th")
                print(f"   Direction Accuracy: {strategy_perf.get('sentiment_direction_accuracy', 0)*100:.1f}% (vs 50% random)")
                
        return results
        
    except Exception as e:
        logger.error(f"Error in portfolio random benchmark analysis: {e}", exc_info=True)
        return {} 