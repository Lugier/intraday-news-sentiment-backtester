"""
Analysis module for financial news sentiment analysis.

This module contains tools for analyzing the relationship between news 
sentiment and stock price reactions, including event study methodology
and statistical significance testing.
"""

from .event_study import (
    estimate_market_model,
    calculate_abnormal_returns,
    run_event_study,
    calculate_aar_and_caar
)

from .bootstrap import run_bootstrap_on_backtest

from .portfolio_bootstrap import run_portfolio_bootstrap_analysis

from .ttest import run_ttest_on_event_study

# Random benchmark analysis
from .random_benchmark import (
    RandomTradingBenchmark,
    run_random_benchmark_analysis
)

__all__ = [
    # Event Study Functions
    'estimate_market_model',
    'calculate_abnormal_returns', 
    'run_event_study',
    'calculate_aar_and_caar',
    
    # Bootstrap Testing Functions
    'run_bootstrap_on_backtest',
    
    # Portfolio Bootstrap Functions
    'run_portfolio_bootstrap_analysis',
    
    # T-Test Functions
    'run_ttest_on_event_study',
    
    # Random Benchmark Functions
    'RandomTradingBenchmark',
    'run_random_benchmark_analysis'
] 