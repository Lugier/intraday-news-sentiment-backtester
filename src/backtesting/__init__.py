"""
Modul für das Backtesting von Trading-Strategien basierend auf Nachrichtensentiment.

Dieses Paket enthält alle Funktionen zur Simulation und Analyse von Handelsstrategien,
die auf Sentiment-Daten von Finanznachrichten basieren.
"""

from src.backtesting.execution.pipeline import run_backtest_pipeline
from src.backtesting.strategies.strategies import (
    SentimentTradingStrategy, 
    simulate_simple_strategy, 
    calculate_backtest_metrics
)
from src.backtesting.analysis.backtest_visualizer import (
    plot_backtest_results, 
    plot_returns_distribution, 
    visualize_backtest_results,
    BacktestVisualizer
)
from src.backtesting.helpers.utilities import (
    create_timestamped_output_dir,
    save_json,
    logger
)
from src.backtesting.helpers.market_data import (
    fetch_stock_prices,
    get_market_hours,
    is_market_open
)

__all__ = [
    'run_backtest_pipeline',
    'SentimentTradingStrategy', 
    'simulate_simple_strategy', 
    'calculate_backtest_metrics',
    'plot_backtest_results', 
    'plot_returns_distribution', 
    'visualize_backtest_results',
    'BacktestVisualizer',
    'create_timestamped_output_dir',
    'save_json',
    'logger',
    'fetch_stock_prices',
    'get_market_hours',
    'is_market_open'
] 