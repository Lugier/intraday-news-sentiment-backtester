"""
Modul für Trading-Strategien basierend auf Nachrichtensentiment.

Dieses Modul enthält Klassen und Funktionen zur Definition und Simulation von
Handelsstrategien, die auf Sentiment-Signalen basieren.
"""

from src.backtesting.strategies.strategies import (
    SentimentTradingStrategy, 
    simulate_simple_strategy, 
    calculate_backtest_metrics
)

__all__ = [
    'SentimentTradingStrategy', 
    'simulate_simple_strategy', 
    'calculate_backtest_metrics'
]
