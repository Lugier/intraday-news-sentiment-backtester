"""
Modul für die Analyse und Visualisierung von Backtesting-Ergebnissen.

Dieses Modul enthält Funktionen zur Erstellung von Diagrammen und Visualisierungen
von Backtesting-Ergebnissen und Performance-Metriken.
"""

from src.backtesting.analysis.backtest_visualizer import (
    plot_backtest_results, 
    plot_returns_distribution, 
    visualize_backtest_results
)

__all__ = [
    'plot_backtest_results', 
    'plot_returns_distribution', 
    'visualize_backtest_results'
]
