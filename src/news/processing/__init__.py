"""
Modul für die Verarbeitung von Finanznachrichten.

Dieses Modul enthält Pipeline-Funktionen, die den gesamten Prozess vom Abrufen
der Nachrichten bis zur Sentimentanalyse koordinieren.
"""

from src.news.processing.pipeline import fetch_news, run_sentiment_analysis_pipeline

__all__ = ['fetch_news', 'run_sentiment_analysis_pipeline']
