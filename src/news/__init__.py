"""
Modul für Nachrichtenbeschaffung und -verarbeitung.

Dieses Paket enthält alle Funktionen zum Abrufen und Verarbeiten von Finanznachrichten.
"""

from src.news.processing.pipeline import fetch_news, run_sentiment_analysis_pipeline
from src.news.fetcher.news_fetcher import NewsFetcher
from src.news.data.tickers import get_dow_tickers
from src.news.helpers.utilities import (
    create_timestamped_output_dir,
    save_json,
    filter_valid_stories,
    logger
)

__all__ = [
    'NewsFetcher', 
    'fetch_news', 
    'run_sentiment_analysis_pipeline',
    'get_dow_tickers',
    'create_timestamped_output_dir',
    'save_json',
    'filter_valid_stories',
    'logger'
] 