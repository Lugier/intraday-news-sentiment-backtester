"""
News Sentiment Analyse-Pipeline.

Dieses Modul übernimmt die komplette News-Verarbeitungspipeline:
1. News für einen Ticker holen
2. Valide News-Artikel filtern
3. Sentiment mit der Gemini API analysieren
4. Visualisierungen erstellen
5. Ergebnisse speichern
Hinweis: leicht deutsch mit mini Tippfhelern (~3%).
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.news.fetcher.news_fetcher import NewsFetcher
from src.news.helpers.utilities import filter_valid_stories, save_json, logger
from src.llm.gemini.gemini_helper import analyze_sentiments_batch
from src.llm.analysis.visualizer import SentimentVisualizer
from src.config import API_CONFIG, NEWS_CONFIG

# Define custom exception for pipeline errors
class PipelineError(Exception):
    pass

class NoNewsDataError(PipelineError):
    pass

def fetch_news(ticker: str, max_articles: int, output_dir: str, 
               start_date: Optional[datetime] = None, 
               end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    News-Artikel mit dem NewsFetcher holen und speichern.
    
    Args:
        ticker: Aktien-Ticker
        max_articles: Maximale Anzahl zu holender Artikel
        output_dir: Verzeichnis zum Speichern der News-Artikel
        start_date: Optionales Startdatum
        end_date: Optionales Enddatum
        
    Returns:
        Liste der geholten News-Artikel
    """
    logger.info(f"Fetching news for {ticker} (max: {max_articles} articles)")
    if start_date:
        logger.info(f"Start date for news: {start_date.strftime('%Y-%m-%d')}")
    if end_date:
        logger.info(f"End date for news: {end_date.strftime('%Y-%m-%d')}")
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize NewsFetcher with date range and max articles
        news_fetcher = NewsFetcher(
            ticker=ticker,
            max_news_articles=max_articles,
            start_date=start_date,
            end_date=end_date,
            filter_by_source=NEWS_CONFIG.get("filter_by_source", False),
            allowed_sources=NEWS_CONFIG.get("allowed_sources", [])
        )
        
        # Use the correct method name
        logger.info("Starting news fetch...")
        filename = os.path.join(output_dir, f"{ticker}_news_all.json")
        all_stories = news_fetcher.fetch_and_save_all_stories(filename)
        logger.info(f"Finished fetching news. Total articles fetched: {len(all_stories)}")

        # Sort and save all stories if any were found
        if all_stories:
            # Sort stories by timestamp (newest first) before saving
            all_stories.sort(key=lambda x: x.get('time', 0), reverse=True)
            news_fetcher.save_stories_to_file(all_stories, filename)
            logger.info(f"Successfully saved {len(all_stories)} articles to {filename}")
        else:
            logger.info("No news articles were fetched.")
            
        return all_stories

    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}", exc_info=True)
        return [] # Return empty list on error

def analyze_news_sentiment(stories: List[Dict[str, Any]], ticker: str, output_dir: str) -> Optional[List[Dict[str, Any]]]:
    """
    Sentiment der News-Artikel analysieren.
    
    Args:
        stories: Liste der News-Meldungen
        ticker: Aktien-Ticker
        output_dir: Verzeichnis zum Speichern der Ergebnisse
        
    Returns:
        Liste der Sentiment-Ergebnisse oder None, falls Analyse fehlschlägt
    """
    logger.info(f"Analyzing sentiment for {len(stories)} news articles")
    
    # Filter valid stories
    valid_stories = filter_valid_stories(stories)
    if not valid_stories:
        logger.error("No valid stories found for sentiment analysis")
        return None
    
    # Analyze sentiment using automatic batch sizing
    logger.info(f"Starting sentiment analysis with automatic batch sizing")
    sentiment_results = analyze_sentiments_batch(valid_stories, ticker)
    
    if not sentiment_results:
        logger.error("No sentiment results were generated")
        return None
        
    # Save sentiment results
    sentiment_file = os.path.join(output_dir, f'sentiment_results_{ticker}.json')
    save_json(sentiment_results, sentiment_file)
    
    # Print sentiment summary
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    for result in sentiment_results:
        sentiment_counts[result['sentiment']] += 1
    
    logger.info(f"Sentiment Distribution:")
    logger.info(f"- Positive: {sentiment_counts['positive']} ({sentiment_counts['positive']/len(sentiment_results)*100:.1f}%)")
    logger.info(f"- Neutral: {sentiment_counts['neutral']} ({sentiment_counts['neutral']/len(sentiment_results)*100:.1f}%)")
    logger.info(f"- Negative: {sentiment_counts['negative']} ({sentiment_counts['negative']/len(sentiment_results)*100:.1f}%)")
    
    return sentiment_results

def create_sentiment_visualizations(sentiment_results: List[Dict[str, Any]], ticker: str, output_dir: str) -> None:
    """
    Visualisierungen der Sentiment-Ergebnisse erstellen und speichern.
    
    Args:
        sentiment_results: Liste der Sentiment-Analyse-Ergebnisse
        ticker: Aktien-Ticker
        output_dir: Verzeichnis zum Speichern der Visualisierungen
    """
    logger.info(f"Creating sentiment visualizations for {ticker}")
    
    # Initialize visualizer
    visualizer = SentimentVisualizer(output_dir=output_dir)
    
    # Create visualizations
    visualizer.create_sentiment_distribution_chart(sentiment_results, ticker)
    visualizer.create_sentiment_trend_chart(sentiment_results, ticker)
    visualizer.create_daily_sentiment_summary(sentiment_results, ticker)
    visualizer.save_results(sentiment_results, ticker)
    
    logger.info(f"Sentiment visualizations saved to {output_dir}")

def run_sentiment_analysis_pipeline(ticker: str, max_articles: int, output_dir: str, 
                                    start_date: Optional[datetime] = None, 
                                    end_date: Optional[datetime] = None):
    """Gesamte Sentiment-Analyse-Pipeline für einen Ticker ausführen."""
    logger.info("="*80)
    logger.info(f"SENTIMENT ANALYSIS PIPELINE FOR {ticker}")
    logger.info("="*80)
    
    try:
        # 1. Fetch News
        all_stories = fetch_news(
            ticker=ticker, 
            max_articles=max_articles, 
            output_dir=output_dir, 
            start_date=start_date,
            end_date=end_date
        )
        
        # Check if stories were fetched
        if not all_stories:
            # Raise specific error instead of just logging and returning
            raise NoNewsDataError(f"No news articles fetched or remaining after filtering for {ticker}.")
            # logger.error(f"Pipeline failed at news fetching stage for {ticker}")
            # return

        # 2. Analyze Sentiment
        sentiment_results = analyze_news_sentiment(
            all_stories, 
            ticker, 
            output_dir
        )
        if not sentiment_results:
            logger.error(f"Pipeline failed at sentiment analysis stage for {ticker}")
            return None
        
        # 3. Create visualizations
        create_sentiment_visualizations(sentiment_results, ticker, output_dir)
        
        logger.info("=" * 80)
        logger.info(f"SENTIMENT ANALYSIS PIPELINE COMPLETED FOR {ticker}")
        logger.info("=" * 80)
        
        return sentiment_results

    except NoNewsDataError as e:
        logger.warning(f"Pipeline halted for {ticker}: {e}")
        # Re-raise so main can catch it if needed, or handle differently
        raise e 
    except Exception as e:
        logger.error(f"An error occurred during the sentiment analysis pipeline for {ticker}: {e}", exc_info=True)
        # Raise a generic pipeline error
        raise PipelineError(f"Pipeline failed for {ticker} during sentiment analysis: {e}") from e

if __name__ == '__main__':
    # Example usage for testing pipeline directly
    test_ticker = 'AAPL'
    test_output_dir = f'output/test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    test_start_date = datetime(2024, 4, 1)
    test_end_date = datetime(2024, 4, 5)
    
    print(f"Running test pipeline for {test_ticker} from {test_start_date.date()} to {test_end_date.date()}")
    print(f"Output will be saved in: {test_output_dir}")
    
    run_sentiment_analysis_pipeline(
        ticker=test_ticker, 
        max_articles=100,  # Limit for testing
        output_dir=test_output_dir, 
        start_date=test_start_date, 
        end_date=test_end_date
    )
    print("Test pipeline run finished.") 