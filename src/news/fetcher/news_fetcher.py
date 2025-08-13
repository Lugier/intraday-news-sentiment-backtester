"""
News-Fetcher Modul

Dieses Modul bietet Funktionalität, um News-Daten aus verschiedenen APIs zu holen
und kümmert sich um Rate-Limiting, Caching und Datenverarbeitung. Leicht deutsch mit mini Tippfehlern (~3%).
"""

import os
import json
import time
import sys
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from src.config import API_CONFIG, NEWS_CONFIG

# For debugging and progress bar output management
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

logger = logging.getLogger(__name__)

# Set up logging for this specific logger
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

# Disable the monitor thread from tqdm to avoid 'daemonic processes' issues
tqdm.monitor_interval = 0

class NewsFilterTracker:
    """Verfolgt die News-Filter-Statistiken bei jedem Schritt."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Alle Tracking-Statistiken zurücksetzen."""
        self.stats = {
            'total_fetched_from_api': 0,
            'filtered_by_source': 0,
            'filtered_by_missing_data': 0,
            'filtered_by_neutral_sentiment': 0,
            'filtered_by_after_hours': 0,
            'filtered_by_weekend': 0,
            'filtered_by_missing_market_data': 0,
            'final_valid_news': 0,
            'final_trades_executed': 0
        }
        self.detailed_logs = []
    
    def log_api_fetch(self, count: int, page: int):
        """News-Fetches von der API protokollieren."""
        self.stats['total_fetched_from_api'] += count
        self.detailed_logs.append(f"Page {page}: Fetched {count} news from API")
    
    def log_source_filter(self, original_count: int, filtered_count: int):
        """Quellen-Filterung protokollieren."""
        removed = original_count - filtered_count
        self.stats['filtered_by_source'] += removed
        self.detailed_logs.append(f"Source filter: Removed {removed} news (kept {filtered_count})")
    
    def log_missing_data_filter(self, count: int):
        """Filterung wegen fehlender Daten protokollieren."""
        self.stats['filtered_by_missing_data'] += count
        self.detailed_logs.append(f"Missing data filter: Removed {count} news")
    
    def log_neutral_sentiment_filter(self, count: int):
        """Filterung neutraler Sentiments protokollieren."""
        self.stats['filtered_by_neutral_sentiment'] += count
        self.detailed_logs.append(f"Neutral sentiment filter: Removed {count} news")
    
    def log_after_hours_filter(self, count: int):
        """Filterung außerhalb der Handelszeiten protokollieren."""
        self.stats['filtered_by_after_hours'] += count
        self.detailed_logs.append(f"After hours filter: Removed {count} news")
    
    def log_weekend_filter(self, count: int):
        """Wochenend-Filterung protokollieren."""
        self.stats['filtered_by_weekend'] += count
        self.detailed_logs.append(f"Weekend filter: Removed {count} news")
    
    def log_missing_market_data_filter(self, count: int):
        """Filterung wegen fehlender Marktdaten protokollieren."""
        self.stats['filtered_by_missing_market_data'] += count
        self.detailed_logs.append(f"Missing market data filter: Removed {count} news")
    
    def log_final_valid_news(self, count: int):
        """Endgültige Anzahl valider News protokollieren."""
        self.stats['final_valid_news'] = count
        self.detailed_logs.append(f"Final valid news: {count}")
    
    def log_trades_executed(self, count: int):
        """Ausgeführte Trades protokollieren."""
        self.stats['final_trades_executed'] = count
        self.detailed_logs.append(f"Trades executed: {count}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Umfassende Filter-Zusammenfassung abrufen."""
        total_removed = (
            self.stats['filtered_by_source'] +
            self.stats['filtered_by_missing_data'] +
            self.stats['filtered_by_neutral_sentiment'] +
            self.stats['filtered_by_after_hours'] +
            self.stats['filtered_by_weekend'] +
            self.stats['filtered_by_missing_market_data']
        )
        
        return {
            'filtering_statistics': self.stats,
            'total_news_removed': total_removed,
            'conversion_rate_news_to_trades': (
                self.stats['final_trades_executed'] / max(self.stats['final_valid_news'], 1) * 100
            ),
            'overall_conversion_rate': (
                self.stats['final_trades_executed'] / max(self.stats['total_fetched_from_api'], 1) * 100
            ),
            'detailed_logs': self.detailed_logs
        }
    
    def save_to_file(self, filepath: str):
        """Filter-Statistiken in JSON-Datei speichern."""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"News filtering statistics saved to {filepath}")

# Global filter tracker instance
filter_tracker = NewsFilterTracker()

class NewsFetcher:
    def __init__(self, ticker: str, 
                 max_news_articles: int = NEWS_CONFIG["default_max_articles"], 
                 api_url: str = API_CONFIG["news_api"]["url"],
                 categories: List[str] = API_CONFIG["news_api"]["categories"], 
                 news_per_page: int = NEWS_CONFIG["articles_per_request"],
                 delay_between_requests: int = API_CONFIG["news_api"]["delay_between_requests"],
                 filter_by_source: bool = NEWS_CONFIG["filter_by_source"],
                 allowed_sources: List[str] = NEWS_CONFIG["allowed_sources"],
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None):
        """Initialize the NewsFetcher."""
        self.ticker = ticker
        self.api_url = api_url
        self.categories = categories
        self.news_per_page = news_per_page
        self.delay_between_requests = delay_between_requests
        self.filter_by_source = filter_by_source
        self.allowed_sources = [s.lower() for s in allowed_sources] if allowed_sources else []
        self.max_news_articles = max_news_articles
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now()
        
        # API request settings
        self.request_timeout = API_CONFIG["news_api"]["request_timeout"]
        self.retry_count = API_CONFIG["news_api"]["retry_count"]
        self.retry_delay = API_CONFIG["news_api"]["retry_delay"]
        
        self.headers = {}
        
        logger.info(f"NewsFetcher initialized for {ticker}.")
        if self.filter_by_source:
            logger.info(f"Source filtering is ENABLED with {len(self.allowed_sources)} allowed sources:")
            if len(self.allowed_sources) <= 10:
                logger.info(f"Allowed sources: {', '.join(self.allowed_sources)}")
            else:
                logger.info(f"Allowed sources (first 10): {', '.join(self.allowed_sources[:10])}...")
        else:
            logger.info("Source filtering is DISABLED - all sources will be included")
            
        if self.start_date:
            logger.info(f"Fetching news from {self.start_date.strftime('%Y-%m-%d')} up to {self.end_date.strftime('%Y-%m-%d')}.")
        else:
            logger.info(f"Fetching latest news (no start date specified).")

    def filter_stories_by_source(self, stories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter stories based on allowed sources."""
        if not self.filter_by_source or not self.allowed_sources:
            return stories
        
        filtered_stories = []
        original_count = len(stories)
        
        logger.debug(f"Filtering stories by source. Using configured allowed sources: {self.allowed_sources}")
        
        # Track unique sources for debugging
        unique_sources = set()
        for story in stories:
            # Use 'site' field which is what the API actually returns
            source = story.get('site', '').lower()
            unique_sources.add(source)
        
        logger.debug(f"Unique sources in stories: {unique_sources}")
        
        for story in stories:
            # Use 'site' field which is what the API actually returns
            source = story.get('site', '').lower()
            
            # Accept stories with empty sources (sometimes legitimate)
            if not source:
                logger.debug("Accepted story with empty source")
                filtered_stories.append(story)
                continue
            
            # Check if any allowed source is contained in the story's source
            is_allowed = any(allowed in source for allowed in self.allowed_sources)
            
            if is_allowed:
                filtered_stories.append(story)
                logger.debug(f"Accepted story from source: {source}")
            else:
                logger.debug(f"Rejected story from source: {source}")
                
        filtered_count = original_count - len(filtered_stories)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} stories from non-approved sources.")
            
        return filtered_stories

    def fetch_stories(self, category: str = None) -> List[Dict[str, Any]]:
        """
        Fetch stories for a specific category with pagination.
        
        Args:
            category: News category to fetch (optional)
            
        Returns:
            List of news stories
        """
        all_stories = []
        last_story_id = None
        page_count = 0
        retry_attempts = 0
        
        # Simple query - only ticker and category if provided
        if category:
            query = f"(and z:{self.ticker} T:{category})"
        else:
            query = f"z:{self.ticker}"
        
        logger.info(f"Starting fetch for ticker {self.ticker} with query: {query}")
        logger.info(f"Max articles to fetch: {self.max_news_articles}")
        
        # Handle case where max_news_articles is None (unlimited)
        max_articles = self.max_news_articles if self.max_news_articles is not None else float('inf')
        
        while len(all_stories) < max_articles:
            params = {
                "q": query,
                "n": self.news_per_page,
                "last": last_story_id
            }
            
            try:
                page_count += 1
                logger.info(f"Sending request {page_count} to API... Query: {query}")
                response = requests.get(self.api_url, params=params, timeout=self.request_timeout)
                
                # Handle rate limiting (429 status)
                if response.status_code == 429:
                    retry_attempts += 1
                    wait_time = self.retry_delay
                    logger.warning(f"Rate limit exceeded (429). Attempt {retry_attempts}/{self.retry_count}. Waiting {wait_time} seconds before retrying...")
                    
                    if retry_attempts >= self.retry_count:
                        logger.warning(f"Maximum retry attempts reached. Moving on...")
                        break
                    
                    time.sleep(wait_time)
                    page_count -= 1  # Don't count failed attempts
                    continue
                
                response.raise_for_status()
                retry_attempts = 0  # Reset on successful request
                
                data = response.json()
                logger.info(f"Response received from API (Page {page_count}), Query: '{query}'.")
                
                stories = data.get("stories", [])
                if not stories:
                    logger.info(f"No more news found (Page {page_count}), Query: '{query}'.")
                    break
                
                # Log API fetch
                filter_tracker.log_api_fetch(len(stories), page_count)
                
                # Filter stories by source if enabled
                if self.filter_by_source:
                    original_count = len(stories)
                    stories = self.filter_stories_by_source(stories)
                    filter_tracker.log_source_filter(original_count, len(stories))
                    filtered_count = original_count - len(stories)
                    if filtered_count > 0:
                        logger.info(f"Filtered out {filtered_count} stories from non-approved sources.")
                
                # Check if we have any stories left after filtering
                if not stories:
                    logger.info(f"All stories on page {page_count} were filtered out. Trying next page...")
                    # Get last_story_id from the original unfiltered data to continue pagination
                    original_stories = data.get("stories", [])
                    if original_stories:
                        last_story_id = original_stories[-1].get('id')
                        if last_story_id:
                            time.sleep(self.delay_between_requests)
                            continue
                        else:
                            logger.warning("No story ID found for pagination. Stopping.")
                            break
                    else:
                        break
                
                # Add stories to our collection
                stories_to_add = min(len(stories), max_articles - len(all_stories)) if self.max_news_articles is not None else len(stories)
                all_stories.extend(stories[:stories_to_add])
                
                logger.info(f"Page {page_count}: {stories_to_add} new articles retrieved. Total so far: {len(all_stories)}.")
                
                # Check if we've reached our limit
                if self.max_news_articles is not None and len(all_stories) >= self.max_news_articles:
                    logger.info(f"Maximum news limit of {self.max_news_articles} reached.")
                    break
                
                # Prepare for next page
                last_story_id = stories[-1].get('id')
                if not last_story_id:
                    logger.warning("No story ID found for pagination. Stopping.")
                    break
                
                # Delay between requests
                time.sleep(self.delay_between_requests)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error (Page {page_count}), Query: '{query}': {e}")
                
                retry_attempts += 1
                if retry_attempts < self.retry_count:
                    wait_time = self.delay_between_requests * 2
                    logger.info(f"Retry attempt {retry_attempts}/{self.retry_count}. Waiting {wait_time} seconds...")
                    page_count -= 1  # Don't count failed attempts
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Max retries reached. Stopping fetch.")
                    break
            
            except Exception as e:
                logger.error(f"Unexpected error (Page {page_count}), Query: '{query}': {e}")
                break
        
        logger.info(f"Fetch completed. Total articles collected: {len(all_stories)}")
        return all_stories

    def save_stories_to_file(self, stories: List[Dict[str, Any]], filename: str) -> None:
        """Save fetched news stories to a JSON file."""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(stories, f, ensure_ascii=False, indent=4)
            logger.info(f"News successfully saved to '{filename}'.")
        except Exception as e:
            logger.error(f"Error saving news: {e}")

    def fetch_and_save_all_stories(self, filename: str = 'all_fetched_stories.json') -> List[Dict[str, Any]]:
        """
        Fetch stories for all categories and save them to a file.
        
        Args:
            filename: Name of the file to save the stories to
            
        Returns:
            List of all fetched stories
        """
        all_stories_overall = []
        
        logger.info(f"Fetching news for ticker: {self.ticker}")
        logger.info(f"Max articles to fetch: {self.max_news_articles}")
        
        # Log date range if provided
        if self.start_date:
            logger.info(f"Start date: {self.start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.end_date:
            logger.info(f"End date: {self.end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # If we have categories, process each one
        if self.categories:
            for category in self.categories:
                logger.info(f"Processing category: {category}")
                stories = self.fetch_stories(category)
                
                if stories:
                    logger.info(f"In category '{category}' a total of {len(stories)} news articles were found.")
                    category_filename = filename.replace(".json", f"_{category}.json")
                    self.save_stories_to_file(stories, category_filename)
                    all_stories_overall.extend(stories)
                else:
                    logger.info(f"No news found in category '{category}'.")
        else:
            # No categories specified, fetch all news for the ticker
            logger.info("No categories specified, fetching all news for ticker")
            stories = self.fetch_stories()
            all_stories_overall.extend(stories)
        
        # Save all stories to the main file
        if all_stories_overall:
            # Sort all stories by timestamp (newest first)
            all_stories_overall.sort(key=lambda x: x.get('time', 0), reverse=True)
            # Limit to max articles if we have more than needed
            if self.max_news_articles is not None and len(all_stories_overall) > self.max_news_articles:
                all_stories_overall = all_stories_overall[:self.max_news_articles]
            self.save_stories_to_file(all_stories_overall, filename)
        
        logger.info(f"Total news found across all categories: {len(all_stories_overall)}")
        return all_stories_overall 