#!/usr/bin/env python
"""
Validation Sample Script

This script fetches a random sample of news articles for AAPL,
analyzes their sentiment using Gemini, and exports the results to an Excel file
for manual validation.

The Excel file will contain columns for:
- News title and description
- Publication date
- Source
- LLM sentiment classification
- LLM explanation
- Manual validation column (to be filled by the reviewer)
"""

import os
import sys
import random
import pandas as pd
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

# Ensure the script can be run from the repository root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from src.news.fetcher.news_fetcher import NewsFetcher
from src.llm.gemini.gemini_helper import analyze_sentiment, query_gemini
from src.config import API_CONFIG, NEWS_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validation_sample')

def main():
    """
    Main function to fetch news, analyze them, and save results to Excel.
    """
    # Specify settings
    ticker = "AAPL"
    sample_size = 100
    
    # Set date range - only fetch news from the last 5 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    logger.info(f"Fetching news for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Initialize news fetcher with date range
    news_fetcher = NewsFetcher(
        ticker=ticker,
        max_news_articles=500,  # Fetch more to have enough to sample from
        start_date=start_date,
        end_date=end_date,
        filter_by_source=True
    )
    
    # Fetch the news articles
    news_articles = news_fetcher.fetch_news_range()
    logger.info(f"Fetched {len(news_articles)} news articles")
    
    # If we have more articles than needed, take a random sample
    if len(news_articles) > sample_size:
        news_articles = random.sample(news_articles, sample_size)
        logger.info(f"Randomly sampled {sample_size} news articles for analysis")
    
    # Process each article and apply sentiment analysis
    processed_articles = []
    for article in tqdm(news_articles, desc="Analyzing sentiment"):
        # Extract article data
        title = article.get('title', '')
        description = article.get('description', '')
        
        # Sometimes description is missing, try to get from original content if available
        if not description and 'original' in article:
            description = article['original'].get('description', '') or article['original'].get('text', '')
        
        # Skip if there's no meaningful content to analyze
        if len(title) < 5 and len(description) < 10:
            logger.warning(f"Skipping article with insufficient content: {title}")
            continue
        
        # Get source and date information
        source = article.get('site', 'Unknown Source')
        
        # Convert timestamp (milliseconds since epoch) to readable date
        timestamp_ms = article.get('time', None)
        if timestamp_ms:
            pub_date = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
        else:
            pub_date = "Unknown Date"
        
        # Analyze sentiment
        try:
            # Use the sentiment analysis function directly
            result = analyze_sentiment(title, description, ticker)
            
            # Extract sentiment and explanation
            sentiment = result.get('sentiment', 'Error')
            explanation = result.get('explanation', 'No explanation provided')
            
            # Add to processed articles
            processed_articles.append({
                'title': title,
                'description': description,
                'published_date': pub_date,
                'source': source,
                'sentiment': sentiment,
                'explanation': explanation,
                'manual_validation': ''  # Empty column for manual validation
            })
        except Exception as e:
            logger.error(f"Error analyzing sentiment for article '{title}': {e}")
    
    # Create DataFrame and save to Excel
    if processed_articles:
        df = pd.DataFrame(processed_articles)
        
        # Reorder columns for better readability
        df = df[['title', 'description', 'published_date', 'source', 'sentiment', 'explanation', 'manual_validation']]
        
        # Save to Excel with proper column widths
        filename = f"{ticker}_sentiment_validation_sample.xlsx"
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=f"{ticker} Sentiment")
            worksheet = writer.sheets[f"{ticker} Sentiment"]
            
            # Set column widths
            worksheet.column_dimensions['A'].width = 50  # Title
            worksheet.column_dimensions['B'].width = 70  # Description
            worksheet.column_dimensions['C'].width = 20  # Published date
            worksheet.column_dimensions['D'].width = 20  # Source
            worksheet.column_dimensions['E'].width = 15  # Sentiment
            worksheet.column_dimensions['F'].width = 70  # Explanation
            worksheet.column_dimensions['G'].width = 20  # Manual validation
        
        logger.info(f"Saved {len(processed_articles)} articles to {filename}")
        print(f"\nValidation sample saved to {filename}")
    else:
        logger.warning("No articles were successfully processed")
        print("\nNo articles were successfully processed.")

if __name__ == "__main__":
    main() 