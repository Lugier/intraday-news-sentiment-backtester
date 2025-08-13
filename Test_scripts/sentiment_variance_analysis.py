#!/usr/bin/env python
"""
Sentiment Variance Analysis Script - Multi-Run Consistency Test

This script fetches ALL available news articles about Apple (AAPL), runs sentiment analysis 
5 times on the entire dataset, and compares the results to measure the consistency 
across multiple complete runs.

The script will:
1. Fetch ALL available news articles for AAPL (no time limit)
2. Run 5 complete sentiment analysis runs on all articles
3. Analyze the consistency of classifications between all runs
4. Calculate median consistency and variance metrics
5. Export detailed results with cross-run comparison statistics
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np
import logging
import time
import concurrent.futures
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from collections import Counter, defaultdict

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
logger = logging.getLogger('sentiment_variance_analysis')

# Rate limiting configuration - maximum API requests per minute
MAX_REQUESTS_PER_MINUTE = 2000
REQUEST_TIMESTAMPS = []

def prepare_article_for_analysis(article):
    """
    Extract relevant fields from article for sentiment analysis.
    
    Args:
        article: News article dictionary
        
    Returns:
        Dictionary with title, description, and metadata for analysis
    """
    ticker = "AAPL"
    title = article.get('title', '')
    description = article.get('description', '')
    
    # Sometimes description is missing, try to get from original content if available
    if not description and 'original' in article:
        description = article['original'].get('description', '') or article['original'].get('text', '')
    
    # Extract timestamp (milliseconds since epoch) and convert to readable date
    timestamp_ms = article.get('time', None)
    pub_date = "Unknown Date"
    if timestamp_ms:
        pub_date = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract source
    source = article.get('site', 'Unknown Source')
    
    return {
        "article_id": article.get('id', ''),
        "title": title,
        "description": description,
        "published_date": pub_date,
        "source": source,
        "ticker": ticker
    }

def wait_if_rate_limited():
    """Check if we're approaching the rate limit and wait if necessary"""
    global REQUEST_TIMESTAMPS
    
    current_time = datetime.now()
    # Remove timestamps older than 1 minute
    REQUEST_TIMESTAMPS = [ts for ts in REQUEST_TIMESTAMPS 
                         if current_time - ts < timedelta(minutes=1)]
    
    # If we've made too many requests in the last minute, wait
    if len(REQUEST_TIMESTAMPS) >= MAX_REQUESTS_PER_MINUTE * 0.95:  # 95% of limit
        oldest_timestamp = min(REQUEST_TIMESTAMPS) if REQUEST_TIMESTAMPS else current_time
        wait_time = max(0, 60 - (current_time - oldest_timestamp).total_seconds()) + 1
        logger.info(f"Rate limit approaching ({len(REQUEST_TIMESTAMPS)}/{MAX_REQUESTS_PER_MINUTE}). Waiting {wait_time:.2f}s...")
        time.sleep(wait_time)

def process_single_article(article):
    """Process a single article with sentiment analysis"""
    global REQUEST_TIMESTAMPS
    
    title = article['title']
    description = article['description']
    ticker = article['ticker']
    
    # Skip if there's no meaningful content to analyze
    if len(title) < 5 and len(description) < 10:
        return None
    
    # Single analysis
    wait_if_rate_limited()
    try:
        result = analyze_sentiment(title, description, ticker)
        REQUEST_TIMESTAMPS.append(datetime.now())
        
        if result:
            return {
                "article_id": article['article_id'],
                "title": article['title'],
                "published_date": article['published_date'],
                "source": article['source'],
                "sentiment": result.get('sentiment'),
                "explanation": result.get('explanation', '')
            }
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return None
    
    return None

def run_sentiment_analysis_single_run(prepared_articles, run_number, batch_size=200, max_workers=50):
    """
    Run sentiment analysis on all articles for a single complete run.
    
    Args:
        prepared_articles: List of dictionaries with title, description for analysis
        run_number: Current run number (1-5)
        batch_size: Size of batches to process at once
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of dictionaries with sentiment analysis results
    """
    logger.info(f"Starting Run {run_number} - Processing {len(prepared_articles)} articles")
    results = []
    
    # Process in batches
    for i in range(0, len(prepared_articles), batch_size):
        end_idx = min(i + batch_size, len(prepared_articles))
        batch = prepared_articles[i:end_idx]
        
        # Filter valid articles before processing
        valid_articles = []
        for article in batch:
            if len(article['title']) >= 5 or len(article['description']) >= 10:
                valid_articles.append(article)
        
        if not valid_articles:
            continue
        
        batch_results = []
        logger.info(f"Run {run_number} - Processing batch of {len(valid_articles)} articles")
        
        # Process articles in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_article = {executor.submit(process_single_article, article): article for article in valid_articles}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_article), 
                             total=len(valid_articles), 
                             desc=f"Run {run_number} - Analyzing articles"):
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error in article processing: {e}")
        
        results.extend(batch_results)
        logger.info(f"Run {run_number} - Completed batch, processed {len(batch_results)} articles successfully")
    
    logger.info(f"Run {run_number} completed - Total articles processed: {len(results)}")
    return results

def calculate_cross_run_consistency(all_runs_results):
    """
    Calculate consistency metrics across all 5 runs.
    
    Args:
        all_runs_results: List of 5 lists, each containing results from one run
        
    Returns:
        Dictionary containing cross-run consistency statistics
    """
    # Create a mapping from article_id to sentiments across runs
    article_sentiments = defaultdict(list)
    
    # Collect sentiments for each article across all runs
    for run_idx, run_results in enumerate(all_runs_results):
        for result in run_results:
            article_id = result['article_id']
            sentiment = result['sentiment']
            article_sentiments[article_id].append(sentiment)
    
    # Only consider articles that were successfully analyzed in all 5 runs
    complete_articles = {aid: sentiments for aid, sentiments in article_sentiments.items() 
                        if len(sentiments) == 5}
    
    logger.info(f"Articles analyzed in all 5 runs: {len(complete_articles)}")
    
    if not complete_articles:
        return {
            "total_articles_all_runs": 0,
            "perfect_consistency_rate": 0,
            "majority_consistency_rate": 0,
            "median_agreement_rate": 0,
            "cross_run_kappa_scores": [],
            "sentiment_stability": {}
        }
    
    # Calculate perfect consistency (all 5 runs agree)
    perfect_consistency_count = 0
    majority_consistency_count = 0
    
    # Calculate pairwise agreement rates for all combinations
    pairwise_agreements = []
    
    # Calculate sentiment stability for each article
    sentiment_stability = []
    
    for article_id, sentiments in complete_articles.items():
        # Perfect consistency: all 5 sentiments are the same
        if len(set(sentiments)) == 1:
            perfect_consistency_count += 1
            majority_consistency_count += 1
            sentiment_stability.append(1.0)  # 100% stable
        else:
            # Majority consistency: most common sentiment appears >= 3 times
            sentiment_counts = Counter(sentiments)
            most_common_count = sentiment_counts.most_common(1)[0][1]
            if most_common_count >= 3:
                majority_consistency_count += 1
            
            # Stability score: proportion of most common sentiment
            sentiment_stability.append(most_common_count / 5.0)
    
    # Calculate pairwise agreements between all run combinations
    for i in range(5):
        for j in range(i + 1, 5):
            agreements = 0
            total_pairs = 0
            
            for article_id, sentiments in complete_articles.items():
                if sentiments[i] == sentiments[j]:
                    agreements += 1
                total_pairs += 1
            
            if total_pairs > 0:
                pairwise_agreements.append(agreements / total_pairs)
    
    # Calculate Cohen's Kappa for each pair of runs
    kappa_scores = []
    sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    
    for i in range(5):
        for j in range(i + 1, 5):
            sentiments_i = []
            sentiments_j = []
            
            for article_id, sentiments in complete_articles.items():
                if sentiments[i] in sentiment_map and sentiments[j] in sentiment_map:
                    sentiments_i.append(sentiment_map[sentiments[i]])
                    sentiments_j.append(sentiment_map[sentiments[j]])
            
            if len(sentiments_i) > 0:
                kappa = cohen_kappa_score(sentiments_i, sentiments_j)
                kappa_scores.append(kappa)
    
    # Calculate overall sentiment distribution consistency
    run_distributions = []
    for run_results in all_runs_results:
        sentiments = [r['sentiment'] for r in run_results if r['sentiment']]
        distribution = {
            'positive': sentiments.count('positive') / len(sentiments) * 100 if sentiments else 0,
            'neutral': sentiments.count('neutral') / len(sentiments) * 100 if sentiments else 0,
            'negative': sentiments.count('negative') / len(sentiments) * 100 if sentiments else 0
        }
        run_distributions.append(distribution)
    
    return {
        "total_articles_all_runs": len(complete_articles),
        "perfect_consistency_rate": (perfect_consistency_count / len(complete_articles)) * 100 if complete_articles else 0,
        "majority_consistency_rate": (majority_consistency_count / len(complete_articles)) * 100 if complete_articles else 0,
        "median_agreement_rate": np.median(pairwise_agreements) * 100 if pairwise_agreements else 0,
        "mean_agreement_rate": np.mean(pairwise_agreements) * 100 if pairwise_agreements else 0,
        "std_agreement_rate": np.std(pairwise_agreements) * 100 if pairwise_agreements else 0,
        "cross_run_kappa_scores": kappa_scores,
        "median_kappa_score": np.median(kappa_scores) if kappa_scores else 0,
        "mean_kappa_score": np.mean(kappa_scores) if kappa_scores else 0,
        "sentiment_stability": {
            "median_stability": np.median(sentiment_stability) * 100 if sentiment_stability else 0,
            "mean_stability": np.mean(sentiment_stability) * 100 if sentiment_stability else 0,
            "std_stability": np.std(sentiment_stability) * 100 if sentiment_stability else 0
        },
        "run_distributions": run_distributions,
        "pairwise_agreement_rates": [rate * 100 for rate in pairwise_agreements]
    }

def save_intermediate_results(results, ticker, timestamp, run_number):
    """Save intermediate results for each run"""
    filename = f"{ticker}_sentiment_analysis_run{run_number}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved Run {run_number} results to {filename}")
    return filename

def main():
    """
    Main function to fetch all news and run 5 complete sentiment analysis runs.
    """
    # Specify settings
    ticker = "AAPL"
    num_runs = 5
    batch_size = 200
    max_workers = 50
    
    # No time limit - fetch ALL available news
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logger.info(f"Fetching ALL available news for {ticker} (no time limit)")
    
    # Initialize news fetcher to get ALL articles
    news_fetcher = NewsFetcher(
        ticker=ticker,
        max_news_articles=50000,  # Very high number to get all available
        filter_by_source=True
    )
    
    # Fetch the news articles
    logger.info(f"Starting news fetch for {ticker}...")
    news_articles = news_fetcher.fetch_and_save_all_stories()
    total_fetched = len(news_articles)
    logger.info(f"Fetched {total_fetched} news articles")
    
    # Prepare articles for processing
    logger.info("Preparing articles for sentiment analysis...")
    prepared_articles = []
    for article in tqdm(news_articles, desc="Preparing articles"):
        prepared = prepare_article_for_analysis(article)
        if len(prepared['title']) >= 5 or len(prepared['description']) >= 10:
            prepared_articles.append(prepared)
    
    logger.info(f"Prepared {len(prepared_articles)} valid articles for analysis")
    
    # Run sentiment analysis 5 times on all articles
    all_runs_results = []
    run_summaries = []
    
    for run_num in range(1, num_runs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"STARTING RUN {run_num}/{num_runs}")
        logger.info(f"{'='*50}")
        
        start_time = time.time()
        
        # Process all articles for this run
        run_results = run_sentiment_analysis_single_run(
            prepared_articles, run_num, batch_size, max_workers
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Save intermediate results
        save_intermediate_results(run_results, ticker, timestamp, run_num)
        
        # Calculate run summary
        sentiments = [r['sentiment'] for r in run_results if r['sentiment']]
        run_summary = {
            "run_number": run_num,
            "articles_processed": len(run_results),
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "sentiment_distribution": {
                'positive': sentiments.count('positive'),
                'neutral': sentiments.count('neutral'),
                'negative': sentiments.count('negative')
            },
            "sentiment_percentages": {
                'positive': sentiments.count('positive') / len(sentiments) * 100 if sentiments else 0,
                'neutral': sentiments.count('neutral') / len(sentiments) * 100 if sentiments else 0,
                'negative': sentiments.count('negative') / len(sentiments) * 100 if sentiments else 0
            }
        }
        
        run_summaries.append(run_summary)
        all_runs_results.append(run_results)
        
        logger.info(f"Run {run_num} completed in {duration/60:.2f} minutes")
        logger.info(f"Processed {len(run_results)} articles")
        logger.info(f"Sentiment distribution: {run_summary['sentiment_distribution']}")
    
    # Calculate cross-run consistency metrics
    logger.info("\nCalculating cross-run consistency metrics...")
    consistency_stats = calculate_cross_run_consistency(all_runs_results)
    
    # Create comprehensive output
    output = {
        "analysis_info": {
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "ticker": ticker,
            "total_news_fetched": total_fetched,
            "valid_articles_prepared": len(prepared_articles),
            "number_of_runs": num_runs,
            "batch_size": batch_size,
            "max_workers": max_workers
        },
        "run_summaries": run_summaries,
        "cross_run_consistency": consistency_stats,
        "detailed_results_by_run": {
            f"run_{i+1}": results for i, results in enumerate(all_runs_results)
        }
    }
    
    # Save comprehensive results
    final_filename = f"{ticker}_multi_run_sentiment_analysis_{timestamp}_final.json"
    with open(final_filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Create Excel summary
    summary_data = []
    for i, summary in enumerate(run_summaries):
        summary_data.append({
            "Run": summary['run_number'],
            "Articles Processed": summary['articles_processed'],
            "Duration (minutes)": round(summary['duration_minutes'], 2),
            "Positive %": round(summary['sentiment_percentages']['positive'], 2),
            "Neutral %": round(summary['sentiment_percentages']['neutral'], 2),
            "Negative %": round(summary['sentiment_percentages']['negative'], 2),
            "Positive Count": summary['sentiment_distribution']['positive'],
            "Neutral Count": summary['sentiment_distribution']['neutral'],
            "Negative Count": summary['sentiment_distribution']['negative']
        })
    
    # Add consistency metrics
    consistency_data = [{
        "Metric": "Total News Fetched",
        "Value": total_fetched
    }, {
        "Metric": "Valid Articles Prepared",
        "Value": len(prepared_articles)
    }, {
        "Metric": "Articles in All 5 Runs",
        "Value": consistency_stats['total_articles_all_runs']
    }, {
        "Metric": "Perfect Consistency Rate (%)",
        "Value": round(consistency_stats['perfect_consistency_rate'], 2)
    }, {
        "Metric": "Majority Consistency Rate (%)",
        "Value": round(consistency_stats['majority_consistency_rate'], 2)
    }, {
        "Metric": "Median Agreement Rate (%)",
        "Value": round(consistency_stats['median_agreement_rate'], 2)
    }, {
        "Metric": "Mean Agreement Rate (%)",
        "Value": round(consistency_stats['mean_agreement_rate'], 2)
    }, {
        "Metric": "Median Kappa Score",
        "Value": round(consistency_stats['median_kappa_score'], 4)
    }, {
        "Metric": "Median Sentiment Stability (%)",
        "Value": round(consistency_stats['sentiment_stability']['median_stability'], 2)
    }]
    
    excel_filename = f"{ticker}_multi_run_analysis_summary_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_filename) as writer:
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Run Summaries", index=False)
        pd.DataFrame(consistency_data).to_excel(writer, sheet_name="Consistency Metrics", index=False)
        
        # Add pairwise agreement rates
        if consistency_stats['pairwise_agreement_rates']:
            pairwise_data = []
            pair_idx = 0
            for i in range(5):
                for j in range(i + 1, 5):
                    pairwise_data.append({
                        "Run Pair": f"Run {i+1} vs Run {j+1}",
                        "Agreement Rate (%)": round(consistency_stats['pairwise_agreement_rates'][pair_idx], 2)
                    })
                    pair_idx += 1
            pd.DataFrame(pairwise_data).to_excel(writer, sheet_name="Pairwise Agreements", index=False)
    
    # Print comprehensive summary
    logger.info(f"\n{'='*60}")
    logger.info(f"MULTI-RUN SENTIMENT ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total news articles fetched: {total_fetched}")
    logger.info(f"Valid articles prepared: {len(prepared_articles)}")
    logger.info(f"Number of complete runs: {num_runs}")
    logger.info(f"Articles analyzed in all runs: {consistency_stats['total_articles_all_runs']}")
    logger.info(f"\nCONSISTENCY METRICS:")
    logger.info(f"- Perfect consistency rate: {consistency_stats['perfect_consistency_rate']:.2f}%")
    logger.info(f"- Majority consistency rate: {consistency_stats['majority_consistency_rate']:.2f}%")
    logger.info(f"- Median agreement rate: {consistency_stats['median_agreement_rate']:.2f}%")
    logger.info(f"- Median Kappa score: {consistency_stats['median_kappa_score']:.4f}")
    logger.info(f"- Median sentiment stability: {consistency_stats['sentiment_stability']['median_stability']:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Total news articles fetched: {total_fetched}")
    print(f"✅ Valid articles prepared: {len(prepared_articles)}")
    print(f"🔄 Number of complete runs: {num_runs}")
    print(f"📈 Articles analyzed in all runs: {consistency_stats['total_articles_all_runs']}")
    print(f"\n🎯 CONSISTENCY RESULTS:")
    print(f"   Perfect consistency: {consistency_stats['perfect_consistency_rate']:.2f}%")
    print(f"   Majority consistency: {consistency_stats['majority_consistency_rate']:.2f}%")
    print(f"   Median agreement rate: {consistency_stats['median_agreement_rate']:.2f}%")
    print(f"   Median Kappa score: {consistency_stats['median_kappa_score']:.4f}")
    print(f"   Median sentiment stability: {consistency_stats['sentiment_stability']['median_stability']:.2f}%")
    print(f"\n📁 Results saved to:")
    print(f"   - {final_filename}")
    print(f"   - {excel_filename}")

if __name__ == "__main__":
    main() 