"""
Centralized configuration for the Financial News Sentiment Analysis & Trading Strategy Backtester.

This module stores all configuration options for the application, making it easier to
maintain consistent settings across all components.
"""

import os
from typing import Dict, Any
import logging
import time

# API Configuration
API_CONFIG = {
    # News API configuration
    "news_api": {
        "url": "https://api.tickertick.com/feed",
        "categories": ["curated"],  # Default category
        "delay_between_requests": 12,  # Increased from 12 to 30 seconds between requests
        "max_requests_per_minute": 10, # API rate limit
        "request_timeout": 10,         # Reduced from 15 to 10 seconds for faster timeouts
        "batch_size": 100,              # Number of articles per request
        "retry_count": 5,              # Increased from 3 to 5 retries on failure
        "retry_delay": 30,             # Increased from 10 to 30 seconds after rate limit is hit
    },
    
    # Gemini API Configuration
    "gemini_api": {
        "api_key": os.environ.get("GEMINI_API_KEY", "AIzaSyALHILnxO9nhOsGspszPXKL1a2MQwGYyEM"),
        "service_account_file": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "applied-primacy-459508-f7-e65289c5b255.json"),
        "model": "gemini-2.0-flash",
        "max_retries": 3,
        "retry_delay": 2,
        "batch_size": 5,
        "max_tokens": 8192,
        "temperature": 0.7,
        "max_requests_per_minute": 2000,
        "max_parallel_requests": 200,
    },
    
    # Polygon API Configuration
    "polygon_api": {
        "api_key": os.environ.get("POLYGON_API_KEY", ""),
        "base_url": "https://api.polygon.io",
        "max_retries": 5,
        "retry_delay": 1,
        "price_check_interval": 5,  # Check price every 5 seconds for stop loss (was implicit before)
    }
}

# News Configuration
NEWS_CONFIG = {
    "default_max_articles": None,  # Fetch all articles by default
    "default_start_date": None, # Default start date for news fetching (None for all available)
    "default_end_date": None,   # Default end date for news fetching (None for all available)
    "min_articles": 50,                                    # Minimum articles to fetch
    "max_articles": 5000,                                  # Maximum articles to fetch (effective if default_max_articles is overridden)
    "max_articles_with_date_filter": 5000,  # Maximum number of articles to fetch with date filter
    "articles_per_request": 100,    # Number of articles to request per API call
    "default_sentiment_batch_size": 5,                     # Batch size for sentiment analysis
    "filter_by_source": True,     # Enable source filtering for quality news sources
    "allowed_sources": [
        # Uncomment your specific allowed sources and comment out others to customize
        "bloomberg.com", "reuters.com", "cnbc.com", "wsj.com", "ft.com", 
        "marketwatch.com", "forbes.com", "economist.com", "barrons.com", 
        "seekingalpha.com", "bbc.com", "axios.com", "techcrunch.com", 
        "arstechnica.com", "theinformation.com", "nikkei.com", "fastcompany.com", 
        "venturebeat.com", "wired.com", "techmeme.com", "technologyreview.com", 
        "protocol.com", "pymnts.com", "notboring.co", "fool.com", "investing.com",
        "stratechery.com", "platformer.news", "pomp.substack.com", 
        "bleepingcomputer.com", "chipsandcheese.com", "fiercebiotech.com", 
        "fierceedronics.com", "fiercetelecom.com", "genengnews.com", "cioinsight.com", 
        "cnet.com", "crn.com", "infoworld.com", "digiday.com", 
        "digitalinformationworld.com", "eetimes.com", "eejournal.com", 
        "modernretail.co", "datacenterdynamics.com", "datacenterknowledge.com", 
        "freightwaves.com", "skift.com", "semiaccurate.com", "semiengineering.com",
        "a16z.com", "finance.yahoo.com", "businessinsider.com", "nytimes.com",
        "washingtonpost.com"
        # Add your custom sources here
    ],
}

# Print the allowed sources at import time for verification
logging.info(f"CONFIGURED NEWS SOURCES: {NEWS_CONFIG['allowed_sources']}")

# Backtesting Configuration
BACKTEST_CONFIG = {
    "default_entry_delay": 2,                              # Default minutes to wait before entering position
    "default_holding_period": 60,                          # Default total minutes to hold position
    "default_stop_loss_pct": 2.0,                          # Default stop loss percentage (2%)
    "default_take_profit_pct": float('inf'),               # Default take profit percentage (infinite, i.e., no TP)
    "default_use_stop_loss": True,                         # Default to use stop loss
    "default_max_workers": 16,                             # Default max workers for parallel processing
    "default_include_after_hours": False,                  # Default to exclude after-hours trading
    "default_use_flip_strategy": True,                   # Default to use flip strategy
    "default_transaction_fee_pct": 0.001, # 0.1% transaction fee per trade (buy/sell)
    "min_news_count_for_backtest": 5,                      # Minimum news count required for backtesting
    "max_parallel_trades": 32,                             # Maximum parallel trade simulations
    "price_check_interval_seconds": 1,                    # How often to check price during trade simulation (reduced from 10 to 5)
    # Min/Max constraints for parameter validation
    "min_entry_delay": 0,                                  # Minimum entry delay (minutes)
    "max_entry_delay": 30,                                 # Maximum entry delay (minutes)
    "min_holding_period": 1,                               # Minimum holding period (minutes)
    "max_holding_period": 240,                             # Maximum holding period (minutes)
    "min_stop_loss_pct": 0.0,                             # Minimum stop loss percentage (5%)
    "max_stop_loss_pct": 0.5,                              # Maximum stop loss percentage (50%)
    "min_max_workers": 1,                                  # Minimum number of worker threads
    "max_max_workers": 32,                                  # Maximum number of worker threads
    "slippage_pct": 0.0005             # 0.05% slippage per trade
}

# Market Hours Configuration
MARKET_HOURS = {
    "regular_start": "09:30",  # Eastern Time
    "regular_end": "16:00",    # Eastern Time
    "extended_start": "04:00", # Eastern Time
    "extended_end": "20:00",   # Eastern Time
}

# Output Configuration
OUTPUT_CONFIG = {
    "base_dir": "output",
    "default_run_prefix": "run",
    "default_batch_prefix": "batch",
    "timestamp_format": "%Y%m%d_%H%M%S",
    "save_raw_news": True,
    "save_sentiment_results": True,
    "save_backtest_results": True,
    "save_summary_statistics": True,
}

# Event Study Configuration
EVENT_STUDY_CONFIG = {
    "default_enabled": True, # Whether to run event study by default
    "default_market_index": "SPY",  # Default market index for comparison (S&P 500 ETF)
    "alternative_indices": {
        "NASDAQ": "QQQ",  # NASDAQ-100 ETF
        "DOW": "DIA",     # Dow Jones Industrial Average ETF
        "RUSSELL": "IWM", # Russell 2000 ETF
        "TECH": "XLK"     # Technology Sector ETF
    },
    "estimation_window": 30,  # Default estimation window in days
    "event_windows": [5, 15, 30, 60],  # Default event windows in minutes
    "min_data_points": 200  # Minimum data points for market model estimation
}

# Visualization Configuration
VIZ_CONFIG = {
    "default_figsize": (12, 8),
    "default_dpi": 120,
    "default_color_palette": "viridis",
    "sentiment_colors": {
        "positive": "#2ca02c",  # Green
        "neutral": "#7f7f7f",   # Gray
        "negative": "#d62728",  # Red
    },
    "generate_sentiment_charts": True,
    "generate_backtest_charts": True,
    "generate_summary_charts": True,
}

# Batch Processing Configuration
BATCH_CONFIG = {
    "default_delay_between_tickers": 15,  # Seconds to wait between processing different tickers
    "max_consecutive_errors": 5,         # Max number of failed tickers before pausing
    "error_cooldown_period": 300,        # Seconds to wait after reaching max consecutive errors
    "default_significance_level": 0.05,  # Default significance level for statistical tests
    "default_bootstrap_simulations": 10000,
    "default_run_statistical_tests": False,
}

# Trading & Financial Constants
TRADING_CONSTANTS = {
    "TRADING_DAYS_PER_YEAR": 252,  # Standard trading days for annualization
    "MINUTES_PER_TRADING_DAY": 390,  # Regular trading hours: 9:30 AM - 4:00 PM
    "HISTORICAL_VOLATILITY_DAYS": 5,  # Days for volatility calculation in trade processing
    "MIN_VOLATILITY_DATA_POINTS": 20,  # Minimum data points needed for volatility calculation
    "VOLATILITY_FACTOR_MIN": 0.5,  # Minimum volatility adjustment factor
    "VOLATILITY_FACTOR_MAX": 2.0,  # Maximum volatility adjustment factor
    "VOLATILITY_BASE_THRESHOLD": 0.5,  # Base volatility threshold for adjustment
    "MARKET_MODEL_MIN_ABSOLUTE_POINTS": 30,  # Absolute minimum data points for market model
    "MARKET_MODEL_ESTIMATION_WINDOW_DAYS": 30,  # Default estimation window in trading days
    "MARKET_MODEL_MIN_DATA_POINTS": 200,  # Minimum data points for market model estimation
    "DATA_FETCH_BUFFER_MINUTES": 5,  # Buffer for data fetching operations
    "BOOTSTRAP_ITERATIONS": 1000,  # Default bootstrap iterations for statistical tests
    "CONFIDENCE_LEVEL": 0.95,  # Default confidence level for statistical tests
    "THREAD_FUTURE_TIMEOUT_SECONDS": 30,  # Timeout for thread futures
}

def get_default_params() -> Dict[str, Any]:
    """
    Get default parameters for processing.
    
    Returns:
        Dictionary of default processing parameters
    """
    return {
        "ticker": None,
        "is_dow_index": False,
        "is_mag7_index": False,
        "start_date": NEWS_CONFIG.get("default_start_date"),
        "end_date": NEWS_CONFIG.get("default_end_date"),
        "max_articles": NEWS_CONFIG.get("default_max_articles"),
        "articles_per_request": NEWS_CONFIG["articles_per_request"],
        "entry_delay": BACKTEST_CONFIG["default_entry_delay"],
        "holding_period": BACKTEST_CONFIG["default_holding_period"],
        "include_after_hours": BACKTEST_CONFIG["default_include_after_hours"],
        "run_backtest": True,
        "max_workers": BACKTEST_CONFIG["default_max_workers"],
        "use_stop_loss": BACKTEST_CONFIG["default_use_stop_loss"],
        "stop_loss_pct": BACKTEST_CONFIG["default_stop_loss_pct"],
        "take_profit_pct": BACKTEST_CONFIG["default_take_profit_pct"],
        "use_flip_strategy": BACKTEST_CONFIG["default_use_flip_strategy"],
        "delay_between_tickers": BATCH_CONFIG["default_delay_between_tickers"],
        "delay_between_requests": API_CONFIG["news_api"]["delay_between_requests"],
        "market_index": EVENT_STUDY_CONFIG["default_market_index"],
        "run_event_study": EVENT_STUDY_CONFIG.get("default_enabled", False),
        "event_windows": EVENT_STUDY_CONFIG["event_windows"],
        "estimation_window": EVENT_STUDY_CONFIG["estimation_window"],
        "min_data_points": EVENT_STUDY_CONFIG["min_data_points"],
        "run_statistical_tests": BATCH_CONFIG.get("default_run_statistical_tests", False),
        "bootstrap_simulations": BATCH_CONFIG["default_bootstrap_simulations"],
        "significance_level": BATCH_CONFIG["default_significance_level"],
    }

def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize processing parameters.
    
    Args:
        params: Dictionary of processing parameters
        
    Returns:
        Validated and sanitized parameters
    """
    validated = params.copy()
    
    # Validate max_articles
    # If params["max_articles"] is None (fetch all), keep it as None. Otherwise, validate.
    if params.get("max_articles") is not None:
        validated["max_articles"] = max(
            NEWS_CONFIG["min_articles"],
            min(params.get("max_articles", NEWS_CONFIG["default_max_articles"] if NEWS_CONFIG["default_max_articles"] is not None else NEWS_CONFIG["max_articles"]),
                NEWS_CONFIG["max_articles"])
        )
    else:
        validated["max_articles"] = None # Explicitly set to None if input was None
    
    # Validate articles_per_request (should always be 50 or less for TickerTick API)
    validated["articles_per_request"] = min(
        params.get("articles_per_request", NEWS_CONFIG["articles_per_request"]),
        NEWS_CONFIG["articles_per_request"]
    )
    
    # Validate entry_delay
    validated["entry_delay"] = max(
        BACKTEST_CONFIG["min_entry_delay"],
        min(params.get("entry_delay", BACKTEST_CONFIG["default_entry_delay"]), 
            BACKTEST_CONFIG["max_entry_delay"])
    )
    
    # Validate holding_period
    validated["holding_period"] = max(
        BACKTEST_CONFIG["min_holding_period"],
        min(params.get("holding_period", BACKTEST_CONFIG["default_holding_period"]), 
            BACKTEST_CONFIG["max_holding_period"])
    )
    
    # Ensure holding_period > entry_delay
    if validated["holding_period"] <= validated["entry_delay"]:
        validated["holding_period"] = validated["entry_delay"] + 1
    
    # Validate stop_loss_pct
    if params.get("use_stop_loss", BACKTEST_CONFIG["default_use_stop_loss"]):
        validated["stop_loss_pct"] = max(
            BACKTEST_CONFIG["min_stop_loss_pct"],
            min(params.get("stop_loss_pct", BACKTEST_CONFIG["default_stop_loss_pct"]),
                BACKTEST_CONFIG["max_stop_loss_pct"])
        )
    else:
        validated["stop_loss_pct"] = float('inf') # Effectively no stop loss if not used
    
    # Validate take_profit_pct
    # If take_profit_pct is 0 or float('inf'), it means no take profit.
    tp_pct = params.get("take_profit_pct", BACKTEST_CONFIG["default_take_profit_pct"])
    if tp_pct == 0 or tp_pct == float('inf'):
        validated["take_profit_pct"] = float('inf')
    else:
        # Assuming similar min/max for take_profit_pct if they were defined
        # For now, just ensure it's positive if not infinite/zero.
        # Add min_take_profit_pct and max_take_profit_pct to BACKTEST_CONFIG if needed.
        validated["take_profit_pct"] = max(0.001, tp_pct) # Must be slightly positive if used
    
    # Validate max_workers
    validated["max_workers"] = max(
        BACKTEST_CONFIG["min_max_workers"],
        min(params.get("max_workers", BACKTEST_CONFIG["default_max_workers"]), 
            BACKTEST_CONFIG["max_max_workers"])
    )
    
    # Validate use_flip_strategy (boolean)
    validated["use_flip_strategy"] = params.get("use_flip_strategy", BACKTEST_CONFIG["default_use_flip_strategy"])
    
    # Validate delay_between_requests
    validated["delay_between_requests"] = max(
        5,  # Minimum 5 seconds
        params.get("delay_between_requests", API_CONFIG["news_api"]["delay_between_requests"])
    )
    
    # Validate delay_between_tickers
    validated["delay_between_tickers"] = max(
        10,  # Minimum 10 seconds
        params.get("delay_between_tickers", BATCH_CONFIG["default_delay_between_tickers"])
    )
    
    return validated 