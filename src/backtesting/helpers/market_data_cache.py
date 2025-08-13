"""
Market Data Cache Module

This module implements a caching mechanism for market data to avoid 
redundant API calls when analyzing multiple news events.
"""

import pandas as pd
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from src.backtesting.helpers.market_data import fetch_stock_prices, fetch_market_index

logger = logging.getLogger(__name__)

class MarketDataCache:
    """
    Thread-safe cache for storing and retrieving market data to minimize API calls.
    """
    
    def __init__(self):
        """Initialize an empty cache with thread safety."""
        self._cache = {}
        self._market_cache = {}
        self._initialized = False
        self._lock = threading.RLock()  # Reentrant lock for thread safety
    
    def initialize_for_period(self, ticker: str, market_index: str, 
                             start_date: datetime, end_date: datetime,
                             timespan: str = "minute") -> None:
        """
        Initialize the cache with data for the entire analysis period.
        Thread-safe operation.
        
        Args:
            ticker: Stock ticker symbol
            market_index: Market index symbol
            start_date: Start date for data
            end_date: End date for data
            timespan: Time granularity
        """
        with self._lock:  # Thread-safe initialization
            logger.info(f"Initializing market data cache for {ticker} and {market_index} from {start_date} to {end_date}")
            
            # Fetch all stock data for the period
            stock_data = fetch_stock_prices(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                timespan=timespan
            )
            
            # Fetch all market data for the period
            market_data = fetch_market_index(
                market_index=market_index,
                start_date=start_date,
                end_date=end_date,
                timespan=timespan
            )
            
            # Store in cache
            self._cache[ticker] = stock_data
            self._market_cache[market_index] = market_data
            self._initialized = True
            
            logger.info(f"Cache initialized with {len(stock_data)} stock data points and {len(market_data)} market data points")
    
    def get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime,
                      timespan: str = "minute") -> pd.DataFrame:
        """
        Get stock data from cache or fetch if not available.
        Thread-safe operation.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            timespan: Time granularity
            
        Returns:
            DataFrame with stock price data
        """
        with self._lock:  # Thread-safe cache access
            # Check if we have this ticker in cache
            if ticker in self._cache:
                # Filter data for the requested period
                data = self._cache[ticker]
                filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
                
                # If we have sufficient data, return from cache
                if not filtered_data.empty and len(filtered_data) > 5:  # Arbitrary threshold
                    logger.debug(f"Retrieved {len(filtered_data)} data points for {ticker} from cache")
                    return filtered_data
            
            # If not in cache or insufficient data, fetch from API
            logger.info(f"Data for {ticker} not in cache or insufficient, fetching from API")
            data = fetch_stock_prices(ticker, start_date, end_date, timespan)
            
            # Store in cache for future use
            if ticker not in self._cache:
                self._cache[ticker] = data
            else:
                # Merge with existing data to avoid duplicates - atomic operation
                combined_data = pd.concat([self._cache[ticker], data]).drop_duplicates()
                combined_data.sort_index(inplace=True)
                self._cache[ticker] = combined_data  # Atomic assignment
            
            return data
    
    def get_market_data(self, market_index: str, start_date: datetime, end_date: datetime,
                      timespan: str = "minute") -> pd.DataFrame:
        """
        Get market data from cache or fetch if not available.
        Thread-safe operation.
        
        Args:
            market_index: Market index symbol
            start_date: Start date for data
            end_date: End date for data
            timespan: Time granularity
            
        Returns:
            DataFrame with market price data
        """
        with self._lock:  # Thread-safe cache access
            # Check if we have this market index in cache
            if market_index in self._market_cache:
                # Filter data for the requested period
                data = self._market_cache[market_index]
                filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
                
                # If we have sufficient data, return from cache
                if not filtered_data.empty and len(filtered_data) > 5:  # Arbitrary threshold
                    logger.debug(f"Retrieved {len(filtered_data)} data points for {market_index} from cache")
                    return filtered_data
            
            # If not in cache or insufficient data, fetch from API
            logger.info(f"Data for {market_index} not in cache or insufficient, fetching from API")
            data = fetch_market_index(market_index, start_date, end_date, timespan)
            
            # Store in cache for future use
            if market_index not in self._market_cache:
                self._market_cache[market_index] = data
            else:
                # Merge with existing data to avoid duplicates - atomic operation
                combined_data = pd.concat([self._market_cache[market_index], data]).drop_duplicates()
                combined_data.sort_index(inplace=True)
                self._market_cache[market_index] = combined_data  # Atomic assignment
            
            return data
    
    def clear(self) -> None:
        """Clear the cache in a thread-safe manner."""
        with self._lock:  # Thread-safe cache clearing
            self._cache = {}
            self._market_cache = {}
            self._initialized = False
            logger.info("Market data cache cleared")

# Global cache instance
market_data_cache = MarketDataCache()

def get_cached_stock_data(ticker: str, start_date: datetime, end_date: datetime,
                         timespan: str = "minute") -> pd.DataFrame:
    """
    Get stock data with caching to minimize API calls.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        timespan: Time granularity
        
    Returns:
        DataFrame with stock price data
    """
    return market_data_cache.get_stock_data(ticker, start_date, end_date, timespan)

def get_cached_market_data(market_index: str, start_date: datetime, end_date: datetime,
                          timespan: str = "minute") -> pd.DataFrame:
    """
    Get market index data with caching to minimize API calls.
    
    Args:
        market_index: Market index symbol
        start_date: Start date for data
        end_date: End date for data
        timespan: Time granularity
        
    Returns:
        DataFrame with market price data
    """
    return market_data_cache.get_market_data(market_index, start_date, end_date, timespan)

def initialize_market_data_cache(ticker: str, market_index: str,
                               start_date: datetime, end_date: datetime,
                               timespan: str = "minute") -> None:
    """
    Initialize cache with all data needed for analysis period.
    Call this once at the beginning of your analysis.
    
    Args:
        ticker: Stock ticker symbol
        market_index: Market index symbol
        start_date: Start date for data
        end_date: End date for data
        timespan: Time granularity
    """
    market_data_cache.initialize_for_period(ticker, market_index, start_date, end_date, timespan)

def clear_market_data_cache() -> None:
    """Clear the market data cache."""
    market_data_cache.clear() 