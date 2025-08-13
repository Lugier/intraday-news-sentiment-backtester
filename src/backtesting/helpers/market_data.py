"""
Market Data Module

Fetches market data (stock prices) using the Polygon API.
"""

import os
import requests
import pandas as pd
from datetime import datetime, time, timedelta
import logging
from typing import Optional, Dict, Any, Tuple
from polygon import RESTClient
import pytz

from src.config import API_CONFIG

logger = logging.getLogger(__name__)

# Eastern Time timezone for US markets
EASTERN_TZ = pytz.timezone('US/Eastern')

# Global API client
_client = None

def convert_to_eastern_time(dt: datetime) -> datetime:
    """
    Convert any datetime to Eastern Time for proper market hours calculation.
    
    Args:
        dt: Input datetime (any timezone or naive)
        
    Returns:
        Datetime converted to Eastern Time
    """
    if dt.tzinfo is None:
        # Assume naive datetime is already in Eastern Time
        # This is a reasonable assumption for US market data
        return EASTERN_TZ.localize(dt)
    else:
        # Convert timezone-aware datetime to Eastern Time
        return dt.astimezone(EASTERN_TZ)

def get_polygon_client() -> RESTClient:
    """
    Get a cached instance of the Polygon REST client.
    
    Returns:
        RESTClient: Initialized Polygon API client
    """
    global _client
    if _client is None:
        api_key = os.getenv("POLYGON_API_KEY")
        
        if not api_key:
            # Versuche, den Schlüssel aus der API_CONFIG zu laden
            api_key = API_CONFIG.get("polygon_api", {}).get("api_key", "")
            
        if not api_key:
            logger.error("POLYGON_API_KEY not found in environment variables or configuration.")
            raise ValueError("POLYGON_API_KEY not set. Please set this in your .env file.")
            
        logger.info("Initializing Polygon API client")
        _client = RESTClient(api_key)
        
    return _client

def get_market_hours() -> Tuple[time, time]:
    """
    Get the standard US market hours in Eastern Time.
    
    Returns:
        Tuple of (market_open, market_close) times in Eastern Time
    """
    market_open = time(9, 30)  # 9:30 AM Eastern
    market_close = time(16, 0)  # 4:00 PM Eastern
    return market_open, market_close

def get_stock_data(
    ticker: str,
    timespan: str = "minute",
    from_date: str = None,
    to_date: str = None,
    limit: int = 50000
) -> pd.DataFrame:
    """
    Fetch stock data from Polygon API.
    
    Args:
        ticker: Stock ticker symbol
        timespan: Time granularity (minute, hour, day, etc.)
        from_date: Start date in format YYYY-MM-DD
        to_date: End date in format YYYY-MM-DD
        limit: Maximum number of data points to fetch
        
    Returns:
        DataFrame with stock price data
    """
    client = get_polygon_client()
    
    # Set default dates if not provided
    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d")
    if from_date is None:
        # Default to 30 days before to_date
        to_datetime = datetime.strptime(to_date, "%Y-%m-%d")
        from_datetime = to_datetime - timedelta(days=30)
        from_date = from_datetime.strftime("%Y-%m-%d")
    
    logger.info(f"Fetching {timespan} data for {ticker} from {from_date} to {to_date}")
    
    try:
        # Fetch data from Polygon API
        aggs = []
        for agg in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            limit=limit,
        ):
            aggs.append(agg)
        
        # Convert to DataFrame
        if not aggs:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        df = pd.DataFrame([{
            'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
            'open': agg.open,
            'high': agg.high,
            'low': agg.low,
            'close': agg.close,
            'volume': agg.volume,
            'vwap': getattr(agg, 'vwap', None),
            'transactions': getattr(agg, 'transactions', None)
        } for agg in aggs])
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"Fetched {len(df)} {timespan} bars for {ticker}")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def fetch_stock_prices(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    timespan: str = "minute"
) -> pd.DataFrame:
    """
    Fetch stock price data for a specific ticker and date range.
    This function only uses real market data from Polygon.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start datetime
        end_date: End datetime
        timespan: Time granularity (minute, hour, day, etc.)
        
    Returns:
        DataFrame with stock price data
    """
    # Format dates for API calls
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Fetching {timespan} data for {ticker} from {from_date} to {to_date}")
    
    # Always use real data from Polygon API - never simulate
    df = get_stock_data(ticker, timespan, from_date, to_date)
    
    if df.empty:
        logger.warning(f"No data returned from Polygon API for {ticker} from {from_date} to {to_date}")
    else:
        logger.info(f"Successfully fetched {len(df)} data points for {ticker}")
        
    return df

def fetch_market_index(
    market_index: str = "SPY",
    start_date: datetime = None,
    end_date: datetime = None,
    timespan: str = "minute"
) -> pd.DataFrame:
    """
    Fetch market index data for use in market models.
    This is a convenience function that maps common index names to their ETF tickers.
    
    Args:
        market_index: Market index name or ETF ticker (e.g., "SPY", "QQQ", "S&P500", "NASDAQ")
        start_date: Start datetime
        end_date: End datetime
        timespan: Time granularity (minute, hour, day, etc.)
        
    Returns:
        DataFrame with market index data
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    # Map common index names to ETF tickers
    index_map = {
        "S&P500": "SPY",
        "S&P": "SPY",
        "SP500": "SPY",
        "NASDAQ": "QQQ",
        "NASDQ": "QQQ",
        "DOW": "DIA",
        "DOWJONES": "DIA",
        "DOW JONES": "DIA",
        "RUSSELL": "IWM",
        "RUSSELL2000": "IWM",
        "RUSSELL 2000": "IWM"
    }
    
    # Convert common names to ticker symbols
    ticker = index_map.get(market_index.upper(), market_index)
    
    logger.info(f"Fetching market index data for {market_index} (using {ticker})")
    
    # Use existing fetch_stock_prices function with the ETF ticker
    return fetch_stock_prices(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        timespan=timespan
    )

def is_market_open(dt: datetime, extended: bool = True) -> bool:
    """
    Determine if the US stock market is open at a given time.
    
    Args:
        dt: Datetime to check (any timezone or naive)
        extended: Whether to include extended hours (pre/post market)
        
    Returns:
        Boolean indicating if market is open
    """
    # Convert to Eastern Time for proper market hours calculation
    et_dt = convert_to_eastern_time(dt)
    t = et_dt.time()
    
    # Check if weekend
    if et_dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Regular trading hours: 9:30 AM - 4:00 PM Eastern
    market_open, market_close = get_market_hours()
    
    if extended:
        # Extended hours: 4:00 AM - 8:00 PM Eastern
        extended_open = time(4, 0)   # 4:00 AM
        extended_close = time(20, 0) # 8:00 PM
        return extended_open <= t <= extended_close
    else:
        # Regular trading hours only
        return market_open <= t <= market_close

def get_minute_data_for_news_event(
    ticker: str,
    news_timestamp: datetime,
    minutes_before: int = 10,
    minutes_after: int = 60
) -> pd.DataFrame:
    """
    Get minute-by-minute stock data around a news event.
    
    Args:
        ticker: Stock ticker symbol
        news_timestamp: Datetime of the news publication
        minutes_before: Minutes of data to fetch before the news
        minutes_after: Minutes of data to fetch after the news
        
    Returns:
        DataFrame with stock data around the news event
    """
    # Calculate date range for query
    start_time = news_timestamp - timedelta(minutes=minutes_before)
    end_time = news_timestamp + timedelta(minutes=minutes_after)
    
    # Format dates for Polygon API
    from_date = start_time.strftime("%Y-%m-%d")
    to_date = end_time.strftime("%Y-%m-%d")
    
    # Get minute data at higher frequency for more accurate stop loss detection
    # We'll use "minute" timespan which is the highest resolution available
    df = get_stock_data(ticker, "minute", from_date, to_date, limit=100000)  # Increased limit for higher granularity
    
    # Filter to our exact time window
    df = df[
        (df.index >= start_time) & 
        (df.index <= end_time)
    ]
    
    # Add news timing information
    df['minutes_from_news'] = (df.index - news_timestamp).total_seconds() / 60
    
    # Log data points to help diagnose stop loss detection
    logger.debug(f"Fetched {len(df)} price points for {ticker} around news at {news_timestamp}")
    
    return df 