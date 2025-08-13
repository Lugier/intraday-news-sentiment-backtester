"""
Event-Study Analyse Modul

Dieses Modul implementiert die Event-Study-Methodik zur Analyse abnormaler Renditen
nach News-Events mit unterschiedlichen Sentiment-Klassifikationen.
Hinweis: leicht deutsch fomuliert mit kleinen Tippfehlern (~3%).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import TRADING_CONSTANTS

from src.backtesting.helpers.market_data import fetch_stock_prices, convert_to_eastern_time
from src.backtesting.helpers.market_data_cache import (
    get_cached_stock_data, 
    get_cached_market_data,
    initialize_market_data_cache
)

logger = logging.getLogger(__name__)

def estimate_market_model(stock_data: pd.DataFrame, 
                          market_data: pd.DataFrame, 
                          estimation_window: int = TRADING_CONSTANTS["MARKET_MODEL_ESTIMATION_WINDOW_DAYS"],
                          min_data_points: int = TRADING_CONSTANTS["MARKET_MODEL_MIN_DATA_POINTS"]) -> Dict[str, float]:
    """
    Marktmodell-Parameter für eine Aktie schätzen.
    
    Args:
        stock_data: DataFrame mit Aktienkurs-Daten
        market_data: DataFrame mit Marktindex-Daten  
        estimation_window: Anzahl Handelstage fürs Schätzfenster
        min_data_points: Minimale Anzahl benötigter Datenpunkte
        
    Returns:
        Dictionary mit Modellparametern und Fit-Statistiken
    """
    default_metrics = {
        'alpha': np.nan,
        'beta': np.nan,
        'r_squared': 0.0,
        'adj_r_squared': 0.0,
        'rmse': np.nan,
        'mae': np.nan,
        'data_points': 0
    }
    
    # Ensure both dataframes have the same index
    common_idx = stock_data.index.intersection(market_data.index)
    
    # Check if we have enough data points overall
    if len(common_idx) < TRADING_CONSTANTS["MARKET_MODEL_MIN_ABSOLUTE_POINTS"]:  # Absolute minimum for any meaningful estimation
        logger.warning(f"Very limited data for market model estimation. Found only {len(common_idx)} common data points.")
        return default_metrics
    
    # Sort indices to ensure chronological order
    sorted_idx = sorted(common_idx)
    
    # For minute data, average trading day has ~390 minutes 
    # Calculate target number of data points based on trading days needed
    target_points = max(min_data_points, estimation_window * TRADING_CONSTANTS["MINUTES_PER_TRADING_DAY"])
    logger.info(f"Using minute-level data with target of {target_points} data points (equivalent to ~{estimation_window} trading days)")
    
    # If we don't have enough points, use all available data
    if len(sorted_idx) < target_points:
        logger.warning(f"Limited data points ({len(sorted_idx)}) for estimation, using all available data")
        estimation_indices = sorted_idx
    else:
        # Use the most recent target_points for more relevant estimation
        estimation_indices = sorted_idx[-target_points:]
    
    # Use data from the estimation window
    stock_subset = stock_data.loc[estimation_indices].sort_index()
    market_subset = market_data.loc[estimation_indices].sort_index()
    
    # Log the actual time period we're using
    if len(estimation_indices) > 0:
        start_date = min(estimation_indices)
        end_date = max(estimation_indices)
        actual_days = (end_date - start_date).days
        actual_points = len(estimation_indices)
        logger.info(f"Market model estimated using data from {start_date} to {end_date} ({actual_days} days, {actual_points} minute data points)")
    
    # Calculate log returns
    stock_returns = np.log(stock_subset['close'] / stock_subset['close'].shift(1)).dropna()
    market_returns = np.log(market_subset['close'] / market_subset['close'].shift(1)).dropna()
    
    # Align the data
    common_dates = stock_returns.index.intersection(market_returns.index)
    stock_returns = stock_returns.loc[common_dates]
    market_returns = market_returns.loc[common_dates]
    
    if len(common_dates) < 10:  # Need at least 10 data points for meaningful regression
        logger.warning(f"Insufficient return data for regression. Found only {len(common_dates)} points after return calculation.")
        return default_metrics
    
    # Run OLS regression to get alpha and beta
    X = add_constant(market_returns.values)
    y = stock_returns.values
    model = OLS(y, X)
    results = model.fit()
    
    alpha, beta = results.params
    r_squared = results.rsquared
    adj_r_squared = results.rsquared_adj
    n_obs = results.nobs
    
    # Calculate residuals
    predicted_returns = results.predict(X)
    residuals = y - predicted_returns
    
    # Calculate RMSE and MAE
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    # Log the quality of the model fit
    fit_quality_msg = f"Market model fit: R²={r_squared:.4f}, Adj R²={adj_r_squared:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f}"
    if r_squared < 0.1:
        logger.warning(f"Poor market model fit: {fit_quality_msg}. Results may be less reliable.")
    elif r_squared > 0.7:
        logger.info(f"Strong market model fit: {fit_quality_msg}")
    else:
         logger.info(f"Moderate market model fit: {fit_quality_msg}")
    
    logger.info(f"Market model estimated with {int(n_obs)} points: alpha={alpha:.6f}, beta={beta:.6f}")
    
    return {
        'alpha': alpha, 'beta': beta, 'r_squared': r_squared,
        'adj_r_squared': adj_r_squared, 'rmse': rmse, 'mae': mae, 'data_points': int(n_obs)
    }

def calculate_abnormal_returns(stock_data: pd.DataFrame, 
                               market_data: pd.DataFrame,
                               alpha: float, 
                               beta: float) -> pd.DataFrame:
    """
    Abnormale Renditen basierend auf dem Marktmodell berechnen.
    
    Args:
        stock_data: DataFrame mit Aktienkurs-Daten (muss 'close' Spalte haben)
        market_data: DataFrame mit Marktindex-Daten (muss 'close' Spalte haben)
        alpha: Marktmodell-Parameter Alpha
        beta: Marktmodell-Parameter Beta
        
    Returns:
        DataFrame mit abnormalen Renditen
    """
    # Ensure both dataframes have the same index
    common_idx = stock_data.index.intersection(market_data.index)
    
    stock_subset = stock_data.loc[common_idx].copy()
    market_subset = market_data.loc[common_idx].copy()
    
    # Calculate returns
    stock_subset['return'] = np.log(stock_subset['close'] / stock_subset['close'].shift(1))
    market_subset['return'] = np.log(market_subset['close'] / market_subset['close'].shift(1))
    
    # Calculate expected returns based on market model
    stock_subset['expected_return'] = alpha + beta * market_subset['return']
    
    # Calculate abnormal returns
    stock_subset['abnormal_return'] = stock_subset['return'] - stock_subset['expected_return']
    
    return stock_subset

def get_abnormal_returns_around_news(
    ticker: str,
    news_timestamp: datetime,
    market_index: str = "SPY",
    estimation_window: int = 30,
    min_data_points: int = 200,
    event_window_pre_minutes: int = 5,
    event_window_post_minutes: int = 60,
    timespan: str = "minute"
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Abnormale Renditen rund um ein News-Event mit Minutendaten berechnen.
    
    Args:
        ticker: Aktien-Ticker
        news_timestamp: Zeitstempel des News-Events
        market_index: Marktindex-Ticker (Standard: SPY für S&P 500 ETF)
        estimation_window: Ziel-Anzahl Handelstage im Schätzfenster
        min_data_points: Minimale Datenpunkte für robuste Schätzung
        event_window_pre_minutes: Minuten vor dem Event zur Berechnung
        event_window_post_minutes: Minuten nach dem Event zur Berechnung
        timespan: Datenauflösung (sollte immer "minute" sein)
        
    Returns:
        Tuple mit DataFrame der abnormalen Renditen im Event-Fenster und Modellmetriken
    """
    default_metrics = {
        'alpha': np.nan, 'beta': np.nan, 'r_squared': 0.0, 'adj_r_squared': 0.0,
        'rmse': np.nan, 'mae': np.nan, 'data_points': 0
    }
    # For minute data, we need a longer lookback to get sufficient data points
    # Average trading day has ~390 minutes, we want at least estimation_window trading days
    # Add 50% buffer for weekends, holidays, and gaps
    days_needed = int(estimation_window * 1.5)
    estimation_start = news_timestamp - timedelta(days=days_needed)
    logger.debug(f"Using {days_needed} calendar days for estimation to cover ~{estimation_window} trading days of minute data")
    
    event_window_start = news_timestamp - timedelta(minutes=event_window_pre_minutes)
    event_window_end = news_timestamp + timedelta(minutes=event_window_post_minutes)
    
    # Convert datetime objects to pandas Timestamps for proper comparison
    event_window_start_ts = pd.Timestamp(event_window_start)
    event_window_end_ts = pd.Timestamp(event_window_end)
    estimation_start_ts = pd.Timestamp(estimation_start)
    
    # Convert timezone-aware timestamps to naive Eastern Time to match stock data
    # Stock data from Polygon is already in Eastern Time but stored as naive
    if event_window_start_ts.tz is not None:
        event_window_start_ts = event_window_start_ts.tz_convert('US/Eastern').tz_localize(None)
    if event_window_end_ts.tz is not None:
        event_window_end_ts = event_window_end_ts.tz_convert('US/Eastern').tz_localize(None)
    if estimation_start_ts.tz is not None:
        estimation_start_ts = estimation_start_ts.tz_convert('US/Eastern').tz_localize(None)
    
    # Fetch stock data for estimation and event windows using cache
    stock_data = get_cached_stock_data(
        ticker=ticker,
        start_date=estimation_start_ts,
        end_date=event_window_end_ts,
        timespan=timespan
    )
    
    # Fetch market data for the same period using cache
    market_data = get_cached_market_data(
        market_index=market_index,
        start_date=estimation_start_ts,
        end_date=event_window_end_ts,
        timespan=timespan
    )
    
    if stock_data.empty:
        logger.error(f"Failed to fetch stock data for {ticker}")
        return pd.DataFrame(columns=['abnormal_return', 'cumulative_abnormal_return']), default_metrics.copy()
        
    if market_data.empty:
        logger.error(f"Failed to fetch market data for {market_index}")
        return pd.DataFrame(columns=['abnormal_return', 'cumulative_abnormal_return']), default_metrics.copy()
    
    # Check if we have data around the news event
    event_data_stock = stock_data[(stock_data.index >= event_window_start_ts) & 
                                 (stock_data.index <= event_window_end_ts)].copy()
    
    if event_data_stock.empty:
        logger.warning(f"No stock data available for {ticker} around news at {news_timestamp}")
        # Return an empty DataFrame and default metrics if stock data is missing
        return pd.DataFrame(columns=['abnormal_return', 'cumulative_abnormal_return']), default_metrics.copy()
    
    # Split data into estimation window and event window
    estimation_data_stock = stock_data[stock_data.index < event_window_start_ts].copy()
    estimation_data_market = market_data[market_data.index < event_window_start_ts].copy()
    
    event_data_market = market_data[(market_data.index >= event_window_start_ts) & 
                                   (market_data.index <= event_window_end_ts)].copy()
    
    # Log data point counts for diagnosis
    minutes_in_estimation = len(estimation_data_stock)
    trading_days_equivalent = minutes_in_estimation / 390 if minutes_in_estimation > 0 else 0
    logger.debug(f"Estimation data points - Stock: {minutes_in_estimation} minutes (~{trading_days_equivalent:.1f} trading days)")
    logger.debug(f"Event window data points - Stock: {len(event_data_stock)} minutes, Market: {len(event_data_market)} minutes")
    
    if estimation_data_stock.empty or estimation_data_market.empty:
        logger.warning(f"Insufficient estimation data for {ticker} prior to news at {news_timestamp}")
        # Use default market model parameters as fallback
        model_metrics = default_metrics.copy()
        alpha, beta = np.nan, np.nan
    else:
        # Estimate market model parameters with improved estimation approach
        try:
            model_metrics = estimate_market_model(
                estimation_data_stock, 
                estimation_data_market,
                estimation_window=estimation_window,
                min_data_points=min_data_points
            )
            alpha = model_metrics['alpha']
            beta = model_metrics['beta']
        except Exception as e:
            logger.error(f"Error estimating market model for {ticker}: {e}")
            # Use default values on error
            model_metrics = default_metrics.copy()
            alpha, beta = np.nan, np.nan
    
    # Check if we have market data for the event window
    if event_data_market.empty:
        logger.warning(f"No market data available for {market_index} around news at {news_timestamp}")
        # Create a basic DataFrame with abnormal returns as raw returns
        event_data_with_ar = event_data_stock[['close']].copy()
        event_data_with_ar.rename(columns={'close': 'stock_return'}, inplace=True)
        event_data_with_ar['stock_return'] = event_data_with_ar['stock_return'].pct_change()
        event_data_with_ar['abnormal_return'] = event_data_with_ar['stock_return'] # No market data, AR = raw return
        # Fill NA for the first row that has no previous data for pct_change()
        event_data_with_ar['abnormal_return'].fillna(0, inplace=True)
        event_data_with_ar['cumulative_abnormal_return'] = event_data_with_ar['abnormal_return'].cumsum()
        return event_data_with_ar[['abnormal_return', 'cumulative_abnormal_return']], default_metrics.copy()
    elif np.isnan(alpha) or np.isnan(beta):
        logger.warning(f"Market model estimation failed for {ticker} around news at {news_timestamp}. Excluding event from analysis to maintain methodological integrity.")
        # When market model fails, exclude the event entirely as described in methodology
        return pd.DataFrame(), default_metrics.copy()
    else:
        # Calculate abnormal returns for event window
        try:
            event_data_with_ar = calculate_abnormal_returns(event_data_stock, event_data_market, alpha, beta)
            # Calculate cumulative abnormal returns
            event_data_with_ar['cumulative_abnormal_return'] = event_data_with_ar['abnormal_return'].cumsum()
        except Exception as e:
            logger.error(f"Error calculating abnormal returns for {ticker}: {e}")
            # Return empty DataFrame and default model metrics
            return pd.DataFrame(), default_metrics
    
    # Add minutes since news - ensure timezone compatibility
    try:
        # Convert news_timestamp to pandas Timestamp if it's not already
        news_ts = pd.Timestamp(news_timestamp)
        # Make sure it's timezone-aware if the index is timezone-aware
        if hasattr(event_data_with_ar.index, 'tz') and event_data_with_ar.index.tz is not None:
            if news_ts.tz is None:
                # If news_ts is naive but index is timezone-aware, localize it
                news_ts = news_ts.tz_localize(event_data_with_ar.index.tz)
            elif news_ts.tz != event_data_with_ar.index.tz:
                # If different timezones, convert to match index timezone
                news_ts = news_ts.tz_convert(event_data_with_ar.index.tz)
        elif news_ts.tz is not None and (not hasattr(event_data_with_ar.index, 'tz') or event_data_with_ar.index.tz is None):
            # If news_ts has timezone but index doesn't, remove timezone
            news_ts = news_ts.tz_localize(None)
            
        event_data_with_ar['minutes_from_news'] = (event_data_with_ar.index - news_ts).total_seconds() / 60
    except Exception as e:
        logger.error(f"Error calculating minutes from news: {e}")
        # Fallback: use position-based indexing
        event_data_with_ar['minutes_from_news'] = range(len(event_data_with_ar))
    
    return event_data_with_ar, model_metrics

def calculate_ar_for_news_events(
    ticker: str,
    sentiment_data: List[Dict],
    market_index: str = "SPY",
    timespan: str = "minute",
    event_windows: List[int] = [5, 15, 30, 60],
    estimation_window: int = 30,
    min_data_points: int = 200,
    use_cache: bool = True
) -> Dict[str, Dict]:
    """
    Calculate abnormal returns for multiple news events with different sentiment categories.
    Always uses minute-level data.
    
    Args:
        ticker: Stock ticker symbol
        sentiment_data: List of dictionaries containing news sentiment data
        market_index: Market index ticker symbol
        timespan: Data granularity (should always be "minute")
        event_windows: List of time windows (in minutes) to calculate ARs for
        estimation_window: Number of days for market model estimation
        min_data_points: Minimum number of data points for robust estimation
        use_cache: Whether to use data caching
        
    Returns:
        Dictionary with abnormal returns grouped by sentiment and time windows
    """
    # Initialize result dictionary
    results = {
        'positive': {window: [] for window in event_windows},
        'negative': {window: [] for window in event_windows},
        'neutral': {window: [] for window in event_windows},
        'all_events': []
    }
    
    # Initialize the data cache if using it
    if use_cache and sentiment_data:
        # Find earliest and latest news timestamps
        all_dates = []
        for news in sentiment_data:
            try:
                # Extract news information with proper timezone handling
                # News timestamps from sentiment analysis are already in Eastern Time (naive format)
                # They should NOT be treated as UTC and converted again
                raw_news_time_str = news['date']
                
                # Parse the datetime
                if 'Z' in raw_news_time_str:
                    # ISO format with UTC indicator
                    raw_news_time = datetime.fromisoformat(raw_news_time_str.replace('Z', '+00:00'))
                    # Convert UTC to Eastern Time
                    news_time = convert_to_eastern_time(raw_news_time)
                elif '+' in raw_news_time_str or '-' in raw_news_time_str[-6:]:
                    # Already has timezone info
                    raw_news_time = datetime.fromisoformat(raw_news_time_str)
                    # Convert to Eastern Time
                    news_time = convert_to_eastern_time(raw_news_time)
                else:
                    # Naive datetime - these are already in Eastern Time from sentiment analysis
                    # Do NOT treat as UTC and convert again!
                    raw_news_time = datetime.strptime(raw_news_time_str, '%Y-%m-%d %H:%M:%S')
                    # Use directly as Eastern Time (naive)
                    news_time = raw_news_time
                
                # Convert to pandas Timestamp for compatibility
                news_time = pd.Timestamp(news_time)
                all_dates.append(news_time)
            except Exception as e:
                logger.warning(f"Error parsing date from news item '{news['date']}': {e}")
                continue
        
        if all_dates:
            # Calculate buffer for minute data
            # Add 50% buffer for weekends, holidays, and gaps
            buffer_days = int(estimation_window * 1.5)
                
            earliest_news = min(all_dates) - timedelta(days=buffer_days)
            latest_news = max(all_dates) + timedelta(minutes=max(event_windows) + 10)
            
            logger.info(f"Initializing market data cache for period: {earliest_news} to {latest_news} ({(latest_news-earliest_news).days} days)")
            
            # Initialize cache for the entire period
            initialize_market_data_cache(
                ticker=ticker,
                market_index=market_index,
                start_date=earliest_news,
                end_date=latest_news,
                timespan=timespan
            )
    
    # Keep track of processed events
    successful_events = 0
    empty_data_events = 0
    error_events = 0
    
    # Store model metrics per event
    model_metrics_results = {
        'positive': [], 'negative': [], 'neutral': []
    }
    
    for news in sentiment_data:
        try:
            # Debug logging to see actual sentiment data structure
            logger.debug(f"Processing news item: {news}")
            
            # Extract news information with proper timezone handling
            # News timestamps from sentiment analysis are already in Eastern Time (naive format)
            # They should NOT be treated as UTC and converted again
            raw_news_time_str = news['date']
            
            # Parse the datetime
            if 'Z' in raw_news_time_str:
                # ISO format with UTC indicator
                raw_news_time = datetime.fromisoformat(raw_news_time_str.replace('Z', '+00:00'))
                # Convert UTC to Eastern Time
                news_time = convert_to_eastern_time(raw_news_time)
            elif '+' in raw_news_time_str or '-' in raw_news_time_str[-6:]:
                # Already has timezone info
                raw_news_time = datetime.fromisoformat(raw_news_time_str)
                # Convert to Eastern Time
                news_time = convert_to_eastern_time(raw_news_time)
            else:
                # Naive datetime - these are already in Eastern Time from sentiment analysis
                # Do NOT treat as UTC and convert again!
                raw_news_time = datetime.strptime(raw_news_time_str, '%Y-%m-%d %H:%M:%S')
                # Use directly as Eastern Time (naive)
                news_time = raw_news_time
            
            # Convert to pandas Timestamp for compatibility
            news_time = pd.Timestamp(news_time)
            sentiment = news['sentiment'].lower()
            
            # Skip if sentiment not in our categories
            if sentiment not in ['positive', 'negative', 'neutral']:
                logger.warning(f"Unknown sentiment category: {sentiment}")
                continue
                
            # Get abnormal returns around this news event
            event_ar_data, model_metrics_for_event = get_abnormal_returns_around_news(
                ticker=ticker,
                news_timestamp=news_time,
                market_index=market_index,
                estimation_window=estimation_window,
                min_data_points=min_data_points,
                event_window_post_minutes=max(event_windows),
                timespan=timespan
            )
            
            if event_ar_data.empty:
                empty_data_events += 1
                if empty_data_events % 100 == 0:  # Log only every 100th empty data event to avoid spam
                    logger.warning(f"No data for news at {news_time} (total empty data events: {empty_data_events})")
                continue
            
            # Store the full AR data for this event
            event_result = {
                'timestamp': news_time,
                'sentiment': sentiment,
                'title': news.get('title', ''),
                'ar_data': event_ar_data
            }
            results['all_events'].append(event_result)
            
            # Extract ARs at specific time points and add to respective lists
            for window in event_windows:
                # Find the closest data point to our target window
                if 'minutes_from_news' not in event_ar_data.columns or event_ar_data['minutes_from_news'].empty:
                    continue
                    
                try:
                    closest_idx = event_ar_data['minutes_from_news'].sub(window).abs().idxmin()
                    # Extract abnormal return for this specific window (minute)
                    ar_at_window = event_ar_data.loc[closest_idx, 'abnormal_return']
                    
                    # Skip NaN values
                    if pd.isna(ar_at_window):
                        continue
                    
                    # Add to the appropriate sentiment category
                    results[sentiment][window].append(ar_at_window) # Now appending individual AR values
                except (KeyError, ValueError) as e:
                    logger.debug(f"Error finding closest data point for window {window}: {e}")
                    continue
            
            # Store the model metrics for this event
            model_metrics_results[sentiment].append(model_metrics_for_event)
            
            successful_events += 1
            
        except Exception as e:
            error_events += 1
            logger.error(f"Error processing news event: {e}")
            continue
    
    # Log summary of event processing
    logger.info(f"Processed {successful_events} events successfully, {empty_data_events} had no data, {error_events} had errors")
    for sentiment in ['positive', 'negative', 'neutral']:
        for window in event_windows:
            valid_count = sum(1 for ar in results[sentiment][window] if ar is not None and not pd.isna(ar))
            logger.info(f"  {sentiment} events with valid AR at {window} minutes: {valid_count}/{len(results[sentiment][window])}")
    
    return results, model_metrics_results

def calculate_aar_and_caar(ar_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Calculate Average Abnormal Returns (AAR) and Cumulative Average Abnormal Returns (CAAR)
    
    AAR is the average abnormal return across all events at a specific time window.
    CAAR is the cumulative sum of AARs from the first window up to the current window.
    
    Args:
        ar_results: Dictionary with abnormal returns by sentiment and time window
        
    Returns:
        Dictionary with AAR and CAAR statistics
    """
    results = {}
    
    # Process each sentiment category
    for sentiment in ['positive', 'negative', 'neutral']:
        results[sentiment] = {
            'count': {},
            'AAR': {},
            'AAR_std': {},
            'CAAR': {},
            't_stat': {},
            'p_value': {}
        }
        
        # Calculate AAR for each window first
        aar_values = {}
        for window in sorted(ar_results[sentiment].keys(), key=lambda x: int(x)):
            # Get ARs for this sentiment and time window
            ars = ar_results[sentiment][window]
            
            # Calculate statistics only if we have data
            if ars and len(ars) > 0:
                n = len(ars)
                # Filter out NaN values
                valid_ars = [ar for ar in ars if ar is not None and not pd.isna(ar)]
                
                if valid_ars and len(valid_ars) > 0:
                    aar = np.mean(valid_ars)
                    aar_std = np.std(valid_ars)
                    
                    # Calculate t-statistic and p-value for significance testing
                    if len(valid_ars) > 1 and aar_std > 1e-10:
                        t_stat = aar / (aar_std / np.sqrt(len(valid_ars)))
                        from scipy import stats
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(valid_ars)-1))
                    else:
                        t_stat = 0.0
                        p_value = 1.0
                    
                    # Store AAR results
                    results[sentiment]['count'][window] = n
                    results[sentiment]['AAR'][window] = float(aar)
                    results[sentiment]['AAR_std'][window] = float(aar_std)
                    results[sentiment]['t_stat'][window] = float(t_stat)
                    results[sentiment]['p_value'][window] = float(p_value)
                    aar_values[window] = aar
                else:
                    # Handle case with no valid data
                    results[sentiment]['count'][window] = n
                    results[sentiment]['AAR'][window] = 0.0
                    results[sentiment]['AAR_std'][window] = 0.0
                    results[sentiment]['t_stat'][window] = 0.0
                    results[sentiment]['p_value'][window] = 1.0
                    aar_values[window] = 0.0
            else:
                # Handle case with no data
                results[sentiment]['count'][window] = 0
                results[sentiment]['AAR'][window] = 0.0
                results[sentiment]['AAR_std'][window] = 0.0
                results[sentiment]['t_stat'][window] = 0.0
                results[sentiment]['p_value'][window] = 1.0
                aar_values[window] = 0.0
        
        # Now calculate CAAR as cumulative sum of AARs
        cumulative_aar = 0.0
        for window in sorted(aar_values.keys(), key=lambda x: int(x)):
            cumulative_aar += aar_values[window]
            results[sentiment]['CAAR'][window] = float(cumulative_aar)
    
    return results

def plot_aar_and_caar(results: Dict[str, Dict], filename: Optional[str] = None):
    """
    Plot Average Abnormal Returns (AAR) and Cumulative Average Abnormal Returns (CAAR)
    
    Args:
        results: Dictionary with AAR and CAAR statistics
        filename: Optional filename to save the plot
    """
    # Check if we have any valid data to plot
    has_valid_data = False
    for sentiment in ['positive', 'negative', 'neutral']:
        if not results.get(sentiment):
            continue
            
        for key in ['AAR', 'CAAR']:
            if key not in results[sentiment]:
                continue
                
            values = results[sentiment][key].values()
            if values and any(v != 0.0 for v in values if v is not None):
                has_valid_data = True
                break
                
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    if not has_valid_data:
        # If no valid data, display a message on the plot
        ax1.text(0.5, 0.5, "No valid abnormal returns data available",
                ha='center', va='center', fontsize=14)
        ax2.text(0.5, 0.5, "No valid cumulative abnormal returns data available",
                ha='center', va='center', fontsize=14)
        
        # Set limits for empty plots
        ax1.set_xlim(-0.05, 0.05)
        ax1.set_ylim(-0.05, 0.05)
        ax2.set_xlim(-0.05, 0.05)
        ax2.set_ylim(-0.05, 0.05)
    else:
        # Plot AAR
        for sentiment, color in zip(['positive', 'negative', 'neutral'], ['green', 'red', 'blue']):
            if sentiment not in results:
                continue
                
            if 'AAR' not in results[sentiment]:
                continue
                
            windows = sorted(results[sentiment]['AAR'].keys(), key=lambda x: int(x))
            values = [results[sentiment]['AAR'][w] for w in windows]
            
            # Filter out None or 0.0 values (which we use as placeholders for no data)
            valid_points = [(w, v) for w, v in zip(windows, values) if v is not None and v != 0.0]
            if valid_points:
                x, y = zip(*valid_points)
                ax1.plot(x, y, marker='o', label=f"{sentiment.capitalize()}", color=color, linewidth=2)
        
        # Plot CAAR
        for sentiment, color in zip(['positive', 'negative', 'neutral'], ['green', 'red', 'blue']):
            if sentiment not in results:
                continue
                
            if 'CAAR' not in results[sentiment]:
                continue
                
            windows = sorted(results[sentiment]['CAAR'].keys(), key=lambda x: int(x))
            values = [results[sentiment]['CAAR'][w] for w in windows]
            
            # Filter out None or 0.0 values (which we use as placeholders for no data)
            valid_points = [(w, v) for w, v in zip(windows, values) if v is not None and v != 0.0]
            if valid_points:
                x, y = zip(*valid_points)
                ax2.plot(x, y, marker='o', label=f"{sentiment.capitalize()}", color=color, linewidth=2)
    
    # Set titles and labels
    ax1.set_title('Average Abnormal Returns (AAR) After News Events', fontsize=14)
    ax1.set_xlabel('Minutes After News', fontsize=12)
    ax1.set_ylabel('AAR (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.legend()
    
    ax2.set_title('Cumulative Average Abnormal Returns (CAAR) After News Events', fontsize=14)
    ax2.set_xlabel('Minutes After News', fontsize=12)
    ax2.set_ylabel('CAAR (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.close(fig)  # Close the figure to prevent showing it in notebooks

def run_event_study(ticker: str, sentiment_data: List[Dict], 
                   market_index: str = "SPY",
                   event_windows: List[int] = [5, 15, 30, 60],
                   estimation_window: int = 30,
                   min_data_points: int = 200,
                   output_dir: Optional[str] = None,
                   use_cache: bool = True,
                   timespan: str = "minute") -> Dict:
    """
    Run a complete event study analysis for news with sentiment classification.
    
    Args:
        ticker: Stock ticker symbol
        sentiment_data: List of dictionaries containing news sentiment data
        market_index: Market index ticker symbol
        event_windows: List of time windows (in minutes) to analyze
        estimation_window: Number of days for estimation window
        min_data_points: Minimum number of data points for robust market model estimation
        output_dir: Optional directory to save results and plots
        use_cache: Whether to use data caching
        timespan: Data granularity (must be "minute")
        
    Returns:
        Dictionary with complete event study results
    """
    # Ensure we're using minute data
    if timespan != "minute":
        logger.warning(f"Timespan '{timespan}' not supported. Forcing 'minute' timespan for event study analysis.")
        timespan = "minute"
        
    logger.info(f"Running event study for {ticker} with {len(sentiment_data)} news events using {market_index} as market index")
    logger.info(f"Using estimation window of {estimation_window} days with minimum {min_data_points} data points")
    
    # Calculate abnormal returns for all events AND get model metrics per event
    ar_results, model_metrics_per_event = calculate_ar_for_news_events(
        ticker=ticker,
        sentiment_data=sentiment_data,
        market_index=market_index,
        event_windows=event_windows,
        estimation_window=estimation_window,
        min_data_points=min_data_points,
        use_cache=use_cache,
        timespan=timespan
    )
    
    # Check if we have any valid results
    has_valid_data = False
    for sentiment in ['positive', 'negative', 'neutral']:
        for window in event_windows:
            if ar_results[sentiment][window] and len(ar_results[sentiment][window]) > 0:
                if any(ar is not None and not pd.isna(ar) for ar in ar_results[sentiment][window]):
                    has_valid_data = True
                    break
    
    # Log warning if no valid data
    if not has_valid_data:
        logger.warning(f"No valid abnormal returns data found for {ticker}. This may be due to missing market data or insufficient price data around news events.")
    
    # Calculate AAR and CAAR
    aar_caar_results = calculate_aar_and_caar(ar_results)
    
    # Aggregate Model Quality Metrics
    aggregated_metrics = {'positive': {}, 'negative': {}, 'neutral': {}, 'overall': {}}
    all_metrics_list = []

    for sentiment in ['positive', 'negative', 'neutral']:
        metrics_list = model_metrics_per_event[sentiment]
        all_metrics_list.extend(metrics_list) # Collect all for overall average

        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list).replace([np.inf, -np.inf], np.nan)
            # Remove rows where any key metric is NaN for consistent aggregation
            key_metrics = ['r_squared', 'adj_r_squared', 'rmse', 'mae', 'data_points']
            metrics_df = metrics_df.dropna(subset=key_metrics)
            
            if not metrics_df.empty:
                 aggregated_metrics[sentiment] = {
                     'avg_r_squared': metrics_df['r_squared'].mean(),
                     'avg_adj_r_squared': metrics_df['adj_r_squared'].mean(),
                     'avg_rmse': metrics_df['rmse'].mean(),
                     'avg_mae': metrics_df['mae'].mean(),
                     'avg_data_points': metrics_df['data_points'].mean(),
                     'count': len(metrics_df) # Count of successful estimations
                 }
            else:
                 # Use consistent default values (NaN instead of mix of NaN and 0)
                 aggregated_metrics[sentiment] = {
                     'avg_r_squared': np.nan,
                     'avg_adj_r_squared': np.nan,
                     'avg_rmse': np.nan,
                     'avg_mae': np.nan,
                     'avg_data_points': np.nan,
                     'count': 0
                 }
        else:
             # Use consistent default values
             aggregated_metrics[sentiment] = {
                 'avg_r_squared': np.nan,
                 'avg_adj_r_squared': np.nan,
                 'avg_rmse': np.nan,
                 'avg_mae': np.nan,
                 'avg_data_points': np.nan,
                 'count': 0
             }

    # Calculate overall average metrics
    if all_metrics_list:
         overall_metrics_df = pd.DataFrame(all_metrics_list).replace([np.inf, -np.inf], np.nan)
         key_metrics = ['r_squared', 'adj_r_squared', 'rmse', 'mae', 'data_points']
         overall_metrics_df = overall_metrics_df.dropna(subset=key_metrics)
         
         if not overall_metrics_df.empty:
             aggregated_metrics['overall'] = {
                 'avg_r_squared': overall_metrics_df['r_squared'].mean(),
                 'avg_adj_r_squared': overall_metrics_df['adj_r_squared'].mean(),
                 'avg_rmse': overall_metrics_df['rmse'].mean(),
                 'avg_mae': overall_metrics_df['mae'].mean(),
                 'avg_data_points': overall_metrics_df['data_points'].mean(),
                 'count': len(overall_metrics_df) # Total successful estimations
             }
         else:
             aggregated_metrics['overall'] = {
                 'avg_r_squared': np.nan,
                 'avg_adj_r_squared': np.nan,
                 'avg_rmse': np.nan,
                 'avg_mae': np.nan,
                 'avg_data_points': np.nan,
                 'count': 0
             }
    else:
         aggregated_metrics['overall'] = {
             'avg_r_squared': np.nan,
             'avg_adj_r_squared': np.nan,
             'avg_rmse': np.nan,
             'avg_mae': np.nan,
             'avg_data_points': np.nan,
             'count': 0
         }

    # Log aggregated metrics
    logger.info("Aggregated Market Model Quality Metrics:")
    for key, metrics in aggregated_metrics.items():
         if metrics.get('count', 0) > 0:
             logger.info(f"  {key.capitalize()} ({metrics['count']} events): "
                         f"Avg R²={metrics.get('avg_r_squared', np.nan):.4f}, "
                         f"Avg Adj R²={metrics.get('avg_adj_r_squared', np.nan):.4f}, "
                         f"Avg RMSE={metrics.get('avg_rmse', np.nan):.6f}, "
                         f"Avg MAE={metrics.get('avg_mae', np.nan):.6f}, "
                         f"Avg Obs={metrics.get('avg_data_points', np.nan):.0f}")
         else:
              logger.info(f"  {key.capitalize()}: No successful model estimations.")


    # Create summary statistics including the new metrics
    summary = {
        'ticker': ticker,
        'market_index': market_index,
        'unique_news_events': len(sentiment_data),  # Changed from 'total_events' to be more precise
        'events_by_sentiment': {
            'positive': len(ar_results['positive'][event_windows[0]]),
            'negative': len(ar_results['negative'][event_windows[0]]),
            'neutral': len(ar_results['neutral'][event_windows[0]])
        },
        'event_windows': event_windows,
        'estimation_window': estimation_window,
        'min_data_points': min_data_points,
        'aar_caar': aar_caar_results,
        'market_model_stats': aggregated_metrics # Add aggregated metrics here
    }
    
    # Generate plot if output directory is specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f"{ticker}_event_study.png")
        plot_aar_and_caar(aar_caar_results, filename=plot_filename)
        
        # Save results as JSON
        import json
        # Convert to JSON-serializable structure
        json_serializable = {
            'ticker': summary['ticker'],
            'market_index': summary['market_index'],
            'unique_news_events': summary['unique_news_events'],  # Updated field name
            'events_by_sentiment': summary['events_by_sentiment'],
            'event_windows': summary['event_windows'],
            'estimation_window': summary['estimation_window'],
            'min_data_points': summary['min_data_points'],
            'aar_caar': {},
            'market_model_stats': {},
            'individual_event_caars_by_sentiment_window': {} # New field for raw CAAR lists
        }
        
        # Handle the AAR and CAAR results (existing code)
        for sentiment_key in aar_caar_results:
            json_serializable['aar_caar'][sentiment_key] = {}
            for stat_key, value_dict in aar_caar_results[sentiment_key].items():
                if isinstance(value_dict, dict):
                    json_serializable['aar_caar'][sentiment_key][stat_key] = {
                        str(k_win): float(v_val) if pd.notna(v_val) else None
                        for k_win, v_val in value_dict.items()
                    }
                else:
                    json_serializable['aar_caar'][sentiment_key][stat_key] = float(value_dict) if pd.notna(value_dict) else None

        # Populate market_model_stats (existing code)
        for main_key in aggregated_metrics:
            json_serializable['market_model_stats'][main_key] = {
                str(k_metric): (float(v_metric) if pd.notna(v_metric) else None) 
                for k_metric, v_metric in aggregated_metrics[main_key].items()
            }

        # Populate individual_event_caars_by_sentiment_window (NEW)
        # ar_results format is: {'positive': {5: [caar1, caar2], 15: [caar1, caar2], ...}, ...}
        for sentiment_cat in ar_results: # 'positive', 'negative', 'neutral', 'all_events'
            if sentiment_cat == 'all_events': # Skip the 'all_events' detailed list for this summary
                continue
            json_serializable['individual_event_caars_by_sentiment_window'][sentiment_cat] = {}
            for window_val in ar_results[sentiment_cat]:
                # Ensure all items in the list are float or None
                caar_list = [float(caar) if pd.notna(caar) else None for caar in ar_results[sentiment_cat][window_val]]
                json_serializable['individual_event_caars_by_sentiment_window'][sentiment_cat][str(window_val)] = caar_list
        
        # Save to file with safety checks
        try:
            results_filename = os.path.join(output_dir, f"{ticker}_event_study_results.json")
            with open(results_filename, 'w') as f:
                json.dump(json_serializable, f, indent=4)
            logger.info(f"Event study results saved to {results_filename}")
        except Exception as e:
            logger.error(f"Error saving event study results to JSON: {e}")
    
    logger.info(f"Event study completed for {ticker}")
    return summary 