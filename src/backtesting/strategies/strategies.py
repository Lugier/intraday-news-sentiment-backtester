"""
Backtesting strategies for trading based on news sentiment.

This module defines the logic for backtesting trading strategies using news sentiment
and historical market data.
"""

import os
import pandas as pd
import numpy as np
import logging
import concurrent.futures
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.backtesting.helpers.market_data import fetch_stock_prices, is_market_open, get_minute_data_for_news_event, convert_to_eastern_time
from src.config import BACKTEST_CONFIG, TRADING_CONSTANTS
from src.news.fetcher.news_fetcher import filter_tracker

logger = logging.getLogger(__name__)

def classify_market_session(dt: datetime) -> str:
    """
    Classify a datetime into market session using proper Eastern Time.
    
    Args:
        dt: Datetime to classify (any timezone or naive)
        
    Returns:
        Market session classification
    """
    # Convert to Eastern Time for US markets
    et_dt = convert_to_eastern_time(dt)
    hour = et_dt.hour
    minute = et_dt.minute
    
    if hour < 9 or (hour == 9 and minute < 30):
        return "Pre-Market"
    elif hour < 12:
        return "Morning"
    elif hour < 14:
        return "Midday"
    elif hour < 16:
        return "Afternoon"
    else:
        return "After-Hours"

def calculate_drawdown(returns: np.ndarray) -> Tuple[float, float]:
    """
    Calculate maximum drawdown from a series of returns.
    
    Args:
        returns: Array of percentage returns
        
    Returns:
        Tuple of (maximum drawdown percentage, drawdown duration in trades)
    """
    # Convert returns to cumulative returns
    cum_returns = (1 + returns/100).cumprod() - 1
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / (1 + running_max)
    
    # Find maximum drawdown
    max_drawdown = drawdown.min() * 100
    
    # Find drawdown duration (in number of trades)
    # This is simplified; a more sophisticated approach would track exact periods
    max_dd_idx = np.argmin(drawdown)
    
    # Find when the drawdown started (last time we were at the previous peak)
    if max_dd_idx > 0:
        peak_idx = np.where(cum_returns[:max_dd_idx] == running_max[max_dd_idx-1])[0]
        if len(peak_idx) > 0:
            last_peak = peak_idx[-1]
            duration = max_dd_idx - last_peak
        else:
            duration = 0
    else:
        duration = 0
        
    return max_drawdown, duration

def calculate_trade_statistics(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate various trading statistics.
    
    Args:
        returns: Array of percentage returns
        
    Returns:
        Dictionary of statistics
    """
    if len(returns) == 0:
        return {
            'total_return': 0,
            'avg_return': 0,
            'median_return': 0,
            'std_dev': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'drawdown_duration': 0,
            'profit_factor': 0,
            'expected_return': 0
        }
    
    # Basic statistics
    total_return = np.sum(returns)
    avg_return = np.mean(returns)
    median_return = np.median(returns)
    std_dev = np.std(returns)
    
    # Sharpe ratio (simplified, assuming risk-free rate = 0)
    sharpe = 0 if std_dev == 0 else (avg_return / std_dev) * np.sqrt(TRADING_CONSTANTS["TRADING_DAYS_PER_YEAR"])  # Annualized
    
    # Win rate
    win_rate = np.mean(returns > 0)
    
    # Maximum drawdown
    max_drawdown, dd_duration = calculate_drawdown(returns)
    
    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = 0 if losses == 0 else gains / losses
    
    # Expected return (average win * win rate - average loss * lose rate)
    avg_win = np.mean(returns[returns > 0]) if any(returns > 0) else 0
    avg_loss = np.mean(returns[returns < 0]) if any(returns < 0) else 0
    expected_return = (avg_win * win_rate) + (avg_loss * (1 - win_rate))
    
    return {
        'total_return': total_return,
        'avg_return': avg_return,
        'median_return': median_return,
        'std_dev': std_dev,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'drawdown_duration': dd_duration,
        'profit_factor': profit_factor,
        'expected_return': expected_return
    }

def process_trade_with_stop_loss(
    ticker: str,
    news: Dict[str, Any],
    entry_delay_minutes: int,
    max_holding_period_minutes: int,
    stop_loss_pct: float = BACKTEST_CONFIG["default_stop_loss_pct"],
    take_profit_pct: float = BACKTEST_CONFIG["default_take_profit_pct"],
    transaction_fee_pct: float = BACKTEST_CONFIG["default_transaction_fee_pct"]
) -> Optional[Dict[str, Any]]:
    """
    Process a single trade with stop loss and take profit logic.
    
    Args:
        ticker: Stock ticker symbol
        news: News item with sentiment data
        entry_delay_minutes: Delay in minutes before entering position
        max_holding_period_minutes: Maximum minutes to hold the position
        stop_loss_pct: Stop loss percentage threshold (positive value, e.g. 2.0 means 2%)
        take_profit_pct: Take profit percentage threshold (positive value, e.g. 4.0 means 4%)
        transaction_fee_pct: Percentage of transaction volume to be charged as fee
        
    Returns:
        Dictionary with trade details or None if trade cannot be executed
    """
    try:
        # Parse news timestamp
        news_time = datetime.strptime(news['date'], '%Y-%m-%d %H:%M:%S')
        
        # Get market session classification
        market_session = classify_market_session(news_time)
        
        # Skip news on weekends
        if news_time.weekday() >= 5:  # 5=Saturday, 6=Sunday
            logger.info(f"Skipping weekend news: {news['date']}")
            filter_tracker.log_weekend_filter(1)
            return None
        
        # Get minute data around news event
        stock_data = get_minute_data_for_news_event(
            ticker, 
            news_time,
            minutes_before=5,
            minutes_after=max_holding_period_minutes + 5  # Add buffer
        )
        
        if stock_data.empty:
            logger.warning(f"No stock data available for news at {news['date']}")
            filter_tracker.log_missing_market_data_filter(1)
            return None
        
        # Calculate entry time
        entry_time = news_time + timedelta(minutes=entry_delay_minutes)
        
        # Ensure the calculated entry time is within regular market hours
        if not is_market_open(entry_time, extended=False):
            logger.info(f"Skipping trade for news at {news['date']} for {ticker}: Calculated entry time {entry_time} is outside regular market hours (9:30 AM - 4:00 PM ET).")
            filter_tracker.log_after_hours_filter(1) # Log this as filtered by after_hours
            return None
        
        # Find entry data point
        entry_data = stock_data[stock_data.index >= entry_time].iloc[0] if not stock_data[stock_data.index >= entry_time].empty else None
        
        if entry_data is None:
            logger.warning(f"Missing entry data for news at {news['date']}")
            filter_tracker.log_missing_market_data_filter(1)
            return None
        
        # Determine position direction based on sentiment
        direction = 1 if news['sentiment'] == 'positive' else -1 if news['sentiment'] == 'negative' else 0
        
        # Skip neutral sentiment
        if direction == 0:
            logger.info(f"Skipping neutral sentiment news at {news['date']}")
            filter_tracker.log_neutral_sentiment_filter(1)
            return None
        
        # Entry price
        entry_price = entry_data['open']
        
        # Calculate recent volatility to adjust stop loss and take profit
        try:
            # Get historical data for volatility calculation (past N days)
            vol_start = news_time - timedelta(days=TRADING_CONSTANTS["HISTORICAL_VOLATILITY_DAYS"])
            hist_data = get_minute_data_for_news_event(
                ticker, 
                news_time, 
                minutes_before=TRADING_CONSTANTS["HISTORICAL_VOLATILITY_DAYS"] * 24 * 60, 
                minutes_after=0
            )
            
            if not hist_data.empty and len(hist_data) > TRADING_CONSTANTS["MIN_VOLATILITY_DATA_POINTS"]:  # Need sufficient data for vol calculation
                # Calculate N-day standard deviation of returns
                hist_returns = hist_data['close'].pct_change().dropna()
                volatility = hist_returns.std() * 100  # Convert to percentage
                
                # Adjust stop loss and take profit based on volatility
                # More volatile stocks get wider stops
                vol_factor = min(
                    max(
                        volatility / TRADING_CONSTANTS["VOLATILITY_BASE_THRESHOLD"], 
                        TRADING_CONSTANTS["VOLATILITY_FACTOR_MIN"]
                    ), 
                    TRADING_CONSTANTS["VOLATILITY_FACTOR_MAX"]
                )  # Limit adjustment between min and max factor
                dynamic_stop_loss = stop_loss_pct * vol_factor
                dynamic_take_profit = take_profit_pct * vol_factor
                
                logger.info(f"Adjusted stop loss for {ticker} based on volatility: {dynamic_stop_loss:.2f}% (factor: {vol_factor:.2f})")
            else:
                # Use default values if insufficient data
                dynamic_stop_loss = stop_loss_pct
                dynamic_take_profit = take_profit_pct
        except Exception as e:
            logger.warning(f"Error calculating volatility adjustment: {e}. Using default stop loss/take profit.")
            dynamic_stop_loss = stop_loss_pct
            dynamic_take_profit = take_profit_pct
        
        # Maximum exit time
        max_exit_time = news_time + timedelta(minutes=max_holding_period_minutes)
        
        # Get price path after entry
        price_path = stock_data[
            (stock_data.index > entry_time) &
            (stock_data.index <= max_exit_time)
        ]
        
        if len(price_path) < 1:
            logger.warning(f"Insufficient price data for trade at {news['date']}")
            return None
            
        # Apply stop loss and take profit logic to determine exit point
        exit_data = None
        exit_reason = "max_holding_period"
        
        for i, (timestamp, row) in enumerate(price_path.iterrows()):
            current_price = row['close']
            
            # Calculate current return
            current_return_pct = direction * (current_price - entry_price) / entry_price * 100
            
            # Check stop loss condition - stop_loss_pct is positive, so check if return is LESS than negative stop_loss
            # Example: If stop_loss_pct is 2.0%, we exit when return <= -2.0%
            if current_return_pct <= -dynamic_stop_loss:
                exit_data = row
                exit_time = timestamp
                exit_reason = "stop_loss"
                logger.debug(f"Stop loss triggered at {timestamp}: return {current_return_pct:.2f}% <= threshold -{dynamic_stop_loss:.2f}%")
                break
            
            # Check take profit condition - take_profit_pct is positive, check if return is MORE than take_profit
            # Example: If take_profit_pct is 4.0%, we exit when return >= 4.0%
            if current_return_pct >= dynamic_take_profit:
                exit_data = row
                exit_time = timestamp
                exit_reason = "take_profit"
                logger.debug(f"Take profit triggered at {timestamp}: return {current_return_pct:.2f}% >= threshold {dynamic_take_profit:.2f}%")
                break
                
        # If no stop loss or take profit triggered, exit at max holding period
        if exit_data is None:
            if not price_path.empty:
                exit_data = price_path.iloc[-1]
                exit_time = exit_data.name
            else:
                logger.warning(f"No exit data available for trade at {news['date']}")
                return None
                
        # Calculate trade return
        exit_price = exit_data['close']
        trade_return_pct = direction * (exit_price - entry_price) / entry_price * 100
        
        # Calculate intra-trade metrics
        trade_price_path = stock_data[
            (stock_data.index >= entry_time) &
            (stock_data.index <= exit_time)
        ]['close'].values
        
        max_favorable_excursion = 0
        max_adverse_excursion = 0
        
        for price in trade_price_path:
            # Calculate return at this point
            current_return = direction * (price - entry_price) / entry_price * 100
            
            # Update MFE (maximum favorable excursion)
            if current_return > max_favorable_excursion:
                max_favorable_excursion = current_return
            
            # Update MAE (maximum adverse excursion)
            if current_return < max_adverse_excursion:
                max_adverse_excursion = current_return
                
        # Calculate P&L
        pnl = (exit_price - entry_price) * direction
        return_pct = (pnl / entry_price) * 100
        
        # Apply transaction fees
        entry_fee = entry_price * transaction_fee_pct
        exit_fee = exit_price * transaction_fee_pct
        total_fees = entry_fee + exit_fee
        
        pnl_after_fees = pnl - total_fees
        # Fix: Calculate return after fees correctly for both long and short positions
        # First calculate base return, then subtract fees as absolute cost
        base_return_pct = (pnl / entry_price) * 100
        fee_cost_pct = (total_fees / entry_price) * 100
        return_pct_after_fees = base_return_pct - fee_cost_pct

        # Create trade record
        trade = {
            'news_time': news_time,
            'market_session': market_session,
            'day_of_week': news_time.strftime('%A'),
            'entry_time': entry_data.name,
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'sentiment': news['sentiment'],
            'title': news['title'],
            'explanation': news.get('explanation', ''),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': return_pct,
            'direction': 'LONG' if direction > 0 else 'SHORT',
            'max_favorable_excursion': max_favorable_excursion,
            'max_adverse_excursion': max_adverse_excursion,
            'trade_duration_min': (exit_time - entry_time).total_seconds() / 60,
            'hit_stop_loss': exit_reason == "stop_loss",
            'pnl': pnl,
            'pnl_after_fees': pnl_after_fees,
            'return_pct_after_fees': return_pct_after_fees,
            'transaction_fees': total_fees
        }
        
        return trade
        
    except Exception as e:
        logger.error(f"Error processing trade for news at {news.get('date', 'unknown')}: {e}")
        return None

def backtest_news_sentiment_strategy(
    ticker: str,
    sentiment_data: List[Dict[str, Any]],
    holding_period_minutes: int = BACKTEST_CONFIG["default_holding_period"],
    entry_delay_minutes: int = BACKTEST_CONFIG["default_entry_delay"],
    include_after_hours: bool = BACKTEST_CONFIG["default_include_after_hours"],
    transaction_fee_pct: float = BACKTEST_CONFIG["default_transaction_fee_pct"]
) -> Dict[str, Any]:
    """
    Backtest a simple news sentiment strategy without stop loss.
    
    Strategy:
    - Buy (go long) if sentiment is positive, X minutes after news
    - Short (go short) if sentiment is negative, X minutes after news
    - Hold position for Y minutes
    
    Args:
        ticker: Stock ticker symbol
        sentiment_data: List of sentiment analysis results with timestamps
        holding_period_minutes: How long to hold the position in total
        entry_delay_minutes: Delay in minutes before entering position
        include_after_hours: Whether to include trades outside regular market hours
        transaction_fee_pct: Percentage of transaction volume to be charged as fee
        
    Returns:
        Dictionary with backtest results
    """
    if not sentiment_data:
        return {"error": "No sentiment data provided"}
    
    trades = []
    total_return_pct = 0
    winning_trades = 0
    losing_trades = 0
    
    # Sort sentiment data by date
    sorted_sentiment = sorted(sentiment_data, key=lambda x: x['date'])
    
    for idx, news in enumerate(sorted_sentiment):
        try:
            # Parse news timestamp
            news_time = datetime.strptime(news['date'], '%Y-%m-%d %H:%M:%S')
            
            # Get market session classification
            market_session = classify_market_session(news_time)
            
            # Check if in trading hours (9:30 AM - 4:00 PM ET)
            is_trading_hours = (
                (news_time.hour > 9 or (news_time.hour == 9 and news_time.minute >= 30)) 
                and news_time.hour < 16
            )
            
            # Skip news outside trading hours if not including after hours
            if not include_after_hours and not is_trading_hours:
                logger.info(f"Skipping news outside trading hours: {news['date']}")
                filter_tracker.log_after_hours_filter(1)
                continue
                
            # Skip news on weekends
            if news_time.weekday() >= 5:  # 5=Saturday, 6=Sunday
                logger.info(f"Skipping weekend news: {news['date']}")
                filter_tracker.log_weekend_filter(1)
                continue
            
            # Get minute data around news event
            stock_data = get_minute_data_for_news_event(
                ticker, 
                news_time,
                minutes_before=5,
                minutes_after=holding_period_minutes + 5  # Add buffer
            )
            
            if stock_data.empty:
                logger.warning(f"No stock data available for news at {news['date']}")
                filter_tracker.log_missing_market_data_filter(1)
                continue
            
            # Calculate entry and exit points
            entry_time = news_time + timedelta(minutes=entry_delay_minutes)
            exit_time = news_time + timedelta(minutes=holding_period_minutes)
            
            # Find closest data points
            entry_data = stock_data[stock_data.index >= entry_time].iloc[0] if not stock_data[stock_data.index >= entry_time].empty else None
            exit_data = stock_data[stock_data.index >= exit_time].iloc[0] if not stock_data[stock_data.index >= exit_time].empty else None
            
            if entry_data is None or exit_data is None:
                logger.warning(f"Missing entry or exit data for news at {news['date']}")
                filter_tracker.log_missing_market_data_filter(1)
                continue
            
            # Calculate trade return based on sentiment
            entry_price = entry_data['open']
            exit_price = exit_data['close']
            
            # Calculate price path during the trade
            price_path = stock_data[
                (stock_data.index >= entry_time) &
                (stock_data.index <= exit_time)
            ]['close'].values
            
            if len(price_path) < 2:
                logger.warning(f"Insufficient price data for trade at {news['date']}")
                filter_tracker.log_missing_market_data_filter(1)
                continue
                
            # Determine position direction based on sentiment
            direction = 1 if news['sentiment'] == 'positive' else -1 if news['sentiment'] == 'negative' else 0
            
            # Skip neutral sentiment
            if direction == 0:
                logger.info(f"Skipping neutral sentiment news at {news['date']}")
                filter_tracker.log_neutral_sentiment_filter(1)
                continue
            
            # Calculate return
            trade_return_pct = direction * (exit_price - entry_price) / entry_price * 100
            total_return_pct += trade_return_pct
            
            # Calculate intra-trade metrics
            max_favorable_excursion = 0
            max_adverse_excursion = 0
            
            for price in price_path:
                # Calculate current P&L at this point
                current_return = direction * (price - entry_price) / entry_price * 100
                
                # Update MFE (maximum favorable excursion)
                if current_return > max_favorable_excursion:
                    max_favorable_excursion = current_return
                
                # Update MAE (maximum adverse excursion)
                if current_return < max_adverse_excursion:
                    max_adverse_excursion = current_return
            
            # Calculate P&L
            pnl = (exit_price - entry_price) * direction
            return_pct = (pnl / entry_price) * 100
            
            # Apply transaction fees
            entry_fee = entry_price * transaction_fee_pct
            exit_fee = exit_price * transaction_fee_pct
            total_fees = entry_fee + exit_fee
            
            pnl_after_fees = pnl - total_fees
            # Fix: Calculate return after fees correctly for both long and short positions
            # First calculate base return, then subtract fees as absolute cost
            base_return_pct = (pnl / entry_price) * 100
            fee_cost_pct = (total_fees / entry_price) * 100
            return_pct_after_fees = base_return_pct - fee_cost_pct

            # Record trade details
            trade = {
                'news_time': news_time,
                'market_session': market_session,
                'day_of_week': news_time.strftime('%A'),
                'entry_time': entry_data.name,
                'exit_time': exit_data.name,
                'sentiment': news['sentiment'],
                'title': news['title'],
                'explanation': news.get('explanation', ''),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': return_pct,
                'direction': 'LONG' if direction > 0 else 'SHORT',
                'max_favorable_excursion': max_favorable_excursion,
                'max_adverse_excursion': max_adverse_excursion,
                'trade_duration_min': holding_period_minutes - entry_delay_minutes,
                'pnl': pnl,
                'pnl_after_fees': pnl_after_fees,
                'return_pct_after_fees': return_pct_after_fees,
                'transaction_fees': total_fees
            }
            trades.append(trade)
            
            # Count winning/losing trades
            if return_pct_after_fees > 0:
                winning_trades += 1
            else:
                losing_trades += 1
                
            logger.info(f"Trade {idx+1}: {trade['direction']} based on {trade['sentiment']} sentiment, return: {return_pct_after_fees:.2f}%")
            
        except Exception as e:
            logger.error(f"Error processing trade for news at {news.get('date', 'unknown')}: {e}")
            continue
    
    # Calculate overall statistics
    total_trades = len(trades)
    
    # Convert trades to DataFrame for analysis
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Initialize results
    results = {
        'ticker': ticker,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'total_return_pct': total_return_pct,
        'average_return_pct': total_return_pct / total_trades if total_trades > 0 else 0,
        'sentiment_stats': {},
        'market_session_stats': {},
        'day_of_week_stats': {},
        'trades': trades
    }
    
    # If no trades were executed, return early
    if total_trades == 0:
        return results
    
    # Extract returns for statistical analysis
    all_returns = trades_df['return_pct_after_fees'].values if not trades_df.empty else np.array([])
    
    # Calculate advanced statistics
    stats = calculate_trade_statistics(all_returns)
    results.update({
        'median_return_pct': stats['median_return'],
        'return_std_dev': stats['std_dev'],
        'sharpe_ratio': stats['sharpe_ratio'],
        'max_drawdown_pct': stats['max_drawdown'],
        'drawdown_duration': stats['drawdown_duration'],
        'profit_factor': stats['profit_factor'],
        'expected_return_pct': stats['expected_return']
    })
    
    # Create summary stats by sentiment
    if not trades_df.empty:
        # Group by sentiment
        for sentiment in ['positive', 'negative']:
            sentiment_trades = trades_df[trades_df['sentiment'] == sentiment]
            if not sentiment_trades.empty:
                sentiment_returns = sentiment_trades['return_pct_after_fees'].values
                sentiment_stats = calculate_trade_statistics(sentiment_returns)
                
                results['sentiment_stats'][sentiment] = {
                    'count': len(sentiment_trades),
                    'total_return': sentiment_returns.sum(),
                    'avg_return': sentiment_stats['avg_return'],
                    'median_return': sentiment_stats['median_return'],
                    'win_rate': sentiment_stats['win_rate'],
                    'std_dev': sentiment_stats['std_dev'],
                    'sharpe_ratio': sentiment_stats['sharpe_ratio'],
                    'profit_factor': sentiment_stats['profit_factor']
                }
        
        # Group by market session
        market_sessions = trades_df['market_session'].unique()
        for session in market_sessions:
            session_trades = trades_df[trades_df['market_session'] == session]
            if not session_trades.empty:
                session_returns = session_trades['return_pct_after_fees'].values
                session_stats = calculate_trade_statistics(session_returns)
                
                results['market_session_stats'][session] = {
                    'count': len(session_trades),
                    'total_return': session_returns.sum(),
                    'avg_return': session_stats['avg_return'],
                    'win_rate': session_stats['win_rate']
                }
        
        # Group by day of week
        days_of_week = trades_df['day_of_week'].unique()
        for day in days_of_week:
            day_trades = trades_df[trades_df['day_of_week'] == day]
            if not day_trades.empty:
                day_returns = day_trades['return_pct_after_fees'].values
                day_stats = calculate_trade_statistics(day_returns)
                
                results['day_of_week_stats'][day] = {
                    'count': len(day_trades),
                    'total_return': day_returns.sum(),
                    'avg_return': day_stats['avg_return'],
                    'win_rate': day_stats['win_rate']
                }
    
    return results

def backtest_news_sentiment_strategy_with_stop_loss(
    ticker: str,
    sentiment_data: List[Dict[str, Any]],
    entry_delay_minutes: int = BACKTEST_CONFIG["default_entry_delay"],
    max_holding_period_minutes: int = BACKTEST_CONFIG["default_holding_period"],
    stop_loss_pct: float = BACKTEST_CONFIG["default_stop_loss_pct"],
    take_profit_pct: float = BACKTEST_CONFIG["default_take_profit_pct"],
    include_after_hours: bool = BACKTEST_CONFIG["default_include_after_hours"],
    max_workers: int = BACKTEST_CONFIG["default_max_workers"],
    transaction_fee_pct: float = BACKTEST_CONFIG["default_transaction_fee_pct"]
) -> Dict[str, Any]:
    """
    Backtest a news sentiment trading strategy with stop loss and take profit.
    
    This is an improved version of the standard strategy that adds stop loss protection
    and take profit targets to limit losses and capture gains.
    
    Args:
        ticker: Stock ticker symbol
        sentiment_data: List of sentiment analysis results
        entry_delay_minutes: Minutes to wait after news before entering position
        max_holding_period_minutes: Maximum minutes to hold from news publication
        stop_loss_pct: Stop loss percentage (positive value, e.g. 3.0 means 3%)
        take_profit_pct: Take profit percentage (positive value, e.g. 6.0 means 6%)
        include_after_hours: Whether to include trades outside market hours
        max_workers: Maximum number of worker threads for parallel processing
        transaction_fee_pct: Percentage of transaction volume to be charged as fee
        
    Returns:
        Dictionary with backtest results
    """
    if not sentiment_data:
        logger.warning(f"No sentiment data provided for {ticker}")
        return {"ticker": ticker, "error": "No sentiment data"}
    
    logger.info(f"Backtesting {ticker} with stop loss at {stop_loss_pct:.1f}% and take profit at {take_profit_pct:.1f}%")
    logger.info(f"Processing {len(sentiment_data)} news items with {max_workers} workers")
    
    # Thread-safe trade collection
    trades = []
    trades_lock = threading.Lock()
    total_news = len(sentiment_data)
    
    # Define a function to process news with logging for parallel execution
    def process_news_item(i, news):
        try:
            logger.debug(f"Processing news {i+1}/{total_news}: {news['date']} - {news['sentiment']}")
            trade = process_trade_with_stop_loss(
                ticker=ticker,
                news=news,
                entry_delay_minutes=entry_delay_minutes,
                max_holding_period_minutes=max_holding_period_minutes,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                transaction_fee_pct=transaction_fee_pct
            )
            
            if trade:
                # Add day of week
                trade_date = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
                trade['day_of_week'] = trade_date.strftime('%A')
                
                # Only include trades during market hours if specified
                if not include_after_hours:
                    if trade['market_session'] in ['Pre-Market', 'After-Hours']:
                        logger.debug(f"Skipping {trade['market_session']} trade at {trade['entry_time']}")
                        return None
                
                logger.debug(f"Trade recorded: {trade['entry_time']} to {trade['exit_time']}, "
                           f"return: {trade['return_pct_after_fees']:.2f}%, "
                           f"exit reason: {trade.get('exit_reason', 'time')}")
                return trade
            return None
        except Exception as e:
            logger.error(f"Error processing news item {i}: {e}", exc_info=True)
            return None
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = [executor.submit(process_news_item, i, news) for i, news in enumerate(sentiment_data)]
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    trade = future.result(timeout=TRADING_CONSTANTS["THREAD_FUTURE_TIMEOUT_SECONDS"])  # Add timeout to prevent hanging
                    if trade:
                        # Thread-safe append
                        with trades_lock:
                            trades.append(trade)
                except Exception as e:
                    logger.error(f"Error getting future result: {e}", exc_info=True)
                finally:
                    # Explicitly delete future reference to prevent memory leaks
                    if 'trade' in locals():
                        del trade
            
            # Clean up all futures explicitly
            for future in futures:
                if not future.done():
                    future.cancel()
                del future
            futures.clear()

    except Exception as e:
        logger.error(f"Error in parallel processing: {e}", exc_info=True)
        # Fall back to sequential processing
        logger.info("Falling back to sequential processing...")
        trades.clear()  # Clear any partial results
        for i, news in enumerate(sentiment_data):
            try:
                trade = process_news_item(i, news)
                if trade:
                    trades.append(trade)
            except Exception as e:
                logger.error(f"Error in sequential processing for news {i}: {e}", exc_info=True)
                continue

    # Sort trades by exit time
    trades = sorted(trades, key=lambda x: datetime.strptime(x['exit_time'], '%Y-%m-%d %H:%M:%S'))

    # Calculate statistics
    total_trades = len(trades)
    
    if total_trades == 0:
        logger.warning("No trades were executed during backtesting")
        return {
            'ticker': ticker,
            'total_trades': 0,
            'error': "No trades executed"
        }
    
    # Convert trades to DataFrame for analysis
    trades_df = pd.DataFrame(trades)
    
    # Calculate basic stats
    winning_trades = (trades_df['return_pct_after_fees'] > 0).sum()
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_return_pct = trades_df['return_pct_after_fees'].sum()
    average_return_pct = trades_df['return_pct_after_fees'].mean()
    
    # Initialize results
    results = {
        'ticker': ticker,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_return_pct': total_return_pct,
        'average_return_pct': average_return_pct,
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct,
        'sentiment_stats': {},
        'market_session_stats': {},
        'day_of_week_stats': {},
        'exit_reason_stats': {},
        'trades': trades
    }
    
    # Extract returns for statistical analysis
    all_returns = trades_df['return_pct_after_fees'].values
    
    # Calculate advanced statistics
    stats = calculate_trade_statistics(all_returns)
    results.update({
        'median_return_pct': stats['median_return'],
        'return_std_dev': stats['std_dev'],
        'sharpe_ratio': stats['sharpe_ratio'],
        'max_drawdown_pct': stats['max_drawdown'],
        'drawdown_duration': stats['drawdown_duration'],
        'profit_factor': stats['profit_factor'],
        'expected_return_pct': stats['expected_return']
    })
    
    # Calculate average trade duration
    results['avg_trade_duration_min'] = trades_df['trade_duration_min'].mean()
    
    # Create summary stats by sentiment
    for sentiment in ['positive', 'negative']:
        sentiment_trades = trades_df[trades_df['sentiment'] == sentiment]
        if not sentiment_trades.empty:
            sentiment_returns = sentiment_trades['return_pct_after_fees'].values
            sentiment_stats = calculate_trade_statistics(sentiment_returns)
            
            results['sentiment_stats'][sentiment] = {
                'count': len(sentiment_trades),
                'total_return': sentiment_returns.sum(),
                'avg_return': sentiment_stats['avg_return'],
                'median_return': sentiment_stats['median_return'],
                'win_rate': sentiment_stats['win_rate'],
                'std_dev': sentiment_stats['std_dev'],
                'sharpe_ratio': sentiment_stats['sharpe_ratio'],
                'profit_factor': sentiment_stats['profit_factor']
            }
    
    # Create summary stats by market session
    market_sessions = trades_df['market_session'].unique()
    for session in market_sessions:
        session_trades = trades_df[trades_df['market_session'] == session]
        if not session_trades.empty:
            session_returns = session_trades['return_pct_after_fees'].values
            session_stats = calculate_trade_statistics(session_returns)
            
            results['market_session_stats'][session] = {
                'count': len(session_trades),
                'total_return': session_returns.sum(),
                'avg_return': session_stats['avg_return'],
                'win_rate': session_stats['win_rate']
            }
    
    # Create summary stats by day of week
    days_of_week = trades_df['day_of_week'].unique()
    for day in days_of_week:
        day_trades = trades_df[trades_df['day_of_week'] == day]
        if not day_trades.empty:
            day_returns = day_trades['return_pct_after_fees'].values
            day_stats = calculate_trade_statistics(day_returns)
            
            results['day_of_week_stats'][day] = {
                'count': len(day_trades),
                'total_return': day_returns.sum(),
                'avg_return': day_stats['avg_return'],
                'win_rate': day_stats['win_rate']
            }
    
    # Create summary stats by exit reason
    exit_reasons = trades_df['exit_reason'].unique()
    for reason in exit_reasons:
        reason_trades = trades_df[trades_df['exit_reason'] == reason]
        if not reason_trades.empty:
            reason_returns = reason_trades['return_pct_after_fees'].values
            reason_stats = calculate_trade_statistics(reason_returns)
            
            results['exit_reason_stats'][reason] = {
                'count': len(reason_trades),
                'total_return': reason_returns.sum(),
                'avg_return': reason_stats['avg_return'],
                'win_rate': reason_stats['win_rate']
            }
    
    return results

def backtest_news_sentiment_flip_strategy(
    ticker: str,
    sentiment_data: List[Dict[str, Any]],
    entry_delay_minutes: int = BACKTEST_CONFIG["default_entry_delay"],
    max_holding_period_minutes: int = BACKTEST_CONFIG["default_holding_period"],
    stop_loss_pct: float = BACKTEST_CONFIG["default_stop_loss_pct"],
    take_profit_pct: float = BACKTEST_CONFIG["default_take_profit_pct"],
    include_after_hours: bool = BACKTEST_CONFIG["default_include_after_hours"],
    transaction_fee_pct: float = BACKTEST_CONFIG["default_transaction_fee_pct"]
) -> Dict[str, Any]:
    """
    Backtest a news sentiment trading strategy that flips positions on conflicting news.
    
    Args:
        ticker: Stock ticker symbol
        sentiment_data: List of sentiment analysis results
        entry_delay_minutes: Minutes to wait after news before entering position
        max_holding_period_minutes: Maximum minutes to hold position
        stop_loss_pct: Stop loss percentage (positive value, e.g. 3.0 means 3%)
        take_profit_pct: Take profit percentage (positive value, e.g. 6.0 means 6%)
        include_after_hours: Whether to include trades outside market hours
        transaction_fee_pct: Percentage of transaction volume to be charged as fee
        
    Returns:
        Dictionary with backtest results
    """
    if not sentiment_data:
        logger.error(f"No sentiment data provided for {ticker}")
        return {"error": "No sentiment data provided"}
        
    # Verify stop_loss_pct is correctly applied (it should be negative)
    logger.info(f"Using stop loss of {stop_loss_pct:.2f}% for {ticker}")
    logger.info(f"Processing {len(sentiment_data)} non-neutral news items in flip strategy")

    trades = []
    active_trade = None  # Stores details of the currently active trade

    # Sort news chronologically
    sorted_sentiment = sorted(sentiment_data, key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d %H:%M:%S'))
    logger.info(f"Sorted sentiment data from {sorted_sentiment[0]['date']} to {sorted_sentiment[-1]['date']}")

    # Filter out after-hours news if not including after-hours trading
    if not include_after_hours:
        original_count = len(sorted_sentiment)
        sorted_sentiment = [
            news for news in sorted_sentiment 
            if is_market_open(datetime.strptime(news['date'], '%Y-%m-%d %H:%M:%S'), extended=False)
        ]
        filtered_count = original_count - len(sorted_sentiment)
        filter_tracker.log_after_hours_filter(filtered_count)
        logger.info(f"Filtered out {filtered_count} news items outside regular trading hours. {len(sorted_sentiment)} remain.")
        
        if not sorted_sentiment:
            logger.error(f"No news items remain after filtering out after-hours news for {ticker}")
            return {'error': "No valid sentiment data after filtering for trading hours"}

    # Determine the full date range needed for price data
    if not sorted_sentiment: 
        logger.error(f"No valid sentiment data for {ticker} after filtering.")
        return {'error': "No valid sentiment data after filtering"}
        
    start_date = datetime.strptime(sorted_sentiment[0]['date'], '%Y-%m-%d %H:%M:%S') - timedelta(minutes=entry_delay_minutes + 5)
    end_date = datetime.strptime(sorted_sentiment[-1]['date'], '%Y-%m-%d %H:%M:%S') + timedelta(minutes=max_holding_period_minutes + 5)
    
    logger.info(f"Fetching price data for {ticker} from {start_date} to {end_date}")
    try:
        all_stock_data = fetch_stock_prices(ticker, start_date, end_date) # Fetch all data upfront
        
        if all_stock_data.empty:
            logger.error(f"Could not fetch any stock data for {ticker} in the required range.")
            return {'error': f'No stock data for {ticker}'}
            
        logger.info(f"Fetched {len(all_stock_data)} price data points from {all_stock_data.index.min()} to {all_stock_data.index.max()}")
    except Exception as e:
        logger.error(f"Error fetching stock prices for {ticker}: {e}", exc_info=True)
        return {'error': f'Failed to fetch stock data: {str(e)}'}

    logger.info(f"Processing {len(sorted_sentiment)} news items sequentially for flip strategy...")

    # Counter for debugging
    news_processed = 0
    entry_attempts = 0
    entries_successful = 0
    exit_count = 0
    
    for news in sorted_sentiment:
        news_processed += 1
        news_time = datetime.strptime(news['date'], '%Y-%m-%d %H:%M:%S')
        sentiment = news['sentiment']
        potential_entry_time = news_time + timedelta(minutes=entry_delay_minutes)
        news_direction = 1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0

        logger.debug(f"Processing news {news_processed}/{len(sorted_sentiment)}: {news_time} - {sentiment}")

        # Filter based on trading hours if needed
        if not include_after_hours and not is_market_open(news_time, extended=False):
            logger.debug(f"Skipping news at {news_time} (outside RTH). Active trade: {active_trade is not None}")
            # Still need to check if an active trade needs closing due to SL/MaxHold during this time
            # This part adds complexity - for now, assume active trades persist but no new entries/flips outside hours
            pass # Simplification: We don't exit active trades based on OOH news time itself, only price action

        # --- Check and Manage Active Trade --- 
        if active_trade:
            # Check if the active trade hit stop loss or max holding BEFORE this news event
            exit_triggered = False
            triggered_exit_time = None
            triggered_exit_price = None
            triggered_reason = None

            # Get price data between active trade entry and current news time
            trade_path = all_stock_data[
                (all_stock_data.index > active_trade['entry_time']) &
                (all_stock_data.index <= news_time) # Up to current news time
            ]
            
            logger.debug(f"Checking active trade from {active_trade['entry_time']} to {news_time}. Found {len(trade_path)} price points.")

            for timestamp, row in trade_path.iterrows():
                # 1. Check Stop Loss - Note: stop_loss_pct is positive, so we check against negative
                current_price = row['close']
                current_return_pct = active_trade['direction'] * (current_price - active_trade['entry_price']) / active_trade['entry_price'] * 100
                if current_return_pct <= -stop_loss_pct:
                    exit_triggered = True
                    triggered_exit_time = timestamp
                    triggered_exit_price = current_price # Exit at the closing price of the triggering bar
                    triggered_reason = "stop_loss"
                    logger.debug(f"Trade stop loss triggered at {timestamp} price {current_price:.2f}, return: {current_return_pct:.2f}%, threshold: -{stop_loss_pct:.2f}%")
                    break
                
                # 2. Check Take Profit - Note: take_profit_pct is positive
                if current_return_pct >= take_profit_pct:
                    exit_triggered = True
                    triggered_exit_time = timestamp
                    triggered_exit_price = current_price
                    triggered_reason = "take_profit"
                    logger.debug(f"Trade take profit triggered at {timestamp} price {current_price:.2f}, return: {current_return_pct:.2f}%, threshold: {take_profit_pct:.2f}%")
                    break
                
                # 3. Check Max Holding Period
                if timestamp >= active_trade['planned_exit_time']:
                    exit_triggered = True
                    triggered_exit_time = timestamp
                    triggered_exit_price = row['close'] # Exit at the closing price of the triggering bar
                    triggered_reason = "max_holding_period"
                    logger.debug(f"Trade max holding period reached at {timestamp}")
                    break
            
            # If an exit was triggered before this news event
            if exit_triggered:
                exit_count += 1
                final_return = active_trade['direction'] * (triggered_exit_price - active_trade['entry_price']) / active_trade['entry_price'] * 100
                trade_duration = (triggered_exit_time - active_trade['entry_time']).total_seconds() / 60
                
                # Calculate P&L and fees for the closed trade
                pnl = (triggered_exit_price - active_trade['entry_price']) * active_trade['direction']
                entry_fee = active_trade['entry_price'] * transaction_fee_pct
                exit_fee = triggered_exit_price * transaction_fee_pct
                total_fees = entry_fee + exit_fee
                pnl_after_fees = pnl - total_fees
                # Fix: Calculate return after fees correctly for both long and short positions
                base_return_pct = (pnl / active_trade['entry_price']) * 100
                fee_cost_pct = (total_fees / active_trade['entry_price']) * 100
                return_pct_after_fees = base_return_pct - fee_cost_pct

                trades.append({
                    # ... (add all trade details: entry time/price, exit time/price, return, duration, reason, sentiment, etc.)
                    'ticker': ticker,
                    'entry_time': active_trade['entry_time'],
                    'entry_price': active_trade['entry_price'],
                    'exit_time': triggered_exit_time,
                    'exit_price': triggered_exit_price,
                    'direction': 'long' if active_trade['direction'] == 1 else 'short',
                    'sentiment': active_trade['sentiment'],
                    'return_pct': final_return, # This is pre-fee return for raw metric
                    'trade_duration_min': trade_duration,
                    'exit_reason': triggered_reason,
                    'market_session': classify_market_session(active_trade['entry_time']),
                    'day_of_week': active_trade['entry_time'].strftime('%A'),
                    'pnl': pnl,
                    'transaction_fees': total_fees,
                    'pnl_after_fees': pnl_after_fees,
                    'return_pct_after_fees': return_pct_after_fees
                })
                active_trade = None # Trade closed
                logger.debug(f"Closed trade with {triggered_reason}, return: {final_return:.2f}%, return after fees: {return_pct_after_fees:.2f}%, duration: {trade_duration:.1f} min")

        # --- Process Current News Event --- 
        # Can only potentially act on news if sentiment is not neutral
        if news_direction != 0:
            # Check if we need to FLIP an existing trade
            if active_trade and news_direction == -active_trade['direction']:
                logger.info(f"Conflicting news at {news_time}. Closing active {active_trade['direction']} trade and flipping.")
                # Close the active trade NOW (at news time)
                exit_price_data = all_stock_data[all_stock_data.index >= news_time]
                if not exit_price_data.empty:
                    exit_count += 1
                    exit_price = exit_price_data.iloc[0]['open'] # Exit at the open of the news minute bar
                    exit_time = exit_price_data.index[0]
                    final_return = active_trade['direction'] * (exit_price - active_trade['entry_price']) / active_trade['entry_price'] * 100
                    trade_duration = (exit_time - active_trade['entry_time']).total_seconds() / 60

                    # Calculate P&L and fees for the flipped trade
                    pnl = (exit_price - active_trade['entry_price']) * active_trade['direction']
                    entry_fee = active_trade['entry_price'] * transaction_fee_pct
                    exit_fee = exit_price * transaction_fee_pct
                    total_fees = entry_fee + exit_fee
                    pnl_after_fees = pnl - total_fees
                    # Fix: Calculate return after fees correctly for both long and short positions
                    base_return_pct = (pnl / active_trade['entry_price']) * 100
                    fee_cost_pct = (total_fees / active_trade['entry_price']) * 100
                    return_pct_after_fees = base_return_pct - fee_cost_pct
                    
                    trades.append({
                        # ... (add all trade details)
                         'ticker': ticker,
                        'entry_time': active_trade['entry_time'],
                        'entry_price': active_trade['entry_price'],
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'direction': 'long' if active_trade['direction'] == 1 else 'short',
                        'sentiment': active_trade['sentiment'],
                        'return_pct': final_return, # Pre-fee
                        'trade_duration_min': trade_duration,
                        'exit_reason': 'conflict_flip',
                        'market_session': classify_market_session(active_trade['entry_time']),
                        'day_of_week': active_trade['entry_time'].strftime('%A'),
                        'pnl': pnl,
                        'transaction_fees': total_fees,
                        'pnl_after_fees': pnl_after_fees,
                        'return_pct_after_fees': return_pct_after_fees
                    })
                    logger.debug(f"Closed trade due to conflicting news, return: {final_return:.2f}%, return after fees: {return_pct_after_fees:.2f}%, duration: {trade_duration:.1f} min")
                    active_trade = None # Trade closed, ready for new entry from this news
                else:
                    logger.warning(f"Could not find price data at {news_time} to close flipped trade.")
                    active_trade = None # Force close if price is missing

            # Enter a NEW trade if no active trade exists
            if active_trade is None:
                # Find entry data point after delay
                entry_attempts += 1
                entry_data = all_stock_data[all_stock_data.index >= potential_entry_time]
                if not entry_data.empty:
                    entries_successful += 1
                    entry_bar = entry_data.iloc[0]
                    entry_price = entry_bar['open']
                    actual_entry_time = entry_bar.name # Use the actual timestamp of the bar
                    
                    # Calculate stop loss price - use negative stop_loss_pct since it's a loss
                    stop_price = entry_price * (1 - (stop_loss_pct / 100 * news_direction))
                    
                    # Calculate planned exit time
                    planned_exit_time = actual_entry_time + timedelta(minutes=max_holding_period_minutes)

                    active_trade = {
                        'entry_time': actual_entry_time,
                        'entry_price': entry_price,
                        'direction': news_direction,
                        'sentiment': sentiment,
                        'planned_exit_time': planned_exit_time,
                        'stop_loss_price': stop_price,
                        'news_time': news_time # Keep track of the triggering news time
                    }
                    logger.info(f"Entering {'LONG' if news_direction==1 else 'SHORT'} trade for {ticker} at {actual_entry_time} price {entry_price:.2f} based on news from {news_time}")
                else:
                    logger.warning(f"No stock data found at or after potential entry time {potential_entry_time} for news at {news_time}")

    # --- Final Check --- 
    # If loop finishes and a trade is still active, close it based on last available data or planned exit
    if active_trade:
        logger.info(f"Closing final active trade at end of data range.")
        exit_time = min(active_trade['planned_exit_time'], all_stock_data.index[-1])
        exit_price_data = all_stock_data[all_stock_data.index >= exit_time]
        exit_price = exit_price_data.iloc[0]['open'] if not exit_price_data.empty else active_trade['entry_price'] # Fallback
        exit_reason = "end_of_data" if exit_time >= active_trade['planned_exit_time'] else "max_holding_period"
        
        # Check one last time for stop loss or take profit between entry and final exit time
        final_trade_path = all_stock_data[
            (all_stock_data.index > active_trade['entry_time']) &
            (all_stock_data.index <= exit_time) 
        ]
        stop_loss_final_check = False
        for timestamp, row in final_trade_path.iterrows():
            current_price = row['close']
            current_return_pct = active_trade['direction'] * (current_price - active_trade['entry_price']) / active_trade['entry_price'] * 100
            
            # Check stop loss
            if current_return_pct <= -stop_loss_pct:
                exit_price = current_price
                exit_time = timestamp
                exit_reason = "stop_loss"
                stop_loss_final_check = True
                logger.debug(f"Final trade stop loss triggered at {timestamp}, return: {current_return_pct:.2f}%, threshold: -{stop_loss_pct:.2f}%")
                break
                
            # Check take profit
            if current_return_pct >= take_profit_pct:
                exit_price = current_price
                exit_time = timestamp
                exit_reason = "take_profit"
                stop_loss_final_check = True
                logger.debug(f"Final trade take profit triggered at {timestamp}, return: {current_return_pct:.2f}%, threshold: {take_profit_pct:.2f}%")
                break
                
        exit_count += 1
        final_return = active_trade['direction'] * (exit_price - active_trade['entry_price']) / active_trade['entry_price'] * 100
        trade_duration = (exit_time - active_trade['entry_time']).total_seconds() / 60
        
        # Calculate P&L and fees for the final trade
        pnl = (exit_price - active_trade['entry_price']) * active_trade['direction']
        entry_fee = active_trade['entry_price'] * transaction_fee_pct
        exit_fee = exit_price * transaction_fee_pct
        total_fees = entry_fee + exit_fee
        pnl_after_fees = pnl - total_fees
        # Fix: Calculate return after fees correctly for both long and short positions
        base_return_pct = (pnl / active_trade['entry_price']) * 100
        fee_cost_pct = (total_fees / active_trade['entry_price']) * 100
        return_pct_after_fees = base_return_pct - fee_cost_pct
        
        trades.append({
            # ... (add all trade details)
            'ticker': ticker,
            'entry_time': active_trade['entry_time'],
            'entry_price': active_trade['entry_price'],
            'exit_time': exit_time,
            'exit_price': exit_price,
            'direction': 'long' if active_trade['direction'] == 1 else 'short',
            'sentiment': active_trade['sentiment'],
            'return_pct': final_return, # Pre-fee
            'trade_duration_min': trade_duration,
            'exit_reason': exit_reason,
            'market_session': classify_market_session(active_trade['entry_time']),
            'day_of_week': active_trade['entry_time'].strftime('%A'),
            'pnl': pnl,
            'transaction_fees': total_fees,
            'pnl_after_fees': pnl_after_fees,
            'return_pct_after_fees': return_pct_after_fees
        })
        active_trade = None

    # --- Results Calculation --- 
    total_trades = len(trades)
    logger.info(f"News processed: {news_processed}, Entry attempts: {entry_attempts}, Successful entries: {entries_successful}, Exits: {exit_count}")
    logger.info(f"Total trades executed: {total_trades}")
    
    if total_trades == 0:
        logger.warning(f"No trades executed for {ticker} with flip strategy.")
        return {
            'ticker': ticker, 'total_trades': 0, 'error': "No trades executed"
        }

    # Ensure trades are chronologically sorted by exit time
    trades = sorted(trades, key=lambda x: x['exit_time'])
    trades_df = pd.DataFrame(trades)
    
    # Calculate cumulative returns correctly
    trades_df['cumulative_return'] = trades_df['return_pct'].cumsum()
    
    # Calculate overall statistics using return_pct_after_fees
    winning_trades = (trades_df['return_pct_after_fees'] > 0).sum()
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_return_pct = trades_df['return_pct_after_fees'].sum() # Use post-fee return
    average_return_pct = trades_df['return_pct_after_fees'].mean() # Use post-fee return
    
    # Process advanced metrics using return_pct_after_fees
    all_returns = trades_df['return_pct_after_fees'].values
    stats = calculate_trade_statistics(all_returns)
    
    # Compile results
    results = {
        'ticker': ticker,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_return_pct': total_return_pct,
        'average_return_pct': average_return_pct,
        'median_return_pct': stats['median_return'],
        'return_std_dev': stats['std_dev'],
        'sharpe_ratio': stats['sharpe_ratio'],
        'max_drawdown_pct': stats['max_drawdown'],
        'drawdown_duration': stats['drawdown_duration'],
        'profit_factor': stats['profit_factor'],
        'expected_return_pct': stats['expected_return'],
        'trades': trades
    }
    
    # Add sentiment statistics
    results['sentiment_stats'] = {}
    for sentiment in ['positive', 'negative']:
        sentiment_trades = trades_df[trades_df['sentiment'] == sentiment]
        if not sentiment_trades.empty:
            sentiment_returns = sentiment_trades['return_pct_after_fees'].values # Use post-fee return
            sentiment_stats = calculate_trade_statistics(sentiment_returns)
            
            results['sentiment_stats'][sentiment] = {
                'count': len(sentiment_trades),
                'total_return': sentiment_returns.sum(),
                'avg_return': sentiment_stats['avg_return'],
                'median_return': sentiment_stats['median_return'],
                'win_rate': sentiment_stats['win_rate'],
                'std_dev': sentiment_stats['std_dev'],
                'sharpe_ratio': sentiment_stats['sharpe_ratio'],
                'profit_factor': sentiment_stats['profit_factor']
            }
    
    # Add exit reason statistics
    results['exit_reason_stats'] = {}
    exit_reasons = trades_df['exit_reason'].unique()
    for reason in exit_reasons:
        reason_trades = trades_df[trades_df['exit_reason'] == reason]
        if not reason_trades.empty:
            reason_returns = reason_trades['return_pct_after_fees'].values # Use post-fee return
            reason_stats = calculate_trade_statistics(reason_returns)
            
            results['exit_reason_stats'][reason] = {
                'count': len(reason_trades),
                'total_return': reason_returns.sum(),
                'avg_return': reason_stats['avg_return'],
                'win_rate': reason_stats['win_rate']
            }
    
    # Add market session statistics
    results['market_session_stats'] = {}
    market_sessions = trades_df['market_session'].unique()
    for session in market_sessions:
        session_trades = trades_df[trades_df['market_session'] == session]
        if not session_trades.empty:
            session_returns = session_trades['return_pct_after_fees'].values # Use post-fee return
            session_stats = calculate_trade_statistics(session_returns)
            
            results['market_session_stats'][session] = {
                'count': len(session_trades),
                'total_return': session_returns.sum(),
                'avg_return': session_stats['avg_return'],
                'win_rate': session_stats['win_rate']
            }
            
    # Add day of week statistics
    results['day_of_week_stats'] = {}
    days_of_week = trades_df['day_of_week'].unique()
    for day in days_of_week:
        day_trades = trades_df[trades_df['day_of_week'] == day]
        if not day_trades.empty:
            day_returns = day_trades['return_pct_after_fees'].values # Use post-fee return
            day_stats = calculate_trade_statistics(day_returns)
            
            results['day_of_week_stats'][day] = {
                'count': len(day_trades),
                'total_return': day_returns.sum(),
                'avg_return': day_stats['avg_return'],
                'win_rate': day_stats['win_rate']
            }

    logger.info(f"Flip strategy backtest for {ticker} completed. Total trades: {results['total_trades']}, Win rate: {results['win_rate']:.2%}, Total return: {results['total_return_pct']:.2f}%" )

    return results

def simulate_simple_strategy(
    ticker: str, 
    sentiment_data: List[Dict[str, Any]],
    entry_delay_minutes: int = BACKTEST_CONFIG["default_entry_delay"],
    holding_period_minutes: int = BACKTEST_CONFIG["default_holding_period"],
    use_stop_loss: bool = BACKTEST_CONFIG["default_use_stop_loss"],
    stop_loss_pct: float = BACKTEST_CONFIG["default_stop_loss_pct"],
    take_profit_pct: float = BACKTEST_CONFIG["default_take_profit_pct"],
    include_after_hours: bool = BACKTEST_CONFIG["default_include_after_hours"],
    transaction_fee_pct: float = BACKTEST_CONFIG["default_transaction_fee_pct"]
) -> Dict[str, Any]:
    """
    Simulate a simple sentiment-based strategy for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        sentiment_data: List of sentiment analysis results
        entry_delay_minutes: Minutes to wait before entry
        holding_period_minutes: Minutes to hold position
        use_stop_loss: Whether to use stop loss
        stop_loss_pct: Stop loss percentage (positive value, e.g. 2.0 means 2%)
        take_profit_pct: Take profit percentage (positive value, e.g. 4.0 means 4%)
        include_after_hours: Whether to include after-hours trading
        transaction_fee_pct: Percentage of transaction volume to be charged as fee
        
    Returns:
        Dictionary with simulation results
    """
    if use_stop_loss:
        # Use stop loss strategy with the direct percentage values
        return backtest_news_sentiment_strategy_with_stop_loss(
            ticker=ticker,
            sentiment_data=sentiment_data,
            entry_delay_minutes=entry_delay_minutes,
            max_holding_period_minutes=holding_period_minutes,
            stop_loss_pct=stop_loss_pct,  # Use direct stop loss value
            take_profit_pct=take_profit_pct,  # Use direct take profit value
            include_after_hours=include_after_hours,
            transaction_fee_pct=transaction_fee_pct
        )
    else:
        # Without stop loss, use the basic strategy with just holding period
        return backtest_news_sentiment_strategy(
            ticker=ticker,
            sentiment_data=sentiment_data,
            holding_period_minutes=holding_period_minutes,
            entry_delay_minutes=entry_delay_minutes,
            include_after_hours=include_after_hours,
            transaction_fee_pct=transaction_fee_pct
        )

def calculate_backtest_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate various metrics from a list of trades.
    """
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_return_pct': 0,
            'average_return_pct': 0,
            'median_return_pct': 0, # Added median return
            'return_std_dev': 0, # Added standard deviation of returns
            'sharpe_ratio': 0, # Added Sharpe ratio
            'max_drawdown_pct': 0, # Added max drawdown
            'profit_factor': 0, # Added profit factor
            'total_fees_paid': 0 # Added total fees
        }

    returns = np.array([trade['return_pct_after_fees'] for trade in trades]) # Use return_pct_after_fees
    pnl_values = np.array([trade['pnl_after_fees'] for trade in trades]) # Use pnl_after_fees
    total_fees_paid = np.sum([trade.get('transaction_fees', 0) for trade in trades])

    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade['pnl_after_fees'] > 0) # Based on pnl_after_fees
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate total return based on P&L after fees relative to initial capital (assuming 1 unit per trade for simplicity here)
    # This is a simplification; a proper portfolio return would consider capital allocation.
    # For now, sum of P&L after fees gives a sense of net profit/loss in currency units.
    # Total percentage return is more complex without knowing initial capital or if reinvesting.
    # We'll use the average of individual trade returns after fees for 'total_return_pct' for consistency with 'average_return_pct'.
    total_return_pct = np.sum(returns) # Sum of individual returns_after_fees
    average_return_pct = np.mean(returns) if total_trades > 0 else 0
    median_return_pct = np.median(returns) if total_trades > 0 else 0
    return_std_dev = np.std(returns) if total_trades > 0 else 0

    # Sharpe Ratio (annualized, assuming daily returns and 252 trading days, risk-free rate = 0)
    # This is an approximation as trades are not necessarily daily.
    sharpe_ratio = (average_return_pct / return_std_dev) * np.sqrt(TRADING_CONSTANTS["TRADING_DAYS_PER_YEAR"]) if return_std_dev > 0 else 0
    
    # Max Drawdown calculation based on cumulative P&L after fees
    cumulative_pnl_after_fees = np.cumsum(pnl_values)
    running_max_pnl = np.maximum.accumulate(cumulative_pnl_after_fees)
    drawdown = cumulative_pnl_after_fees - running_max_pnl
    max_drawdown_pct = np.min(drawdown) if len(drawdown) > 0 else 0 # This is absolute drawdown, not percentage of capital easily

    # Profit Factor
    gross_profit = np.sum(pnl_values[pnl_values > 0])
    gross_loss = np.abs(np.sum(pnl_values[pnl_values < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0


    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_return_pct': total_return_pct, # Sum of individual trade returns after fees
        'average_return_pct': average_return_pct, # Average of individual trade returns after fees
        'median_return_pct': median_return_pct,
        'return_std_dev': return_std_dev,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown_pct, # This is absolute P&L drawdown
        'profit_factor': profit_factor,
        'total_fees_paid': total_fees_paid
    }

class SentimentTradingStrategy:
    """
    A class-based implementation of a news sentiment trading strategy.
    """
    
    def __init__(
        self, 
        ticker: str,
        entry_delay_minutes: int = BACKTEST_CONFIG["default_entry_delay"],
        holding_period_minutes: int = BACKTEST_CONFIG["default_holding_period"],
        use_stop_loss: bool = BACKTEST_CONFIG["default_use_stop_loss"],
        stop_loss_pct: float = BACKTEST_CONFIG["default_stop_loss_pct"],
        take_profit_pct: float = BACKTEST_CONFIG["default_take_profit_pct"],
        include_after_hours: bool = BACKTEST_CONFIG["default_include_after_hours"],
        transaction_fee_pct: float = BACKTEST_CONFIG["default_transaction_fee_pct"]
    ):
        """
        Initialize a sentiment trading strategy.
        
        Args:
            ticker: Stock ticker symbol
            entry_delay_minutes: Minutes to wait after news before entering position
            holding_period_minutes: Maximum minutes to hold the position
            use_stop_loss: Whether to use stop loss
            stop_loss_pct: Stop loss percentage threshold (positive value, e.g. 2.0 means 2%)
            take_profit_pct: Take profit percentage threshold (positive value, e.g. 4.0 means 4%)
            include_after_hours: Whether to include trades outside regular market hours
            transaction_fee_pct: Percentage of transaction volume to be charged as fee
        """
        self.ticker = ticker
        self.entry_delay_minutes = entry_delay_minutes
        self.holding_period_minutes = holding_period_minutes
        self.use_stop_loss = use_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.include_after_hours = include_after_hours
        self.transaction_fee_pct = transaction_fee_pct
        
    def backtest(self, sentiment_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a backtest of the strategy using historical data.
        
        Args:
            sentiment_data: List of sentiment analysis results with timestamps
            
        Returns:
            Dictionary with backtest results
        """
        if self.use_stop_loss:
            return backtest_news_sentiment_strategy_with_stop_loss(
                ticker=self.ticker,
                sentiment_data=sentiment_data,
                entry_delay_minutes=self.entry_delay_minutes,
                max_holding_period_minutes=self.holding_period_minutes,
                stop_loss_pct=self.stop_loss_pct,  # Use direct value without negation
                take_profit_pct=self.take_profit_pct,  # Use direct take profit value
                include_after_hours=self.include_after_hours,
                transaction_fee_pct=self.transaction_fee_pct
            )
        else:
            return backtest_news_sentiment_strategy(
                ticker=self.ticker,
                sentiment_data=sentiment_data,
                holding_period_minutes=self.holding_period_minutes,
                entry_delay_minutes=self.entry_delay_minutes,
                include_after_hours=self.include_after_hours,
                transaction_fee_pct=self.transaction_fee_pct
            ) 