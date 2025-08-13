"""
Backtesting pipeline module.

Handles the execution of backtesting strategies using sentiment data and market data.
"""

import os
import logging
import csv
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import timedelta

from src.backtesting.strategies.strategies import (
    simulate_simple_strategy, 
    calculate_backtest_metrics, 
    backtest_news_sentiment_strategy_with_stop_loss,
    backtest_news_sentiment_strategy,
    backtest_news_sentiment_flip_strategy
)
from src.backtesting.helpers.market_data import fetch_stock_prices
from src.backtesting.analysis.backtest_visualizer import visualize_backtest_results, BacktestVisualizer
from src.backtesting.helpers.utilities import save_json, logger
from src.config import BACKTEST_CONFIG

def run_backtest_pipeline(
    ticker: str, 
    sentiment_data: List[Dict[str, Any]], 
    entry_delay: int, 
    holding_period: int,
    use_stop_loss: bool = BACKTEST_CONFIG["default_use_stop_loss"],
    stop_loss_pct: float = BACKTEST_CONFIG["default_stop_loss_pct"],
    take_profit_pct: float = BACKTEST_CONFIG["default_take_profit_pct"],
    include_after_hours: bool = BACKTEST_CONFIG["default_include_after_hours"],
    max_workers: int = BACKTEST_CONFIG["default_max_workers"],
    use_flip_strategy: bool = BACKTEST_CONFIG["default_use_flip_strategy"],
    output_dir: str = "output",
    transaction_fee_pct: float = BACKTEST_CONFIG["default_transaction_fee_pct"]
) -> Optional[Dict[str, Any]]:
    """
    Run the complete backtesting pipeline.
    
    Args:
        ticker: Stock ticker symbol
        sentiment_data: List of sentiment analysis results
        entry_delay: Minutes to wait after news before entering position
        holding_period: Total minutes from news to exit position
        use_stop_loss: Whether to use stop loss strategy
        stop_loss_pct: Stop loss percentage threshold (positive number)
        take_profit_pct: Take profit percentage threshold (positive number)
        include_after_hours: Whether to include trades outside regular market hours
        max_workers: Maximum number of worker threads for parallel processing
        use_flip_strategy: Whether to use the flip strategy for conflicting news
        output_dir: Directory to save results
        transaction_fee_pct: Percentage of transaction volume to be charged as fee
        
    Returns:
        Dictionary with backtest results or None if pipeline failed
    """
    logger.info("=" * 80)
    logger.info(f"BACKTESTING PIPELINE FOR {ticker}")
    logger.info("=" * 80)
    
    # Create a timestamped directory for this specific backtest
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backtest_dir = os.path.join(output_dir, f"backtest_{ticker}_{timestamp}")
    os.makedirs(backtest_dir, exist_ok=True)
    
    # Add debug information about sentiment data
    positive_count = sum(1 for item in sentiment_data if item.get('sentiment') == 'positive')
    negative_count = sum(1 for item in sentiment_data if item.get('sentiment') == 'negative')
    neutral_count = sum(1 for item in sentiment_data if item.get('sentiment') == 'neutral')
    logger.info(f"Processing sentiment data: {len(sentiment_data)} total items")
    logger.info(f"Sentiment breakdown: {positive_count} positive, {negative_count} negative, {neutral_count} neutral")
    
    # Filter out neutral sentiment now and log
    actionable_sentiment = [news for news in sentiment_data if news.get('sentiment') in ['positive', 'negative']]
    logger.info(f"Actionable sentiment items (excluding neutral): {len(actionable_sentiment)}")
    
    if not actionable_sentiment:
        logger.error(f"No actionable sentiment data for {ticker} (all are neutral)")
        return {"ticker": ticker, "error": "No actionable sentiment data"}
    
    # Display strategy configuration
    logger.info(f"Strategy details:")
    
    if use_flip_strategy:
        logger.info(f"- Using flip strategy: positions are reversed when conflicting news arrives")
    else:
        logger.info(f"- Long position for positive news, short position for negative news")
    
    logger.info(f"- Enter {entry_delay} minutes after news publication")
    
    if use_stop_loss:
        logger.info(f"- Stop loss at {stop_loss_pct:.2f}% loss (from config)")
        logger.info(f"- Take profit at {take_profit_pct:.2f}% gain (from config)")
        logger.info(f"- Maximum holding period {holding_period} minutes from news publication")
    else:
        logger.info(f"- Exit {holding_period} minutes after news publication")
        
    logger.info(f"- Trading {ticker} based on {len(actionable_sentiment)} news items")
    logger.info(f"- Include after-hours trading: {'Yes' if include_after_hours else 'No'}")
    logger.info(f"- Parallel processing with {max_workers} worker threads")
    logger.info(f"- Transaction Fee: {transaction_fee_pct*100:.3f}% per trade")
    logger.info("-" * 80)
    
    # Save configuration to file
    strategy_config = {
        "ticker": ticker,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_news": len(sentiment_data),
        "positive_news": positive_count,
        "negative_news": negative_count,
        "neutral_news": neutral_count,
        "actionable_news": len(actionable_sentiment),
        "strategy_type": "flip_strategy" if use_flip_strategy else "standard_strategy",
        "entry_delay_minutes": entry_delay,
        "holding_period_minutes": holding_period,
        "use_stop_loss": use_stop_loss,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "include_after_hours": include_after_hours,
        "max_workers": max_workers,
        "transaction_fee_pct": transaction_fee_pct
    }
    
    # Save strategy config as JSON
    strategy_config_file = os.path.join(backtest_dir, "strategy_config.json")
    save_json(strategy_config, strategy_config_file)
    
    # Also save as CSV for easy reading
    strategy_csv_file = os.path.join(backtest_dir, "strategy_config.csv")
    with open(strategy_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        for key, value in strategy_config.items():
            writer.writerow([key, value])
    
    logger.info(f"Strategy configuration saved to {strategy_config_file} and {strategy_csv_file}")
    
    # No conversion - use the values directly from config
    logger.info(f"Using stop loss setting: {stop_loss_pct:.2f}%")
    logger.info(f"Using take profit setting: {take_profit_pct:.2f}%")
    
    # Run the appropriate backtesting strategy
    try:
        if use_flip_strategy:
            # Use the flip strategy which handles conflicting news by reversing positions
            logger.info(f"Using flip strategy for backtesting {ticker}")
            backtest_results = backtest_news_sentiment_flip_strategy(
                ticker=ticker,
                sentiment_data=actionable_sentiment,  # Use non-neutral sentiment only
                entry_delay_minutes=entry_delay,
                max_holding_period_minutes=holding_period,
                stop_loss_pct=stop_loss_pct,  # Use direct value
                take_profit_pct=take_profit_pct,  # Use direct value
                include_after_hours=include_after_hours,
                transaction_fee_pct=transaction_fee_pct
            )
        elif use_stop_loss:
            logger.info(f"Using stop loss strategy for backtesting {ticker}")
            backtest_results = backtest_news_sentiment_strategy_with_stop_loss(
                ticker=ticker,
                sentiment_data=actionable_sentiment,  # Use non-neutral sentiment only
                entry_delay_minutes=entry_delay,
                max_holding_period_minutes=holding_period,
                stop_loss_pct=stop_loss_pct,  # Use direct value
                take_profit_pct=take_profit_pct,  # Use direct value
                include_after_hours=include_after_hours,
                max_workers=max_workers,
                transaction_fee_pct=transaction_fee_pct
            )
        else:
            logger.info(f"Using standard strategy for backtesting {ticker}")
            backtest_results = backtest_news_sentiment_strategy(
                ticker=ticker,
                sentiment_data=actionable_sentiment,  # Use non-neutral sentiment only
                entry_delay_minutes=entry_delay,
                holding_period_minutes=holding_period,
                include_after_hours=include_after_hours,
                transaction_fee_pct=transaction_fee_pct
            )
    except Exception as e:
        logger.error(f"Error during backtesting: {e}", exc_info=True)
        # Re-raise the exception to be handled by the caller
        raise
    
    # Check for error
    if not backtest_results:
        logger.error(f"No backtest results returned for {ticker}")
        return None
        
    if "error" in backtest_results:
        error_msg = backtest_results.get("error", "Unknown error")
        logger.error(f"Backtesting error: {error_msg}")
        return None
    
    # Log trade summary
    logger.info(f"Trades generated: {backtest_results.get('total_trades', 0)}")
    
    if backtest_results.get('total_trades', 0) == 0:
        logger.warning(f"No trades were generated for {ticker}. Check market data availability and settings.")
        return backtest_results
    
    # Print summary results
    print_backtest_summary(backtest_results, use_stop_loss)
    
    # Create exit reason analysis file for flip strategy
    if use_flip_strategy and 'trades' in backtest_results:
        create_exit_reason_analysis(backtest_results, backtest_dir)
    
    # Add buy & hold comparison
    calculate_buyhold_comparison(ticker, backtest_results)
    
    # Create and save visualizations
    create_backtest_visualizations(backtest_results, backtest_dir)
    
    # Save backtest results to file
    result_file = os.path.join(backtest_dir, f'backtest_results_{ticker}.json')
    save_json(backtest_results, result_file)
    
    # Also save to the output directory root for easy access by other functions
    root_result_file = os.path.join(output_dir, f'backtest_results_{ticker}.json')
    save_json(backtest_results, root_result_file)
    
    logger.info("=" * 80)
    logger.info(f"BACKTESTING PIPELINE COMPLETED FOR {ticker}")
    logger.info(f"Results saved to {backtest_dir}")
    logger.info("=" * 80)
    
    return backtest_results

def print_backtest_summary(backtest_results: Dict[str, Any], use_stop_loss: bool) -> None:
    """
    Print a summary of backtest results.
    
    Args:
        backtest_results: Dictionary with backtest results
        use_stop_loss: Whether stop loss was used
    """
    logger.info("BACKTEST RESULTS SUMMARY")
    logger.info("-" * 80)
    logger.info(f"Total trades: {backtest_results['total_trades']}")
    logger.info(f"Winning trades: {backtest_results['winning_trades']}")
    logger.info(f"Losing trades: {backtest_results['losing_trades']}")
    logger.info(f"Win rate: {backtest_results['win_rate'] * 100:.2f}%")
    logger.info(f"Total return: {backtest_results['total_return_pct']:.2f}%")
    logger.info(f"Average return per trade: {backtest_results['average_return_pct']:.2f}%")
    
    # Display Buy & Hold comparison if available
    if 'buy_hold_comparison' in backtest_results:
        bh_compare = backtest_results['buy_hold_comparison']
        logger.info("-" * 80)
        logger.info("STRATEGY vs BUY & HOLD COMPARISON")
        logger.info(f"Period: {bh_compare.get('start_date')} to {bh_compare.get('end_date')}")
        logger.info(f"Buy & Hold return: {bh_compare.get('buyhold_return_pct', 0):.2f}%")
        logger.info(f"Strategy return: {bh_compare.get('strategy_return_pct', 0):.2f}%")
        
        outperformance = bh_compare.get('outperformance_pct', 0)
        if outperformance > 0:
            logger.info(f"Outperformance: +{outperformance:.2f}% (strategy outperformed buy & hold)")
        else:
            logger.info(f"Outperformance: {outperformance:.2f}% (strategy underperformed buy & hold)")
    
    # Display stop loss and take profit statistics if applicable
    if use_stop_loss:
        # Count trades by exit reason
        exit_reasons = {'stop_loss': 0, 'take_profit': 0, 'max_holding_period': 0}
        trades_df = pd.DataFrame(backtest_results['trades'])
        
        if 'exit_reason' in trades_df.columns:
            for reason in exit_reasons.keys():
                exit_reasons[reason] = (trades_df['exit_reason'] == reason).sum()
                
            total = backtest_results['total_trades']
            logger.info(f"Exit reasons:")
            logger.info(f"  - Take profit: {exit_reasons['take_profit']} trades ({exit_reasons['take_profit']/total*100:.1f}%)")
            logger.info(f"  - Stop loss: {exit_reasons['stop_loss']} trades ({exit_reasons['stop_loss']/total*100:.1f}%)")
            logger.info(f"  - Max holding period: {exit_reasons['max_holding_period']} trades ({exit_reasons['max_holding_period']/total*100:.1f}%)")
            
        logger.info(f"Average trade duration: {backtest_results.get('avg_trade_duration_min', 0):.1f} minutes")
    
    # Display advanced metrics if available
    advanced_metrics = [
        ('Median return', 'median_return_pct', '%.2f%%'),
        ('Sharpe ratio', 'sharpe_ratio', '%.2f'),
        ('Max drawdown', 'max_drawdown_pct', '%.2f%%'),
        ('Profit factor', 'profit_factor', '%.2f')
    ]
    
    for label, key, fmt in advanced_metrics:
        if key in backtest_results:
            logger.info(f"{label}: {fmt % backtest_results[key]}")
    
    # Display sentiment-specific stats
    for sentiment, stats in backtest_results.get('sentiment_stats', {}).items():
        logger.info("-" * 40)
        logger.info(f"{sentiment.upper()} NEWS STATISTICS")
        logger.info(f"Count: {stats['count']} trades")
        logger.info(f"Average return: {stats['avg_return']:.2f}%")
        logger.info(f"Win rate: {stats['win_rate'] * 100:.2f}%")
        if 'profit_factor' in stats:
            logger.info(f"Profit factor: {stats['profit_factor']:.2f}")
    
    # Display market session stats
    if backtest_results.get('market_session_stats'):
        logger.info("-" * 40)
        logger.info("MARKET SESSION STATISTICS")
        for session, stats in backtest_results['market_session_stats'].items():
            logger.info(f"{session}: {stats['count']} trades, {stats['win_rate']*100:.1f}% win rate, "
                       f"{stats['avg_return']:.2f}% avg return")
    
    # Display exit reason stats
    if use_stop_loss and backtest_results.get('exit_reason_stats'):
        logger.info("-" * 40)
        logger.info("EXIT REASON STATISTICS")
        for reason, stats in backtest_results['exit_reason_stats'].items():
            logger.info(f"{reason}: {stats['count']} trades, {stats['win_rate']*100:.1f}% win rate, "
                      f"{stats['avg_return']:.2f}% avg return")

def create_backtest_visualizations(backtest_results: Dict[str, Any], output_dir: str) -> None:
    """
    Create and save visualizations for backtest results.
    
    Args:
        backtest_results: Dictionary with backtest results
        output_dir: Directory to save visualizations
    """
    logger.info(f"Creating backtest visualizations")
    
    try:
        visualizer = BacktestVisualizer(output_dir=output_dir)
        backtest_dir = visualizer.save_all_visualizations(backtest_results)
        logger.info(f"Backtest visualizations saved to: {backtest_dir}")
    except Exception as e:
        logger.error(f"Error creating backtest visualizations: {e}")

def create_exit_reason_analysis(backtest_results: Dict[str, Any], output_dir: str) -> None:
    """
    Create detailed exit reason analysis for flip strategy.
    
    Args:
        backtest_results: Dictionary with backtest results
        output_dir: Directory to save analysis
    """
    if 'trades' not in backtest_results:
        logger.warning("No trades found in backtest results for exit reason analysis")
        return
    
    trades = backtest_results['trades']
    if not trades:
        logger.warning("Empty trades list in backtest results for exit reason analysis")
        return
    
    # Count exit reasons
    exit_reasons = {}
    for trade in trades:
        reason = trade.get('exit_reason', 'unknown')
        if reason not in exit_reasons:
            exit_reasons[reason] = {
                'count': 0,
                'total_return': 0,
                'positive_returns': 0,
                'negative_returns': 0,
                'avg_duration': 0,
                'total_duration': 0
            }
        
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['total_return'] += trade.get('return_pct', 0)
        
        if trade.get('return_pct', 0) > 0:
            exit_reasons[reason]['positive_returns'] += 1
        else:
            exit_reasons[reason]['negative_returns'] += 1
            
        duration = trade.get('trade_duration_min', 0)
        exit_reasons[reason]['total_duration'] += duration
    
    # Calculate averages
    for reason in exit_reasons:
        count = exit_reasons[reason]['count']
        if count > 0:
            exit_reasons[reason]['avg_return'] = exit_reasons[reason]['total_return'] / count
            exit_reasons[reason]['win_rate'] = exit_reasons[reason]['positive_returns'] / count * 100
            exit_reasons[reason]['avg_duration'] = exit_reasons[reason]['total_duration'] / count
    
    # Prepare CSV data
    csv_data = [
        ["Exit Reason", "Count", "Percentage", "Total Return", "Avg Return", "Win Rate", "Avg Duration (min)"]
    ]
    
    total_trades = backtest_results.get('total_trades', 0)
    
    for reason, stats in exit_reasons.items():
        percentage = stats['count'] / total_trades * 100 if total_trades > 0 else 0
        csv_data.append([
            reason,
            stats['count'],
            f"{percentage:.2f}%",
            f"{stats['total_return']:.2f}%",
            f"{stats.get('avg_return', 0):.2f}%",
            f"{stats.get('win_rate', 0):.2f}%",
            f"{stats.get('avg_duration', 0):.2f}"
        ])
    
    # Create direction analysis (long vs short)
    direction_analysis = {
        'long': {'count': 0, 'total_return': 0},
        'short': {'count': 0, 'total_return': 0}
    }
    
    for trade in trades:
        direction = trade.get('direction', 'unknown')
        if direction in direction_analysis:
            direction_analysis[direction]['count'] += 1
            direction_analysis[direction]['total_return'] += trade.get('return_pct', 0)
    
    # Calculate averages for directions
    for direction in direction_analysis:
        count = direction_analysis[direction]['count']
        if count > 0:
            direction_analysis[direction]['avg_return'] = direction_analysis[direction]['total_return'] / count
    
    # Add direction analysis to CSV
    csv_data.append([])  # Empty row as separator
    csv_data.append(["Direction", "Count", "Percentage", "Total Return", "Avg Return"])
    
    for direction, stats in direction_analysis.items():
        percentage = stats['count'] / total_trades * 100 if total_trades > 0 else 0
        csv_data.append([
            direction,
            stats['count'],
            f"{percentage:.2f}%",
            f"{stats['total_return']:.2f}%",
            f"{stats.get('avg_return', 0):.2f}%"
        ])
    
    # Add conflict flip analysis if applicable
    flip_trades = [trade for trade in trades if trade.get('exit_reason') == 'conflict_flip']
    if flip_trades:
        csv_data.append([])  # Empty row as separator
        csv_data.append(["Conflict Flip Analysis"])
        csv_data.append(["Total Flips", len(flip_trades)])
        csv_data.append(["Average Time to Flip (min)", sum(t.get('trade_duration_min', 0) for t in flip_trades) / len(flip_trades)])
        
        # Analyze percentage of trades that were flipped
        csv_data.append(["Percentage of Trades Flipped", f"{len(flip_trades)/total_trades*100:.2f}%"])
    
    # Write to CSV
    output_file = os.path.join(output_dir, "exit_reason_analysis.csv")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    logger.info(f"Exit reason analysis saved to {output_file}")
    
    # Create a summary file specifically for flips
    if flip_trades:
        flip_file = os.path.join(output_dir, "flip_trades_details.csv")
        
        with open(flip_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Entry Time", "Exit Time", "Duration (min)", "Direction", 
                "Entry Price", "Exit Price", "Return (%)"
            ])
            
            for trade in flip_trades:
                writer.writerow([
                    trade.get('entry_time', ''),
                    trade.get('exit_time', ''),
                    trade.get('trade_duration_min', 0),
                    trade.get('direction', ''),
                    trade.get('entry_price', 0),
                    trade.get('exit_price', 0),
                    f"{trade.get('return_pct', 0):.2f}%"
                ])
        
        logger.info(f"Flip trades details saved to {flip_file}")

def calculate_buyhold_comparison(ticker: str, backtest_results: Dict[str, Any]) -> None:
    """
    Calculate buy & hold comparison for the backtesting period and add to results.
    
    Args:
        ticker: Stock ticker symbol
        backtest_results: Dictionary with backtest results to be updated
    """
    if not backtest_results or 'trades' not in backtest_results or not backtest_results['trades']:
        logger.warning(f"No trades data available for {ticker} buy & hold comparison")
        return
    
    trades = backtest_results['trades']
    
    try:
        # Extract earliest entry and latest exit times
        all_entry_times = [pd.to_datetime(trade['entry_time']) for trade in trades]
        all_exit_times = [pd.to_datetime(trade['exit_time']) for trade in trades]
        
        first_entry = min(all_entry_times)
        last_exit = max(all_exit_times)
        
        # Fetch stock price data covering the entire period
        logger.info(f"Fetching buy & hold data for {ticker} from {first_entry} to {last_exit}")
        
        # Add a buffer for data fetching
        buffer = timedelta(minutes=5)
        buy_hold_data = fetch_stock_prices(ticker, first_entry - buffer, last_exit + buffer)
        
        if buy_hold_data.empty:
            logger.warning(f"Could not fetch buy & hold data for {ticker}")
            return
            
        # Find the first valid price at or after the first trade entry
        first_price_mask = buy_hold_data.index >= first_entry
        if not first_price_mask.any():
            logger.warning(f"No valid starting price found for buy & hold comparison for {ticker}")
            return
            
        # Get first price and last price
        first_price_idx = buy_hold_data[first_price_mask].index[0]
        first_price = buy_hold_data.loc[first_price_idx, 'open']
        
        last_price_mask = buy_hold_data.index <= last_exit
        if not last_price_mask.any():
            logger.warning(f"No valid ending price found for buy & hold comparison for {ticker}")
            return
            
        last_price_idx = buy_hold_data[last_price_mask].index[-1]
        last_price = buy_hold_data.loc[last_price_idx, 'close']
        
        # Calculate buy & hold return
        buyhold_return_pct = (last_price / first_price - 1) * 100
        
        # Calculate outperformance
        strategy_return_pct = backtest_results.get('total_return_pct', 0)
        outperformance_pct = strategy_return_pct - buyhold_return_pct
        
        # Add to backtest results
        backtest_results['buy_hold_comparison'] = {
            'start_date': first_entry.strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': last_exit.strftime('%Y-%m-%d %H:%M:%S'),
            'start_price': float(first_price),
            'end_price': float(last_price),
            'buyhold_return_pct': float(buyhold_return_pct),
            'strategy_return_pct': float(strategy_return_pct),
            'outperformance_pct': float(outperformance_pct)
        }
        
        logger.info(f"Buy & Hold comparison for {ticker}:")
        logger.info(f"  Period: {first_entry.strftime('%Y-%m-%d %H:%M:%S')} to {last_exit.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Buy & Hold return: {buyhold_return_pct:.2f}%")
        logger.info(f"  Strategy return: {strategy_return_pct:.2f}%")
        logger.info(f"  Outperformance: {outperformance_pct:.2f}%")
        
    except Exception as e:
        logger.error(f"Error calculating buy & hold comparison for {ticker}: {e}", exc_info=True) 