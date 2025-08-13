"""
Main entry point for the Financial News Sentiment Analysis & Trading Strategy Backtester.

This script coordinates the workflow by:
1. Getting user input
2. Dispatching to appropriate pipeline modules
3. Handling high-level program flow
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import time
import shutil
import json
import numpy as np
import pandas as pd # Assuming pandas is used for DataFrames
from scipy import stats # For T-Test
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root directory to sys.path for both direct execution and module imports
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from src.news.processing.pipeline import run_sentiment_analysis_pipeline, PipelineError, NoNewsDataError, fetch_news
from src.news.data.tickers import get_dow_tickers, get_mag7_tickers, get_custom_tickers, get_top5_tickers
from src.llm.helpers.utilities import create_timestamped_output_dir
from src.news.helpers.utilities import logger
from src.backtesting.execution.pipeline import run_backtest_pipeline
from src.config import (
    API_CONFIG, 
    NEWS_CONFIG, 
    BACKTEST_CONFIG, 
    OUTPUT_CONFIG,
    BATCH_CONFIG,
    EVENT_STUDY_CONFIG
)
from src.backtesting.analysis.backtest_visualizer import BacktestVisualizer
from src.analysis.event_study import run_event_study, calculate_ar_for_news_events
from src.backtesting.helpers.market_data_cache import clear_market_data_cache
from src.news.fetcher.news_fetcher import filter_tracker
from src.analysis.bootstrap import run_bootstrap_on_backtest
from src.analysis.random_benchmark import run_random_benchmark_analysis
from src.analysis.portfolio_random_benchmark import run_portfolio_random_benchmark_analysis
from src.analysis.portfolio_event_study_analyzer import calculate_and_save_portfolio_caar_significance

def save_json(data: Any, file_path: str) -> None:
    """Helper function to save data to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.debug(f"Successfully saved JSON data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")

def clean_output_folder(output_dir: str):
    """
    Placeholder function to clean output folder.
    Currently just a pass-through function.
    """
    # Simply pass as this function isn't needed anymore but is still called
    pass

# --- Multi-Ticker Processing --- 
def process_multiple_tickers_main(ticker_list: List[str], output_dir: str, **kwargs):
    """Processes multiple tickers by calling the sentiment analysis pipeline for each."""
    if not ticker_list:
        logger.error("No ticker list provided for multi-ticker processing.")
        return []
        
    is_dow_index = kwargs.get("is_dow_index", False)
    is_mag7_index = kwargs.get("is_mag7_index", False)
    is_log_index = kwargs.get("is_log_index", False)
    is_top5_index = kwargs.get("is_top5_index", False)
    index_name = kwargs.get("index_name", "INDEX")
    
    logger.info(f"Starting multi-ticker processing for {len(ticker_list)} tickers.")
    results = []
    total_tickers = len(ticker_list)
    
    successful_sentiment = 0
    successful_backtest = 0
    successful_event_study = 0
    failed_tickers = []
    
    for i, ticker in enumerate(ticker_list):
        print("\n" + "="*80)
        print(f" PROCESSING TICKER {i+1}/{total_tickers}: {ticker} ".center(80, "="))
        print("="*80)
        
        logger.info("="*60)
        logger.info(f"Processing Ticker {i+1}/{total_tickers}: {ticker}")
        logger.info("="*60)
        
        # Reset filter tracker for this ticker
        filter_tracker.reset()
        
        logger.info(f"Processing ticker: {ticker}")
        
        # Create ticker-specific output directory
        ticker_output_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_output_dir, exist_ok=True)
        
        pipeline_params = kwargs.copy()
        pipeline_params["output_dir"] = ticker_output_dir
        pipeline_params["ticker"] = ticker
        
        ticker_result = {"ticker": ticker, "status": "failed", "sentiment_success": False, "backtest_success": False, "event_study_success": False}
        
        try:
            print(f"\n--- Starting Sentiment Analysis for {ticker} ---")
            
            start_date = None
            end_date = None
            if pipeline_params.get("start_date"):
                start_date = datetime.strptime(pipeline_params["start_date"], '%Y-%m-%d') if isinstance(pipeline_params["start_date"], str) else pipeline_params["start_date"]
            if pipeline_params.get("end_date"):
                end_date = datetime.strptime(pipeline_params["end_date"], '%Y-%m-%d') if isinstance(pipeline_params["end_date"], str) else pipeline_params["end_date"]
            
            sentiment_results = run_sentiment_analysis_pipeline(
                ticker=ticker,
                max_articles=pipeline_params.get("max_articles", 5000),
                output_dir=ticker_output_dir,
                start_date=start_date,
                end_date=end_date
            )
            
            if sentiment_results:
                ticker_result["sentiment_success"] = True
                successful_sentiment += 1
                print(f"✓ Sentiment Analysis for {ticker} completed successfully with {len(sentiment_results)} articles")
                
                print(f"\n--- Starting Backtesting for {ticker} ---")
                
                backtest_results = run_backtest_pipeline(
                    ticker=ticker,
                    sentiment_data=sentiment_results,
                    entry_delay=pipeline_params.get("entry_delay", BACKTEST_CONFIG["default_entry_delay"]),
                    holding_period=pipeline_params.get("holding_period", BACKTEST_CONFIG["default_holding_period"]),
                    use_stop_loss=pipeline_params.get("use_stop_loss", BACKTEST_CONFIG["default_use_stop_loss"]),
                    stop_loss_pct=pipeline_params.get("stop_loss_pct", BACKTEST_CONFIG["default_stop_loss_pct"]),
                    take_profit_pct=pipeline_params.get("take_profit_pct", BACKTEST_CONFIG["default_take_profit_pct"]),
                    include_after_hours=pipeline_params.get("include_after_hours", BACKTEST_CONFIG["default_include_after_hours"]),
                    max_workers=pipeline_params.get("max_workers", BACKTEST_CONFIG["default_max_workers"]),
                    use_flip_strategy=pipeline_params.get("use_flip_strategy", BACKTEST_CONFIG["default_use_flip_strategy"]),
                    output_dir=ticker_output_dir
                )
                
                if backtest_results and "total_trades" in backtest_results:
                    ticker_result["backtest_success"] = True
                    successful_backtest += 1
                    print(f"✓ Backtesting for {ticker} completed successfully with {backtest_results['total_trades']} trades")
                    
                    # Log final valid news and trades executed for filter tracking
                    filter_tracker.log_final_valid_news(len(sentiment_results))
                    filter_tracker.log_trades_executed(backtest_results['total_trades'])
                    
                    # Save filter statistics for this ticker
                    save_news_filter_statistics(ticker_output_dir, ticker)
                    
                    # Bootstrap analysis
                    try:
                        from src.analysis.bootstrap import run_bootstrap_on_backtest
                        
                        print("\n" + "="*80)
                        print(f" BOOTSTRAP ANALYSIS FOR {ticker} ".center(80, "="))
                        print("="*80 + "\n")
                        
                        logger.info(f"Running bootstrap tests on trading strategy results for {ticker}")
                        print(f"Running Bootstrap Analysis with 10000 simulations...")
                        
                        # Create statistical tests directory
                        stats_dir = os.path.join(ticker_output_dir, "statistical_tests")
                        os.makedirs(stats_dir, exist_ok=True)
                        
                        # Set backtest file path
                        backtest_file = os.path.join(ticker_output_dir, f"backtest_results_{ticker}.json")
                        
                        # Run bootstrap test
                        bootstrap_results = run_bootstrap_on_backtest(
                            backtest_file=backtest_file,
                            output_dir=stats_dir,
                            num_simulations=pipeline_params.get("bootstrap_simulations", 10000),
                            significance_level=pipeline_params.get("significance_level", 0.05)
                        )
                        
                        # Show key results
                        if bootstrap_results and "bootstrap_results" in bootstrap_results:
                            trade_p_value = bootstrap_results["bootstrap_results"]["trade"]["p_value"]
                            significant = bootstrap_results["bootstrap_results"]["trade"]["significant"]
                            significance_str = "SIGNIFICANT" if significant else "NOT SIGNIFICANT"
                            
                            print(f"Bootstrap analysis completed")
                            print(f"\nTrading strategy p-value: {trade_p_value:.4f}")
                            print(f"Significance: {significance_str}")
                            print(f"\nBootstrap results saved to {stats_dir}")
                            
                            logger.info(f"Bootstrap analysis completed for {ticker} - p-value: {trade_p_value:.4f}, significant: {significant}")
                        else:
                            logger.warning(f"Bootstrap analysis did not return expected results structure for {ticker}")
                    except Exception as e:
                        logger.error(f"Error running bootstrap tests for {ticker}: {e}", exc_info=True)
                        print(f"\nError in bootstrap analysis for {ticker}: {e}")
                    
                    # Run Random Trading Benchmark Analysis
                    try:
                        print(f"\n--- Starting Random Trading Benchmark for {ticker} ---")
                        logger.info(f"Running random trading benchmark for {ticker}")
                        
                        # Get price data for random benchmark
                        trades = backtest_results.get('trades', [])
                        if trades:
                            from src.backtesting.helpers.market_data import fetch_stock_prices
                            
                            # Get date range from trades
                            start_date = min(pd.to_datetime(trade['entry_time']) for trade in trades) - timedelta(days=1)
                            end_date = max(pd.to_datetime(trade['exit_time']) for trade in trades) + timedelta(days=1)
                            
                            price_data = fetch_stock_prices(ticker, start_date, end_date)
                            
                            if not price_data.empty:
                                random_benchmark_results = run_random_benchmark_analysis(
                                    ticker=ticker,
                                    sentiment_trades=trades,
                                    strategy_results=backtest_results,
                                    price_data=price_data,
                                    output_dir=ticker_output_dir
                                )
                                
                                if random_benchmark_results and 'strategy_vs_random_comparison' in random_benchmark_results:
                                    comparison = random_benchmark_results['strategy_vs_random_comparison']
                                    percentile = comparison['strategy_percentile_rank']
                                    p_value = comparison['statistical_significance']['p_value']
                                    significant = comparison['statistical_significance']['significant_at_5pct']
                                    
                                    print(f"Random benchmark analysis completed")
                                    print(f"Strategy Performance: {percentile:.1f}th percentile vs random")
                                    print(f"P-value: {p_value:.4f} ({'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'})")
                                    print(f"Results saved to {ticker_output_dir}")
                                    
                                    logger.info(f"Random benchmark completed for {ticker} - percentile: {percentile:.1f}%, p-value: {p_value:.4f}, significant: {significant}")
                                else:
                                    logger.warning(f"Random benchmark analysis did not return expected results for {ticker}")
                            else:
                                logger.warning(f"Could not fetch price data for random benchmark for {ticker}")
                                print(f"Could not fetch price data for random benchmark analysis")
                        else:
                            logger.warning(f"No trades available for random benchmark for {ticker}")
                            print(f"No trades available for random benchmark analysis")
                            
                    except Exception as e:
                        logger.error(f"Error running random benchmark for {ticker}: {e}", exc_info=True)
                        print(f"\nError in random benchmark analysis for {ticker}: {e}")
                    
                    # Event study analysis
                    print(f"\n--- Starting Event Study Analysis for {ticker} ---")
                    
                    # Check if event study analysis should be run
                    enable_event_study = pipeline_params.get("enable_event_study", True)
                    logger.info(f"Event Study enabled for {ticker}: {enable_event_study}")
                    print(f"Event Study enabled for {ticker}: {enable_event_study}")
                    
                    if enable_event_study:
                        # Create event study output directory
                        event_study_dir = os.path.join(ticker_output_dir, "event_study")
                        os.makedirs(event_study_dir, exist_ok=True)
                        
                        try:
                            # Debug logging for sentiment data
                            logger.info(f"Sentiment data for Event Study: {len(sentiment_results) if sentiment_results else 0} articles")
                            if sentiment_results and len(sentiment_results) > 0:
                                sample_item = sentiment_results[0]
                                logger.info(f"Sample sentiment item structure: {list(sample_item.keys()) if hasattr(sample_item, 'keys') else type(sample_item)}")
                            
                            # Run event study analysis
                            event_study_results = run_event_study(
                                ticker=ticker,
                                sentiment_data=sentiment_results,
                                market_index=pipeline_params.get("market_index", "SPY"),
                                event_windows=[5, 15, 30, 60],
                                output_dir=event_study_dir,
                                use_cache=True
                            )
                            
                            if event_study_results:
                                ticker_result["event_study_success"] = True
                                successful_event_study += 1
                                print(f"✓ Event Study Analysis for {ticker} completed successfully with {event_study_results.get('unique_news_events', event_study_results.get('total_events', 0))} unique news events")
                                
                                # Run statistical tests if enabled
                                if pipeline_params.get("run_statistical_tests", False):
                                    print(f"\n--- Starting Statistical Tests for {ticker} ---")
                                    
                                    # Create statistical tests output directory
                                    stats_dir = os.path.join(ticker_output_dir, "statistical_tests")
                                    os.makedirs(stats_dir, exist_ok=True)
                                    
                                    try:
                                        # Run t-tests on event study results
                                        from src.analysis.ttest import run_ttest_on_event_study
                                        
                                        event_study_file = os.path.join(event_study_dir, f"{ticker}_event_study_results.json")
                                        
                                        run_ttest_on_event_study(
                                            event_study_file=event_study_file,
                                            output_dir=stats_dir,
                                            significance_level=pipeline_params.get("significance_level", 0.05)
                                        )
                                        
                                        ticker_result["statistical_tests_success"] = True
                                        print(f"✓ Statistical Tests for {ticker} completed successfully")
                                    except Exception as e:
                                        logger.error(f"Statistical Tests error for {ticker}: {e}", exc_info=True)
                                        print(f"✗ Statistical Tests for {ticker} failed: {str(e)}")
                            else:
                                print(f"✗ Event Study Analysis for {ticker} failed: No results returned")
                        except Exception as e:
                            logger.error(f"Event Study error for {ticker}: {e}", exc_info=True)
                            print(f"✗ Event Study Analysis for {ticker} failed: {str(e)}")
                    else:
                        print(f"ℹ Event Study Analysis for {ticker} skipped (disabled in config)")
                else:
                    print(f"✗ Backtesting for {ticker} failed: No valid results returned")
                    # Still save filter statistics even if backtest failed
                    filter_tracker.log_final_valid_news(len(sentiment_results))
                    filter_tracker.log_trades_executed(0)
                    save_news_filter_statistics(ticker_output_dir, ticker)
                
            else:
                print(f"✗ Sentiment Analysis for {ticker} failed: No sentiment data returned")
                # Save filter statistics even if sentiment analysis failed
                filter_tracker.log_final_valid_news(0)
                filter_tracker.log_trades_executed(0)
                save_news_filter_statistics(ticker_output_dir, ticker)
            
            ticker_result["status"] = "success"
            
        except NoNewsDataError as e:
            logger.warning(f"No news data found for {ticker}: {e}")
            print(f"✗ Sentiment Analysis for {ticker} failed: No news data found - {e}")
            ticker_result["status"] = "no_news"
            failed_tickers.append(ticker)
            
        except PipelineError as e:
            logger.error(f"Pipeline processing error for {ticker}: {e}", exc_info=True)
            print(f"✗ Processing pipeline for {ticker} failed: {str(e)}")
            ticker_result["status"] = "pipeline_error"
            failed_tickers.append(ticker)
                
        except Exception as e:
            logger.error(f"Unhandled error processing {ticker}: {e}", exc_info=True)
            print(f"✗ Processing {ticker} failed with an unhandled error: {str(e)}")
            ticker_result["status"] = "error"
            failed_tickers.append(ticker)
            
        results.append(ticker_result)
        
        # Print separator after each ticker
        print("\n" + "-"*80)
    
    # Print final summary
    print("\n" + "="*80)
    print(" MULTI-TICKER PROCESSING SUMMARY ".center(80, "="))
    print("="*80)
    print(f"Total Tickers Processed: {total_tickers}")
    print(f"Successful Sentiment Analysis: {successful_sentiment}/{total_tickers}")
    print(f"Successful Backtesting: {successful_backtest}/{total_tickers}")
    print(f"Successful Event Study: {successful_event_study}/{total_tickers}")
    
    if failed_tickers:
        print(f"\nFailed Tickers ({len(failed_tickers)}):")
        for ticker in failed_tickers:
            print(f"  - {ticker}")
    
    print("="*80 + "\n")
    
    # Run consolidated analysis to generate index-level reports and charts 
    logger.info(f"All individual tickers processed. Running comprehensive portfolio analysis for {index_name}...")
    print(f"\nAll individual tickers processed. Running comprehensive portfolio analysis for {index_name}...")
    
    # Use the enhanced portfolio report function instead of the basic generate_index_analysis
    fix_portfolio_report(ticker_list, output_dir, index_name)
    
    clean_output_folder(output_dir)
    return results

def generate_index_analysis(output_dir: str, ticker_list: List[str], index_name: str, **kwargs):
    """Generate accumulated analysis for the entire index (DOW Jones or MAG 7)."""
    logger.info(f"Starting index analysis generation for {index_name}")
    index_dir = os.path.join(output_dir, f"{index_name}_INDEX_ANALYSIS")
    os.makedirs(index_dir, exist_ok=True)

    all_ticker_summaries = []
    all_event_study_results = []
    all_trades = []
    all_bootstrap_results = []  # Neu: Sammle alle Bootstrap-Ergebnisse
    successful_tickers = 0
    failed_tickers = []

    # Accumulators for index-level averages
    total_r_squared = 0
    total_adj_r_squared = 0
    total_rmse = 0
    total_mae = 0
    total_market_model_estimations = 0 # Count successful estimations across all tickers

    total_buyhold_return = 0
    total_strategy_return = 0
    total_outperformance = 0
    buyhold_comparison = []
    total_win_rate = 0
    total_trades_overall = 0
    
    # Neu: Für AAR/CAAR Konsolidierung
    all_aar_data = {}
    all_caar_data = {}
    all_sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    all_ttest_results = {}  # Neu: Für konsolidierte T-Test-Ergebnisse
    
    # Dictionary für Fenstergrößen initialisieren
    for window in EVENT_STUDY_CONFIG.get("event_windows", [5, 15, 30, 60]):
        all_aar_data[window] = []
        all_caar_data[window] = []
        all_ttest_results[window] = {"t_statistic": 0, "p_value": 0, "significant": False, "count": 0}

    for ticker in ticker_list:
        ticker_dir = os.path.join(output_dir, ticker)
        backtest_file = os.path.join(ticker_dir, f'backtest_results_{ticker}.json')
        event_study_file = os.path.join(ticker_dir, "event_study", f"{ticker}_event_study_results.json")
        bootstrap_file = os.path.join(ticker_dir, "statistical_tests", "bootstrap_results.json")  # Neu: Bootstrap-Ergebnisdatei

        ticker_data_found = False
        try:
            # Load backtest results
            if os.path.exists(backtest_file):
                with open(backtest_file, 'r') as f:
                    backtest_results = json.load(f)
                
                # Skip if no trades
                if backtest_results.get('total_trades', 0) > 0:
                    ticker_data_found = True
                    # Collect summary data
                    ticker_summary = {
                        'ticker': ticker,
                        'total_trades': backtest_results.get('total_trades', 0),
                        'winning_trades': backtest_results.get('winning_trades', 0),
                        'win_rate': backtest_results.get('win_rate', 0),
                        'total_return_pct': backtest_results.get('total_return_pct', 0),
                        'average_return_pct': backtest_results.get('average_return_pct', 0),
                        'sharpe_ratio': backtest_results.get('sharpe_ratio', np.nan),
                        'max_drawdown_pct': backtest_results.get('max_drawdown_pct', np.nan),
                        'profit_factor': backtest_results.get('profit_factor', np.nan)
                    }

                    # Accumulate metrics
                    total_strategy_return += ticker_summary['total_return_pct']
                    total_win_rate += ticker_summary['win_rate']
                    total_trades_overall += ticker_summary['total_trades']

                    # Extract Buy & Hold return if available
                    if 'buy_hold_comparison' in backtest_results:
                        buyhold_return = backtest_results['buy_hold_comparison'].get('buyhold_return_pct', 0)
                        outperformance = backtest_results['buy_hold_comparison'].get('outperformance_pct', 0)
                        ticker_summary['buyhold_return_pct'] = buyhold_return
                        ticker_summary['outperformance_pct'] = outperformance
                        total_buyhold_return += buyhold_return
                        total_outperformance += outperformance
                        
                        buyhold_comparison.append({
                            'ticker': ticker,
                            'strategy_return': backtest_results.get('total_return_pct', 0),
                            'buyhold_return': buyhold_return,
                            'outperformance': outperformance
                        })
                    
                    all_ticker_summaries.append(ticker_summary)
                    all_trades.extend(backtest_results.get('trades', []))
            
            # Load event study results
            if os.path.exists(event_study_file):
                with open(event_study_file, 'r') as f:
                    event_results = json.load(f)
                all_event_study_results.append(event_results)
                ticker_data_found = True
                
                # Neu: Extrahiere AAR/CAAR-Daten für die Konsolidierung
                if 'sentiment_distribution' in event_results:
                    all_sentiment_counts["positive"] += event_results['sentiment_distribution'].get('positive', 0)
                    all_sentiment_counts["negative"] += event_results['sentiment_distribution'].get('negative', 0)
                    all_sentiment_counts["neutral"] += event_results['sentiment_distribution'].get('neutral', 0)
                
                # AAR und CAAR Daten aus Events für alle Fenstergrößen sammeln
                for window_size in all_aar_data.keys():
                    window_key = f"window_{window_size}"
                    if window_key in event_results and 'aar_data' in event_results[window_key]:
                        all_aar_data[window_size].append(event_results[window_key]['aar_data'])
                    if window_key in event_results and 'caar_data' in event_results[window_key]:
                        all_caar_data[window_size].append(event_results[window_key]['caar_data'])
                
                # Accumulate market model metrics if available and valid
                if 'market_model_stats' in event_results and 'overall' in event_results['market_model_stats']:
                    overall_stats = event_results['market_model_stats']['overall']
                    count = overall_stats.get('count', 0)
                    if count > 0:
                        r_sq = overall_stats.get('avg_r_squared')
                        adj_r_sq = overall_stats.get('avg_adj_r_squared')
                        rmse = overall_stats.get('avg_rmse')
                        mae = overall_stats.get('avg_mae')
                        
                        if r_sq is not None and not np.isnan(r_sq): total_r_squared += r_sq
                        if adj_r_sq is not None and not np.isnan(adj_r_sq): total_adj_r_squared += adj_r_sq
                        if rmse is not None and not np.isnan(rmse): total_rmse += rmse
                        if mae is not None and not np.isnan(mae): total_mae += mae
                        total_market_model_estimations += 1 # Count tickers with successful overall estimation
                
                # Neu: Lade T-Test-Ergebnisse falls vorhanden
                ttest_file = os.path.join(ticker_dir, "event_study", "statistical_tests", "ttest_results.json")
                if os.path.exists(ttest_file):
                    with open(ttest_file, 'r') as f:
                        ttest_results = json.load(f)
                    
                    # Konsolidiere T-Test-Ergebnisse
                    for window_size in all_ttest_results.keys():
                        window_key = f"window_{window_size}"
                        if window_key in ttest_results:
                            window_result = ttest_results[window_key]
                            if not np.isnan(window_result.get('t_statistic', np.nan)):
                                all_ttest_results[window_size]["t_statistic"] += window_result.get('t_statistic', 0)
                                all_ttest_results[window_size]["p_value"] += window_result.get('p_value', 0)
                                all_ttest_results[window_size]["count"] += 1
                                if window_result.get('significant', False):
                                    all_ttest_results[window_size]["significant_count"] = all_ttest_results[window_size].get("significant_count", 0) + 1
            
            # Neu: Lade Bootstrap-Ergebnisse falls vorhanden
            if os.path.exists(bootstrap_file):
                with open(bootstrap_file, 'r') as f:
                    bootstrap_results = json.load(f)
                
                # Sammle nur die Bootstrap-Ergebnisse mit gültigen p-Werten
                if "bootstrap_results" in bootstrap_results and "trade" in bootstrap_results["bootstrap_results"]:
                    trade_result = bootstrap_results["bootstrap_results"]["trade"]
                    if "p_value" in trade_result and not np.isnan(trade_result["p_value"]):
                        bootstrap_entry = {
                            "ticker": ticker,
                            "p_value": trade_result["p_value"],
                            "significant": trade_result.get("significant", False),
                            "observed_value": trade_result.get("observed_value", np.nan),
                            "confidence_interval": trade_result.get("confidence_interval", {})
                        }
                        all_bootstrap_results.append(bootstrap_entry)

            if ticker_data_found:
                successful_tickers += 1
            else:
                failed_tickers.append(ticker)
                logger.warning(f"No usable results found for {ticker} in {ticker_dir}")

        except Exception as e:
            logger.error(f"Error processing data for ticker {ticker}: {e}", exc_info=True)
            failed_tickers.append(ticker)

    if successful_tickers == 0:
        logger.error(f"No successful tickers found for {index_name} analysis. Aborting.")
        return

    logger.info(f"Processed {successful_tickers}/{len(ticker_list)} tickers successfully.")
    if failed_tickers:
        logger.warning(f"Failed tickers: {failed_tickers}")

    # Create DataFrame from summaries
    ticker_summaries_df = pd.DataFrame(all_ticker_summaries)
    if not ticker_summaries_df.empty:
        ticker_summaries_df.to_csv(os.path.join(index_dir, f'{index_name}_ticker_summary.csv'), index=False)

    # Calculate overall averages
    avg_strategy_return = total_strategy_return / successful_tickers
    avg_buyhold_return = total_buyhold_return / successful_tickers if successful_tickers > 0 else 0
    avg_outperformance = total_outperformance / successful_tickers if successful_tickers > 0 else 0
    avg_win_rate = total_win_rate / successful_tickers if successful_tickers > 0 else 0

    # Calculate average market model stats
    avg_r_squared_index = total_r_squared / total_market_model_estimations if total_market_model_estimations > 0 else np.nan
    avg_adj_r_squared_index = total_adj_r_squared / total_market_model_estimations if total_market_model_estimations > 0 else np.nan
    avg_rmse_index = total_rmse / total_market_model_estimations if total_market_model_estimations > 0 else np.nan
    avg_mae_index = total_mae / total_market_model_estimations if total_market_model_estimations > 0 else np.nan
    
    # Neu: Verarbeite konsolidierte AAR/CAAR-Daten
    consolidated_aar = {}
    consolidated_caar = {}
    
    for window_size, aar_data_list in all_aar_data.items():
        if aar_data_list:
            # Durchschnittliche AAR für alle Aktien pro Tag im Event-Fenster
            consolidated_aar[window_size] = {}
            days_count = {}
            
            # Sammle alle AAR-Werte für jeden Tag
            for aar_data in aar_data_list:
                for day, value in aar_data.items():
                    if day not in consolidated_aar[window_size]:
                        consolidated_aar[window_size][day] = 0
                        days_count[day] = 0
                    
                    if not np.isnan(value):
                        consolidated_aar[window_size][day] += value
                        days_count[day] += 1
            
            # Berechne Durchschnitt für jeden Tag
            for day in consolidated_aar[window_size]:
                if days_count[day] > 0:
                    consolidated_aar[window_size][day] /= days_count[day]
    
    # Ähnliche Verarbeitung für CAAR
    for window_size, caar_data_list in all_caar_data.items():
        if caar_data_list:
            consolidated_caar[window_size] = {}
            days_count = {}
            
            for caar_data in caar_data_list:
                for day, value in caar_data.items():
                    if day not in consolidated_caar[window_size]:
                        consolidated_caar[window_size][day] = 0
                        days_count[day] = 0
                    
                    if not np.isnan(value):
                        consolidated_caar[window_size][day] += value
                        days_count[day] += 1
            
            for day in consolidated_caar[window_size]:
                if days_count[day] > 0:
                    consolidated_caar[window_size][day] /= days_count[day]
    
    # Neu: Berechne durchschnittliche T-Test-Ergebnisse
    avg_ttest_results = {}
    for window_size, results in all_ttest_results.items():
        if results["count"] > 0:
            avg_ttest_results[window_size] = {
                "avg_t_statistic": results["t_statistic"] / results["count"],
                "avg_p_value": results["p_value"] / results["count"],
                "significant_rate": results.get("significant_count", 0) / results["count"],
                "count": results["count"]
            }
    
    # Neu: Berechne zusammengefasste Bootstrap-Statistiken
    bootstrap_summary = {}
    if all_bootstrap_results:
        p_values = [result["p_value"] for result in all_bootstrap_results]
        significant_count = sum(1 for result in all_bootstrap_results if result.get("significant", False))
        
        bootstrap_summary = {
            "count": len(all_bootstrap_results),
            "avg_p_value": sum(p_values) / len(p_values) if p_values else np.nan,
            "median_p_value": np.median(p_values) if p_values else np.nan,
            "significant_count": significant_count,
            "significant_percentage": (significant_count / len(all_bootstrap_results)) * 100 if all_bootstrap_results else 0
        }

    # Create overall summary dictionary
    index_summary = {
        'index': index_name,
        'total_tickers_processed': len(ticker_list),
        'successful_tickers': successful_tickers,
        'failed_tickers': failed_tickers,
        'average_strategy_return_pct': avg_strategy_return,
        'average_buyhold_return_pct': avg_buyhold_return,
        'average_outperformance_pct': avg_outperformance,
        'average_win_rate': avg_win_rate,
        'total_trades_overall': total_trades_overall,
        'sentiment_distribution': all_sentiment_counts,
        'average_market_model_metrics': {
            'avg_r_squared': float(avg_r_squared_index) if pd.notna(avg_r_squared_index) else None,
            'avg_adj_r_squared': float(avg_adj_r_squared_index) if pd.notna(avg_adj_r_squared_index) else None,
            'avg_rmse': float(avg_rmse_index) if pd.notna(avg_rmse_index) else None,
            'avg_mae': float(avg_mae_index) if pd.notna(avg_mae_index) else None,
            'tickers_with_successful_estimation': total_market_model_estimations
        },
        'consolidated_event_study': {
            'aar': consolidated_aar,
            'caar': consolidated_caar,
            'ttest_summary': avg_ttest_results
        },
        'bootstrap_summary': bootstrap_summary
    }

    # Save overall summary
    index_summary_file = os.path.join(index_dir, f'{index_name}_index_summary.json')
    try:
        with open(index_summary_file, 'w') as f:
            json.dump(index_summary, f, indent=4, default=lambda x: None if pd.isna(x) else x) # Handle potential NaN
        logger.info(f"Index summary saved to {index_summary_file}")
    except Exception as e:
        logger.error(f"Failed to save index summary JSON: {e}")
    
    # Speichere konsolidierte AAR/CAAR-Daten separat für einfacheren Zugriff
    try:
        with open(os.path.join(index_dir, f'{index_name}_consolidated_aar.json'), 'w') as f:
            json.dump(consolidated_aar, f, indent=4, default=lambda x: None if pd.isna(x) else x)
        with open(os.path.join(index_dir, f'{index_name}_consolidated_caar.json'), 'w') as f:
            json.dump(consolidated_caar, f, indent=4, default=lambda x: None if pd.isna(x) else x)
        logger.info(f"Consolidated AAR/CAAR data saved to {index_dir}")
    except Exception as e:
        logger.error(f"Failed to save consolidated AAR/CAAR data: {e}")
    
    # Speichere Bootstrap-Zusammenfassungsdaten
    if bootstrap_summary:
        try:
            with open(os.path.join(index_dir, f'{index_name}_bootstrap_summary.json'), 'w') as f:
                json.dump(bootstrap_summary, f, indent=4, default=lambda x: None if pd.isna(x) else x)
            logger.info(f"Bootstrap summary saved to {index_dir}")
        except Exception as e:
            logger.error(f"Failed to save bootstrap summary: {e}")

    # ---- Save Index Market Model KPIs ----
    index_market_model_kpis = index_summary.get('average_market_model_metrics', {})
    if index_market_model_kpis:
        kpi_file = os.path.join(index_dir, f'{index_name}_market_model_kpis.json')
        try:
            with open(kpi_file, 'w') as f:
                 json.dump(index_market_model_kpis, f, indent=4, default=lambda x: None if pd.isna(x) else x) # Handle potential NaN
            logger.info(f"Index market model KPIs saved to {kpi_file}")
        except Exception as e:
            logger.error(f"Failed to save index market model KPIs JSON: {e}")
    # ---- End Save KPIs ----

    # Print key summary stats
    logger.info(f"--- {index_name} Index Analysis Summary ---")
    logger.info(f"Successful Tickers: {successful_tickers}/{len(ticker_list)}")
    logger.info(f"Average Strategy Return: {avg_strategy_return:.2f}%" )
    logger.info(f"Average Buy & Hold Return: {avg_buyhold_return:.2f}%" )
    logger.info(f"Average Outperformance: {avg_outperformance:.2f}%" )
    logger.info(f"Average Win Rate: {avg_win_rate*100:.1f}%" )
    logger.info(f"Total Trades Across Index: {total_trades_overall}")
    logger.info(f"Market Model Quality (Avg across {total_market_model_estimations} tickers):")
    logger.info(f"  - Avg R²: {avg_r_squared_index:.4f}")
    logger.info(f"  - Avg Adj R²: {avg_adj_r_squared_index:.4f}")
    logger.info(f"  - Avg RMSE: {avg_rmse_index:.6f}")
    logger.info(f"  - Avg MAE: {avg_mae_index:.6f}")
    
    # Neu: Bootstrap-Statistiken ausgeben
    if bootstrap_summary:
        logger.info(f"Bootstrap Statistics:")
        logger.info(f"  - Tickers mit gültigen Bootstrap-Tests: {bootstrap_summary['count']}")
        logger.info(f"  - Durschnittlicher p-Wert: {bootstrap_summary['avg_p_value']:.4f}")
        logger.info(f"  - Anteil signifikanter Ergebnisse: {bootstrap_summary['significant_percentage']:.1f}%")
    
    # ---- Add Console Output for Index Metrics ----
    print("\n" + "-"*80)
    print(f" {index_name} INDEX ANALYSIS - KEY METRICS ".center(80, "-"))
    print("-"*80)
    print(f"Successfully Processed Tickers: {successful_tickers}/{len(ticker_list)}")
    print(f"Average Strategy Return: {avg_strategy_return:.2f}%")
    print(f"Average Buy & Hold Return: {avg_buyhold_return:.2f}%")
    print(f"Average Outperformance: {avg_outperformance:.2f}%")
    print(f"Average Win Rate: {avg_win_rate*100:.1f}%")
    print(f"Total Trades Across Index: {total_trades_overall}")
    print(f"\nSentiment Distribution:")
    total_sentiment = sum(all_sentiment_counts.values())
    if total_sentiment > 0:
        print(f"  - Positive: {all_sentiment_counts['positive']} ({all_sentiment_counts['positive']/total_sentiment*100:.1f}%)")
        print(f"  - Negative: {all_sentiment_counts['negative']} ({all_sentiment_counts['negative']/total_sentiment*100:.1f}%)")
        print(f"  - Neutral: {all_sentiment_counts['neutral']} ({all_sentiment_counts['neutral']/total_sentiment*100:.1f}%)")
    
    if total_market_model_estimations > 0:
        print(f"\nAverage Market Model Quality (across {total_market_model_estimations} tickers):")
        print(f"  - R²:         {avg_r_squared_index:.4f}")
        print(f"  - Adj. R²:    {avg_adj_r_squared_index:.4f}")
        print(f"  - RMSE:       {avg_rmse_index:.6f}")
        print(f"  - MAE:        {avg_mae_index:.6f}")
    else:
        print("\nAverage Market Model Quality: Not available (no successful estimations)")
    
    # Neu: Bootstrap-Zusammenfassung anzeigen
    if bootstrap_summary:
        print(f"\nBootstrap Significance Testing (across {bootstrap_summary['count']} tickers):")
        print(f"  - Avg p-value: {bootstrap_summary['avg_p_value']:.4f}")
        print(f"  - Significant Results: {bootstrap_summary['significant_count']}/{bootstrap_summary['count']} ({bootstrap_summary['significant_percentage']:.1f}%)")
    
    # Neu: T-Test Zusammenfassung anzeigen
    if avg_ttest_results:
        print(f"\nT-Test Summary for Event Study Windows:")
        for window_size, results in avg_ttest_results.items():
            if results["count"] > 0:
                print(f"  - Window {window_size}min: Avg t-stat: {results['avg_t_statistic']:.4f}, Avg p-value: {results['avg_p_value']:.4f}, Significance Rate: {results['significant_rate']*100:.1f}%")
    
    # Add Random Benchmark Summary
    if 'random_benchmark' in index_summary:
        rb_data = index_summary['random_benchmark']
        print(f"\nRandom Trading Benchmark Analysis:")
        print(f"  - Tickers Analyzed: {rb_data['tickers_with_benchmark']}")
        print(f"  - Average Strategy Percentile: {rb_data['average_strategy_percentile_rank']:.1f}th")
        print(f"  - Significant Tickers: {rb_data['tickers_significant_at_5pct']}/{rb_data['tickers_with_benchmark']} ({rb_data['percentage_significant_tickers']:.1f}%)")
        print(f"  - Interpretation: {rb_data['interpretation']}")
    
    print("-"*80)
    # ---- End Console Output ----

    # Run visualizations if data exists
    if not ticker_summaries_df.empty:
        try:
            visualizer = BacktestVisualizer()
            logger.info("Generating index performance visualizations...")
            
            # Combine all trades into a single dataframe for portfolio analysis
            all_trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
            
            # Visualize index-level performance
            visualizer.visualize_index_performance(
                index_summary=index_summary,
                ticker_summaries_df=ticker_summaries_df,
                all_trades_df=all_trades_df,
                buyhold_comparison=buyhold_comparison,
                index_dir=index_dir
            )
            
            # NEU: Visualize aggregated sentiment if event study results exist
            if all_event_study_results and consolidated_aar and consolidated_caar:
                logger.info("Generating aggregated event study visualization...")
                
                # Visualisiere konsolidierte AAR/CAAR-Daten für alle Event-Fenster
                for window_size in consolidated_aar.keys():
                    if window_size in consolidated_aar and window_size in consolidated_caar:
                        try:
                            # Konvertiere AAR/CAAR-Daten in DataFrame-Format für die Visualisierung
                            aar_days = sorted([int(day) for day in consolidated_aar[window_size].keys()])
                            aar_values = [consolidated_aar[window_size].get(str(day), np.nan) for day in aar_days]
                            caar_values = [consolidated_caar[window_size].get(str(day), np.nan) for day in aar_days]
                            
                            # Erzeuge Dataframe für die Visualisierung
                            event_df = pd.DataFrame({
                                'day': aar_days,
                                'aar': aar_values,
                                'caar': caar_values
                            })
                            
                            # Generiere AAR/CAAR-Plots
                            visualizer.visualize_consolidated_event_study(
                                event_df=event_df,
                                window_size=window_size,
                                index_name=index_name,
                                output_dir=index_dir
                            )
                            
                            logger.info(f"Generated visualization for {window_size}min event window")
                        except Exception as e:
                            logger.error(f"Error generating visualization for {window_size}min event window: {e}", exc_info=True)
            
            # Create KPI Dashboard with extended metrics
            visualizer.create_index_kpi_dashboard(
                index_summary=index_summary,
                ticker_summaries_df=ticker_summaries_df,
                index_dir=index_dir
            )
            
            logger.info("Index visualizations generated.")
        except Exception as e:
            logger.error(f"Error generating index visualizations: {e}", exc_info=True)

    logger.info(f"Index analysis generation for {index_name} completed.")
    
    # NEU: Portfolio-Bootstrap-Analyse am Ende hinzufügen
    try:
        from src.analysis.portfolio_bootstrap import run_portfolio_bootstrap_analysis
        
        print("\n" + "="*80)
        print(f" PORTFOLIO BOOTSTRAP ANALYSIS FOR {index_name} ".center(80, "="))
        print("="*80)
        
        # Check if consolidated trades file exists
        trades_file = os.path.join(index_dir, "trades", f"{index_name}_all_trades.csv")
        
        if os.path.exists(trades_file) and os.path.getsize(trades_file) > 0:
            logger.info(f"Running comprehensive portfolio bootstrap analysis for {index_name}")
            print(f"Running Portfolio Bootstrap Analysis with 10,000 simulations...")
            print(f"Analyzing consolidated trades from: {trades_file}")
            
            # Create portfolio bootstrap directory
            portfolio_bootstrap_dir = os.path.join(index_dir, "portfolio_bootstrap")
            
            # Run portfolio bootstrap analysis
            portfolio_results = run_portfolio_bootstrap_analysis(
                trades_file=trades_file,
                output_dir=portfolio_bootstrap_dir,
                num_simulations=kwargs.get("bootstrap_simulations", 10000)
            )
            
            # Display key results
            if portfolio_results and 'portfolio_metrics' in portfolio_results:
                metrics = portfolio_results['portfolio_metrics']
                bootstrap = portfolio_results['bootstrap_results']
                
                print(f"\n📊 PORTFOLIO PERFORMANCE METRICS:")
                print(f"   Total Trades: {metrics['total_trades']}")
                print(f"   Total Return: {metrics['total_return']:.4f}%")
                print(f"   Mean Return per Trade: {metrics['mean_return']:.4f}%")
                print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                print(f"   Sortino Ratio: {metrics['sortino_ratio']:.4f}")
                print(f"   Win Rate: {metrics['win_rate']:.2%}")
                print(f"   Max Drawdown: {metrics['max_drawdown']:.4f}%")
                print(f"   Profit Factor: {metrics['profit_factor']:.4f}")
                
                print(f"\n🔬 STATISTICAL SIGNIFICANCE (Bootstrap Tests):")
                print(f"   Mean Return - P-value: {bootstrap['mean_return']['p_value']:.4f} " +
                      f"({'✓ SIGNIFICANT' if bootstrap['mean_return']['significant'] else '✗ NOT SIGNIFICANT'})")
                print(f"   Sharpe Ratio - P-value: {bootstrap['sharpe_ratio']['p_value']:.4f} " +
                      f"({'✓ SIGNIFICANT' if bootstrap['sharpe_ratio']['significant'] else '✗ NOT SIGNIFICANT'})")
                print(f"   Sortino Ratio - P-value: {bootstrap['sortino_ratio']['p_value']:.4f} " +
                      f"({'✓ SIGNIFICANT' if bootstrap['sortino_ratio']['significant'] else '✗ NOT SIGNIFICANT'})")
                
                # Confidence intervals
                print(f"\n📈 95% CONFIDENCE INTERVALS:")
                print(f"   Mean Return: [{bootstrap['mean_return']['ci_lower']:.4f}%, {bootstrap['mean_return']['ci_upper']:.4f}%]")
                print(f"   Sharpe Ratio: [{bootstrap['sharpe_ratio']['ci_lower']:.4f}, {bootstrap['sharpe_ratio']['ci_upper']:.4f}]")
                
                # Drawdown analysis
                if 'drawdown_analysis' in portfolio_results:
                    dd = portfolio_results['drawdown_analysis']
                    print(f"\n📉 DRAWDOWN ANALYSIS:")
                    print(f"   Maximum Drawdown: {dd['max_drawdown']:.4f}%")
                    print(f"   Average Drawdown: {dd['avg_drawdown']:.4f}%")
                    print(f"   Drawdown Periods: {dd['num_drawdown_periods']}")
                    print(f"   Time in Drawdown: {dd['time_in_drawdown']:.2%}")
                
                # Sentiment performance
                if 'sentiment_analysis' in portfolio_results:
                    sentiment_data = portfolio_results['sentiment_analysis']
                    print(f"\n🎭 PERFORMANCE BY SENTIMENT:")
                    
                    for sentiment_type, data in sentiment_data.items():
                        metrics = data['portfolio_metrics']
                        bootstrap_sent = data['bootstrap_results']
                        
                        print(f"   {sentiment_type.upper()}:")
                        print(f"     Trades: {data['trade_count']}")
                        print(f"     Mean Return: {metrics['mean_return']:.4f}%")
                        print(f"     Win Rate: {metrics['win_rate']:.2%}")
                        print(f"     Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                        print(f"     P-value: {bootstrap_sent['mean_return']['p_value']:.4f} " +
                              f"({'✓ SIG' if bootstrap_sent['mean_return']['significant'] else '✗ NS'})")
                
                print(f"\n📁 Detailed results saved to: {portfolio_bootstrap_dir}")
                print(f"   - portfolio_bootstrap_analysis.json (detailed data)")
                print(f"   - portfolio_bootstrap_summary.txt (human-readable)")
                print(f"   - portfolio_analysis.png (performance charts)")
                print(f"   - bootstrap_analysis.png (statistical charts)")
                
        else:
            logger.warning(f"No consolidated trades file found at {trades_file} or file is empty. Skipping portfolio bootstrap analysis.")
            print(f"⚠️  No consolidated trades data found. Skipping portfolio bootstrap analysis.")
            
    except ImportError as e:
        logger.error(f"Portfolio bootstrap module not available: {e}")
        print(f"⚠️  Portfolio bootstrap analysis not available. Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"Error running portfolio bootstrap analysis: {e}", exc_info=True)
        print(f"❌ Error in portfolio bootstrap analysis: {e}")
    
    print("="*80)
    print(f"COMPREHENSIVE PORTFOLIO REPORT FOR {index_name} COMPLETED")
    print(f"Output directory: {index_analysis_dir}")
    print(f"{'='*80}\n")
    
    # Create consolidated filter statistics for the index
    print(f"Creating consolidated filter statistics for {index_name}...")
    create_index_filter_statistics(output_dir, ticker_list, index_name)
    
    # Calculate and save portfolio-level CAAR significance tests
    try:
        print(f"\nCalculating portfolio-level CAAR significance tests for {index_name}...")
        index_summary_file = os.path.join(index_analysis_dir, "summaries", f"{index_name}_index_summary.json")
        calculate_and_save_portfolio_caar_significance(
            portfolio_name=index_name,
            ticker_list=ticker_list,
            run_output_dir=output_dir,
            index_summary_file_path=index_summary_file
        )
        print(f"✅ Portfolio CAAR significance tests completed for {index_name}")
    except Exception as e:
        logger.error(f"Error calculating portfolio CAAR significance: {e}", exc_info=True)
        print(f"⚠️  Error in portfolio CAAR significance calculation: {e}")
    
    # NEU: Portfolio Event Study Statistical Analysis
    try:
        from src.analysis.portfolio_event_study_statistics import run_portfolio_statistics_analysis
        
        print("\n" + "="*80)
        print(f" PORTFOLIO EVENT STUDY STATISTICAL ANALYSIS FOR {index_name} ".center(80, "="))
        print("="*80)
        
        logger.info(f"Running comprehensive portfolio statistical analysis for {index_name}")
        print(f"Running Portfolio Statistical Analysis...")
        print(f"- T-Tests for AAR/CAAR significance")
        print(f"- Effect Size Analysis (Cohen's d)")
        print(f"- Model Fit Statistics (R², RMSE)")
        print(f"- Cross-Sectional Analysis")
        print(f"- Consistency Measures")
        
        # Run portfolio statistics analysis
        portfolio_stats_results = run_portfolio_statistics_analysis(output_dir, index_name)
        
        if portfolio_stats_results:
            print(f"\n📊 PORTFOLIO EVENT STUDY STATISTICAL RESULTS:")
            
            # Show meta statistics
            meta = portfolio_stats_results.get('meta_statistics', {})
            print(f"   Stocks Analyzed: {meta.get('total_stocks_analyzed', 0)}")
            print(f"   Total News Events: {meta.get('total_events', 0)}")
            print(f"   Event Distribution: {meta.get('event_distribution', {})}")
            
            # Show significant t-test results
            t_tests = portfolio_stats_results.get('portfolio_t_tests', {}).get('aar_tests', {})
            significant_results = []
            
            print(f"\n🔬 STATISTICAL SIGNIFICANCE TESTS (AAR T-Tests):")
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in t_tests:
                    print(f"   {sentiment.upper()} Sentiment:")
                    for window, test_result in t_tests[sentiment].items():
                        p_val = test_result.get('p_value', 1.0)
                        t_stat = test_result.get('t_statistic', 0.0)
                        significant = test_result.get('significant_5pct', False)
                        sig_marker = " ✓ SIGNIFICANT" if significant else " ✗ NOT SIGNIFICANT"
                        print(f"     {window}min: t={t_stat:.3f}, p={p_val:.4f}{sig_marker}")
                        
                        if significant:
                            significant_results.append(f"{sentiment} {window}min")
            
            # Show effect sizes
            effect_sizes = portfolio_stats_results.get('effect_sizes', {})
            print(f"\n📈 EFFECT SIZES (Cohen's d vs Neutral):")
            for sentiment in ['positive', 'negative']:
                if sentiment in effect_sizes:
                    print(f"   {sentiment.upper()} vs Neutral:")
                    for window, results in effect_sizes[sentiment].items():
                        cohens_d = results.get('cohens_d', 0)
                        interpretation = results.get('effect_interpretation', 'unknown')
                        print(f"     {window}min: d={cohens_d:.3f} ({interpretation})")
            
            # Show model fit
            model_fit = portfolio_stats_results.get('model_fit', {}).get('sentiment_prediction', {})
            print(f"\n🎯 MODEL FIT STATISTICS:")
            if model_fit:
                print(f"   Sentiment Prediction Model (R²):")
                for window, results in model_fit.items():
                    r2 = results.get('r_squared', 0)
                    p_val = results.get('regression', {}).get('p_value', 1)
                    print(f"     {window}min: R²={r2:.4f}, p={p_val:.4f}")
            
            # Summary
            if significant_results:
                print(f"\n✅ STATISTICALLY SIGNIFICANT RESULTS FOUND:")
                for result in significant_results:
                    print(f"   - {result}")
            else:
                print(f"\n⚠️  NO STATISTICALLY SIGNIFICANT RESULTS at p < 0.05 level")
            
            print(f"\n📁 Detailed portfolio statistics saved to:")
            print(f"   - {index_dir}/portfolio_statistics.json (detailed data)")
            print(f"   - {index_dir}/portfolio_statistics_summary.txt (summary report)")
            
        else:
            logger.warning(f"Portfolio statistical analysis returned no results for {index_name}")
            print(f"⚠️  Portfolio statistical analysis returned no results.")
            
    except ImportError as e:
        logger.error(f"Portfolio statistics module not available: {e}")
        print(f"⚠️  Portfolio statistical analysis not available. Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"Error running portfolio statistical analysis: {e}", exc_info=True)
        print(f"❌ Error in portfolio statistical analysis: {e}")
    
    return index_analysis_dir

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Financial News Sentiment Analysis & Trading Strategy Backtester')
    
    # Ticker selection
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--dow', action='store_true', help='Process all Dow Jones Industrial Average stocks')
    parser.add_argument('--mag7', action='store_true', help='Process all Magnificent 7 stocks')
    parser.add_argument('--log', action='store_true', help='Process custom list of stocks (LOG)')
    parser.add_argument('--top5', action='store_true', help='Process top 5 tech stocks')
    
    # Portfolio options
    parser.add_argument('--portfolio-only', action='store_true', help='Skip individual ticker processing and only generate consolidated portfolio analysis')
    
    # News processing configuration
    parser.add_argument('--max-articles', type=int, 
                        default=NEWS_CONFIG.get("default_max_articles"),
                        help=f'Maximum articles to fetch per ticker (default: {NEWS_CONFIG.get("default_max_articles", "All")})')
    parser.add_argument('--start-date', type=str, 
                        default=NEWS_CONFIG.get("default_start_date"),
                        help='Start date for news (YYYY-MM-DD, default: Earliest available)')
    parser.add_argument('--end-date', type=str, 
                        default=NEWS_CONFIG.get("default_end_date"),
                        help='End date for news (YYYY-MM-DD, default: Latest available)')
    
    # Backtest configuration
    parser.add_argument('--entry-delay', type=int, 
                        default=BACKTEST_CONFIG["default_entry_delay"],
                        help=f'Minutes to wait before entering position (default: {BACKTEST_CONFIG["default_entry_delay"]})')
    parser.add_argument('--holding-period', type=int, 
                        default=BACKTEST_CONFIG["default_holding_period"],
                        help=f'Minutes to hold position (default: {BACKTEST_CONFIG["default_holding_period"]})')
    parser.add_argument('--max-workers', type=int, 
                        default=BACKTEST_CONFIG["default_max_workers"],
                        help=f'Max parallel workers for backtesting (default: {BACKTEST_CONFIG["default_max_workers"]})')
    parser.add_argument('--use-stop-loss', action='store_true',
                        default=BACKTEST_CONFIG["default_use_stop_loss"],
                        help='Enable stop loss')
    parser.add_argument('--stop-loss-pct', type=float, 
                        default=BACKTEST_CONFIG["default_stop_loss_pct"],
                        help=f'Stop loss percentage (default: {BACKTEST_CONFIG["default_stop_loss_pct"]}%%)')
    parser.add_argument('--take-profit-pct', type=float, 
                        default=BACKTEST_CONFIG["default_take_profit_pct"],
                        help=f'Take profit percentage (default: {BACKTEST_CONFIG["default_take_profit_pct"]}%%)')
    parser.add_argument('--include-after-hours', action='store_true',
                        default=BACKTEST_CONFIG["default_include_after_hours"],
                        help='Include after-hours trading in backtests')
    parser.add_argument('--use-flip-strategy', action='store_true',
                        default=BACKTEST_CONFIG["default_use_flip_strategy"],
                        help='Use inverse sentiment trading strategy')
    parser.add_argument('--transaction-fee-pct', type=float,
                        default=BACKTEST_CONFIG["default_transaction_fee_pct"],
                        help=f'Transaction fee percentage per trade (buy/sell) (default: {BACKTEST_CONFIG["default_transaction_fee_pct"]*100}%%)')
    
    # Event study configuration
    parser.add_argument('--market-index', type=str, 
                        default="SPY",
                        help='Market index for event study (default: SPY)')
    parser.add_argument('--disable-event-study', action='store_true',
                        help='Disable event study analysis')
    
    # Statistical tests configuration
    parser.add_argument('--run-statistical-tests', action='store_true',
                        help='Run statistical tests on event study and backtest results')
    parser.add_argument('--significance-level', type=float, 
                        default=0.05,
                        help='Significance level for statistical tests (default: 0.05)')
    parser.add_argument('--bootstrap-simulations', type=int, 
                        default=10000,
                        help='Number of bootstrap simulations (default: 10000)')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, 
                        default=OUTPUT_CONFIG["base_dir"],
                        help=f'Output directory (default: {OUTPUT_CONFIG["base_dir"]})')
    parser.add_argument('--clean', action='store_true',
                        help='Clean output directory before running')
    
    return parser.parse_args()

def get_user_input():
    """Get user input for configuration parameters, using new defaults."""
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS & TRADING STRATEGY BACKTESTER")
    print("="*80)
    
    # Get ticker symbol
    ticker_input = input("\nEnter ticker symbol (or 'DOW' for Dow 30, 'MAG7' for Magnificent 7, 'LOG' for Custom List, 'TOP5' for Top 5 Tech stocks): ").strip().upper()
    
    # Handle ticker selection
    ticker = None
    is_dow_index = False
    is_mag7_index = False
    is_log_index = False
    is_top5_index = False
    
    if ticker_input == "DOW":
        print("Processing all Dow Jones Industrial Average stocks.")
        is_dow_index = True
    elif ticker_input == "MAG7":
        print("Processing all Magnificent 7 stocks.")
        is_mag7_index = True
    elif ticker_input == "LOG":
        print("Processing custom list of stocks (LOG).")
        is_log_index = True
    elif ticker_input == "TOP5":
        print("Processing top 5 tech stocks.")
        is_top5_index = True
    else:
        ticker = ticker_input
        print(f"Processing single ticker: {ticker}")
    
    # --- Use hardcoded defaults based on user request ---
    start_date_str = NEWS_CONFIG.get('default_start_date') # None for all available
    end_date_str = NEWS_CONFIG.get('default_end_date')     # None for all available
    max_articles = NEWS_CONFIG.get('default_max_articles') # None for all articles
    entry_delay = BACKTEST_CONFIG['default_entry_delay']
    holding_period = BACKTEST_CONFIG['default_holding_period']
    use_stop_loss = BACKTEST_CONFIG['default_use_stop_loss']
    stop_loss_pct = BACKTEST_CONFIG['default_stop_loss_pct']
    take_profit_pct = BACKTEST_CONFIG['default_take_profit_pct']
    include_after_hours = BACKTEST_CONFIG['default_include_after_hours']
    use_flip_strategy = BACKTEST_CONFIG['default_use_flip_strategy']
    # --- End of hardcoded defaults ---

    # --- Get transaction fee ---
    transaction_fee_pct_input = input(f"Enter transaction fee percentage per trade (e.g., 0.1 for 0.1%) [default: {BACKTEST_CONFIG['default_transaction_fee_pct']*100}]: ").strip()
    if not transaction_fee_pct_input:
        transaction_fee_pct = BACKTEST_CONFIG['default_transaction_fee_pct']
    else:
        try:
            transaction_fee_pct = float(transaction_fee_pct_input) / 100.0
            if not (0 <= transaction_fee_pct <= 1): # Assuming fee is between 0% and 100%
                print("Invalid fee percentage. Using default.")
                transaction_fee_pct = BACKTEST_CONFIG['default_transaction_fee_pct']
        except ValueError:
            print("Invalid input for fee. Using default.")
            transaction_fee_pct = BACKTEST_CONFIG['default_transaction_fee_pct']

    # --- Automatically set event study and statistical tests based on config defaults ---
    run_event_study_flag = EVENT_STUDY_CONFIG.get("default_enabled", True) # Use new default
    
    run_statistical_tests_flag = True  # Statt False
    significance_level = BATCH_CONFIG["default_significance_level"]
    bootstrap_simulations = BATCH_CONFIG["default_bootstrap_simulations"]

    if run_event_study_flag:
        # Automatically enable statistical tests if event study is enabled, based on its own default
        run_statistical_tests_flag = BATCH_CONFIG.get("default_run_statistical_tests", True)
        # Significance level and bootstrap simulations will use their defaults from BATCH_CONFIG
        # No need to prompt for them.
    
    # Removed run_robustness_flag and related conditional printing.
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    if ticker:
        print(f"Ticker: {ticker}") # Corrected potential indentation error here
    elif is_dow_index:
        print("Processing: Dow Jones Industrial Average")
    elif is_mag7_index:
        print("Processing: Magnificent 7")
    elif is_log_index:
        print("Processing: Custom Stock List (LOG)")
    elif is_top5_index:
        print("Processing: Top 5 Tech Stocks")
    
    print(f"Start Date: {start_date_str if start_date_str else 'All available'}")
    print(f"End Date: {end_date_str if end_date_str else 'All available'}")
    print(f"Max Articles: {max_articles if max_articles is not None else 'All available'}")
    print(f"Entry Delay: {entry_delay} minutes")
    print(f"Holding Period: {holding_period} minutes")
    print(f"Transaction Fee: {transaction_fee_pct*100:.3f}% per trade")
    
    if use_stop_loss:
        print(f"Stop Loss: {stop_loss_pct}%")
    else:
        print("Stop Loss: Disabled")
    
    if take_profit_pct == float('inf') or take_profit_pct == 0:
        print("Take Profit: Disabled (Infinite/0)")
    else:
        print(f"Take Profit: {take_profit_pct}%")
    
    print(f"Include After-Hours: {'Yes' if include_after_hours else 'No'}")
    print(f"Flip Strategy: {'Yes' if use_flip_strategy else 'No'}")
    print(f"Event Study: {'Yes (default)' if run_event_study_flag else 'No (default)'}")
    
    if run_event_study_flag and run_statistical_tests_flag:
        print(f"Statistical Tests: Enabled (p={significance_level}, bootstrap={bootstrap_simulations})")
    elif run_event_study_flag and not run_statistical_tests_flag:
        print("Statistical Tests: Disabled (default for event study)")
    else:
        print("Statistical Tests: Disabled (event study also disabled)")
    
    # Removed print statement for Robustness Testing status
    
    # Confirm before proceeding - User only confirms the Ticker/Index choice now
    if not get_yes_no_input(f"\nProceed with processing {ticker_input if ticker_input else 'the selected index'} and the default settings above? (y/n)", True):
        print("Exiting...")
        sys.exit(0)
    
    # Return configuration as dictionary
    return {
        "ticker": ticker,
        "is_dow_index": is_dow_index,
        "is_mag7_index": is_mag7_index,
        "is_log_index": is_log_index,
        "is_top5_index": is_top5_index,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "max_articles": max_articles,
        "entry_delay": entry_delay,
        "holding_period": holding_period,
        "use_stop_loss": use_stop_loss,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "include_after_hours": include_after_hours,
        "use_flip_strategy": use_flip_strategy,
        "run_event_study": run_event_study_flag, 
        "run_statistical_tests": run_statistical_tests_flag, 
        "significance_level": significance_level,
        "bootstrap_simulations": bootstrap_simulations,
        "transaction_fee_pct": transaction_fee_pct
        # Removed 'run_robustness_test' from returned dict
    }

def get_int_input(prompt: str, default: int, min_val: int, max_val: int) -> int:
    """Get integer input from user with validation."""
    while True:
        try:
            value_input = input(prompt).strip()
            if not value_input:  # Use default if empty
                return default
            
            value = int(value_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid number.")

def get_float_input(prompt: str, default: float, min_val: float, max_val: float) -> float:
    """Get float input from user with validation."""
    while True:
        try:
            value_input = input(prompt).strip()
            if not value_input:  # Use default if empty
                return default
            
            value = float(value_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid number.")

def get_yes_no_input(prompt: str, default: bool) -> bool:
    """Get yes/no input from user."""
    while True:
        value_input = input(prompt).strip().lower()
        if not value_input:
            return default
        elif value_input in ['yes', 'y']:
            return True
        elif value_input in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")

def get_date_input(prompt: str) -> str:
    """Get date input from user in YYYY-MM-DD format."""
    while True:
        date_input = input(prompt).strip()
        if not date_input:
            return None
        
        try:
            # Validate date format
            datetime.strptime(date_input, '%Y-%m-%d')
            return date_input
        except ValueError:
            print("Please enter a valid date in YYYY-MM-DD format.")

def process_ticker(ticker: str, 
                  max_articles: int, 
                  entry_delay: int, 
                  holding_period: int,
                  use_stop_loss: bool,
                  stop_loss_pct: float,
                  take_profit_pct: float,
                  include_after_hours: bool,
                  max_workers: int,
                  use_flip_strategy: bool,
                  transaction_fee_pct: float,
                  output_dir: str) -> Optional[Dict[str, Any]]:
    """Process a single ticker with sentiment analysis and backtesting."""
    
    try:
        logger.info(f"Processing ticker {ticker}...")
        
        # Create ticker-specific output directory
        ticker_dir = os.path.join(output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Fetch and analyze sentiment - USE THE CORRECT FUNCTION
        sentiment_results = run_sentiment_analysis_pipeline(
            ticker=ticker,
            max_articles=max_articles,
            output_dir=ticker_dir
        )
        
        if not sentiment_results:
            logger.error(f"No sentiment results for {ticker}")
            return None
            
        # Run backtesting - sentiment_results is already the list, not a dict
        backtest_results = run_backtest_pipeline(
            ticker=ticker,
            sentiment_data=sentiment_results,  # Direct list, not ['news_with_sentiment']
            entry_delay=entry_delay,
            holding_period=holding_period,
            use_stop_loss=use_stop_loss,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            include_after_hours=include_after_hours,
            max_workers=max_workers,
            use_flip_strategy=use_flip_strategy,
            output_dir=ticker_dir,
            transaction_fee_pct=transaction_fee_pct
        )
        
        if not backtest_results:
            logger.error(f"No backtest results for {ticker}")
            return None
            
        # Run event study analysis - sentiment_results is already the list, not a dict
        logger.info(f"Running event study analysis for {ticker}...")
        event_study_results = run_event_study(
            ticker=ticker,
            sentiment_data=sentiment_results,  # Direct list, not ['news_with_sentiment']
            output_dir=ticker_dir
        )
        
        # Run bootstrap analysis
        logger.info(f"Running bootstrap analysis for {ticker}...")
        bootstrap_results = run_bootstrap_on_backtest(
            backtest_file=os.path.join(ticker_dir, f"backtest_results_{ticker}.json"),
            output_dir=os.path.join(ticker_dir, "statistical_tests"),
            num_simulations=10000
        )
        
        # Run random benchmark analysis
        logger.info(f"Running random trading benchmark for {ticker}...")
        try:
            # Need to fetch price data for random benchmark
            from backtesting.helpers.market_data import fetch_stock_prices
            from datetime import datetime, timedelta
            
            # Get date range from trades
            trades = backtest_results.get('trades', [])
            if trades:
                start_date = min(pd.to_datetime(trade['entry_time']) for trade in trades) - timedelta(days=1)
                end_date = max(pd.to_datetime(trade['exit_time']) for trade in trades) + timedelta(days=1)
                
                price_data = fetch_stock_prices(ticker, start_date, end_date)
                
                if not price_data.empty:
                    random_benchmark_results = run_random_benchmark_analysis(
                        ticker=ticker,
                        sentiment_trades=trades,
                        strategy_results=backtest_results,
                        price_data=price_data,
                        output_dir=ticker_dir
                    )
                    logger.info(f"Random benchmark completed for {ticker}")
                else:
                    logger.warning(f"Could not fetch price data for random benchmark for {ticker}")
                    random_benchmark_results = None
            else:
                logger.warning(f"No trades available for random benchmark for {ticker}")
                random_benchmark_results = None
                
        except Exception as e:
            logger.error(f"Error running random benchmark for {ticker}: {e}")
            random_benchmark_results = None
        
        # Combine all results
        combined_results = {
            'ticker': ticker,
            'sentiment_results': sentiment_results,
            'backtest_results': backtest_results,
            'event_study_results': event_study_results,
            'bootstrap_results': bootstrap_results,
            'random_benchmark_results': random_benchmark_results
        }
        
        logger.info(f"Successfully completed analysis for {ticker}")
        return combined_results
        
    except Exception as e:
        logger.error(f"Error processing ticker {ticker}: {e}", exc_info=True)
        return None

def main():
    """Main function to run the application."""
    start_time = time.time()
    # Configure logging (basic config, file handler added in create_output_dirs if single ticker)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("Application started") # This uses the root logger
    
    # Parse arguments first
    args = parse_args()
    params = vars(args) # Convert argparse.Namespace to dict

    # If no CLI args that define the run mode (ticker, dow, mag7), then get user input.
    # Otherwise, CLI args take precedence.
    if not (args.ticker or args.dow or args.mag7 or args.log or args.top5):
        params = get_user_input()
    else: # CLI mode, print summary of effective params
        print("\n" + "="*80)
        print(" RUNNING WITH COMMAND-LINE ARGUMENTS ".center(80, "="))
        print("="*80)
        print("Effective Configuration:")
        if params.get("ticker"):
            print(f"  Ticker: {params['ticker']}")
        if params.get("is_dow_index") or args.dow: # Check both, args.dow for direct CLI flag
            print("  Processing: Dow Jones Industrial Average")
        if params.get("is_mag7_index") or args.mag7: # Check both, args.mag7 for direct CLI flag
            print("  Processing: Magnificent 7")
        if params.get("is_log_index") or args.log: # Check both, args.log for direct CLI flag
            print("  Processing: Custom Stock List (LOG)")
        if params.get("is_top5_index") or args.top5: # Check both, args.top5 for direct CLI flag
            print("  Processing: Top 5 Tech Stocks")
        
        # If portfolio-only mode
        if args.portfolio_only:
            print("  Mode: Portfolio Analysis Only (skipping individual ticker processing)")
        
        start_date_display = params.get('start_date') if params.get('start_date') else 'All available'
        end_date_display = params.get('end_date') if params.get('end_date') else 'All available'
        max_articles_display = params.get('max_articles') if params.get('max_articles') is not None else 'All available'

        print(f"  Start Date: {start_date_display}")
        print(f"  End Date: {end_date_display}")
        print(f"  Max Articles: {max_articles_display}")
        print(f"  Entry Delay: {params['entry_delay']} minutes")
        print(f"  Holding Period: {params['holding_period']} minutes")
        print(f"  Transaction Fee: {params['transaction_fee_pct']*100:.3f}% per trade")
        
        if params['use_stop_loss']:
            print(f"  Stop Loss: {params['stop_loss_pct']}%")
        else:
            print("  Stop Loss: Disabled")
        
        tp_pct_val = params.get('take_profit_pct', float('inf'))
        if tp_pct_val == float('inf') or tp_pct_val == 0:
            print("  Take Profit: Disabled (Infinite/0)")
        else:
            print(f"  Take Profit: {tp_pct_val}%")
            
        print(f"  Include After-Hours: {'Yes' if params['include_after_hours'] else 'No'}")
        print(f"  Flip Strategy: {'Yes' if params['use_flip_strategy'] else 'No'}")
        print(f"  Event Study: {'No' if params.get('disable_event_study') else 'Yes'}")
        if not params.get('disable_event_study') and params.get('run_statistical_tests'):
            print(f"  Statistical Tests: Enabled (p={params['significance_level']}, bootstrap={params['bootstrap_simulations']})")
        else:
            print("  Statistical Tests: Disabled")
        print(f"  Transaction Fee: {params['transaction_fee_pct']*100:.3f}% per trade")
        print("="*80 + "\n")

    # Configure parameters from user input or CLI args
    ticker = params.get("ticker") # Use .get for safety
    start_date_str = params.get("start_date")
    end_date_str = params.get("end_date")
    output_base_dir = params.get("output_dir", OUTPUT_CONFIG["base_dir"]) # Use CLI or default 'output'
    
    # Create ticker_list based on parameters
    ticker_list = []
    index_name = None
    if params.get("is_dow_index") or getattr(args, 'dow', False): # Check params from get_user_input or direct CLI arg
        ticker_list = get_dow_tickers()
        index_name = "DOW_Jones"
        logger.info(f"Using Dow Jones tickers: {ticker_list}")
        if not ticker: ticker = index_name # For output dir naming
    elif params.get("is_mag7_index") or getattr(args, 'mag7', False):
        ticker_list = get_mag7_tickers()
        index_name = "MAG7"
        logger.info(f"Using Magnificent 7 tickers: {ticker_list}")
        if not ticker: ticker = index_name # For output dir naming
    elif params.get("is_log_index") or getattr(args, 'log', False):
        ticker_list = get_custom_tickers()
        index_name = "LOG"
        logger.info(f"Using Custom tickers (LOG): {ticker_list}")
        if not ticker: ticker = index_name # For output dir naming
    elif params.get("is_top5_index") or getattr(args, 'top5', False):
        ticker_list = get_top5_tickers()
        index_name = "TOP5"
        logger.info(f"Using Top 5 tech tickers: {ticker_list}")
        if not ticker: ticker = index_name # For output dir naming
    elif ticker:
        ticker_list = [ticker] # Single ticker run
    
    if not ticker_list and not ticker: # Should not happen if CLI/interactive logic is correct
        logger.error("No ticker or index specified. Exiting.")
        print("Error: No ticker or index specified. Please use --ticker, --dow, or --mag7, or provide input interactively.")
        return 1

    # Parse dates if provided
    start_date_dt = None
    end_date_dt = None
    if start_date_str:
        try:
            start_date_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid start_date format: {start_date_str}. Expected YYYY-MM-DD.")
            print(f"Error: Invalid start_date format: {start_date_str}. Expected YYYY-MM-DD.")
            return 1
    if end_date_str:
        try:
            end_date_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid end_date format: {end_date_str}. Expected YYYY-MM-DD.")
            print(f"Error: Invalid end_date format: {end_date_str}. Expected YYYY-MM-DD.")
            return 1
    
    # Create a unique run directory based on timestamp and primary ticker/index
    # This is the main output directory for the entire run (single or multi-ticker)
    run_timestamp = datetime.now().strftime(OUTPUT_CONFIG["timestamp_format"])
    primary_identifier = ticker if ticker else index_name if index_name else "multi_run"
    
    # Use output_base_dir from params (CLI or default 'output')
    main_output_dir_name = f"{OUTPUT_CONFIG.get('default_run_prefix', 'run')}_{primary_identifier}_{run_timestamp}"
    main_output_dir = os.path.join(output_base_dir, main_output_dir_name)
    
    if args.clean and os.path.exists(main_output_dir):
        logger.info(f"Cleaning output directory: {main_output_dir}")
        shutil.rmtree(main_output_dir)
    os.makedirs(main_output_dir, exist_ok=True)
    logger.info(f"Main output directory for this run: {main_output_dir}")

    # Configure file logging to the main output directory for the run
    log_file_path = os.path.join(main_output_dir, 'run_log.log')
    file_handler = logging.FileHandler(log_file_path)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler) # Add to root logger
    logger.info(f"Logging to file: {log_file_path}")
    
    # Process single ticker or multiple tickers
    # Pass the main_output_dir and other params to processing functions
    processing_kwargs = params.copy() # Start with all params
    processing_kwargs["start_date"] = start_date_dt # Pass datetime objects
    processing_kwargs["end_date"] = end_date_dt
    processing_kwargs["index_name"] = index_name # Pass the determined index name
    
    # Convert disable_event_study to enable_event_study for backward compatibility
    if 'disable_event_study' in processing_kwargs:
        processing_kwargs["enable_event_study"] = not processing_kwargs.pop('disable_event_study')
    
    if len(ticker_list) > 1 and (params.get("is_dow_index") or params.get("is_mag7_index") or params.get("is_log_index") or params.get("is_top5_index") or getattr(args, 'dow', False) or getattr(args, 'mag7', False) or getattr(args, 'log', False) or getattr(args, 'top5', False)):
        # If portfolio-only mode is enabled, skip individual ticker processing
        if args.portfolio_only:
            logger.info(f"Portfolio-only mode enabled. Skipping individual ticker processing.")
            print(f"\nPortfolio-only mode enabled. Skipping individual ticker processing.")
            
            # Create empty directories for each ticker that would have been processed
            # This is necessary for the portfolio analysis to find existing ticker directories
            for ticker in ticker_list:
                ticker_dir = os.path.join(main_output_dir, ticker)
                os.makedirs(ticker_dir, exist_ok=True)
            
            # Run only the portfolio analysis
            logger.info(f"Running comprehensive portfolio analysis for {index_name}...")
            print(f"\nRunning comprehensive portfolio analysis for {index_name}...")
            fix_portfolio_report(ticker_list, main_output_dir, index_name)
        else:
            # Normal mode - process each ticker individually, then run portfolio analysis
            processing_kwargs.pop("output_dir", None) # Remove to avoid duplicate keyword argument
            process_multiple_tickers_main(ticker_list, output_dir=main_output_dir, **processing_kwargs)
            
            # After processing all individual tickers, run the comprehensive portfolio analysis
            logger.info(f"Individual ticker processing complete. Running comprehensive portfolio analysis for {index_name}...")
            print(f"\nIndividual ticker processing complete. Running comprehensive portfolio analysis for {index_name}...")
            
            # Run the enhanced portfolio report generation
            fix_portfolio_report(ticker_list, main_output_dir, index_name)
            
    elif ticker_list: # Single ticker specified via CLI or chosen interactively
        # For a single ticker run, the main_output_dir is the ticker's specific output dir
        processing_kwargs.pop("is_dow_index", None) # Not relevant for single ticker
        processing_kwargs.pop("is_mag7_index", None)
        processing_kwargs.pop("is_log_index", None)
        processing_kwargs.pop("is_top5_index", None)
        # index_name is still in processing_kwargs but not used by run_robustness_test or process_single_ticker directly for naming
        # It might be used if those functions were to call generate_index_analysis, but they don't.
        # We can keep it or remove it; let's remove to be clean for single ticker path.
        processing_kwargs.pop("index_name", None)
        
        # Remove output_dir from processing_kwargs to avoid duplication
        processing_kwargs.pop("output_dir", None)
        
        # Standard single ticker processing (sentiment, backtest, event study)
        # For single ticker, use the same logic as multi-ticker but with one ticker
        process_multiple_tickers_main(
            ticker_list, 
            output_dir=main_output_dir,
            **processing_kwargs
        ) 
    else:
        logger.error("No tickers to process. This should have been caught earlier.")
        print("Error: No tickers available for processing.")
        return 1
    
    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Application completed in {elapsed_time:.2f} seconds")
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds.")
    print(f"Check the '{main_output_dir}' directory for results.")
    return 0 # Indicate success

def configure_logging():
    """Configure logging for the application."""
    # Basic configuration, file handler is added later if not a multi-ticker master log
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Get the root logger
    # logger = logging.getLogger() # This is the root logger
    # logger.info("Root logger configured.") # Example message

# This function is more for per-ticker output dir, main run dir is created in main()
# def create_output_dirs(ticker):
#     """Create necessary output directories."""
#     # Generate a unique run identifier based on timestamp
#     run_id = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    
#     # Create output directory
#     # For single ticker, it's output/run_TICKER_timestamp
#     # For multi-ticker, this might be a sub-directory
#     dir_name = f"run_{ticker}_{run_id}" if ticker else f"run_multi_{run_id}"
#     output_dir = os.path.join(OUTPUT_CONFIG.get('base_dir', 'output'), dir_name)
#     os.makedirs(output_dir, exist_ok=True)
#     # logger.info(f"Created output directory: {output_dir}") # Logger not configured with file handler yet
    
#     # Configure logging to file in this specific directory
#     # file_handler = logging.FileHandler(os.path.join(output_dir, 'run.log'))
#     # file_handler.setFormatter(logging.Formatter(
#     #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
#     #     datefmt='%Y-%m-%d %H:%M:%S'
#     # ))
#     # logging.getLogger().addHandler(file_handler) # Add to root logger
    
#     # Print header banner
#     # logger.info("="*80)
#     # logger.info("FINANCIAL NEWS SENTIMENT ANALYSIS & BACKTESTING")
#     # logger.info("="*80)
    
#     return output_dir

def process_single_ticker_fix(ticker_list: List[str], output_dir: str, index_name: str = "TOP5"):
    """
    Fix-Funktion, um einen manuellen Portfolio-Report zu erstellen.
    """
    logger.info(f"Erstelle manuellen Portfolio-Report für {index_name}")
    
    index_kwargs = {}
    generate_index_analysis(output_dir, ticker_list, index_name, **index_kwargs)
    
    return "Portfolio-Report wurde manuell erstellt."

def fix_portfolio_report(ticker_list: List[str], output_dir: str, index_name: str = "TOP5"):
    """
    Creates a comprehensive portfolio report with full consolidation of:
    - Performance metrics across all tickers
    - Consolidated day-by-day AAR/CAAR from individual events (not just ticker averages)
    - Complete trade analysis and statistics at the portfolio level
    - Detailed visualizations and summaries
    
    Properly organizes results into the following structure:
    - index_name_INDEX_ANALYSIS/
        - summaries/
        - event_study/
        - trades/
        - visualizations/
    """
    from src.backtesting.analysis.backtest_visualizer import BacktestVisualizer
    import pandas as pd
    import numpy as np
    import json
    from datetime import datetime
    from scipy import stats
    import os
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter

    logger.info(f"Creating comprehensive portfolio report for {index_name} in {output_dir}")
    print(f"\n{'='*80}\nCREATING COMPREHENSIVE PORTFOLIO REPORT FOR {index_name}\n{'='*80}")

    # Create main index directory and subdirectories
    index_analysis_dir = os.path.join(output_dir, f"{index_name}_INDEX_ANALYSIS")
    summaries_dir = os.path.join(index_analysis_dir, "summaries")
    event_study_dir = os.path.join(index_analysis_dir, "event_study")
    trades_dir = os.path.join(index_analysis_dir, "trades")
    visualizations_dir = os.path.join(index_analysis_dir, "visualizations")
    
    # Create directories if they don't exist
    for dir_path in [index_analysis_dir, summaries_dir, event_study_dir, trades_dir, visualizations_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Initialize data collectors
    ticker_summaries = []
    all_trades = []
    all_events = {}  # Dictionary to collect all events by window size
    all_ar_values = {}  # Day-by-day abnormal returns for all tickers
    all_event_dates = []  # Collect all event dates for timeline analysis
    market_model_stats = []
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
    
    # Initialize AR collectors for each window size
    for window in [5, 15, 30, 60]:
        all_events[window] = {"positive": [], "negative": [], "neutral": []}
        all_ar_values[window] = {}  # Will store by day index
        
    # Process each ticker's data
    successful_tickers = []
    for ticker in ticker_list:
        ticker_dir = os.path.join(output_dir, ticker)
        print(f"Processing data for {ticker}...")
        ticker_processed = False
        
        # --- 1. BACKTEST RESULTS ---
        backtest_file = os.path.join(ticker_dir, f"backtest_results_{ticker}.json")
        if os.path.exists(backtest_file):
            try:
                with open(backtest_file, 'r') as f:
                    backtest_data = json.load(f)
                
                # Extract ticker summary stats
                ticker_summary = {
                    'ticker': ticker,
                    'total_trades': backtest_data.get('total_trades', 0),
                    'winning_trades': backtest_data.get('winning_trades', 0),
                    'losing_trades': backtest_data.get('losing_trades', 0),
                    'win_rate': backtest_data.get('win_rate', 0),
                    'total_return_pct': backtest_data.get('total_return_pct', 0),
                    'average_return_pct': backtest_data.get('average_return_pct', 0),
                    'median_return_pct': backtest_data.get('median_return_pct', 0),
                    'return_std_dev': backtest_data.get('return_std_dev', 0),
                    'sharpe_ratio': backtest_data.get('sharpe_ratio', 0),
                    'max_drawdown_pct': backtest_data.get('max_drawdown_pct', 0),
                    'profit_factor': backtest_data.get('profit_factor', 0)
                }
                
                # Add buy & hold comparison if available
                if 'buy_hold_comparison' in backtest_data:
                    ticker_summary['buyhold_return_pct'] = backtest_data['buy_hold_comparison'].get('buyhold_return_pct', 0)
                    ticker_summary['outperformance_pct'] = backtest_data['buy_hold_comparison'].get('outperformance_pct', 0)
                
                ticker_summaries.append(ticker_summary)
                
                # Extract individual trades
                if 'trades' in backtest_data and backtest_data['trades']:
                    for trade in backtest_data['trades']:
                        # Ensure ticker is set in trade data
                        trade['ticker'] = ticker
                        all_trades.append(trade)
                
                ticker_processed = True
                print(f"  ✓ Extracted backtest results with {backtest_data.get('total_trades', 0)} trades")
            except Exception as e:
                logger.error(f"Error processing backtest data for {ticker}: {e}", exc_info=True)
                print(f"  ✗ Error processing backtest data: {e}")
        else:
            print(f"  ✗ No backtest results file found for {ticker}")
        
        # --- 2. EVENT STUDY RESULTS ---
        event_study_file = os.path.join(ticker_dir, "event_study", f"{ticker}_event_study_results.json")
        if os.path.exists(event_study_file):
            try:
                with open(event_study_file, 'r') as f:
                    event_data = json.load(f)
                
                # Collect overall sentiment distribution
                if 'events_by_sentiment' in event_data:
                    sentiment_counts['positive'] += event_data['events_by_sentiment'].get('positive', 0)
                    sentiment_counts['negative'] += event_data['events_by_sentiment'].get('negative', 0)
                    sentiment_counts['neutral'] += event_data['events_by_sentiment'].get('neutral', 0)
                    sentiment_counts['total'] += event_data.get('unique_news_events', event_data.get('total_events', 0))
                
                # Collect market model stats
                if 'market_model_stats' in event_data and 'overall' in event_data['market_model_stats']:
                    market_model = event_data['market_model_stats']['overall']
                    market_model['ticker'] = ticker
                    market_model_stats.append(market_model)
                
                # Collect day-by-day AAR/CAAR data from each window
                for window_size in all_events.keys():
                    window_key = f"window_{window_size}"
                    
                    # Process detailed AR data if available
                    # First check if we have detailed event-level data
                    if 'detailed_events' in event_data and window_key in event_data['detailed_events']:
                        for sentiment, events in event_data['detailed_events'][window_key].items():
                            for event in events:
                                # Add ticker and window info
                                event['ticker'] = ticker
                                event['window'] = window_size
                                all_events[window_size][sentiment].append(event)
                                
                                # Collect event date
                                if 'event_date' in event:
                                    all_event_dates.append({
                                        'ticker': ticker,
                                        'date': event['event_date'],
                                        'sentiment': sentiment,
                                        'window': window_size
                                    })
                                
                                # Process daily ARs if available
                                if 'daily_ar' in event:
                                    for day, ar_value in event['daily_ar'].items():
                                        if day not in all_ar_values[window_size]:
                                            all_ar_values[window_size][day] = []
                                        all_ar_values[window_size][day].append(ar_value)
                    
                    # If no detailed data, try to get aggregate AAR/CAAR from the summary data
                    elif 'aar_caar' in event_data:
                        for sentiment in ['positive', 'negative', 'neutral']:
                            if sentiment in event_data['aar_caar']:
                                # Check if specific window size exists in the data
                                if 'AAR' in event_data['aar_caar'][sentiment] and window_size in event_data['aar_caar'][sentiment]['AAR']:
                                    aar_value = event_data['aar_caar'][sentiment]['AAR'][window_size]
                                    
                                    # Create a single-point representation of AAR (day 0)
                                    if '0' not in all_ar_values[window_size]:
                                        all_ar_values[window_size]['0'] = []
                                    all_ar_values[window_size]['0'].append(aar_value)
                
                ticker_processed = True
                print(f"  ✓ Extracted event study results with {event_data.get('unique_news_events', event_data.get('total_events', 0))} unique news events")
            except Exception as e:
                logger.error(f"Error processing event study data for {ticker}: {e}", exc_info=True)
                print(f"  ✗ Error processing event study data: {e}")
        else:
            print(f"  ✗ No event study results file found for {ticker}")
        
        if ticker_processed:
            successful_tickers.append(ticker)
    
    # Convert to DataFrame for easier analysis
    ticker_summaries_df = pd.DataFrame(ticker_summaries)
    all_trades_df = pd.DataFrame(all_trades)
    market_model_stats_df = pd.DataFrame(market_model_stats)
    
    # Create ticker CSV summary
    if not ticker_summaries_df.empty:
        ticker_summaries_df.to_csv(os.path.join(summaries_dir, f"{index_name}_ticker_summaries.csv"), index=False)
        print(f"Saved {index_name}_ticker_summaries.csv with {len(ticker_summaries_df)} tickers")
    else:
        print("No ticker summary data available")
    
    # Create trades CSV summary
    if not all_trades_df.empty:
        all_trades_df.to_csv(os.path.join(trades_dir, f"{index_name}_all_trades.csv"), index=False)
        print(f"Saved {index_name}_all_trades.csv with {len(all_trades_df)} trades")
        
        # Also save trades by sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_trades = all_trades_df[all_trades_df['sentiment'] == sentiment]
            if not sentiment_trades.empty:
                sentiment_trades.to_csv(os.path.join(trades_dir, f"{index_name}_{sentiment}_trades.csv"), index=False)
                print(f"Saved {index_name}_{sentiment}_trades.csv with {len(sentiment_trades)} trades")
    else:
        print("No trade data available")
    
    # --- CALCULATE CONSOLIDATED AAR/CAAR ---
    # Process for each window size
    consolidated_aar_caar = {}
    
    for window_size in all_ar_values.keys():
        if all_ar_values[window_size]:
            # Sort the days by their integer value
            days = sorted([int(day) for day in all_ar_values[window_size].keys()])
            
            # Check if we have meaningful data
            if len(days) == 0:
                print(f"No data points found for {window_size}min window")
                continue
                
            # If we have only one day (typically day 0), expand to create a better visualization
            if len(days) == 1:
                print(f"Only one day point found for {window_size}min window, expanding for better visualization")
                if days[0] == 0:  # If the single day is event day
                    days = [-3, -2, -1, 0, 1, 2, 3]  # Create a range around event day
            
            # Initialize the AAR and standard deviations lists
            aar_values = []
            aar_stds = []
            caar_values = []
            aar_t_stats = []
            aar_p_values = []
            
            # Calculate AAR for each day
            for day in days:
                day_str = str(day)
                # For expanded days that don't exist in original data, use zero values
                if day_str not in all_ar_values[window_size]:
                    aar_values.append(0)
                    aar_stds.append(0)
                    aar_t_stats.append(None)
                    aar_p_values.append(None)
                    continue
                
                day_values = [v for v in all_ar_values[window_size][day_str] if v is not None and not np.isnan(v)]
                
                if day_values:
                    # Calculate AAR for this day
                    aar = np.mean(day_values)
                    aar_std = np.std(day_values, ddof=1)
                    
                    # Calculate t-statistic and p-value
                    if len(day_values) > 1:
                        t_stat, p_value = stats.ttest_1samp(day_values, 0)
                    else:
                        t_stat, p_value = np.nan, np.nan
                    
                    aar_values.append(aar)
                    aar_stds.append(aar_std)
                    aar_t_stats.append(float(t_stat) if not np.isnan(t_stat) else None)
                    aar_p_values.append(float(p_value) if not np.isnan(p_value) else None)
                else:
                    aar_values.append(0)
                    aar_stds.append(0)
                    aar_t_stats.append(None)
                    aar_p_values.append(None)
            
            # Calculate CAAR as the cumulative sum of AAR values
            caar_values = list(np.cumsum(aar_values))
            
            # Create consolidated result
            consolidated_aar_caar[window_size] = {
                "days": days,
                "aar": [float(x) for x in aar_values],
                "aar_std": [float(x) for x in aar_stds],
                "caar": [float(x) for x in caar_values],
                "t_stats": aar_t_stats,
                "p_values": aar_p_values,
                "num_events": len(day_values) if 'day_values' in locals() else 0
            }
            
            # Save individual window results
            aar_file = os.path.join(event_study_dir, f"{index_name}_consolidated_AAR_{window_size}min.json")
            caar_file = os.path.join(event_study_dir, f"{index_name}_consolidated_CAAR_{window_size}min.json")
            
            aar_json = {"days": days, "aar": [float(x) for x in aar_values]}
            caar_json = {"days": days, "caar": [float(x) for x in caar_values]}
            
            save_json(aar_json, aar_file)
            save_json(caar_json, caar_file)
            print(f"Saved consolidated AAR/CAAR for {window_size}min window")
        else:
            print(f"No AR data available for {window_size}min window")
    
    # Save overall AAR/CAAR data
    if consolidated_aar_caar:
        save_json(consolidated_aar_caar, os.path.join(event_study_dir, f"{index_name}_all_consolidated_aar_caar.json"))
    
    # --- T-TEST RESULTS ---
    # Perform t-tests on the AAR data for each window using REAL event study results
    consolidated_ttest_results = {}
    
    # Load the actual detailed event study results to get proper statistics
    detailed_event_file = os.path.join(event_study_dir, f"{index_name}_all_consolidated_aar_caar.json")
    if os.path.exists(detailed_event_file):
        try:
            with open(detailed_event_file, 'r') as f:
                detailed_data = json.load(f)
            
            # Use the real event study t-test results
            for window_str, window_data in detailed_data.items():
                if 't_stats' in window_data and 'p_values' in window_data:
                    # Find the event day (day 0) t-stat and p-value
                    days = window_data.get('days', [])
                    t_stats = window_data.get('t_stats', [])
                    p_values = window_data.get('p_values', [])
                    aar_values = window_data.get('aar', [])
                    
                    if 0 in days:  # Find event day
                        event_day_idx = days.index(0)
                        if (event_day_idx < len(t_stats) and event_day_idx < len(p_values) and 
                            event_day_idx < len(aar_values) and t_stats[event_day_idx] is not None):
                            
                            consolidated_ttest_results[window_str] = {
                                "window_size": int(window_str),
                                "mean_aar": float(aar_values[event_day_idx]),
                                "t_statistic": float(t_stats[event_day_idx]),
                                "p_value": float(p_values[event_day_idx]),
                                "significant_at_0.05": bool(p_values[event_day_idx] < 0.05),
                                "num_observations": int(window_data.get('num_events', 0))
                            }
        except Exception as e:
            logger.warning(f"Could not load detailed event study results for proper t-tests: {e}")
            # Fallback to old method but with warning
            for window_size, data in consolidated_aar_caar.items():
                if data["aar"]:
                    # Calculate mean AAR
                    mean_aar = np.mean(data["aar"])
                    
                    # Collect all non-null AAR values
                    valid_aars = [aar for aar in data["aar"] if aar is not None and not np.isnan(aar)]
                    
                    if valid_aars and len(valid_aars) > 1:
                        # Perform t-test
                        t_stat, p_value = stats.ttest_1samp(valid_aars, 0)
                        
                        consolidated_ttest_results[window_size] = {
                            "window_size": window_size,
                            "mean_aar": float(mean_aar),
                            "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
                            "p_value": float(p_value) if not np.isnan(p_value) else None,
                            "significant_at_0.05": bool(p_value < 0.05) if not np.isnan(p_value) else None,
                            "num_observations": len(valid_aars)
                        }
                        logger.warning(f"Using fallback t-test calculation for window {window_size}")
    else:
        # Original fallback method
        for window_size, data in consolidated_aar_caar.items():
            if data["aar"]:
                mean_aar = np.mean(data["aar"])
                valid_aars = [aar for aar in data["aar"] if aar is not None and not np.isnan(aar)]
                
                if valid_aars and len(valid_aars) > 1:
                    t_stat, p_value = stats.ttest_1samp(valid_aars, 0)
                    
                    consolidated_ttest_results[window_size] = {
                        "window_size": window_size,
                        "mean_aar": float(mean_aar),
                        "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
                        "p_value": float(p_value) if not np.isnan(p_value) else None,
                        "significant_at_0.05": bool(p_value < 0.05) if not np.isnan(p_value) else None,
                        "num_observations": len(valid_aars)
                    }
    
    # Save consolidated t-test results
    if consolidated_ttest_results:
        save_json(consolidated_ttest_results, os.path.join(summaries_dir, f"{index_name}_consolidated_ttest_summary.json"))
        print(f"Saved consolidated t-test results for {len(consolidated_ttest_results)} windows")
    
    # --- PORTFOLIO STATISTICS AND SUMMARY ---
    # Calculate key portfolio metrics
    portfolio_summary = {
        "index_name": index_name,
        "total_tickers": len(ticker_list),
        "successful_tickers": len(successful_tickers),
        "ticker_list": ticker_list,
        "successful_ticker_list": successful_tickers,
        "timestamp": datetime.now().isoformat(),
        "sentiment_distribution": sentiment_counts
    }
    
    # Add backtest stats if available
    if not ticker_summaries_df.empty:
        portfolio_summary.update({
            "total_trades": int(ticker_summaries_df["total_trades"].sum()),
            "average_win_rate": float(ticker_summaries_df["win_rate"].mean()),
            "average_return_pct": float(ticker_summaries_df["total_return_pct"].mean()),
            "median_return_pct": float(ticker_summaries_df["total_return_pct"].median()),
            "max_return_pct": float(ticker_summaries_df["total_return_pct"].max()),
            "min_return_pct": float(ticker_summaries_df["total_return_pct"].min()),
            "average_sharpe_ratio": float(ticker_summaries_df["sharpe_ratio"].mean()),
            "average_profit_factor": float(ticker_summaries_df["profit_factor"].mean()),
            "average_max_drawdown_pct": float(ticker_summaries_df["max_drawdown_pct"].mean())
        })
        
        # Add buy & hold comparison if available
        if "buyhold_return_pct" in ticker_summaries_df.columns:
            portfolio_summary.update({
                "average_buyhold_return_pct": float(ticker_summaries_df["buyhold_return_pct"].mean()),
                "average_outperformance_pct": float(ticker_summaries_df["outperformance_pct"].mean()),
            })
    
    # Add market model stats if available
    if not market_model_stats_df.empty:
        portfolio_summary["market_model"] = {
            "average_r_squared": float(market_model_stats_df["avg_r_squared"].mean()),
            "average_adj_r_squared": float(market_model_stats_df["avg_adj_r_squared"].mean()),
            "average_rmse": float(market_model_stats_df["avg_rmse"].mean()),
            "average_mae": float(market_model_stats_df["avg_mae"].mean()),
            "tickers_with_valid_models": int(market_model_stats_df["count"].sum())
        }
    
    # Add event study stats
    portfolio_summary["event_study"] = {
                    "unique_news_events": sentiment_counts["total"],  # Total unique news events across all tickers
        "event_windows": list(consolidated_aar_caar.keys()),
        "event_distribution": {
            "positive": sentiment_counts["positive"],
            "negative": sentiment_counts["negative"],
            "neutral": sentiment_counts["neutral"]
        },
        "consolidated_ttest_results": consolidated_ttest_results
    }
    
    # Add random benchmark aggregation if available
    random_benchmark_summaries = []
    for ticker in ticker_list:
        ticker_dir = os.path.join(output_dir, ticker)
        random_benchmark_file = os.path.join(ticker_dir, f"{ticker}_random_benchmark.json")
        if os.path.exists(random_benchmark_file):
            try:
                with open(random_benchmark_file, 'r') as f:
                    random_data = json.load(f)
                    if 'strategy_vs_random_comparison' in random_data:
                        comparison = random_data['strategy_vs_random_comparison']
                        random_benchmark_summaries.append({
                            'ticker': ticker,
                            'strategy_return_pct': comparison['strategy_return_pct'],
                            'random_mean_return_pct': comparison['random_mean_return_pct'],
                            'strategy_percentile_rank': comparison['strategy_percentile_rank'],
                            'outperforming_percentage': comparison['outperforming_percentage'],
                            'p_value': comparison['statistical_significance']['p_value'],
                            'significant_at_5pct': comparison['statistical_significance']['significant_at_5pct']
                        })
            except Exception as e:
                logger.warning(f"Error reading random benchmark for {ticker}: {e}")
    
    if random_benchmark_summaries:
        # Calculate portfolio-level random benchmark statistics
        avg_percentile_rank = np.mean([rb['strategy_percentile_rank'] for rb in random_benchmark_summaries])
        avg_strategy_return = np.mean([rb['strategy_return_pct'] for rb in random_benchmark_summaries])
        avg_random_return = np.mean([rb['random_mean_return_pct'] for rb in random_benchmark_summaries])
        significant_tickers = sum(1 for rb in random_benchmark_summaries if rb['significant_at_5pct'])
        
        portfolio_summary["random_benchmark"] = {
            "tickers_with_benchmark": len(random_benchmark_summaries),
            "average_strategy_percentile_rank": float(avg_percentile_rank),
            "average_strategy_return_pct": float(avg_strategy_return),
            "average_random_return_pct": float(avg_random_return),
            "tickers_significant_at_5pct": int(significant_tickers),
            "percentage_significant_tickers": float(significant_tickers / len(random_benchmark_summaries) * 100),
            "interpretation": f"Portfolio sentiment strategy performs at {avg_percentile_rank:.1f}th percentile vs random trading, with {significant_tickers}/{len(random_benchmark_summaries)} tickers showing statistically significant outperformance",
            "ticker_details": random_benchmark_summaries
        }
        
        logger.info(f"Random benchmark: Portfolio at {avg_percentile_rank:.1f}th percentile, {significant_tickers}/{len(random_benchmark_summaries)} significant")
    
    # Save portfolio summary
    save_json(portfolio_summary, os.path.join(summaries_dir, f"{index_name}_index_summary.json"))
    print(f"Saved comprehensive index summary")
    
    # --- VISUALIZATIONS ---
    # Initialize the visualizer 
    visualizer = BacktestVisualizer()
    
    try:
        # Create visualizations if we have ticker data
        if not ticker_summaries_df.empty:
            # 1. Ticker Performance Dashboard - Use the actual methods available
            visualizer.visualize_index_ticker_performance(
                ticker_summaries_df=ticker_summaries_df,
                index_dir=visualizations_dir
            )
            print("Created ticker performance dashboard")
            
            # 2. Create KPI dashboard with detailed metrics
            try:
                # Make sure portfolio_summary has the correct key for total_trades
                if 'total_trades' in portfolio_summary and 'total_trades_overall' not in portfolio_summary:
                    portfolio_summary['total_trades_overall'] = portfolio_summary['total_trades']
                
                visualizer.create_index_kpi_dashboard(
                    index_summary=portfolio_summary,
                    ticker_summaries_df=ticker_summaries_df,
                    output_dir=visualizations_dir
                )
                print("Created KPI dashboard")
            except Exception as e:
                logger.error(f"Error creating KPI dashboard: {e}", exc_info=True)
                print(f"Error creating KPI dashboard: {e}")
            
            # 3. Additional visualizations if trades data exists
            if not all_trades_df.empty:
                # Prepare ticker KPIs format that the visualization methods expect
                ticker_kpis = ticker_summaries_df.set_index('ticker').copy()
                
                # Make column names match what the visualizer expects
                if 'total_return_pct' in ticker_kpis.columns and 'total_return' not in ticker_kpis.columns:
                    ticker_kpis['total_return'] = ticker_kpis['total_return_pct']
                
                # Create portfolio performance visualization
                visualizer.visualize_index_portfolio_performance(
                    trades_df=all_trades_df,
                    ticker_summaries_df=ticker_summaries_df,
                    index_dir=visualizations_dir,
                    index_summary=portfolio_summary
                )
                print("Created portfolio performance visualization")
                
                # Create sentiment performance visualization
                visualizer.visualize_index_sentiment_performance(
                    trades_df=all_trades_df,
                    ticker_kpis=ticker_kpis,
                    index_dir=visualizations_dir
                )
                print("Created sentiment performance visualization")
                
                # Create strategy vs buy & hold visualization
                visualizer.visualize_index_strategy_vs_buyhold(
                    trades_df=all_trades_df,
                    index_dir=visualizations_dir,
                    ticker_list=ticker_list,
                    ticker_summaries_df=ticker_summaries_df
                )
                print("Created strategy vs buy & hold visualization")
                
                # Create portfolio cumulative returns vs index visualization
                visualizer.visualize_portfolio_cumulative_returns_vs_index(
                    trades_df=all_trades_df,
                    ticker_list=ticker_list,
                    index_dir=visualizations_dir,
                    index_name=index_name
                )
                print("Created portfolio cumulative returns vs index visualization")
                
                # Create portfolio event study timeline visualization
                visualizer.visualize_portfolio_event_study_timeline(
                    ticker_list=ticker_list,
                    index_dir=visualizations_dir,
                    index_name=index_name
                )
                print("Created portfolio event study timeline visualization")
                
                # Create portfolio event study by sentiment (like individual stocks)
                visualizer.create_portfolio_event_study_by_sentiment(
                    index_dir=visualizations_dir,
                    index_name=index_name
                )
                print("Created portfolio event study by sentiment visualization")
                
                # Create portfolio event study (JPM-style) 
                visualizer.create_portfolio_event_study_sentiment_styled(
                    index_dir=visualizations_dir,
                    index_name=index_name
                )
                print("Created portfolio event study (JPM-style) visualization")
                
                # Create comprehensive event study timeline (replaces individual window plots)
                visualizer.create_comprehensive_event_study_timeline(
                    ticker_list=ticker_list,
                    index_dir=visualizations_dir,
                    index_name=index_name
                )
                print("Created comprehensive event study timeline visualization")
                
                print("Created all visualizations")
            
        # Create event study visualizations if available
        for window_size, data in consolidated_aar_caar.items():
            if data["days"] and data["aar"] and data["caar"]:
                try:
                    # Create dataframe for the visualizer
                    event_df = pd.DataFrame({
                        'day': data["days"],
                        'aar': data["aar"],
                        'caar': data["caar"]
                    })
                    
                    # If we only have a single day point, skip visualization to avoid synthetic data
                    if len(data["days"]) == 1:
                        logger.warning(f"Insufficient data for {window_size}min window - only single point available. Skipping visualization to avoid synthetic data.")
                        print(f"Skipping {window_size}min window visualization - insufficient real data points")
                        continue
                    
                    # Generate plots for this window using matplotlib directly if visualizer method fails
                    try:
                        visualizer.visualize_consolidated_event_study(
                            event_df=event_df,
                            window_size=window_size,
                            index_name=index_name,
                            output_dir=event_study_dir
                        )
                        print(f"Created event study visualization for {window_size}min window using visualizer")
                    except Exception as e:
                        logger.warning(f"Error using visualizer for event study, falling back to manual plotting: {e}")
                        
                        # Create manual plots for AAR/CAAR
                        plt.figure(figsize=(14, 10))
                        
                        # Plot AAR
                        plt.subplot(2, 1, 1)
                        plt.plot(event_df["day"], event_df["aar"], 'b-', marker='o', label='AAR')
                        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                        plt.axvline(x=0, color='g', linestyle='--', alpha=0.3)
                        plt.title(f'{index_name} Average Abnormal Returns ({window_size}min window)', fontsize=14)
                        plt.xlabel('Minuten um Nachrichtenereignis (0 = Ereigniszeitpunkt)', fontsize=12)
                        plt.ylabel('AAR (%)', fontsize=12)
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        
                        # Plot CAAR
                        plt.subplot(2, 1, 2)
                        plt.plot(event_df["day"], event_df["caar"], 'g-', marker='o', label='CAAR')
                        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                        plt.axvline(x=0, color='g', linestyle='--', alpha=0.3)
                        plt.title(f'{index_name} Cumulative Average Abnormal Returns ({window_size}min window)', fontsize=14)
                        plt.xlabel('Minuten um Nachrichtenereignis (0 = Ereigniszeitpunkt)', fontsize=12)
                        plt.ylabel('CAAR (%)', fontsize=12)
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        
                        plt.tight_layout()
                        
                        # Save figure to PNG format only (no PDF)
                        plt.savefig(os.path.join(event_study_dir, f"{index_name}_event_study_{window_size}min.png"), dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Created manual event study visualization for {window_size}min window")
                except Exception as e:
                    logger.error(f"Error creating event study visualization for {window_size}min window: {e}", exc_info=True)
                    print(f"Error creating event study visualization for {window_size}min window: {e}")
        
        # Create comprehensive event study timeline if we have ticker data
        if ticker_list:
            try:
                visualizer.create_comprehensive_event_study_timeline(
                    ticker_list=ticker_list,
                    index_dir=event_study_dir,
                    index_name=index_name
                )
                print("Created comprehensive event study timeline replacing individual window plots")
            except Exception as e:
                logger.error(f"Error creating comprehensive event study timeline: {e}", exc_info=True)
                print(f"Error creating comprehensive event study timeline: {e}")
        
        # DISABLED:         # Create portfolio event study by sentiment visualization
        # DISABLED:         try:
        # DISABLED:             visualizer.create_portfolio_event_study_by_sentiment(
        # DISABLED:                 index_dir=event_study_dir,
        # DISABLED:                 index_name=index_name
        # DISABLED:             )
        # DISABLED:             print("Created portfolio event study by sentiment visualization")
        # DISABLED:         except Exception as e:
        # DISABLED:             logger.error(f"Error creating portfolio event study by sentiment: {e}", exc_info=True)
        # DISABLED:             print(f"Error creating portfolio event study by sentiment: {e}")
        
        # DISABLED:         # Create portfolio event study (JPM-style) visualization
        # DISABLED:         try:
        # DISABLED:             visualizer.create_portfolio_event_study_sentiment_styled(
        # DISABLED:                 index_dir=event_study_dir,
        # DISABLED:                 index_name=index_name
        # DISABLED:             )
        # DISABLED:             print("Created portfolio event study (JPM-style) visualization")
        # DISABLED:         except Exception as e:
        # DISABLED:             logger.error(f"Error creating portfolio event study (JPM-style): {e}", exc_info=True)
        # DISABLED:             print(f"Error creating portfolio event study (JPM-style): {e}")
        
            except Exception as e:
                logger.error(f"Error creating trades timeline: {e}", exc_info=True)
                print(f"Error creating trades timeline: {e}")
        
        # Custom: Create sentiment distribution pie chart
        if sum(sentiment_counts.values()) > 0:
            try:
                labels = ['Positive', 'Negative', 'Neutral']
                sizes = [sentiment_counts['positive'], sentiment_counts['negative'], sentiment_counts['neutral']]
                colors = ['#4CAF50', '#F44336', '#2196F3']
                
                plt.figure(figsize=(10, 8))
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, shadow=False)
                plt.axis('equal')
                plt.title(f'{index_name} Sentiment Distribution', fontsize=14)
                
                # Save figure
                pie_path = os.path.join(visualizations_dir, f"{index_name}_sentiment_distribution.png")
                plt.savefig(pie_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Created sentiment distribution visualization")
            except Exception as e:
                logger.error(f"Error creating sentiment distribution: {e}", exc_info=True)
                print(f"Error creating sentiment distribution: {e}")
    
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}", exc_info=True)
        print(f"Error creating visualizations: {e}")
    
    # --- PORTFOLIO RANDOM TRADING BENCHMARK ANALYSIS ---
    try:
        print("\n" + "="*80)
        print(f" PORTFOLIO RANDOM TRADING BENCHMARK FOR {index_name} ".center(80, "="))
        print("="*80 + "\n")
        
        logger.info(f"Running portfolio-level random trading benchmark for {index_name}")
        print(f"Running Portfolio Random Trading Benchmark with 1,000 simulations...")
        print(f"Methodology: Identical entry/exit times, randomized trade directions")
        
        # Check if trades file exists
        trades_file = os.path.join(trades_dir, f"{index_name}_all_trades.csv")
        
        if os.path.exists(trades_file) and os.path.getsize(trades_file) > 0:
            # Run portfolio random benchmark analysis
            portfolio_random_results = run_portfolio_random_benchmark_analysis(
                all_trades_file=trades_file,
                index_name=index_name,
                output_dir=index_analysis_dir,
                n_simulations=1000
            )
            
            if portfolio_random_results and 'portfolio_vs_random_comparison' in portfolio_random_results:
                comparison = portfolio_random_results['portfolio_vs_random_comparison']
                strategy_perf = comparison.get('strategy_performance', {})
                random_perf = comparison.get('random_performance', {})
                vs_random = comparison.get('strategy_vs_random', {})
                significance = comparison.get('statistical_significance', {})
                insights = comparison.get('key_insights', {})
                
                print(f"\n🎯 Portfolio Random Benchmark Results:")
                print(f"   Strategy Total Return: {strategy_perf.get('total_return_pct', 0):.2f}%")
                print(f"   Random Mean Return: {random_perf.get('mean_total_return', 0):.2f}%")
                print(f"   Strategy Percentile: {vs_random.get('return_percentile_rank', 0):.1f}th")
                print(f"   Outperformed Simulations: {vs_random.get('outperformed_simulations', 0):,}/{vs_random.get('total_simulations', 0):,}")
                
                print(f"\n📊 Direction Accuracy Analysis:")
                print(f"   Sentiment Prediction Accuracy: {strategy_perf.get('sentiment_direction_accuracy', 0)*100:.1f}%")
                print(f"   Random Direction Accuracy: {random_perf.get('mean_direction_accuracy', 0.5)*100:.1f}%")
                print(f"   Direction Advantage: +{vs_random.get('direction_accuracy_advantage', 0)*100:.1f} percentage points")
                
                print(f"\n📈 Statistical Significance:")
                print(f"   P-value: {significance.get('p_value_return', 1):.6f}")
                print(f"   Z-score: {significance.get('z_score', 0):.4f}")
                print(f"   Significant at 5%: {'✓ YES' if significance.get('is_significant_5pct', False) else '✗ NO'}")
                print(f"   Significant at 1%: {'✓ YES' if significance.get('is_significant_1pct', False) else '✗ NO'}")
                
                portfolio_random_dir = os.path.join(index_analysis_dir, "portfolio_random_benchmark")
                print(f"\n📁 Detailed results saved to: {portfolio_random_dir}")
                print(f"   - portfolio_random_benchmark.json (detailed data)")
                print(f"   - portfolio_benchmark_summary.txt (human-readable)")
                
                # Add to portfolio summary
                portfolio_summary["portfolio_random_benchmark"] = {
                    "strategy_total_return_pct": strategy_perf.get('total_return_pct', 0),
                    "random_mean_return_pct": random_perf.get('mean_total_return', 0),
                    "strategy_percentile_rank": vs_random.get('return_percentile_rank', 0),
                    "outperformed_simulations": vs_random.get('outperformed_simulations', 0),
                    "total_simulations": vs_random.get('total_simulations', 0),
                    "p_value": significance.get('p_value_return', 1),
                    "significant_at_5pct": significance.get('is_significant_5pct', False),
                    "significant_at_1pct": significance.get('is_significant_1pct', False),
                    "sentiment_direction_accuracy": strategy_perf.get('sentiment_direction_accuracy', 0),
                    "direction_accuracy_advantage": vs_random.get('direction_accuracy_advantage', 0),
                    "methodology": "Identical entry/exit times with randomized trade directions"
                }
                
                # Save updated portfolio summary with random benchmark results
                save_json(portfolio_summary, os.path.join(summaries_dir, f"{index_name}_index_summary.json"))
                print(f"✅ Updated index summary with portfolio random benchmark results")
                
            else:
                logger.warning(f"No portfolio random benchmark results generated")
                print(f"⚠️  Portfolio random benchmark analysis failed. Check logs for details.")
        else:
            logger.warning(f"No trades file found at {trades_file} or file is empty")
            print(f"⚠️  No trades data available for portfolio random benchmark analysis.")
            
    except ImportError as e:
        logger.error(f"Portfolio random benchmark module not available: {e}")
        print(f"⚠️  Portfolio random benchmark analysis not available. Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"Error running portfolio random benchmark analysis: {e}", exc_info=True)
        print(f"❌ Error in portfolio random benchmark analysis: {e}")
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PORTFOLIO REPORT FOR {index_name} COMPLETED")
    print(f"Output directory: {index_analysis_dir}")
    print(f"{'='*80}\n")
    
    # Create consolidated filter statistics for the index
    print(f"Creating consolidated filter statistics for {index_name}...")
    create_index_filter_statistics(output_dir, ticker_list, index_name)
    
    # NEU: Portfolio Event Study Statistical Analysis
    try:
        from src.analysis.portfolio_event_study_statistics import run_portfolio_statistics_analysis
        
        print("\n" + "="*80)
        print(f" PORTFOLIO EVENT STUDY STATISTICAL ANALYSIS FOR {index_name} ".center(80, "="))
        print("="*80)
        
        logger.info(f"Running comprehensive portfolio statistical analysis for {index_name}")
        print(f"Running Portfolio Statistical Analysis...")
        print(f"- T-Tests for AAR/CAAR significance")
        print(f"- Effect Size Analysis (Cohen's d)")
        print(f"- Model Fit Statistics (R², RMSE)")
        print(f"- Cross-Sectional Analysis")
        print(f"- Consistency Measures")
        
        # Run portfolio statistics analysis
        portfolio_stats_results = run_portfolio_statistics_analysis(output_dir, index_name)
        
        if portfolio_stats_results:
            print(f"\n📊 PORTFOLIO EVENT STUDY STATISTICAL RESULTS:")
            
            # Show meta statistics
            meta = portfolio_stats_results.get('meta_statistics', {})
            print(f"   Stocks Analyzed: {meta.get('total_stocks_analyzed', 0)}")
            print(f"   Total News Events: {meta.get('total_events', 0)}")
            print(f"   Event Distribution: {meta.get('event_distribution', {})}")
            
            # Show significant t-test results
            t_tests = portfolio_stats_results.get('portfolio_t_tests', {}).get('aar_tests', {})
            significant_results = []
            
            print(f"\n🔬 STATISTICAL SIGNIFICANCE TESTS (AAR T-Tests):")
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in t_tests:
                    print(f"   {sentiment.upper()} Sentiment:")
                    for window, test_result in t_tests[sentiment].items():
                        p_val = test_result.get('p_value', 1.0)
                        t_stat = test_result.get('t_statistic', 0.0)
                        significant = test_result.get('significant_5pct', False)
                        sig_marker = " ✓ SIGNIFICANT" if significant else " ✗ NOT SIGNIFICANT"
                        print(f"     {window}min: t={t_stat:.3f}, p={p_val:.4f}{sig_marker}")
                        
                        if significant:
                            significant_results.append(f"{sentiment} {window}min")
            
            # Show effect sizes
            effect_sizes = portfolio_stats_results.get('effect_sizes', {})
            print(f"\n📈 EFFECT SIZES (Cohen's d vs Neutral):")
            for sentiment in ['positive', 'negative']:
                if sentiment in effect_sizes:
                    print(f"   {sentiment.upper()} vs Neutral:")
                    for window, results in effect_sizes[sentiment].items():
                        cohens_d = results.get('cohens_d', 0)
                        interpretation = results.get('effect_interpretation', 'unknown')
                        print(f"     {window}min: d={cohens_d:.3f} ({interpretation})")
            
            # Show model fit
            model_fit = portfolio_stats_results.get('model_fit', {}).get('sentiment_prediction', {})
            print(f"\n🎯 MODEL FIT STATISTICS:")
            if model_fit:
                print(f"   Sentiment Prediction Model (R²):")
                for window, results in model_fit.items():
                    r2 = results.get('r_squared', 0)
                    p_val = results.get('regression', {}).get('p_value', 1)
                    print(f"     {window}min: R²={r2:.4f}, p={p_val:.4f}")
            
            # Summary
            if significant_results:
                print(f"\n✅ STATISTICALLY SIGNIFICANT RESULTS FOUND:")
                for result in significant_results:
                    print(f"   - {result}")
            else:
                print(f"\n⚠️  NO STATISTICALLY SIGNIFICANT RESULTS at p < 0.05 level")
            
            print(f"\n📁 Detailed portfolio statistics saved to:")
            print(f"   - {index_analysis_dir}/portfolio_statistics.json (detailed data)")
            print(f"   - {index_analysis_dir}/portfolio_statistics_summary.txt (summary report)")
            
        else:
            logger.warning(f"Portfolio statistical analysis returned no results for {index_name}")
            print(f"⚠️  Portfolio statistical analysis returned no results.")
            
    except ImportError as e:
        logger.error(f"Portfolio statistics module not available: {e}")
        print(f"⚠️  Portfolio statistical analysis not available. Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"Error running portfolio statistical analysis: {e}", exc_info=True)
        print(f"❌ Error in portfolio statistical analysis: {e}")
    
    return index_analysis_dir

def create_correct_portfolio_event_study(ticker_list: List[str], output_dir: str, index_name: str = "Portfolio"):
    """
    Create the correct portfolio event study using ALL ticker events,
    not the buggy 168 events limitation from the consolidated function.
    """
    from backtesting.analysis.backtest_visualizer import BacktestVisualizer
    
    event_study_dir = os.path.join(output_dir, f"{index_name}_INDEX_ANALYSIS", "event_study")
    
    # Load all individual ticker event study results
    all_events_data = {}
    total_events_by_window = {}
    
    # Initialize for each window
    for window in [5, 15, 30, 60]:
        all_events_data[window] = {
            'positive': {'aar': [], 'events': 0},
            'negative': {'aar': [], 'events': 0}, 
            'neutral': {'aar': [], 'events': 0},
            'all': {'aar': [], 'events': 0}
        }
    
    successful_tickers = 0
    total_events_processed = 0
    
    for ticker in ticker_list:
        event_file = os.path.join(output_dir, ticker, "event_study", f"{ticker}_event_study_results.json")
        if os.path.exists(event_file):
            try:
                with open(event_file, 'r') as f:
                    ticker_data = json.load(f)
                
                total_events_processed += ticker_data.get('unique_news_events', ticker_data.get('total_events', 0))
                successful_tickers += 1
                
                # Process each sentiment and window
                for sentiment in ['positive', 'negative', 'neutral']:
                    if sentiment in ticker_data.get('aar_caar', {}):
                        sentiment_data = ticker_data['aar_caar'][sentiment]
                        
                        for window in [5, 15, 30, 60]:
                            window_str = str(window)
                            if 'AAR' in sentiment_data and window_str in sentiment_data['AAR']:
                                aar_value = sentiment_data['AAR'][window_str]
                                event_count = sentiment_data.get('count', {}).get(window_str, 0)
                                
                                # Add weighted by event count
                                for _ in range(int(event_count)):
                                    all_events_data[window][sentiment]['aar'].append(aar_value)
                                    all_events_data[window]['all']['aar'].append(aar_value)
                                
                                all_events_data[window][sentiment]['events'] += int(event_count)
                                all_events_data[window]['all']['events'] += int(event_count)
                                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
    
    # Calculate consolidated statistics
    consolidated_results = {}
    
    for window in [5, 15, 30, 60]:
        window_data = all_events_data[window]['all']
        
        if len(window_data['aar']) > 0:
            aar_array = np.array(window_data['aar'])
            
            # Calculate statistics
            mean_aar = np.mean(aar_array)
            std_aar = np.std(aar_array, ddof=1)
            n_events = len(aar_array)
            
            # T-test against zero
            if n_events > 1 and std_aar > 0:
                t_stat = mean_aar / (std_aar / np.sqrt(n_events))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_events - 1))
            else:
                t_stat = 0
                p_value = 1
            
            consolidated_results[window] = {
                "days": [-3, -2, -1, 0, 1, 2, 3],
                "aar": [0.0, 0.0, 0.0, float(mean_aar), 0.0, 0.0, 0.0],
                "aar_std": [0.0, 0.0, 0.0, float(std_aar), 0.0, 0.0, 0.0],
                "caar": [0.0, 0.0, 0.0, float(mean_aar), float(mean_aar), float(mean_aar), float(mean_aar)],
                "t_stats": [None, None, None, float(t_stat), None, None, None],
                "p_values": [None, None, None, float(p_value), None, None, None],
                "num_events": n_events
            }
    
    # Save corrected results
    corrected_file = os.path.join(event_study_dir, f"{index_name}_CORRECTED_consolidated_aar_caar.json")
    save_json(consolidated_results, corrected_file)
    
    print(f"✅ CORRECTED Portfolio Event Study:")
    print(f"   📊 {successful_tickers} tickers processed")
    print(f"   🎯 {total_events_processed} unique news events in source data")
    
    for window in [5, 15, 30, 60]:
        if window in consolidated_results:
            data = consolidated_results[window]
            aar_pct = data['aar'][3] * 100
            t_stat = data['t_stats'][3]
            p_val = data['p_values'][3]
            n_events = data['num_events']
            significance = "✅ Significant" if p_val < 0.05 else "❌ Not Significant"
            
            print(f"   {window}min: AAR={aar_pct:.4f}%, t={t_stat:.3f}, p={p_val:.3f}, n={n_events} {significance}")
    
    return corrected_file

def save_news_filter_statistics(output_dir: str, ticker: str = None):
    """
    Save comprehensive news filtering statistics.
    
    Args:
        output_dir: Output directory for the statistics
        ticker: Optional ticker name for individual ticker stats
    """
    try:
        if ticker:
            filename = f"{ticker}_news_filter_statistics.json"
        else:
            filename = "news_filter_statistics.json"
        
        filepath = os.path.join(output_dir, filename)
        filter_tracker.save_to_file(filepath)
        
        # Also create a human-readable summary
        summary = filter_tracker.get_summary()
        summary_filename = filename.replace('.json', '_summary.txt')
        summary_filepath = os.path.join(output_dir, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            f.write("NEWS FILTERING STATISTICS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            stats = summary['filtering_statistics']
            f.write(f"📊 OVERALL STATISTICS:\n")
            f.write(f"Total news fetched from API: {stats['total_fetched_from_api']:,}\n")
            f.write(f"Total news removed by filters: {summary['total_news_removed']:,}\n")
            f.write(f"Final valid news events: {stats['final_valid_news']:,}\n")
            f.write(f"Final trades executed: {stats['final_trades_executed']:,}\n\n")
            
            f.write(f"🔄 CONVERSION RATES:\n")
            f.write(f"News to Trades: {summary['conversion_rate_news_to_trades']:.2f}%\n")
            f.write(f"Overall API to Trades: {summary['overall_conversion_rate']:.2f}%\n\n")
            
            f.write(f"🚫 FILTERING BREAKDOWN:\n")
            f.write(f"Filtered by source restrictions: {stats['filtered_by_source']:,}\n")
            f.write(f"Filtered by missing data: {stats['filtered_by_missing_data']:,}\n")
            f.write(f"Filtered by neutral sentiment: {stats['filtered_by_neutral_sentiment']:,}\n")
            f.write(f"Filtered by after-hours timing: {stats['filtered_by_after_hours']:,}\n")
            f.write(f"Filtered by weekend timing: {stats['filtered_by_weekend']:,}\n")
            f.write(f"Filtered by missing market data: {stats['filtered_by_missing_market_data']:,}\n\n")
            
            f.write(f"📝 DETAILED LOGS:\n")
            for log_entry in summary['detailed_logs']:
                f.write(f"- {log_entry}\n")
        
        logger.info(f"News filter statistics saved to {filepath} and {summary_filepath}")
        
    except Exception as e:
        logger.error(f"Error saving news filter statistics: {e}", exc_info=True)

def create_index_filter_statistics(output_dir: str, ticker_list: List[str], index_name: str):
    """
    Create consolidated filter statistics for the entire index.
    
    Args:
        output_dir: Output directory containing individual ticker results
        ticker_list: List of tickers to analyze
        index_name: Name of the index (e.g., "TOP5", "LOG")
    """
    try:
        # Initialize consolidated statistics
        consolidated_stats = {
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
        
        ticker_details = []
        
        # Collect statistics from each ticker
        for ticker in ticker_list:
            ticker_dir = os.path.join(output_dir, ticker)
            filter_stats_file = os.path.join(ticker_dir, f"{ticker}_news_filter_statistics.json")
            
            if os.path.exists(filter_stats_file):
                try:
                    with open(filter_stats_file, 'r') as f:
                        ticker_stats = json.load(f)
                    
                    # Add to consolidated stats
                    stats = ticker_stats.get('filtering_statistics', {})
                    for key in consolidated_stats.keys():
                        consolidated_stats[key] += stats.get(key, 0)
                    
                    # Store ticker details
                    ticker_details.append({
                        'ticker': ticker,
                        'stats': stats,
                        'conversion_rates': {
                            'news_to_trades': ticker_stats.get('conversion_rate_news_to_trades', 0),
                            'overall_conversion': ticker_stats.get('overall_conversion_rate', 0)
                        }
                    })
                    
                except Exception as e:
                    logger.warning(f"Could not load filter statistics for {ticker}: {e}")
        
        # Calculate consolidated conversion rates
        total_removed = (
            consolidated_stats['filtered_by_source'] +
            consolidated_stats['filtered_by_missing_data'] +
            consolidated_stats['filtered_by_neutral_sentiment'] +
            consolidated_stats['filtered_by_after_hours'] +
            consolidated_stats['filtered_by_weekend'] +
            consolidated_stats['filtered_by_missing_market_data']
        )
        
        consolidated_summary = {
            'index_name': index_name,
            'total_tickers': len(ticker_list),
            'tickers_with_data': len(ticker_details),
            'filtering_statistics': consolidated_stats,
            'total_news_removed': total_removed,
            'conversion_rate_news_to_trades': (
                consolidated_stats['final_trades_executed'] / max(consolidated_stats['final_valid_news'], 1) * 100
            ),
            'overall_conversion_rate': (
                consolidated_stats['final_trades_executed'] / max(consolidated_stats['total_fetched_from_api'], 1) * 100
            ),
            'ticker_details': ticker_details
        }
        
        # Save consolidated statistics
        index_analysis_dir = os.path.join(output_dir, f"{index_name}_INDEX_ANALYSIS")
        os.makedirs(index_analysis_dir, exist_ok=True)
        
        # JSON file
        json_filepath = os.path.join(index_analysis_dir, f"{index_name}_consolidated_filter_statistics.json")
        with open(json_filepath, 'w') as f:
            json.dump(consolidated_summary, f, indent=2)
        
        # Human-readable summary
        summary_filepath = os.path.join(index_analysis_dir, f"{index_name}_filter_statistics_summary.txt")
        with open(summary_filepath, 'w') as f:
            f.write(f"{index_name} INDEX - NEWS FILTERING STATISTICS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"📊 INDEX OVERVIEW:\n")
            f.write(f"Index Name: {index_name}\n")
            f.write(f"Total Tickers: {len(ticker_list)}\n")
            f.write(f"Tickers with Data: {len(ticker_details)}\n\n")
            
            f.write(f"📈 OVERALL STATISTICS:\n")
            f.write(f"Total news fetched from API: {consolidated_stats['total_fetched_from_api']:,}\n")
            f.write(f"Total news removed by filters: {total_removed:,}\n")
            f.write(f"Final valid news events: {consolidated_stats['final_valid_news']:,}\n")
            f.write(f"Final trades executed: {consolidated_stats['final_trades_executed']:,}\n\n")
            
            f.write(f"🔄 CONVERSION RATES:\n")
            f.write(f"Valid News to Trades: {consolidated_summary['conversion_rate_news_to_trades']:.2f}%\n")
            f.write(f"Overall API to Trades: {consolidated_summary['overall_conversion_rate']:.2f}%\n\n")
            
            f.write(f"🚫 FILTERING BREAKDOWN:\n")
            f.write(f"Filtered by source restrictions: {consolidated_stats['filtered_by_source']:,}\n")
            f.write(f"Filtered by missing data: {consolidated_stats['filtered_by_missing_data']:,}\n")
            f.write(f"Filtered by neutral sentiment: {consolidated_stats['filtered_by_neutral_sentiment']:,}\n")
            f.write(f"Filtered by after-hours timing: {consolidated_stats['filtered_by_after_hours']:,}\n")
            f.write(f"Filtered by weekend timing: {consolidated_stats['filtered_by_weekend']:,}\n")
            f.write(f"Filtered by missing market data: {consolidated_stats['filtered_by_missing_market_data']:,}\n\n")
            
            f.write(f"📋 TICKER BREAKDOWN:\n")
            for ticker_detail in ticker_details:
                ticker = ticker_detail['ticker']
                stats = ticker_detail['stats']
                rates = ticker_detail['conversion_rates']
                f.write(f"\n{ticker}:\n")
                f.write(f"  API Fetched: {stats.get('total_fetched_from_api', 0):,}\n")
                f.write(f"  Valid News: {stats.get('final_valid_news', 0):,}\n")
                f.write(f"  Trades: {stats.get('final_trades_executed', 0):,}\n")
                f.write(f"  Conversion Rate: {rates['news_to_trades']:.2f}%\n")
        
        logger.info(f"Index filter statistics saved to {json_filepath} and {summary_filepath}")
        print(f"📊 Index filter statistics saved for {index_name}")
        
    except Exception as e:
        logger.error(f"Error creating index filter statistics: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        return_code = main()
        sys.exit(return_code if isinstance(return_code, int) else 0)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        logger.warning("Program terminated by user (KeyboardInterrupt).")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnhandled error in __main__: {e}")
        logger.critical("Unhandled exception in __main__", exc_info=True)
        sys.exit(1)  