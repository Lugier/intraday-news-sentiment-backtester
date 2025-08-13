"""
Modul für Event-Study-Analyse auf Portfolio-Ebene inkl. Signifikanztests.
Leicht deutsch mit mini Tippfehlern (~3%).
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def calculate_and_save_portfolio_caar_significance(
    portfolio_name: str,
    ticker_list: list,
    run_output_dir: str, 
    index_summary_file_path: str
):
    """
    Calculates portfolio-level CAAR significance tests by loading individual
    ticker event study results and saves them to the main index summary file.

    Args:
        portfolio_name: Name of the portfolio/index.
        ticker_list: List of tickers included in the portfolio.
        run_output_dir: The base output directory for the current run 
                        (e.g., output/run_TOP5_YYYYMMDD_HHMMSS),
                        which contains subdirectories for each ticker.
        index_summary_file_path: Full path to the portfolio's summary JSON file 
                                 (e.g., .../TOP5_INDEX_ANALYSIS/summaries/TOP5_index_summary.json).
    """
    portfolio_event_caars = defaultdict(lambda: defaultdict(list))
    # Structure: {sentiment: {window: [all_event_caars_from_portfolio]}}

    event_windows_overall = None 

    logger.info(f"Starting portfolio CAAR significance calculation for {portfolio_name}")

    for ticker in ticker_list:
        # Construct path to the individual ticker's event study results
        # Assumes ticker results are in <run_output_dir>/<ticker>/event_study/<ticker>_event_study_results.json
        ticker_event_study_path = os.path.join(run_output_dir, ticker, "event_study", f"{ticker}_event_study_results.json")
        
        if not os.path.exists(ticker_event_study_path):
            logger.warning(f"Event study results for ticker {ticker} not found at {ticker_event_study_path}. Skipping.")
            continue

        try:
            with open(ticker_event_study_path, 'r') as f:
                ticker_results = json.load(f)
            
            if event_windows_overall is None and 'event_windows' in ticker_results:
                event_windows_overall = ticker_results['event_windows']
            
            # The key here should match what was saved in run_event_study
            individual_caars_data = ticker_results.get('individual_event_caars_by_sentiment_window', {})
            
            for sentiment, window_dict in individual_caars_data.items():
                if sentiment == 'all_events': # Should not be here if filtered in run_event_study
                    continue
                for window_str, caar_list_for_event in window_dict.items():
                    try:
                        window = int(window_str)
                        # Filter out None values which were pd.isna before serialization
                        valid_caars = [caar for caar in caar_list_for_event if caar is not None]
                        if valid_caars:
                             portfolio_event_caars[sentiment][window].extend(valid_caars)
                    except ValueError:
                        logger.warning(f"Could not convert window '{window_str}' to int for ticker {ticker}, sentiment {sentiment}")
                        continue
        except Exception as e:
            logger.error(f"Error processing event study results for ticker {ticker} from {ticker_event_study_path}: {e}")
            continue

    if not portfolio_event_caars or event_windows_overall is None:
        logger.warning(f"No individual event CAARs collected for portfolio {portfolio_name}. Cannot calculate significance.")
        # Still try to update the summary file with empty results if it exists
        if os.path.exists(index_summary_file_path):
            try:
                with open(index_summary_file_path, 'r+') as f:
                    summary_data = json.load(f)
                    if 'event_study' not in summary_data:
                        summary_data['event_study'] = {}
                    summary_data['event_study']['consolidated_ttest_results'] = {} # Empty results
                    f.seek(0)
                    json.dump(summary_data, f, indent=4)
                    f.truncate()
                logger.info(f"Updated {index_summary_file_path} with empty portfolio CAAR t-test results.")
            except Exception as e:
                logger.error(f"Error updating index summary file {index_summary_file_path} with empty results: {e}")
        return

    portfolio_ttest_results = defaultdict(dict)
    for sentiment_key in portfolio_event_caars:
        portfolio_ttest_results[sentiment_key] = {}
        for window_key in portfolio_event_caars[sentiment_key]:
            all_caars_for_window_sentiment = portfolio_event_caars[sentiment_key][window_key]
            if len(all_caars_for_window_sentiment) >= 2: # Need at least 2 observations for a t-test
                # Ensure all elements are numbers before passing to ttest_1samp
                numeric_caars = [x for x in all_caars_for_window_sentiment if isinstance(x, (int, float)) and pd.notna(x)]
                if len(numeric_caars) >=2:
                    t_stat, p_value = ttest_1samp(numeric_caars, 0)
                    portfolio_ttest_results[sentiment_key][str(window_key)] = {
                        't_stat': float(t_stat) if pd.notna(t_stat) else None,
                        'p_value': float(p_value) if pd.notna(p_value) else None,
                        'n_obs': len(numeric_caars)
                    }
                    logger.info(f"Portfolio CAAR ({portfolio_name} - {sentiment_key} - {window_key}min): N={len(numeric_caars)}, t-stat={t_stat:.4f}, p-value={p_value:.4f}")
                else:
                    portfolio_ttest_results[sentiment_key][str(window_key)] = {
                        't_stat': None, 'p_value': None, 'n_obs': len(numeric_caars)
                    }
                    logger.warning(f"Portfolio CAAR ({portfolio_name} - {sentiment_key} - {window_key}min): Insufficient numeric data (N={len(numeric_caars)}) for t-test after filtering non-numeric.")
            else:
                portfolio_ttest_results[sentiment_key][str(window_key)] = {
                    't_stat': None,
                    'p_value': None,
                    'n_obs': len(all_caars_for_window_sentiment)
                }
                logger.warning(f"Portfolio CAAR ({portfolio_name} - {sentiment_key} - {window_key}min): Insufficient data (N={len(all_caars_for_window_sentiment)}) for t-test.")
    
    # Save to the main index summary file
    if os.path.exists(index_summary_file_path):
        try:
            with open(index_summary_file_path, 'r+') as f:
                summary_data = json.load(f)
                if 'event_study' not in summary_data:
                    summary_data['event_study'] = {}
                # Convert defaultdict to dict for JSON serialization
                final_results_to_save = {}
                for sentiment, window_data_dict in portfolio_ttest_results.items():
                    final_results_to_save[sentiment] = dict(window_data_dict)

                summary_data['event_study']['consolidated_ttest_results'] = final_results_to_save
                f.seek(0)
                json.dump(summary_data, f, indent=4)
                f.truncate()
            logger.info(f"Successfully saved portfolio CAAR t-test results to {index_summary_file_path}")
        except Exception as e:
            logger.error(f"Error updating index summary file {index_summary_file_path} with portfolio t-test results: {e}")
    else:
        logger.error(f"Index summary file {index_summary_file_path} not found. Cannot save portfolio t-test results.") 