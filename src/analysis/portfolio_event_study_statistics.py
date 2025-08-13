"""
Portfolio Event Study Statistical Analysis

This module calculates comprehensive statistical measures for portfolio-level event study results,
including t-tests for significance, R-squared for explanatory power, RMSE for model fit,
and other robustness indicators.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PortfolioEventStudyStatistics:
    """
    Calculates comprehensive statistical measures for portfolio-level event study results.
    """
    
    def __init__(self, event_study_file: str, individual_results_dir: str):
        """
        Initialize with portfolio event study results and individual stock results.
        
        Args:
            event_study_file: Path to portfolio event study JSON file
            individual_results_dir: Directory containing individual stock event study results
        """
        self.event_study_file = event_study_file
        self.individual_results_dir = individual_results_dir
        self.portfolio_data = None
        self.individual_data = {}
        self.statistics = {}
        
    def load_data(self):
        """Load portfolio and individual event study data."""
        # Load portfolio data
        with open(self.event_study_file, 'r') as f:
            self.portfolio_data = json.load(f)
            
        # Load individual stock data
        # Look for event study files in subdirectories like TICKER/event_study/TICKER_event_study_results.json
        for item in os.listdir(self.individual_results_dir):
            item_path = os.path.join(self.individual_results_dir, item)
            if os.path.isdir(item_path) and not item.endswith('_INDEX_ANALYSIS'):
                # This might be a ticker directory
                ticker = item
                possible_paths = [
                    os.path.join(item_path, 'event_study', f'{ticker}_event_study_results.json'),
                    os.path.join(item_path, f'{ticker}_event_study_results.json')
                ]
                
                for event_study_path in possible_paths:
                    if os.path.exists(event_study_path):
                        try:
                            with open(event_study_path, 'r') as f:
                                self.individual_data[ticker] = json.load(f)
                            logger.info(f"Loaded event study data for {ticker}")
                            break
                        except Exception as e:
                            logger.warning(f"Could not load {event_study_path}: {e}")
        
        logger.info(f"Loaded individual event study data for {len(self.individual_data)} tickers: {list(self.individual_data.keys())}")
                    
    def calculate_portfolio_t_tests(self) -> Dict[str, Dict]:
        """
        Calculate t-tests for portfolio-level AAR and CAAR values.
        
        Returns:
            Dictionary with t-test results for each sentiment and time window
        """
        results = {
            'aar_tests': {},
            'caar_tests': {}
        }
        
        sentiments = self.portfolio_data.get('sentiments', ['positive', 'negative', 'neutral'])
        time_windows = self.portfolio_data.get('timeline_minutes', [5, 15, 30, 60])
        
        for sentiment in sentiments:
            results['aar_tests'][sentiment] = {}
            results['caar_tests'][sentiment] = {}
            
            # Get portfolio-level data
            portfolio_aar = self.portfolio_data['aar_data'].get(sentiment, [])
            portfolio_caar = self.portfolio_data['caar_data'].get(sentiment, [])
            event_count = self.portfolio_data['event_counts'].get(sentiment, 0)
            
            # Collect individual stock AARs for each time window
            for i, window in enumerate(time_windows):
                window_str = str(window)
                
                # Collect individual AARs for this sentiment and window
                individual_aars = []
                individual_caars = []
                
                for ticker, data in self.individual_data.items():
                    if sentiment in data.get('aar_caar', {}):
                        aar_value = data['aar_caar'][sentiment].get('AAR', {}).get(window_str)
                        caar_value = data['aar_caar'][sentiment].get('CAAR', {}).get(window_str)
                        
                        if aar_value is not None and not np.isnan(aar_value):
                            individual_aars.append(aar_value)
                        if caar_value is not None and not np.isnan(caar_value):
                            individual_caars.append(caar_value)
                
                # Calculate AAR t-test
                if len(individual_aars) > 1:
                    aar_mean = np.mean(individual_aars)
                    aar_std = np.std(individual_aars, ddof=1)
                    aar_se = aar_std / np.sqrt(len(individual_aars))
                    aar_t_stat = aar_mean / aar_se if aar_se > 0 else 0
                    aar_p_value = 2 * (1 - stats.t.cdf(abs(aar_t_stat), len(individual_aars) - 1))
                    
                    results['aar_tests'][sentiment][window] = {
                        'mean': aar_mean,
                        'std': aar_std,
                        'se': aar_se,
                        't_statistic': aar_t_stat,
                        'p_value': aar_p_value,
                        'significant_5pct': aar_p_value < 0.05,
                        'significant_1pct': aar_p_value < 0.01,
                        'n_stocks': len(individual_aars),
                        'portfolio_aar': portfolio_aar[i] if i < len(portfolio_aar) else None
                    }
                
                # Calculate CAAR t-test
                if len(individual_caars) > 1:
                    caar_mean = np.mean(individual_caars)
                    caar_std = np.std(individual_caars, ddof=1)
                    caar_se = caar_std / np.sqrt(len(individual_caars))
                    caar_t_stat = caar_mean / caar_se if caar_se > 0 else 0
                    caar_p_value = 2 * (1 - stats.t.cdf(abs(caar_t_stat), len(individual_caars) - 1))
                    
                    results['caar_tests'][sentiment][window] = {
                        'mean': caar_mean,
                        'std': caar_std,
                        'se': caar_se,
                        't_statistic': caar_t_stat,
                        'p_value': caar_p_value,
                        'significant_5pct': caar_p_value < 0.05,
                        'significant_1pct': caar_p_value < 0.01,
                        'n_stocks': len(individual_caars),
                        'portfolio_caar': portfolio_caar[i] if i < len(portfolio_caar) else None
                    }
        
        return results
    
    def calculate_cross_sectional_tests(self) -> Dict[str, Any]:
        """
        Calculate cross-sectional tests to examine the distribution of abnormal returns
        across stocks for each event type.
        
        Returns:
            Dictionary with cross-sectional test results
        """
        results = {}
        sentiments = self.portfolio_data.get('sentiments', ['positive', 'negative', 'neutral'])
        time_windows = self.portfolio_data.get('timeline_minutes', [5, 15, 30, 60])
        
        for sentiment in sentiments:
            results[sentiment] = {}
            
            for window in time_windows:
                window_str = str(window)
                
                # Collect individual stock AARs for this sentiment and window
                individual_aars = []
                stock_names = []
                
                for ticker, data in self.individual_data.items():
                    if sentiment in data.get('aar_caar', {}):
                        aar_value = data['aar_caar'][sentiment].get('AAR', {}).get(window_str)
                        if aar_value is not None and not np.isnan(aar_value):
                            individual_aars.append(aar_value)
                            stock_names.append(ticker)
                
                if len(individual_aars) > 2:
                    # Kolmogorov-Smirnov test for normality
                    ks_stat, ks_p_value = stats.kstest(individual_aars, 'norm', 
                                                      args=(np.mean(individual_aars), np.std(individual_aars)))
                    
                    # Shapiro-Wilk test for normality
                    if len(individual_aars) <= 5000:  # Shapiro-Wilk has sample size limitations
                        sw_stat, sw_p_value = stats.shapiro(individual_aars)
                    else:
                        sw_stat, sw_p_value = None, None
                    
                    # Calculate percentiles for distribution analysis
                    percentiles = np.percentile(individual_aars, [5, 10, 25, 50, 75, 90, 95])
                    
                    # Test for significance of positive vs negative AARs
                    positive_aars = [x for x in individual_aars if x > 0]
                    negative_aars = [x for x in individual_aars if x < 0]
                    
                    results[sentiment][window] = {
                        'n_stocks': len(individual_aars),
                        'mean': np.mean(individual_aars),
                        'std': np.std(individual_aars, ddof=1),
                        'min': np.min(individual_aars),
                        'max': np.max(individual_aars),
                        'percentiles': {
                            '5th': percentiles[0],
                            '10th': percentiles[1],
                            '25th': percentiles[2],
                            '50th': percentiles[3],
                            '75th': percentiles[4],
                            '90th': percentiles[5],
                            '95th': percentiles[6]
                        },
                        'n_positive': len(positive_aars),
                        'n_negative': len(negative_aars),
                        'pct_positive': len(positive_aars) / len(individual_aars) * 100,
                        'normality_tests': {
                            'ks_statistic': ks_stat,
                            'ks_p_value': ks_p_value,
                            'sw_statistic': sw_stat,
                            'sw_p_value': sw_p_value,
                            'normal_at_5pct': ks_p_value > 0.05 if ks_p_value else None
                        }
                    }
        
        return results
    
    def calculate_consistency_measures(self) -> Dict[str, Any]:
        """
        Calculate measures of consistency across stocks and time windows.
        
        Returns:
            Dictionary with consistency measures
        """
        results = {
            'sentiment_consistency': {},
            'time_window_consistency': {},
            'overall_consistency': {}
        }
        
        sentiments = self.portfolio_data.get('sentiments', ['positive', 'negative', 'neutral'])
        time_windows = self.portfolio_data.get('timeline_minutes', [5, 15, 30, 60])
        
        # Sentiment consistency: Do individual stocks show consistent directional effects?
        for sentiment in sentiments:
            consistent_stocks = 0
            total_stocks = 0
            
            for ticker, data in self.individual_data.items():
                if sentiment in data.get('aar_caar', {}):
                    aars = []
                    for window in time_windows:
                        aar_value = data['aar_caar'][sentiment].get('AAR', {}).get(str(window))
                        if aar_value is not None and not np.isnan(aar_value):
                            aars.append(aar_value)
                    
                    if len(aars) >= 3:  # Need at least 3 time windows
                        total_stocks += 1
                        # Check if majority of AARs have same sign as expected
                        expected_sign = 1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0
                        
                        if expected_sign != 0:
                            correct_sign_count = sum(1 for aar in aars if np.sign(aar) == expected_sign)
                            if correct_sign_count >= len(aars) * 0.6:  # 60% threshold
                                consistent_stocks += 1
            
            results['sentiment_consistency'][sentiment] = {
                'consistent_stocks': consistent_stocks,
                'total_stocks': total_stocks,
                'consistency_rate': consistent_stocks / total_stocks if total_stocks > 0 else 0
            }
        
        # Time window consistency: Are effects persistent across time windows?
        for window in time_windows:
            consistent_sentiments = 0
            total_sentiments = len(sentiments)
            
            for sentiment in sentiments:
                portfolio_aar = self.portfolio_data['aar_data'].get(sentiment, [])
                window_idx = time_windows.index(window)
                
                if window_idx < len(portfolio_aar):
                    aar_value = portfolio_aar[window_idx]
                    expected_sign = 1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0
                    
                    if expected_sign == 0 or np.sign(aar_value) == expected_sign:
                        consistent_sentiments += 1
            
            results['time_window_consistency'][window] = {
                'consistent_sentiments': consistent_sentiments,
                'total_sentiments': total_sentiments,
                'consistency_rate': consistent_sentiments / total_sentiments
            }
        
        return results
    
    def calculate_effect_sizes(self) -> Dict[str, Any]:
        """
        Calculate effect sizes (Cohen's d) for the sentiment effects.
        
        Returns:
            Dictionary with effect size measures
        """
        results = {}
        sentiments = self.portfolio_data.get('sentiments', ['positive', 'negative', 'neutral'])
        time_windows = self.portfolio_data.get('timeline_minutes', [5, 15, 30, 60])
        
        for sentiment in sentiments:
            results[sentiment] = {}
            
            for window in time_windows:
                window_str = str(window)
                
                # Collect individual stock AARs
                sentiment_aars = []
                neutral_aars = []
                
                for ticker, data in self.individual_data.items():
                    # Get AAR for current sentiment
                    if sentiment in data.get('aar_caar', {}):
                        aar_value = data['aar_caar'][sentiment].get('AAR', {}).get(window_str)
                        if aar_value is not None and not np.isnan(aar_value):
                            sentiment_aars.append(aar_value)
                    
                    # Get AAR for neutral sentiment (as baseline)
                    if 'neutral' in data.get('aar_caar', {}):
                        neutral_aar = data['aar_caar']['neutral'].get('AAR', {}).get(window_str)
                        if neutral_aar is not None and not np.isnan(neutral_aar):
                            neutral_aars.append(neutral_aar)
                
                if len(sentiment_aars) > 1 and len(neutral_aars) > 1:
                    # Calculate Cohen's d
                    mean_diff = np.mean(sentiment_aars) - np.mean(neutral_aars)
                    pooled_std = np.sqrt(((len(sentiment_aars) - 1) * np.var(sentiment_aars, ddof=1) + 
                                         (len(neutral_aars) - 1) * np.var(neutral_aars, ddof=1)) / 
                                        (len(sentiment_aars) + len(neutral_aars) - 2))
                    
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    # Effect size interpretation
                    if abs(cohens_d) < 0.2:
                        effect_interpretation = "negligible"
                    elif abs(cohens_d) < 0.5:
                        effect_interpretation = "small"
                    elif abs(cohens_d) < 0.8:
                        effect_interpretation = "medium"
                    else:
                        effect_interpretation = "large"
                    
                    results[sentiment][window] = {
                        'cohens_d': cohens_d,
                        'effect_interpretation': effect_interpretation,
                        'sentiment_mean': np.mean(sentiment_aars),
                        'neutral_mean': np.mean(neutral_aars),
                        'mean_difference': mean_diff,
                        'pooled_std': pooled_std,
                        'n_sentiment': len(sentiment_aars),
                        'n_neutral': len(neutral_aars)
                    }
        
        return results
    
    def calculate_model_fit_statistics(self) -> Dict[str, Any]:
        """
        Calculate R-squared and RMSE measures for model fit.
        
        Returns:
            Dictionary with model fit statistics
        """
        results = {
            'sentiment_prediction': {},
            'time_series_fit': {}
        }
        
        sentiments = self.portfolio_data.get('sentiments', ['positive', 'negative', 'neutral'])
        time_windows = self.portfolio_data.get('timeline_minutes', [5, 15, 30, 60])
        
        # Model 1: Can sentiment predict the direction of abnormal returns?
        for window in time_windows:
            window_str = str(window)
            
            sentiment_labels = []
            aar_values = []
            
            for ticker, data in self.individual_data.items():
                for sentiment in sentiments:
                    if sentiment in data.get('aar_caar', {}):
                        aar_value = data['aar_caar'][sentiment].get('AAR', {}).get(window_str)
                        if aar_value is not None and not np.isnan(aar_value):
                            # Encode sentiment as numeric: positive=1, neutral=0, negative=-1
                            if sentiment == 'positive':
                                sentiment_numeric = 1
                            elif sentiment == 'negative':
                                sentiment_numeric = -1
                            else:
                                sentiment_numeric = 0
                            
                            sentiment_labels.append(sentiment_numeric)
                            aar_values.append(aar_value)
            
            if len(sentiment_labels) > 5:
                # Calculate correlation and R-squared
                correlation = np.corrcoef(sentiment_labels, aar_values)[0, 1]
                r_squared = correlation ** 2
                
                # Calculate RMSE
                mean_aar = np.mean(aar_values)
                rmse = np.sqrt(np.mean([(aar - mean_aar) ** 2 for aar in aar_values]))
                
                # Linear regression for more detailed analysis
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(sentiment_labels, aar_values)
                    
                    results['sentiment_prediction'][window] = {
                        'correlation': correlation,
                        'r_squared': r_squared,
                        'rmse': rmse,
                        'regression': {
                            'slope': slope,
                            'intercept': intercept,
                            'r_value': r_value,
                            'p_value': p_value,
                            'std_err': std_err
                        },
                        'n_observations': len(sentiment_labels)
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate regression for window {window}: {e}")
        
        # Model 2: Time series fit for CAAR evolution
        for sentiment in sentiments:
            portfolio_caar = self.portfolio_data['caar_data'].get(sentiment, [])
            
            if len(portfolio_caar) >= 3:
                # Fit polynomial trends
                x = np.array(time_windows[:len(portfolio_caar)])
                y = np.array(portfolio_caar)
                
                # Linear fit
                try:
                    linear_coef = np.polyfit(x, y, 1)
                    linear_pred = np.polyval(linear_coef, x)
                    linear_r2 = 1 - np.sum((y - linear_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
                    linear_rmse = np.sqrt(np.mean((y - linear_pred) ** 2))
                    
                    results['time_series_fit'][sentiment] = {
                        'linear_r2': linear_r2,
                        'linear_rmse': linear_rmse,
                        'linear_slope': linear_coef[0],
                        'linear_intercept': linear_coef[1]
                    }
                    
                    # Quadratic fit if enough data points
                    if len(portfolio_caar) >= 4:
                        quad_coef = np.polyfit(x, y, 2)
                        quad_pred = np.polyval(quad_coef, x)
                        quad_r2 = 1 - np.sum((y - quad_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
                        quad_rmse = np.sqrt(np.mean((y - quad_pred) ** 2))
                        
                        results['time_series_fit'][sentiment].update({
                            'quadratic_r2': quad_r2,
                            'quadratic_rmse': quad_rmse,
                            'quadratic_coef': quad_coef.tolist()
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not calculate time series fit for {sentiment}: {e}")
        
        return results
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run all statistical analyses and return comprehensive results.
        
        Returns:
            Dictionary with all statistical analysis results
        """
        logger.info("Loading event study data...")
        self.load_data()
        
        logger.info("Calculating portfolio t-tests...")
        t_test_results = self.calculate_portfolio_t_tests()
        
        logger.info("Calculating cross-sectional tests...")
        cross_sectional_results = self.calculate_cross_sectional_tests()
        
        logger.info("Calculating consistency measures...")
        consistency_results = self.calculate_consistency_measures()
        
        logger.info("Calculating effect sizes...")
        effect_size_results = self.calculate_effect_sizes()
        
        logger.info("Calculating model fit statistics...")
        model_fit_results = self.calculate_model_fit_statistics()
        
        # Compile comprehensive results
        self.statistics = {
            'portfolio_t_tests': t_test_results,
            'cross_sectional_analysis': cross_sectional_results,
            'consistency_measures': consistency_results,
            'effect_sizes': effect_size_results,
            'model_fit': model_fit_results,
            'meta_statistics': {
                'total_stocks_analyzed': len(self.individual_data),
                'total_events': self.portfolio_data.get('unique_news_events', 0),
                'event_distribution': self.portfolio_data.get('event_counts', {}),
                'time_windows': self.portfolio_data.get('timeline_minutes', []),
                'sentiments': self.portfolio_data.get('sentiments', [])
            }
        }
        
        return self.statistics
    
    def save_results(self, output_path: str):
        """
        Save statistical analysis results to JSON file.
        
        Args:
            output_path: Path to save the results JSON file
        """
        if self.statistics:
            with open(output_path, 'w') as f:
                json.dump(self.statistics, f, indent=2, default=str)
            logger.info(f"Statistical analysis results saved to {output_path}")
        else:
            logger.warning("No statistical results to save. Run comprehensive analysis first.")
    
    def generate_summary_report(self) -> str:
        """
        Generate a human-readable summary report of the statistical analysis.
        
        Returns:
            String containing formatted summary report
        """
        if not self.statistics:
            return "No statistical analysis results available. Run comprehensive analysis first."
        
        report = []
        report.append("=" * 80)
        report.append("PORTFOLIO EVENT STUDY STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Meta statistics
        meta = self.statistics['meta_statistics']
        report.append(f"Total stocks analyzed: {meta['total_stocks_analyzed']}")
        report.append(f"Total news events: {meta['total_events']}")
        report.append(f"Event distribution: {meta['event_distribution']}")
        report.append(f"Time windows: {meta['time_windows']} minutes")
        report.append("")
        
        # Portfolio t-test results
        report.append("PORTFOLIO T-TEST RESULTS (AAR)")
        report.append("-" * 40)
        t_tests = self.statistics['portfolio_t_tests']['aar_tests']
        
        for sentiment in meta['sentiments']:
            report.append(f"\n{sentiment.upper()} Sentiment:")
            if sentiment in t_tests:
                for window, results in t_tests[sentiment].items():
                    sig_5 = "***" if results['significant_5pct'] else ""
                    sig_1 = "**" if results['significant_1pct'] else ""
                    report.append(f"  {window}min: AAR={results['mean']:.6f}, "
                                f"t={results['t_statistic']:.3f}, "
                                f"p={results['p_value']:.4f} {sig_5}{sig_1}")
        
        # Effect sizes
        report.append("\n\nEFFECT SIZES (Cohen's d vs Neutral)")
        report.append("-" * 40)
        effect_sizes = self.statistics['effect_sizes']
        
        for sentiment in ['positive', 'negative']:
            if sentiment in effect_sizes:
                report.append(f"\n{sentiment.upper()} vs Neutral:")
                for window, results in effect_sizes[sentiment].items():
                    report.append(f"  {window}min: d={results['cohens_d']:.3f} "
                                f"({results['effect_interpretation']})")
        
        # Model fit
        report.append("\n\nMODEL FIT STATISTICS")
        report.append("-" * 40)
        model_fit = self.statistics['model_fit']['sentiment_prediction']
        
        if model_fit:
            report.append("Sentiment Prediction Model (R²):")
            for window, results in model_fit.items():
                report.append(f"  {window}min: R²={results['r_squared']:.4f}, "
                            f"p={results['regression']['p_value']:.4f}")
        
        # Consistency measures
        report.append("\n\nCONSISTENCY MEASURES")
        report.append("-" * 40)
        consistency = self.statistics['consistency_measures']['sentiment_consistency']
        
        for sentiment, results in consistency.items():
            rate = results['consistency_rate'] * 100
            report.append(f"{sentiment}: {rate:.1f}% of stocks show consistent directional effects")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def run_portfolio_statistics_analysis(run_dir: str, index_name: str = "LOG", output_filename: str = "portfolio_statistics.json"):
    """
    Run comprehensive portfolio-level statistical analysis for an entire run.
    
    Args:
        run_dir: Directory containing the run results
        index_name: Name of the index (e.g., "LOG", "TOP5", "DOW")
        output_filename: Name of output file for results
    """
    # Paths - use the provided index name
    index_analysis_dir = os.path.join(run_dir, f"{index_name}_INDEX_ANALYSIS")
    
    # Try different possible locations for event study file
    possible_event_study_paths = [
        os.path.join(index_analysis_dir, "event_study", f"{index_name}_event_study.json"),
        os.path.join(index_analysis_dir, "visualizations", "event_study", f"{index_name}_event_study.json"),
        os.path.join(index_analysis_dir, f"{index_name}_event_study.json")
    ]
    
    event_study_file = None
    for path in possible_event_study_paths:
        if os.path.exists(path):
            event_study_file = path
            break
    
    # Individual results directory
    individual_results_dir = run_dir
    
    # Check if event study file exists
    if not event_study_file:
        logger.error(f"Portfolio event study file not found in any of these locations:")
        for path in possible_event_study_paths:
            logger.error(f"  - {path}")
        return None
    
    logger.info(f"Using event study file: {event_study_file}")
    
    # Initialize analyzer
    analyzer = PortfolioEventStudyStatistics(event_study_file, individual_results_dir)
    
    # Run analysis
    try:
        results = analyzer.run_comprehensive_analysis()
        
        # Save results
        output_path = os.path.join(index_analysis_dir, output_filename)
        analyzer.save_results(output_path)
        
        # Generate and save summary report
        summary_report = analyzer.generate_summary_report()
        summary_path = os.path.join(index_analysis_dir, "portfolio_statistics_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"Portfolio statistical analysis completed. Results saved to {output_path}")
        logger.info(f"Summary report saved to {summary_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in portfolio statistical analysis: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
        run_portfolio_statistics_analysis(run_dir)
    else:
        # Default to latest run
        output_base = "output"
        if os.path.exists(output_base):
            runs = [d for d in os.listdir(output_base) if d.startswith("run_LOG_")]
            if runs:
                latest_run = sorted(runs)[-1]
                run_dir = os.path.join(output_base, latest_run)
                print(f"Running analysis on latest run: {run_dir}")
                run_portfolio_statistics_analysis(run_dir)
            else:
                print("No LOG runs found in output directory")
        else:
            print("Output directory not found")