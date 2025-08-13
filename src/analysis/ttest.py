"""
T-Test Modul für Event-Study Analyse

Dieses Modul implementiert t-Tests für Event-Study-Ergebnisse, um die statistische Signifikanz
abnormaler Renditen nach News-Ereignissen zu prüfen. Leicht deutsch mit mini Tippfehlern (~3%).
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_event_study_results(file_path: str) -> Dict[str, Any]:
    """
    Event-Study-Ergebnisse aus einer JSON-Datei laden.
    
    Args:
        file_path: Pfad zur Event-Study-JSON-Datei
        
    Returns:
        Dictionary mit den Event-Study-Ergebnissen
    """
    logger.info(f"Loading event study results from {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded event study results for {data.get('ticker', 'unknown ticker')}")
        return data
    except Exception as e:
        logger.error(f"Error loading event study results: {e}")
        raise

def run_ttest_on_event_study(
    event_study_file: str,
    output_dir: str,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    t-Tests auf Event-Study-Ergebnissen ausführen, um die statistische Signifikanz
    abnormaler Renditen nach News-Ereignissen zu prüfen.
    
    Args:
        event_study_file: Pfad zur Event-Study-JSON-Datei
        output_dir: Verzeichnis zum Speichern der Testergebnisse und Visualisierungen
        significance_level: Alpha-Niveau für Signifikanztests (Standard: 0.05)
        
    Returns:
        Dictionary mit t-Test-Ergebnissen
    """
    # Event-Study-Ergebnisse laden
    data = load_event_study_results(event_study_file)
    
    # Output-Verzeichnis erstellen, falls nicht vorhanden
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Ticker aus den Daten extrahieren
    ticker = data.get('ticker', 'unknown')
    
    # Ergebnis-Struktur initialisieren
    results = {
        'ticker': ticker,
        'unique_news_events': data.get('unique_news_events', data.get('total_events', 0)),  # Updated field name with fallback
        'event_counts': {}
    }
    
    # Event-Anzahlen nach Sentiment extrahieren
    if 'events_by_sentiment' in data:
        results['event_counts'] = data['events_by_sentiment']
    
    # Tests-Dictionary initialisieren
    results['tests'] = {}
    
    # Die tatsächlichen Event-Study-Ergebnisse haben eine andere Struktur
    # Prüfen, ob der Schlüssel 'aar_caar' in den Daten existiert
    if 'aar_caar' in data:
        # Process each sentiment category
        for sentiment in data['aar_caar']:
            logger.info(f"Processing t-test results for {sentiment} sentiment events")
            
            # Ergebnisse für dieses Sentiment initialisieren
            results['tests'][sentiment] = {
                'aar_tests': {},
                'caar_tests': {}
            }
            
            # AAR-Ergebnisse verarbeiten
            if 'AAR' in data['aar_caar'][sentiment]:
                aar_data = data['aar_caar'][sentiment]['AAR']
                aar_std = data['aar_caar'][sentiment].get('AAR_std', {})
                counts = data['aar_caar'][sentiment].get('count', {})
                t_stats = data['aar_caar'][sentiment].get('t_stat', {})
                p_values = data['aar_caar'][sentiment].get('p_value', {})
                
                for window in aar_data:
                    mean_aar = aar_data[window]
                    std_aar = aar_std.get(window, None)
                    count = counts.get(window, 0)
                    t_stat = t_stats.get(window, None)
                    p_value = p_values.get(window, None)
                    
                    # Falls t_stat und p_value fehlen, mit std und count berechnen
                    if (t_stat is None or p_value is None) and std_aar is not None and count > 0:
                        # Calculate standard error
                        se = std_aar / np.sqrt(count)
                        # Calculate t-statistic
                        t_stat = mean_aar / se if se > 0 else 0
                        # Calculate p-value (two-tailed)
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), count - 1)) if count > 1 else 1
                    
                    # Ergebnisse speichern
                    results['tests'][sentiment]['aar_tests'][window] = {
                        'mean': mean_aar,
                        'std': std_aar,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < significance_level if p_value is not None else False,
                        'n': count
                    }
                    
                    # t_stat und p_value fürs Logging formatieren
                    t_stat_str = f"{t_stat:.4f}" if t_stat is not None else "N/A"
                    p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
                    significant_str = str(p_value < significance_level if p_value is not None else False)
                    logger.info(f"{sentiment} AAR {window} min: t={t_stat_str}, p={p_value_str}, sig={significant_str}")
            
            # CAAR-Ergebnisse verarbeiten
            if 'CAAR' in data['aar_caar'][sentiment]:
                caar_data = data['aar_caar'][sentiment]['CAAR']
                counts = data['aar_caar'][sentiment].get('count', {})
                
                for window in caar_data:
                    mean_caar = caar_data[window]
                    count = counts.get(window, 0)
                    
                    # t-Statistiken und p-Werte extrahieren, falls vorhanden
                    t_stat = None
                    p_value = None
                    significant = False
                    
                    if 't_stat' in data['aar_caar'][sentiment] and window in data['aar_caar'][sentiment]['t_stat']:
                        t_stat = data['aar_caar'][sentiment]['t_stat'][window]
                    
                    if 'p_value' in data['aar_caar'][sentiment] and window in data['aar_caar'][sentiment]['p_value']:
                        p_value = data['aar_caar'][sentiment]['p_value'][window]
                        significant = p_value < significance_level if p_value is not None else False
                    
                    # Ergebnisse speichern
                    results['tests'][sentiment]['caar_tests'][window] = {
                        'mean': mean_caar,
                        'std': None,  # CAAR std not provided in the data
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': significant,
                        'n': count
                    }
                    
                    # t_stat und p_value fürs Logging formatieren
                    t_stat_str = f"{t_stat:.4f}" if t_stat is not None else "N/A"
                    p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
                    significant_str = str(significant)
                    
                    logger.info(f"{sentiment} CAAR {window} min: t={t_stat_str}, p={p_value_str}, sig={significant_str}, mean={mean_caar:.8f}")
    else:
        logger.warning("Event-Study-Ergebnisse enthalten keine 'aar_caar' Daten")
    
    # Zusammenfassende Visualisierungen erstellen
    create_ttest_visualizations(results, output_dir)
    
    # Ergebnisse als JSON speichern
    output_file = os.path.join(output_dir, f"{ticker}_ttest_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"T-test results saved to {output_file}")
    
    # Zusammenfassungsbericht erstellen
    create_ttest_summary(results, output_dir)
    
    return results

def create_ttest_visualizations(results: Dict[str, Any], output_dir: str):
    """
    Create visualizations of t-test results.
    
    Args:
        results: Dictionary containing t-test results
        output_dir: Directory to save visualizations
    """
    ticker = results['ticker']
    logger.info(f"Creating t-test visualizations for {ticker}")
    
    # Set plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 16))  # Increased figure height for 4 panels
    
    sentiments = ['positive', 'negative', 'neutral']
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
    
    # Plot AAR t-statistics by sentiment and event window
    ax1 = plt.subplot(4, 1, 1)  # Changed to 4 rows instead of 2
    
    for sentiment in sentiments:
        if sentiment not in results['tests']:
            continue
            
        windows = []
        t_stats = []
        
        for window in sorted(results['tests'][sentiment]['aar_tests'].keys(), key=lambda x: int(x)):
            t_stat = results['tests'][sentiment]['aar_tests'][window]['t_statistic']
            if t_stat is not None:
                windows.append(int(window))
                t_stats.append(t_stat)
        
        if windows:
            plt.plot(windows, t_stats, marker='o', label=f"{sentiment.capitalize()}", color=colors[sentiment])
    
    # Add horizontal lines at t-critical values
    df = 0
    for sentiment in sentiments:
        if sentiment in results['tests'] and results['tests'][sentiment]['aar_tests']:
            first_window = next(iter(results['tests'][sentiment]['aar_tests']))
            df = results['tests'][sentiment]['aar_tests'][first_window].get('n', 0)
            if df > 0:
                break
    
    if df > 0:
        t_crit_95 = stats.t.ppf(0.975, df - 1)  # Two-tailed, 95% confidence
        plt.axhline(y=t_crit_95, linestyle='--', color='gray', alpha=0.7, label=f"t-critical (95%)")
        plt.axhline(y=-t_crit_95, linestyle='--', color='gray', alpha=0.7)
    
    plt.title(f"{ticker} - AAR T-Statistics by Event Window")
    plt.xlabel("Minutes from News")
    plt.ylabel("T-Statistic")
    plt.legend()
    plt.grid(True)
    
    # Plot AAR means by sentiment and event window
    ax2 = plt.subplot(4, 1, 2)  # Changed to 4 rows instead of 2
    
    for sentiment in sentiments:
        if sentiment not in results['tests']:
            continue
            
        windows = []
        means = []
        
        for window in sorted(results['tests'][sentiment]['aar_tests'].keys(), key=lambda x: int(x)):
            windows.append(int(window))
            means.append(results['tests'][sentiment]['aar_tests'][window]['mean'])
        
        if windows:
            plt.plot(windows, means, marker='o', label=f"{sentiment.capitalize()}", color=colors[sentiment])
    
    plt.title(f"{ticker} - AAR Means by Event Window")
    plt.xlabel("Minutes from News")
    plt.ylabel("Average Abnormal Return")
    plt.legend()
    plt.grid(True)
    
    # Plot CAAR t-statistics by sentiment and event window
    ax3 = plt.subplot(4, 1, 3)
    
    for sentiment in sentiments:
        if sentiment not in results['tests']:
            continue
            
        windows = []
        t_stats = []
        
        for window in sorted(results['tests'][sentiment]['caar_tests'].keys(), key=lambda x: int(x)):
            t_stat = results['tests'][sentiment]['caar_tests'][window]['t_statistic']
            if t_stat is not None:
                windows.append(int(window))
                t_stats.append(t_stat)
        
        if windows:
            plt.plot(windows, t_stats, marker='o', label=f"{sentiment.capitalize()}", color=colors[sentiment])
    
    # Add horizontal lines at t-critical values (reusing the same df from earlier)
    if df > 0:
        plt.axhline(y=t_crit_95, linestyle='--', color='gray', alpha=0.7, label=f"t-critical (95%)")
        plt.axhline(y=-t_crit_95, linestyle='--', color='gray', alpha=0.7)
    
    plt.title(f"{ticker} - CAAR T-Statistics by Event Window")
    plt.xlabel("Minutes from News")
    plt.ylabel("T-Statistic")
    plt.legend()
    plt.grid(True)
    
    # Plot CAAR means by sentiment and event window
    ax4 = plt.subplot(4, 1, 4)
    
    for sentiment in sentiments:
        if sentiment not in results['tests']:
            continue
            
        windows = []
        means = []
        
        for window in sorted(results['tests'][sentiment]['caar_tests'].keys(), key=lambda x: int(x)):
            windows.append(int(window))
            means.append(results['tests'][sentiment]['caar_tests'][window]['mean'])
        
        if windows:
            plt.plot(windows, means, marker='o', label=f"{sentiment.capitalize()}", color=colors[sentiment])
    
    plt.title(f"{ticker} - CAAR Means by Event Window")
    plt.xlabel("Minutes from News")
    plt.ylabel("Cumulative Average Abnormal Return")
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{ticker}_ttest_visualization.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    logger.info(f"T-test visualization saved to {output_file}")

def create_ttest_summary(results: Dict[str, Any], output_dir: str):
    """
    Create a summary report of t-test results.
    
    Args:
        results: Dictionary containing t-test results
        output_dir: Directory to save the summary report
    """
    ticker = results['ticker']
    logger.info(f"Creating t-test summary report for {ticker}")
    
    # Create a DataFrame of AAR test results
    aar_rows = []
    caar_rows = []
    
    for sentiment in results['tests']:
        for window in results['tests'][sentiment]['aar_tests']:
            test = results['tests'][sentiment]['aar_tests'][window]
            aar_rows.append({
                'Sentiment': sentiment.capitalize(),
                'Window (min)': window,
                'Mean AAR': test['mean'],
                'Std Dev': test['std'],
                'T-Statistic': test['t_statistic'],
                'P-Value': test['p_value'],
                'Significant': test['significant'],
                'Sample Size': test['n']
            })
        
        for window in results['tests'][sentiment]['caar_tests']:
            test = results['tests'][sentiment]['caar_tests'][window]
            caar_rows.append({
                'Sentiment': sentiment.capitalize(),
                'Window (min)': window,
                'Mean CAAR': test['mean'],
                'Std Dev': test['std'],
                'T-Statistic': test['t_statistic'],
                'P-Value': test['p_value'],
                'Significant': test['significant'],
                'Sample Size': test['n']
            })
    
    # Create DataFrames
    aar_df = pd.DataFrame(aar_rows)
    caar_df = pd.DataFrame(caar_rows)
    
    # Sort by sentiment and window
    if not aar_df.empty:
        aar_df['Window (min)'] = aar_df['Window (min)'].astype(int)
        aar_df = aar_df.sort_values(['Sentiment', 'Window (min)'])
    
    if not caar_df.empty:
        caar_df['Window (min)'] = caar_df['Window (min)'].astype(int)
        caar_df = caar_df.sort_values(['Sentiment', 'Window (min)'])
    
    # Save as CSV
    aar_csv = os.path.join(output_dir, f"{ticker}_aar_ttest_summary.csv")
    caar_csv = os.path.join(output_dir, f"{ticker}_caar_ttest_summary.csv")
    
    if not aar_df.empty:
        aar_df.to_csv(aar_csv, index=False)
        logger.info(f"AAR t-test summary saved to {aar_csv}")
    
    if not caar_df.empty:
        caar_df.to_csv(caar_csv, index=False)
        logger.info(f"CAAR t-test summary saved to {caar_csv}")
    
    # Create HTML reports with color highlighting for significance
    def style_pvalue(val):
        if val is None:
            return 'color: gray'
        color = 'green' if val < 0.01 else 'orange' if val < 0.05 else 'black'
        return f'color: {color}; font-weight: {"bold" if val < 0.05 else "normal"}'
    
    if not aar_df.empty:
        # Handle None values for formatting
        format_dict = {
            'Mean AAR': lambda x: '{:.8f}'.format(x) if x is not None else 'N/A',
            'Std Dev': lambda x: '{:.8f}'.format(x) if x is not None else 'N/A',
            'T-Statistic': lambda x: '{:.4f}'.format(x) if x is not None else 'N/A',
            'P-Value': lambda x: '{:.4f}'.format(x) if x is not None else 'N/A'
        }
        
        styled_aar = aar_df.style.format(format_dict).map(style_pvalue, subset=['P-Value'])
        
        aar_html = os.path.join(output_dir, f"{ticker}_aar_ttest_summary.html")
        with open(aar_html, 'w') as f:
            f.write(f"<h1>{ticker} - AAR T-Test Results</h1>\n")
            f.write(styled_aar.to_html())
        logger.info(f"AAR t-test HTML summary saved to {aar_html}")
    
    if not caar_df.empty:
        # Handle None values for formatting
        format_dict = {
            'Mean CAAR': lambda x: '{:.8f}'.format(x) if x is not None else 'N/A',
            'Std Dev': lambda x: '{:.8f}'.format(x) if x is not None else 'N/A',
            'T-Statistic': lambda x: '{:.4f}'.format(x) if x is not None else 'N/A',
            'P-Value': lambda x: '{:.4f}'.format(x) if x is not None else 'N/A'
        }
        
        styled_caar = caar_df.style.format(format_dict).map(style_pvalue, subset=['P-Value'])
        
        caar_html = os.path.join(output_dir, f"{ticker}_caar_ttest_summary.html")
        with open(caar_html, 'w') as f:
            f.write(f"<h1>{ticker} - CAAR T-Test Results</h1>\n")
            f.write(styled_caar.to_html())
        logger.info(f"CAAR t-test HTML summary saved to {caar_html}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run t-tests on event study results")
    parser.add_argument("--event-study-file", required=True, help="Path to the event study results JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory to save test results and visualizations")
    parser.add_argument("--significance-level", type=float, default=0.05, help="Alpha level for significance tests (default: 0.05)")
    
    args = parser.parse_args()
    
    run_ttest_on_event_study(
        event_study_file=args.event_study_file,
        output_dir=args.output_dir,
        significance_level=args.significance_level
    ) 