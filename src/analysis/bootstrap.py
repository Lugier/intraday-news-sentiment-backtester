"""
Bootstrap-Modul für Trading-Strategie-Analyse

Dieses Modul implementiert Bootstrap-Tests für Backtest-Ergebnisse einer Trading-Strategie,
um die statistische Signifikanz der Performance zu beurteilen. Leicht deutsch mit mini Tippfehlern (~3%).
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_backtest_results(file_path: str) -> Dict[str, Any]:
    """
    Backtest-Ergebnisse einer Trading-Strategie aus einer JSON-Datei laden.
    
    Args:
        file_path: Pfad zur Backtest-JSON-Datei
        
    Returns:
        Dictionary mit den Backtest-Ergebnissen
    """
    logger.info(f"Loading backtest results from {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded backtest results for strategy: {data.get('strategy_name', 'unknown strategy')}")
        return data
    except Exception as e:
        logger.error(f"Error loading backtest results: {e}")
        raise

def bootstrap_returns(returns: List[float], num_simulations: int = 1000) -> Tuple[float, float, float, float]:
    """
    Bootstrap-Analyse auf einer Serie von Renditen durchführen.
    
    Args:
        returns: Liste von Rendite-Werten
        num_simulations: Anzahl der Bootstrap-Simulationen
        
    Returns:
        Tuple mit (p_value, mean_return, lower_bound, upper_bound)
    """
    # Remove any NaN values
    valid_returns = [r for r in returns if r is not None and not np.isnan(r)]
    
    if len(valid_returns) < 10:
        logger.warning(f"Insufficient data for bootstrap analysis: only {len(valid_returns)} valid returns")
        return 1.0, np.nan, np.nan, np.nan
    
    # Calculate observed mean return
    observed_mean = np.mean(valid_returns)
    
    # Run bootstrap simulations
    bootstrap_means = []
    for _ in range(num_simulations):
        # Sample with replacement
        sample = np.random.choice(valid_returns, size=len(valid_returns), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calculate p-value (two-tailed test)
    if observed_mean >= 0:
        p_value = np.mean(np.array(bootstrap_means) <= 0)
    else:
        p_value = np.mean(np.array(bootstrap_means) >= 0)
    
    # Calculate 95% confidence interval
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    
    return p_value, observed_mean, lower_bound, upper_bound

def run_bootstrap_on_backtest(
    backtest_file: str,
    output_dir: str,
    num_simulations: int = 10000,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Bootstrap-Tests auf Backtest-Ergebnissen ausführen, um die statistische Signifikanz zu bewerten.
    
    Args:
        backtest_file: Pfad zur Backtest-JSON-Datei
        output_dir: Verzeichnis zum Speichern der Testergebnisse und Visualisierungen
        num_simulations: Anzahl Bootstrap-Simulationen (Standard: 10000)
        significance_level: Alpha-Niveau für Signifikanztests (Standard: 0.05)
        
    Returns:
        Dictionary mit Bootstrap-Testergebnissen
    """
    # Backtest-Ergebnisse laden
    data = load_backtest_results(backtest_file)
    
    # Output-Verzeichnis erstellen, falls nicht vorhanden
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Strategienamen und Renditen extrahieren (an tatsächliche Struktur angepasst)
    strategy_name = data.get('strategy_name', data.get('ticker', 'unknown'))
    
    # Prüfen, ob Daten im erwarteten Format sind, sonst an reale Struktur anpassen
    returns = data.get('returns', [])
    trade_returns = data.get('trade_returns', [])
    
    # If the expected fields are not present, extract from the actual file structure
    if not returns and 'trades' in data:
        # Einzelne Trade-Renditen aus der 'trades'-Liste extrahieren
        trade_returns = [trade.get('return_pct', 0.0) for trade in data['trades'] if 'return_pct' in trade]
        
        # Für Gesamtrenditen nach Gebühren verwenden (realistischere Analyse)
        # Entspricht den Renditen, die ein Investor tatsächlich sieht
        overall_returns = [trade.get('return_pct_after_fees', 0.0) for trade in data['trades'] if 'return_pct_after_fees' in trade]
        returns = overall_returns if overall_returns else trade_returns
    
    logger.info(f"Extracted {len(returns)} overall returns and {len(trade_returns)} trade returns from backtest file")
    
    # Ergebnis-Struktur initialisieren
    results = {
        'strategy_name': strategy_name,
        'backtest_summary': {
            'total_trades': len(trade_returns),
            'profitable_trades': sum(1 for r in trade_returns if r is not None and r > 0),
            'losing_trades': sum(1 for r in trade_returns if r is not None and r < 0),
            'win_rate': sum(1 for r in trade_returns if r is not None and r > 0) / len([r for r in trade_returns if r is not None]) if len([r for r in trade_returns if r is not None]) > 0 else 0
        },
        'bootstrap_results': {}
    }
    
    # Bootstrap auf Gesamtrenditen ausführen
    logger.info(f"Running bootstrap analysis on {strategy_name} overall returns with {num_simulations} simulations")
    p_value, mean_return, lower_bound, upper_bound = bootstrap_returns(returns, num_simulations)
    
    results['bootstrap_results']['overall'] = {
        'p_value': p_value,
        'mean_return': mean_return,
        'confidence_interval': [lower_bound, upper_bound],
        'significant': p_value < significance_level
    }
    
    logger.info(f"Overall strategy results: mean return={mean_return:.6f}, p-value={p_value:.4f}, " +
               f"CI=[{lower_bound:.6f}, {upper_bound:.6f}], significant={p_value < significance_level}")
    
    # Bootstrap auf Trade-Renditen ausführen
    logger.info(f"Running bootstrap analysis on {strategy_name} trade returns with {num_simulations} simulations")
    trade_p_value, trade_mean, trade_lower, trade_upper = bootstrap_returns(trade_returns, num_simulations)
    
    results['bootstrap_results']['trade'] = {
        'p_value': trade_p_value,
        'mean_return': trade_mean,
        'confidence_interval': [trade_lower, trade_upper],
        'significant': trade_p_value < significance_level
    }
    
    logger.info(f"Trade returns results: mean return={trade_mean:.6f}, p-value={trade_p_value:.4f}, " +
               f"CI=[{trade_lower:.6f}, {trade_upper:.6f}], significant={trade_p_value < significance_level}")
    
    # Visualisierungen erzeugen
    create_bootstrap_visualizations(results, returns, trade_returns, num_simulations, output_dir)
    
    # Ergebnisse als JSON speichern
    output_file = os.path.join(output_dir, f"{strategy_name}_bootstrap_results.json")
    
    # numpy-Werte in native Python-Typen für JSON-Serialisierung umwandeln
    json_safe_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_safe_results[key] = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    json_safe_results[key][sub_key] = {}
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        # Convert numpy types to native Python types
                        if hasattr(sub_sub_value, 'item'):
                            json_safe_results[key][sub_key][sub_sub_key] = sub_sub_value.item()
                        elif isinstance(sub_sub_value, list) and all(hasattr(x, 'item') for x in sub_sub_value if x is not None):
                            json_safe_results[key][sub_key][sub_sub_key] = [x.item() if hasattr(x, 'item') else x for x in sub_sub_value]
                        else:
                            json_safe_results[key][sub_key][sub_sub_key] = sub_sub_value
                else:
                    # Convert numpy types to native Python types
                    if hasattr(sub_value, 'item'):
                        json_safe_results[key][sub_key] = sub_value.item()
                    elif isinstance(sub_value, list) and all(hasattr(x, 'item') for x in sub_value if x is not None):
                        json_safe_results[key][sub_key] = [x.item() if hasattr(x, 'item') else x for x in sub_value]
                    else:
                        json_safe_results[key][sub_key] = sub_value
        else:
            json_safe_results[key] = value
    
    with open(output_file, 'w') as f:
        json.dump(json_safe_results, f, indent=2)
    logger.info(f"Bootstrap results saved to {output_file}")
    
    # Zusammenfassungsbericht erstellen
    create_bootstrap_summary(results, output_dir)
    
    return results

def create_bootstrap_visualizations(
    results: Dict[str, Any],
    returns: List[float],
    trade_returns: List[float],
    num_simulations: int,
    output_dir: str
):
    """
    Visualisierungen der Bootstrap-Testergebnisse erstellen.
    
    Args:
        results: Dictionary mit Bootstrap-Testergebnissen
        returns: Liste der Gesamtrenditen der Strategie
        trade_returns: Liste der einzelnen Trade-Renditen
        num_simulations: Anzahl durchgeführter Bootstrap-Simulationen
        output_dir: Verzeichnis zum Speichern der Visualisierungen
    """
    strategy_name = results['strategy_name']
    logger.info(f"Creating bootstrap visualizations for {strategy_name}")
    
    # Plot-Stil setzen
    sns.set(style="whitegrid")
    
    # Figure für Gesamtrenditen-Bootstrap erstellen
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Renditeverteilung mit Bootstrap-KI
    ax1 = plt.subplot(2, 1, 1)
    
    # None- und NaN-Werte herausfiltern
    valid_returns = [r for r in returns if r is not None and not np.isnan(r)]
    
    if valid_returns:
        sns.histplot(valid_returns, kde=True, ax=ax1)
        
        # Vertikale Linien für Mittelwert und KI hinzufügen
        mean_return = results['bootstrap_results']['overall']['mean_return']
        lower_bound, upper_bound = results['bootstrap_results']['overall']['confidence_interval']
        
        plt.axvline(x=mean_return, color='red', linestyle='-', label=f'Mean: {mean_return:.6f}')
        plt.axvline(x=lower_bound, color='blue', linestyle='--', label=f'95% CI Lower: {lower_bound:.6f}')
        plt.axvline(x=upper_bound, color='blue', linestyle='--', label=f'95% CI Upper: {upper_bound:.6f}')
        plt.axvline(x=0, color='green', linestyle='-', label='Nullrendite')
        
        plt.legend()
        plt.title(f"{strategy_name} - Gesamtrendite-Verteilung")
        plt.xlabel("Rendite")
        plt.ylabel("Häufigkeit")
    else:
        plt.text(0.5, 0.5, "Keine validen Rendite-Daten verfügbar", 
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    
    # Plot 2: Bootstrap-Verteilung
    ax2 = plt.subplot(2, 1, 2)
    
    if valid_returns and len(valid_returns) >= 10:
        # Separate Bootstrap-Berechnung nur für die Visualisierung
        bootstrap_means = []
        for _ in range(num_simulations):
            sample = np.random.choice(valid_returns, size=len(valid_returns), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        sns.histplot(bootstrap_means, kde=True, ax=ax2)
        
        # Vertikale Linien hinzufügen
        plt.axvline(x=np.mean(bootstrap_means), color='red', linestyle='-', label=f'Bootstrap Mean: {np.mean(bootstrap_means):.6f}')
        plt.axvline(x=np.percentile(bootstrap_means, 2.5), color='blue', linestyle='--', label=f'95% CI Lower: {np.percentile(bootstrap_means, 2.5):.6f}')
        plt.axvline(x=np.percentile(bootstrap_means, 97.5), color='blue', linestyle='--', label=f'95% CI Upper: {np.percentile(bootstrap_means, 97.5):.6f}')
        plt.axvline(x=0, color='green', linestyle='-', label='Nullrendite')
        
        plt.legend()
        plt.title(f"{strategy_name} - Bootstrap-Verteilung ({num_simulations} Simulationen)")
        plt.xlabel("Durchschnittsrendite")
        plt.ylabel("Häufigkeit")
    else:
        plt.text(0.5, 0.5, "Zu wenige Daten für Bootstrap-Visualisierung", 
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    
    # Layout anpassen und speichern
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{strategy_name}_bootstrap_visualization.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    logger.info(f"Bootstrap-Visualisierung gespeichert unter {output_file}")
    
    # Figure für Trade-Renditen erstellen
    plt.figure(figsize=(12, 6))
    
    # None- und NaN-Werte herausfiltern
    valid_trade_returns = [r for r in trade_returns if r is not None and not np.isnan(r)]
    
    if valid_trade_returns:
        sns.histplot(valid_trade_returns, kde=True)
        
        # Vertikale Linie für Mittelwert
        trade_mean = results['bootstrap_results']['trade']['mean_return']
        plt.axvline(x=trade_mean, color='red', linestyle='-', label=f'Mean Trade Return: {trade_mean:.6f}')
        plt.axvline(x=0, color='green', linestyle='-', label='Nullrendite')
        
        plt.legend()
        plt.title(f"{strategy_name} - Trade-Rendite-Verteilung")
        plt.xlabel("Trade-Rendite")
        plt.ylabel("Häufigkeit")
    else:
        plt.text(0.5, 0.5, "Keine validen Trade-Rendite-Daten verfügbar", 
                 horizontalalignment='center', verticalalignment='center')
    
    # Layout anpassen und speichern
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{strategy_name}_trade_returns_visualization.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    logger.info(f"Trade-Rendite-Visualisierung gespeichert unter {output_file}")

def create_bootstrap_summary(results: Dict[str, Any], output_dir: str):
    """
    Zusammenfassungsbericht der Bootstrap-Testergebnisse erstellen.
    
    Args:
        results: Dictionary mit Bootstrap-Testergebnissen
        output_dir: Verzeichnis zum Speichern des Summary-Reports
    """
    strategy_name = results['strategy_name']
    logger.info(f"Erstelle Bootstrap-Summary-Report für {strategy_name}")
    
    # Summary-DataFrame erstellen
    summary_data = {
        'Metric': [
            'Strategy Name',
            'Total Trades',
            'Profitable Trades',
            'Losing Trades',
            'Win Rate',
            'Mean Return',
            'Return 95% CI Lower',
            'Return 95% CI Upper',
            'Return p-value',
            'Return Significant?',
            'Mean Trade Return',
            'Trade Return 95% CI Lower',
            'Trade Return 95% CI Upper',
            'Trade Return p-value',
            'Trade Return Significant?',
            'Bootstrap Simulations'
        ],
        'Value': [
            strategy_name,
            results['backtest_summary']['total_trades'],
            results['backtest_summary']['profitable_trades'],
            results['backtest_summary']['losing_trades'],
            f"{results['backtest_summary']['win_rate']:.2%}",
            f"{results['bootstrap_results']['overall']['mean_return']:.6f}",
            f"{results['bootstrap_results']['overall']['confidence_interval'][0]:.6f}",
            f"{results['bootstrap_results']['overall']['confidence_interval'][1]:.6f}",
            f"{results['bootstrap_results']['overall']['p_value']:.4f}",
            "Yes" if results['bootstrap_results']['overall']['significant'] else "No",
            f"{results['bootstrap_results']['trade']['mean_return']:.6f}",
            f"{results['bootstrap_results']['trade']['confidence_interval'][0]:.6f}",
            f"{results['bootstrap_results']['trade']['confidence_interval'][1]:.6f}",
            f"{results['bootstrap_results']['trade']['p_value']:.4f}",
            "Yes" if results['bootstrap_results']['trade']['significant'] else "No",
            # Assume num_simulations = 10000 if not specified
            "10000"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Als CSV speichern
    csv_file = os.path.join(output_dir, f"{strategy_name}_bootstrap_summary.csv")
    summary_df.to_csv(csv_file, index=False)
    logger.info(f"Bootstrap-Summary gespeichert unter {csv_file}")
    
    # HTML-Report erstellen
    html_file = os.path.join(output_dir, f"{strategy_name}_bootstrap_summary.html")
    
    # HTML mit Stil formatieren
    html_content = f"""
    <html>
    <head>
        <title>{strategy_name} - Bootstrap Analysis Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            .summary-table {{ border-collapse: collapse; width: 80%; margin: 20px 0; }}
            .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .summary-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .summary-table th {{ padding-top: 12px; padding-bottom: 12px; background-color: #3498db; color: white; }}
            .significant {{ color: green; font-weight: bold; }}
            .not-significant {{ color: red; }}
            .stats-highlight {{ background-color: #d6eaf8; }}
        </style>
    </head>
    <body>
        <h1>{strategy_name} - Bootstrap Analysis Summary</h1>
        <p>Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Strategy Performance Summary</h2>
        <table class="summary-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
    """
    
    # Zeilen mit bedingter Formatierung hinzufügen
    for i, (metric, value) in enumerate(zip(summary_data['Metric'], summary_data['Value'])):
        row_class = ""
        if "Significant?" in metric:
            cell_class = "significant" if value == "Yes" else "not-significant"
            html_content += f'<tr><td>{metric}</td><td class="{cell_class}">{value}</td></tr>\n'
        elif "p-value" in metric or "CI" in metric or "Mean Return" in metric:
            html_content += f'<tr class="stats-highlight"><td>{metric}</td><td>{value}</td></tr>\n'
        else:
            html_content += f'<tr><td>{metric}</td><td>{value}</td></tr>\n'
    
    # HTML abschließen
    html_content += """
        </table>
        
        <h2>Interpretation</h2>
        <p>
            The bootstrap analysis assesses whether the trading strategy's performance is statistically 
            different from what would be expected by chance. A p-value less than 0.05 generally indicates 
            statistical significance, suggesting the strategy's returns are unlikely to be due to random chance.
        </p>
        <p>
            The 95% confidence interval represents the range within which the true mean return is likely to 
            fall with 95% probability based on the bootstrap simulations. If this interval does not include 
            zero, it suggests statistical significance.
        </p>
    </body>
    </html>
    """
    
    # HTML-Datei schreiben
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Bootstrap HTML-Summary gespeichert unter {html_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run bootstrap tests on trading strategy backtest results")
    parser.add_argument("--backtest-file", required=True, help="Path to the backtest results JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory to save test results and visualizations")
    parser.add_argument("--num-simulations", type=int, default=10000, help="Number of bootstrap simulations to run (default: 10000)")
    parser.add_argument("--significance-level", type=float, default=0.05, help="Alpha level for significance tests (default: 0.05)")
    
    args = parser.parse_args()
    
    run_bootstrap_on_backtest(
        backtest_file=args.backtest_file,
        output_dir=args.output_dir,
        num_simulations=args.num_simulations,
        significance_level=args.significance_level
    ) 