import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# --- Konfiguration ---
RUN_LOG_DIR = 'output/run_LOG_20250617_200833'
INITIAL_CAPITAL = 100.0
OUTPUT_FILE = 'sequential_portfolio_performance.png'

# --- Datensammlung ---
print("Sammle Trades aus allen Backtest-Dateien...")
all_trades = []
tickers = [d for d in os.listdir(RUN_LOG_DIR) if os.path.isdir(os.path.join(RUN_LOG_DIR, d)) and d.isupper() and d != 'LOG_INDEX_ANALYSIS']

for ticker in tickers:
    ticker_dir = os.path.join(RUN_LOG_DIR, ticker)
    backtest_file = None
    
    # Finde die richtige Backtest-Datei (könnte Zeitstempel enthalten)
    if os.path.isdir(ticker_dir):
        for f in os.listdir(ticker_dir):
            if f.startswith('backtest_results_') and f.endswith('.json'):
                backtest_file = os.path.join(ticker_dir, f)
                break
    
    if backtest_file and os.path.exists(backtest_file):
        try:
            with open(backtest_file, 'r') as f:
                data = json.load(f)
                if 'trades' in data and data['trades']:
                    all_trades.extend(data['trades'])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Fehler beim Verarbeiten von {backtest_file}: {e}")

if not all_trades:
    print("Keine Trades gefunden. Skript wird beendet.")
    exit()

print(f"Gesamtanzahl der gefundenen Trades: {len(all_trades)}")

# --- Datenverarbeitung ---
print("Sortiere Trades und simuliere Portfolio-Performance...")
df_trades = pd.DataFrame(all_trades)
df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
df_trades.sort_values('entry_time', inplace=True)
df_trades.reset_index(drop=True, inplace=True)

# --- Simulation der Portfolio-Performance ---
gross_portfolio_values = []
net_portfolio_values = []
dates = []

current_gross_capital = INITIAL_CAPITAL
current_net_capital = INITIAL_CAPITAL

# Startwerte hinzufügen
start_date = df_trades['entry_time'].iloc[0] - pd.Timedelta(minutes=1)
dates.append(start_date)
gross_portfolio_values.append(INITIAL_CAPITAL)
net_portfolio_values.append(INITIAL_CAPITAL)


for _, trade in df_trades.iterrows():
    # Bruttorendite
    gross_return_pct = trade['return_pct'] / 100.0
    current_gross_capital *= (1 + gross_return_pct)
    
    # Nettorendite
    net_return_pct = trade['return_pct_after_fees'] / 100.0
    current_net_capital *= (1 + net_return_pct)
    
    dates.append(trade['entry_time'])
    gross_portfolio_values.append(current_gross_capital)
    net_portfolio_values.append(current_net_capital)

df_portfolio = pd.DataFrame({
    'date': dates,
    'gross_value': gross_portfolio_values,
    'net_value': net_portfolio_values
}).set_index('date')

# --- Drawdown-Berechnung ---
net_high_water_mark = df_portfolio['net_value'].cummax()
df_portfolio['net_drawdown_pct'] = ((df_portfolio['net_value'] - net_high_water_mark) / net_high_water_mark) * 100

# --- Chart-Erstellung ---
print(f"Erstelle Chart und speichere als '{OUTPUT_FILE}'...")
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(16, 10), 
    gridspec_kw={'height_ratios': [3, 1]}, 
    sharex=True
)

# Oberer Plot: Portfolio-Performance
ax1.plot(df_portfolio.index, df_portfolio['gross_value'], label='Bruttorendite (ohne Gebühren)', color='cornflowerblue', linewidth=2)
ax1.plot(df_portfolio.index, df_portfolio['net_value'], label='Nettorendite (mit 0.1% Gebühr pro Trade)', color='mediumblue', linewidth=2)
ax1.set_title('Sequentielle Portfolio-Performance der LOG-Strategie', fontsize=18, pad=20)
ax1.set_ylabel('Portfolio-Wert ($)', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Annotationen für finale Werte
start_val = df_portfolio['gross_value'].iloc[0]
final_gross_val = df_portfolio['gross_value'].iloc[-1]
final_net_val = df_portfolio['net_value'].iloc[-1]
gross_return = (final_gross_val / start_val - 1) * 100
net_return = (final_net_val / start_val - 1) * 100

annotation_text = (
    f"Startkapital: ${start_val:.2f}\n"
    f"Brutto-Endwert: ${final_gross_val:.2f} (Rendite: {gross_return:.2f}%)\n"
    f"Netto-Endwert: ${final_net_val:.2f} (Rendite: {net_return:.2f}%)"
)
ax1.text(0.02, 0.1, annotation_text, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9))

# Unterer Plot: Drawdown
max_drawdown = df_portfolio['net_drawdown_pct'].min()
ax2.fill_between(df_portfolio.index, df_portfolio['net_drawdown_pct'], 0, color='salmon', alpha=0.7)
ax2.set_title(f"Drawdown der Nettoperformance (Max: {max_drawdown:.2f}%)", fontsize=14, pad=10)
ax2.set_ylabel('Drawdown (%)', fontsize=14)
ax2.set_xlabel('Datum', fontsize=14)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

# Datumsformatierung
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=30, ha='right')

plt.tight_layout(pad=2.0)
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')

print("Chart erfolgreich erstellt.")
print(f"\n--- Finale Werte ---")
print(f"Brutto-Rendite: {gross_return:.2f}%")
print(f"Netto-Rendite: {net_return:.2f}%")
print(f"Maximaler Drawdown: {max_drawdown:.2f}%") 