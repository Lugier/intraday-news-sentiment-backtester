## Intraday News Sentiment & Event-Study Backtester

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Issues](https://img.shields.io/github/issues/Lugier/intraday-news-sentiment-backtester.svg)](https://github.com/Lugier/intraday-news-sentiment-backtester/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/Lugier/intraday-news-sentiment-backtester.svg)](https://github.com/Lugier/intraday-news-sentiment-backtester/pulls)
[![Last Commit](https://img.shields.io/github/last-commit/Lugier/intraday-news-sentiment-backtester.svg)](https://github.com/Lugier/intraday-news-sentiment-backtester/commits/main)

Reproduzierbare End‑to‑End‑Pipeline für den Preis‑Impact von Finanznachrichten im Minutenbereich:
- Hochqualitative News‑Beschaffung und -Filterung (Quellen‑Whitelist)
- LLM‑basierte Sentiment‑Analyse (Gemini) im Batchbetrieb
- Realistisches Intraday‑Backtesting (Entry‑Delay, Stop‑Loss, Gebühren, Slippage, Flip)
- Event‑Study (AR/AAR/CAAR), Bootstrap‑Signifikanz und Random‑Benchmarks
- Portfolio‑Aggregation, Visualisierungen und Reports

Enthaltene Komponenten:
- `Finaler Run/` konsolidierte Ergebnisordner und Abbildungen pro Ticker und Index‑Analysen
- `src/` Anwendungscode (Pipelines, Analyse, Backtesting, News, LLM)
- `Test_scripts/` Skripte für zusätzliche Auswertungen
- `requirements.txt` Abhängigkeiten

> [!TIP]
> Für einen schnellen Einstieg siehe „Schnellstart (TL;DR)“.

### Inhaltsverzeichnis
- [Schnellstart (TL;DR)](#schnellstart-tldr)
- [Funktionen](#funktionen)
- [Architektur](#architektur)
- [Verzeichnisstruktur](#verzeichnisstruktur)
- [Installation](#installation)
- [Konfiguration (.env)](#konfiguration-env)
- [Ausführen der Pipeline](#ausführen-der-pipeline)
- [CLI und Beispiele](#cli-und-beispiele)
- [Wissenschaftlicher Kontext & Ergebnisse](#wissenschaftlicher-kontext-und-kernergebnisse-aus-der-bachelorarbeit)
- [Visuals (Beispiele)](#visuals-beispiele)
- [Output, Reproduzierbarkeit und Nachnutzung](#output-reproduzierbarkeit-und-nachnutzung)
- [Post‑Processing‑Skripte](#post-processing-skripte-testscripts)
- [Performance‑Hinweise](#performance-hinweise)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Roadmap](#roadmap)
- [Lizenz & Zitation](#lizenz--zitation)

### Schnellstart (TL;DR)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# .env anlegen (siehe Abschnitt „Konfiguration (.env)“)

# Einzelticker
python -m src.main --ticker AAPL --output-dir "Finaler Run"

# Dow 30 (Batch)
python -m src.main --dow --output-dir "Finaler Run"
```

### Architektur
```mermaid
graph TD
    A[News Fetcher<br/>(TickerTick API)] --> B[Quellen-Filter<br/>& Qualitätsregeln]
    B --> C[LLM-Sentiment<br/>(Gemini)]
    C --> D[Trading-Backtester]
    D --> E[Event Study
    • AR/AAR
    • CAAR]
    D --> F[Bootstrap-Tests
    • p-Werte
    • Konfidenzintervalle]
    D --> G[Random Benchmarks]
    E --> H[Portfolio-Aggregation]
    F --> H
    G --> H
    H --> I[Visualisierungen & Reports]
    I --> J[Finaler Run/<TICKER>/*]
```

### Funktionen
- **LLM-basiertes Sentiment**: Batch-Verarbeitung über Gemini (Modell/Temperatur/Parallelität konfigurierbar).
- **Realistisches Backtesting**: Entry-Delay, Stop-Loss/Take-Profit, Gebühren, Slippage, After-Hours, Parallelisierung.
- **Event-Study**: Marktmodell, mehrfache Zeitfenster (5, 15, 30, 60 Min), AAR/CAAR mit Signifikanztests.
- **Statistik & Robustheit**: Bootstrap, Konfidenzintervalle, Random-Benchmark-Vergleiche.
- **Portfolio & Visuals**: Zusammenfassungen, Modell-Fit-Kennzahlen, PNG/JSON-Ausgaben.

## Verzeichnisstruktur
```
Finaler Run/            # Ergebnisse pro Ticker inkl. Backtests, Event-Study, Statistiken
src/                    # Applikationscode (Pipelines, Analyse, Backtesting, News, LLM)
Test_scripts/           # Zusatz-Analysen für fertige Ergebnisse
requirements.txt        # Python-Abhängigkeiten
```

### Pipeline-Module
- `src/news/processing/pipeline.py`: End‑to‑End News → Sentiment Pipeline je Ticker
- `src/backtesting/execution/pipeline.py`: Regelbasiertes Intraday‑Backtesting inkl. Gebühren/Slippage
- `src/analysis/event_study.py`: AAR/CAAR, Marktmodell, Fenster 5/15/30/60 Min
- `src/analysis/bootstrap.py`: Bootstrap‑Signifikanztests der Backtest‑Ergebnisse
- `src/analysis/portfolio_*.py`: Portfolio‑Aggregation, Benchmarks, Statistik
- `src/backtesting/analysis/backtest_visualizer.py`: Visuals (Equity, Win‑Rate, Sentiment‑Plots)

Typische Inhalte von `Finaler Run/<TICKER>/`:
- `<TICKER>_news_all.json`, `<TICKER>_news_all_curated.json` (Roh- und kuratierte Nachrichten)
- `backtest_<DATUM_UHRZEIT>_*` und `backtest_results_<TICKER>.json`
- `event_study/` Grafiken und JSON-Outputs
- `statistical_tests/` Bootstrap-Ergebnisse und HTML-Reports
- `*_filter_statistics_summary.txt` Filter-Tracking je Ticker

## Installation

### Voraussetzungen
- Python 3.10+
- API-Zugänge:
  - TickerTick (News) via `pytickertick`
  - Gemini (LLM) via `google-generativeai`
  - Polygon (Marktdaten, optional/empfohlen) via `polygon-api-client`

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Konfiguration (.env)
Erzeuge `.env` im Projektroot:
```bash
GEMINI_API_KEY=dein_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=/absoluter/pfad/zu/service_account.json
POLYGON_API_KEY=dein_polygon_api_key
```
Hinweise:
- `.env` und Credentials nicht committen (durch `.gitignore` ausgeschlossen).
- `src/config.py` liest Umgebungsvariablen und bietet Defaults – setze deine Schlüssel.

Empfohlene Variablen:

| Variable | Beschreibung | Beispiel |
|---|---|---|
| `GEMINI_API_KEY` | API‑Key für Google Gemini | `AIza...` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Pfad zur Service‑Account‑JSON | `/Users/du/key.json` |
| `POLYGON_API_KEY` | API‑Key für Marktdaten | `abcd-...` |

## Ausführen der Pipeline

Empfohlen aus dem Projektroot:
```bash
python -m src.main --help
```

### CLI und Beispiele

| Flag | Typ | Beschreibung |
|---|---|---|
| `--ticker` | str | Einzelner Ticker, z. B. `AAPL`. |
| `--dow` | flag | Dow-Jones-Komponenten verarbeiten. |
| `--mag7` | flag | Magnificent 7 verarbeiten. |
| `--log` | flag | Benutzerdefinierte Liste (LOG) verarbeiten. |
| `--top5` | flag | Top-5-Tech verarbeiten. |
| `--portfolio-only` | flag | Nur konsolidierte Portfolio-Analyse erstellen. |
| `--max-articles` | int | Max. Artikel pro Ticker (Default aus Config; `None` = alle). |
| `--start-date` | str | Startdatum `YYYY-MM-DD`. |
| `--end-date` | str | Enddatum `YYYY-MM-DD`. |
| `--entry-delay` | int | Minuten Verzögerung vor Einstieg. |
| `--holding-period` | int | Haltedauer in Minuten. |
| `--max-workers` | int | Parallele Worker für Backtesting. |
| `--use-stop-loss` | flag | Stop-Loss aktivieren. |
| `--stop-loss-pct` | float | Stop-Loss in %. |
| `--take-profit-pct` | float | Take-Profit in % (`inf` zum Deaktivieren). |
| `--include-after-hours` | flag | Extended Trading einbeziehen. |
| `--use-flip-strategy` | flag | Signale invertieren (Kontra). |
| `--transaction-fee-pct` | float | Gebührenanteil pro Trade (z. B. `0.001` = 0,1%). |
| `--market-index` | str | Marktindex für Event-Study (Default `SPY`). |
| `--disable-event-study` | flag | Event-Study deaktivieren. |
| `--run-statistical-tests` | flag | Bootstrap/Statistik aktivieren. |
| `--significance-level` | float | Signifikanzniveau (Default `0.05`). |
| `--bootstrap-simulations` | int | Bootstrap-Iterationen (Default `10000`). |
| `--output-dir` | str | Ausgabeordner (z. B. `"Finaler Run"`). |
| `--clean` | flag | Ausgabeordner vorab leeren. |

### Beispiele

- Einzelner Ticker mit sinnvollen Defaults, Ausgabe nach `Finaler Run/`:
```bash
python -m src.main --ticker AAPL \
  --max-articles 3000 \
  --entry-delay 2 \
  --holding-period 60 \
  --use-stop-loss --stop-loss-pct 2.0 \
  --transaction-fee-pct 0.001 \
  --output-dir "Finaler Run"
```

- Dow-30 Batch mit Event-Study:
```bash
python -m src.main --dow --output-dir "Finaler Run"
```

- Nur Portfolio-Zusammenfassung (setzt vorhandene Einzelergebnisse voraus):
```bash
python -m src.main --dow --portfolio-only --output-dir "Finaler Run"
```

## Wissenschaftlicher Kontext und Kernergebnisse (aus der Bachelorarbeit)

Die in `BA.txt` dokumentierte Studie nutzt exakt diese Pipeline. Untersuchungsrahmen und Kennzahlen:

- **Stichprobe & Zeitraum**: 57 Large-Cap-Aktien (S&P‑500-Segmente), 03.03.2025–17.06.2025
- **Nachrichtenbasis**: 27.864 Artikel abgerufen; nach Qualitätsfiltern 20.297 valide News-Events
- **Sentiment-Events**: 62.218 (ein Artikel kann mehrere Ticker betreffen)
- **Handelssignale**: 2.386 Trades (Einstieg t+2 Min, Haltedauer bis 60 Min, Stop‑Loss 2%)

### Event-Study (AR/AAR/CAAR)
- Für 5/15/30/60 Minuten-Fenster: **keine statistisch signifikanten AAR/CAAR** (alle p > 0,05)
- Beispiel 60‑Minuten‑CAAR (positiv): 0,07 %; 95‑%-CI (Bootstrap, Portfolioebene): [-0,053 %, +0,228 %]
- Erklärungsgüte des einfachen Querschnitts (AAR ~ Sentiment‑Score): **R² 0,0029–0,0143**

### Backtesting (Regelbasierte Strategie)
- Brutto-Gesamtrendite: **-27,89 %**; mit Gebühren (0,1 % pro Trade) Netto: **-99,39 %**
- Win Rate: **29,8 %** (711/2.386 Trades)
- Asymmetrie: Positives News‑Sentiment kumuliert **-53,69 % (1.016 Trades)**, negatives **+31,95 % (1.370 Trades)** – ökonomisch dennoch nicht tragfähig nach Kosten

### Weitere Muster
- **Aufmerksamkeits-Paradoxon**: Starke negative Korrelation zwischen Nachrichtenintensität und Performance (**r = -0,895**); besonders medienpräsente Tech‑Titel (z. B. TSLA, NVDA, AAPL) mit schwächster Strategieperformance
- **Sektorales Bild** (Ø‑Renditen): Konsumgüter ≈ -1,68 %, Health‑Care ≈ -2,48 %, Finanzen ≈ -8,48 %, Technologie ≈ -21,40 %
- **Wochentagseffekt**: Dienstag leicht positiv (+0,04 %), ansonsten nahe Null/negativ – ohne statistische oder ökonomische Relevanz
- **Random‑Benchmark**: Strategie im 32,4‑Perzentil (p ≈ 0,64) – unterlegen gegenüber Zufallsstrategien

### Einordnung
- Ergebnisse stützen für das untersuchte Segment die **mittelstarke EMH**: Öffentliches Sentiment wird offenbar innerhalb sehr kurzer Zeit eingepreist (vermutlich < 1 Minute), sodass **minutenbasierte** Analysen keine handelbaren Effekte mehr erfassen.
- Beobachtete Asymmetrien (Negatives > Positives) bleiben **nach Kosten** ökonomisch irrelevant; können aber auf höhere Klassifikationsschärfe bei negativen Meldungen hindeuten.

### Limitationen und Ausblick
- Minutenauflösung erfasst nicht Sub‑Sekunden‑Reaktionen; Tick‑Daten oder Sekundenbar‑Analysen könnten verborgene, extrem kurzfristige Effekte zeigen.
- Benchmark SPY erzeugt endogene Kopplung; individuelle Index‑Exklusion pro Aktie wäre ideal (erhöhter Aufwand).
- Gebühren, Slippage, Marktimpact konservativ modelliert – reale Kosten oft höher.
- Zeitraum (≈ 3,5 Monate), Large‑Caps, ein LLM (Gemini), trinitäre Sentiment‑Skala – begrenzte Generalisierbarkeit.
- Perspektiven: Small‑Caps/Emerging/Crypto, Ereignisfenster um Earnings/FOMC, **Multi‑LLM‑Ensembles**, hybride Features (z. B. Optionsdaten), Sentiment‑Intensität/Konfidenz.

### Zentrale Ergebnisse in Kürze
- **Keine intraday-signifikanten AAR/CAAR** in 5–60 Minuten-Fenstern (starkes EMH-Signal)
- **Strategie unprofitabel nach Kosten** (Netto ≈ −99 %), Random-Benchmark überlegen (p ≈ 0,64)
- **Negatives Sentiment > Positives** in Bruttoeffekten, aber ökonomisch nicht tragfähig
- **Aufmerksamkeits-Paradoxon**: Mehr News-Intensität korreliert mit schlechterer Performance (r ≈ −0,90)
- **Tech-Sektor** zeigt schwächste Ergebnisse; einzelne Ausreißer (z. B. UNH) ohne robuste Generalisierbarkeit

### Daten- und Ereignisübersicht (Studie)

| Kennzahl | Wert |
|---|---|
| Zeitraum | 03.03.2025–17.06.2025 |
| Aktien | 57 (Large-Cap, S&P-Segmente) |
| Nachrichten abgerufen | 27.864 |
| Valide News-Events | 20.297 |
| Sentiment-Events | 62.218 |
| Handelssignale (Trades) | 2.386 |

### Event-Study: Ergebnisse je Sentiment (Auswahl)

| Sentiment | Fenster | Kennzahl | Wert |
|---|---|---|---|
| Positiv | 60 Min | CAAR | 0,07 % |
| Positiv | 60 Min | 95 %-CI (Bootstrap, Portfolio) | [−0,053 %, +0,228 %] |
| Negativ | 60 Min | CAAR | ≈ 0,00 % |
| Neutral | 60 Min | CAAR | 0,02 % |

Hinweis: In allen Fenstern keine statistische Signifikanz (p > 0,05).

### Sektorale Performance (Backtesting, Brutto)

| Sektor | Ø Rendite (%) | Median (%) | Std (%) | Ø Trades | Gesamt Trades |
|---|---:|---:|---:|---:|---:|
| Konsumgüter | −1,68 | −1,14 | 2,96 | 7 | 91 |
| Health‑Care | −2,48 | −1,54 | 10,22 | 15 | 236 |
| Finanzen | −8,48 | −4,88 | 27,00 | 27 | 388 |
| Technologie | −21,40 | −9,87 | 23,77 | 111 | 1.671 |

### Wochentagseffekte (Backtesting, Ø Rendite)

| Tag | Trades | Ø Rendite (%) | Std (%) |
|---|---:|---:|---:|
| Montag | 429 | −0,04 | 0,98 |
| Dienstag | 545 | +0,04 | 0,85 |
| Mittwoch | 499 | −0,02 | 0,87 |
| Donnerstag | 501 | 0,00 | 1,18 |
| Freitag | 412 | −0,04 | 0,89 |

### Modellgüte und Fehlermaße

| Metrik | Wert |
|---|---|
| R² (AAR ~ Sentiment-Score, 5–60 Min) | 0,0029–0,0143 |
| RMSE (Marktmodell, log-Returns) | ≈ 0,00123 |
| MAE (Marktmodell, log-Returns) | ≈ 0,00064 |

### Formeln (Auszug)

\[ AR_{i,t} = R_{i,t} - E\left[R_{i,t}\right] \]

\[ R_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right), \quad R_{m,t} = \ln\left(\frac{P_{m,t}}{P_{m,t-1}}\right) \]

\[ R_{i,t} = \alpha_i + \beta_i \cdot R_{m,t} + \varepsilon_{i,t} \]\
\[ AAR_{s,t} = \frac{1}{N_s} \sum_{i\in s} AR_{i,t}, \quad CAAR_{s,t} = \sum_{\tau \le t} AAR_{s,\tau} \]

\[ t = \frac{AAR_{s,t}}{\frac{s_{s,t}}{\sqrt{N_s}}} \]

### Interpretation & Implikationen
- Ergebnisse stützen die **mittelstarke EMH** für hochliquide Large‑Caps im untersuchten Zeitraum.
- **Minutenbasierte** Signalnutzung reicht nicht aus; Einpreisung erfolgt vermutlich schneller (Sekunden‑Bereich).
- **Transaktionskosten** sind in schnellen Strategien entscheidend – Bruttoeffekte reichen nicht aus.
- **Qualität statt Quantität**: Mehr News‑Volumen verschlechtert Performance (Wettbewerb, Beobachtung, Effizienz).

### Reproduktion der Studienergebnisse mit dieser Pipeline

1) Umgebung und `.env` wie oben beschrieben einrichten
2) Für die untersuchten Tickergruppen Run(s) starten, z. B. Dow + MAG7:
```bash
python -m src.main --dow --output-dir "Finaler Run"
python -m src.main --mag7 --output-dir "Finaler Run"
```
3) Optional: Fenster/Parameter analog der Studie setzen (Entry‑Delay 2, Haltedauer 60, Stop‑Loss 2 %, Fees 0,1 %)
```bash
python -m src.main --ticker AAPL \
  --entry-delay 2 --holding-period 60 --use-stop-loss --stop-loss-pct 2.0 \
  --transaction-fee-pct 0.001 --output-dir "Finaler Run"
```
4) Portfolio‑Zusammenfassung erzeugen (wenn Einzelläufe vorliegen):
```bash
python -m src.main --dow --portfolio-only --output-dir "Finaler Run"
```
5) Post‑Processing: Bootstrap‑CIs und Sequenz‑Charts
```bash
python Test_scripts/compute_portfolio_bootstrap_ci.py
python Test_scripts/create_sequential_performance_chart.py
```

<details>
<summary>Methodik‑Highlights (ausführlich)</summary>

- Quellen‑Whitelist für Nachrichtenqualität; strikte Filterung irrelevanter/mehrdeutiger Artikel
- LLM‑Prompt mit 5‑Kriterien‑Framework (Impact, Überraschung, Trigger, Zahlen, Katalysatoren)
- Strikte 60‑Minuten‑Definition; Entry t+2 Minuten; Stop‑Loss 2 %; optional Flip‑Strategie
- Marktmodell‑Schätzung mit Mindestdatenpunkten; Ausschluss nicht robuster Events
- Bootstrap‑Tests, Random‑Benchmark, Portfolio‑Aggregation, Modell‑Fit‑Statistiken

</details>

## Visuals (Beispiele)
> [!NOTE]
> Beispiel‑Abbildungen. Weitere Visuals liegen unter `Finaler Run/<TICKER>/...`.

| Event‑Study (TSLA) | Sentiment‑Verteilung (MSFT) |
|---|---|
| ![TSLA Event Study](Finaler%20Run/TSLA/event_study/TSLA_event_study.png) | ![MSFT Sentiment Distribution](Finaler%20Run/MSFT/sentiment_distribution_MSFT.png) |

| Kumulierte Renditen (NVDA) | Win‑Rate (PEP) |
|---|---|
| ![NVDA Cumulative](Finaler%20Run/NVDA/backtest_NVDA_20250617_212212/backtest_NVDA_20250617_212217/cumulative_returns.png) | ![PEP Win Rate](Finaler%20Run/PEP/backtest_PEP_20250617_223733/backtest_PEP_20250617_223735/win_rate.png) |

## Output, Reproduzierbarkeit und Nachnutzung
- Alle Ergebnisse werden zeitgestempelt unter `Finaler Run/<TICKER>/...` abgelegt.
- Filter‑Entscheidungen und Trade‑Zahlen pro Ticker: `*_filter_statistics_summary.txt`.
- Event‑Study: JSON + PNG je Fenster; Backtests: per‑Trade JSON inkl. Gebühren, Slippage, PnL, Equity.
- Portfolio‑Analysen: CAAR‑Signifikanz, Modell‑Fit (R²/p‑Werte), Summary‑Reports.

## Post‑Processing‑Skripte (`Test_scripts/`)
- `create_sequential_performance_chart.py`: Sequenzielle Performance‑Charts aus Backtests.
- `compute_portfolio_bootstrap_ci.py`: Bootstrap‑Konfidenzintervalle auf Portfolioebene.
- `redraw_event_study_clean.py`: Event‑Study‑Plots aus gespeicherten JSONs neu zeichnen.

Aufruf jeweils mit `--help`:
```bash
python Test_scripts/create_sequential_performance_chart.py --help
```

## Tipps & Troubleshooting
- Bei API‑Rate‑Limits Delays in `src/config.py` (`API_CONFIG`) erhöhen.
- Für schnelle Testläufe `--max-articles` und `--bootstrap-simulations` reduzieren.
- `--include-after-hours` nur nutzen, wenn Marktdatenanbieter Extended Hours sauber liefert.
- Secrets/Credentials niemals committen (`.env`, Service‑Accounts sind ausgeschlossen).

## Performance‑Hinweise
- Parallelisierung über `--max-workers` sowie Batch‑Größen in `src/config.py`.
- Für schnelle Iterationen `--max-articles` und `--bootstrap-simulations` reduzieren.
- Bei API‑Limits Pausen in `API_CONFIG` erhöhen.

## Troubleshooting & FAQ
> [!WARNING]
> „No API key provided“ – Prüfe `.env` und `src/config.py`.

- „Zu wenige Events für Event‑Study“: Estimationsfenster zu klein; mehr Daten oder anderen Index wählen.
- „Bilder fehlen im Output“: Stelle sicher, dass `VIZ_CONFIG` das Rendern aktiviert.
- „After‑Hours wirkt inkonsistent“: Verifiziere, ob der Datenanbieter Extended Hours konsistent liefert.

## Roadmap
- Sekundengenaue Daten / Tick‑Daten
- Multi‑LLM‑Ensembles
- Erweiterte Benchmarks (z. B. Optionen, sektorale Indizes ohne Eigengewicht)
- CI‑Workflow (Lint/Tests) via GitHub Actions

## Lizenz & Zitation
- Für akademische Nutzung bitte mit Repository‑URL und Commit‑Hash zitieren, um genaue Ergebnisse zu fixieren.
 - Ergänze optional eine Lizenzdatei (`LICENSE`), z. B. MIT oder Apache‑2.0.

