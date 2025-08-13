# Analysis Module Documentation

This module contains statistical analysis tools for financial news sentiment analysis and trading strategy evaluation.

## 📁 Module Structure

### Core Files

#### 1. `event_study.py` 📊
**Purpose**: Event study analysis around news events
- **What it does**: Calculates abnormal returns (AR) and cumulative abnormal returns (CAAR) around news events
- **Key functions**:
  - `run_event_study()` - Main event study pipeline
  - `calculate_ar_for_news_events()` - Calculate abnormal returns
  - `calculate_aar_and_caar()` - Calculate average abnormal returns
- **Output**: Event study results with statistical measures
- **Used for**: Measuring market reaction to news events

#### 2. `bootstrap.py` 🎲
**Purpose**: Bootstrap statistical testing for trading strategies
- **What it does**: Tests if trading strategy returns are statistically significant
- **Key functions**:
  - `run_bootstrap_on_backtest()` - Main bootstrap pipeline for individual tickers
  - `bootstrap_returns()` - Core bootstrap resampling function
- **Output**: P-values, confidence intervals, significance tests
- **Used for**: Individual ticker statistical validation

#### 3. `ttest.py` 📈
**Purpose**: T-test analysis for event studies
- **What it does**: Tests statistical significance of abnormal returns
- **Key functions**:
  - `run_ttest_on_event_study()` - Main t-test pipeline
  - `load_event_study_results()` - Load and process event study data
- **Output**: T-statistics, p-values for different time windows
- **Used for**: Event study statistical validation

#### 4. `portfolio_bootstrap.py` 🏛️ **[NEW]**
**Purpose**: Comprehensive portfolio-wide statistical analysis
- **What it does**: Advanced statistical testing for consolidated portfolio performance
- **Key features**:
  - Portfolio-wide Bootstrap tests (consolidated P-values)
  - Sharpe ratio significance testing
  - Sortino ratio analysis
  - Maximum drawdown analysis with statistical validation
  - Performance analysis by sentiment categories
  - Comprehensive visualizations and reports
- **Key functions**:
  - `PortfolioBootstrapAnalyzer` - Main analysis class
  - `run_portfolio_bootstrap_analysis()` - Convenience function
- **Output**: 
  - Detailed JSON results (`portfolio_bootstrap_analysis.json`)
  - Human-readable summary (`portfolio_bootstrap_summary.txt`)
  - Performance visualizations (`portfolio_analysis.png`)
  - Statistical charts (`bootstrap_analysis.png`)
- **Used for**: Portfolio-level statistical validation and comprehensive performance analysis

## 🔄 Analysis Workflow

### Standard Analysis Pipeline:
1. **Individual Tickers**: `bootstrap.py` + `ttest.py` for each ticker
2. **Event Studies**: `event_study.py` for market reaction analysis  
3. **Portfolio Consolidation**: `portfolio_bootstrap.py` for comprehensive portfolio analysis

### Portfolio Bootstrap Features:
- **10,000+ Bootstrap simulations** for robust statistical testing
- **Multiple risk metrics**: Sharpe, Sortino, Calmar ratios
- **Drawdown analysis**: Maximum, average, duration, frequency
- **Sentiment-based performance**: Statistical validation by positive/negative/neutral news
- **Confidence intervals**: 95% CI for all key metrics
- **P-value testing**: Two-tailed tests for statistical significance

## 📊 Usage Examples

### Individual Ticker Analysis:
```python
from src.analysis.bootstrap import run_bootstrap_on_backtest

bootstrap_results = run_bootstrap_on_backtest(
    backtest_file="results/AAPL_backtest.json",
    output_dir="analysis/AAPL",
    num_simulations=10000
)
```

### Portfolio-Wide Analysis:
```python
from src.analysis.portfolio_bootstrap import run_portfolio_bootstrap_analysis

portfolio_results = run_portfolio_bootstrap_analysis(
    trades_file="trades/portfolio_all_trades.csv",
    output_dir="analysis/portfolio",
    num_simulations=10000
)
```

### Integration in Main Pipeline:
The portfolio bootstrap analysis is automatically triggered when running index analysis:
```bash
python src/main.py --top5 --max-articles 500
```

## 📈 Statistical Methods

### Bootstrap Testing:
- **Resampling**: With replacement from original trade returns
- **Null Hypothesis**: Mean return = 0 (no strategy effect)
- **P-value calculation**: Proportion of bootstrap samples ≤ 0
- **Confidence intervals**: 2.5% and 97.5% percentiles

### Risk Metrics:
- **Sharpe Ratio**: (Mean Return - Risk-free Rate) / Standard Deviation
- **Sortino Ratio**: (Mean Return - Risk-free Rate) / Downside Deviation  
- **Calmar Ratio**: Mean Return / |Maximum Drawdown|
- **Profit Factor**: Gross Profit / |Gross Loss|

### Drawdown Analysis:
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Drawdown Duration**: Number of periods in drawdown
- **Time in Drawdown**: Percentage of time below previous high
- **Recovery Analysis**: Time to recover from drawdowns

## 🎯 Output Interpretation

### Statistical Significance:
- **P-value < 0.05**: Statistically significant results
- **P-value ≥ 0.05**: Results not distinguishable from random chance
- **Confidence Intervals**: Range of likely true values

### Performance Quality:
- **Sharpe Ratio > 1.0**: Good risk-adjusted performance
- **Sortino Ratio > 1.5**: Strong downside-adjusted performance  
- **Max Drawdown < 10%**: Conservative risk profile
- **Win Rate > 50%**: Majority of trades profitable

## 🔧 Configuration

All analysis modules support configurable parameters:
- **num_simulations**: Bootstrap iterations (default: 10,000)
- **significance_level**: Alpha level (default: 0.05)
- **risk_free_rate**: Reference rate for Sharpe calculations (default: 0.02% daily)

## 📁 File Structure Summary

```
src/analysis/
├── __init__.py           # Module exports
├── README.md            # This documentation
├── event_study.py       # Event study analysis
├── bootstrap.py         # Individual ticker bootstrap
├── ttest.py            # T-test analysis  
└── portfolio_bootstrap.py  # Portfolio-wide analysis [NEW]
```

The portfolio bootstrap module represents a significant enhancement to the statistical analysis capabilities, providing comprehensive portfolio-level validation that was previously missing from the analysis pipeline. 