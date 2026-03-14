# MMA Value Finder

A Python toolkit for modeling MMA fight outcomes, estimating win probabilities, and identifying value in betting lines through backtesting.

## Project Structure

```
mma-value-finder/
├── config/
│   └── settings.py          # Central configuration (feature lists, model params, thresholds)
├── scraping/
│   └── ufcstats_scraper.py  # Scrape fight/fighter data from UFCStats.com
├── data/
│   └── data_loader.py       # Load, clean, and merge raw data into modeling-ready DataFrames
├── features/
│   └── feature_engineering.py # Derive all features from raw stats
├── models/
│   ├── base_model.py        # Abstract base class for all models
│   ├── logistic_model.py    # Logistic regression baseline
│   ├── xgboost_model.py     # Gradient boosted trees
│   └── elo_model.py         # Elo rating system
├── backtesting/
│   ├── backtester.py        # Core backtesting engine
│   └── metrics.py           # ROI, Brier score, calibration, CLV
├── notebooks/
│   └── exploration.ipynb    # Starter notebook for EDA and iteration
├── main.py                  # CLI entry point: train, evaluate, backtest
├── requirements.txt
└── README.md
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1. Scrape data (builds local CSV cache)
python main.py scrape

# 2. Train a model
python main.py train --model logistic

# 3. Backtest against historical lines
python main.py backtest --model logistic --min-edge 0.05

# 4. Evaluate model calibration
python main.py evaluate --model logistic
```

## How It Works

1. **Scrape** historical fight data (stats, results) from UFCStats.com
2. **Engineer features** per fighter per fight (rolling averages, differentials, matchup indicators)
3. **Train a model** that outputs P(fighter A wins)
4. **Compare** model probability vs. implied probability from betting line
5. **Backtest** by simulating bets on historical fights where model found "value"

## Key Concepts

- **Implied probability**: A -150 line implies 60% win probability. Convert with: `implied = abs(line) / (abs(line) + 100)` for favorites, `implied = 100 / (line + 100)` for underdogs.
- **Edge**: `model_prob - implied_prob`. Positive edge = potential value.
- **Brier score**: Measures calibration — does your 70% really win 70% of the time?
- **CLV (Closing Line Value)**: Did you beat the closing line? The gold standard for sharp betting.

## Iteration Tips

- Start with logistic regression on 5-10 features. Get the pipeline working end to end.
- Add features one at a time and measure impact on Brier score and backtest ROI.
- Watch for lookahead bias — only use data available BEFORE each fight.
- Small sample sizes are your enemy. Trust calibration metrics over raw ROI.
- Weight class and era matter — a model trained on all data may miss context.
