"""
CLI entry point for the MMA Value Finder.

Usage:
    python main.py scrape                          # Scrape fight data
    python main.py train --model logistic           # Train a model
    python main.py backtest --model logistic         # Run backtest
    python main.py evaluate --model logistic         # Model quality report
    python main.py predict --model logistic          # Predict upcoming fights
"""

import argparse
import sys

import pandas as pd
import numpy as np

from scraping.ufcstats_scraper import UFCStatsScraper
from data.data_loader import load_fights, load_historical_lines, merge_fights_with_lines
from features.feature_engineering import build_feature_matrix
from models import MODEL_REGISTRY
from backtesting.backtester import Backtester
from config.settings import MODEL_FEATURES, MIN_EDGE_THRESHOLD


def cmd_scrape(args):
    """Scrape UFC fight data from UFCStats.com."""
    print("Starting scrape of UFCStats.com...")
    scraper = UFCStatsScraper()
    df = scraper.scrape_all_events(save=True)
    print(f"\nDone! Scraped {len(df)} fights.")
    print(f"Saved to data/raw_fights.csv")


def cmd_train(args):
    """Train a model and save it."""
    print(f"Loading fight data...")
    fights = load_fights()
    print(f"Loaded {len(fights)} fights")

    print(f"Building features...")
    fights = build_feature_matrix(fights)

    # Filter to fights with valid features and non-draw outcomes
    valid = fights[(fights["has_features"] == True) & (fights["is_draw_nc"] == False)]
    print(f"{len(valid)} fights with valid features")

    # Determine which features are actually available
    available_features = [f for f in MODEL_FEATURES if f in valid.columns]
    if not available_features:
        # Fall back to any _diff columns that exist
        available_features = [c for c in valid.columns if c.endswith("_diff")]

    if not available_features:
        print("[ERROR] No features available. You may need to enrich the scraper with per-fight stats.")
        sys.exit(1)

    print(f"Using {len(available_features)} features: {available_features[:5]}...")

    # Instantiate and train
    model_name = args.model
    if model_name not in MODEL_REGISTRY:
        print(f"[ERROR] Unknown model: {model_name}. Options: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    model = MODEL_REGISTRY[model_name]()

    if model_name == "elo":
        model.fit(valid, valid["fighter_a_won"])
    else:
        X = valid[available_features]
        y = valid["fighter_a_won"]
        model.fit(X, y)

    model.save()
    print(f"Model '{model_name}' trained and saved.")


def cmd_backtest(args):
    """Run walk-forward backtest."""
    print(f"Loading fight data...")
    fights = load_fights()

    # Try to merge betting lines
    lines = load_historical_lines()
    fights = merge_fights_with_lines(fights, lines)

    print(f"Building features...")
    fights = build_feature_matrix(fights)

    # Determine available features
    available_features = [f for f in MODEL_FEATURES if f in fights.columns]
    if not available_features:
        available_features = [c for c in fights.columns if c.endswith("_diff")]

    model_name = args.model
    model = MODEL_REGISTRY[model_name]()

    min_edge = args.min_edge if hasattr(args, "min_edge") else MIN_EDGE_THRESHOLD

    backtester = Backtester(model, fights, available_features)
    results = backtester.run(
        test_start_date=args.test_start if hasattr(args, "test_start") else None,
        min_edge=min_edge,
    )
    backtester.print_report(results)


def cmd_evaluate(args):
    """Quick model evaluation (same as backtest but focused on model quality)."""
    # Reuse backtest with reporting focused on calibration
    cmd_backtest(args)


def cmd_predict(args):
    """
    Predict upcoming fights.

    Placeholder — once you have a trained model and upcoming fight card data,
    this will generate predictions and flag value spots.
    """
    print("Predict mode — coming soon!")
    print()
    print("To use this, you'll need to:")
    print("  1. Have a trained model (run `python main.py train --model logistic`)")
    print("  2. Create a CSV of upcoming fights with fighter names")
    print("  3. The model will compute features from historical data and output probabilities")
    print()
    print("Example upcoming_fights.csv format:")
    print("  fighter_a,fighter_b,fighter_a_line,fighter_b_line")
    print('  "Jon Jones","Stipe Miocic",-400,+300')


def main():
    parser = argparse.ArgumentParser(
        description="MMA Value Finder — fight prediction and betting value tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scrape
    subparsers.add_parser("scrape", help="Scrape fight data from UFCStats.com")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a prediction model")
    train_parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default="logistic",
        help="Model type to train (default: logistic)"
    )

    # Backtest
    bt_parser = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    bt_parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default="logistic"
    )
    bt_parser.add_argument(
        "--min-edge", type=float, default=MIN_EDGE_THRESHOLD,
        help="Minimum edge to place a bet (default: 0.05)"
    )
    bt_parser.add_argument(
        "--test-start", type=str, default=None,
        help="Start date for test period (YYYY-MM-DD). Default: last 20%%."
    )

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model calibration")
    eval_parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default="logistic"
    )

    # Predict
    subparsers.add_parser("predict", help="Predict upcoming fights")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "scrape": cmd_scrape,
        "train": cmd_train,
        "backtest": cmd_backtest,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
