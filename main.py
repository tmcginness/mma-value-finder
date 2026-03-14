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
    Predict upcoming fights using a trained model and historical fighter data.

    Loads all historical fights, builds features for the upcoming matchups,
    and outputs win probabilities + value assessment against betting lines.
    """
    from models.base_model import BaseFightModel
    from config.settings import american_to_implied_prob, implied_prob_to_american

    # Load upcoming fights
    upcoming_path = args.fights if hasattr(args, "fights") and args.fights else "data/upcoming_fights.csv"
    try:
        upcoming = pd.read_csv(upcoming_path)
    except FileNotFoundError:
        print(f"[ERROR] {upcoming_path} not found.")
        print()
        print("Create a CSV with columns: fighter_a,fighter_b[,fighter_a_line,fighter_b_line]")
        print("Example:")
        print('  fighter_a,fighter_b,fighter_a_line,fighter_b_line')
        print('  "Jon Jones","Stipe Miocic",-400,+300')
        sys.exit(1)

    # Load historical fights and build fighter histories
    print("Loading historical fight data...")
    fights = load_fights()
    print(f"Loaded {len(fights)} historical fights")

    print("Building feature matrix from historical fights...")
    fights_with_features = build_feature_matrix(fights)

    # Build fighter histories for upcoming fight feature computation
    from features.feature_engineering import _build_fighter_histories, _get_fighter_stats_before
    fighter_histories = _build_fighter_histories(
        fights_with_features.sort_values("date").reset_index(drop=True)
    )

    # Load or train model
    model_name = args.model
    model = MODEL_REGISTRY[model_name]()

    if model_name == "elo":
        # Elo needs to process all historical fights to build ratings
        valid = fights_with_features[
            (fights_with_features.get("has_features", False) == True)
            & (fights_with_features["is_draw_nc"] == False)
        ]
        model.fit(valid, valid["fighter_a_won"])
    else:
        model_path = f"models/{model_name}_model.pkl"
        try:
            model = BaseFightModel.load(model_path)
            print(f"Loaded saved {model_name} model")
        except FileNotFoundError:
            print(f"No saved model found. Training {model_name} on all historical data...")
            valid = fights_with_features[
                (fights_with_features.get("has_features", False) == True)
                & (fights_with_features["is_draw_nc"] == False)
            ]
            available_features = [f for f in MODEL_FEATURES if f in valid.columns]
            model.fit(valid[available_features], valid["fighter_a_won"])

    # Determine available features
    available_features = [f for f in MODEL_FEATURES if f in fights_with_features.columns]

    # Use latest date as reference for computing features
    reference_date = fights["date"].max() + pd.Timedelta(days=1)

    print(f"\n{'='*70}")
    print(f"PREDICTIONS — {model_name.upper()} MODEL")
    print(f"{'='*70}")

    has_lines = "fighter_a_line" in upcoming.columns

    for _, matchup in upcoming.iterrows():
        a_name = matchup["fighter_a"].strip()
        b_name = matchup["fighter_b"].strip()

        print(f"\n{a_name} vs {b_name}")
        print(f"{'─'*50}")

        if model_name == "elo":
            # Elo uses ratings directly
            prob_a = model._expected_score(
                model._get_rating(a_name), model._get_rating(b_name)
            )
            rating_a = model._get_rating(a_name)
            rating_b = model._get_rating(b_name)
            print(f"  Elo Ratings: {a_name} {rating_a:.0f} | {b_name} {rating_b:.0f}")
        else:
            # Feature-based model
            a_stats = _get_fighter_stats_before(fighter_histories, a_name, reference_date)
            b_stats = _get_fighter_stats_before(fighter_histories, b_name, reference_date)

            if a_stats is None or b_stats is None:
                missing = []
                if a_stats is None:
                    missing.append(a_name)
                if b_stats is None:
                    missing.append(b_name)
                print(f"  [SKIP] Not enough history for: {', '.join(missing)}")
                continue

            features = {}
            for stat_name in a_stats:
                features[f"{stat_name}_diff"] = a_stats[stat_name] - b_stats[stat_name]

            feature_df = pd.DataFrame([features])
            X = feature_df[[f for f in available_features if f in feature_df.columns]]
            prob_a = model.predict_proba(X)[0]

        prob_b = 1 - prob_a
        print(f"  Model: {a_name} {prob_a:.1%} | {b_name} {prob_b:.1%}")
        print(f"  Fair line: {a_name} {implied_prob_to_american(prob_a):+d} | {b_name} {implied_prob_to_american(prob_b):+d}")

        if has_lines and pd.notna(matchup.get("fighter_a_line")):
            line_a = int(matchup["fighter_a_line"])
            line_b = int(matchup["fighter_b_line"])
            implied_a = american_to_implied_prob(line_a)
            implied_b = american_to_implied_prob(line_b)

            edge_a = prob_a - implied_a
            edge_b = prob_b - implied_b

            print(f"  Market: {a_name} {line_a:+d} ({implied_a:.1%}) | {b_name} {line_b:+d} ({implied_b:.1%})")

            if edge_a > MIN_EDGE_THRESHOLD:
                print(f"  >>> VALUE: {a_name} ({edge_a:+.1%} edge)")
            elif edge_b > MIN_EDGE_THRESHOLD:
                print(f"  >>> VALUE: {b_name} ({edge_b:+.1%} edge)")
            else:
                print(f"  No value found (edges: {edge_a:+.1%} / {edge_b:+.1%})")

    print()


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
    pred_parser = subparsers.add_parser("predict", help="Predict upcoming fights")
    pred_parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()), default="elo",
        help="Model type (default: elo)"
    )
    pred_parser.add_argument(
        "--fights", type=str, default="data/upcoming_fights.csv",
        help="Path to upcoming fights CSV"
    )

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
