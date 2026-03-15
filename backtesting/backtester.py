"""
Backtesting engine — simulates your model's performance on historical fights.

Key design decisions:
1. Walk-forward validation: train on fights before date X, predict fights after X.
   This prevents lookahead bias and simulates real-world usage.
2. Time-based splits (not random): MMA evolves — training on 2024 fights to
   predict 2018 fights would be cheating.
3. Configurable edge thresholds: only simulate bets where the model sees enough value.

Usage:
    backtester = Backtester(model, fights_df, feature_cols)
    results = backtester.run(min_edge=0.05)
    backtester.print_report(results)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from models.base_model import BaseFightModel
from backtesting.metrics import (
    model_accuracy,
    brier_score,
    log_loss_score,
    calibration_table,
    compute_bet_results,
    betting_summary,
)
from config.settings import (
    MIN_EDGE_THRESHOLD,
    DEFAULT_BET_SIZE,
    american_to_implied_prob,
    american_to_decimal,
)


class Backtester:
    """Walk-forward backtesting engine for fight prediction models."""

    def __init__(
        self,
        model: BaseFightModel,
        fights_df: pd.DataFrame,
        feature_cols: list[str],
    ):
        """
        Args:
            model: A BaseFightModel instance (untrained — we retrain during walk-forward)
            fights_df: DataFrame with features, outcomes, and optionally betting lines
            feature_cols: Column names to use as model features
        """
        self.model = model
        self.fights = fights_df.sort_values("date").reset_index(drop=True)
        self.feature_cols = feature_cols

    def run(
        self,
        test_start_date: Optional[str] = None,
        min_train_fights: int = 200,
        min_edge: float = MIN_EDGE_THRESHOLD,
        bet_size: float = DEFAULT_BET_SIZE,
        retrain_frequency: str = "quarterly",  # "per_event", "monthly", "quarterly"
    ) -> dict:
        """
        Run walk-forward backtest.

        1. Split data into train (before test_start_date) and test (after)
        2. Train model on train set
        3. Generate predictions for test set
        4. Evaluate model quality and betting performance

        If test_start_date is None, uses the last 20% of fights.
        """
        fights = self.fights.copy()

        # ── Train/test split ─────────────────────────────────────
        if test_start_date:
            split_date = pd.to_datetime(test_start_date)
        else:
            # Default: last 20% of fights
            split_idx = int(len(fights) * 0.8)
            split_date = fights.iloc[split_idx]["date"]

        train = fights[fights["date"] < split_date].copy()
        test = fights[fights["date"] >= split_date].copy()

        # Filter to rows with valid features
        valid_features = self.feature_cols
        train = train.dropna(subset=["has_features"])
        train = train[train["has_features"] == True]
        test = test.dropna(subset=["has_features"])
        test = test[test["has_features"] == True]

        # Remove draws/NCs
        train = train[train["is_draw_nc"] == False]
        test = test[test["is_draw_nc"] == False]

        if len(train) < min_train_fights:
            return {"error": f"Only {len(train)} training fights (need {min_train_fights})"}

        print(f"\n{'='*60}")
        print(f"BACKTEST: {self.model.name}")
        print(f"{'='*60}")
        print(f"Train: {len(train)} fights ({train['date'].min().date()} to {train['date'].max().date()})")
        print(f"Test:  {len(test)} fights ({test['date'].min().date()} to {test['date'].max().date()})")

        # ── Train ────────────────────────────────────────────────
        X_train = train[valid_features]
        y_train = train["fighter_a_won"]

        if self.model.name == "elo":
            # Elo needs the full DataFrame (fighter names, methods, etc.)
            self.model.fit(train, y_train)
        else:
            self.model.fit(X_train, y_train)

        # ── Predict ──────────────────────────────────────────────
        if self.model.name == "elo":
            test_probs = self.model.predict_proba(test)
        else:
            X_test = test[valid_features]
            test_probs = self.model.predict_proba(X_test)

        y_test = test["fighter_a_won"].values

        # ── Model Quality ────────────────────────────────────────
        accuracy = model_accuracy(y_test, test_probs)
        brier = brier_score(y_test, test_probs)
        logloss = log_loss_score(y_test, test_probs)
        cal_table = calibration_table(y_test, test_probs)

        model_metrics = {
            "accuracy": accuracy,
            "brier_score": brier,
            "log_loss": logloss,
            "calibration": cal_table,
            "n_test_fights": len(test),
        }

        # ── Betting Performance ──────────────────────────────────
        bet_metrics = None
        bet_results_df = pd.DataFrame()

        has_lines = (
            "fighter_a_line" in test.columns
            and test["fighter_a_line"].notna().sum() > 0
        )

        if has_lines:
            mask = test["fighter_a_line"].notna()
            test_with_lines = test[mask]
            probs_with_lines = test_probs[mask.values]

            implied_probs_a = test_with_lines["fighter_a_line"].apply(
                lambda x: american_to_implied_prob(int(x))
            ).values
            decimal_odds_a = test_with_lines["fighter_a_line"].apply(
                lambda x: american_to_decimal(int(x))
            ).values

            # Also compute fighter_b lines if available
            has_b_lines = (
                "fighter_b_line" in test_with_lines.columns
                and test_with_lines["fighter_b_line"].notna().sum() > 0
            )
            implied_probs_b = None
            decimal_odds_b = None
            if has_b_lines:
                implied_probs_b = test_with_lines["fighter_b_line"].apply(
                    lambda x: american_to_implied_prob(int(x)) if pd.notna(x) else 1.0
                ).values
                decimal_odds_b = test_with_lines["fighter_b_line"].apply(
                    lambda x: american_to_decimal(int(x)) if pd.notna(x) else 1.0
                ).values

            bet_results_df = compute_bet_results(
                y_true=test_with_lines["fighter_a_won"].values,
                model_prob=probs_with_lines,
                implied_prob_a=implied_probs_a,
                decimal_odds_a=decimal_odds_a,
                min_edge=min_edge,
                bet_size=bet_size,
                implied_prob_b=implied_probs_b,
                decimal_odds_b=decimal_odds_b,
            )
            bet_metrics = betting_summary(bet_results_df)
        else:
            print("\n[INFO] No historical betting lines found. Skipping betting simulation.")
            print("       Add lines to data/historical_lines.csv to enable backtesting.")

        return {
            "model_metrics": model_metrics,
            "bet_metrics": bet_metrics,
            "predictions": pd.DataFrame({
                "date": test["date"].values,
                "fighter_a": test["fighter_a"].values,
                "fighter_b": test["fighter_b"].values,
                "model_prob_a": test_probs,
                "actual_winner_a": y_test,
            }),
            "bet_results": bet_results_df,
        }

    def print_report(self, results: dict) -> None:
        """Print a formatted backtest report."""
        if "error" in results:
            print(f"\n[ERROR] {results['error']}")
            return

        mm = results["model_metrics"]
        print(f"\n{'─'*40}")
        print(f"MODEL QUALITY")
        print(f"{'─'*40}")
        print(f"  Accuracy:    {mm['accuracy']:.1%}")
        print(f"  Brier Score: {mm['brier_score']:.4f}  (baseline 0.25 = always predict 50%)")
        print(f"  Log Loss:    {mm['log_loss']:.4f}")
        print(f"  Test Fights: {mm['n_test_fights']}")

        print(f"\n  Calibration:")
        cal = mm["calibration"]
        if not cal.empty:
            for _, row in cal.iterrows():
                predicted = f"{row['avg_predicted']:.0%}"
                actual = f"{row['avg_actual']:.0%}"
                err = f"{row['calibration_error']:.1%}"
                print(f"    {row['bin']:>10}  pred={predicted:>4}  actual={actual:>4}  err={err:>4}  (n={row['count']})")

        bm = results.get("bet_metrics")
        if bm and bm.get("total_bets", 0) > 0:
            print(f"\n{'─'*40}")
            print(f"BETTING PERFORMANCE")
            print(f"{'─'*40}")
            print(f"  Total Bets:    {bm['total_bets']}")
            print(f"  Record:        {bm['wins']}W - {bm['losses']}L ({bm['win_rate']:.1%})")
            print(f"  Total P&L:     ${bm['total_pnl']:+.2f}")
            print(f"  ROI:           {bm['roi']:+.1%}")
            print(f"  Avg Edge:      {bm['avg_edge']:.1%}")
            print(f"  Max Drawdown:  ${bm['max_drawdown']:.2f}")
            print(f"  Longest Losing Streak: {bm['longest_losing_streak']}")
        elif bm:
            print(f"\n  {bm.get('message', 'No bets placed.')}")

        print()
