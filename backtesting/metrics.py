"""
Evaluation metrics for fight prediction models and betting performance.

Two categories:
1. Model quality metrics (how good are the probability estimates?)
2. Betting performance metrics (does the model make money against the lines?)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score


# ── Model Quality Metrics ────────────────────────────────────────────

def model_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Simple accuracy at a given threshold."""
    return accuracy_score(y_true, (y_prob >= threshold).astype(int))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Brier score — mean squared error of probability estimates.
    Lower is better. Range [0, 1].
    A model that always predicts 0.5 scores 0.25.
    """
    return brier_score_loss(y_true, y_prob)


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Log loss — heavily penalizes confident wrong predictions."""
    # Clip to avoid log(0)
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return log_loss(y_true, y_prob)


def calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Calibration analysis: bin predictions by confidence and compare
    predicted vs actual win rates.

    A well-calibrated model: when it says 70%, the fighter wins ~70% of the time.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{bins[i]:.0%}-{bins[i+1]:.0%}" for i in range(n_bins)]

    rows = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": bin_labels[i],
            "count": int(mask.sum()),
            "avg_predicted": float(y_prob[mask].mean()),
            "avg_actual": float(y_true[mask].mean()),
            "calibration_error": float(abs(y_prob[mask].mean() - y_true[mask].mean())),
        })

    return pd.DataFrame(rows)


# ── Betting Performance Metrics ──────────────────────────────────────

def compute_bet_results(
    y_true: np.ndarray,
    model_prob: np.ndarray,
    implied_prob: np.ndarray,
    decimal_odds: np.ndarray,
    min_edge: float = 0.05,
    bet_size: float = 10.0,
) -> pd.DataFrame:
    """
    Simulate flat-bet results for fights where model found value.

    A bet is placed when: model_prob - implied_prob > min_edge

    Returns DataFrame with per-bet P&L.
    """
    edge = model_prob - implied_prob
    bet_mask = edge > min_edge

    results = []
    for i in range(len(y_true)):
        if not bet_mask[i]:
            continue

        won_bet = bool(y_true[i] == 1)  # We always bet on fighter_a when edge is positive
        pnl = bet_size * (decimal_odds[i] - 1) if won_bet else -bet_size

        results.append({
            "fight_idx": i,
            "model_prob": model_prob[i],
            "implied_prob": implied_prob[i],
            "edge": edge[i],
            "decimal_odds": decimal_odds[i],
            "won_bet": won_bet,
            "bet_size": bet_size,
            "pnl": pnl,
        })

    return pd.DataFrame(results)


def betting_summary(bet_results: pd.DataFrame) -> dict:
    """Compute summary betting statistics from bet results."""
    if bet_results.empty:
        return {"total_bets": 0, "message": "No bets placed (edge threshold too high?)"}

    total_bets = len(bet_results)
    wins = bet_results["won_bet"].sum()
    total_wagered = bet_results["bet_size"].sum()
    total_pnl = bet_results["pnl"].sum()
    roi = total_pnl / total_wagered if total_wagered > 0 else 0

    return {
        "total_bets": total_bets,
        "wins": int(wins),
        "losses": total_bets - int(wins),
        "win_rate": wins / total_bets,
        "total_wagered": total_wagered,
        "total_pnl": total_pnl,
        "roi": roi,
        "avg_edge": bet_results["edge"].mean(),
        "avg_odds": bet_results["decimal_odds"].mean(),
        "max_drawdown": _max_drawdown(bet_results["pnl"].values),
        "longest_losing_streak": _longest_streak(~bet_results["won_bet"]),
    }


def kelly_bet_size(model_prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """
    Kelly criterion bet sizing.

    Full Kelly: f = (p * (d - 1) - (1 - p)) / (d - 1)
    where p = win probability, d = decimal odds

    We use fractional Kelly (default 1/4) because full Kelly is way too aggressive
    for the uncertainty in our estimates.
    """
    edge = model_prob * (decimal_odds - 1) - (1 - model_prob)
    if edge <= 0:
        return 0.0

    full_kelly = edge / (decimal_odds - 1)
    return full_kelly * fraction


# ── Helpers ──────────────────────────────────────────────────────────

def _max_drawdown(pnl_series: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown in cumulative P&L."""
    cumulative = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(drawdown.max()) if len(drawdown) > 0 else 0.0


def _longest_streak(bool_series: pd.Series) -> int:
    """Longest consecutive True streak."""
    if bool_series.empty:
        return 0
    groups = bool_series.ne(bool_series.shift()).cumsum()
    streak_lengths = bool_series.groupby(groups).sum()
    return int(streak_lengths.max()) if not streak_lengths.empty else 0
