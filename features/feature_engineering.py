"""
Feature engineering for MMA fight prediction.

This is where most of your iteration will happen. The idea:
1. For each fighter entering a fight, compute rolling stats from their PRIOR fights only
   (no lookahead bias).
2. Compute differentials between the two fighters (fighter_a_stat - fighter_b_stat).
3. Feed differentials into the model.

Key principle: Every feature must only use information available BEFORE the fight.
"""

import pandas as pd
import numpy as np
from typing import Optional

from config.settings import ROLLING_WINDOW


def build_feature_matrix(fights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from raw fight data.

    For each fight, computes rolling stats for both fighters from their
    prior fights, then takes the differential.

    Returns a DataFrame aligned with fights_df, with feature columns added.
    """
    fights = fights_df.copy().sort_values("date").reset_index(drop=True)

    # Build per-fighter history lookup
    fighter_histories = _build_fighter_histories(fights)

    feature_rows = []

    for idx, fight in fights.iterrows():
        a_name = fight["fighter_a"]
        b_name = fight["fighter_b"]
        fight_date = fight["date"]

        # Get rolling stats for each fighter using only fights BEFORE this one
        a_stats = _get_fighter_stats_before(fighter_histories, a_name, fight_date)
        b_stats = _get_fighter_stats_before(fighter_histories, b_name, fight_date)

        if a_stats is None or b_stats is None:
            # Not enough history — can't compute features
            feature_rows.append({"has_features": False})
            continue

        # Compute differentials (fighter_a - fighter_b)
        features = {}
        for stat_name in a_stats:
            features[f"{stat_name}_diff"] = a_stats[stat_name] - b_stats[stat_name]

        features["has_features"] = True
        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)
    result = pd.concat([fights, features_df], axis=1)

    return result


def _build_fighter_histories(fights: pd.DataFrame) -> dict:
    """
    Build a dict mapping fighter_name -> list of per-fight stat dicts,
    sorted chronologically.

    Each fight produces TWO entries (one per fighter).
    """
    histories = {}

    for _, fight in fights.iterrows():
        date = fight["date"]
        winner = fight["winner"]
        method = fight.get("method_clean", "unknown")
        rnd = fight.get("round", None)

        # Parse round to int for round-based features
        try:
            rnd_int = int(rnd) if pd.notna(rnd) else None
        except (ValueError, TypeError):
            rnd_int = None

        for role in ["a", "b"]:
            name = fight[f"fighter_{role}"]
            opponent = fight[f"fighter_{'b' if role == 'a' else 'a'}"]

            if name not in histories:
                histories[name] = []

            entry = {
                "date": date,
                "opponent": opponent,
                "won": 1 if name == winner else 0,
                "method": method,
                "round": rnd_int,
                "weight_class": fight.get("weight_class_clean", ""),
            }
            histories[name].append(entry)

    # Sort each fighter's history chronologically
    for name in histories:
        histories[name].sort(key=lambda x: x["date"])

    return histories


def _get_fighter_stats_before(
    histories: dict, fighter_name: str, before_date, min_fights: int = 2
) -> Optional[dict]:
    """
    Compute rolling averages for a fighter using only fights before `before_date`.

    Returns None if fewer than `min_fights` prior fights exist.
    """
    if fighter_name not in histories:
        return None

    prior_fights = [
        f for f in histories[fighter_name] if f["date"] < before_date
    ]

    if len(prior_fights) < min_fights:
        return None

    # Use last N fights (rolling window)
    recent = prior_fights[-ROLLING_WINDOW:]

    stats = {}

    # ── Win metrics ──────────────────────────────────────────────
    stats["win_rate"] = np.mean([f["won"] for f in recent])
    stats["win_streak"] = _current_streak(prior_fights)
    stats["ufc_experience"] = len(prior_fights)

    # ── Finish rate ──────────────────────────────────────────────
    wins = [f for f in prior_fights if f["won"] == 1]
    if wins:
        finishes = [f for f in wins if f["method"] in ("ko_tko", "submission")]
        stats["finish_rate"] = len(finishes) / len(wins)
    else:
        stats["finish_rate"] = 0.0

    # ── KO rate and submission rate (separated) ──────────────────
    if wins:
        ko_wins = [f for f in wins if f["method"] == "ko_tko"]
        sub_wins = [f for f in wins if f["method"] == "submission"]
        stats["ko_rate"] = len(ko_wins) / len(wins)
        stats["sub_rate"] = len(sub_wins) / len(wins)
    else:
        stats["ko_rate"] = 0.0
        stats["sub_rate"] = 0.0

    # ── Decision rate (goes the distance) ────────────────────────
    decision_methods = ("unanimous_decision", "split_decision", "majority_decision")
    decision_fights = [f for f in recent if f["method"] in decision_methods]
    stats["decision_rate"] = len(decision_fights) / len(recent)

    # ── Days since last fight (ring rust) ────────────────────────
    last_fight_date = prior_fights[-1]["date"]
    stats["days_since_last_fight"] = (before_date - last_fight_date).days

    # ── Recent form (win rate in last 3) ─────────────────────────
    last_3 = prior_fights[-3:] if len(prior_fights) >= 3 else prior_fights
    stats["recent_form"] = np.mean([f["won"] for f in last_3])

    # ── Loss recovery: did they win after their last loss? ───────
    stats["loss_recovery"] = _loss_recovery_score(prior_fights)

    # ── Avg fight duration (round finished) ──────────────────────
    rounds = [f["round"] for f in recent if f["round"] is not None]
    stats["avg_fight_rounds"] = np.mean(rounds) if rounds else 2.5

    # ── First round finish rate ──────────────────────────────────
    if wins:
        r1_finishes = [f for f in wins if f["round"] == 1 and f["method"] in ("ko_tko", "submission")]
        stats["first_round_finish_rate"] = len(r1_finishes) / len(wins)
    else:
        stats["first_round_finish_rate"] = 0.0

    # ── Activity rate (fights per year over career) ──────────────
    if len(prior_fights) >= 2:
        career_days = (prior_fights[-1]["date"] - prior_fights[0]["date"]).days
        if career_days > 0:
            stats["fights_per_year"] = len(prior_fights) / (career_days / 365.25)
        else:
            stats["fights_per_year"] = 1.0
    else:
        stats["fights_per_year"] = 1.0

    return stats


def _current_streak(fight_list: list[dict]) -> int:
    """Count current win (positive) or loss (negative) streak."""
    if not fight_list:
        return 0

    streak = 0
    last_result = fight_list[-1]["won"]

    for f in reversed(fight_list):
        if f["won"] == last_result:
            streak += 1
        else:
            break

    return streak if last_result == 1 else -streak


def _loss_recovery_score(fight_list: list[dict]) -> float:
    """
    Measures how well a fighter bounces back after losses.
    Returns the fraction of losses followed by a win (0 to 1).
    Fighters who recover well from losses are more resilient.
    """
    if len(fight_list) < 2:
        return 0.5  # neutral default

    losses_followed_by = []
    for i in range(len(fight_list) - 1):
        if fight_list[i]["won"] == 0:
            losses_followed_by.append(fight_list[i + 1]["won"])

    if not losses_followed_by:
        return 1.0  # never lost, or lost only in last fight

    return np.mean(losses_followed_by)
