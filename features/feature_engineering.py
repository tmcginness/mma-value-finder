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

        for role in ["a", "b"]:
            name = fight[f"fighter_{role}"]
            opponent = fight[f"fighter_{'b' if role == 'a' else 'a'}"]

            if name not in histories:
                histories[name] = []

            entry = {
                "date": date,
                "opponent": opponent,
                "won": 1 if name == winner else 0,
                "method": fight.get("method_clean", "unknown"),
                "weight_class": fight.get("weight_class_clean", ""),
                # TODO: When you have per-fight detailed stats from scraping
                # fight detail pages, add them here:
                # "sig_strikes_landed": ...,
                # "sig_strikes_attempted": ...,
                # "sig_strikes_absorbed": ...,
                # "takedowns_landed": ...,
                # "takedowns_attempted": ...,
                # "sub_attempts": ...,
                # "control_time_seconds": ...,
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

    # ── Days since last fight (ring rust) ────────────────────────
    last_fight_date = prior_fights[-1]["date"]
    stats["days_since_last_fight"] = (before_date - last_fight_date).days

    # ── Placeholder stats (fill in when you have per-fight detail data) ──
    # For now, use career-level stats or zeros as placeholders.
    # Once you scrape fight detail pages, replace these with rolling avgs.
    stats["sig_strikes_landed_pm"] = 0.0
    stats["sig_strike_accuracy"] = 0.0
    stats["sig_strikes_absorbed_pm"] = 0.0
    stats["strike_defense"] = 0.0
    stats["takedown_avg"] = 0.0
    stats["takedown_accuracy"] = 0.0
    stats["takedown_defense"] = 0.0
    stats["sub_attempts_avg"] = 0.0
    stats["reach"] = 0.0
    stats["age"] = 0.0
    stats["striking_differential"] = 0.0  # landed - absorbed per min

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


# ── Additional Feature Ideas (implement as you iterate) ──────────────
#
# Stylistic matchup features:
#   - striker_vs_grappler: classify each fighter's style, encode matchup
#   - southpaw_advantage: reach/stance interaction
#   - pace_mismatch: one fighter high output vs low output opponent
#
# Contextual features:
#   - title_fight: championship fights may produce different dynamics
#   - main_card: position on card as proxy for perceived competitiveness
#   - altitude: some venues at elevation affect cardio
#   - short_notice: replacement fighters have worse stats
#
# Opponent quality:
#   - strength_of_schedule: average opponent win rate
#   - best_win_quality: Elo of best defeated opponent
#   - level_of_competition_jump: moving from regional to UFC, etc.
