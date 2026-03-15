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
from pathlib import Path
from typing import Optional

from config.settings import ROLLING_WINDOW

FIGHT_DETAILS_CSV = "data/fight_details.csv"


def build_feature_matrix(fights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from raw fight data.

    For each fight, computes rolling stats for both fighters from their
    prior fights, then takes the differential.

    Returns a DataFrame aligned with fights_df, with feature columns added.
    """
    fights = fights_df.copy().sort_values("date").reset_index(drop=True)

    # Load detailed fight stats if available
    detail_lookup = _load_fight_details()

    # Build per-fighter history lookup
    fighter_histories = _build_fighter_histories(fights, detail_lookup)

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


def _load_fight_details() -> dict:
    """Load per-fight detail stats into a lookup by fight_url."""
    if not Path(FIGHT_DETAILS_CSV).exists():
        return {}

    df = pd.read_csv(FIGHT_DETAILS_CSV)
    lookup = {}
    for _, row in df.iterrows():
        url = row.get("fight_url", "")
        if url:
            lookup[url] = row.to_dict()
    print(f"Loaded {len(lookup)} fight detail records")
    return lookup


def _build_fighter_histories(fights: pd.DataFrame, detail_lookup: dict = None) -> dict:
    """
    Build a dict mapping fighter_name -> list of per-fight stat dicts,
    sorted chronologically.

    Each fight produces TWO entries (one per fighter).
    Includes detailed per-fight stats when available.
    """
    if detail_lookup is None:
        detail_lookup = {}

    histories = {}

    for _, fight in fights.iterrows():
        date = fight["date"]
        winner = fight["winner"]
        method = fight.get("method_clean", "unknown")
        rnd = fight.get("round", None)
        fight_url = fight.get("fight_url", "")

        # Parse round to int for round-based features
        try:
            rnd_int = int(rnd) if pd.notna(rnd) else None
        except (ValueError, TypeError):
            rnd_int = None

        # Look up detailed stats for this fight
        detail = detail_lookup.get(fight_url)

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

            # Add detailed stats if available
            # Need to figure out which side (a/b) in the detail matches this fighter
            if detail:
                my_prefix, opp_prefix = _match_detail_side(
                    name, detail
                )
                if my_prefix:
                    fight_mins = _estimate_fight_minutes(rnd_int, fight.get("time", ""))
                    entry.update(_extract_detail_stats(
                        detail, my_prefix, opp_prefix, fight_mins
                    ))

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

    # ── Detailed per-fight stats (rolling averages) ────────────
    detail_keys = [
        "sig_str_landed_pm", "sig_str_absorbed_pm", "total_str_landed_pm",
        "td_landed_pm", "td_absorbed_pm", "sub_att_pm",
        "ctrl_sec_pm", "ctrl_absorbed_pm", "kd_pm",
        "sig_str_accuracy", "td_accuracy",
        "sig_str_defense", "td_defense",
        "head_pct", "body_pct", "leg_pct",
        "distance_pct", "clinch_pct", "ground_pct",
    ]

    # Only compute if at least some recent fights have detail stats
    recent_with_details = [f for f in recent if "sig_str_landed_pm" in f]
    if recent_with_details:
        for key in detail_keys:
            vals = [f[key] for f in recent_with_details if key in f]
            if vals:
                stats[key] = np.mean(vals)

    return stats


def _match_detail_side(
    fighter_name: str, detail: dict
) -> tuple[str, str]:
    """Figure out which prefix (a_ or b_) corresponds to this fighter in the detail data."""
    detail_a = str(detail.get("fighter_a", "")).strip().lower()
    detail_b = str(detail.get("fighter_b", "")).strip().lower()
    fighter_lower = fighter_name.strip().lower()

    # Try exact match
    if fighter_lower == detail_a:
        return "a_", "b_"
    if fighter_lower == detail_b:
        return "b_", "a_"

    # Try last name match
    fighter_last = fighter_lower.split()[-1] if fighter_lower.split() else ""
    if fighter_last and len(fighter_last) > 2:
        if fighter_last in detail_a:
            return "a_", "b_"
        if fighter_last in detail_b:
            return "b_", "a_"

    return "", ""


def _estimate_fight_minutes(rnd: int | None, time_str: str) -> float:
    """Estimate total fight time in minutes from round + time."""
    if rnd is None:
        return 10.0  # 2-round default

    # Parse time string (e.g., "4:32")
    secs_in_round = 0
    if isinstance(time_str, str):
        import re
        m = re.match(r"(\d+):(\d+)", time_str.strip())
        if m:
            secs_in_round = int(m.group(1)) * 60 + int(m.group(2))

    # Each completed round = 5 minutes, plus partial final round
    total_secs = (rnd - 1) * 300 + secs_in_round
    return max(total_secs / 60.0, 1.0)


def _extract_detail_stats(
    detail: dict, my: str, opp: str, fight_mins: float
) -> dict:
    """Extract per-minute and ratio stats from fight detail for one fighter."""
    def _get(key, default=0):
        v = detail.get(key, default)
        try:
            return float(v) if pd.notna(v) else default
        except (ValueError, TypeError):
            return default

    stats = {}

    # Per-minute rates (normalize by fight duration)
    stats["sig_str_landed_pm"] = _get(f"{my}sig_str_landed") / fight_mins
    stats["sig_str_absorbed_pm"] = _get(f"{opp}sig_str_landed") / fight_mins
    stats["total_str_landed_pm"] = _get(f"{my}total_str_landed") / fight_mins
    stats["td_landed_pm"] = _get(f"{my}td_landed") / fight_mins
    stats["td_absorbed_pm"] = _get(f"{opp}td_landed") / fight_mins
    stats["sub_att_pm"] = _get(f"{my}sub_att") / fight_mins
    stats["ctrl_sec_pm"] = _get(f"{my}ctrl_sec") / fight_mins
    stats["ctrl_absorbed_pm"] = _get(f"{opp}ctrl_sec") / fight_mins
    stats["kd_pm"] = _get(f"{my}kd") / fight_mins

    # Accuracy ratios
    att = _get(f"{my}sig_str_att")
    stats["sig_str_accuracy"] = _get(f"{my}sig_str_landed") / att if att > 0 else 0.0

    td_att = _get(f"{my}td_att")
    stats["td_accuracy"] = _get(f"{my}td_landed") / td_att if td_att > 0 else 0.0

    # Defense (1 - opponent accuracy)
    opp_att = _get(f"{opp}sig_str_att")
    if opp_att > 0:
        stats["sig_str_defense"] = 1.0 - (_get(f"{opp}sig_str_landed") / opp_att)
    else:
        stats["sig_str_defense"] = 0.5

    opp_td_att = _get(f"{opp}td_att")
    if opp_td_att > 0:
        stats["td_defense"] = 1.0 - (_get(f"{opp}td_landed") / opp_td_att)
    else:
        stats["td_defense"] = 0.5

    # Strike targeting distribution (what % of sig strikes go to head/body/leg)
    total_sig = _get(f"{my}sig_str_landed")
    if total_sig > 0:
        stats["head_pct"] = _get(f"{my}head_landed") / total_sig
        stats["body_pct"] = _get(f"{my}body_landed") / total_sig
        stats["leg_pct"] = _get(f"{my}leg_landed") / total_sig
    else:
        stats["head_pct"] = 0.33
        stats["body_pct"] = 0.33
        stats["leg_pct"] = 0.33

    # Range distribution (distance vs clinch vs ground)
    if total_sig > 0:
        stats["distance_pct"] = _get(f"{my}dist_landed") / total_sig
        stats["clinch_pct"] = _get(f"{my}clinch_landed") / total_sig
        stats["ground_pct"] = _get(f"{my}ground_landed") / total_sig
    else:
        stats["distance_pct"] = 0.5
        stats["clinch_pct"] = 0.25
        stats["ground_pct"] = 0.25

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
