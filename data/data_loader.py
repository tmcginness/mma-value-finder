"""
Data loader — reads raw scraped CSVs and produces clean, modeling-ready DataFrames.

Handles:
- Date parsing and sorting
- Deduplication
- Merging fight results with fighter career stats
- Joining historical betting lines (when available)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import RAW_FIGHTS_CSV, RAW_FIGHTERS_CSV, HISTORICAL_LINES_CSV


def load_fights(path: str = RAW_FIGHTS_CSV) -> pd.DataFrame:
    """Load and clean the raw fights CSV."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python main.py scrape` first."
        )

    df = pd.read_csv(path)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Normalize fighter names
    df["fighter_a"] = df["fighter_a"].str.strip()
    df["fighter_b"] = df["fighter_b"].str.strip()
    df["winner"] = df["winner"].str.strip()

    # Binary outcome: did fighter_a win?
    df["fighter_a_won"] = (df["winner"] == df["fighter_a"]).astype(int)

    # Flag draws / no contests
    df["is_draw_nc"] = df["winner"].str.contains("Draw|NC", case=False, na=False)

    # Clean method
    df["method_clean"] = df["method"].apply(_clean_method)

    # Weight class normalization
    df["weight_class_clean"] = df["weight_class"].str.strip().str.title()

    return df


def load_fighters(path: str = RAW_FIGHTERS_CSV) -> pd.DataFrame:
    """Load and clean fighter career stats."""
    if not Path(path).exists():
        print(f"[WARN] {path} not found. Fighter stats won't be available.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["name"] = df["name"].str.strip()

    # Parse reach to numeric inches
    if "reach" in df.columns:
        df["reach_inches"] = pd.to_numeric(df["reach"], errors="coerce")

    # Parse DOB to age-usable format
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], format="mixed", errors="coerce")

    return df


def load_historical_lines(path: str = HISTORICAL_LINES_CSV) -> pd.DataFrame:
    """
    Load historical betting lines.

    Expected columns:
        date, fighter_a, fighter_b, fighter_a_line, fighter_b_line

    Lines should be American odds (e.g., -150, +130).
    You'll need to source this data yourself — some options:
        - bestfightodds.com (manual or scrape archive)
        - Kaggle datasets
        - odds-api.com historical endpoint
    """
    if not Path(path).exists():
        print(f"[INFO] {path} not found. Backtesting will use synthetic lines.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")

    for col in ["fighter_a_line", "fighter_b_line"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def merge_fights_with_lines(
    fights: pd.DataFrame, lines: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge fight results with historical lines on date + fighter names.
    Fuzzy matching on names is a common pain point — start with exact match.
    """
    if lines.empty:
        fights["fighter_a_line"] = np.nan
        fights["fighter_b_line"] = np.nan
        return fights

    merged = fights.merge(
        lines,
        on=["date", "fighter_a", "fighter_b"],
        how="left",
        suffixes=("", "_line"),
    )

    return merged


def _clean_method(method: str) -> str:
    """Normalize method of victory into clean categories."""
    if not isinstance(method, str):
        return "unknown"
    method = method.lower().strip()
    if "ko" in method or "tko" in method:
        return "ko_tko"
    elif "sub" in method:
        return "submission"
    elif "dec" in method:
        if "split" in method:
            return "split_decision"
        elif "majority" in method:
            return "majority_decision"
        else:
            return "unanimous_decision"
    elif "draw" in method:
        return "draw"
    elif "nc" in method or "no contest" in method:
        return "no_contest"
    else:
        return "other"
