"""
Central configuration for the MMA Value Finder.
Tweak these knobs as you iterate.
"""

# ── Scraping ─────────────────────────────────────────────────────────
UFCSTATS_BASE_URL = "http://www.ufcstats.com/statistics/events/completed"
UFCSTATS_FIGHT_URL = "http://www.ufcstats.com/fight-details/"
UFCSTATS_FIGHTER_URL = "http://www.ufcstats.com/fighter-details/"
REQUEST_DELAY_SECONDS = 1.5  # Be polite to the server

# ── Data Paths ───────────────────────────────────────────────────────
RAW_FIGHTS_CSV = "data/raw_fights.csv"
RAW_FIGHTERS_CSV = "data/raw_fighters.csv"
PROCESSED_FIGHTS_CSV = "data/processed_fights.csv"
HISTORICAL_LINES_CSV = "data/historical_lines.csv"  # You'll need to source this

# ── Feature Engineering ──────────────────────────────────────────────
# Number of past fights to use for rolling averages
ROLLING_WINDOW = 5

# Features fed into the model (per fighter, computed as fighter_a - fighter_b differential)
MODEL_FEATURES = [
    # Striking
    "sig_strikes_landed_pm_diff",   # Significant strikes landed per minute
    "sig_strike_accuracy_diff",     # Significant strike accuracy %
    "sig_strikes_absorbed_pm_diff", # Sig strikes absorbed per minute
    "strike_defense_diff",          # Strike defense %

    # Grappling
    "takedown_avg_diff",            # Takedowns per 15 min
    "takedown_accuracy_diff",       # Takedown accuracy %
    "takedown_defense_diff",        # Takedown defense %
    "sub_attempts_avg_diff",        # Submission attempts per 15 min

    # Physical / contextual
    "reach_diff",                   # Reach in inches
    "age_diff",                     # Age at fight time
    "days_since_last_fight_diff",   # Ring rust indicator
    "win_streak_diff",              # Current win streak
    "ufc_experience_diff",          # Number of UFC fights

    # Derived
    "finish_rate_diff",             # % of wins by finish
    "striking_differential_diff",   # Landed - absorbed per min (net pressure)
]

# ── Model Settings ───────────────────────────────────────────────────
RANDOM_SEED = 42

LOGISTIC_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
}

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,  # Conservative — small dataset
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": RANDOM_SEED,
}

ELO_PARAMS = {
    "initial_rating": 1500,
    "k_factor": 32,            # How fast ratings change
    "finish_bonus": 0.5,       # Extra K for finishes
    "scale_factor": 400,       # Logistic curve width
}

# ── Backtesting ──────────────────────────────────────────────────────
DEFAULT_BANKROLL = 1000.0
DEFAULT_BET_SIZE = 10.0         # Flat bet size in units
MIN_EDGE_THRESHOLD = 0.05       # Only bet when model_prob - implied_prob > this
KELLY_FRACTION = 0.25           # Quarter-Kelly for conservative sizing

# ── Betting Math Helpers ─────────────────────────────────────────────
def american_to_implied_prob(line: int) -> float:
    """Convert American odds to implied probability."""
    if line < 0:
        return abs(line) / (abs(line) + 100)
    else:
        return 100 / (line + 100)

def american_to_decimal(line: int) -> float:
    """Convert American odds to decimal odds."""
    if line < 0:
        return 1 + (100 / abs(line))
    else:
        return 1 + (line / 100)

def implied_prob_to_american(prob: float) -> int:
    """Convert probability to American odds."""
    if prob >= 0.5:
        return int(-prob / (1 - prob) * 100)
    else:
        return int((1 - prob) / prob * 100)
