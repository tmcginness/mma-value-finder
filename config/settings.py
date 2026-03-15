"""
Central configuration for the MMA Value Finder.
Tweak these knobs as you iterate.
"""

# ── Scraping ─────────────────────────────────────────────────────────
UFCSTATS_BASE_URL = "http://www.ufcstats.com/statistics/events/completed"
UFCSTATS_FIGHT_URL = "http://www.ufcstats.com/fight-details/"
UFCSTATS_FIGHTER_URL = "http://www.ufcstats.com/fighter-details/"
REQUEST_DELAY_SECONDS = 1.5  # Be polite to the server

# ── Tapology ────────────────────────────────────────────────────────
TAPOLOGY_BASE_URL = "https://www.tapology.com"
TAPOLOGY_DELAY_SECONDS = 3.0  # More conservative — smaller community site

# ── Data Paths ───────────────────────────────────────────────────────
RAW_FIGHTS_CSV = "data/raw_fights.csv"
RAW_FIGHTERS_CSV = "data/raw_fighters.csv"
PROCESSED_FIGHTS_CSV = "data/processed_fights.csv"
HISTORICAL_LINES_CSV = "data/historical_lines.csv"  # You'll need to source this
TAPOLOGY_RECORDS_CSV = "data/tapology_records.csv"

# ── Feature Engineering ──────────────────────────────────────────────
# Number of past fights to use for rolling averages
ROLLING_WINDOW = 5

# Features fed into the model (per fighter, computed as fighter_a - fighter_b differential)
MODEL_FEATURES = [
    # Win/loss record
    "win_rate_diff",                # Rolling win rate
    "win_streak_diff",              # Current win/loss streak
    "recent_form_diff",             # Win rate in last 3 fights
    "ufc_experience_diff",          # Number of UFC fights

    # Finish tendencies
    "finish_rate_diff",             # % of wins by finish
    "ko_rate_diff",                 # % of wins by KO/TKO
    "sub_rate_diff",                # % of wins by submission
    "decision_rate_diff",           # % of fights going to decision
    "first_round_finish_rate_diff", # % of wins in round 1

    # Activity / conditioning
    "days_since_last_fight_diff",   # Ring rust indicator
    "fights_per_year_diff",         # Career activity rate
    "avg_fight_rounds_diff",        # Average fight duration (rounds)

    # Resilience
    "loss_recovery_diff",           # How well they bounce back from losses

    # ── Detailed per-fight stats (from fight detail scrape) ────
    # Striking volume & power
    "sig_str_landed_pm_diff",       # Sig. strikes landed per minute
    "sig_str_absorbed_pm_diff",     # Sig. strikes absorbed per minute
    "kd_pm_diff",                   # Knockdowns per minute

    # Striking accuracy & defense
    "sig_str_accuracy_diff",        # Striking accuracy (landed/attempted)
    "sig_str_defense_diff",         # Strike defense (1 - opp accuracy)

    # Grappling
    "td_landed_pm_diff",            # Takedowns landed per minute
    "td_accuracy_diff",             # Takedown accuracy
    "td_defense_diff",              # Takedown defense
    "ctrl_sec_pm_diff",             # Control time per minute
    "sub_att_pm_diff",              # Submission attempts per minute

    # Style indicators
    "head_pct_diff",                # % strikes to head
    "body_pct_diff",                # % strikes to body
    "distance_pct_diff",            # % strikes at distance
    "ground_pct_diff",              # % strikes on ground
]

# ── Model Settings ───────────────────────────────────────────────────
RANDOM_SEED = 42

LOGISTIC_PARAMS = {
    "C": 1.0,
    "l1_ratio": 0,  # equivalent to L2 penalty
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
