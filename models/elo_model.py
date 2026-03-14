"""
Elo Rating System for MMA.

How it works:
- Every fighter starts at 1500
- After each fight, the winner gains rating points and the loser drops
- The K-factor controls how much ratings change per fight
- Finishes cause bigger rating swings (optional bonus)
- Win probability is derived from the rating gap via logistic curve

Why it's useful:
- No feature engineering needed — it learns purely from win/loss sequences
- Naturally handles opponent quality (beating a 1800-rated fighter matters
  more than beating a 1200-rated fighter)
- Great as a standalone baseline or as an additional feature for other models
- Very interpretable: "Fighter A is rated 1650 vs Fighter B at 1450"
"""

import numpy as np
import pandas as pd

from models.base_model import BaseFightModel
from config.settings import ELO_PARAMS


class EloFightModel(BaseFightModel):
    """Elo-based fight prediction model."""

    def __init__(self):
        super().__init__(name="elo")
        self.ratings: dict[str, float] = {}
        self.k_factor = ELO_PARAMS["k_factor"]
        self.finish_bonus = ELO_PARAMS["finish_bonus"]
        self.scale = ELO_PARAMS["scale_factor"]
        self.initial = ELO_PARAMS["initial_rating"]
        self.history: list[dict] = []  # Track rating changes over time

    def _get_rating(self, fighter: str) -> float:
        """Get a fighter's current rating, initializing if new."""
        if fighter not in self.ratings:
            self.ratings[fighter] = self.initial
        return self.ratings[fighter]

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Logistic expected score: P(A beats B) given their ratings."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / self.scale))

    def _update(self, winner: str, loser: str, is_finish: bool = False):
        """Update ratings after a fight result."""
        r_w = self._get_rating(winner)
        r_l = self._get_rating(loser)

        expected_w = self._expected_score(r_w, r_l)
        expected_l = 1 - expected_w

        k = self.k_factor
        if is_finish:
            k *= (1 + self.finish_bonus)

        # Winner: actual score = 1, Loser: actual score = 0
        self.ratings[winner] = r_w + k * (1 - expected_w)
        self.ratings[loser] = r_l + k * (0 - expected_l)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        'Fit' the Elo model by processing fights chronologically.

        X must contain columns: fighter_a, fighter_b, date, method_clean
        y is fighter_a_won (1 or 0)
        """
        # X here is actually the full fight DataFrame (we need names)
        self.ratings = {}
        self.history = []

        for idx in range(len(X)):
            fighter_a = X.iloc[idx]["fighter_a"]
            fighter_b = X.iloc[idx]["fighter_b"]
            a_won = y.iloc[idx]
            method = X.iloc[idx].get("method_clean", "")
            is_finish = method in ("ko_tko", "submission")
            date = X.iloc[idx].get("date", None)

            # Record pre-fight ratings
            r_a_pre = self._get_rating(fighter_a)
            r_b_pre = self._get_rating(fighter_b)
            expected_a = self._expected_score(r_a_pre, r_b_pre)

            # Update based on result
            if a_won:
                self._update(fighter_a, fighter_b, is_finish)
            else:
                self._update(fighter_b, fighter_a, is_finish)

            self.history.append({
                "date": date,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "rating_a_pre": r_a_pre,
                "rating_b_pre": r_b_pre,
                "expected_a": expected_a,
                "a_won": a_won,
                "rating_a_post": self.ratings[fighter_a],
                "rating_b_post": self.ratings[fighter_b],
            })

        self.is_fitted = True

        # Summary
        top_rated = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        print(f"\nElo model processed {len(X)} fights, {len(self.ratings)} fighters")
        print(f"\nTop 10 rated fighters:")
        for name, rating in top_rated[:10]:
            print(f"  {name:<30} {rating:.0f}")
        print()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return P(fighter_a wins) based on current Elo ratings.
        X must contain fighter_a and fighter_b columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        probs = []
        for idx in range(len(X)):
            fighter_a = X.iloc[idx]["fighter_a"]
            fighter_b = X.iloc[idx]["fighter_b"]
            r_a = self._get_rating(fighter_a)
            r_b = self._get_rating(fighter_b)
            probs.append(self._expected_score(r_a, r_b))

        return np.array(probs)

    def get_rating_history(self) -> pd.DataFrame:
        """Return the full rating update history for analysis."""
        return pd.DataFrame(self.history)

    def get_current_ratings(self) -> pd.DataFrame:
        """Return current ratings for all fighters."""
        return (
            pd.DataFrame(
                list(self.ratings.items()),
                columns=["fighter", "rating"],
            )
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )
