"""
XGBoost gradient boosted trees model.

Why use this:
- Handles non-linear feature interactions (e.g., reach advantage matters more
  when combined with a striking style)
- Built-in regularization helps with small datasets
- Feature importance via gain/SHAP values
- Generally the go-to for structured tabular prediction tasks

Caveat: Easier to overfit on small MMA datasets. Use conservative hyperparameters
and always validate on held-out time periods.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from models.base_model import BaseFightModel
from config.settings import XGBOOST_PARAMS, RANDOM_SEED


class XGBoostFightModel(BaseFightModel):
    """XGBoost fight outcome model."""

    def __init__(self):
        super().__init__(name="xgboost")

        self.model = XGBClassifier(**XGBOOST_PARAMS)
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train XGBoost on features X and outcomes y."""
        self.feature_names = list(X.columns)
        X_clean = X.fillna(-999)  # XGBoost handles missing natively, but explicit is safer

        self.model.fit(
            X_clean, y,
            verbose=False,
        )
        self.is_fitted = True

        # Print feature importance (gain-based)
        importance = self.get_feature_importance()
        print(f"\n{'Feature':<40} {'Importance':>12}")
        print("-" * 54)
        for _, row in importance.head(15).iterrows():
            print(f"  {row['feature']:<38} {row['importance']:>12.4f}")
        print()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(fighter_a wins) for each row."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_clean = X.fillna(-999)
        return self.model.predict_proba(X_clean)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance (gain) as a DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        importance = self.model.feature_importances_
        return (
            pd.DataFrame({
                "feature": self.feature_names,
                "importance": importance,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
