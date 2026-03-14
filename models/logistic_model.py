"""
Logistic Regression baseline model.

Why start here:
- Interpretable: you can see exactly which features matter and how much
- Fast to train: iterate quickly on feature engineering
- Surprisingly competitive: for structured tabular data, often within a few
  percentage points of more complex models
- Good calibration: logistic regression naturally outputs well-calibrated probabilities
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from models.base_model import BaseFightModel
from config.settings import LOGISTIC_PARAMS, RANDOM_SEED


class LogisticFightModel(BaseFightModel):
    """Logistic regression fight outcome model."""

    def __init__(self):
        super().__init__(name="logistic")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                random_state=RANDOM_SEED,
                **LOGISTIC_PARAMS,
            )),
        ])

        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train logistic regression on features X and outcomes y."""
        self.feature_names = list(X.columns)

        # Handle any remaining NaNs with median imputation
        X_clean = X.fillna(X.median())

        self.pipeline.fit(X_clean, y)
        self.is_fitted = True

        # Print feature importance (coefficients)
        coefs = self.pipeline.named_steps["clf"].coef_[0]
        importance = sorted(
            zip(self.feature_names, coefs),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        print(f"\n{'Feature':<40} {'Coefficient':>12}")
        print("-" * 54)
        for feat, coef in importance:
            print(f"  {feat:<38} {coef:>12.4f}")
        print()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(fighter_a wins) for each row."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_clean = X.fillna(X.median() if len(X) > 1 else 0)
        # predict_proba returns [[P(0), P(1)]], we want P(1) = P(fighter_a wins)
        return self.pipeline.predict_proba(X_clean)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature coefficients as a DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        coefs = self.pipeline.named_steps["clf"].coef_[0]
        return (
            pd.DataFrame({"feature": self.feature_names, "coefficient": coefs})
            .assign(abs_coef=lambda df: df["coefficient"].abs())
            .sort_values("abs_coef", ascending=False)
            .drop(columns="abs_coef")
            .reset_index(drop=True)
        )
