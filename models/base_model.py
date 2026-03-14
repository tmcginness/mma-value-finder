"""
Abstract base class for all fight prediction models.

Every model must:
1. Accept a feature matrix and binary outcomes for training
2. Output calibrated probabilities (not just class predictions)
3. Support serialization for backtesting reproducibility
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class BaseFightModel(ABC):
    """Interface that all fight prediction models implement."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on features X and binary outcomes y."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return P(fighter_a wins) for each row.
        Must return values in [0, 1].
        """
        pass

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction with configurable threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: Optional[str] = None):
        """Serialize model to disk."""
        path = path or f"models/{self.name}_model.pkl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseFightModel":
        """Load a serialized model."""
        model = joblib.load(path)
        if not isinstance(model, BaseFightModel):
            raise TypeError(f"Loaded object is not a BaseFightModel: {type(model)}")
        return model

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
