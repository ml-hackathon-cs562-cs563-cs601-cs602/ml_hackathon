"""XGBoost model wrapper for PathSnap candidate scoring.

Family: Boosting (gradient boosting decision trees)
Owner: Member D / Pipeline V2
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


class XGBoostModel:
    """XGBoost binary classifier for candidate scoring."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        default_params = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": 2,
        }
        self.params = {**default_params, **self.config.get("params", {})}
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self.train_time_s: Optional[float] = None
        self.infer_time_s: Optional[float] = None

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        trip_ids_train: Optional[np.ndarray] = None,
        trip_ids_val: Optional[np.ndarray] = None,
    ) -> "XGBoostModel":
        """Fit XGBoost model."""
        start_time = time.perf_counter()

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_mat = X_train.values
        else:
            X_mat = np.asarray(X_train, dtype=np.float32)
            self.feature_names = [f"feature_{i}" for i in range(X_mat.shape[1])]

        y_vec = np.asarray(y_train, dtype=int).ravel()

        self.model = xgb.XGBClassifier(**self.params)
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(X_mat, y_vec, eval_set=eval_set, verbose=False)
        self.train_time_s = time.perf_counter() - start_time
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        start_time = time.perf_counter()
        if isinstance(X, pd.DataFrame):
            X_mat = X.values
        else:
            X_mat = np.asarray(X, dtype=np.float32)

        proba = self.model.predict_proba(X_mat)
        self.infer_time_s = time.perf_counter() - start_time
        return proba

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary labels."""
        p = self.predict_proba(X)
        return (p[:, 1] >= threshold).astype(int)

    def save(self, path: Union[str, Path]) -> None:
        """Save model and configuration."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "config": self.config,
            "params": self.params,
            "feature_names": self.feature_names,
            "train_time_s": self.train_time_s,
        }, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "XGBoostModel":
        """Load model from path."""
        data = joblib.load(path)
        instance = cls(config=data.get("config", {}))
        instance.model = data["model"]
        instance.params = data.get("params", {})
        instance.feature_names = data.get("feature_names")
        instance.train_time_s = data.get("train_time_s")
        return instance
