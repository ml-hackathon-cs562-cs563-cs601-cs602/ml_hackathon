"""Logistic Regression model wrapper for PathSnap candidate scoring.

Family: Linear probabilistic classifier
Owner: Member C
Phase 2: Baseline Models

Features:
- Handles imputation (median) and scaling (StandardScaler fit on train only)
- Predicts calibrated probabilities
- Saves and loads via joblib
- Measures training and inference time
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:
    """Logistic Regression wrapper with internal preprocessing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.params = self.config.get("params", {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 500,
            "class_weight": "balanced",
            "random_state": 42,
        })
        self.feature_names: Optional[List[str]] = None
        self.train_time_s: Optional[float] = None
        self.infer_time_s: Optional[float] = None
        self.pipeline: Optional[Pipeline] = None

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        trip_ids_train: Optional[np.ndarray] = None,
        trip_ids_val: Optional[np.ndarray] = None,
    ) -> "LogisticRegressionModel":
        """Fit preprocessing pipeline and logistic regression on training data only."""
        start_time = time.perf_counter()

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_mat = X_train.values
        else:
            X_mat = np.asarray(X_train, dtype=np.float32)
            self.feature_names = [f"feature_{i}" for i in range(X_mat.shape[1])]

        y_vec = np.asarray(y_train, dtype=int).ravel()

        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**self.params)),
        ])

        self.pipeline.fit(X_mat, y_vec)
        self.train_time_s = time.perf_counter() - start_time
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities (shape: [N, 2])."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted.")

        start_time = time.perf_counter()
        if isinstance(X, pd.DataFrame):
            X_mat = X.values
        else:
            X_mat = np.asarray(X, dtype=np.float32)

        proba = self.pipeline.predict_proba(X_mat)
        self.infer_time_s = time.perf_counter() - start_time

        # Handle single class edge case
        if proba.shape[1] == 1:
            classes = list(self.pipeline.named_steps["classifier"].classes_)
            if classes[0] == 1:
                proba = np.column_stack([np.zeros(len(X_mat)), proba[:, 0]])
            else:
                proba = np.column_stack([proba[:, 0], np.zeros(len(X_mat))])

        return proba

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary labels based on positive probability threshold."""
        p = self.predict_proba(X)
        return (p[:, 1] >= threshold).astype(int)

    def save(self, path: Union[str, Path]) -> None:
        """Save model pipeline and config to path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "config": self.config,
            "params": self.params,
            "feature_names": self.feature_names,
            "train_time_s": self.train_time_s,
        }, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LogisticRegressionModel":
        """Load model pipeline from path."""
        data = joblib.load(path)
        instance = cls(config=data.get("config", {}))
        instance.pipeline = data["pipeline"]
        instance.params = data.get("params", {})
        instance.feature_names = data.get("feature_names")
        instance.train_time_s = data.get("train_time_s")
        return instance
