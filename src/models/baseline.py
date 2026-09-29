"""Baseline model wrappers for tabular feature classification.
Supports Logistic Regression, Random Forest, and LightGBM / XGBoost.
Handles feature scaling, class balancing, threshold tuning, and model persistence.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.utils.metrics import find_optimal_threshold, calculate_metrics, measure_inference_latency


class PhishingClassifier:
    """Unified wrapper around tabular classification models."""

    def __init__(self, model_type: str = "lightgbm", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.scaler = None
        self.optimal_threshold = 0.5
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == "logistic_regression":
            self.scaler = StandardScaler()
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
                solver="liblinear"
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type in ["lightgbm", "lgb"]:
            return lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=10,
                learning_rate=0.08,
                num_leaves=31,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Fit model with optional threshold optimization on validation data."""
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X)
        else:
            X_train = X

        self.model.fit(X_train, y)

        # Optimize decision threshold on validation set only
        if X_val is not None and y_val is not None and len(y_val) > 0:
            if self.scaler is not None:
                X_val_scaled = self.scaler.transform(X_val)
            else:
                X_val_scaled = X_val
            val_probs = self.model.predict_proba(X_val_scaled)[:, 1]
            self.optimal_threshold = find_optimal_threshold(y_val, val_probs)
        else:
            self.optimal_threshold = 0.5

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.scaler is not None:
            X_proc = self.scaler.transform(X)
        else:
            X_proc = X
        return self.model.predict_proba(X_proc)

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Predict binary class using calibrated threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)[:, 1]
        return (probs >= thresh).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, model_path: str = None) -> dict:
        """Evaluate model and return complete metrics dictionary."""
        probs = self.predict_proba(X_test)[:, 1]
        latency = measure_inference_latency(self, X_test)
        size_mb = 0.0
        if model_path and os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)

        return calculate_metrics(
            y_true=y_test,
            y_probs=probs,
            threshold=self.optimal_threshold,
            latency_ms=latency,
            model_size_mb=size_mb
        )

    def save(self, filepath: str):
        """Save model and scaler to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model_type": self.model_type,
            "random_state": self.random_state,
            "optimal_threshold": self.optimal_threshold,
            "scaler": self.scaler,
            "model": self.model
        }, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load trained model from disk."""
        data = joblib.load(filepath)
        clf = cls(model_type=data["model_type"], random_state=data["random_state"])
        clf.scaler = data["scaler"]
        clf.optimal_threshold = data["optimal_threshold"]
        clf.model = data["model"]
        return clf
