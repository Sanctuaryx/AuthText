from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


@dataclass
class SVMConfig:
    C: float = 1.0
    calibrated: bool = True
    calibration: str = "sigmoid"
    cv: int = 3


class BaselineSVM:
    def __init__(self, cfg: SVMConfig) -> None:
        base = LinearSVC(C=cfg.C)
        if cfg.calibrated:
            self.clf = CalibratedClassifierCV(
                base, method=cfg.calibration, cv=cfg.cv
            )
        else:
            self.clf = base
        self._scaler = MinMaxScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineSVM":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)
        scores = self.clf.decision_function(X).reshape(-1, 1)
        probs = self._scaler.fit_transform(scores)
        return np.hstack([1.0 - probs, probs])


def evaluate_binary(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    y_hat = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_hat)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
    }
