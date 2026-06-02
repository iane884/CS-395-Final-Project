"""Tiger-vs-amateur discriminator.

A small supervised classifier (imputer -> scaler -> logistic regression) trained
on biomechanical feature vectors. Its positive-class probability is the
"Tiger-likeness" score, and its standardized coefficients tell us *which*
biomechanical variables most distinguish a Tiger-like swing — used both to weight
the similarity score and to prioritize coaching feedback.

Important framing: this measures match to Tiger's *signature mechanics*, not an
objective "good vs. bad" verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config
from .features import SwingFeatures


@dataclass
class TigerDiscriminator:
    pipeline: object                 # sklearn Pipeline
    keys: List[str]

    # ------------------------------------------------------------------ infer
    def tiger_likeness(self, feats: SwingFeatures) -> float:
        x = feats.vector(self.keys).reshape(1, -1)
        return float(self.pipeline.predict_proba(x)[0, 1])

    def feature_importance(self) -> Dict[str, float]:
        """Absolute standardized logistic-regression coefficients, normalized
        to sum to 1. Falls back to uniform weights for non-linear models."""
        clf = self.pipeline.named_steps.get("clf")
        coef = getattr(clf, "coef_", None)
        if coef is None:
            return {k: 1.0 / len(self.keys) for k in self.keys}
        w = np.abs(coef[0])
        total = w.sum() or 1.0
        return {k: float(w[i] / total) for i, k in enumerate(self.keys)}

    # ------------------------------------------------------------------ I/O
    def save(self, path: Path = config.DISCRIMINATOR_PATH) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "keys": self.keys}, path)
        return path

    @classmethod
    def load(cls, path: Path = config.DISCRIMINATOR_PATH) -> "TigerDiscriminator":
        import joblib
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No discriminator at {path}. Run scripts/train_discriminator.py first.")
        d = joblib.load(path)
        return cls(pipeline=d["pipeline"], keys=d["keys"])


def _build_pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])


def train_discriminator(features_list: List[SwingFeatures], labels: List[int],
                        keys: Optional[List[str]] = None
                        ) -> Tuple[TigerDiscriminator, Dict]:
    """Fit the discriminator and return it alongside cross-validated metrics."""
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 roc_auc_score, confusion_matrix)

    keys = keys or config.flat_feature_keys()
    X = np.stack([f.vector(keys) for f in features_list])
    y = np.asarray(labels, int)

    pipe = _build_pipeline()
    folds = min(5, int(np.bincount(y).min()))
    metrics: Dict = {}
    if folds >= 2:
        y_pred = cross_val_predict(pipe, X, y, cv=folds)
        y_prob = cross_val_predict(pipe, X, y, cv=folds, method="predict_proba")[:, 1]
        metrics = {
            "cv_folds": folds,
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_prob)) if len(set(y)) > 1 else float("nan"),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "n_samples": int(len(y)),
        }

    pipe.fit(X, y)   # final fit on all data
    return TigerDiscriminator(pipeline=pipe, keys=keys), metrics
