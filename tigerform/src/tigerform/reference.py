"""The Tiger reference template.

Aggregates a set of reference (Tiger) swings into per-feature mean/std
distributions plus mean time-series curves for DTW. Stored as JSON (derived
features only — never the source footage). User swings are later compared
against this template via z-scores in `compare.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config
from .features import SwingFeatures

SERIES_LEN = 100   # series resampled to this length before averaging


@dataclass
class ReferenceTemplate:
    keys: List[str]
    mean: Dict[str, float]
    std: Dict[str, float]
    n: int
    series: Dict[str, List[float]] = field(default_factory=dict)

    def std_floored(self, key: str) -> float:
        """Std with a floor so z-scores stay finite and well-scaled."""
        m, s = self.mean.get(key, 0.0), self.std.get(key, 0.0)
        return max(s, 0.02 * abs(m), 1e-3)

    def zscores(self, feats: SwingFeatures, clip: float = 10.0) -> Dict[str, float]:
        """Per-feature z-scores, clipped to +/-`clip` so a single degenerate
        feature (e.g. near-zero reference variance) can't dominate the score."""
        z: Dict[str, float] = {}
        for k in self.keys:
            v = feats.scalars.get(k, np.nan)
            if np.isfinite(v) and k in self.mean:
                z[k] = float(np.clip((v - self.mean[k]) / self.std_floored(k), -clip, clip))
        return z

    # ------------------------------------------------------------------ I/O
    def save(self, path: Path = config.REFERENCE_TEMPLATE_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "keys": self.keys, "mean": self.mean, "std": self.std,
            "n": self.n, "series": self.series,
        }, indent=2))
        return path

    @classmethod
    def load(cls, path: Path = config.REFERENCE_TEMPLATE_PATH) -> "ReferenceTemplate":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No reference template at {path}. Run scripts/build_reference.py first.")
        d = json.loads(path.read_text())
        return cls(keys=d["keys"], mean=d["mean"], std=d["std"],
                   n=d["n"], series=d.get("series", {}))


def _resample(series: np.ndarray, length: int = SERIES_LEN) -> np.ndarray:
    series = np.asarray(series, float)
    good = np.isfinite(series)
    if good.sum() < 2:
        return np.zeros(length)
    xs = np.linspace(0, 1, good.sum())
    return np.interp(np.linspace(0, 1, length), xs, series[good])


def build_reference(features_list: List[SwingFeatures]) -> ReferenceTemplate:
    if not features_list:
        raise ValueError("Need at least one reference swing.")
    keys = config.flat_feature_keys()

    mean, std = {}, {}
    for k in keys:
        vals = np.array([f.scalars.get(k, np.nan) for f in features_list], float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            mean[k] = float(np.mean(vals))
            std[k] = float(np.std(vals))

    series_names = features_list[0].series.keys()
    series: Dict[str, List[float]] = {}
    for name in series_names:
        stacked = np.stack([_resample(f.series[name]) for f in features_list
                            if name in f.series])
        series[name] = np.nanmean(stacked, axis=0).tolist()

    return ReferenceTemplate(keys=list(mean.keys()), mean=mean, std=std,
                             n=len(features_list), series=series)
