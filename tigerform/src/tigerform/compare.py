"""Compare a user swing against the Tiger reference.

Produces:
* per-feature z-score deviations (how many std from Tiger's mean),
* a 0-100 similarity score (discriminator-importance-weighted, optionally
  blended with the discriminator's Tiger-likeness probability),
* a DTW-based sequence similarity on normalized swing time-series,
* a ranked list of the biggest deviations to drive coaching feedback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import config
from .features import SwingFeatures
from .model import TigerDiscriminator
from .reference import ReferenceTemplate


@dataclass
class Deviation:
    key: str
    base: str
    event: Optional[str]
    value: float
    tiger_mean: float
    tiger_std: float
    z: float
    direction: str          # "high" (above Tiger) or "low" (below Tiger)
    weight: float

    @property
    def severity(self) -> float:
        return self.weight * abs(self.z)


@dataclass
class ComparisonResult:
    similarity_score: float          # 0-100 overall match to Tiger
    sequence_similarity: float       # 0-100 from DTW on swing curves
    tiger_likeness: Optional[float]  # discriminator probability (0-1) or None
    zscores: Dict[str, float]
    deviations: List[Deviation]      # sorted, most severe first
    dtw: Dict[str, float]
    weights: Dict[str, float]


def _zscore_similarity(zscores: Dict[str, float], weights: Dict[str, float]) -> float:
    if not zscores:
        return 0.0
    num = sum(weights.get(k, 0.0) * abs(z) for k, z in zscores.items())
    den = sum(weights.get(k, 0.0) for k in zscores) or 1.0
    weighted_abs_z = num / den
    return float(100.0 * np.exp(-0.5 * weighted_abs_z))


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Shape-focused DTW distance between two series (z-normalized first)."""
    from fastdtw import fastdtw

    def zn(s):
        s = np.asarray(s, float)
        s = s[np.isfinite(s)]
        sd = s.std()
        return (s - s.mean()) / sd if sd > 1e-6 else s - s.mean()

    za, zb = zn(a), zn(b)
    if len(za) < 2 or len(zb) < 2:
        return float("nan")
    dist, _ = fastdtw(za, zb, dist=lambda x, y: abs(x - y))
    return float(dist / max(len(za), len(zb)))   # length-normalized


def compare_swing(features: SwingFeatures, reference: ReferenceTemplate,
                  discriminator: Optional[TigerDiscriminator] = None
                  ) -> ComparisonResult:
    zscores = reference.zscores(features)

    # Feature weights: discriminator importance if available, else uniform.
    if discriminator is not None:
        weights = discriminator.feature_importance()
    else:
        weights = {k: 1.0 / max(len(zscores), 1) for k in zscores}

    score_sim = _zscore_similarity(zscores, weights)

    tiger_likeness = None
    final = score_sim
    if discriminator is not None:
        tiger_likeness = discriminator.tiger_likeness(features)
        final = 0.6 * score_sim + 0.4 * (100.0 * tiger_likeness)

    # DTW on shared time series.
    dtw: Dict[str, float] = {}
    for name, ref_curve in reference.series.items():
        if name in features.series:
            dtw[name] = _dtw_distance(features.series[name], np.array(ref_curve))
    valid = [d for d in dtw.values() if np.isfinite(d)]
    seq_sim = float(100.0 * np.exp(-0.4 * np.mean(valid))) if valid else float("nan")

    # Rank deviations.
    deviations: List[Deviation] = []
    for k, z in zscores.items():
        if abs(z) < config.ZSCORE_FLAG_THRESHOLD:
            continue
        deviations.append(Deviation(
            key=k, base=config.base_of(k), event=config.event_of(k),
            value=features.scalars.get(k, float("nan")),
            tiger_mean=reference.mean.get(k, float("nan")),
            tiger_std=reference.std_floored(k),
            z=z, direction="high" if z > 0 else "low",
            weight=weights.get(k, 0.0),
        ))
    deviations.sort(key=lambda d: d.severity, reverse=True)

    return ComparisonResult(
        similarity_score=round(final, 1),
        sequence_similarity=round(seq_sim, 1) if np.isfinite(seq_sim) else float("nan"),
        tiger_likeness=round(tiger_likeness, 3) if tiger_likeness is not None else None,
        zscores=zscores, deviations=deviations, dtw=dtw, weights=weights,
    )
