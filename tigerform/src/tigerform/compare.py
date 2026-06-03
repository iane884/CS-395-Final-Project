"""Compare a user swing against the Tiger reference.

Produces two orthogonal 0-100 scores plus a ranked breakdown:

* **Position match** — how close the *static* body angles/posture are to Tiger's
  at the key events (address/top/impact). Excludes timing.
* **Tempo match** — how close the *rhythm* (backswing:downswing ratio) is to
  Tiger's. Independent of positions.
* **Deviations** — the biggest per-feature differences, ranked by the trained
  discriminator's feature importances (the one thing only the supervised model
  does). The discriminator's raw probability is kept as a diagnostic, not a
  headline number, since it duplicates "position match" for users.
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
    position_match: float            # 0-100 static angles/posture vs Tiger
    tempo_match: float               # 0-100 rhythm (back:down ratio) vs Tiger
    tiger_likeness: Optional[float]  # discriminator probability (0-1); diagnostic only
    zscores: Dict[str, float]
    deviations: List[Deviation]      # sorted, most severe first
    weights: Dict[str, float]


def _zscore_similarity(zscores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Map a group of z-scores to 0-100 (100 = on Tiger's mean). Per-feature |z|
    is capped so one very-different feature can't dominate."""
    if not zscores:
        return float("nan")
    cap = config.ZSCORE_SIM_CAP
    num = sum(weights.get(k, 0.0) * min(abs(z), cap) for k, z in zscores.items())
    den = sum(weights.get(k, 0.0) for k in zscores) or 1.0
    return float(100.0 * np.exp(-0.5 * num / den))


def compare_swing(features: SwingFeatures, reference: ReferenceTemplate,
                  discriminator: Optional[TigerDiscriminator] = None
                  ) -> ComparisonResult:
    zscores = reference.zscores(features)

    # Feature weights: discriminator importance if available, else uniform...
    if discriminator is not None:
        weights = discriminator.feature_importance()
    else:
        weights = {k: 1.0 / max(len(zscores), 1) for k in zscores}
    # ...scaled by each feature's single-camera reliability so noisy / unreliable
    # features (head sway, depth-axis spine tilt, clip-dependent swing time) don't
    # dominate.
    weights = {k: weights.get(k, 0.0) *
               config.FEATURE_RELIABILITY.get(config.base_of(k), config.DEFAULT_RELIABILITY)
               for k in set(weights) | set(zscores)}

    # Split into orthogonal axes: timing (tempo) vs everything else (positions).
    tempo_z = {k: v for k, v in zscores.items() if config.base_of(k) in config.TEMPO_FEATURES}
    position_z = {k: v for k, v in zscores.items() if config.base_of(k) not in config.TEMPO_FEATURES}

    position_match = _zscore_similarity(position_z, weights)
    tempo_match = _zscore_similarity(tempo_z, weights)

    # Discriminator probability is kept only as a diagnostic (not a headline).
    tiger_likeness = discriminator.tiger_likeness(features) if discriminator else None

    # Rank deviations across all features by importance-weighted magnitude.
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
        position_match=round(position_match, 1) if np.isfinite(position_match) else float("nan"),
        tempo_match=round(tempo_match, 1) if np.isfinite(tempo_match) else float("nan"),
        tiger_likeness=round(tiger_likeness, 3) if tiger_likeness is not None else None,
        zscores=zscores, deviations=deviations, weights=weights,
    )
