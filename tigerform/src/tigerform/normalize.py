"""Scale / position normalization.

Joint angles are already scale-invariant, so normalization here exists for the
*positional* features (head sway/bob, weight shift): we express image positions
in torso-length units relative to the hip center. That makes "how far did the
head move" comparable across golfers of different size and camera distance.
"""
from __future__ import annotations

import numpy as np

from . import config
from .geometry import midpoint
from .pose import PoseSequence


def torso_scale(seq: PoseSequence) -> float:
    """Median shoulder-center-to-hip-center distance in normalized image units."""
    sh = midpoint(seq.image_xy[:, config.LEFT_SHOULDER], seq.image_xy[:, config.RIGHT_SHOULDER])
    hp = midpoint(seq.image_xy[:, config.LEFT_HIP], seq.image_xy[:, config.RIGHT_HIP])
    d = np.linalg.norm(sh - hp, axis=1)
    scale = float(np.nanmedian(d))
    return scale if np.isfinite(scale) and scale > 1e-6 else 1.0


def normalized_positions(seq: PoseSequence) -> np.ndarray:
    """(T, 33, 2) image positions re-centered on the hip center and divided by
    torso length, so distances are in torso units."""
    hp = midpoint(seq.image_xy[:, config.LEFT_HIP], seq.image_xy[:, config.RIGHT_HIP])
    return (seq.image_xy - hp[:, None, :]) / torso_scale(seq)
