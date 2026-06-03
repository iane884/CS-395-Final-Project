"""Swing-event segmentation.

MVP heuristic: the lead hand traces a low -> high (top) -> low (impact) ->
high (finish) height profile through the swing. We locate the two height peaks
(top, finish) and the troughs (address, impact) from the lead-wrist vertical
trajectory, then interpolate the four intermediate GolfDB events.

This is intentionally simple and robust; the plan's optional upgrade is a
learned SwingNet model trained on GolfDB, which would replace `segment_swing`
while keeping the same `SwingEvents` output contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from . import config
from .pose import PoseSequence


@dataclass
class SwingEvents:
    frames: Dict[str, int]            # event name -> frame index
    confidence: float                 # 0..1 heuristic confidence

    def __getitem__(self, name: str) -> int:
        return self.frames[name]

    def ordered(self):
        return [(e, self.frames[e]) for e in config.EVENT_NAMES if e in self.frames]


def _interp_nan(y: np.ndarray) -> np.ndarray:
    """Linear-fill NaNs in a 1D series so peak finding is stable."""
    y = y.astype(float).copy()
    idx = np.arange(len(y))
    good = ~np.isnan(y)
    if good.sum() < 2:
        return np.nan_to_num(y, nan=0.0)
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


def segment_swing(seq: PoseSequence) -> SwingEvents:
    n = seq.n_frames
    lt = config.lead_trail_indices(seq.handedness)
    # Height of the lead hand: image y grows downward, so height = -y.
    y = _interp_nan(seq.image_xy[:, lt["lead_wrist"], 1])
    height = -y
    rng = float(np.ptp(height)) or 1.0

    # Top of backswing = highest the lead hand gets in the first ~70% of the
    # clip (the follow-through can also be high, so we exclude the tail).
    search_hi = max(2, int(0.7 * n))
    top = int(np.clip(np.argmax(height[:search_hi]), 1, n - 2))

    # Address = lowest hand point before the top (hands start low and rise).
    address = int(np.argmin(height[: top + 1])) if top > 0 else 0

    # Impact = lowest hand point after the top (hands return to the ball).
    impact = top + int(np.argmin(height[top:]))
    impact = min(max(impact, top + 1), n - 1)

    # Finish = highest hand point after impact (or the last frame).
    finish = impact + int(np.argmax(height[impact:])) if impact < n - 1 else n - 1
    finish = min(max(finish, impact + 1), n - 1)

    frames = {
        "address": address,
        "top": top,
        "impact": min(impact, n - 1),
        "finish": min(finish, n - 1),
    }
    # Interpolate intermediate events between the anchors.
    frames["toe_up"] = _between(address, top, 0.4)
    frames["mid_backswing"] = _between(address, top, 0.7)
    frames["mid_downswing"] = _between(top, frames["impact"], 0.6)
    frames["mid_follow_through"] = _between(frames["impact"], finish, 0.5)

    frames = {k: int(np.clip(v, 0, n - 1)) for k, v in frames.items()}
    conf = _confidence(height, frames, rng)
    return SwingEvents(frames=frames, confidence=conf)


def _between(a: int, b: int, frac: float) -> int:
    return int(round(a + (b - a) * frac))


def _confidence(height: np.ndarray, frames: Dict[str, int], rng: float) -> float:
    """Higher when the address/top/impact heights show the expected ordering."""
    try:
        a, t, i = height[frames["address"]], height[frames["top"]], height[frames["impact"]]
    except Exception:
        return 0.3
    score = 0.0
    if t > a:           # top above address
        score += 0.4
    if t > i:           # top above impact
        score += 0.4
    if abs(a - i) < 0.5 * rng:  # impact returns near address height
        score += 0.2
    return round(score, 2)
