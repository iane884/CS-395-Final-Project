"""Approximate club tracking from body landmarks.

MediaPipe does not see the club, so we approximate the shaft from the arms and
hands. The grip sits at the hands (midpoint of the wrists); the shaft direction
is taken from the lead forearm (lead elbow -> lead wrist), which points roughly
down the shaft at address and through much of the swing. This is an explicit
approximation — it gives a usable shaft angle / swing-plane proxy and a
plausible clubhead position for the overlay, not exact club geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .geometry import line_angle_from_horizontal
from .pose import PoseSequence


@dataclass
class ClubTrack:
    grip_xy: np.ndarray         # (T, 2) image coords of the hands/grip
    head_xy: np.ndarray         # (T, 2) estimated clubhead image coords
    shaft_angle: np.ndarray     # (T,)   degrees from horizontal (image plane)


def estimate_club(seq: PoseSequence) -> ClubTrack:
    lt = config.lead_trail_indices(seq.handedness)
    lead_w = seq.image_xy[:, lt["lead_wrist"], :]
    trail_w = seq.image_xy[:, lt["trail_wrist"], :]
    lead_e = seq.image_xy[:, lt["lead_elbow"], :]
    lead_s = seq.image_xy[:, lt["lead_shoulder"], :]

    grip = np.nanmean(np.stack([lead_w, trail_w], axis=0), axis=0)  # (T, 2)

    # Shaft direction ~ lead forearm direction (elbow -> wrist), extended.
    fore = lead_w - lead_e
    norm = np.linalg.norm(fore, axis=1, keepdims=True)
    unit = np.divide(fore, norm, out=np.full_like(fore, np.nan), where=norm > 1e-6)

    # Scale clubhead reach to the golfer: ~1.4x the lead-arm (shoulder->wrist).
    arm_len = np.nanmedian(np.linalg.norm(lead_w - lead_s, axis=1))
    reach = (arm_len if np.isfinite(arm_len) else 0.25) * 1.4
    head = grip + unit * reach

    angle = np.array([line_angle_from_horizontal(grip[t], head[t])
                      for t in range(seq.n_frames)])
    return ClubTrack(grip_xy=grip, head_xy=head, shaft_angle=angle)
