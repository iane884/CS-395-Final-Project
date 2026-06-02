"""Biomechanical feature extraction.

Produces a flat scalar feature vector (keyed by `<base>` or `<base>@<event>` per
`config.flat_feature_keys()`) plus a few whole-swing time series used for DTW in
`compare.py`. Joint angles come from MediaPipe *world* coordinates (view-robust
3D); positional features come from torso-normalized *image* coordinates.

Conventions:
* "flex" features are flexion amounts: 0 deg = straight, larger = more bend.
* "turn" features are absolute rotation (deg) of the shoulder/hip line about the
  vertical axis relative to the address frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from . import config
from .club import ClubTrack
from .events import SwingEvents
from .geometry import angle_between, midpoint
from .normalize import normalized_positions
from .pose import PoseSequence


@dataclass
class SwingFeatures:
    scalars: Dict[str, float]
    series: Dict[str, np.ndarray] = field(default_factory=dict)
    handedness: str = "right"

    def vector(self, keys=None) -> np.ndarray:
        keys = keys or config.flat_feature_keys()
        return np.array([self.scalars.get(k, np.nan) for k in keys], float)


# --------------------------------------------------------------------------- #
# Per-frame primitives (world coordinates)
# --------------------------------------------------------------------------- #
def _w(seq: PoseSequence, idx: int, t: int) -> np.ndarray:
    return seq.world[t, idx]


def _knee_flex(seq, t, lt, side: str) -> float:
    interior = angle_between(_w(seq, lt[f"{side}_hip"], t),
                             _w(seq, lt[f"{side}_knee"], t),
                             _w(seq, lt[f"{side}_ankle"], t))
    return 180.0 - interior if np.isfinite(interior) else float("nan")


def _elbow_angle(seq, t, lt, side: str) -> float:
    return angle_between(_w(seq, lt[f"{side}_shoulder"], t),
                         _w(seq, lt[f"{side}_elbow"], t),
                         _w(seq, lt[f"{side}_wrist"], t))


def _spine_vec(seq, t) -> np.ndarray:
    sh = midpoint(_w(seq, config.LEFT_SHOULDER, t), _w(seq, config.RIGHT_SHOULDER, t))
    hp = midpoint(_w(seq, config.LEFT_HIP, t), _w(seq, config.RIGHT_HIP, t))
    return sh - hp


def _spine_tilt(seq, t) -> tuple:
    """(forward, lateral) tilt of the spine from vertical, in degrees.

    World axes: x = lateral, y = vertical (down positive), z = depth.
    forward tilt uses the depth component, lateral tilt the x component.
    """
    v = _spine_vec(seq, t)
    if np.any(np.isnan(v)) or abs(v[1]) < 1e-6:
        return float("nan"), float("nan")
    forward = float(np.degrees(np.arctan2(abs(v[2]), abs(v[1]))))
    lateral = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]))))
    return forward, lateral


def _heading_xz(p_left: np.ndarray, p_right: np.ndarray) -> float:
    """Heading (deg) of the left->right segment in the world x-z (top-down)
    plane; used to measure shoulder / hip turn."""
    d = p_right - p_left
    if np.any(np.isnan(d)) or (abs(d[0]) < 1e-9 and abs(d[2]) < 1e-9):
        return float("nan")
    return float(np.degrees(np.arctan2(d[2], d[0])))


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# Series across the swing window (used for turn deltas + DTW)
# --------------------------------------------------------------------------- #
def _turn_series(seq, left_idx, right_idx, address_frame) -> np.ndarray:
    headings = np.array([_heading_xz(_w(seq, left_idx, t), _w(seq, right_idx, t))
                         for t in range(seq.n_frames)])
    base = headings[address_frame]
    if not np.isfinite(base):
        base = np.nanmedian(headings)
    return np.array([abs(_wrap(h - base)) if np.isfinite(h) else np.nan
                     for h in headings])


def _elbow_series(seq, lt, side: str) -> np.ndarray:
    return np.array([_elbow_angle(seq, t, lt, side) for t in range(seq.n_frames)])


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract_features(seq: PoseSequence, events: SwingEvents,
                     club: ClubTrack) -> SwingFeatures:
    lt = config.lead_trail_indices(seq.handedness)
    n = seq.n_frames
    addr = events["address"]

    shoulder_turn = _turn_series(seq, lt["lead_shoulder"], lt["trail_shoulder"], addr)
    hip_turn = _turn_series(seq, lt["lead_hip"], lt["trail_hip"], addr)
    lead_elbow = _elbow_series(seq, lt, "lead")

    def at(name: str) -> int:
        return int(np.clip(events[name], 0, n - 1))

    sc: Dict[str, float] = {}

    # ---- per-event angle features ----
    for ev in config.EVENT_NAMES:
        if ev not in events.frames:
            continue
        t = at(ev)
        per_ev = {
            "lead_knee_flex": _knee_flex(seq, t, lt, "lead"),
            "trail_knee_flex": _knee_flex(seq, t, lt, "trail"),
            "lead_elbow_angle": lead_elbow[t],
            "shoulder_turn": shoulder_turn[t],
            "hip_turn": hip_turn[t],
            "club_shaft_angle": club.shaft_angle[t],
        }
        fwd, lat = _spine_tilt(seq, t)
        per_ev["spine_tilt_forward"] = fwd
        per_ev["spine_tilt_lateral"] = lat
        for base, val in per_ev.items():
            key = f"{base}@{ev}"
            if key in config.flat_feature_keys():
                sc[key] = float(val)

    # X-factor at the top = shoulder turn minus hip turn.
    st, ht = shoulder_turn[at("top")], hip_turn[at("top")]
    sc["x_factor@top"] = float(st - ht) if np.isfinite(st) and np.isfinite(ht) else float("nan")

    # ---- global features ----
    fps = seq.fps or config.TARGET_FPS
    back = max(at("top") - addr, 1)
    down = max(at("impact") - at("top"), 1)
    sc["tempo_ratio"] = float(back / down)
    sc["total_swing_time"] = float((at("finish") - addr) / fps)

    npos = normalized_positions(seq)
    lo, hi = addr, max(at("finish"), addr + 1)
    nose = npos[lo:hi, config.NOSE, :]
    base_nose = npos[addr, config.NOSE, :]
    sc["head_sway"] = float(np.nanmax(np.abs(nose[:, 0] - base_nose[0]))) if len(nose) else float("nan")
    sc["head_bob"] = float(np.nanmax(np.abs(nose[:, 1] - base_nose[1]))) if len(nose) else float("nan")

    hip_center = midpoint(npos[:, config.LEFT_HIP, :], npos[:, config.RIGHT_HIP, :])
    base_hip = hip_center[addr]
    sc["com_lateral_shift"] = float(np.nanmax(np.abs(hip_center[lo:hi, 0] - base_hip[0]))) \
        if hi > lo else float("nan")

    # ---- time series for DTW (address..finish slice) ----
    series = {
        "shaft_angle": club.shaft_angle[lo:hi],
        "lead_elbow_angle": lead_elbow[lo:hi],
        "shoulder_turn": shoulder_turn[lo:hi],
    }
    return SwingFeatures(scalars=sc, series=series, handedness=seq.handedness)
