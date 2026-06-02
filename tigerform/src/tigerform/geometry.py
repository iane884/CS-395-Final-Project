"""Small, dependency-light geometry helpers shared across feature extraction.

All functions operate on NumPy arrays of 2D or 3D points and are written to be
robust to degenerate inputs (zero-length vectors return NaN rather than raising)
so a few bad frames don't crash the pipeline.
"""
from __future__ import annotations

import warnings

import numpy as np


def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.full_like(v, np.nan), where=n > 1e-9)


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at vertex `b` formed by points a-b-c, in degrees.

    Works for 2D or 3D points. Returns NaN if either segment is degenerate.
    """
    ba = _safe_unit(np.asarray(a, float) - np.asarray(b, float))
    bc = _safe_unit(np.asarray(c, float) - np.asarray(b, float))
    if np.any(np.isnan(ba)) or np.any(np.isnan(bc)):
        return float("nan")
    cos = float(np.clip(np.dot(ba, bc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def line_angle_from_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
    """Angle (deg) of segment p1->p2 away from the vertical (image y) axis.

    0 deg means perfectly vertical. Uses the 2D (x, y) components only. Note
    image y grows downward, which does not affect the magnitude.
    """
    d = np.asarray(p2, float)[:2] - np.asarray(p1, float)[:2]
    if np.linalg.norm(d) < 1e-9:
        return float("nan")
    # angle from vertical = atan2(|dx|, |dy|)
    return float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]))))


def line_angle_from_horizontal(p1: np.ndarray, p2: np.ndarray) -> float:
    """Angle (deg) of segment p1->p2 away from the horizontal (image x) axis."""
    d = np.asarray(p2, float)[:2] - np.asarray(p1, float)[:2]
    if np.linalg.norm(d) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))


def signed_rotation_2d(ref_vec: np.ndarray, vec: np.ndarray) -> float:
    """Signed rotation (deg, -180..180) from `ref_vec` to `vec` in the xy plane.

    Used to measure shoulder/hip turn relative to the address frame.
    """
    a = np.asarray(ref_vec, float)[:2]
    b = np.asarray(vec, float)[:2]
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return float("nan")
    ang = np.degrees(np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0]))
    return float((ang + 180.0) % 360.0 - 180.0)


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, float) + np.asarray(b, float)) / 2.0


def moving_average(series: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along axis 0, preserving length and NaNs-as-gaps.

    `series` may be (T,) or (T, ...). Edges use shrinking windows.
    """
    arr = np.asarray(series, float)
    if window <= 1 or arr.shape[0] <= 2:
        return arr
    out = np.empty_like(arr)
    half = window // 2
    with warnings.catch_warnings():
        # All-NaN windows (frames with no detection) are expected; the result
        # stays NaN and is treated as a gap downstream.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(arr.shape[0]):
            lo, hi = max(0, i - half), min(arr.shape[0], i + half + 1)
            with np.errstate(invalid="ignore"):
                out[i] = np.nanmean(arr[lo:hi], axis=0)
    return out
