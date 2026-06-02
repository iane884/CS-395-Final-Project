"""Video ingestion: decode a swing clip into frames + metadata.

Swing clips are short (a few seconds), so we load frames into memory once and
reuse them for pose estimation and overlay rendering. Wide videos are
downscaled to `RESIZE_WIDTH` for speed while preserving aspect ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from . import config


@dataclass
class VideoData:
    path: str
    frames: List[np.ndarray]          # BGR uint8, possibly downscaled
    fps: float
    width: int                        # post-resize frame width
    height: int                       # post-resize frame height
    orig_width: int
    orig_height: int
    warnings: List[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def _resize_keep_aspect(frame: np.ndarray, target_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    scale = target_w / float(w)
    return cv2.resize(frame, (target_w, int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def load_video(path: str | Path, resize_width: int = config.RESIZE_WIDTH,
               max_frames: int = 900) -> VideoData:
    """Decode a video file into a `VideoData` bundle.

    `max_frames` guards against accidentally loading a very long clip (default
    ~30 s at 30 fps). Raises FileNotFoundError / ValueError on bad input.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video (unsupported codec?): {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.TARGET_FPS
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: List[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(_resize_keep_aspect(frame, resize_width))
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from: {path}")

    h, w = frames[0].shape[:2]
    vid = VideoData(path=path, frames=frames, fps=float(fps), width=w, height=h,
                    orig_width=orig_w, orig_height=orig_h)
    vid.warnings = _framing_warnings(vid)
    return vid


def from_frames(frames: List[np.ndarray], fps: float = config.TARGET_FPS,
                path: str = "<memory>") -> VideoData:
    """Build a VideoData from in-memory BGR frames (used by tests/tools)."""
    h, w = frames[0].shape[:2]
    return VideoData(path=path, frames=frames, fps=float(fps), width=w, height=h,
                     orig_width=w, orig_height=h)


def _framing_warnings(vid: VideoData) -> List[str]:
    """Cheap heuristics that flag clips likely to give poor pose estimates."""
    warns: List[str] = []
    if vid.duration_s < 0.7:
        warns.append("Clip is very short (<0.7s); a full swing may be cut off.")
    if vid.duration_s > 12:
        warns.append("Clip is long (>12s); trim to a single swing for best results.")
    if vid.orig_width and vid.orig_height:
        ar = vid.orig_width / vid.orig_height
        if ar > 2.2 or ar < 0.4:
            warns.append("Unusual aspect ratio; use a standard front-on or "
                         "down-the-line framing.")
    if vid.fps < 24:
        warns.append(f"Low frame rate ({vid.fps:.0f} fps); fast motion near "
                     "impact may be blurred or skipped.")
    return warns
