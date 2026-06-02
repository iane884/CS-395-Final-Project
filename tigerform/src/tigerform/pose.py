"""Pose estimation: turn a `VideoData` into a smoothed landmark time series.

Uses MediaPipe BlazePose (lazy-imported so synthetic `PoseSequence`s can be
built in tests without the heavy dependency). We keep two coordinate systems:

* ``image_xy`` — normalized [0, 1] image coords (x right, y down). Best for
  drawing overlays and for image-plane positional features (head sway/bob).
* ``world``    — metric 3D coords (meters), origin at the hip center. Best for
  view-robust joint angles and rotation (turn) features.

Frames where no person is detected are stored as NaN so downstream code can
treat them as gaps rather than (0, 0) landmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config
from .geometry import moving_average
from .ingest import VideoData


@dataclass
class PoseSequence:
    image_xy: np.ndarray      # (T, 33, 2) normalized image coords
    world: np.ndarray         # (T, 33, 3) metric world coords
    visibility: np.ndarray    # (T, 33) in [0, 1]
    fps: float
    width: int                # source frame width  (pixels)
    height: int               # source frame height (pixels)
    handedness: str = "right"

    @property
    def n_frames(self) -> int:
        return self.image_xy.shape[0]

    def pixels(self, t: int) -> np.ndarray:
        """Landmark pixel coords (33, 2) for frame t."""
        return self.image_xy[t] * np.array([self.width, self.height], float)

    def joint(self, idx: int, space: str = "world") -> np.ndarray:
        """Trajectory (T, D) of a single landmark across the whole sequence."""
        return (self.world[:, idx, :] if space == "world"
                else self.image_xy[:, idx, :])

    def valid_frames(self) -> np.ndarray:
        """Boolean mask (T,) of frames with a usable detection."""
        return ~np.isnan(self.image_xy[:, config.NOSE, 0])


def estimate_pose(video: VideoData, handedness: str = "right",
                  model_complexity: int = 1,
                  smooth: bool = True) -> PoseSequence:
    """Run MediaPipe Pose over every frame and return a `PoseSequence`."""
    import mediapipe as mp  # lazy: only needed when actually processing video

    n = video.n_frames
    image_xy = np.full((n, config.NUM_LANDMARKS, 2), np.nan)
    world = np.full((n, config.NUM_LANDMARKS, 3), np.nan)
    visibility = np.zeros((n, config.NUM_LANDMARKS))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        import cv2
        for t, frame in enumerate(video.frames):
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    image_xy[t, i] = (lm.x, lm.y)
                    visibility[t, i] = lm.visibility
            if res.pose_world_landmarks:
                for i, lm in enumerate(res.pose_world_landmarks.landmark):
                    world[t, i] = (lm.x, lm.y, lm.z)
    finally:
        pose.close()

    seq = PoseSequence(image_xy=image_xy, world=world, visibility=visibility,
                       fps=video.fps, width=video.width, height=video.height,
                       handedness=handedness)
    return smooth_sequence(seq) if smooth else seq


def smooth_sequence(seq: PoseSequence,
                    window: Optional[int] = None) -> PoseSequence:
    """Temporally smooth landmark positions to reduce jitter (NaN-aware)."""
    w = config.SMOOTH_WINDOW if window is None else window
    seq.image_xy = moving_average(seq.image_xy, w)
    seq.world = moving_average(seq.world, w)
    return seq
