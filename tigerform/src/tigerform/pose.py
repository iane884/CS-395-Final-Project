"""Pose estimation: turn a `VideoData` into a smoothed landmark time series.

Uses MediaPipe BlazePose. Newer MediaPipe (>=~0.10.20) ships only the **Tasks**
API (`mediapipe.tasks.python.vision.PoseLandmarker`), while older builds expose
the legacy `mediapipe.solutions.pose` API. We prefer Tasks and fall back to the
legacy solution, so the code works across MediaPipe versions. MediaPipe is
lazy-imported so synthetic `PoseSequence`s can be built in tests without it.

The Tasks backend needs a `.task` model asset, which we download once into
``artifacts/models/`` on first use.

Coordinate systems kept:
* ``image_xy`` — normalized [0, 1] image coords (x right, y down): overlays + 2D
  positional features (head sway/bob).
* ``world``    — metric 3D coords (meters), hip-centered: view-robust angles.

Frames with no detection are stored as NaN (treated as gaps downstream).
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import config
from .geometry import moving_average
from .ingest import VideoData

# Pose model assets (downloaded on first use of the Tasks backend).
_MODEL_DIR = config.ARTIFACTS_DIR / "models"
_MODEL_NAMES = {0: "lite", 1: "full", 2: "heavy"}
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_{name}/float16/latest/pose_landmarker_{name}.task")


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


# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #
def _tasks_available() -> bool:
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        return True
    except Exception:
        return False


def _legacy_available() -> bool:
    try:
        import mediapipe as mp
        return hasattr(mp, "solutions")
    except Exception:
        return False


def _resolve_model(complexity: int) -> str:
    """Return a local path to the pose `.task` model, downloading if needed."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    name = _MODEL_NAMES.get(complexity, "full")
    dest = _MODEL_DIR / f"pose_landmarker_{name}.task"
    if not dest.exists():
        url = _MODEL_URL.format(name=name)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"Could not download the MediaPipe pose model from {url} ({e}). "
                f"Download it manually and place it at {dest}.") from e
    return str(dest)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def estimate_pose(video: VideoData, handedness: str = "right",
                  model_complexity: int = 1,
                  smooth: bool = True) -> PoseSequence:
    """Run MediaPipe Pose over every frame and return a `PoseSequence`."""
    if _tasks_available():
        image_xy, world, visibility = _estimate_tasks(video, model_complexity)
    elif _legacy_available():
        image_xy, world, visibility = _estimate_legacy(video, model_complexity)
    else:  # pragma: no cover
        raise RuntimeError(
            "No usable MediaPipe pose backend found. Install 'mediapipe' "
            "(Tasks API) or an older 'mediapipe' with solutions support.")

    seq = PoseSequence(image_xy=image_xy, world=world, visibility=visibility,
                       fps=video.fps, width=video.width, height=video.height,
                       handedness=handedness)
    return smooth_sequence(seq) if smooth else seq


def _empty_arrays(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.full((n, config.NUM_LANDMARKS, 2), np.nan),
            np.full((n, config.NUM_LANDMARKS, 3), np.nan),
            np.zeros((n, config.NUM_LANDMARKS)))


def _estimate_tasks(video: VideoData, complexity: int):
    """Modern MediaPipe Tasks API (PoseLandmarker)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = _resolve_model(complexity)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    n = video.n_frames
    image_xy, world, visibility = _empty_arrays(n)

    landmarker = vision.PoseLandmarker.create_from_options(options)
    try:
        fps = video.fps or config.TARGET_FPS
        last_ts = -1
        for t, frame in enumerate(video.frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(last_ts + 1, int(round(t * 1000.0 / fps)))  # strictly increasing
            last_ts = ts
            res = landmarker.detect_for_video(mp_image, ts)
            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks[0]):
                    image_xy[t, i] = (lm.x, lm.y)
                    visibility[t, i] = getattr(lm, "visibility", 0.0) or 0.0
            if res.pose_world_landmarks:
                for i, lm in enumerate(res.pose_world_landmarks[0]):
                    world[t, i] = (lm.x, lm.y, lm.z)
    finally:
        landmarker.close()
    return image_xy, world, visibility


def _estimate_legacy(video: VideoData, complexity: int):
    """Legacy MediaPipe solutions API (older installs)."""
    import cv2
    import mediapipe as mp

    n = video.n_frames
    image_xy, world, visibility = _empty_arrays(n)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=complexity,
        enable_segmentation=False, min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    try:
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
    return image_xy, world, visibility


def smooth_sequence(seq: PoseSequence,
                    window: Optional[int] = None) -> PoseSequence:
    """Temporally smooth landmark positions to reduce jitter (NaN-aware)."""
    w = config.SMOOTH_WINDOW if window is None else window
    seq.image_xy = moving_average(seq.image_xy, w)
    seq.world = moving_average(seq.world, w)
    return seq
