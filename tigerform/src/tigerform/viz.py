"""Visualization: annotated overlay video + comparison charts.

* `render_overlay` draws the skeleton, approximate club shaft, and event labels
  onto the source frames and writes an MP4.
* The chart helpers return Matplotlib figures (Agg backend, headless-safe) that
  the Streamlit app and CLI report embed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import config
from .club import ClubTrack
from .compare import ComparisonResult
from .events import SwingEvents
from .features import SwingFeatures
from .pose import PoseSequence
from .reference import ReferenceTemplate

_SKELETON_COLOR = (0, 220, 0)
_CLUB_COLOR = (0, 165, 255)
_JOINT_COLOR = (0, 0, 255)


def draw_frame(frame: np.ndarray, seq: PoseSequence, t: int,
               club: Optional[ClubTrack] = None, label: str = "") -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    px = seq.image_xy[t] * np.array([w, h], float)

    for a, b in config.POSE_EDGES:
        if np.all(np.isfinite(px[a])) and np.all(np.isfinite(px[b])):
            cv2.line(out, tuple(px[a].astype(int)), tuple(px[b].astype(int)),
                     _SKELETON_COLOR, 2)
    for p in px:
        if np.all(np.isfinite(p)):
            cv2.circle(out, tuple(p.astype(int)), 3, _JOINT_COLOR, -1)

    if club is not None:
        g, hd = club.grip_xy[t], club.head_xy[t]
        if np.all(np.isfinite(g)) and np.all(np.isfinite(hd)):
            gp = (g * [w, h]).astype(int)
            hp = (hd * [w, h]).astype(int)
            cv2.line(out, tuple(gp), tuple(hp), _CLUB_COLOR, 3)

    if label:
        cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out


def render_overlay(video, seq: PoseSequence, club: ClubTrack,
                   events: SwingEvents, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_to_event = {f: e for e, f in events.frames.items()}
    rgb_frames = []
    for t, frame in enumerate(video.frames):
        label = frame_to_event.get(t, "")
        label = label.replace("_", " ").title() if label else ""
        bgr = draw_frame(frame, seq, t, club, label)
        rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    _write_video(out_path, rgb_frames, video.fps)
    return out_path


def _write_video(out_path: Path, rgb_frames, fps: float) -> None:
    """Write an H.264 MP4 (yuv420p) so it plays in browsers / st.video.

    OpenCV's default 'mp4v' codec produces MPEG-4 Part 2, which HTML5 <video>
    can't decode. We prefer imageio's bundled ffmpeg (libx264); if that's
    unavailable we fall back to OpenCV mp4v (file still written, may not preview
    in-browser).
    """
    fps = max(1.0, float(fps or 30.0))
    try:
        import imageio
        writer = imageio.get_writer(
            str(out_path), fps=fps, codec="libx264", quality=8,
            pixelformat="yuv420p", macro_block_size=16, format="FFMPEG")
        try:
            for f in rgb_frames:
                writer.append_data(f)
        finally:
            writer.close()
        return
    except Exception:
        pass  # fall back to OpenCV below

    h, w = rgb_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    try:
        for f in rgb_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


# --------------------------------------------------------------------------- #
# Charts (Matplotlib figures)
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def deviation_chart(result: ComparisonResult, top_n: int = 6):
    plt = _mpl()
    devs = result.deviations[:top_n]
    fig, ax = plt.subplots(figsize=(7, 0.6 * max(len(devs), 1) + 1))
    if not devs:
        ax.text(0.5, 0.5, "No significant deviations", ha="center", va="center")
        ax.axis("off")
        return fig
    labels = [d.base.replace("_", " ") + (f"@{d.event}" if d.event else "") for d in devs]
    zs = [d.z for d in devs]
    colors = ["#d9534f" if abs(z) >= 2 else "#f0ad4e" for z in zs]
    ax.barh(range(len(devs)), zs, color=colors)
    ax.set_yticks(range(len(devs)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Deviation from Tiger (z-score)   ← lower    higher →")
    ax.set_title("Biggest mechanical differences")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def series_chart(features: SwingFeatures, reference: ReferenceTemplate,
                 name: str = "shaft_angle"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if name in features.series:
        u = np.asarray(features.series[name], float)
        ax.plot(np.linspace(0, 1, len(u)), u, label="You", color="#0275d8", lw=2)
    if name in reference.series:
        r = np.asarray(reference.series[name], float)
        ax.plot(np.linspace(0, 1, len(r)), r, label="Tiger (avg)",
                color="#5cb85c", lw=2, ls="--")
    ax.set_xlabel("Swing progress (address → finish)")
    ax.set_ylabel(name.replace("_", " "))
    ax.set_title(f"Swing trace: {name.replace('_', ' ')}")
    ax.legend()
    fig.tight_layout()
    return fig
