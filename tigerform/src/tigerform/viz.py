"""Visualization: annotated overlay video.

`render_overlay` draws the pose skeleton and event labels onto the source frames
and writes a browser-playable (H.264) MP4. The user-facing comparison is a
plain-language scorecard (see `scorecard.py`), rendered by the app.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config
from .events import SwingEvents
from .pose import PoseSequence

_SKELETON_COLOR = (0, 220, 0)
_JOINT_COLOR = (0, 0, 255)


def draw_frame(frame: np.ndarray, seq: PoseSequence, t: int,
               label: str = "") -> np.ndarray:
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

    if label:
        cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out


def render_overlay(video, seq: PoseSequence, events: SwingEvents,
                   out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_to_event = {f: e for e, f in events.frames.items()}
    rgb_frames = []
    for t, frame in enumerate(video.frames):
        label = frame_to_event.get(t, "")
        label = label.replace("_", " ").title() if label else ""
        bgr = draw_frame(frame, seq, t, label)
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
# Deviation bar chart: biggest differences from Tiger
# --------------------------------------------------------------------------- #
def _dev_label(dev) -> str:
    spec = config.SPEC_BY_BASE.get(dev.base)
    name = spec.label if spec else dev.base.replace("_", " ")
    return f"{name} ({dev.event})" if dev.event else name


def deviation_chart(comparison, top_n: int = 6):
    """Horizontal z-score bars of your biggest differences from Tiger. Bars to
    the right = you do MORE than Tiger, left = LESS; longer = further off; red =
    big difference, orange = moderate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    devs = comparison.deviations[:top_n]
    fig, ax = plt.subplots(figsize=(7.5, 0.62 * max(len(devs), 1) + 1.2))
    if not devs:
        ax.text(0.5, 0.5, "No significant differences from Tiger — nice work!",
                ha="center", va="center")
        ax.axis("off")
        return fig

    labels = [_dev_label(d) for d in devs]
    zs = [float(np.clip(d.z, -3.5, 3.5)) for d in devs]
    colors = ["#d9534f" if abs(z) >= 2 else "#f0ad4e" for z in zs]

    y = list(range(len(devs)))
    ax.barh(y, zs, color=colors, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlim(-3.6, 3.6)
    ax.set_xticks([])
    ax.set_xlabel("← you do LESS than Tiger          you do MORE than Tiger →",
                  fontsize=9)
    ax.set_title("Your biggest differences from Tiger", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    return fig
