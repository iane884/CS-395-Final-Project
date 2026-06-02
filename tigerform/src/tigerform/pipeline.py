"""End-to-end TigerForm pipeline for a single swing.

ingest -> pose -> (club, events, features) -> compare -> feedback -> viz.

Usable two ways:
* `run_pipeline(video_path)` for a real video file, and
* `run_on_sequence(seq)` for an already-extracted/synthetic `PoseSequence`
  (used by the demo and tests, no video decoding required).

Also exposes a CLI: ``python -m tigerform.pipeline path/to/swing.mp4``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import config
from .analyze import analyze_sequence
from .compare import ComparisonResult, compare_swing
from .events import SwingEvents
from .feedback import FeedbackResult, generate_feedback
from .features import SwingFeatures
from .model import TigerDiscriminator
from .pose import PoseSequence
from .reference import ReferenceTemplate


@dataclass
class PipelineResult:
    events: SwingEvents
    features: SwingFeatures
    comparison: ComparisonResult
    feedback: FeedbackResult
    overlay_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def report(self) -> dict:
        c = self.comparison
        return {
            "similarity_score": c.similarity_score,
            "sequence_similarity": c.sequence_similarity,
            "tiger_likeness": c.tiger_likeness,
            "events": self.events.frames,
            "event_confidence": self.events.confidence,
            "features": self.features.scalars,
            "top_deviations": [
                {"feature": d.base, "event": d.event, "your_value": round(d.value, 2),
                 "tiger_mean": round(d.tiger_mean, 2), "z": round(d.z, 2),
                 "direction": d.direction} for d in c.deviations[:config.TOP_N_DEVIATIONS]
            ],
            "feedback": {
                "headline": self.feedback.headline,
                "coaching_text": self.feedback.coaching_text,
                "generated_by": self.feedback.generated_by,
                "tips": [t.__dict__ for t in self.feedback.tips],
            },
            "overlay_path": self.overlay_path,
            "warnings": self.warnings,
        }

    def save_report(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2))
        return path


def _load_artifacts(reference: Optional[ReferenceTemplate],
                    discriminator: Optional[TigerDiscriminator]):
    ref = reference or ReferenceTemplate.load()
    disc = discriminator
    if disc is None and config.DISCRIMINATOR_PATH.exists():
        disc = TigerDiscriminator.load()
    return ref, disc


def run_on_sequence(seq: PoseSequence, video=None, use_claude: bool = True,
                    render: bool = False, out_dir: Optional[Path] = None,
                    reference: Optional[ReferenceTemplate] = None,
                    discriminator: Optional[TigerDiscriminator] = None,
                    warnings: Optional[List[str]] = None) -> PipelineResult:
    ref, disc = _load_artifacts(reference, discriminator)
    analyzed = analyze_sequence(seq)
    comparison = compare_swing(analyzed.features, ref, disc)
    feedback = generate_feedback(comparison, use_claude=use_claude)

    overlay_path = None
    if render and video is not None:
        from .viz import render_overlay
        out_dir = Path(out_dir or config.ARTIFACTS_DIR)
        overlay_path = str(render_overlay(
            video, seq, analyzed.club, analyzed.events,
            out_dir / "overlay.mp4"))

    return PipelineResult(events=analyzed.events, features=analyzed.features,
                          comparison=comparison, feedback=feedback,
                          overlay_path=overlay_path,
                          warnings=warnings or [])


def run_pipeline(video_path: str | Path, handedness: str = "right",
                 use_claude: bool = True, render: bool = True,
                 out_dir: Optional[Path] = None,
                 reference: Optional[ReferenceTemplate] = None,
                 discriminator: Optional[TigerDiscriminator] = None) -> PipelineResult:
    from .ingest import load_video
    from .pose import estimate_pose

    video = load_video(video_path)
    seq = estimate_pose(video, handedness=handedness)
    return run_on_sequence(seq, video=video, use_claude=use_claude, render=render,
                           out_dir=out_dir, reference=reference,
                           discriminator=discriminator, warnings=video.warnings)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze a golf swing vs. Tiger Woods.")
    ap.add_argument("video", help="Path to a swing video (front-on or down-the-line).")
    ap.add_argument("--handedness", choices=["right", "left"], default="right")
    ap.add_argument("--no-claude", action="store_true", help="Skip Claude phrasing.")
    ap.add_argument("--no-render", action="store_true", help="Skip overlay video.")
    ap.add_argument("--out-dir", default=str(config.ARTIFACTS_DIR))
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    result = run_pipeline(args.video, handedness=args.handedness,
                          use_claude=not args.no_claude, render=not args.no_render,
                          out_dir=out_dir)
    report_path = result.save_report(out_dir / "report.json")

    print(f"\n=== TigerForm ===")
    print(result.feedback.headline)
    if result.comparison.tiger_likeness is not None:
        print(f"Tiger-likeness (discriminator): {result.comparison.tiger_likeness:.2f}")
    print(f"\n{result.feedback.coaching_text}\n")
    if result.overlay_path:
        print(f"Overlay video: {result.overlay_path}")
    print(f"Report JSON:   {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
