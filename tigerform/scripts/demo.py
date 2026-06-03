"""Run the full TigerForm analysis on a synthetic swing (no video needed).

Handy for verifying the end-to-end pipeline and producing a sample overlay +
report without any footage or MediaPipe.

    python scripts/demo.py [--kind amateur|tiger] [--no-claude]
"""
import _bootstrap  # noqa: F401
import argparse
import numpy as np

from tigerform import config
from tigerform.ingest import from_frames
from tigerform.pipeline import run_on_sequence
from tigerform.synthetic import TIGER, AMATEUR, synthetic_pose


def _blank_video(seq, w=540, h=960):
    frames = [np.full((h, w, 3), 30, np.uint8) for _ in range(seq.n_frames)]
    seq.width, seq.height = w, h
    return from_frames(frames, fps=seq.fps, path="<synthetic>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["amateur", "tiger"], default="amateur")
    ap.add_argument("--no-claude", action="store_true")
    args = ap.parse_args()

    params = TIGER if args.kind == "tiger" else AMATEUR
    seq = synthetic_pose(params, seed=123)
    video = _blank_video(seq)

    out_dir = config.ARTIFACTS_DIR / "demo"
    result = run_on_sequence(seq, video=video, use_claude=not args.no_claude,
                             render=True, out_dir=out_dir)
    report_path = result.save_report(out_dir / "report.json")

    print(f"\n=== TigerForm demo ({args.kind}-like synthetic swing) ===")
    print(result.feedback.headline)
    print(f"Position match: {result.comparison.position_match}  "
          f"Tempo match: {result.comparison.tempo_match}")
    print(f"\n{result.feedback.coaching_text}\n")
    print(f"Overlay: {result.overlay_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
