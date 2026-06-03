"""Build the Tiger reference template.

Uses real Tiger swing clips in ``data/reference/*.mp4`` when present; otherwise
falls back to synthetic Tiger-like swings so the project is runnable out of the
box. Only derived features are stored (never the footage).

    python scripts/build_reference.py [--synthetic] [--n 25]
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from tigerform import config
from tigerform.analyze import analyze_sequence
from tigerform.reference import build_reference
from tigerform.synthetic import reference_swings


def _features_from_clips(clip_dir: Path):
    from tigerform.ingest import load_video
    from tigerform.pose import estimate_pose
    feats = []
    clips = sorted(p for p in clip_dir.glob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".m4v"})
    for clip in clips:
        print(f"  • {clip.name}")
        video = load_video(clip)
        seq = estimate_pose(video)
        feats.append(analyze_sequence(seq).features)
    return feats, len(clips)


def _features_synthetic(n: int):
    return [analyze_sequence(seq).features for seq in reference_swings(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="Force synthetic data.")
    ap.add_argument("--n", type=int, default=25, help="Synthetic swing count.")
    args = ap.parse_args()

    real, n_clips = ([], 0)
    if not args.synthetic:
        real, n_clips = _features_from_clips(config.REFERENCE_DIR)

    if n_clips >= 3:
        print(f"Building reference from {n_clips} real Tiger clips.")
        feats = real
    else:
        if not args.synthetic:
            print(f"Found {n_clips} clips in {config.REFERENCE_DIR} (need >=3); "
                  "using synthetic Tiger swings instead.")
        print(f"Building reference from {args.n} synthetic Tiger swings.")
        feats = _features_synthetic(args.n)

    template = build_reference(feats)
    path = template.save()
    print(f"\nSaved reference template ({template.n} swings) -> {path}")
    print(f"Features captured: {len(template.keys)}")


if __name__ == "__main__":
    main()
