"""Train the Tiger-vs-amateur discriminator.

Uses real clips when available (Tiger in ``data/reference``, amateur swings in
``data/amateur``); otherwise trains on synthetic labeled swings so the project
runs out of the box. Reports cross-validated precision / recall / F1 / ROC-AUC.

    python scripts/train_discriminator.py [--synthetic]
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from tigerform import config
from tigerform.analyze import analyze_sequence
from tigerform.model import train_discriminator
from tigerform.synthetic import make_dataset


def _features_from_dir(d: Path, label: int):
    from tigerform.ingest import load_video
    from tigerform.pose import estimate_pose
    out = []
    for clip in sorted(d.glob("*")):
        if clip.suffix.lower() not in {".mp4", ".mov", ".avi", ".m4v"}:
            continue
        video = load_video(clip)
        seq = estimate_pose(video)
        out.append((analyze_sequence(seq).features, label))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    amateur_dir = config.DATA_DIR / "amateur"
    labeled = []
    if not args.synthetic:
        labeled += _features_from_dir(config.REFERENCE_DIR, 1)
        if amateur_dir.exists():
            labeled += _features_from_dir(amateur_dir, 0)

    if sum(1 for _, y in labeled if y == 1) >= 5 and sum(1 for _, y in labeled if y == 0) >= 5:
        print(f"Training on {len(labeled)} real clips.")
        feats = [f for f, _ in labeled]
        labels = [y for _, y in labeled]
    else:
        if not args.synthetic:
            print("Not enough real clips (need >=5 Tiger + >=5 amateur); "
                  "training on synthetic swings.")
        print("Generating synthetic labeled dataset...")
        dataset = make_dataset(n_tiger=25, n_amateur=30)
        feats = [analyze_sequence(seq).features for seq, _ in dataset]
        labels = [y for _, y in dataset]

    disc, metrics = train_discriminator(feats, labels)
    path = disc.save()
    print(f"\nSaved discriminator -> {path}")
    print("Cross-validated metrics:")
    print(json.dumps(metrics, indent=2))

    print("\nTop discriminating features:")
    imp = sorted(disc.feature_importance().items(), key=lambda kv: kv[1], reverse=True)
    for k, v in imp[:8]:
        print(f"  {k:28s} {v:.3f}")


if __name__ == "__main__":
    main()
