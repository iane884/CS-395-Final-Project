"""Fetch GolfDB metadata and print setup guidance.

GolfDB (McNally et al., CVPRW 2019, "GolfDB: A Video Database for Golf Swing
Sequencing") is distributed as an annotation file (swing clips + 8 event-frame
labels) plus a list of source YouTube videos — the raw videos are NOT
redistributed for copyright reasons. This script downloads the annotation
metadata if a URL is provided and otherwise prints where to get it.

    python scripts/download_golfdb.py [--url <golfDB.pkl url>]

TigerForm runs fully on synthetic data without GolfDB; GolfDB is the upgrade
path for (a) training a learned swing-event segmenter and (b) sourcing real
amateur/pro swings for the discriminator's negative class.
"""
import _bootstrap  # noqa: F401
import argparse
import urllib.request
from pathlib import Path

from tigerform import config

PROJECT_PAGE = "https://github.com/wmcnally/golfdb"
PAPER = "https://arxiv.org/abs/1903.06528"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Direct URL to the GolfDB annotation file (e.g. golfDB.pkl).")
    args = ap.parse_args()

    config.GOLFDB_DIR.mkdir(parents=True, exist_ok=True)

    if args.url:
        dest = config.GOLFDB_DIR / Path(args.url).name
        print(f"Downloading GolfDB metadata -> {dest}")
        urllib.request.urlretrieve(args.url, dest)
        print("Done.")
    else:
        print("No --url provided. GolfDB is optional for TigerForm.\n")
        print("To use it:")
        print(f"  1. Read the project page:  {PROJECT_PAGE}")
        print(f"     (paper: {PAPER})")
        print("  2. Download the annotation file (golfDB.pkl) per their instructions.")
        print(f"  3. Re-run with: python scripts/download_golfdb.py --url <pkl_url>")
        print(f"  4. Place any downloaded clips under: {config.GOLFDB_DIR}")
        print("\nTigerForm works without GolfDB using synthetic swings — this is")
        print("only needed for the learned-segmentation upgrade and real swing data.")


if __name__ == "__main__":
    main()
