"""Download and (optionally) trim a swing clip into a TigerForm data folder.

You supply the URL of a clip you have the right to use (educational fair use for
this course project — TigerForm stores only derived features, never the video).
The script uses yt-dlp; install it first:  pip install yt-dlp

Examples
--------
# A real Tiger swing into the reference set (trim to a single swing):
python scripts/fetch_clip.py "https://youtu.be/XXXX" --to reference --start 0:04 --end 0:09

# One of your own swings into the amateur (negative) class:
python scripts/fetch_clip.py "https://youtu.be/YYYY" --to amateur --name my_swing

Then rebuild: python scripts/build_reference.py && python scripts/train_discriminator.py
"""
import _bootstrap  # noqa: F401
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from tigerform import config

TARGETS = {
    "reference": config.REFERENCE_DIR,                 # real Tiger / pro swings
    "amateur": config.DATA_DIR / "amateur",            # negative class
    "raw": config.RAW_DIR,                             # just to analyze
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="Video URL (a clip you have the right to use).")
    ap.add_argument("--to", choices=list(TARGETS), default="reference")
    ap.add_argument("--start", help="Trim start, e.g. 0:04 (use with --end).")
    ap.add_argument("--end", help="Trim end, e.g. 0:09.")
    ap.add_argument("--name", help="Output filename stem.")
    args = ap.parse_args()

    if importlib.util.find_spec("yt_dlp") is None:
        print("yt-dlp not found. Install it first:  pip install yt-dlp")
        return 1

    out_dir = TARGETS[args.to]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or "%(title).50s-%(id)s"
    cmd = [sys.executable, "-m", "yt_dlp", "-f", "mp4", "--no-playlist",
           "-o", str(out_dir / (stem + ".%(ext)s"))]

    # Use the ffmpeg bundled with imageio-ffmpeg if present (for trimming/merge).
    try:
        import imageio_ffmpeg
        cmd += ["--ffmpeg-location", imageio_ffmpeg.get_ffmpeg_exe()]
    except Exception:
        pass

    if args.start and args.end:
        cmd += ["--download-sections", f"*{args.start}-{args.end}",
                "--force-keyframes-at-cuts"]
    cmd.append(args.url)

    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc == 0:
        print(f"\n✓ Saved into {out_dir}")
        print("  Next: python scripts/build_reference.py  (and/or train_discriminator.py)")
    else:
        print("\n✗ Download failed. Check the URL and that you have rights to use it.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
