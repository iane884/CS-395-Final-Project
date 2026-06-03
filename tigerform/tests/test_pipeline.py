"""End-to-end pipeline on an in-memory synthetic swing (no video, no artifacts on disk)."""
import numpy as np

from tigerform.ingest import from_frames
from tigerform.pipeline import run_on_sequence
from tigerform.synthetic import AMATEUR, synthetic_pose


def test_run_on_sequence_report(reference, discriminator):
    seq = synthetic_pose(AMATEUR, seed=3)
    frames = [np.full((480, 270, 3), 30, np.uint8) for _ in range(seq.n_frames)]
    seq.width, seq.height = 270, 480
    video = from_frames(frames, fps=seq.fps)

    result = run_on_sequence(seq, video=video, use_claude=False, render=True,
                             out_dir=None, reference=reference,
                             discriminator=discriminator)
    report = result.report()

    assert 0 <= report["position_match"] <= 100
    assert 0 <= report["tempo_match"] <= 100
    assert set(report["events"]) >= {"address", "top", "impact", "finish"}
    assert "coaching_text" in report["feedback"]
    assert result.overlay_path and result.overlay_path.endswith(".mp4")


def test_compare_without_discriminator(reference, amateur_features):
    # compare_swing must work (and report no Tiger-likeness) when no
    # discriminator is supplied. Tested directly so it doesn't depend on whether
    # a trained artifact happens to exist on disk (run_on_sequence auto-loads one
    # if present — that's intended app/CLI behavior).
    from tigerform.compare import compare_swing
    result = compare_swing(amateur_features, reference, discriminator=None)
    assert result.tiger_likeness is None
    assert 0 <= result.position_match <= 100
