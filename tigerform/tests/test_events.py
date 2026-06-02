"""Swing-event segmentation sanity checks on synthetic swings."""
from tigerform.analyze import analyze_sequence
from tigerform.synthetic import TIGER, synthetic_pose


def test_event_ordering(tiger_swing):
    ev = analyze_sequence(tiger_swing).events.frames
    assert ev["address"] < ev["top"] < ev["impact"] < ev["finish"]


def test_top_near_expected():
    seq = synthetic_pose(TIGER, seed=1)
    analyzed = analyze_sequence(seq)
    n = seq.n_frames
    # TIGER tempo 3:1 -> top at ~0.5625 of the swing (see synthetic._phase_anchors).
    expected_top = 0.5625 * (n - 1)
    assert abs(analyzed.events.frames["top"] - expected_top) < 0.1 * n


def test_segmentation_confident(tiger_swing):
    assert analyze_sequence(tiger_swing).events.confidence >= 0.6
