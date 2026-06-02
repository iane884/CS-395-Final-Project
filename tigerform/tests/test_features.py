"""Feature extraction should recover the synthetic swing parameters and reflect
the known Tiger-vs-amateur differences."""
import numpy as np


def test_shoulder_turn_recovered(tiger_features):
    # TIGER shoulder_turn_top = 95 deg.
    assert abs(tiger_features.scalars["shoulder_turn@top"] - 95) < 12


def test_x_factor_tiger_gt_amateur(tiger_features, amateur_features):
    assert tiger_features.scalars["x_factor@top"] > amateur_features.scalars["x_factor@top"]
    # TIGER 95-45=50, AMATEUR 76-54=22.
    assert tiger_features.scalars["x_factor@top"] > 35
    assert amateur_features.scalars["x_factor@top"] < 35


def test_lead_arm_straighter_for_tiger(tiger_features, amateur_features):
    assert tiger_features.scalars["lead_elbow_angle@top"] > amateur_features.scalars["lead_elbow_angle@top"]


def test_tempo_recovered(tiger_features, amateur_features):
    assert abs(tiger_features.scalars["tempo_ratio"] - 3.0) < 0.6
    assert amateur_features.scalars["tempo_ratio"] < tiger_features.scalars["tempo_ratio"]


def test_knee_flex_recovered(tiger_features):
    assert abs(tiger_features.scalars["lead_knee_flex@address"] - 22) < 10


def test_spine_tilt_recovered(tiger_features):
    assert abs(tiger_features.scalars["spine_tilt_forward@address"] - 34) < 10


def test_head_steadier_for_tiger(tiger_features, amateur_features):
    assert tiger_features.scalars["head_sway"] < amateur_features.scalars["head_sway"]


def test_all_feature_keys_present(tiger_features):
    from tigerform import config
    vec = tiger_features.vector()
    assert len(vec) == len(config.flat_feature_keys())
    # Most features should be finite for a clean synthetic swing.
    assert np.isfinite(vec).mean() > 0.8
