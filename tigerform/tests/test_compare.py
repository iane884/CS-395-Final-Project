"""Comparison + scoring + feedback behavior."""
import numpy as np

from tigerform.compare import compare_swing
from tigerform.feedback import build_tips, generate_feedback


def test_tiger_scores_higher_than_amateur(tiger_features, amateur_features,
                                          reference, discriminator):
    tiger = compare_swing(tiger_features, reference, discriminator)
    amateur = compare_swing(amateur_features, reference, discriminator)
    assert tiger.position_match > amateur.position_match
    # Tiger should match its own reference well.
    assert tiger.position_match > 60


def test_discriminator_separates(tiger_features, amateur_features, discriminator):
    assert discriminator.tiger_likeness(tiger_features) > 0.5
    assert discriminator.tiger_likeness(amateur_features) < 0.5


def test_amateur_deviations_flag_known_faults(amateur_features, reference, discriminator):
    result = compare_swing(amateur_features, reference, discriminator)
    flagged = {d.base for d in result.deviations}
    # The amateur under-rotates and loses X-factor; at least one should surface.
    assert {"x_factor", "shoulder_turn", "lead_elbow_angle"} & flagged


def test_scores_bounded(amateur_features, reference, discriminator):
    r = compare_swing(amateur_features, reference, discriminator)
    assert 0 <= r.position_match <= 100
    assert 0 <= r.tempo_match <= 100


def test_feature_importance_normalized(discriminator):
    imp = discriminator.feature_importance()
    assert abs(sum(imp.values()) - 1.0) < 1e-6


def test_feedback_offline_template(amateur_features, reference, discriminator):
    result = compare_swing(amateur_features, reference, discriminator)
    fb = generate_feedback(result, use_claude=False)
    assert fb.generated_by == "template"
    assert fb.coaching_text
    assert len(build_tips(result)) >= 1


def test_scorecard(amateur_features, reference):
    from tigerform.scorecard import build_scorecard
    items = build_scorecard(amateur_features, reference)
    assert items, "scorecard should not be empty"
    for it in items:
        assert it.status in {"good", "minor", "off"}
        assert it.title and it.your_value and it.tiger_value and it.instruction


def test_reference_roundtrip(reference, tmp_path):
    from tigerform.reference import ReferenceTemplate
    path = reference.save(tmp_path / "ref.json")
    loaded = ReferenceTemplate.load(path)
    assert loaded.keys == reference.keys
    assert loaded.n == reference.n
