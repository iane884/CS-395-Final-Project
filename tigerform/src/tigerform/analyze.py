"""Light orchestration: PoseSequence -> events + features.

Kept separate from `pipeline.py` so the reference builder, discriminator
trainer, and tests can extract features without importing the heavier feedback
(Anthropic) and visualization (OpenCV) modules.
"""
from __future__ import annotations

from dataclasses import dataclass

from .events import SwingEvents, segment_swing
from .features import SwingFeatures, extract_features
from .pose import PoseSequence


@dataclass
class AnalyzedSwing:
    events: SwingEvents
    features: SwingFeatures


def analyze_sequence(seq: PoseSequence) -> AnalyzedSwing:
    events = segment_swing(seq)
    features = extract_features(seq, events)
    return AnalyzedSwing(events=events, features=features)
