"""Light orchestration: PoseSequence -> club + events + features.

Kept separate from `pipeline.py` so the reference builder, discriminator
trainer, and tests can extract features without importing the heavier feedback
(Anthropic) and visualization (matplotlib) modules.
"""
from __future__ import annotations

from dataclasses import dataclass

from .club import ClubTrack, estimate_club
from .events import SwingEvents, segment_swing
from .features import SwingFeatures, extract_features
from .pose import PoseSequence


@dataclass
class AnalyzedSwing:
    club: ClubTrack
    events: SwingEvents
    features: SwingFeatures


def analyze_sequence(seq: PoseSequence) -> AnalyzedSwing:
    club = estimate_club(seq)
    events = segment_swing(seq, club)
    features = extract_features(seq, events, club)
    return AnalyzedSwing(club=club, events=events, features=features)
