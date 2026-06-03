"""Plain-language swing scorecard.

Turns the biomechanical comparison into golfer-readable cards: each key position
with a plain title, a one-line explanation of what it is and why it matters,
your value vs Tiger's, a status (good / slightly off / work on it), and a simple
instruction. This is what the app shows instead of an abstract chart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from . import config
from .features import SwingFeatures
from .reference import ReferenceTemplate

# key, plain title, what-it-means, advice if you do LESS than Tiger, advice if MORE.
POSITIONS = [
    ("shoulder_turn@top", "Shoulder turn at the top",
     "How far your shoulders rotate away from the ball at the top of your backswing. A bigger turn coils your body and stores power.",
     "Turn your shoulders more — feel your back face the target at the top.",
     "Shorten your shoulder turn a touch for more control."),
    ("x_factor@top", "X-Factor (shoulder–hip separation)",
     "How much more your shoulders turn than your hips at the top. This 'coil' between upper and lower body is a major power source.",
     "Keep your hips quieter while your shoulders turn fully — feel the stretch across your torso.",
     "You have lots of separation; focus on unwinding it in sequence (hips first)."),
    ("hip_turn@top", "Hip turn at the top",
     "How far your hips rotate at the top. Too much and you lose the coil with your shoulders; too little and the backswing gets restricted.",
     "Allow your hips to turn a little more to complete your backswing.",
     "Quiet your hips going back — resist with your trail leg to build more coil."),
    ("lead_elbow_angle@top", "Lead-arm extension at the top",
     "How straight your lead arm (left arm for a right-handed golfer) is at the top. A straighter lead arm widens your swing arc.",
     "Keep your lead arm straighter — but not rigid — to widen your swing.",
     "Your lead arm is very straight; keep a touch of softness to avoid tension."),
    ("lead_knee_flex@address", "Knee flex at address",
     "How much your lead knee is bent at setup. A little flex gives you an athletic, stable base to turn against.",
     "Add a bit more knee bend at setup for a more athletic base.",
     "Ease off the knee bend slightly so you're not sitting too low."),
    ("spine_tilt_forward@address", "Posture (spine tilt) at address",
     "How much you bend forward from your hips at setup. This tilt sets the plane your swing travels on.",
     "Hinge a bit more forward from your hips at address (bend, don't slouch).",
     "Stand a touch taller — you're bent over more than Tiger."),
    ("tempo_ratio", "Tempo (backswing : downswing)",
     "The ratio of how long your backswing takes versus your downswing. Tiger's is about 3 to 1 — smooth back, then a quicker move down.",
     "Smooth out your backswing so it's slower relative to your downswing (toward a 3:1 feel).",
     "Your backswing is slow relative to your downswing; even it out toward a 3:1 feel."),
]


# Short labels for the radar-chart axes (keyed by feature key).
SHORT_LABELS = {
    "shoulder_turn@top": "Shoulder\nturn",
    "x_factor@top": "X-Factor",
    "hip_turn@top": "Hip turn",
    "lead_elbow_angle@top": "Lead arm",
    "lead_knee_flex@address": "Knee flex",
    "spine_tilt_forward@address": "Posture",
    "tempo_ratio": "Tempo",
}


@dataclass
class ScoreItem:
    title: str
    description: str
    your_value: str
    tiger_value: str
    status: str            # "good" | "minor" | "off"
    instruction: str


def _fmt(base: str, v: float) -> str:
    spec = config.SPEC_BY_BASE.get(base)
    unit = spec.unit if spec else ""
    if unit == "deg":
        return f"{v:.0f}°"
    if unit == "ratio":
        return f"{v:.1f}:1"
    if unit == "s":
        return f"{v:.1f}s"
    return f"{v:.1f}"


def build_scorecard(features: SwingFeatures,
                    reference: ReferenceTemplate) -> List[ScoreItem]:
    items: List[ScoreItem] = []
    for key, title, desc, advise_more, advise_less in POSITIONS:
        base = config.base_of(key)
        v = features.scalars.get(key)
        m = reference.mean.get(key)
        if v is None or m is None or not (np.isfinite(v) and np.isfinite(m)):
            continue
        z = (v - m) / reference.std_floored(key)
        if abs(z) <= 1:
            status, instruction = "good", "Right in Tiger's range — keep it."
        else:
            status = "minor" if abs(z) <= 2 else "off"
            instruction = advise_more if z < 0 else advise_less
        items.append(ScoreItem(title=title, description=desc,
                               your_value=_fmt(base, v),
                               tiger_value="~" + _fmt(base, m),
                               status=status, instruction=instruction))
    return items
