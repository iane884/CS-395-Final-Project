"""Central configuration for TigerForm.

Single source of truth for filesystem paths, MediaPipe landmark indices, the
canonical swing events, and the biomechanical feature registry. Other modules
import from here so feature names stay consistent across extraction, the
reference template, the discriminator, comparison, and feedback.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# config.py lives at <root>/src/tigerform/config.py -> parents[2] == <root>.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
GOLFDB_DIR = DATA_DIR / "golfdb"
ARTIFACTS_DIR = ROOT / "artifacts"

REFERENCE_TEMPLATE_PATH = ARTIFACTS_DIR / "reference_template.json"
DISCRIMINATOR_PATH = ARTIFACTS_DIR / "discriminator.joblib"

for _d in (RAW_DIR, REFERENCE_DIR, GOLFDB_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Load <root>/.env (e.g. ANTHROPIC_API_KEY) so the key set there is available
# to os.environ everywhere. Existing environment variables take precedence.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Swing events (GolfDB ordering)
# --------------------------------------------------------------------------- #
EVENT_NAMES: List[str] = [
    "address",
    "toe_up",
    "mid_backswing",
    "top",
    "mid_downswing",
    "impact",
    "mid_follow_through",
    "finish",
]
# Events we actually evaluate biomechanics at (others are mostly for timing/viz).
KEY_EVENTS: List[str] = ["address", "top", "impact", "finish"]

# --------------------------------------------------------------------------- #
# MediaPipe BlazePose (33 landmarks) indices
# --------------------------------------------------------------------------- #
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
NUM_LANDMARKS = 33

# Skeleton edges for drawing overlays (subset that matters for a golf swing).
POSE_EDGES = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]

# --------------------------------------------------------------------------- #
# Handedness: a right-handed golfer's LEAD side is the left side of the body.
# --------------------------------------------------------------------------- #
def lead_trail_indices(handedness: str = "right") -> Dict[str, int]:
    """Return landmark indices for the lead/trail shoulder, elbow, wrist, hip,
    knee, ankle given golfer handedness."""
    left = dict(shoulder=LEFT_SHOULDER, elbow=LEFT_ELBOW, wrist=LEFT_WRIST,
                hip=LEFT_HIP, knee=LEFT_KNEE, ankle=LEFT_ANKLE)
    right = dict(shoulder=RIGHT_SHOULDER, elbow=RIGHT_ELBOW, wrist=RIGHT_WRIST,
                 hip=RIGHT_HIP, knee=RIGHT_KNEE, ankle=RIGHT_ANKLE)
    if handedness == "right":          # lead = left side
        return {f"lead_{k}": v for k, v in left.items()} | \
               {f"trail_{k}": v for k, v in right.items()}
    return {f"lead_{k}": v for k, v in right.items()} | \
           {f"trail_{k}": v for k, v in left.items()}

# --------------------------------------------------------------------------- #
# Processing constants
# --------------------------------------------------------------------------- #
TARGET_FPS = 30.0          # frames are resampled toward this when computing time
SMOOTH_WINDOW = 5          # moving-average window (frames) for landmark smoothing
MIN_VISIBILITY = 0.3       # landmarks below this are treated as unreliable
RESIZE_WIDTH = 720         # downscale wide videos for speed (keeps aspect ratio)

# --------------------------------------------------------------------------- #
# Feature registry
# --------------------------------------------------------------------------- #
# A "base" feature is a semantic biomechanical quantity. Most are evaluated at
# one or more key events; a few are global (whole-swing). features.py emits a
# flat dict keyed by `<base>` (global) or `<base>@<event>` (per-event). The
# coaching rule table in feedback.py is keyed by `base`.

@dataclass(frozen=True)
class FeatureSpec:
    base: str
    label: str
    unit: str
    events: List[str]            # [] => global feature, else per-event
    description: str

PER_EVENT_FEATURES: List[FeatureSpec] = [
    FeatureSpec("lead_knee_flex", "Lead knee flex", "deg", ["address", "top", "impact"],
                "Bend in the lead knee; reflects athletic posture and stability."),
    FeatureSpec("trail_knee_flex", "Trail knee flex", "deg", ["address", "top"],
                "Bend in the trail knee; should stay loaded during the backswing."),
    FeatureSpec("spine_tilt_forward", "Spine tilt (forward)", "deg", ["address", "top", "impact"],
                "Forward bend of the spine from vertical; the swing's posture angle."),
    FeatureSpec("spine_tilt_lateral", "Spine tilt (lateral)", "deg", ["top", "impact"],
                "Side bend of the spine; secondary axis tilt away from target at impact."),
    FeatureSpec("lead_elbow_angle", "Lead arm extension", "deg", ["top", "impact"],
                "Lead-arm straightness (180 deg = fully straight) at the top and impact."),
    FeatureSpec("shoulder_turn", "Shoulder turn", "deg", ["top", "impact"],
                "Rotation of the shoulder line relative to address."),
    FeatureSpec("hip_turn", "Hip turn", "deg", ["top", "impact"],
                "Rotation of the hip line relative to address."),
    FeatureSpec("x_factor", "X-Factor (shoulder-hip separation)", "deg", ["top"],
                "Shoulder turn minus hip turn at the top; a key power source."),
    FeatureSpec("club_shaft_angle", "Club shaft angle", "deg", ["address", "top", "impact"],
                "Approximate shaft angle from horizontal; swing-plane proxy."),
]

GLOBAL_FEATURES: List[FeatureSpec] = [
    FeatureSpec("tempo_ratio", "Tempo ratio (back:down)", "ratio", [],
                "Backswing duration divided by downswing duration; Tiger ~ 3:1."),
    FeatureSpec("total_swing_time", "Total swing time", "s", [],
                "Seconds from address to finish."),
    FeatureSpec("head_sway", "Head sway", "norm", [],
                "Peak lateral head movement (torso-normalized); lower is steadier."),
    FeatureSpec("head_bob", "Head bob", "norm", [],
                "Peak vertical head movement (torso-normalized); lower is steadier."),
    FeatureSpec("com_lateral_shift", "Weight shift", "norm", [],
                "Lateral shift of body center of mass; a weight-transfer proxy."),
]

ALL_SPECS: List[FeatureSpec] = PER_EVENT_FEATURES + GLOBAL_FEATURES
SPEC_BY_BASE: Dict[str, FeatureSpec] = {s.base: s for s in ALL_SPECS}


def flat_feature_keys() -> List[str]:
    """Canonical, ordered list of flat feature-vector keys."""
    keys: List[str] = []
    for spec in PER_EVENT_FEATURES:
        for ev in spec.events:
            keys.append(f"{spec.base}@{ev}")
    for spec in GLOBAL_FEATURES:
        keys.append(spec.base)
    return keys


def base_of(flat_key: str) -> str:
    """`x_factor@top` -> `x_factor`; `tempo_ratio` -> `tempo_ratio`."""
    return flat_key.split("@", 1)[0]


def event_of(flat_key: str) -> Optional[str]:
    """`x_factor@top` -> `top`; `tempo_ratio` -> None."""
    return flat_key.split("@", 1)[1] if "@" in flat_key else None


# --------------------------------------------------------------------------- #
# Feedback / model knobs
# --------------------------------------------------------------------------- #
FEEDBACK_MODEL = os.environ.get("TIGERFORM_FEEDBACK_MODEL", "claude-opus-4-8")
TOP_N_DEVIATIONS = 4          # how many deviations to surface in feedback
ZSCORE_FLAG_THRESHOLD = 1.0   # |z| above this counts as a meaningful deviation
