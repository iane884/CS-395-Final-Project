"""Synthetic golf-swing generator.

Real Tiger Woods footage is copyrighted and hard to label, so for development,
testing, and a self-contained demo we synthesize biomechanically *plausible*
swings directly as `PoseSequence`s. Each swing is parameterized (shoulder/hip
turn, arm extension, knee flex, spine tilt, tempo, head movement); the feature
extractor recovers those parameters, which lets us:

* build a "Tiger" reference template from `TIGER` params + noise,
* train the Tiger-vs-amateur discriminator (Tiger vs `AMATEUR` params),
* unit-test the full pipeline without any video files.

The motion model is a deliberate simplification (orthographic front-on camera,
rigid-segment kinematics) — enough for the features to vary monotonically with
the params, not a substitute for real capture. Swap in real clips via
`pose.estimate_pose` once available; downstream code is identical.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

from . import config
from .pose import PoseSequence


@dataclass
class SwingParams:
    shoulder_turn_top: float = 90.0     # deg at top of backswing
    hip_turn_top: float = 45.0          # deg at top
    lead_elbow_top: float = 168.0       # deg (180 = straight)
    knee_flex: float = 20.0             # deg flexion (0 = straight)
    spine_tilt: float = 32.0            # deg forward from vertical
    tempo_ratio: float = 3.0            # backswing:downswing duration
    head_sway: float = 0.04             # torso-units lateral head travel
    head_bob: float = 0.04              # torso-units vertical head travel
    hip_slide: float = 0.04             # lower-body lateral slide (sway vs turn)
    n_frames: int = 90
    fps: float = 60.0
    handedness: str = "right"
    noise: float = 0.004                # world-coord jitter (meters)


# Tiger-like (big turn, large X-factor, straight lead arm, steady head, 3:1 tempo)
TIGER = SwingParams(shoulder_turn_top=95, hip_turn_top=45, lead_elbow_top=172,
                    knee_flex=22, spine_tilt=34, tempo_ratio=3.0,
                    head_sway=0.03, head_bob=0.03, hip_slide=0.03)

# Amateur-like (under-rotation, small X-factor, bent lead arm, sway, quick tempo)
AMATEUR = SwingParams(shoulder_turn_top=76, hip_turn_top=54, lead_elbow_top=148,
                      knee_flex=12, spine_tilt=24, tempo_ratio=2.0,
                      head_sway=0.13, head_bob=0.10, hip_slide=0.13)


# --------------------------------------------------------------------------- #
# Timing + profile helpers
# --------------------------------------------------------------------------- #
def _phase_anchors(tempo_ratio: float):
    """Return (p_top, p_impact) phase fractions in [0, 1] for a given tempo."""
    follow = 0.25
    down = (1.0 - follow) / (tempo_ratio + 1.0)
    p_top = 1.0 - follow - down
    p_impact = 1.0 - follow
    return p_top, p_impact


def _piecewise(p: float, anchors: List[tuple]) -> float:
    """Cosine-eased interpolation through sorted (phase, value) anchors."""
    for (p0, v0), (p1, v1) in zip(anchors, anchors[1:]):
        if p0 <= p <= p1:
            t = 0.0 if p1 == p0 else (p - p0) / (p1 - p0)
            e = 0.5 - 0.5 * np.cos(np.pi * t)   # smoothstep
            return v0 + (v1 - v0) * e
    return anchors[-1][1] if p > anchors[-1][0] else anchors[0][1]


def _rot_y(point: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a 3D point about the vertical (y) axis through the origin."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    x, y, z = point
    return np.array([x * c - z * s, y, x * s + z * c])


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def synthetic_pose(params: SwingParams = TIGER,
                   seed: Optional[int] = None) -> PoseSequence:
    rng = np.random.default_rng(seed)
    n = params.n_frames
    p_top, p_impact = _phase_anchors(params.tempo_ratio)

    # Turn profiles: ramp up to the top, unwind through impact, rotate through.
    sh_anchors = [(0.0, 0.0), (p_top, params.shoulder_turn_top),
                  (p_impact, 0.35 * params.shoulder_turn_top), (1.0, -55.0)]
    hp_anchors = [(0.0, 0.0), (p_top, params.hip_turn_top),
                  (p_impact, 0.30 * params.hip_turn_top), (1.0, -60.0)]
    # Lead-hand height: low (address) -> high (top) -> low (impact) -> high (finish)
    h_anchors = [(0.0, 0.0), (p_top, 1.0), (p_impact, -0.05), (1.0, 0.85)]

    spine = np.radians(params.spine_tilt)
    L_spine = 0.52
    hip_hw, sh_hw = 0.16, 0.20
    arm = 0.62                      # shoulder->grip reach
    leg_upper, leg_lower = 0.45, 0.46

    world = np.full((n, config.NUM_LANDMARKS, 3), np.nan)

    for t in range(n):
        p = t / (n - 1)
        th_s = _piecewise(p, sh_anchors)
        th_h = _piecewise(p, hp_anchors)
        h = _piecewise(p, h_anchors)

        # Lower-body lateral slide toward the lead side through the downswing
        # (a sway-vs-rotate signal). Lead side is -x for a right-handed golfer.
        slide_prog = _piecewise(p, [(0.0, 0.0), (p_top, 0.0), (1.0, 1.0)])
        slide = np.array([-params.hip_slide * slide_prog, 0.0, 0.0])

        # Hips (origin at hip center) + lateral slide.
        l_hip = _rot_y(np.array([-hip_hw, 0.0, 0.0]), th_h) + slide
        r_hip = _rot_y(np.array([+hip_hw, 0.0, 0.0]), th_h) + slide

        # Shoulder center sits up and forward (spine tilt), then turns with torso.
        sh_center = np.array([0.0, -L_spine * np.cos(spine), L_spine * np.sin(spine)])
        sh_center = _rot_y(sh_center, th_s)
        l_sh = sh_center + _rot_y(np.array([-sh_hw, 0.0, 0.0]), th_s)
        r_sh = sh_center + _rot_y(np.array([+sh_hw, 0.0, 0.0]), th_s)

        # Grip: hangs down+forward from the shoulders at address, lifts with h,
        # and swings around with the shoulder turn.
        grip_off = np.array([0.0, arm * np.cos(np.radians(20)) - 0.42 * h,
                             arm * np.sin(np.radians(20))])
        grip = sh_center + _rot_y(grip_off, th_s)

        lead_sh, trail_sh = (l_sh, r_sh) if params.handedness == "right" else (r_sh, l_sh)
        lead_hip, trail_hip = (l_hip, r_hip) if params.handedness == "right" else (r_hip, l_hip)

        lead_wrist = grip + np.array([0.03, 0.0, 0.0])
        trail_wrist = grip - np.array([0.03, 0.0, 0.0])

        lead_elbow = _bent_joint(lead_sh, lead_wrist, params.lead_elbow_top)
        trail_elbow = _bent_joint(trail_sh, trail_wrist, params.lead_elbow_top - 10)

        # Knees / ankles with the requested flexion.
        l_knee = l_hip + np.array([0.0, leg_upper, 0.0])
        r_knee = r_hip + np.array([0.0, leg_upper, 0.0])
        kfi = np.radians(180.0 - params.knee_flex)
        ank_dir = np.array([0.0, -np.cos(kfi), np.sin(kfi)]) * leg_lower
        l_ankle, r_ankle = l_knee + ank_dir, r_knee + ank_dir

        # Head: held near a fixed point above the address position (real golfers
        # keep the head steady while the shoulders turn under it), plus the
        # swing's own sway/bob. Decoupled from shoulder rotation so head_sway /
        # head_bob features reflect the params rather than torso turn.
        sway = params.head_sway * np.sin(np.pi * p)
        bob = params.head_bob * (1 - np.cos(2 * np.pi * p)) * 0.5
        head_base_y = -L_spine * np.cos(spine) - 0.22
        nose = np.array([sway, head_base_y + bob, 0.05])

        pts = {
            config.NOSE: nose,
            config.LEFT_SHOULDER: l_sh, config.RIGHT_SHOULDER: r_sh,
            config.LEFT_HIP: l_hip, config.RIGHT_HIP: r_hip,
            config.LEFT_KNEE: l_knee, config.RIGHT_KNEE: r_knee,
            config.LEFT_ANKLE: l_ankle, config.RIGHT_ANKLE: r_ankle,
        }
        if params.handedness == "right":
            pts.update({config.LEFT_ELBOW: lead_elbow, config.LEFT_WRIST: lead_wrist,
                        config.RIGHT_ELBOW: trail_elbow, config.RIGHT_WRIST: trail_wrist})
        else:
            pts.update({config.RIGHT_ELBOW: lead_elbow, config.RIGHT_WRIST: lead_wrist,
                        config.LEFT_ELBOW: trail_elbow, config.LEFT_WRIST: trail_wrist})
        # Fingers near the wrists so club approximation has hand landmarks.
        pts[config.LEFT_INDEX] = pts[config.LEFT_WRIST] + np.array([0.0, 0.06, 0.02])
        pts[config.RIGHT_INDEX] = pts[config.RIGHT_WRIST] + np.array([0.0, 0.06, 0.02])

        for idx, xyz in pts.items():
            world[t, idx] = xyz + rng.normal(0, params.noise, 3)

    image_xy = _project_front_on(world)
    visibility = np.where(np.isnan(world[:, :, 0]), 0.0, 0.95)
    return PoseSequence(image_xy=image_xy, world=world, visibility=visibility,
                        fps=params.fps, width=720, height=1280,
                        handedness=params.handedness)


def _bent_joint(a: np.ndarray, b: np.ndarray, target_angle_deg: float) -> np.ndarray:
    """Place a middle joint so the interior angle a-mid-b ~= target (degrees)."""
    mid = (a + b) / 2.0
    d = b - a
    half = np.linalg.norm(d) / 2.0
    if half < 1e-6:
        return mid
    half_ang = np.radians(target_angle_deg) / 2.0
    bow = half / max(np.tan(half_ang), 1e-3)
    # A stable perpendicular to d (project the +z axis out of d).
    ref = np.array([0.0, 0.0, 1.0])
    perp = ref - d * np.dot(ref, d) / np.dot(d, d)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([0.0, 1.0, 0.0])
    perp /= np.linalg.norm(perp)
    return mid + perp * bow


def _project_front_on(world: np.ndarray) -> np.ndarray:
    """Orthographic front-on camera: world (x right, y down) -> image [0, 1]."""
    image = np.full((world.shape[0], world.shape[1], 2), np.nan)
    image[:, :, 0] = 0.5 + 0.42 * world[:, :, 0]
    image[:, :, 1] = 0.42 + 0.42 * world[:, :, 1]
    return image


def reference_params(n: int = 25, seed: int = 7):
    """Tiger-like `SwingParams` with realistic swing-to-swing human variance.

    The reference template's value comes from having *spread*: if every "Tiger"
    swing were identical the per-feature std would be ~0 and any real swing would
    saturate the score. We jitter the biomechanically meaningful params (and the
    pose-noise level) to model how a pro's own swings vary clip to clip.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(replace(
            TIGER,
            shoulder_turn_top=TIGER.shoulder_turn_top + rng.normal(0, 7),
            hip_turn_top=TIGER.hip_turn_top + rng.normal(0, 5),
            lead_elbow_top=TIGER.lead_elbow_top + rng.normal(0, 6),
            knee_flex=TIGER.knee_flex + rng.normal(0, 4),
            spine_tilt=TIGER.spine_tilt + rng.normal(0, 4),
            tempo_ratio=max(2.2, TIGER.tempo_ratio + rng.normal(0, 0.35)),
            head_sway=max(0.0, TIGER.head_sway + rng.normal(0, 0.025)),
            head_bob=max(0.0, TIGER.head_bob + rng.normal(0, 0.025)),
            hip_slide=max(0.0, TIGER.hip_slide + rng.normal(0, 0.025)),
            noise=float(np.clip(rng.normal(0.006, 0.002), 0.002, 0.012)),
            n_frames=int(rng.integers(80, 115)),
        ))
    return out


def reference_swings(n: int = 25, seed: int = 7):
    """PoseSequences for building the Tiger reference template."""
    return [synthetic_pose(p, seed=500 + i)
            for i, p in enumerate(reference_params(n, seed))]


def make_dataset(n_tiger: int = 25, n_amateur: int = 30, seed: int = 0):
    """Generate labeled (PoseSequence, label) pairs for the discriminator.

    label 1 = Tiger-like, 0 = amateur. Params are jittered per sample.
    """
    rng = np.random.default_rng(seed)
    data = []

    def jitter(base: SwingParams) -> SwingParams:
        return replace(
            base,
            shoulder_turn_top=base.shoulder_turn_top + rng.normal(0, 4),
            hip_turn_top=base.hip_turn_top + rng.normal(0, 3),
            lead_elbow_top=base.lead_elbow_top + rng.normal(0, 4),
            knee_flex=base.knee_flex + rng.normal(0, 2),
            spine_tilt=base.spine_tilt + rng.normal(0, 2),
            tempo_ratio=max(1.2, base.tempo_ratio + rng.normal(0, 0.3)),
            head_sway=max(0.0, base.head_sway + rng.normal(0, 0.02)),
            head_bob=max(0.0, base.head_bob + rng.normal(0, 0.02)),
            n_frames=int(rng.integers(80, 110)),
        )

    for i in range(n_tiger):
        data.append((synthetic_pose(jitter(TIGER), seed=1000 + i), 1))
    for i in range(n_amateur):
        data.append((synthetic_pose(jitter(AMATEUR), seed=2000 + i), 0))
    return data
