"""Corrective feedback generation.

Two layers:
1. A deterministic rule table maps each biomechanical deviation (feature +
   direction vs. Tiger) to a concrete coaching message and drill.
2. The structured tips are optionally handed to the Claude API, which phrases
   them into a short, encouraging coaching note. The system prompt + rules are
   prompt-cached. If no ANTHROPIC_API_KEY is set (or the SDK/call fails), a
   deterministic template note is used instead, so the app always works offline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from . import config
from .compare import ComparisonResult, Deviation

# --------------------------------------------------------------------------- #
# Coaching rule table: base feature -> messages for being high/low vs. Tiger.
# --------------------------------------------------------------------------- #
RULES: Dict[str, Dict[str, str]] = {
    "shoulder_turn": {
        "low": "You're under-rotating your shoulders. A fuller turn stores more power, like Tiger's deep backswing.",
        "high": "You're over-rotating your shoulders, which can get the club across the line. Feel a slightly shorter, more controlled turn.",
        "drill": "Shoulder-turn drill: lay a club across your chest and turn back until the grip end points behind the ball — that's a full, Tiger-like shoulder turn.",
    },
    "hip_turn": {
        "low": "Your hips are too quiet in the backswing. Allow a little more hip turn to load the trail side.",
        "high": "Your hips are spinning too early/much, which leaks the X-factor. Keep the lower body more stable going back.",
        "drill": "Trail-foot-back drill: pull your trail foot a few inches off the line at setup, which restricts the hips so you feel them resist while your shoulders turn fully.",
    },
    "x_factor": {
        "low": "Your shoulder-hip separation (X-factor) at the top is smaller than Tiger's. More separation = more stored power.",
        "high": "Your X-factor is very large; great for power but make sure you can sequence it without losing balance.",
        "drill": "Pause-at-the-top drill: swing to the top and freeze — feel your back to the target (shoulders fully turned) while your belt buckle stays quieter (hips less turned).",
    },
    "lead_elbow_angle": {
        "low": "Your lead arm is more bent at the top than Tiger's. A straighter (not rigid) lead arm widens your arc.",
        "high": "Your lead arm is locked very straight; keep some softness to avoid tension.",
        "drill": "Towel drill: tuck a towel under your lead armpit and keep it pinned into the backswing — it trains a connected, extended (not rigid) lead arm.",
    },
    "lead_knee_flex": {
        "low": "Your lead knee is fairly straight; add a little flex for a more athletic, stable base.",
        "high": "Lots of lead-knee flex; make sure you're not sinking or losing posture.",
        "drill": "Chair drill: set up with your backside just touching the edge of a chair and keep light contact during the backswing, so you hold your knee flex and posture.",
    },
    "trail_knee_flex": {
        "low": "Your trail knee straightens in the backswing; keep it flexed to stay loaded.",
        "high": "Trail knee is very bent; keep stability so you don't sway off the ball.",
        "drill": "Alignment-stick drill: stand a stick just outside your trail knee and keep light pressure on it going back, so the knee stays flexed instead of straightening.",
    },
    "spine_tilt_forward": {
        "low": "You're standing a bit tall; add forward spine tilt from the hips for a better swing plane.",
        "high": "You're bent over more than Tiger; ease the forward tilt to free up rotation.",
        "drill": "Posture drill: hold a club vertically against your spine (touching head, mid-back, tailbone), then tilt forward from your hip joints — not by rounding your back.",
    },
    "spine_tilt_lateral": {
        "low": "Add a touch of secondary (away-from-target) tilt at impact to help shallow the club.",
        "high": "Too much lateral tilt can cause early extension; keep the tilt moderate.",
        "drill": "Impact-bag drill: push a padded bag (or stack of towels) at impact while keeping your spine tilted slightly away from the target and stable.",
    },
    "tempo_ratio": {
        "low": "Your transition is quick (downswing too fast relative to backswing). Tiger's tempo is near 3:1.",
        "high": "Your backswing is slow relative to the downswing; smooth it toward a 3:1 feel.",
        "drill": "Counting drill: count '1-2-3' on the way back and a quick '1' coming down — this grooves the roughly 3:1 backswing-to-downswing rhythm.",
    },
    "total_swing_time": {
        "low": "Your overall swing is quick; make sure you complete the backswing.",
        "high": "Your swing is slow overall; a touch more pace can improve sequencing.",
        "drill": "Metronome drill: set a metronome and match your backswing and downswing to its beats so your overall timing repeats.",
    },
    "head_sway": {
        "low": "Nice steady head laterally.",
        "high": "Your head sways off the ball. Keep it centered like Tiger for a consistent low point.",
        "drill": "Wall drill: rest your head lightly against a wall (or a partner's hand) and make slow swings without losing contact, so your head stops sliding off the ball.",
    },
    "head_bob": {
        "low": "Good vertical head stability.",
        "high": "Your head moves up/down during the swing; maintain your spine height for solid contact.",
        "drill": "Background drill: pick a fixed spot behind the ball and keep your head level with it through practice swings, so you don't rise up or dip down.",
    },
    "com_lateral_shift": {
        "low": "Limited weight shift; allow some pressure move to the lead side through impact.",
        "high": "You're sliding laterally rather than rotating; turn into the lead side instead of sway.",
        "drill": "Step-through drill: after impact, let your trail foot step toward the target — this trains rotating through the ball instead of sliding sideways.",
    },
}

EVENT_LABELS = {
    "address": "at address", "toe_up": "during takeaway",
    "mid_backswing": "mid-backswing", "top": "at the top",
    "mid_downswing": "in transition", "impact": "at impact",
    "mid_follow_through": "post-impact", "finish": "at the finish",
}


@dataclass
class Tip:
    title: str
    message: str
    drill: str
    detail: str          # numeric context (your value vs Tiger)


@dataclass
class FeedbackResult:
    headline: str
    coaching_text: str
    tips: List[Tip]
    generated_by: str    # "claude" or "template"


# --------------------------------------------------------------------------- #
# Structured tips from the rule table
# --------------------------------------------------------------------------- #
def _fmt(dev: Deviation) -> str:
    spec = config.SPEC_BY_BASE.get(dev.base)
    unit = spec.unit if spec else ""
    where = f" {EVENT_LABELS[dev.event]}" if dev.event else ""
    return (f"Yours{where}: {dev.value:.1f}{unit} vs Tiger "
            f"{dev.tiger_mean:.1f}{unit} (±{dev.tiger_std:.1f}).")


def build_tips(result: ComparisonResult, top_n: int = config.TOP_N_DEVIATIONS) -> List[Tip]:
    tips: List[Tip] = []
    for dev in result.deviations[:top_n]:
        rule = RULES.get(dev.base)
        if not rule:
            continue
        msg = rule.get(dev.direction) or rule.get("high", "")
        spec = config.SPEC_BY_BASE.get(dev.base)
        title = spec.label if spec else dev.base
        if dev.event:
            title += f" ({EVENT_LABELS[dev.event]})"
        tips.append(Tip(title=title, message=msg, drill=rule.get("drill", ""),
                        detail=_fmt(dev)))
    return tips


# --------------------------------------------------------------------------- #
# Natural-language coaching
# --------------------------------------------------------------------------- #
def generate_feedback(result: ComparisonResult, use_claude: bool = True,
                      top_n: int = config.TOP_N_DEVIATIONS) -> FeedbackResult:
    tips = build_tips(result, top_n)
    headline = _headline(result)
    if use_claude:
        text = _claude_coaching(result, tips, headline)
        if text:
            return FeedbackResult(headline, text, tips, "claude")
    return FeedbackResult(headline, _template_coaching(result, tips, headline),
                          tips, "template")


def _headline(result: ComparisonResult) -> str:
    s = result.position_match
    band = ("a strong match to" if s >= 80 else
            "a solid foundation compared to" if s >= 60 else
            "some clear differences from" if s >= 40 else
            "significant differences from")
    return f"Your positions scored {s:.0f}/100 — {band} Tiger's mechanics."


def _template_coaching(result: ComparisonResult, tips: List[Tip], headline: str) -> str:
    if not tips:
        return headline + " No major biomechanical deviations stood out — nice work."
    lines = [headline, "", "Top things to work on:"]
    for i, t in enumerate(tips, 1):
        lines.append(f"{i}. {t.title}: {t.message} {t.detail} Drill: {t.drill}")
    return "\n".join(lines)


def _claude_coaching(result: ComparisonResult, tips: List[Tip],
                     headline: str) -> Optional[str]:
    """Phrase the structured tips via Claude. Returns None on any failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    system = [
        {"type": "text",
         "text": "You are TigerForm, an encouraging, concise golf swing coach. "
                 "You are given how a golfer's positions and tempo compare to Tiger "
                 "Woods' and a ranked list of biomechanical deviations with drills. "
                 "Write a short, motivating coaching note (120-180 words): open with "
                 "the score context, then give the 2-4 prioritized fixes in plain "
                 "language a golfer understands, weaving in the suggested drills. "
                 "Whenever you mention a drill, briefly explain how to do it in the "
                 "same sentence — assume the golfer has never heard of it. "
                 "Be specific and positive; never invent numbers beyond those given.",
         "cache_control": {"type": "ephemeral"}},
    ]
    payload = {
        "headline": headline,
        "position_match": result.position_match,
        "tempo_match": result.tempo_match,
        "tips": [asdict(t) for t in tips],
    }
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=config.FEEDBACK_MODEL,
            max_tokens=500,
            system=system,
            messages=[{"role": "user",
                       "content": json.dumps(payload, indent=2)}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    except Exception:
        return None
