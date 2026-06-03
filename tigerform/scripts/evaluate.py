"""Evaluate TigerForm on a held-out synthetic test set.

Reports:
* discriminator precision / recall / F1 / ROC-AUC on fresh swings,
* a score-separation test (Tiger-like swings should score clearly higher than
  amateur-like swings),
* event-segmentation accuracy vs. the known synthetic event frames (a PCE-style
  check), with a tolerance of +/- a few frames.

    python scripts/evaluate.py
"""
import _bootstrap  # noqa: F401
import numpy as np
from dataclasses import replace

from tigerform.analyze import analyze_sequence
from tigerform.compare import compare_swing
from tigerform.model import TigerDiscriminator
from tigerform.reference import ReferenceTemplate
from tigerform.synthetic import TIGER, AMATEUR, synthetic_pose, _phase_anchors
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def _test_set(n=20, seed=42):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        tp = replace(TIGER, n_frames=int(rng.integers(85, 105)))
        ap = replace(AMATEUR, n_frames=int(rng.integers(85, 105)))
        items.append((synthetic_pose(tp, seed=9000 + i), 1, tp))
        items.append((synthetic_pose(ap, seed=9500 + i), 0, ap))
    return items


def main():
    ref = ReferenceTemplate.load()
    disc = TigerDiscriminator.load()

    test = _test_set()
    analyzed = [(analyze_sequence(seq), label, seq, params) for seq, label, params in test]

    # --- discriminator metrics ---
    y_true = [lab for _, lab, _, _ in analyzed]
    y_prob = [disc.tiger_likeness(a.features) for a, _, _, _ in analyzed]
    y_pred = [int(p >= 0.5) for p in y_prob]
    print("=== Discriminator (held-out synthetic) ===")
    print(f"precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"f1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"roc_auc:   {roc_auc_score(y_true, y_prob):.3f}")

    # --- score separation ---
    tiger_scores, am_scores = [], []
    for a, lab, _, _ in analyzed:
        s = compare_swing(a.features, ref, disc).position_match
        (tiger_scores if lab == 1 else am_scores).append(s)
    print("\n=== Similarity-score separation ===")
    print(f"Tiger-like mean score:   {np.mean(tiger_scores):.1f}")
    print(f"Amateur-like mean score: {np.mean(am_scores):.1f}")
    print(f"Separation (Tiger - amateur): {np.mean(tiger_scores) - np.mean(am_scores):.1f}")

    # --- event segmentation (PCE-style) on known synthetic events ---
    print("\n=== Event segmentation accuracy (±3 frames) ===")
    hits, total = 0, 0
    for a, _, seq, params in analyzed:
        n = seq.n_frames
        p_top, p_impact = _phase_anchors(params.tempo_ratio)
        truth = {"address": 0, "top": int(p_top * (n - 1)),
                 "impact": int(p_impact * (n - 1)), "finish": n - 1}
        for ev, tf in truth.items():
            total += 1
            if abs(a.events.frames.get(ev, -99) - tf) <= 3:
                hits += 1
    print(f"PCE@3 (address/top/impact/finish): {hits / total:.2f}")


if __name__ == "__main__":
    main()
