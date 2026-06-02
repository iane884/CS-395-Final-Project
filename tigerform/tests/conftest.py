"""Shared fixtures: synthetic swings + in-memory reference/discriminator.

Everything is built from the synthetic generator so the test suite needs no
video files, MediaPipe, or trained artifacts on disk.
"""
from dataclasses import replace

import numpy as np
import pytest

from tigerform.analyze import analyze_sequence
from tigerform.model import train_discriminator
from tigerform.reference import build_reference
from tigerform.synthetic import TIGER, AMATEUR, synthetic_pose, make_dataset


@pytest.fixture(scope="session")
def tiger_swing():
    return synthetic_pose(TIGER, seed=1)


@pytest.fixture(scope="session")
def amateur_swing():
    return synthetic_pose(AMATEUR, seed=2)


@pytest.fixture(scope="session")
def tiger_features(tiger_swing):
    return analyze_sequence(tiger_swing).features


@pytest.fixture(scope="session")
def amateur_features(amateur_swing):
    return analyze_sequence(amateur_swing).features


@pytest.fixture(scope="session")
def reference():
    rng = np.random.default_rng(7)
    feats = [analyze_sequence(synthetic_pose(
        replace(TIGER, n_frames=int(rng.integers(85, 105))), seed=500 + i)).features
        for i in range(20)]
    return build_reference(feats)


@pytest.fixture(scope="session")
def discriminator():
    dataset = make_dataset(n_tiger=20, n_amateur=24, seed=0)
    feats = [analyze_sequence(seq).features for seq, _ in dataset]
    labels = [y for _, y in dataset]
    disc, _ = train_discriminator(feats, labels)
    return disc
