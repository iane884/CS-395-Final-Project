import numpy as np

from tigerform.geometry import (angle_between, line_angle_from_horizontal,
                                line_angle_from_vertical, moving_average,
                                signed_rotation_2d)


def test_angle_between_right_angle():
    a, b, c = [1, 0, 0], [0, 0, 0], [0, 1, 0]
    assert abs(angle_between(a, b, c) - 90.0) < 1e-6


def test_angle_between_straight():
    a, b, c = [1, 0, 0], [0, 0, 0], [-1, 0, 0]
    assert abs(angle_between(a, b, c) - 180.0) < 1e-6


def test_angle_degenerate_is_nan():
    assert np.isnan(angle_between([0, 0], [0, 0], [1, 1]))


def test_line_angles():
    assert abs(line_angle_from_vertical([0, 0], [0, 5])) < 1e-6
    assert abs(line_angle_from_horizontal([0, 0], [5, 0])) < 1e-6
    assert abs(line_angle_from_horizontal([0, 0], [0, 5]) - 90.0) < 1e-6


def test_signed_rotation():
    assert abs(signed_rotation_2d([1, 0], [0, 1]) - 90.0) < 1e-6
    assert abs(signed_rotation_2d([1, 0], [0, -1]) + 90.0) < 1e-6


def test_moving_average_preserves_length_and_smooths():
    x = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
    y = moving_average(x, 3)
    assert y.shape == x.shape
    assert y[1] < 10.0 and y[1] > 0.0


def test_moving_average_nan_aware():
    x = np.array([1.0, np.nan, 3.0, 4.0])
    y = moving_average(x, 3)
    assert np.all(np.isfinite(y))
