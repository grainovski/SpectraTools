import pytest

from calibration import Calibration, CalibrationError


def test_linear_apply_matches_hand_computation():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    # E = a + b*channel
    assert cal.apply(0) == pytest.approx(10.0)
    assert cal.apply(100) == pytest.approx(10.0 + 0.5 * 100)
    assert cal.apply(200) == pytest.approx(10.0 + 0.5 * 200)


def test_quadratic_apply_matches_hand_computation():
    cal = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.001)
    # E = a + b*channel + c*channel**2
    assert cal.apply(0) == pytest.approx(10.0)
    assert cal.apply(100) == pytest.approx(10.0 + 0.5 * 100 + 0.001 * 100 ** 2)


def test_apply_broadcasts_over_a_numpy_array():
    import numpy as np

    cal = Calibration(kind="linear", a=10.0, b=0.5)
    channels = np.array([0.0, 100.0, 200.0])
    result = cal.apply(channels)
    expected = 10.0 + 0.5 * channels
    np.testing.assert_allclose(result, expected)


def test_b_zero_raises_calibration_error():
    with pytest.raises(CalibrationError):
        Calibration(kind="linear", a=10.0, b=0.0)


def test_derivative_linear_is_constant_b():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    assert cal.derivative(0) == pytest.approx(0.5)
    assert cal.derivative(500) == pytest.approx(0.5)


def test_derivative_quadratic_matches_hand_computation():
    cal = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.001)
    # dE/dchannel = b + 2*c*channel
    assert cal.derivative(100) == pytest.approx(0.5 + 2 * 0.001 * 100)
