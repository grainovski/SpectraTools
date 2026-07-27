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


def test_invert_linear_round_trips_apply():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    for channel in (0.0, 1.0, 100.0, 500.0, 4095.0):
        energy = cal.apply(channel)
        assert cal.invert(energy) == pytest.approx(channel, abs=1e-6)


def test_invert_quadratic_round_trips_apply():
    cal = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.0002)
    # TV's 0.01 keV convergence criterion bounds the ENERGY residual, not
    # the channel residual directly -- channel error is ~0.01/derivative,
    # coarser here than the linear case above (where the initial guess is
    # already the exact algebraic solution, needing no iteration at all).
    # Tolerance verified against these specific sampled channels only, not
    # a domain-wide worst-case search.
    for channel in (0.0, 1.0, 100.0, 500.0, 4095.0):
        energy = cal.apply(channel)
        assert cal.invert(energy) == pytest.approx(channel, abs=0.01)


def test_invert_matches_hand_worked_newton_example():
    # Hand-worked: E = 10 + 0.5*ch + 0.0002*ch**2, solve for E=120.
    # Linear initial guess: x0 = (120-10)/0.5 = 220.
    # apply(220) = 10 + 110 + 0.0002*48400 = 10 + 110 + 9.68 = 129.68
    # de = 129.68 - 120 = 9.68; gradient = 0.5 + 2*0.0002*220 = 0.588
    # x1 = 220 - 9.68/0.588 = 220 - 16.462... = 203.537...
    # Iterate to convergence (|de| < 0.01) and confirm apply(x) == 120.
    cal = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.0002)
    channel = cal.invert(120.0)
    assert cal.apply(channel) == pytest.approx(120.0, abs=0.01)
    # Confirms the iteration actually moved from the linear-only guess
    # of 220 rather than returning it unrefined.
    assert channel != pytest.approx(220.0, abs=1.0)


def test_invert_negative_b_still_round_trips():
    # b < 0 is a valid (if unusual) calibration -- energy decreasing
    # with channel. Newton's method must still converge.
    cal = Calibration(kind="linear", a=1000.0, b=-0.5)
    energy = cal.apply(300.0)
    assert cal.invert(energy) == pytest.approx(300.0, abs=1e-6)


def test_invert_raises_on_zero_derivative_at_vertex():
    # x0 = (energy - a) / b = (-615 - 10) / 0.5 = -1250, which is exactly
    # this calibration's vertex (x = -b/(2c) = -0.5/0.0004 = -1250), where
    # derivative(x) = b + 2*c*x = 0.5 + 2*0.0002*(-1250) = 0 -- Newton's
    # method cannot divide by this and must fail clearly, not crash with
    # an unguarded ZeroDivisionError.
    cal = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.0002)
    with pytest.raises(CalibrationError):
        cal.invert(-615.0)
