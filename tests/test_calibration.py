import pytest

import calibration as calibration_module
from calibration import Calibration, CalibrationError, CalibrationFileError


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


def test_read_coefficients_file_linear(tmp_path):
    path = tmp_path / "cal.txt"
    path.write_text("10.5\n0.487\n")
    from calibration import read_coefficients_file
    assert read_coefficients_file(str(path), quadratic=False) == [10.5, 0.487]


def test_read_coefficients_file_quadratic(tmp_path):
    path = tmp_path / "cal.txt"
    path.write_text("10.5\n0.487\n0.0002\n")
    from calibration import read_coefficients_file
    assert read_coefficients_file(str(path), quadratic=True) == [10.5, 0.487, 0.0002]


def test_read_coefficients_file_ignores_blank_lines(tmp_path):
    path = tmp_path / "cal.txt"
    path.write_text("10.5\n\n0.487\n\n")
    from calibration import read_coefficients_file
    assert read_coefficients_file(str(path), quadratic=False) == [10.5, 0.487]


def test_read_coefficients_file_wrong_count_raises(tmp_path):
    path = tmp_path / "cal.txt"
    path.write_text("10.5\n")  # only 1 line, linear needs 2
    with pytest.raises(CalibrationFileError):
        from calibration import read_coefficients_file
        read_coefficients_file(str(path), quadratic=False)


def test_read_coefficients_file_non_numeric_raises(tmp_path):
    path = tmp_path / "cal.txt"
    path.write_text("10.5\nnot-a-number\n")
    with pytest.raises(CalibrationFileError):
        from calibration import read_coefficients_file
        read_coefficients_file(str(path), quadratic=False)


def test_read_coefficients_file_missing_file_raises(tmp_path):
    with pytest.raises(CalibrationFileError):
        from calibration import read_coefficients_file
        read_coefficients_file(str(tmp_path / "does_not_exist.txt"), quadratic=False)


def test_read_coefficients_file_strips_utf8_bom(tmp_path):
    from calibration import read_coefficients_file
    path = tmp_path / "cal.txt"
    path.write_bytes("10.5\n0.487\n".encode("utf-8-sig"))
    assert read_coefficients_file(str(path), quadratic=False) == [10.5, 0.487]


def test_read_coefficients_file_non_utf8_raises(tmp_path):
    from calibration import CalibrationFileError, read_coefficients_file
    path = tmp_path / "cal.txt"
    # A byte sequence that's invalid UTF-8 (0xFF is never valid in UTF-8).
    path.write_bytes(b"10.5\n\xff\xfe0.487\n")
    with pytest.raises(CalibrationFileError):
        read_coefficients_file(str(path), quadratic=False)


def test_rescaled_linear_calibration():
    # Rebinning by 4 sums old channels [4k, 4k+3] into new channel k, whose
    # CENTRE is old channel 4k + 1.5. So the new constant term is the old
    # energy at old channel 1.5, not at old channel 0.
    cal = Calibration(kind="linear", a=5.0, b=2.0)
    rescaled = cal.rescaled(4)
    assert rescaled.kind == "linear"
    assert rescaled.a == pytest.approx(5.0 + 2.0 * 1.5)
    assert rescaled.b == 8.0


def test_rescaled_quadratic_calibration():
    cal = Calibration(kind="quadratic", a=1.0, b=2.0, c=3.0)
    rescaled = cal.rescaled(2)
    offset = 0.5  # (2 - 1) / 2
    assert rescaled.a == pytest.approx(1.0 + 2.0 * offset + 3.0 * offset ** 2)
    assert rescaled.b == pytest.approx(2 * (2.0 + 2 * 3.0 * offset))
    assert rescaled.c == pytest.approx(12.0)


def test_rescaled_is_algebraically_equivalent_at_the_group_centre():
    """Defining property of the transform: a rebinned channel must report
    the energy of the CENTRE of the old channels it was summed from.

    This test previously asserted equivalence at old channel
    new_channel*factor -- the FIRST channel of the group -- which left
    every rebinned spectrum's energy axis low by half a bin at factor 2
    and by 3.5 bins at factor 8. The centre is the right anchor because a
    rebinned bin's counts come from the whole group, and it is the
    convention used everywhere else here (root_io reads a ROOT axis as
    edges[0] + width/2, the centre of bin 0).
    """
    cal = Calibration(kind="quadratic", a=3.0, b=1.5, c=0.02)
    factor = 5
    rescaled = cal.rescaled(factor)
    for new_channel in (0, 1, 37, 512):
        centre = new_channel * factor + (factor - 1) / 2.0
        assert rescaled.apply(new_channel) == pytest.approx(cal.apply(centre))


def test_rescaling_by_one_changes_nothing():
    """factor 1 is the identity: the group is one channel, so its centre
    is itself."""
    cal = Calibration(kind="quadratic", a=3.0, b=1.5, c=0.02)
    rescaled = cal.rescaled(1)
    assert rescaled.a == pytest.approx(cal.a)
    assert rescaled.b == pytest.approx(cal.b)
    assert rescaled.c == pytest.approx(cal.c)


# --- non-finite coefficient rejection (v3.1.0 audit, Minor) -------------
# float("nan")/float("inf") parse happily everywhere a coefficient can
# enter the app, and `nan == 0.0` is False so the pre-existing b-is-zero
# guard never caught them. An accepted non-finite coefficient makes
# apply() return NaN for every channel -- an active calibration that
# silently displays a blank energy everywhere.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_calibration_rejects_non_finite_a(bad):
    with pytest.raises(CalibrationError):
        Calibration(kind="linear", a=bad, b=0.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_calibration_rejects_non_finite_b(bad):
    with pytest.raises(CalibrationError):
        Calibration(kind="linear", a=1.0, b=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_calibration_rejects_non_finite_c(bad):
    with pytest.raises(CalibrationError):
        Calibration(kind="quadratic", a=1.0, b=0.5, c=bad)


def test_calibration_error_names_the_offending_coefficient():
    with pytest.raises(CalibrationError, match="'b'"):
        Calibration(kind="linear", a=1.0, b=float("nan"))


def test_finite_coefficients_still_accepted():
    # Guards against the new check being too aggressive: a very large but
    # finite coefficient is legal and must still construct.
    cal = Calibration(kind="quadratic", a=-1e300, b=1e300, c=0.0)
    assert cal.b == 1e300


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "Infinity", "NaN"])
def test_read_coefficients_file_rejects_non_finite(tmp_path, text):
    from calibration import read_coefficients_file

    path = tmp_path / "cal.txt"
    path.write_text(f"10.5\n{text}\n")
    with pytest.raises(CalibrationFileError, match="finite"):
        read_coefficients_file(str(path), quadratic=False)


# --- X3: calibrating from assigned peak energies (v4.0.0) --------------


def test_from_points_recovers_a_known_linear_calibration():
    import calibration as calibration_module

    channels = [100.0, 300.0, 700.0, 1500.0]
    energies = [10.0 + 0.5 * ch for ch in channels]
    cal = calibration_module.from_points(channels, energies)

    assert cal.kind == "linear"
    assert cal.a == pytest.approx(10.0)
    assert cal.b == pytest.approx(0.5)


def test_from_points_recovers_a_known_quadratic_calibration():
    import calibration as calibration_module

    channels = [50.0, 400.0, 900.0, 1600.0, 2000.0]
    energies = [2.0 + 0.4 * ch + 1e-5 * ch ** 2 for ch in channels]
    cal = calibration_module.from_points(channels, energies, quadratic=True)

    assert cal.kind == "quadratic"
    assert cal.a == pytest.approx(2.0, abs=1e-6)
    assert cal.b == pytest.approx(0.4, abs=1e-8)
    assert cal.c == pytest.approx(1e-5, abs=1e-12)


def test_from_points_needs_enough_points_to_say_anything():
    """Fewer than the minimum would pass exactly through the points and
    describe nothing."""
    import calibration as calibration_module
    from calibration import CalibrationError

    with pytest.raises(CalibrationError, match="at least 2"):
        calibration_module.from_points([100.0], [50.0])
    with pytest.raises(CalibrationError, match="at least 3"):
        calibration_module.from_points([100.0, 200.0], [50.0, 90.0], quadratic=True)


def test_from_points_refuses_duplicate_channels():
    """Two peaks at the same channel constrain one point, not two --
    numpy would otherwise return a fit with a silently meaningless
    slope."""
    import calibration as calibration_module
    from calibration import CalibrationError

    with pytest.raises(CalibrationError, match="different channels"):
        calibration_module.from_points([500.0, 500.0], [100.0, 200.0])


def test_from_points_requires_one_energy_per_channel():
    import calibration as calibration_module
    from calibration import CalibrationError

    with pytest.raises(CalibrationError, match="exactly one energy"):
        calibration_module.from_points([1.0, 2.0, 3.0], [10.0, 20.0])


def test_residuals_expose_a_single_mistyped_energy():
    """The reason residuals are reported and not just the coefficients: one
    wrong energy moves the whole fit, and a and b look equally plausible
    either way."""
    import calibration as calibration_module

    channels = [100.0, 300.0, 700.0, 1500.0]
    energies = [10.0 + 0.5 * ch for ch in channels]
    energies[2] += 40.0  # a fat-fingered entry

    cal = calibration_module.from_points(channels, energies)
    residuals = calibration_module.residuals(cal, channels, energies)

    assert max(abs(r) for r in residuals) > 10.0
    # The bad point is the one that stands out.
    assert abs(residuals[2]) == pytest.approx(max(abs(r) for r in residuals))


def test_residuals_are_essentially_zero_for_consistent_assignments():
    import calibration as calibration_module

    channels = [100.0, 300.0, 700.0, 1500.0]
    energies = [10.0 + 0.5 * ch for ch in channels]
    cal = calibration_module.from_points(channels, energies)
    assert max(abs(r) for r in calibration_module.residuals(cal, channels, energies)) < 1e-9


# ---------------------------------------------------------------------------
# v4.1.0 audit S6: calibrating from fitted peaks must use the fit's own
# precision, and report how well the coefficients came out.
# ---------------------------------------------------------------------------


def test_from_points_weights_by_the_centroid_uncertainties():
    """A badly-determined centroid must not pull as hard as a precise one.

    Four points on a true E = 10 + 0.5*ch, with one deliberately displaced
    and carrying a large uncertainty that says so.
    """
    channels = [100.0, 200.0, 300.0, 400.0]
    truth = [10.0 + 0.5 * ch for ch in channels]
    energies = list(truth)
    energies[2] += 20.0                       # a badly-placed peak
    errors = [0.05, 0.05, 10.0, 0.05]         # and the fit knew it was bad

    unweighted = calibration_module.from_points(channels, energies)
    weighted = calibration_module.from_points(
        channels, energies, channel_errors=errors
    )

    assert abs(weighted.b - 0.5) < abs(unweighted.b - 0.5)
    assert weighted.b == pytest.approx(0.5, abs=0.01)


def test_from_points_without_errors_is_unchanged():
    channels = [100.0, 300.0, 500.0]
    energies = [60.0, 160.0, 260.0]
    assert calibration_module.from_points(channels, energies) == \
        calibration_module.from_points(channels, energies, channel_errors=None)


def test_from_points_falls_back_when_an_error_is_unusable():
    """A fixed or undetermined position has no usable weight. Weighting
    the others around it would be worse than weighting none of them."""
    channels = [100.0, 300.0, 500.0]
    energies = [60.0, 160.0, 260.0]
    plain = calibration_module.from_points(channels, energies)
    for unusable in ([0.1, 0.0, 0.2], [0.1, float("nan"), 0.2], [0.1, -1.0, 0.2]):
        result = calibration_module.from_points(
            channels, energies, channel_errors=unusable
        )
        assert result == plain


def test_from_points_rejects_a_mismatched_error_count():
    with pytest.raises(CalibrationError, match="exactly one uncertainty"):
        calibration_module.from_points(
            [1.0, 2.0, 3.0], [10.0, 20.0, 30.0], channel_errors=[0.1, 0.2]
        )


def test_from_points_reports_coefficient_uncertainties():
    channels = [100.0, 200.0, 300.0, 400.0, 500.0]
    energies = [10.0 + 0.5 * ch for ch in channels]
    energies[1] += 0.4                        # a little real scatter
    cal = calibration_module.from_points(channels, energies)
    assert len(cal.coefficient_errors) == 2   # (a_err, b_err), lowest first
    assert all(e > 0.0 for e in cal.coefficient_errors)
    # b is far better determined than a over a long lever arm.
    assert cal.coefficient_errors[1] < cal.coefficient_errors[0]


def test_from_points_reports_no_uncertainty_at_the_minimum_points():
    """Two points define a line exactly. There is no scatter to estimate
    from, and a fabricated zero would read as a perfect calibration."""
    cal = calibration_module.from_points([100.0, 500.0], [60.0, 260.0])
    assert cal.coefficient_errors == ()


def test_coefficient_errors_do_not_affect_calibration_equality():
    """Provenance, not identity: the same coefficients transform channels
    the same way however they were arrived at."""
    fitted = calibration_module.from_points(
        [100.0, 200.0, 300.0, 400.0],
        [10.0 + 0.5 * ch for ch in [100.0, 200.0, 300.0, 400.0]],
    )
    typed = Calibration(kind="linear", a=fitted.a, b=fitted.b)
    assert fitted == typed


# --- turning point / folded calibrations -------------------------------


def test_turning_point_is_none_for_a_line():
    """A line never reverses, so there is nothing to warn about."""
    from calibration import turning_point

    assert turning_point(Calibration(kind="linear", a=1.0, b=2.0)) is None


def test_turning_point_is_none_when_the_quadratic_term_is_zero():
    """kind='quadratic' with c == 0 IS a line, whatever it is called."""
    from calibration import turning_point

    assert turning_point(Calibration(kind="quadratic", a=1.0, b=2.0, c=0.0)) is None


def test_turning_point_is_where_the_derivative_vanishes():
    from calibration import turning_point

    calibration = Calibration(kind="quadratic", a=0.0, b=2.0, c=1.0)
    turning = turning_point(calibration)
    assert turning == pytest.approx(-1.0)
    assert calibration.derivative(turning) == pytest.approx(0.0)


def test_turning_point_reproduces_the_folded_case_a_sweep_found():
    """The concrete calibration a randomised sweep produced, whose
    inversion returned the other branch: its vertex sits at ~6967,
    inside an 8192-channel spectrum."""
    from calibration import turning_point

    calibration = Calibration(kind="quadratic", a=1.242, b=0.960, c=-6.89e-05)
    turning = turning_point(calibration)
    assert turning == pytest.approx(6966.6, abs=0.1)
    # Both sides of the fold genuinely share an energy -- which is why
    # inverting is ambiguous rather than wrong.
    left, right = turning - 500.0, turning + 500.0
    assert calibration.apply(left) == pytest.approx(calibration.apply(right))
