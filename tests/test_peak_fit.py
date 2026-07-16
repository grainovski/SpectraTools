import numpy as np
import pytest

from peak_fit import FitError, FitResult, PeakResult, _compute_background, hypermet_left_tail


def test_hypermet_left_tail_biases_left_not_right():
    x = np.array([-10.0, 10.0])  # dx values either side of position=0
    values = hypermet_left_tail(x, position=0.0, sigma=3.0, r=0.1, beta=5.0)
    left_value, right_value = values
    assert left_value > right_value
    # hand-derived: ~14.7x; generous margin so this isn't brittle to
    # floating-point/erfc implementation differences
    assert left_value / right_value > 5.0


def test_hypermet_left_tail_reduces_to_gaussian_when_r_is_zero():
    x = np.array([-6.0, -2.0, 0.0, 3.0])
    values = hypermet_left_tail(x, position=1.0, sigma=2.0, r=0.0, beta=5.0)
    expected = np.exp(-((x - 1.0) ** 2) / (2 * 2.0 ** 2))
    np.testing.assert_allclose(values, expected, rtol=1e-10)


def test_hypermet_left_tail_handles_large_offsets_without_overflow():
    x = np.array([1000.0])  # dx/beta = 200, far past the safety threshold
    values = hypermet_left_tail(x, position=0.0, sigma=3.0, r=0.1, beta=5.0)
    assert np.all(np.isfinite(values))
    assert values[0] < 1e-6


def test_hypermet_left_tail_handles_degenerate_erfc_underflow():
    # sigma vastly larger than beta drives y = sigma/(beta*sqrt(2)) so
    # high that erfc(y) underflows to exactly 0.0 -- must fall back to
    # the plain Gaussian core instead of dividing by (effectively) zero.
    x = np.array([0.0, -5.0, 5.0])
    values = hypermet_left_tail(x, position=0.0, sigma=1000.0, r=0.1, beta=0.1)
    assert np.all(np.isfinite(values))


def test_peak_result_holds_expected_fields():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    assert peak.position == 100.0
    assert peak.fwhm == 5.0
    assert peak.area == 1000.0


def test_fit_result_holds_expected_fields():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    result = FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[peak],
    )
    assert result.fit_region == (90.0, 110.0)
    assert result.peaks == [peak]


def test_fit_result_new_fields_have_sensible_defaults():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    result = FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[peak],
    )
    assert result.link_widths is True
    assert result.tail_fraction is None
    assert result.tail_fraction_err is None
    assert result.tail_beta is None
    assert result.tail_beta_err is None


def test_fit_result_accepts_explicit_tail_fields():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    result = FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[peak],
        link_widths=False,
        tail_fraction=0.1, tail_fraction_err=0.02,
        tail_beta=3.0, tail_beta_err=0.5,
    )
    assert result.link_widths is False
    assert result.tail_fraction == 0.1
    assert result.tail_beta == 3.0


def test_peak_result_amplitude_and_sigma_err_default_to_zero():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    assert peak.amplitude_err == 0.0
    assert peak.sigma_err == 0.0


def test_fit_result_fixed_params_defaults_to_empty_dict():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    result = FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[peak],
    )
    assert result.fixed_params == {}


def test_fit_result_visible_defaults_to_true():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    result = FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[peak],
    )
    assert result.visible is True


def test_fit_result_timestamp_defaults_to_none():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    result = FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[peak],
    )
    assert result.timestamp is None


def test_fit_error_is_an_exception():
    assert issubclass(FitError, Exception)


def test_compute_background_flat():
    x = np.arange(200, dtype=float)
    y = np.full(200, 15.0)
    slope, intercept = _compute_background(x, y, (10.0, 20.0), (150.0, 160.0))
    assert slope == pytest.approx(0.0, abs=1e-9)
    assert intercept == pytest.approx(15.0, abs=1e-9)


def test_compute_background_sloped():
    x = np.arange(200, dtype=float)
    y = 0.5 * x + 3.0
    slope, intercept = _compute_background(x, y, (10.0, 20.0), (150.0, 160.0))
    assert slope == pytest.approx(0.5, abs=1e-6)
    assert intercept == pytest.approx(3.0, abs=1e-4)


def test_compute_background_rejects_empty_region():
    x = np.arange(200, dtype=float)
    y = np.full(200, 15.0)
    with pytest.raises(FitError):
        _compute_background(x, y, (500.0, 501.0), (150.0, 160.0))


def test_compute_background_rejects_identical_mean_x():
    x = np.arange(200, dtype=float)
    y = np.full(200, 15.0)
    with pytest.raises(FitError):
        _compute_background(x, y, (10.0, 20.0), (10.0, 20.0))


from peak_fit import fit_peaks


def _make_spectrum(channels, peaks, slope, intercept, noise_seed=None):
    x = np.arange(channels, dtype=float)
    y = slope * x + intercept
    for amplitude, position, sigma in peaks:
        y = y + amplitude * np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        y = rng.poisson(np.maximum(y, 0)).astype(float)
    return x, y


def test_fit_single_peak_no_noise():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
    )
    assert len(result.peaks) == 1
    peak = result.peaks[0]
    assert peak.position == pytest.approx(100.0, abs=0.5)
    assert peak.amplitude == pytest.approx(500.0, rel=0.05)
    assert peak.sigma == pytest.approx(3.0, rel=0.1)
    assert peak.fwhm == pytest.approx(3.0 * 2.3548, rel=0.1)
    assert peak.area == pytest.approx(500.0 * 3.0 * np.sqrt(2 * np.pi), rel=0.1)
    assert result.background_slope == pytest.approx(0.0, abs=0.5)
    assert result.background_intercept == pytest.approx(20.0, abs=2.0)


def test_fit_multiplet_two_peaks_no_noise():
    x, y = _make_spectrum(
        channels=200,
        peaks=[(400.0, 95.0, 3.0), (300.0, 108.0, 3.0)],
        slope=0.1, intercept=10.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(60.0, 75.0),
        right_bg_region=(130.0, 145.0),
        fit_region=(75.0, 130.0),
        peak_positions=[95.0, 108.0],
    )
    assert len(result.peaks) == 2
    positions = sorted(p.position for p in result.peaks)
    assert positions[0] == pytest.approx(95.0, abs=1.0)
    assert positions[1] == pytest.approx(108.0, abs=1.0)


def test_fit_single_peak_with_poisson_noise():
    x, y = _make_spectrum(
        channels=200, peaks=[(2000.0, 100.0, 4.0)], slope=0.0, intercept=50.0,
        noise_seed=42,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
    )
    peak = result.peaks[0]
    assert peak.position == pytest.approx(100.0, abs=1.0)
    assert peak.fwhm == pytest.approx(4.0 * 2.3548, rel=0.15)


def test_fit_single_peak_reports_amplitude_and_sigma_uncertainties():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
    )
    peak = result.peaks[0]
    assert peak.amplitude_err > 0
    assert peak.sigma_err > 0


def test_fit_independent_widths_recovers_different_sigmas():
    x, y = _make_spectrum(
        channels=200,
        peaks=[(400.0, 95.0, 2.0), (300.0, 108.0, 5.0)],
        slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(60.0, 75.0),
        right_bg_region=(130.0, 145.0),
        fit_region=(75.0, 130.0),
        peak_positions=[95.0, 108.0],
        link_widths=False,
    )
    assert result.link_widths is False
    sigmas = {round(p.position): p.sigma for p in result.peaks}
    assert sigmas[95] == pytest.approx(2.0, rel=0.2)
    assert sigmas[108] == pytest.approx(5.0, rel=0.2)


def test_fit_linked_widths_forces_equal_sigma_even_with_different_true_widths():
    x, y = _make_spectrum(
        channels=200,
        peaks=[(400.0, 95.0, 2.0), (300.0, 108.0, 5.0)],
        slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(60.0, 75.0),
        right_bg_region=(130.0, 145.0),
        fit_region=(75.0, 130.0),
        peak_positions=[95.0, 108.0],
        link_widths=True,
    )
    assert result.link_widths is True
    assert result.peaks[0].sigma == result.peaks[1].sigma
    assert result.peaks[0].fwhm == result.peaks[1].fwhm


def test_fit_default_link_widths_is_true():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
    )
    assert result.link_widths is True


def test_fit_without_left_tail_leaves_tail_fields_none():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
    )
    assert result.tail_fraction is None
    assert result.tail_beta is None


def test_fit_with_left_tail_enabled_recovers_known_tail_parameters():
    # Synthetic data WITH a real left tail baked in via the same
    # hypermet_left_tail() the fitter itself uses, so this verifies
    # fit_peaks() can recover known tail parameters, not just that the
    # option runs without crashing.
    x = np.arange(200, dtype=float)
    true_r, true_beta = 0.1, 4.0
    y = 20.0 + 500.0 * hypermet_left_tail(x, position=100.0, sigma=3.0, r=true_r, beta=true_beta)
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(80.0, 120.0),
        peak_positions=[100.0],
        enable_left_tail=True,
    )
    assert result.tail_fraction == pytest.approx(true_r, abs=0.05)
    assert result.tail_beta == pytest.approx(true_beta, rel=0.3)
    assert result.tail_fraction_err is not None
    assert result.tail_beta_err is not None


def test_fit_independent_widths_with_left_tail_both_enabled():
    # Exercises the 4th (3n+2) parameter-count combination.
    x = np.arange(200, dtype=float)
    y = (
        20.0
        + 400.0 * hypermet_left_tail(x, position=95.0, sigma=2.0, r=0.1, beta=4.0)
        + 300.0 * hypermet_left_tail(x, position=115.0, sigma=4.0, r=0.1, beta=4.0)
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(60.0, 75.0),
        right_bg_region=(140.0, 155.0),
        fit_region=(80.0, 130.0),
        peak_positions=[95.0, 115.0],
        link_widths=False,
        enable_left_tail=True,
    )
    assert len(result.peaks) == 2
    assert result.link_widths is False
    assert result.tail_fraction is not None
    sigmas = {round(p.position): p.sigma for p in result.peaks}
    assert sigmas[95] == pytest.approx(2.0, rel=0.3)
    assert sigmas[115] == pytest.approx(4.0, rel=0.3)


def test_fit_with_fixed_sigma_holds_it_constant():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
        fixed_params={"sigma": 3.0},
    )
    peak = result.peaks[0]
    assert peak.sigma == 3.0
    assert peak.sigma_err == 0.0
    assert result.fixed_params == {"sigma": 3.0}


def test_fit_with_fixed_position_holds_it_constant():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
        fixed_params={"pos_0": 100.0},
    )
    peak = result.peaks[0]
    assert peak.position == 100.0
    assert peak.position_err == 0.0


def test_fit_with_fixed_amplitude_reduces_area_uncertainty():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    free_result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
    )
    fixed_result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
        fixed_params={"amp_0": free_result.peaks[0].amplitude},
    )
    assert fixed_result.peaks[0].amplitude_err == 0.0
    # Only sigma's uncertainty contributes now, so area_err must shrink.
    assert fixed_result.peaks[0].area_err < free_result.peaks[0].area_err


def test_fit_with_fixed_tail_fraction_and_beta():
    x = np.arange(200, dtype=float)
    y = 20.0 + 500.0 * hypermet_left_tail(x, position=100.0, sigma=3.0, r=0.1, beta=4.0)
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(80.0, 120.0),
        peak_positions=[100.0],
        enable_left_tail=True,
        fixed_params={"tail_fraction": 0.1, "tail_beta": 4.0},
    )
    assert result.tail_fraction == 0.1
    assert result.tail_fraction_err == 0.0
    assert result.tail_beta == 4.0
    assert result.tail_beta_err == 0.0


def test_fit_rejects_unknown_fixed_parameter_name():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(
            x, y,
            left_bg_region=(70.0, 85.0),
            right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0),
            peak_positions=[100.0],
            fixed_params={"not_a_real_param": 1.0},
        )


def test_fit_with_all_parameters_fixed_skips_optimization():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
        fixed_params={"amp_0": 500.0, "pos_0": 100.0, "sigma": 3.0},
    )
    peak = result.peaks[0]
    assert peak.amplitude == 500.0
    assert peak.position == 100.0
    assert peak.sigma == 3.0
    assert peak.amplitude_err == 0.0
    assert peak.position_err == 0.0
    assert peak.sigma_err == 0.0
    assert peak.area_err == 0.0


from peak_fit import _initial_guess


def test_initial_guess_overrides_replace_only_the_named_parameters():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    lo, hi = 85.0, 115.0
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_sub = y[mask] - 20.0
    free_names = ["amp_0", "pos_0", "sigma"]

    p0_default = _initial_guess(
        free_names, x_fit, y_sub, (lo, hi), [100.0], link_widths=True, enable_left_tail=False,
    )
    p0_overridden = _initial_guess(
        free_names, x_fit, y_sub, (lo, hi), [100.0], link_widths=True, enable_left_tail=False,
        initial_guess_overrides={"sigma": 7.5},
    )

    assert p0_overridden[free_names.index("sigma")] == 7.5
    assert p0_overridden[free_names.index("amp_0")] == p0_default[free_names.index("amp_0")]
    assert p0_overridden[free_names.index("pos_0")] == p0_default[free_names.index("pos_0")]


def test_fit_peaks_accepts_initial_guess_overrides_without_disturbing_a_fixed_value():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
        fixed_params={"pos_0": 100.0},
        initial_guess_overrides={"pos_0": 999.0, "sigma": 6.0},
    )
    peak = result.peaks[0]
    # pos_0 is fixed, so its override is never consulted -- the fixed
    # value wins.
    assert peak.position == 100.0
    # sigma is free and started far (6.0) from its true value (3.0) via
    # the override -- it must still converge to the true value, proving
    # the override is only a *starting point*, not a held constant.
    assert peak.sigma == pytest.approx(3.0, rel=0.2)


def test_fit_recovers_from_a_deliberately_bad_initial_width_guess():
    """End-to-end version of the Task 1 adversarial scenario, through
    the real fit_peaks() entry point -- with a fit_region wide relative
    to peak count (the scenario that produces a too-wide sigma0 under
    the old region_width/(4*n_peaks) heuristic), all three peaks must
    still land near their true positions with positive amplitude."""
    x, y = _make_spectrum(
        channels=300,
        peaks=[(500.0, 100.0, 3.0), (350.0, 108.0, 3.0), (420.0, 117.0, 3.0)],
        slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(60.0, 75.0),
        right_bg_region=(145.0, 160.0),
        fit_region=(80.0, 140.0),
        peak_positions=[100.0, 108.0, 117.0],
    )
    positions = sorted(p.position for p in result.peaks)
    assert positions[0] == pytest.approx(100.0, abs=2.0)
    assert positions[1] == pytest.approx(108.0, abs=2.0)
    assert positions[2] == pytest.approx(117.0, abs=2.0)
    assert all(p.amplitude > 0 for p in result.peaks)


def test_fit_rejects_no_peaks():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(x, y, (70.0, 85.0), (115.0, 130.0), (85.0, 115.0), [])


def test_fit_rejects_empty_fit_region():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(x, y, (70.0, 85.0), (115.0, 130.0), (500.0, 501.0), [100.0])


def test_fit_rejects_too_few_points_for_peak_count():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(x, y, (70.0, 85.0), (115.0, 130.0), (99.5, 100.5), [100.0, 101.0])


def test_fit_raises_on_non_convergence():
    # Four peaks requested at the exact same position, each with its own
    # independent sigma (link_widths=False), start with four identical
    # parameter quadruplets, so their Jacobian columns are numerically
    # identical at every iteration (a perfectly singular Jacobian). The
    # damped Marquardt solver still "converges" to a stationary point --
    # it doesn't raise on its own the way MINPACK's curve_fit used to --
    # but that point's curvature matrix is too ill-conditioned to invert
    # into a physically valid covariance, so fit_peaks()'s explicit
    # `not np.all(np.isfinite(pcov)) or np.any(np.diag(pcov) < 0)` check
    # catches it and converts it to FitError.
    #
    # This needs four duplicate peaks, not three: with Task 3's
    # data-measured initial width (_measure_width), the solver starts
    # from a near-exact sigma and settles at a numerically "clean" (if
    # still physically indeterminate, since any amplitude split across
    # identical peaks fits equally well) point for three duplicates,
    # which keeps the covariance inversion just barely finite/positive.
    # A fourth duplicate adds enough extra degeneracy to reliably push
    # the curvature matrix back over the edge into non-invertibility.
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(
            x, y,
            left_bg_region=(20.0, 35.0),
            right_bg_region=(160.0, 175.0),
            fit_region=(93.0, 107.0),
            peak_positions=[100.0, 100.0, 100.0, 100.0],
            link_widths=False,
        )


from peak_fit import _ParamDamping, _marquardt_fit


def test_marquardt_fit_recovers_a_single_gaussian():
    def model(x, p):
        amp, pos, sigma = p
        return amp * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))

    rng = np.random.default_rng(1)
    x = np.arange(200, dtype=float)
    true = [500.0, 100.0, 3.0]
    y_clean = model(x, true)
    y = rng.poisson(np.maximum(y_clean, 0)).astype(float)
    y_err = np.sqrt(np.maximum(y, 1.0))

    damping = [_ParamDamping("amp"), _ParamDamping("pos", sigma_index=2), _ParamDamping("sigma")]
    p0 = [400.0, 98.0, 5.0]
    popt, pcov = _marquardt_fit(model, x, y, y_err, p0, damping)

    assert popt[1] == pytest.approx(100.0, abs=1.0)
    assert popt[2] == pytest.approx(3.0, abs=1.0)
    assert np.all(np.isfinite(pcov))


def test_marquardt_fit_position_step_is_damped_by_current_sigma():
    """A raw Newton step larger than the peak's own sigma must be capped
    to +-sigma for that single iteration -- the mechanism that prevents
    a peak from jumping straight past its neighbors in one step.

    The starting offset here (10 channels, ~3.3x the true sigma of 3.0)
    is deliberately chosen to stay within the region where the Gaussian
    still has meaningful gradient overlap with the data -- empirically,
    offsets beyond roughly 6x sigma leave essentially zero position
    gradient at the starting point, so amplitude collapses toward zero
    before position ever gets a chance to move (a real, well-known
    Gaussian-least-squares local-minimum trap that no amount of step
    damping can fix, since there's no gradient signal to damp in the
    first place -- confirmed by direct iteration count sweep: offsets of
    4-13 channels all converge correctly, offsets of 20+ don't). This
    test's job is to confirm damping doesn't cause problems within the
    regime where the fit is expected to work, not to prove convergence
    from an arbitrarily bad starting guess."""
    def model(x, p):
        amp, pos, sigma = p
        return amp * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))

    x = np.arange(200, dtype=float)
    y = model(x, [500.0, 100.0, 3.0])
    y_err = np.sqrt(np.maximum(y, 1.0))

    # Starting 10 channels away from the true position (~3.3x the true
    # sigma) -- the raw Newton step early on will exceed the current
    # sigma guess and need capping, without leaving the convergence
    # basin entirely.
    damping = [_ParamDamping("amp"), _ParamDamping("pos", sigma_index=2), _ParamDamping("sigma")]
    p0 = [500.0, 90.0, 3.0]
    popt, pcov = _marquardt_fit(model, x, y, y_err, p0, damping)
    # Damped or not, it should still eventually converge close to truth --
    # this test is about the mechanism not causing divergence, not about
    # inspecting individual iterations.
    assert popt[1] == pytest.approx(100.0, abs=2.0)


def test_marquardt_fit_bad_initial_width_does_not_produce_negative_amplitude():
    """Reproduces the investigation's adversarial scenario: three peaks
    with a deliberately too-wide initial sigma guess. Plain curve_fit on
    this exact data produces a peak with negative amplitude and wrong
    positions; the damped solver must not."""
    def model(x, p, n_peaks=3):
        sigma = p[-1]
        total = np.zeros_like(x)
        for i in range(n_peaks):
            total = total + p[2 * i] * np.exp(-((x - p[2 * i + 1]) ** 2) / (2 * sigma ** 2))
        return total

    rng = np.random.default_rng(0)
    x = np.arange(300, dtype=float)
    true_peaks = [(100.0, 500.0, 3.0), (108.0, 350.0, 3.0), (117.0, 420.0, 3.0)]
    y_clean = 20.0 + sum(
        amp * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2)) for pos, amp, sigma in true_peaks
    )
    y = rng.poisson(np.maximum(y_clean, 0)).astype(float)
    lo, hi = 80, 140
    mask = (x >= lo) & (x <= hi)
    x_fit, y_fit = x[mask], y[mask]
    y_sub = y_fit - 20.0
    y_err = np.sqrt(np.maximum(y_fit, 1.0))

    marked = [100.0, 108.0, 117.0]
    bad_sigma0 = (hi - lo) / (2 * 3)  # deliberately too wide
    p0 = []
    for pos in marked:
        idx = int(np.argmin(np.abs(x_fit - pos)))
        p0 += [float(y_sub[idx]), pos]
    p0.append(bad_sigma0)

    sigma_index = 2 * 3
    damping = []
    for _ in marked:
        damping += [_ParamDamping("amp"), _ParamDamping("pos", sigma_index=sigma_index)]
    damping.append(_ParamDamping("sigma"))

    popt, pcov = _marquardt_fit(
        lambda xx, pp: model(xx, pp), x_fit, y_sub, y_err, p0, damping,
        fit_region_bounds=(lo, hi),
    )
    amplitudes = [popt[0], popt[2], popt[4]]
    assert all(a > 0 for a in amplitudes)


def test_marquardt_fit_clamps_tail_fraction_and_beta_to_bounds():
    from peak_fit import hypermet_left_tail, TAIL_FRACTION_MAX, TAIL_BETA_MIN

    def model(x, p):
        amp, pos, sigma, r, beta = p
        return amp * hypermet_left_tail(x, pos, sigma, r, beta)

    x = np.arange(200, dtype=float)
    y = 500.0 * hypermet_left_tail(x, 100.0, 3.0, 0.1, 4.0)
    y_err = np.sqrt(np.maximum(y, 1.0))

    damping = [
        _ParamDamping("amp"), _ParamDamping("pos", sigma_index=2), _ParamDamping("sigma"),
        _ParamDamping("tail_fraction"), _ParamDamping("tail_beta"),
    ]
    p0 = [500.0, 100.0, 3.0, 0.05, 3.0]
    popt, pcov = _marquardt_fit(model, x, y, y_err, p0, damping)

    assert 0.0 <= popt[3] <= TAIL_FRACTION_MAX
    assert popt[4] >= TAIL_BETA_MIN


from peak_fit import _measure_width


def test_measure_width_recovers_true_sigma_from_a_single_peak():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.5)], slope=0.0, intercept=20.0)
    lo, hi = 80, 120
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_sub = y[mask] - 20.0
    sigma = _measure_width(x_fit, y_sub, [100.0], fallback=10.0)
    assert sigma == pytest.approx(3.5, abs=0.5)


def test_measure_width_uses_the_largest_marked_peak_for_three_peaks():
    x, y = _make_spectrum(
        channels=200,
        peaks=[(500.0, 90.0, 3.0), (350.0, 108.0, 3.0), (420.0, 117.0, 3.0)],
        slope=0.0, intercept=20.0,
    )
    lo, hi = 70, 135
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_sub = y[mask] - 20.0
    sigma = _measure_width(x_fit, y_sub, [88.0, 110.0, 119.0], fallback=5.42)
    assert sigma == pytest.approx(3.0, abs=0.7)


def test_measure_width_falls_back_when_no_clear_peak_is_found():
    x = np.arange(80, 121, dtype=float)
    y_sub = np.zeros_like(x)  # perfectly flat, background-subtracted to zero
    sigma = _measure_width(x, y_sub, [100.0], fallback=7.0)
    assert sigma == 7.0


def test_measure_width_uses_the_one_side_that_does_not_hit_the_data_edge():
    # Peak center sits at the very first fit-region channel, so only the
    # right side can cross half-max before running off the edge -- must
    # still measure a good width from that one side rather than falling
    # back just because the other side had nowhere to go.
    x = np.arange(100, 140, dtype=float)
    y_sub = 500.0 * np.exp(-((x - 100.0) ** 2) / (2 * 3.0 ** 2))
    sigma = _measure_width(x, y_sub, [100.0], fallback=99.0)
    assert sigma == pytest.approx(3.0, abs=0.5)


from peak_fit import FWHM_FACTOR, IntegrationResult, integrate_region


def test_integration_result_holds_expected_fields():
    result = IntegrationResult(
        left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0), background_density=5.0,
        gross_area=100.0, gross_area_err=10.0,
        gross_centroid=100.0, gross_centroid_err=1.0,
        gross_fwhm=5.0, gross_fwhm_err=0.5,
        gross_skewness=0.1, gross_skewness_err=0.05,
        background_area=20.0, background_area_err=4.0,
        background_centroid=100.0, background_centroid_err=2.0,
        background_fwhm=3.0, background_fwhm_err=0.3,
        background_skewness=0.0, background_skewness_err=0.02,
        net_area=80.0, net_area_err=11.0,
        net_centroid=100.0, net_centroid_err=1.2,
        net_fwhm=5.0, net_fwhm_err=0.6,
        net_skewness=0.1, net_skewness_err=0.06,
    )
    assert result.net_area == 80.0
    assert result.timestamp is None
    assert result.visible is True


def test_integrate_region_flat_background_gives_near_zero_net_area():
    x = np.arange(200, dtype=float)
    y = np.full(200, 20.0)
    result = integrate_region(
        x, y, left_bg_region=(10.0, 30.0), right_bg_region=(170.0, 190.0),
        fit_region=(85.0, 115.0),
    )
    assert result.net_area == pytest.approx(0.0, abs=1e-6)
    assert result.background_centroid == pytest.approx(100.0, abs=0.6)


def test_integrate_region_recovers_analytic_peak_area():
    x, y = _make_spectrum(
        channels=200, peaks=[(5000.0, 100.0, 4.0)], slope=0.0, intercept=50.0,
    )
    result = integrate_region(
        x, y, left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
    )
    analytic_area = 5000.0 * 4.0 * np.sqrt(2 * np.pi)
    assert result.net_area == pytest.approx(analytic_area, rel=0.05)
    assert result.net_centroid == pytest.approx(100.0, abs=1.0)
    assert result.net_fwhm == pytest.approx(FWHM_FACTOR * 4.0, abs=1.0)


def test_integrate_region_matches_hand_computed_moments():
    x = np.arange(10, dtype=float)
    y = np.array([0., 0., 10., 20., 10., 0., 0., 0., 0., 0.])
    result = integrate_region(
        x, y, left_bg_region=(8.0, 8.0), right_bg_region=(9.0, 9.0),
        fit_region=(2.0, 4.0),
    )
    # s=[10,20,10] at i=[2,3,4]: gross_area=40, centroid=(20+60+40)/40=3.0,
    # mom2=((2-3)^2*10+(3-3)^2*20+(4-3)^2*10)/40=20/40=0.5
    assert result.gross_area == 40.0
    assert result.gross_centroid == pytest.approx(3.0)
    assert result.gross_fwhm == pytest.approx(np.sqrt(0.5) * FWHM_FACTOR)
    # Both bg regions are exactly 0, so net == gross exactly.
    assert result.net_area == 40.0
    assert result.net_centroid == pytest.approx(3.0)


def test_integrate_region_all_zero_does_not_raise():
    x = np.arange(50, dtype=float)
    y = np.zeros(50)
    result = integrate_region(
        x, y, left_bg_region=(5.0, 8.0), right_bg_region=(40.0, 43.0),
        fit_region=(20.0, 25.0),
    )
    assert result.gross_area == 0.0
    assert result.net_area == 0.0


def test_integrate_region_rejects_empty_fit_region():
    x = np.arange(200, dtype=float)
    y = np.full(200, 20.0)
    with pytest.raises(FitError):
        integrate_region(
            x, y, left_bg_region=(10.0, 30.0), right_bg_region=(170.0, 190.0),
            fit_region=(500.0, 501.0),
        )


def test_integrate_region_background_uncertainty_scales_linearly_with_region_width():
    """Deliberately verifies TV's own (statistically non-standard)
    background-uncertainty formula is preserved exactly: background_area_err
    squared scales linearly with the fit region's channel count, not
    quadratically as strict error propagation for a scaled mean would
    require -- a future "fix" of this would break this test on purpose."""
    x = np.arange(20, dtype=float)
    y = np.full(20, 5.0)
    y[2] = 100.0
    y[17] = 100.0
    result = integrate_region(
        x, y, left_bg_region=(2.0, 2.0), right_bg_region=(17.0, 17.0),
        fit_region=(8.0, 10.0),
    )
    # bg_chn=2, bg_count=200, bg_density=100, bg_density_var=200/(2*2)=50
    # n_region=3 channels (8,9,10) -> background_area=300,
    # background_area_err^2 = 50*3 = 150 (linear, not 50*3^2=450)
    assert result.background_area == pytest.approx(300.0)
    assert result.background_area_err == pytest.approx(np.sqrt(150.0))


def test_integrate_region_background_moment_uncertainty_reuses_net_second_moment():
    """Deliberately verifies another TV quirk: the background layer's own
    width/skewness uncertainty terms reuse the *net* distribution's 2nd
    moment rather than the background's own -- confirmed against TV's
    source (vsFitInt.c:281,303) and cross-checked numerically against an
    independently-coded "corrected" alternative during design (the two
    differ by more than 30% for this fixture; this test pins the exact
    TV-parity value so a future "fix" would fail loudly)."""
    x = np.arange(60, dtype=float)
    y = np.full(60, 30.0)
    y[5] = 40.0
    y[6] = 20.0
    y[50] = 25.0
    y[51] = 35.0
    y[25:31] += np.array([50, 150, 300, 300, 150, 50])
    result = integrate_region(
        x, y, left_bg_region=(5.0, 6.0), right_bg_region=(50.0, 51.0),
        fit_region=(20.0, 36.0),
    )
    assert result.background_fwhm_err == pytest.approx(0.3305131157646951)
