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
    # Three peaks requested at the exact same position, each with its own
    # independent sigma (link_widths=False), start curve_fit with three
    # identical parameter triplets, so their Jacobian columns are identical
    # at every iteration (a perfectly singular Jacobian). MINPACK's
    # Levenberg-Marquardt exhausts its default maxfev before ever breaking
    # that symmetry, so curve_fit raises RuntimeError, which fit_peaks must
    # convert to FitError. (With the default link_widths=True, the shared
    # single sigma removes enough degrees of freedom from this degenerate
    # setup that curve_fit actually settles onto a stationary point along
    # the still-undetermined amplitude/position split, so this test needs
    # link_widths=False to keep exercising the non-convergence path.)
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(
            x, y,
            left_bg_region=(20.0, 35.0),
            right_bg_region=(160.0, 175.0),
            fit_region=(93.0, 107.0),
            peak_positions=[100.0, 100.0, 100.0],
            link_widths=False,
        )
