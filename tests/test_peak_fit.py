import math

import numpy as np
import pytest

from peak_fit import FitError, FitResult, PeakResult, compute_background, hypermet_left_tail


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
    slope, intercept = compute_background(x, y, (10.0, 20.0), (150.0, 160.0))
    assert slope == pytest.approx(0.0, abs=1e-9)
    assert intercept == pytest.approx(15.0, abs=1e-9)


def test_compute_background_sloped():
    x = np.arange(200, dtype=float)
    y = 0.5 * x + 3.0
    slope, intercept = compute_background(x, y, (10.0, 20.0), (150.0, 160.0))
    assert slope == pytest.approx(0.5, abs=1e-6)
    assert intercept == pytest.approx(3.0, abs=1e-4)


def test_compute_background_rejects_empty_region():
    x = np.arange(200, dtype=float)
    y = np.full(200, 15.0)
    with pytest.raises(FitError):
        compute_background(x, y, (500.0, 501.0), (150.0, 160.0))


def test_compute_background_rejects_identical_mean_x():
    x = np.arange(200, dtype=float)
    y = np.full(200, 15.0)
    with pytest.raises(FitError):
        compute_background(x, y, (10.0, 20.0), (10.0, 20.0))


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


def test_fit_peaks_reports_full_and_net_region_areas():
    """FitResult.gross_area is the raw (background-included) count total
    over the fit region, independent of how well the fit converges; for a
    single peak, net_area is just that peak's own (background-excluded)
    area, since it's the only fitted component."""
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
    mask = (x >= 85.0) & (x <= 115.0)
    expected_gross = float(np.sum(y[mask]))
    assert result.gross_area == pytest.approx(expected_gross)
    assert result.gross_area_err == pytest.approx(np.sqrt(expected_gross))
    assert len(result.peaks) == 1
    assert result.net_area == pytest.approx(result.peaks[0].area)
    assert result.net_area_err == pytest.approx(result.peaks[0].area_err)
    # The background under the peak is real, non-zero signal in the raw
    # counts, so the full (gross) total must exceed the net (peak-only)
    # total by roughly that background contribution.
    assert result.gross_area > result.net_area


def test_fit_peaks_gross_area_err_is_nan_not_crash_for_negative_region():
    """fit_peaks' own gross_area_err sibling to the integrate_region fix:
    gross_area is the raw (background-still-included) sum over the fit
    region. np.sqrt() on a negative scalar doesn't crash like math.sqrt
    does -- it silently produces NaN, matching TV's own lack of a guard
    here. Deliberately different resolution from integrate_region()'s
    FitError: gross_area_err is one auxiliary summary field on an
    otherwise-valid, already-successful fit (the fit itself doesn't
    care about the sign of y_sub), so this fit is allowed to succeed
    with a NaN in that one field rather than being rejected outright."""
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=-50.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(0.0, 5.0),
        right_bg_region=(195.0, 199.0),
        fit_region=(10.0, 190.0),
        peak_positions=[100.0],
    )
    assert result.gross_area < 0.0  # sanity: this really exercises the negative branch
    assert math.isnan(result.gross_area_err)
    # The rest of the fit is unaffected -- a real, well-defined peak was
    # still found and reported normally.
    assert len(result.peaks) == 1
    assert math.isfinite(result.peaks[0].area)


def test_fit_peaks_reports_full_area_per_peak():
    """Each peak's own full_area is its net area plus the local linear
    background (evaluated at the peak's own center) integrated over that
    peak's own FWHM -- the standard gamma-spectroscopy way to split a
    shared background back out per peak. full_area_err equals area_err
    since the background term is a deterministic offset with no
    uncertainty of its own."""
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
    background_under_peak = (
        result.background_slope * peak.position + result.background_intercept
    ) * peak.fwhm
    assert peak.full_area == pytest.approx(peak.area + background_under_peak)
    assert peak.full_area_err == pytest.approx(peak.area_err)


def test_fit_peaks_reports_reduced_chi2():
    """A noiseless single-peak fit should recover chi^2 essentially at
    zero (the model reproduces the data almost exactly); a Poisson-noisy
    fit of a well-specified model should land near reduced chi^2 ~ 1, the
    textbook signature of a good fit under correct weighting."""
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    noiseless = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0), peak_positions=[100.0],
    )
    assert noiseless.reduced_chi2 == pytest.approx(0.0, abs=1e-6)

    x2, y2 = _make_spectrum(
        channels=200, peaks=[(2000.0, 100.0, 4.0)], slope=0.0, intercept=50.0,
        noise_seed=42,
    )
    noisy = fit_peaks(
        x2, y2,
        left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0), peak_positions=[100.0],
    )
    assert noisy.reduced_chi2 == pytest.approx(1.0, abs=0.5)


def test_fit_peaks_reduced_chi2_is_actually_divided_by_degrees_of_freedom():
    """Direct proof that FitResult.reduced_chi2 is really reduced (raw
    chi^2 / degrees of freedom), not just raw chi^2 under another name:
    independently reconstructs the model from the fitted peak's own
    values (no call into any peak_fit.py internals) and hand-computes
    both raw chi^2 and degrees of freedom (data points minus free
    parameters, matching TV's own CurFreedom convention) separately."""
    from peak_fit import parameter_names

    x, y = _make_spectrum(
        channels=200, peaks=[(2000.0, 100.0, 4.0)], slope=0.0, intercept=50.0,
        noise_seed=42,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0), peak_positions=[100.0],
    )
    peak = result.peaks[0]

    mask = (x >= 85.0) & (x <= 115.0)
    x_fit, y_fit = x[mask], y[mask]
    y_err = np.sqrt(np.maximum(y_fit, 1.0))
    model = (
        result.background_slope * x_fit + result.background_intercept
        + peak.amplitude * np.exp(-((x_fit - peak.position) ** 2) / (2 * peak.sigma ** 2))
    )
    raw_chi2 = float(np.sum(((y_fit - model) / y_err) ** 2))
    n_free_params = len(parameter_names(1, link_widths=True, enable_left_tail=False))
    dof = x_fit.size - n_free_params

    assert dof > 0
    assert raw_chi2 / dof != pytest.approx(raw_chi2)  # sanity: dof != 1 for this fixture
    assert result.reduced_chi2 == pytest.approx(raw_chi2 / dof)


def test_fit_peaks_reduced_chi2_is_none_at_zero_degrees_of_freedom():
    """When the fit region has exactly as many data points as free
    parameters, degrees of freedom is zero -- reduced_chi2 must be None
    (undefined), not a divide-by-zero crash or a silently wrong value.
    Fixing sigma leaves amp_0/pos_0 free (2 params); a 2-channel fit
    region gives exactly 2 data points."""
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(100.0, 101.0), peak_positions=[100.0],
        fixed_params={"sigma": 3.0},
    )
    assert result.reduced_chi2 is None


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
    # net_area sums across both peaks, not just the first.
    assert result.net_area == pytest.approx(sum(p.area for p in result.peaks))
    assert result.net_area_err == pytest.approx(
        np.sqrt(sum(p.area_err ** 2 for p in result.peaks))
    )


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


def test_marquardt_does_not_recompute_jacobian_for_rejected_trials():
    """Task 7 (Important 2) regression guard for the audit's measured
    12.4x evaluation blowup on multi-peak fits: the full numeric
    central-difference Jacobian costs 2*n_params model evaluations, but
    a rejected trial only ever needs the scalar measure for its
    accept/reject test -- a rejected trial's Jacobian is thrown away
    immediately and never feeds the next iteration, so computing it is
    pure waste. An 8-peak, independent-width (24 free parameter)
    overlapping multiplet with a modestly-off initial guess forces many
    lambda-growth (rejected-trial) rounds -- this exact fixture produces
    68 evaluation-events (1 initial + 16 accepted + 50 rejected + 1
    give-up half-step) before the fix, i.e. 68 * (2*24 + 1) = 3332 model
    calls. The ceiling below is a generous 60% of that measured pre-fix
    count -- comfortably clears if
    rejected trials stop paying for a Jacobian, comfortably fails if they
    don't."""
    n_peaks = 8

    def raw_model(x, p):
        total = np.zeros_like(x)
        for i in range(n_peaks):
            amp, pos, sigma = p[3 * i], p[3 * i + 1], p[3 * i + 2]
            total = total + amp * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))
        return total

    call_count = {"n": 0}

    def counting_model(xx, p):
        call_count["n"] += 1
        return raw_model(xx, p)

    x = np.linspace(0.0, 100.0, 400)
    true_sigma = 3.0
    spacing = 2.9 * true_sigma  # overlapping, not fully resolved peaks
    true_positions = [50.0 + (i - (n_peaks - 1) / 2.0) * spacing for i in range(n_peaks)]
    true_p = []
    for pos in true_positions:
        true_p += [500.0, pos, true_sigma]
    y = raw_model(x, true_p)
    y_err = np.sqrt(np.maximum(y, 1.0))

    # A modestly (not wildly) bad initial guess -- close enough to
    # converge in 17 outer iterations, well under _CUR_MAX_ITERATIONS,
    # but off enough that many individual steps need lambda-growth
    # rounds first. Found by direct sweep against this exact fixture,
    # not guessed.
    p0 = []
    for pos in true_positions:
        p0 += [450.0, pos + 1.3, true_sigma * 1.2]

    damping = []
    for i in range(n_peaks):
        sigma_index = 3 * i + 2
        damping += [
            _ParamDamping("amp"),
            _ParamDamping("pos", sigma_index=sigma_index),
            _ParamDamping("sigma"),
        ]

    popt, pcov = _marquardt_fit(
        counting_model, x, y, y_err, p0, damping, fit_region_bounds=(0.0, 100.0)
    )

    assert np.all(np.isfinite(popt))  # sanity: this really is a working fit, not a crash
    assert call_count["n"] < 3332 * 0.6


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


def test_integration_result_has_background_true_when_bg_regions_set():
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
    assert result.has_background is True


def test_integration_result_has_background_false_when_bg_regions_none():
    result = IntegrationResult(
        left_bg_region=None, right_bg_region=None,
        fit_region=(90.0, 110.0), background_density=0.0,
        gross_area=100.0, gross_area_err=10.0,
        gross_centroid=100.0, gross_centroid_err=1.0,
        gross_fwhm=5.0, gross_fwhm_err=0.5,
        gross_skewness=0.1, gross_skewness_err=0.05,
        background_area=0.0, background_area_err=0.0,
        background_centroid=0.0, background_centroid_err=0.0,
        background_fwhm=0.0, background_fwhm_err=0.0,
        background_skewness=0.0, background_skewness_err=0.0,
        net_area=100.0, net_area_err=10.0,
        net_centroid=100.0, net_centroid_err=1.0,
        net_fwhm=5.0, net_fwhm_err=0.5,
        net_skewness=0.1, net_skewness_err=0.05,
    )
    assert result.has_background is False


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


def test_integrate_region_without_background_omits_bg_regions_and_zeroes_density():
    x, y = _make_spectrum(
        channels=200, peaks=[(5000.0, 100.0, 4.0)], slope=0.0, intercept=50.0,
    )
    result = integrate_region(
        x, y, left_bg_region=None, right_bg_region=None,
        fit_region=(85.0, 115.0),
    )
    assert result.left_bg_region is None
    assert result.right_bg_region is None
    assert result.background_density == 0.0
    assert result.background_area == 0.0
    assert result.has_background is False


def test_integrate_region_without_background_makes_net_equal_gross():
    x, y = _make_spectrum(
        channels=200, peaks=[(5000.0, 100.0, 4.0)], slope=0.0, intercept=50.0,
    )
    result = integrate_region(
        x, y, left_bg_region=None, right_bg_region=None,
        fit_region=(85.0, 115.0),
    )
    # No background to subtract -- net reduces to gross exactly (both the
    # sum and, for this fixture's positive gross sum, every moment too).
    assert result.net_area == result.gross_area
    assert result.net_centroid == result.gross_centroid
    assert result.net_fwhm == result.gross_fwhm


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
    FWHM uncertainty term (background_fwhm_err) reuses the *net*
    distribution's 2nd moment rather than the background's own --
    confirmed against TV's source (vsFitInt.c:281) and cross-checked
    numerically against an independently-coded "corrected" alternative
    during design (the two differ by more than 30% for this fixture;
    this test pins the exact TV-parity value so a future "fix" would
    fail loudly). NOT the same quirk as background_skewness_err, a
    DIFFERENT term (vsFitInt.c:303) that turned out to use the
    background's own 2nd moment after all -- see
    test_background_skewness_err_uses_backgrounds_own_m2_not_nets."""
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


def _hand_compute_background_skewness_err(x, y, left_bg_region, right_bg_region, fit_region):
    """Independent re-derivation of integrate_region()'s background
    3rd-moment uncertainty (background_skewness_err), NOT a call into
    integrate_region() itself. Mirrors that function's own mask
    construction, bg_density/bg_M1/dltb/dltb2 machinery exactly, but
    deliberately uses the background's OWN 2nd moment (bg_M2) in the
    `termb` line -- matching tv-1.9.13/lib/tv/vsFitInt.c:303 (`dltb =
    dltb * (dltb2 - 3.0 * bgMom2) - bgMom3;`), unlike the net-reusing
    `n_M2` the pre-fix production code used there. If this helper's
    result matches integrate_region()'s, the production code must
    really be using bg_M2 (not n_M2) in that term."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lo, hi = fit_region
    mask = (x >= lo) & (x <= hi)
    idx = x[mask]
    s = y[mask]
    n = idx.size

    bg_chn = 0
    bg_count = 0.0
    bg_dcount = 0.0
    for region in (left_bg_region, right_bg_region):
        blo, bhi = region
        bmask = (x >= blo) & (x <= bhi)
        bg_chn += int(np.sum(bmask))
        bg_y = y[bmask]
        bg_count += float(np.sum(bg_y))
        bg_dcount += float(np.sum(bg_y))

    bg_density = bg_count / bg_chn
    bg_density_var = bg_dcount / (bg_chn * bg_chn)
    background_area = bg_density * n

    b = bg_density
    db = bg_density_var
    bg_sum = background_area

    bgmom1_raw = float(np.sum(idx * b))
    bg_M1 = bgmom1_raw / abs(bg_sum)

    dltb = idx - bg_M1
    dltb2 = dltb ** 2
    bgmom2_raw = float(np.sum(dltb2 * b))
    bg_M2 = bgmom2_raw / abs(bg_sum)  # background's OWN 2nd moment
    bgmom3_raw = float(np.sum((dltb2 * dltb) * b))
    bg_M3 = bgmom3_raw / abs(bg_sum)

    termb = dltb * (dltb2 - 3.0 * bg_M2) - bg_M3  # bg_M2, not n_M2 -- the fix
    dBgMom3 = float(np.sum((termb ** 2) * db))
    bg_DM3 = math.sqrt(dBgMom3) / abs(bg_sum)
    return bg_DM3  # == background_skewness_err (DM3 passes through _to_reported unchanged)


def test_background_skewness_err_uses_backgrounds_own_m2_not_nets():
    # Constructed so bg_M2 and n_M2 are numerically different -- an
    # asymmetric background window vs. a peak-dominated net region --
    # so the bug (using n_M2 in termb) produces a different, wrong
    # background_skewness_err than the fix (using bg_M2). Verified
    # numerically before writing this test: bg_M2=850.0 vs n_M2=33.25
    # for this exact fixture (a 96% relative difference), producing a
    # ~29% difference in the final background_skewness_err between the
    # buggy and fixed formula -- comfortably far from a coincidental
    # match.
    x = np.arange(300, dtype=float)
    y = np.full(300, 3.0)
    y[140:160] += 100.0  # a sharp peak, skews the NET moments a lot
    result = integrate_region(x, y, (10, 15), (280, 295), (100, 200))
    expected = _hand_compute_background_skewness_err(x, y, (10, 15), (280, 295), (100, 200))
    assert result.background_skewness_err == pytest.approx(expected, rel=1e-9)


def test_integrate_region_net_second_moment_uses_abs_net_area():
    """Regression test for a real port bug (found and fixed 2026-07-17):
    TV's own source (vsFitInt.c:264) normalizes the net layer's second
    moment by ABS(sum), not plain sum, like every other bg/net moment in
    that block -- only the gross layer's M1/M2 use a plain (non-abs) sum
    (vsFitInt.c:54,62). A prior version of this port copied the gross
    layer's plain-sum convention into the net layer's M2 by mistake. That
    only diverges from TV when net_area goes negative (background
    estimate exceeding gross counts -- a normal outcome for a weak or
    absent peak), where it silently flips the *sign* of the reported
    net_fwhm. This fixture forces net_area negative and pins the
    TV-correct (negative) net_fwhm/net_fwhm_err, so a regression back to
    the plain-sum formula fails loudly instead of silently reporting the
    wrong sign."""
    x = np.arange(30, dtype=float)
    y = np.full(30, 5.0)
    y[2] = 100.0
    y[27] = 100.0
    result = integrate_region(
        x, y, left_bg_region=(2.0, 2.0), right_bg_region=(27.0, 27.0),
        fit_region=(13.0, 16.0),
    )
    # bg_density=100 pooled from the two single-channel bg regions, far
    # above the flat 5.0/channel in the fit region -> net_area =
    # gross_area(20) - background_area(100*4=400) = -380 (negative,
    # the case that exercises the abs-vs-plain divergence).
    assert result.net_area == pytest.approx(-380.0)
    assert result.net_fwhm == pytest.approx(-68.34051289398487)
    assert result.net_fwhm_err == pytest.approx(5.338971539536898)


def test_integrate_region_raises_clear_error_on_negative_counts_in_fit_region():
    # Simulates a Subtract-Spectra-derived spectrum: unclamped negative
    # counts in the fit region itself. TV's own uncertainty formulas have
    # no guard for this case (confirmed directly against vsFitInt.c --
    # ABS() there wraps only the sum/divisor, never the sqrt argument),
    # so rather than manufacturing a number TV was never designed to
    # produce, this is rejected outright with a clear, actionable message
    # -- matching this file's own existing precedent of raising FitError
    # for an analogous negative-covariance case (see the pcov check
    # above in fit_peaks).
    #
    # Background regions are kept entirely positive here (unlike the fit
    # region) so this test genuinely isolates the fit-region check from
    # its sibling test below -- a regression that dropped the fit-region
    # check but kept the background one must NOT be able to pass this
    # test by accident.
    x = np.arange(200, dtype=float)
    y = np.full(200, 5.0)  # positive baseline everywhere...
    y[60:140] = -5.0  # ...except made negative specifically inside the fit region...
    y[90:110] += 40.0  # ...with a "peak" sitting on top of that negative background
    with pytest.raises(FitError, match="negative counts"):
        integrate_region(x, y, (150, 155), (160, 165), (60, 140))


def test_integrate_region_raises_clear_error_on_negative_counts_in_background_region():
    # Same rejection, but triggered by a background region alone -- the
    # fit region itself is entirely non-negative here, proving the check
    # on the background data path is independent of the one on the fit
    # region.
    x = np.arange(200, dtype=float)
    y = np.full(200, 5.0)
    y[90:110] += 40.0
    y[150:155] = -3.0  # only the left background region goes negative
    with pytest.raises(FitError, match="negative counts"):
        integrate_region(x, y, (150, 155), (170, 175), (60, 140))


def test_integrate_region_positive_counts_unaffected_by_negative_count_guard():
    # Regression guard: for ordinary non-negative data (the overwhelming
    # majority of real usage), the new negative-count check must never
    # fire -- same result as before this fix existed.
    x = np.linspace(0, 200, 201)
    y = np.zeros(201)
    y[90:110] = 40.0
    y += 5.0
    result = integrate_region(x, y, (10, 30), (150, 170), (60, 140))
    assert result.gross_area_err == pytest.approx(math.sqrt(sum(y[(x >= 60) & (x <= 140)])))


# --- channel_indices caching (v3.1.0 audit, Minor) ---------------------


def test_channel_indices_returns_the_channel_axis():
    from peak_fit import channel_indices

    assert list(channel_indices(5)) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_channel_indices_reuses_one_array_per_length():
    from peak_fit import channel_indices

    assert channel_indices(4096) is channel_indices(4096)
    assert channel_indices(4096) is not channel_indices(2048)


def test_channel_indices_is_read_only():
    # The array is shared between every caller of the same length, so a
    # caller mutating it would corrupt the axis for all the others. Must
    # fail loudly rather than silently.
    import numpy as np
    import pytest as _pytest

    from peak_fit import channel_indices

    axis = channel_indices(16)
    with _pytest.raises(ValueError):
        axis[0] = 99.0
    assert axis[0] == 0.0
    assert np.array_equal(channel_indices(16), np.arange(16, dtype=float))


def test_channel_indices_is_unaffected_by_in_place_data_mutation():
    # The reason caching is safe at all: the channel axis depends only on
    # the channel count, so mutating a spectrum's counts in place (what
    # Multiply and Add/Subtract do) cannot make a cached axis stale.
    import numpy as np

    from peak_fit import channel_indices

    data = np.array([1, 2, 3, 4], dtype=np.int64)
    before = list(channel_indices(len(data)))
    data *= 1000
    assert list(channel_indices(len(data))) == before


# --- TV parity: initial width seed (v3.1.0 audit, M1) ------------------


def test_initial_sigma_seed_is_half_the_measured_width_matching_tv():
    # TV's FSInitWidth (vsFitSetup.c:384) sets width = w * FWHM_TO_SIGMA
    # with w a HALF-width, so its seed is half the sigma the measured
    # FWHM implies. This port previously seeded the full sigma (2x TV).
    from peak_fit import _initial_guess, _measure_width, parameter_names

    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.5)], slope=0.0, intercept=20.0)
    lo, hi = 80, 120
    mask = (x >= lo) & (x <= hi)
    x_fit, y_sub = x[mask], y[mask] - 20.0

    measured = _measure_width(x_fit, y_sub, [100.0], fallback=10.0)
    names = parameter_names(1, link_widths=True, enable_left_tail=False)
    guess = _initial_guess(names, x_fit, y_sub, (lo, hi), [100.0],
                           link_widths=True, enable_left_tail=False)

    seeded_sigma = guess[names.index("sigma")]
    assert seeded_sigma == pytest.approx(measured / 2.0)
    # ... and the measurement itself still reports the real width.
    assert measured == pytest.approx(3.5, abs=0.5)


def test_tail_beta_seed_uses_the_measured_width_not_the_halved_seed():
    # TV's halving is specific to its width parameter; nothing in
    # vsFitSetup.c propagates it to the tail decay length.
    from peak_fit import TAIL_BETA_MIN, _initial_guess, _measure_width, parameter_names

    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.5)], slope=0.0, intercept=20.0)
    lo, hi = 80, 120
    mask = (x >= lo) & (x <= hi)
    x_fit, y_sub = x[mask], y[mask] - 20.0

    measured = _measure_width(x_fit, y_sub, [100.0], fallback=10.0)
    names = parameter_names(1, link_widths=True, enable_left_tail=True)
    guess = _initial_guess(names, x_fit, y_sub, (lo, hi), [100.0],
                           link_widths=True, enable_left_tail=True)

    assert guess[names.index("tail_beta")] == pytest.approx(max(measured, TAIL_BETA_MIN))
    assert guess[names.index("sigma")] == pytest.approx(measured / 2.0)


def test_initial_guess_falls_back_without_halving_when_width_cannot_be_measured():
    # The fallback heuristic is this port's own, not TV's, so the
    # halving must not be applied to it.
    from peak_fit import _initial_guess, parameter_names

    x_fit = np.arange(80, 121, dtype=float)
    y_sub = np.zeros_like(x_fit)  # perfectly flat -> no measurable peak
    names = parameter_names(1, link_widths=True, enable_left_tail=False)
    guess = _initial_guess(names, x_fit, y_sub, (80, 120), [100.0],
                           link_widths=True, enable_left_tail=False)

    expected_fallback = max((120 - 80) / 4.0, 1e-6)
    assert guess[names.index("sigma")] == pytest.approx(expected_fallback)


# --- TV parity: Marquardt half-step fallback (v3.1.0 audit, M2) --------


def test_reversed_half_step_is_re_damped_and_stays_inside_the_fit_region():
    # TV's lambda-runaway fallback (vsCurFit.c:130-134) negates and
    # halves the step and then pushes it back through the SAME
    # CurChangeTry damping pass the forward step used. This port applied
    # the reversed step raw. Halving alone can only shrink a step, but
    # NEGATING it can break a constraint the forward step honoured: the
    # position rule keeps a peak inside the marked region, and
    # _clamp_trial does not cover positions at all -- only sigma and the
    # tail parameters.
    from peak_fit import _apply_step_damping, _build_param_damping, _clamp_trial

    names = ["amp_0", "pos_0", "sigma"]
    damping = _build_param_damping(names, {}, link_widths=True)
    region = (10.0, 60.0)
    p = np.array([100.0, 10.5, 3.0])  # position just inside the low edge

    forward = _apply_step_damping(p, np.array([0.0, 2.0, 0.0]), damping, region)

    # What the old code did: reverse the damped step, value-clamp only.
    undamped = _clamp_trial(p - 0.5 * forward, damping)
    assert undamped[1] < region[0], "precondition: the raw reversal leaves the region"

    # What TV does, and what the fix does: re-damp the reversed step.
    back = _apply_step_damping(p, -0.5 * forward, damping, region)
    redamped = _clamp_trial(p + back, damping)
    assert region[0] <= redamped[1] <= region[1]


def test_fits_never_place_a_peak_outside_the_marked_region():
    # End-to-end guard for the same property: whichever path the solver
    # exits through, including the give-up fallback, a fitted position
    # must lie inside the region the user marked.
    rng = np.random.default_rng(11)
    for trial in range(25):
        sigma = float(rng.uniform(1.5, 6.0))
        pos = float(rng.uniform(95.0, 105.0))
        x = np.arange(200, dtype=float)
        y = 30.0 + 400.0 * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))
        y = rng.poisson(np.maximum(y, 0)).astype(np.int64)
        lo, hi = 80.0, 120.0
        try:
            result = fit_peaks(x, y, (60, 78), (122, 140), (lo, hi), [pos], link_widths=True)
        except FitError:
            continue
        for peak in result.peaks:
            assert lo <= peak.position <= hi, (
                f"trial {trial}: fitted position {peak.position} escaped region ({lo}, {hi})"
            )


# --- deliberate divergence from gf3: the tail is never disabled -------


def test_tail_stays_active_where_gf3_would_disable_it():
    """gf3 switches to a pure Gaussian once y = sigma/(beta*sqrt(2)) > 4
    (its `notail` flag, gf3_subs.c:2896-2898) -- a float32-era numerical
    safeguard. This port keeps computing the true Hypermet function
    there, which is a DELIBERATE, user-confirmed divergence (2026-08-16),
    not an oversight. Locked in by this test so a future parity pass
    cannot quietly "restore" gf3's behaviour and change fitted results.
    """
    sigma, beta, r = 3.0, 0.2, 0.3
    y = sigma / (beta * np.sqrt(2))
    assert y > 4.0, "precondition: this is the regime gf3 would disable the tail in"

    x = np.linspace(-40.0, 40.0, 2001)
    shape = hypermet_left_tail(x, 0.0, sigma, r, beta)
    pure_gaussian = np.exp(-((x / (sigma * np.sqrt(2))) ** 2))

    # If the tail were disabled, `shape` would BE the pure Gaussian.
    assert np.max(np.abs(shape - pure_gaussian)) > 0.05

    # And it must still lean left -- the physical point of the tail.
    left = shape[x < 0].sum()
    right = shape[x > 0].sum()
    assert left > right


def test_tail_and_gaussian_agree_when_gf3_would_also_keep_the_tail():
    """Guard from the other side: for a long tail (y well under 4, where
    gf3 keeps its tail too) the shape must still be a real tail, so the
    test above is checking the regime and not just any old difference."""
    sigma, beta, r = 3.0, 10.0, 0.3
    assert sigma / (beta * np.sqrt(2)) < 4.0

    x = np.linspace(-40.0, 40.0, 2001)
    shape = hypermet_left_tail(x, 0.0, sigma, r, beta)
    assert shape[x < 0].sum() > shape[x > 0].sum()


def test_singular_tail_covariance_still_suggests_unchecking_left_tail():
    """A fit whose tail parameters the data cannot constrain fails with a
    SINGULAR covariance (np.linalg.inv raises -> pcov all-inf), not the
    negative-variance case. Both are the same underlying situation -- once
    tail_fraction is driven to ~0 the tail contributes nothing, so
    d(model)/d(tail_beta) vanishes and J.T @ J loses rank -- so both must
    carry the actionable hint. Previously only the negative-variance
    branch did, leaving this one saying just "non-finite covariance
    matrix".
    """
    rng = np.random.default_rng(5)
    messages = []
    for sigma, beta in [(1.5, 0.3), (1.5, 1.0), (1.5, 4.0), (3.0, 0.3), (3.0, 1.0),
                        (3.0, 4.0), (5.0, 0.3), (5.0, 1.0), (5.0, 4.0)]:
        x = np.arange(300, dtype=float)
        y = 30.0 + 800.0 * hypermet_left_tail(x, 150.0, sigma, 0.2, beta)
        y = rng.poisson(np.maximum(y, 0)).astype(np.int64)
        try:
            fit_peaks(x, y, (100, 120), (180, 200), (120, 180), [150.0],
                      link_widths=True, enable_left_tail=True)
        except FitError as exc:
            messages.append(str(exc))

    assert messages, "expected at least one tail-parameter fit failure in this sweep"
    singular = [m for m in messages if "non-finite covariance" in m]
    assert singular, "expected the singular-covariance branch specifically"
    for message in singular:
        assert "Left tail" in message


def test_no_tail_hint_when_the_left_tail_is_disabled():
    # The hint must not appear for fits that never enabled the tail --
    # it would be actively misleading.
    from peak_fit import _marquardt_fit

    def singular_fit(model, x, y, y_err, p0, damping, fit_region_bounds=None):
        return np.asarray(p0, dtype=float), np.full((len(p0), len(p0)), np.inf)

    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    import peak_fit as pf

    pf._marquardt_fit = singular_fit
    try:
        with pytest.raises(FitError) as excinfo:
            fit_peaks(x, y, (60, 78), (122, 140), (80, 120), [100.0],
                      link_widths=True, enable_left_tail=False)
        assert "Left tail" not in str(excinfo.value)
    finally:
        pf._marquardt_fit = _marquardt_fit
