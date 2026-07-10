import numpy as np
import pytest

from peak_fit import FitError, FitResult, PeakResult, _compute_background


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
    # Three peaks requested at the exact same position start curve_fit with
    # three identical parameter triplets, so their Jacobian columns are
    # identical at every iteration (a perfectly singular Jacobian). MINPACK's
    # Levenberg-Marquardt exhausts its default maxfev before ever breaking
    # that symmetry, so curve_fit raises RuntimeError, which fit_peaks must
    # convert to FitError.
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(
            x, y,
            left_bg_region=(20.0, 35.0),
            right_bg_region=(160.0, 175.0),
            fit_region=(93.0, 107.0),
            peak_positions=[100.0, 100.0, 100.0],
        )
