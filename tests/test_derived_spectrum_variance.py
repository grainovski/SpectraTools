"""C3 (v4.0.0): a derived spectrum carries its propagated variance, and
the fitter weights by it instead of assuming Poisson.

The defect these cover: a matrix cut or an Add/Subtract result is not
Poisson in its own counts -- its variance is `A + f**2 * B`, which
exceeds those counts and stays positive where they cancel to zero.
Fitting it with sqrt(max(y, 1.0)) misweights every channel and reports
optimistic parameter errors and chi-square.
"""

import numpy as np
import pytest

from matrix_cut import compute_cut_with_variance
from peak_fit import FitError, fit_peaks
from spectrum import LoadedSpectrum
from spectrum_operations import combined_variance, poisson_variance


def _gaussian_spectrum(length=200, centre=100.0, sigma=4.0, amplitude=800.0, pedestal=50.0):
    x = np.arange(length, dtype=float)
    return pedestal + amplitude * np.exp(-((x - centre) ** 2) / (2 * sigma ** 2))


# --- the variance travels with the spectrum -----------------------------


def test_a_file_backed_spectrum_has_no_variance():
    """None means "assume Poisson", which is right for raw counts. Only
    derived spectra set it, so the default must stay None."""
    spectrum = LoadedSpectrum("some.spe", np.arange(10), "#123456")
    assert spectrum.variance is None


def test_a_loaded_spectrum_carries_a_variance_when_given_one():
    variance = np.full(10, 3.0)
    spectrum = LoadedSpectrum("cut.mtx", np.arange(10), "#123456", variance=variance)
    assert spectrum.variance is variance


def test_poisson_variance_is_the_counts_floored_at_zero():
    assert poisson_variance(np.array([4.0, 0.0, -9.0])) == pytest.approx([4.0, 0.0, 0.0])


def test_combined_variance_adds_and_squares_the_factor():
    a = np.array([100.0, 100.0])
    b = np.array([25.0, 25.0])
    # Both Poisson: 100 + 2**2 * 25 = 200.
    assert combined_variance(a, b, 2.0) == pytest.approx([200.0, 200.0])
    # Explicit variances are used in place of the counts.
    assert combined_variance(a, b, 2.0, np.array([1.0, 1.0]), np.array([1.0, 1.0])) \
        == pytest.approx([5.0, 5.0])


def test_subtracting_a_spectrum_from_itself_leaves_zero_counts_but_real_uncertainty():
    """The case that makes Poisson weighting indefensible: the counts
    cancel exactly, so sqrt(N) claims perfect knowledge of a channel about
    which nothing is known. The variance doubles instead.
    """
    data = np.full(8, 400.0)
    variance = combined_variance(data, data, 1.0)
    assert variance == pytest.approx(np.full(8, 800.0))


# --- the fitter uses it ------------------------------------------------


def test_fit_without_variance_still_assumes_poisson():
    """The fallback has to be untouched -- every file-backed spectrum
    depends on it, and no existing result may move."""
    y = _gaussian_spectrum()
    x = np.arange(len(y), dtype=float)
    result = fit_peaks(x, y, (0, 20), (180, 199), (80, 120), [100.0])
    assert result.peaks[0].position == pytest.approx(100.0, abs=0.5)


def test_a_supplied_variance_changes_the_reported_uncertainties():
    """The point of the whole change. Inflating the variance by a constant
    factor must inflate the fitted parameter errors, and leave the fitted
    VALUES essentially alone -- a uniform reweighting does not move the
    least-squares minimum, it only changes how well determined it is.
    """
    y = _gaussian_spectrum()
    x = np.arange(len(y), dtype=float)

    poisson = fit_peaks(x, y, (0, 20), (180, 199), (80, 120), [100.0])
    inflated = fit_peaks(x, y, (0, 20), (180, 199), (80, 120), [100.0],
                         variance=poisson_variance(y) * 9.0)

    assert inflated.peaks[0].position == pytest.approx(poisson.peaks[0].position, abs=1e-3)
    # 9x the variance is 3x the error on every channel, so every fitted
    # uncertainty scales by 3 while the values stay put.
    assert inflated.peaks[0].position_err == pytest.approx(
        3.0 * poisson.peaks[0].position_err, rel=1e-2
    )
    assert inflated.peaks[0].area_err == pytest.approx(
        3.0 * poisson.peaks[0].area_err, rel=1e-2
    )


def test_a_supplied_variance_is_masked_to_the_fit_region():
    """The variance is spectrum-length; y_err must come from the same
    mask as y_fit. If it were used unmasked the shapes would not even
    broadcast, and if it were mis-sliced the weights would silently
    belong to the wrong channels -- so compare against a fit whose
    variance differs ONLY outside the fit region, which must be
    identical.
    """
    y = _gaussian_spectrum()
    x = np.arange(len(y), dtype=float)
    variance = poisson_variance(y)

    tampered = variance.copy()
    tampered[:40] *= 1000.0     # entirely left of the fit region
    tampered[160:] *= 1000.0    # entirely right of it

    inside_only = fit_peaks(x, y, (0, 20), (180, 199), (80, 120), [100.0], variance=variance)
    tampered_fit = fit_peaks(x, y, (0, 20), (180, 199), (80, 120), [100.0], variance=tampered)

    assert tampered_fit.peaks[0].position == pytest.approx(inside_only.peaks[0].position, abs=1e-9)
    assert tampered_fit.peaks[0].position_err == pytest.approx(
        inside_only.peaks[0].position_err, rel=1e-9
    )


def test_a_mismatched_variance_length_is_rejected():
    y = _gaussian_spectrum()
    x = np.arange(len(y), dtype=float)
    with pytest.raises(FitError, match="variance has length"):
        fit_peaks(x, y, (0, 20), (180, 199), (80, 120), [100.0],
                  variance=np.ones(len(y) - 1))


def test_zero_variance_channels_do_not_take_infinite_weight():
    """A derived spectrum can legitimately have channels with no counts
    and hence zero propagated variance. Left unfloored those would carry
    infinite weight and dominate the fit by themselves.
    """
    y = _gaussian_spectrum()
    x = np.arange(len(y), dtype=float)
    variance = poisson_variance(y)
    variance[90:95] = 0.0

    result = fit_peaks(x, y, (0, 20), (180, 199), (80, 120), [100.0], variance=variance)
    assert np.isfinite(result.peaks[0].position)
    assert np.isfinite(result.peaks[0].position_err)
    assert result.peaks[0].position == pytest.approx(100.0, abs=1.0)


# --- end to end: a cut is fitted with the cut's own variance -----------


def test_a_matrix_cut_produces_a_variance_the_fitter_accepts():
    """Joins the two halves: the variance compute_cut_with_variance
    returns is the right length and shape to hand straight to fit_peaks,
    which is exactly what MatrixPanel._activate_cut now relies on.
    """
    rng = np.random.default_rng(3)
    matrix = rng.integers(0, 20, size=(200, 200)).astype(np.int64)
    # Put a real peak into the gated rows so there is something to fit.
    peak = 600.0 * np.exp(-((np.arange(200) - 120.0) ** 2) / (2 * 5.0 ** 2))
    matrix[40:60, :] += peak.astype(np.int64)

    net, variance = compute_cut_with_variance(matrix, "y", (40, 59), [(100, 119)])
    assert variance.shape == net.shape

    x = np.arange(len(net), dtype=float)
    result = fit_peaks(x, net, (0, 30), (170, 199), (100, 140), [120.0], variance=variance)
    assert result.peaks[0].position == pytest.approx(120.0, abs=1.5)
    assert np.isfinite(result.peaks[0].area_err)

    # And the propagated variance genuinely differs from the Poisson
    # assumption on the same data -- otherwise this test would pass even
    # if the wiring were ignored.
    assert not np.allclose(variance, poisson_variance(net))
