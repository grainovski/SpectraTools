"""Dividing a spectrum by the efficiency curve, bin by bin."""

import os

import numpy as np
import pytest

from efficiency_apply import apply_efficiency

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")


class _FlatCurve:
    """A stand-in for EfficiencyResult exposing only what apply needs."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def curve(self, grid, model=None):
        grid = np.asarray(grid, dtype=float)
        return np.interp(grid, np.arange(len(self._values), dtype=float),
                         self._values)


class _Calibration:
    """channel -> energy, one keV per channel."""

    def apply(self, channel):
        return np.asarray(channel, dtype=float)


def test_counts_are_divided_bin_by_bin():
    counts = np.array([100.0, 200.0, 300.0, 400.0])
    curve = _FlatCurve([0.5, 0.5, 0.25, 0.25])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts == pytest.approx([200.0, 400.0, 1200.0, 1600.0])
    assert out.zeroed == 0


def test_variance_is_scaled_by_one_over_eff_squared():
    """The user asked for no efficiency uncertainty, so deff does not
    propagate -- but the spectrum's own variance still must, or the stored
    variance stops matching the counts and quietly corrupts any later fit."""
    counts = np.array([100.0, 100.0])
    variance = np.array([100.0, 100.0])
    curve = _FlatCurve([0.5, 0.25])
    out = apply_efficiency(counts, _Calibration(), curve, variance=variance)
    assert out.variance == pytest.approx([400.0, 1600.0])


def test_a_negative_efficiency_zeroes_the_bin():
    counts = np.array([100.0, 100.0, 100.0])
    curve = _FlatCurve([0.5, -0.5, 0.5])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[1] == 0.0
    assert out.zeroed == 1
    assert out.zeroed_nonpositive == 1


def test_a_zero_efficiency_zeroes_the_bin():
    counts = np.array([100.0, 100.0])
    curve = _FlatCurve([0.5, 0.0])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[1] == 0.0
    assert out.zeroed_nonpositive == 1


def test_a_nan_efficiency_zeroes_the_bin():
    """THE case a literal `eff <= 0` implementation fails. Under IEEE
    comparison NaN <= 0 is FALSE, so such a test lets NaN through, the bin
    becomes NaN, and that NaN then spreads into autoscaling, plotting,
    fitting and integration -- far worse than a zero."""
    counts = np.array([100.0, 100.0])
    curve = _FlatCurve([0.5, float("nan")])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[1] == 0.0, "a NaN efficiency slipped through"
    assert out.zeroed_nonfinite == 1


def test_an_infinite_efficiency_zeroes_the_bin():
    counts = np.array([100.0, 100.0])
    curve = _FlatCurve([0.5, float("inf")])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[1] == 0.0
    assert out.zeroed_nonfinite == 1


def test_nothing_non_finite_ever_escapes():
    """The blanket guarantee, over a curve deliberately built to go positive,
    zero, negative, infinite and NaN across its range."""
    counts = np.full(5, 100.0)
    curve = _FlatCurve([1.0, 0.0, -1.0, float("inf"), float("nan")])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert np.all(np.isfinite(out.counts)), out.counts
    assert out.zeroed == 4


def test_the_zeroing_check_can_fail():
    """Control. A correction that returned its input untouched would satisfy
    every finiteness assertion above."""
    counts = np.array([100.0, 100.0])
    curve = _FlatCurve([0.5, 0.5])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts != pytest.approx(counts)


def _real_result(name="demo1.txt"):
    """A genuine EfficiencyResult, not the stub above."""
    from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo

    d = np.loadtxt(os.path.join(FIXTURES, name), ndmin=2)
    N, dN, E, I, dI = d[:, 2], d[:, 3], d[:, 4], d[:, 5], d[:, 6]
    fit = fit_efficiency(E, N, dN, I, dI)
    mc = run_monte_carlo(fit, N, dN, I, dI, iterations=200)
    return EfficiencyResult(fit=fit, mc=mc, model="kfr")


def test_a_real_efficiency_result_plugs_in():
    """The stub above pins the arithmetic; this pins that the real object
    fits the same hole. A signature or return-shape mismatch would satisfy
    every test above and fail the first time a user pressed Apply."""
    from calibration import Calibration

    result = _real_result()
    E = result.fit.E
    slope = (float(E.max()) - float(E.min())) / 511.0
    counts = np.full(512, 1000.0)

    out = apply_efficiency(
        counts, Calibration("linear", float(E.min()), slope), result)

    assert np.all(np.isfinite(out.counts))
    assert out.zeroed == 0, "nothing should be zeroed inside the fitted range"
    # The applied curve is normalised to peak at 1, so dividing by it can
    # only raise counts -- never lower them.
    assert np.all(out.counts >= 1000.0 - 1e-9)


def test_a_negative_calibration_offset_zeroes_those_bins():
    """Real calibrations routinely have a negative offset, so channel 0 maps
    to a negative energy. Both models fail there, and each fails a DIFFERENT
    way -- which is exactly why the rule is "finite and > 0" rather than a
    literal `eff <= 0`:

        KFR      (aE + b/E)exp(...) stays finite and goes <= 0
        Radware  takes ln(E) and returns NaN

    So this one ordinary case exercises both branches of the rule with real
    curves rather than a hand-built stub.
    """
    from calibration import Calibration

    result = _real_result()
    counts = np.full(64, 500.0)
    # -20 keV at channel 0, reaching 400 keV at the top.
    out = apply_efficiency(
        counts, Calibration("linear", -20.0, 420.0 / 63.0), result)

    assert np.all(np.isfinite(out.counts)), "a bad energy produced a bad count"
    assert out.zeroed > 0, "negative energies were silently corrected"
    assert out.zeroed_nonpositive > 0, "expected KFR's finite <= 0 branch"

    result.model = "rw"
    out_rw = apply_efficiency(
        counts, Calibration("linear", -20.0, 420.0 / 63.0), result)
    assert np.all(np.isfinite(out_rw.counts))
    assert out_rw.zeroed_nonfinite > 0, "expected Radware's NaN branch"


def test_the_original_counts_are_not_modified():
    counts = np.array([100.0, 200.0])
    original = counts.copy()
    apply_efficiency(counts, _Calibration(), _FlatCurve([0.5, 0.5]))
    assert counts == pytest.approx(original)
