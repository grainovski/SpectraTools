"""Dividing a spectrum by the efficiency curve, bin by bin."""

import os

import numpy as np
import pytest

from efficiency_apply import ZEROED_LOW_CHANNELS, apply_efficiency

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")

LOW = ZEROED_LOW_CHANNELS


def _pad(values, fill=1.0):
    """Prepend the unconditionally-zeroed low band.

    apply_efficiency zeroes the first ZEROED_LOW_CHANNELS bins whatever
    their efficiency evaluates to, so a four-bin fixture would come back
    entirely zeroed and prove nothing about the arithmetic. Padding moves
    the bins a test cares about above the band; they are then read back at
    [LOW:] rather than from zero.
    """
    return np.concatenate([np.full(LOW, fill, dtype=float),
                           np.asarray(values, dtype=float)])


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
    counts = _pad([100.0, 200.0, 300.0, 400.0], 100.0)
    curve = _FlatCurve(_pad([0.5, 0.5, 0.25, 0.25], 0.5))
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[LOW:] == pytest.approx([200.0, 400.0, 1200.0, 1600.0])
    assert out.zeroed == LOW


def test_variance_is_scaled_by_one_over_eff_squared():
    """The user asked for no efficiency uncertainty, so deff does not
    propagate -- but the spectrum's own variance still must, or the stored
    variance stops matching the counts and quietly corrupts any later fit."""
    counts = _pad([100.0, 100.0], 100.0)
    variance = _pad([100.0, 100.0], 100.0)
    curve = _FlatCurve(_pad([0.5, 0.25], 0.5))
    out = apply_efficiency(counts, _Calibration(), curve, variance=variance)
    assert out.variance[LOW:] == pytest.approx([400.0, 1600.0])
    assert out.variance[:LOW] == pytest.approx(np.zeros(LOW))


def test_a_negative_efficiency_zeroes_the_bin():
    counts = _pad([100.0, 100.0, 100.0], 100.0)
    curve = _FlatCurve(_pad([0.5, -0.5, 0.5], 0.5))
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[LOW + 1] == 0.0
    assert out.zeroed == LOW + 1
    assert out.zeroed_nonpositive == 1


def test_a_zero_efficiency_zeroes_the_bin():
    counts = _pad([100.0, 100.0], 100.0)
    curve = _FlatCurve(_pad([0.5, 0.0], 0.5))
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[LOW + 1] == 0.0
    assert out.zeroed_nonpositive == 1


def test_a_nan_efficiency_zeroes_the_bin():
    """THE case a literal `eff <= 0` implementation fails. Under IEEE
    comparison NaN <= 0 is FALSE, so such a test lets NaN through, the bin
    becomes NaN, and that NaN then spreads into autoscaling, plotting,
    fitting and integration -- far worse than a zero."""
    counts = _pad([100.0, 100.0], 100.0)
    curve = _FlatCurve(_pad([0.5, float("nan")], 0.5))
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[LOW + 1] == 0.0, "a NaN efficiency slipped through"
    assert out.zeroed_nonfinite == 1


def test_an_infinite_efficiency_zeroes_the_bin():
    counts = _pad([100.0, 100.0], 100.0)
    curve = _FlatCurve(_pad([0.5, float("inf")], 0.5))
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[LOW + 1] == 0.0
    assert out.zeroed_nonfinite == 1


def test_nothing_non_finite_ever_escapes():
    """The blanket guarantee, over a curve deliberately built to go positive,
    zero, negative, infinite and NaN across its range."""
    counts = _pad(np.full(5, 100.0), 100.0)
    curve = _FlatCurve(_pad([1.0, 0.0, -1.0, float("inf"), float("nan")], 0.5))
    out = apply_efficiency(counts, _Calibration(), curve)
    assert np.all(np.isfinite(out.counts)), out.counts
    assert out.zeroed == LOW + 4


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
    assert out.zeroed == LOW, "only the low band should be zeroed here"
    assert out.zeroed_nonpositive == 0 and out.zeroed_nonfinite == 0, (
        "nothing should be zeroed for its efficiency inside the fitted range")
    # The applied curve is normalised to peak at 1, so dividing by it can
    # only raise counts -- never lower them.
    assert np.all(out.counts[LOW:] >= 1000.0 - 1e-9)


def test_a_negative_calibration_offset_zeroes_those_bins():
    """Real calibrations routinely have a negative offset, so channel 0 maps
    to a negative energy. Both models fail there, and each fails a DIFFERENT
    way -- which is exactly why the rule is "finite and > 0" rather than a
    literal `eff <= 0`:

        KFR      (aE + b/E)exp(...) stays finite and goes <= 0
        Radware  takes ln(E) and returns NaN

    So this one ordinary case exercises both branches of the rule with real
    curves rather than a hand-built stub.

    The offset is large enough that negative energies reach past the
    unconditionally zeroed low band. With a small one they would all fall
    inside it, both branches would go untested, and this would quietly
    become a test of nothing.
    """
    from calibration import Calibration

    result = _real_result()
    counts = np.full(64, 500.0)
    # -200 keV at channel 0, crossing zero at channel 20 -- so channels
    # LOW..19 carry negative energies and are judged on their efficiency.
    cal = Calibration("linear", -200.0, 10.0)
    assert cal.apply(np.array([float(LOW)]))[0] < 0.0, (
        "the fixture no longer puts a negative energy above the low band")
    out = apply_efficiency(counts, cal, result)

    assert np.all(np.isfinite(out.counts)), "a bad energy produced a bad count"
    assert out.zeroed > LOW, "negative energies were silently corrected"
    assert out.zeroed_nonpositive > 0, "expected KFR's finite <= 0 branch"

    result.model = "rw"
    out_rw = apply_efficiency(counts, cal, result)
    assert np.all(np.isfinite(out_rw.counts))
    assert out_rw.zeroed_nonfinite > 0, "expected Radware's NaN branch"


def test_the_original_counts_are_not_modified():
    counts = np.array([100.0, 200.0])
    original = counts.copy()
    apply_efficiency(counts, _Calibration(), _FlatCurve([0.5, 0.5]))
    assert counts == pytest.approx(original)


def test_the_lowest_channels_are_zeroed_however_good_their_efficiency():
    """User request, 2026-09-21. Below the lowest calibration line the curve
    falls towards zero, so dividing by it explodes: on a 0.5 keV/channel
    calibration against a curve fitted from 121.8 keV, channel 1 turned a
    flat 1000 counts into 5.7e+96, which sets the autoscale and hides the
    spectrum. The band goes whatever the efficiency there works out to --
    here it is a perfectly healthy 0.5 and the bins are still zeroed."""
    counts = np.full(LOW + 3, 100.0)
    curve = _FlatCurve(np.full(LOW + 3, 0.5))
    out = apply_efficiency(counts, _Calibration(), curve)

    assert out.counts[:LOW] == pytest.approx(np.zeros(LOW))
    assert out.zeroed_low == LOW
    # The boundary, stated explicitly: the band is the FIRST ten bins, so
    # channel LOW itself is ordinary and must survive.
    assert out.counts[LOW] == pytest.approx(200.0)
    assert out.counts[LOW:] == pytest.approx(np.full(3, 200.0))


def test_the_three_zeroed_counts_are_disjoint_and_add_up():
    """Channel 0 is routinely both non-finite and inside the low band.
    Counting it in two categories would report more zeroed bins than the
    spectrum has -- the fit log double-counted every auto-calibrated line
    exactly this way before v5.2.1."""
    values = np.full(LOW + 4, 0.5)
    values[0] = float("nan")      # inside the band AND non-finite
    values[LOW + 1] = float("nan")
    values[LOW + 2] = -1.0
    counts = np.full(LOW + 4, 100.0)
    out = apply_efficiency(counts, _Calibration(), _FlatCurve(values))

    assert out.zeroed_low == LOW
    assert out.zeroed_nonfinite == 1, "the in-band NaN was counted twice"
    assert out.zeroed_nonpositive == 1
    assert out.zeroed == LOW + 2
    assert out.zeroed == int(np.count_nonzero(out.counts == 0.0))


def test_a_spectrum_shorter_than_the_band_is_zeroed_entirely():
    """Degenerate, but it must not report zeroing more bins than exist."""
    counts = np.full(4, 100.0)
    out = apply_efficiency(counts, _Calibration(), _FlatCurve(np.full(4, 0.5)))
    assert out.counts == pytest.approx(np.zeros(4))
    assert out.zeroed_low == 4
    assert out.zeroed == 4


def test_the_low_band_zeroing_can_be_detected():
    """Control. Every assertion above would also hold if apply_efficiency
    zeroed the whole spectrum, so pin that it does not."""
    counts = np.full(LOW + 3, 100.0)
    out = apply_efficiency(counts, _Calibration(),
                           _FlatCurve(np.full(LOW + 3, 0.5)))
    assert np.count_nonzero(out.counts) == 3, (
        "everything was zeroed, so the band assertions prove nothing")


def test_exactly_ten_channels_are_zeroed():
    """The width, pinned as a LITERAL rather than through the constant.

    Every other test in this file reads ZEROED_LOW_CHANNELS, so their
    expectations move with it: setting it to 0 leaves them asserting over
    empty slices and they pass having checked nothing. Measured -- that
    mutation left two of them green. Ten is the number the user asked for,
    so ten is what gets written down here.
    """
    assert ZEROED_LOW_CHANNELS == 10
    counts = np.full(20, 100.0)
    out = apply_efficiency(counts, _Calibration(), _FlatCurve(np.full(20, 0.5)))
    assert int(np.count_nonzero(out.counts == 0.0)) == 10
    assert int(np.count_nonzero(out.counts == 200.0)) == 10
