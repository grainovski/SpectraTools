"""Dividing a spectrum by the efficiency curve, bin by bin."""

import os

import numpy as np
import pytest

from efficiency_apply import ZEROED_BELOW_KEV, apply_efficiency

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")


class _FlatCurve:
    """A stand-in for EfficiencyResult exposing only what apply needs.

    Indexed from ZEROED_BELOW_KEV so that values[i] is the efficiency at
    the energy _Calibration gives channel i.
    """

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def curve(self, grid, model=None):
        grid = np.asarray(grid, dtype=float)
        return np.interp(
            grid,
            np.arange(len(self._values), dtype=float) + ZEROED_BELOW_KEV,
            self._values)


class _Calibration:
    """channel -> energy, one keV per channel, starting AT the threshold.

    Channel 0 lands on ZEROED_BELOW_KEV exactly, which is not below it, so
    no bin is zeroed for its energy. That keeps the tests below about the
    arithmetic they are each named for; the threshold gets its own tests,
    with a calibration that actually dips under it.
    """

    def apply(self, channel):
        return np.asarray(channel, dtype=float) + ZEROED_BELOW_KEV


class _LowCalibration:
    """channel -> energy at 10 keV per channel from zero, so channels 0-4
    fall below a 50 keV threshold and channel 5 lands exactly on it."""

    def apply(self, channel):
        return np.asarray(channel, dtype=float) * 10.0


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


def test_everything_below_the_threshold_is_zeroed():
    """User request, 2026-09-21, refined from "the first 10 bins".

    Below the lowest calibration line the curve falls away towards zero, so
    dividing by it explodes: on a 0.5 keV/channel calibration against a
    curve fitted from 121.8 keV, channel 1 turned a flat 1000 counts into
    5.7e+96, which sets the autoscale and hides the spectrum. The band goes
    whatever the efficiency there works out to -- here it is a healthy 0.5
    throughout and the low bins are still zeroed.
    """
    counts = np.full(8, 100.0)
    out = apply_efficiency(counts, _LowCalibration(), _FlatCurve(np.full(8, 0.5)))

    # 10 keV per channel: channels 0-4 are under 50, channel 5 is exactly on
    # it. The boundary is "below", not "at or below", so channel 5 survives.
    assert out.counts[:5] == pytest.approx(np.zeros(5))
    assert out.zeroed_low == 5
    assert out.counts[5] > 0.0, "the bin exactly ON the threshold was zeroed"


def test_the_threshold_is_in_keV_so_it_holds_at_any_gain():
    """The point of the rewrite from a channel count. A fixed number of
    channels covers the blow-up at one gain only; the same 50 keV covers it
    at every gain, and zeroes a different number of bins each time."""
    counts = np.full(40, 100.0)
    curve = _FlatCurve(np.full(200, 0.5))

    class _Gain:
        def __init__(self, keV_per_channel):
            self.g = keV_per_channel

        def apply(self, channel):
            return np.asarray(channel, dtype=float) * self.g

    # 50 keV is 25 channels at 2 keV/ch, 10 at 5, 5 at 10.
    assert apply_efficiency(counts, _Gain(2.0), curve).zeroed_low == 25
    assert apply_efficiency(counts, _Gain(5.0), curve).zeroed_low == 10
    assert apply_efficiency(counts, _Gain(10.0), curve).zeroed_low == 5


def test_the_threshold_is_fifty_keV():
    """The value, pinned as a LITERAL rather than through the constant.

    Every other test here reads ZEROED_BELOW_KEV, so their expectations
    move with it: setting it to 0 leaves them asserting over empty slices
    and they pass having checked nothing. Measured on the earlier
    channel-based version -- that mutation left two of them green. Fifty is
    the number the user asked for, so fifty is what gets written down.
    """
    assert ZEROED_BELOW_KEV == 50.0
    counts = np.full(20, 100.0)

    class _OneKeV:
        def apply(self, channel):
            return np.asarray(channel, dtype=float)

    out = apply_efficiency(counts, _OneKeV(), _FlatCurve(np.full(200, 0.5)))
    assert int(np.count_nonzero(out.counts == 0.0)) == 20, (
        "at 1 keV/channel every one of 20 bins is under 50 keV")

    counts = np.full(60, 100.0)
    out = apply_efficiency(counts, _OneKeV(), _FlatCurve(np.full(200, 0.5)))
    assert int(np.count_nonzero(out.counts == 0.0)) == 50


def test_the_three_zeroed_counts_are_disjoint_and_add_up():
    """A bin below the threshold usually ALSO has a non-finite or
    non-positive efficiency. Counting it in two categories would report more
    zeroed bins than the spectrum has -- the fit log double-counted every
    auto-calibrated line exactly this way before v5.2.1."""
    class _TenKeV:
        def apply(self, channel):
            return np.asarray(channel, dtype=float) * 10.0

    class _Curve:
        """NaN below 50 keV -- so those bins are BOTH below the threshold
        and non-finite, which is the overlap being tested -- plus one NaN
        and one negative above it."""

        def curve(self, grid, model=None):
            g = np.asarray(grid, dtype=float)
            out = np.full(g.shape, 0.5)
            out[g < ZEROED_BELOW_KEV] = float("nan")
            out[np.isclose(g, 60.0)] = float("nan")
            out[np.isclose(g, 70.0)] = -1.0
            return out

    out = apply_efficiency(np.full(10, 100.0), _TenKeV(), _Curve())

    assert out.zeroed_low == 5        # channels 0-4 at 0-40 keV, NaN as well
    assert out.zeroed_nonfinite == 1  # channel 6 at 60 keV
    assert out.zeroed_nonpositive == 1  # channel 7 at 70 keV
    assert out.zeroed == 7
    assert out.zeroed == int(np.count_nonzero(out.counts == 0.0))
    # The five overlapping bins were counted once, not twice.
    assert out.zeroed < 5 + 5 + 1 + 1


def test_a_spectrum_entirely_below_the_threshold_is_zeroed_entirely():
    """Degenerate, but it must not report zeroing more bins than exist."""
    counts = np.full(4, 100.0)

    class _OneKeV:
        def apply(self, channel):
            return np.asarray(channel, dtype=float)

    out = apply_efficiency(counts, _OneKeV(), _FlatCurve(np.full(200, 0.5)))
    assert out.counts == pytest.approx(np.zeros(4))
    assert out.zeroed_low == 4
    assert out.zeroed == 4


def test_a_spectrum_starting_above_the_threshold_loses_nothing():
    """The rule self-limits, which a fixed channel count could not do: a
    spectrum whose first channel is already above 50 keV keeps every bin."""
    counts = np.full(6, 100.0)
    out = apply_efficiency(counts, _Calibration(), _FlatCurve(np.full(6, 0.5)))
    assert out.zeroed_low == 0
    assert out.counts == pytest.approx(np.full(6, 200.0))


def _real_result(name="demo1.txt"):
    """A genuine EfficiencyResult, not the stub above."""
    from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo

    d = np.loadtxt(os.path.join(FIXTURES, name), ndmin=2)
    N, dN, E, I, dI = d[:, 2], d[:, 3], d[:, 4], d[:, 5], d[:, 6]
    fit = fit_efficiency(E, N, dN, I, dI)
    mc = run_monte_carlo(fit, N, dN, I, dI, iterations=200)
    return EfficiencyResult(fit=fit, mc=mc, model="krf")


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
    # demo1 is fitted from 53.2 keV, so the whole spectrum sits above the
    # threshold and nothing is zeroed at all.
    assert out.zeroed == 0, "nothing should be zeroed inside the fitted range"
    # The applied curve is normalised to peak at 1, so dividing by it can
    # only raise counts -- never lower them.
    assert np.all(out.counts >= 1000.0 - 1e-9)


def test_a_negative_calibration_offset_zeroes_those_bins():
    """Real calibrations routinely have a negative offset, so the lowest
    channels map to negative energies. With a real curve those are the bins
    that used to produce NaN (Radware takes ln E) or a negative efficiency
    (KRF stays finite and goes <= 0).

    The threshold now catches them first, and that is the point: a negative
    energy is below 50 keV by definition, so the user never sees either
    failure. The `finite and > 0` rule underneath is still there and still
    tested -- by the stub curves above, which can put a NaN or a negative
    at any energy. It is no longer reachable from below with a real curve,
    and above it needs about 1e6 keV, so this test no longer claims to
    exercise both branches; it pins the guarantee the user actually gets.
    """
    from calibration import Calibration

    result = _real_result()
    counts = np.full(64, 500.0)
    # -20 keV at channel 0, reaching 400 keV at the top.
    cal = Calibration("linear", -20.0, 420.0 / 63.0)
    out = apply_efficiency(counts, cal, result)

    assert np.all(np.isfinite(out.counts)), "a bad energy produced a bad count"
    assert out.zeroed_low > 0, "negative energies were silently corrected"
    # Every zeroed bin is accounted for by the threshold, not by a failure.
    assert out.zeroed == out.zeroed_low
    energies = cal.apply(np.arange(64, dtype=float))
    assert int(np.count_nonzero(energies < ZEROED_BELOW_KEV)) == out.zeroed_low

    result.model = "rw"
    out_rw = apply_efficiency(counts, cal, result)
    assert np.all(np.isfinite(out_rw.counts)), "Radware's NaN escaped"
    assert out_rw.zeroed == out_rw.zeroed_low


def test_the_original_counts_are_not_modified():
    counts = np.array([100.0, 200.0])
    original = counts.copy()
    apply_efficiency(counts, _Calibration(), _FlatCurve([0.5, 0.5]))
    assert counts == pytest.approx(original)
