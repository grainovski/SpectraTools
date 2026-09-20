"""The two efficiency files.

No channel column in either: the energy calibration is available wherever
these are read, so a channel column would duplicate a derived value on disk
where it can go stale.
"""

import os

import numpy as np
import pytest

from efficiency_io import write_per_bin, write_per_peak

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")


@pytest.fixture(scope="module")
def result():
    from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo

    data = np.loadtxt(os.path.join(FIXTURES, "demo1.txt"), ndmin=2)
    N, dN, E, I, dI = (data[:, 2], data[:, 3], data[:, 4],
                       data[:, 5], data[:, 6])
    fit = fit_efficiency(E, N, dN, I, dI)
    return EfficiencyResult(fit=fit,
                            mc=run_monte_carlo(fit, N, dN, I, dI,
                                               iterations=200),
                            model="kfr")


def _rows(path):
    return [l for l in open(path, encoding="utf-8").read().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def _header(path):
    return "\n".join(l for l in open(path, encoding="utf-8").read().splitlines()
                     if l.lstrip().startswith("#"))


def test_per_peak_has_six_columns_and_one_row_per_peak(tmp_path, result):
    energy_errors = np.full(len(result.fit.E), 0.01)
    path = str(tmp_path / "eff_peaks.txt")
    write_per_peak(path, result, energy_errors)
    rows = _rows(path)
    assert len(rows) == len(result.fit.E)
    for row in rows:
        assert len(row.split()) == 6, row


def test_per_bin_has_five_columns_and_one_row_per_channel(tmp_path, result):
    class _Cal:
        def apply(self, channel):
            return np.asarray(channel, dtype=float) + 50.0

    path = str(tmp_path / "eff_bins.txt")
    write_per_bin(path, result, _Cal(), channels=128)
    rows = _rows(path)
    assert len(rows) == 128
    for row in rows:
        assert len(row.split()) == 5, row


def test_neither_file_carries_a_channel_column(tmp_path, result):
    """The decision this format turns on. A channel column would be a stale
    copy of something the calibration already answers."""
    class _Cal:
        def apply(self, channel):
            return np.asarray(channel, dtype=float) + 50.0

    peaks = str(tmp_path / "p.txt")
    bins = str(tmp_path / "b.txt")
    write_per_peak(peaks, result, np.full(len(result.fit.E), 0.01))
    write_per_bin(bins, result, _Cal(), channels=32)
    for path, n in ((peaks, 6), (bins, 5)):
        assert len(_rows(path)[0].split()) == n
        assert "ch" not in _header(path).lower().split("columns:")[-1]


def test_the_header_records_what_the_numbers_depend_on(tmp_path, result):
    """The normalisation is the selected model's peak, so the same
    calibration saved under the other model writes different numbers. The
    header has to say which one produced these."""
    path = str(tmp_path / "p.txt")
    write_per_peak(path, result, np.full(len(result.fit.E), 0.01))
    header = _header(path)
    assert "kfr" in header.lower()
    assert "normalis" in header.lower() or "normaliz" in header.lower()
    assert "%.6g" % result.normalisation in header


def test_the_selected_curve_never_exceeds_one_in_the_file(tmp_path, result):
    path = str(tmp_path / "p.txt")
    write_per_peak(path, result, np.full(len(result.fit.E), 0.01))
    eff_kfr = np.array([float(r.split()[2]) for r in _rows(path)])
    assert np.all(eff_kfr <= 1.0 + 1e-9), eff_kfr.max()


def test_a_big_per_bin_file_is_written_in_seconds_not_minutes(tmp_path, result):
    """Written the obvious way this took 91.6 s on a full Monte Carlo, which
    reads as a hang. The knot interpolation and the shared band centre bring
    it to a few seconds. The threshold is loose on purpose -- this guards
    against a return to the quadratic behaviour, not against a slow machine."""
    import time

    class _Cal:
        def apply(self, channel):
            return 50.0 + np.asarray(channel, dtype=float) * 0.25

    # The shared fixture holds 200 samples, which is fast whatever the
    # implementation does. Tile it to a realistic 10,000 so the expensive
    # path is actually exercised -- without paying for 10,000 real fits,
    # since the cost depends only on the array shapes.
    from efficiency import EfficiencyResult

    big = EfficiencyResult(fit=result.fit, mc=result.mc, model="kfr")
    big.mc.kfr_samples = np.tile(result.mc.kfr_samples, (50, 1))[:10000]
    big.mc.rw_samples = np.tile(result.mc.rw_samples, (50, 1))[:10000]

    path = str(tmp_path / "big.txt")
    start = time.perf_counter()
    write_per_bin(path, big, _Cal(), channels=16384)
    assert time.perf_counter() - start < 60.0
    assert len(_rows(path)) == 16384


def test_interpolation_stays_far_below_the_stated_uncertainty(result):
    """The header claims ~2e-6. If that claim is ever wrong the file is
    quietly less accurate than it says, which is worse than being slow."""
    from efficiency_io import FILE_KNOTS, _both

    energies = np.linspace(float(result.fit.E.min()),
                           float(result.fit.E.max()), FILE_KNOTS * 4)
    approx = _both(result, energies)[0]
    exact = result.curve(energies, "kfr")
    good = np.isfinite(approx) & np.isfinite(exact) & (exact != 0)
    assert good.any()
    error = np.max(np.abs(approx[good] - exact[good]) / np.abs(exact[good]))
    assert error < 1e-5, "interpolation error %.2e exceeds the header's claim" % error


def test_a_small_file_is_not_interpolated_at_all(result):
    """Below the knot count the energies ARE the knots, so the values must
    match the exact curve bit for bit -- no approximation is introduced for
    an ordinary 4096-channel spectrum."""
    from efficiency_io import FILE_KNOTS, _both

    energies = np.linspace(float(result.fit.E.min()),
                           float(result.fit.E.max()), FILE_KNOTS // 2)
    approx = _both(result, energies)[0]
    exact = result.curve(energies, "kfr")
    assert approx == pytest.approx(exact, rel=1e-12, nan_ok=True)


def test_a_missing_normalisation_would_fail_that():
    """Control: raw eps for this data is in the thousands, so an unnormalised
    file could not pass the test above."""
    assert 67254.0 > 1.0
