"""Step 3 of the list-mode operation: the gain-matching dialog.

Driven by calling its methods and clicking its buttons, with QFileDialog
and QMessageBox patched (no pytest-qt). Synthetic singles carry a known
gain, so the fitted map can be checked against the truth; the three
guards from Task 3's review (pair range drawn, extrapolation warning,
low-channel cutoff) each have a case that must trigger and one that must
not, so a guard that never fires cannot pass as one that works. The real
Na-22 run pins the case the guards were written for.
"""

import os

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox
from scipy.special import ndtr

import gain_match as G
import gain_match_dialog as D
from listmode_session import NEXT, ListModeSession

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA22_RUN = os.path.join(REPO, "Na22_261126_ANI_STEF_180")
NA22_FILTERED = os.path.join(NA22_RUN, "FILTERED")
NA22_STAMP = "Na22_261126_ANI_STEF_20261126_115129"
NCHAN = 16384


# ----------------------------------------------------------- synthetic


def _spectrum(centroids, fwhms, areas, rng, bg_amp=2000.0, bg_scale=1500.0,
              bg_flat=20.0):
    edges = np.arange(NCHAN + 1, dtype=float) - 0.5
    mean = bg_amp * np.exp(-np.arange(NCHAN) / bg_scale) + bg_flat
    for c, f, a in zip(centroids, fwhms, areas):
        sigma = f / 2.3548200450309493
        mean = mean + a * np.diff(ndtr((edges - c) / sigma))
    return rng.poisson(mean).astype(np.int64)


def _inverse(coefficients, targets):
    a = np.asarray(coefficients, dtype=float)
    t = np.asarray(targets, dtype=float)
    x = (t - a[0]) / a[1]
    for _ in range(50):
        p = np.polynomial.polynomial.polyval(x, a)
        dp = np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(a))
        x = x - (p - t) / dp
    return x


def _singles(truth, c0, areas, seed, fwhm_at_2700=50.0):
    """{0: CH0, 1: CH1} with CH1 = the same lines seen through the
    inverse of `truth` (ch0 = truth(ch1))."""
    rng = np.random.default_rng(seed)
    c0 = np.asarray(c0, dtype=float)
    fwhm0 = fwhm_at_2700 * np.sqrt(c0 / 2700.0)
    c1 = _inverse(truth, c0)
    slope = np.polynomial.polynomial.polyval(c1, np.polynomial.polynomial.polyder(truth))
    s0 = _spectrum(c0, fwhm0, areas, rng)
    s1 = _spectrum(c1, fwhm0 / slope, areas, rng)
    return {0: s0, 1: s1}


QUAD_TRUTH = [-20.0, 0.95, 3.0e-6]
C0 = [700.0, 1600.0, 2700.0, 4300.0, 6400.0, 9100.0, 12600.0]
AREAS = [40000.0, 30000.0, 60000.0, 25000.0, 30000.0, 20000.0, 15000.0]


@pytest.fixture(scope="module")
def quad_singles():
    return _singles(QUAD_TRUTH, C0, AREAS, seed=11)


@pytest.fixture
def paired(qapp, quad_singles):
    dlg = D.GainMatchDialog(ListModeSession(singles=quad_singles))
    assert dlg.find_and_pair(), dlg.status.text()
    return dlg


def test_find_and_pair_fills_the_table_and_fits(paired):
    dlg = paired
    assert len(dlg.pairs) == len(C0)
    assert dlg.table.rowCount() == len(C0)
    assert all(dlg.is_included(r) for r in range(len(C0)))  # 700 > cutoff 300
    assert dlg.selected_order() == 2          # the automatic starting value
    assert dlg.gain is not None and dlg.gain.order == 2
    for got, want in zip(dlg.gain.coefficients, QUAD_TRUTH):
        assert got == pytest.approx(want, rel=0.05, abs=1.0)
    # every row shows centroids with errors and a residual
    assert "±" in dlg.table.item(0, 1).text()
    assert "±" in dlg.table.item(0, 2).text()
    assert dlg.table.item(0, 6).text() != ""
    assert "pairs cover CH1 channels" in dlg.summary.text()
    assert dlg.next_button.isEnabled()


def test_unticking_a_pair_refits(paired):
    dlg = paired
    before = list(dlg.gain.coefficients)
    dlg.set_included(3, False)
    assert dlg.pairs[3].included is False
    assert dlg.gain.coefficients != before
    assert not dlg.gain.pairs[3].included
    # the excluded pair still gets a residual, from the new fit
    assert dlg.table.item(3, 6).text() != ""
    dlg.set_included(3, True)
    assert dlg.gain.coefficients == pytest.approx(before)


def test_order_two_is_disabled_under_four_included_pairs(paired):
    dlg = paired
    item = dlg.order_combo.model().item(1)
    assert item.isEnabled()
    for row in range(4):                      # 7 -> 3 included
        dlg.set_included(row, False)
    assert not item.isEnabled()
    assert dlg.selected_order() == 1
    assert dlg.gain.order == 1
    assert "at least 4" in dlg.order_note.text()
    dlg.set_included(0, True)                 # back to 4
    assert item.isEnabled()
    assert dlg.order_note.text() == ""
    # the user decides: it stays at 1 until they choose 2
    assert dlg.selected_order() == 1
    dlg.order_combo.setCurrentIndex(1)
    assert dlg.gain.order == 2


def test_user_can_choose_order_one_with_many_pairs(paired):
    dlg = paired
    dlg.order_combo.setCurrentIndex(0)
    assert dlg.gain.order == 1
    assert not dlg.extrapolation_warning_shown()


def test_cutoff_unticks_low_pairs_and_reapplies(paired):
    dlg = paired
    dlg.cutoff.setValue(1000)
    assert [dlg.is_included(r) for r in range(len(C0))] == [False] + [True] * 6
    assert dlg.gain.pairs[0].included is False
    dlg.set_included(1, False)                # a manual change...
    dlg.cutoff.setValue(200)                  # ...is reset to the defaults
    assert all(dlg.is_included(r) for r in range(len(C0)))


def test_pairs_below_default_cutoff_start_unticked(qapp):
    pairs = [G.GainPair(c, 0.2, 1.01 * c + 5.0, 0.2, fwhm0=40.0)
             for c in (150.0, 2000.0, 5000.0, 9000.0)]
    dlg = D.GainMatchDialog(ListModeSession())
    dlg.set_pairs(pairs)
    assert D.DEFAULT_LOW_CUTOFF == dlg.cutoff.value()
    assert [p.included for p in dlg.pairs] == [False, True, True, True]
    assert dlg.selected_order() == 1          # 3 included -> automatic 1


def _low_peak_pairs():
    """A linear truth through three real lines, plus a low feature that
    does not follow it (a threshold edge 18 channels off the line): a
    quadratic forced through all four takes its curvature from it."""
    line = lambda x: 0.99 * x + 10.0
    pairs = [G.GainPair(150.0, 0.2, 140.0, 0.2, fwhm1=30.0, fwhm0=30.0)]
    for c, f in ((2000.0, 66.0), (5000.0, 100.0), (6000.0, 150.0)):
        pairs.append(G.GainPair(c, 0.2, line(c), 0.2, fwhm1=f, fwhm0=f))
    return pairs


def test_extrapolation_warning_for_curvature_from_a_low_peak(qapp):
    dlg = D.GainMatchDialog(ListModeSession())
    dlg.set_pairs(_low_peak_pairs())
    assert not dlg.extrapolation_warning_shown()   # low pair unticked, order 1
    dlg.set_included(0, True)
    dlg.order_combo.setCurrentIndex(1)
    assert dlg.gain.order == 2
    x = dlg.extrapolation
    assert x is not None and x.warn
    assert x.fwhm == pytest.approx(150.0)
    assert x.deviation > x.fwhm
    assert x.channel > 6000                         # beyond the last pair
    assert dlg.extrapolation_warning_shown()
    assert f"{x.deviation:.0f} channels" in dlg.extrapolation_label.text()
    dlg.order_combo.setCurrentIndex(0)
    assert not dlg.extrapolation_warning_shown()


def test_no_extrapolation_warning_for_a_linear_truth(qapp):
    singles = _singles([15.0, 1.08, 0.0], [2000.0, 3500.0, 5500.0, 8000.0,
                                           11000.0, 14000.0],
                       [40000.0, 30000.0, 50000.0, 30000.0, 20000.0, 15000.0],
                       seed=3)
    dlg = D.GainMatchDialog(ListModeSession(singles=singles))
    assert dlg.find_and_pair(), dlg.status.text()
    assert dlg.gain.order == 2
    x = dlg.extrapolation
    assert x is not None and not x.warn, x       # computed, and within a FWHM
    assert not dlg.extrapolation_warning_shown()


def test_map_plot_marks_the_extrapolated_part(paired):
    dlg = paired
    lines = dlg.map_axes.get_lines()
    styles = {l.get_linestyle() for l in lines}
    assert "-" in styles and "--" in styles       # fitted solid, extrapolated dashed
    assert len(dlg.map_axes.patches) >= 1          # the shaded range
    # the overlay shows both singles
    labels = [l.get_label() for l in dlg.overlay_axes.get_lines()]
    assert labels == ["CH0", "CH1 mapped"]


def test_mapped_histogram_puts_ch1_peaks_on_ch0(quad_singles):
    gm = G.GainMatch(QUAD_TRUTH)
    x, y = D.mapped_histogram(gm, quad_singles[1])
    # the strongest line (2700 in CH0) lands there after mapping
    window = (x > 2000) & (x < 3400)
    assert abs(x[window][np.argmax(y[window])] - 2700.0) < 5.0


def test_save_and_load_round_trip(paired, tmp_path, monkeypatch):
    dlg = paired
    path = str(tmp_path / "gain.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    dlg.save_button.click()
    assert os.path.exists(path)

    other = D.GainMatchDialog(ListModeSession())   # no singles: needs no peaks
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    other.load_button.click()
    assert other.gain is not None
    assert other.gain.coefficients == dlg.gain.coefficients
    assert other.table.rowCount() == len(dlg.pairs)
    assert other.selected_order() == dlg.gain.order
    assert "loaded" in other.status.text()
    assert other.apply_to_session() is None
    assert other.session.gain.coefficients == dlg.gain.coefficients


def test_load_refuses_a_wrong_file(qapp, tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"format": "other"}', encoding="utf-8")
    dlg = D.GainMatchDialog(ListModeSession())
    assert not dlg.load_coefficients(str(path))
    assert "not a SpectraTools gain-match" in dlg.status.text()
    assert dlg.gain is None


def test_skip_sets_identity_on_the_session(qapp):
    session = ListModeSession(gain=G.GainMatch([0.0, 1.0]))
    dlg = D.GainMatchDialog(session)
    dlg.skip_check.setChecked(True)
    assert not dlg.skip_label.isHidden()
    assert "own ADC scale" in dlg.skip_label.text()
    dlg.next_button.click()
    assert dlg.step_result == NEXT
    assert session.gain is None and session.gain_skipped is True


def test_next_without_fit_or_skip_is_refused(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: shown.append(a[2])))
    session = ListModeSession()
    dlg = D.GainMatchDialog(session)
    message = dlg.apply_to_session()
    assert message and "Skip" in message
    dlg.next_button.click()
    assert dlg.step_result != NEXT
    assert shown and "no gain match" in shown[0]
    assert session.gain is None and not session.gain_skipped


def test_pairing_failure_is_shown_verbatim_and_blocks_next(qapp):
    rng = np.random.default_rng(5)
    s1 = _spectrum([3000.0, 3400.0], [8.0, 8.0], [20000.0, 20000.0], rng,
                   bg_amp=50.0, bg_flat=2.0)
    s0 = _spectrum([3000.0, 3400.0, 3800.0], [8.0] * 3, [20000.0] * 3, rng,
                   bg_amp=50.0, bg_flat=2.0)
    dlg = D.GainMatchDialog(ListModeSession(singles={0: s0, 1: s1}))
    assert not dlg.find_and_pair()
    p1, p0 = G.find_peaks(s1), G.find_peaks(s0)
    reason = G.pair_peaks(p1, p0).reason
    assert reason
    assert reason in dlg.status.text()
    assert dlg.gain is None
    message = dlg.apply_to_session()
    assert reason in message


def test_back_visit_shows_the_stored_gain(paired):
    gm = paired.gain
    session = ListModeSession(singles=paired.session.singles, gain=gm)
    dlg = D.GainMatchDialog(session)
    assert dlg.gain is gm
    assert dlg.table.rowCount() == len(gm.pairs)
    assert dlg.selected_order() == gm.order
    assert not dlg.skip_check.isChecked()
    assert dlg.apply_to_session() is None and session.gain is gm


def test_back_visit_shows_a_skip(qapp):
    dlg = D.GainMatchDialog(ListModeSession(gain_skipped=True))
    assert dlg.skip_check.isChecked()


def test_calibration_spectra_are_used_for_the_peaks(qapp, quad_singles, tmp_path):
    from histogram_io import save_histogram

    p1, p0 = tmp_path / "cal_ch1.txt", tmp_path / "cal_ch0.txt"
    save_histogram(str(p1), quad_singles[1])
    save_histogram(str(p0), quad_singles[0])
    dlg = D.GainMatchDialog(ListModeSession())      # no singles of its own
    assert not dlg.find_button.isEnabled()
    assert dlg.load_calibration_spectra(str(p1), str(p0))
    assert dlg.calibration_radio.isChecked()
    assert "cal_ch1.txt" in dlg.peak_source_label.text()
    assert dlg.find_and_pair(), dlg.status.text()
    assert len(dlg.pairs) == len(C0)
    assert "cal_ch1.txt" in dlg.gain.source


def test_calibration_sdata_is_sorted_with_the_excluded_flags(qapp, tmp_path):
    from synthetic_sdata import write_sdata

    truth = [10.0, 1.1, 0.0]
    rng = np.random.default_rng(8)
    c0 = np.array([2000.0, 4500.0, 7000.0, 10000.0])
    c1 = _inverse(truth, c0)
    e0, e1 = [], []
    for a, b in zip(c0, c1):
        e0.append(rng.normal(a, 15.0, 6000))
        e1.append(rng.normal(b, 15.0, 6000))
    e0 = np.concatenate(e0 + [rng.uniform(0, 16000, 20000)])
    e1 = np.concatenate(e1 + [rng.uniform(0, 16000, 20000)])
    energy = np.round(np.concatenate([e0, e1])).astype(int)
    channel = np.concatenate([np.zeros(e0.size), np.ones(e1.size)]).astype(int)
    flags = np.zeros(energy.size, dtype=int)
    flags[:500] = 0x80                                  # excluded value
    order = rng.permutation(energy.size)
    path = write_sdata(str(tmp_path / "SDataR_cal.root"), channel[order],
                       np.arange(energy.size) * 10000, energy[order], flags[order])
    session = ListModeSession(exclude_flags={0x80})
    dlg = D.GainMatchDialog(session)
    assert dlg.load_calibration_sdata(path), dlg.status.text()
    ch1, ch0 = dlg.peak_spectra()
    assert ch0.sum() + ch1.sum() == energy.size - 500
    assert dlg.find_and_pair(), dlg.status.text()
    assert dlg.gain.map(c1) == pytest.approx(c0, abs=3.0)


def test_source_file_switches_to_energy_pairing(qapp):
    from synthetic_calibration import source_file

    dlg = D.GainMatchDialog(ListModeSession())
    path = source_file("eu152.sou")
    assert dlg.load_source(path)
    assert dlg.source_lines and dlg.source_path == path
    assert dlg.source_combo.currentData() == path
    assert dlg.load_source(None)
    assert dlg.source_lines is None and dlg.source_combo.currentIndex() == 0


# ------------------------------------------------------------ real Na-22


@pytest.fixture(scope="module")
def na22_singles():
    if not os.path.isdir(NA22_RUN):
        pytest.skip("untracked run Na22_261126_ANI_STEF_180/ is absent")
    singles = {}
    for ch in (0, 1):
        path = os.path.join(NA22_FILTERED,
                            f"CH{ch}@DT5780SCP_58363_EspectrumF_{NA22_STAMP}.txt")
        counts = np.loadtxt(path)
        singles[ch] = np.concatenate([counts, np.zeros(NCHAN - counts.size)])
    return singles


def test_na22_default_cutoff_and_forced_quadratic(qapp, na22_singles):
    dlg = D.GainMatchDialog(ListModeSession(singles=na22_singles))
    assert dlg.find_and_pair(), dlg.status.text()
    ch0 = [p.ch0 for p in dlg.pairs]
    assert len(ch0) == 4
    assert ch0[0] < 300                       # the ~150 threshold feature...
    assert not dlg.is_included(0)             # ...starts unticked
    used = [p.ch0 for p in dlg.pairs if p.included]
    for want in (2173.0, 5391.0, 6176.0):     # 511, 1275, 1461 keV in CH0
        assert min(abs(c - want) for c in used) < 30
    assert dlg.gain.order == 1
    assert dlg.gain.rms < 5.0
    assert not dlg.extrapolation_warning_shown()

    dlg.set_included(0, True)
    dlg.order_combo.setCurrentIndex(1)        # order 2 forced on all four
    assert dlg.gain.order == 2
    x = dlg.extrapolation
    assert x.warn and x.channel > 6200
    assert x.deviation > 300                  # ~500 ch against the linear fit
    assert dlg.extrapolation_warning_shown()
    print(f"\nNa-22 forced order 2: deviation {x.deviation:.0f} ch at CH1 "
          f"{x.channel}, largest FWHM {x.fwhm:.0f} ch")
