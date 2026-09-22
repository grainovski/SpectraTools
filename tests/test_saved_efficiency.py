"""Reading a saved efficiency -- and the energy calibration it was made
under -- back into the program.

The files written by efficiency_io are the only place a finished calibration
is kept, and the only place an energy calibration the program produced is
written down at all. These tests hold the loader to one standard: what comes
back must be what was applied.
"""

import os

import numpy as np
import pytest

import efficiency_io as eio
from calibration import Calibration, from_points
from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo
from efficiency_io import (
    SavedEfficiencyError, is_saved_efficiency, parse_energy_calibration,
    read_energy_calibration, read_saved_efficiency,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")
CAL = Calibration(kind="quadratic", a=-0.35, b=0.5, c=2e-8)
CHANNELS = 8192


def _load(name):
    d = np.loadtxt(os.path.join(FIXTURES, name), ndmin=2)
    return d[:, 2], d[:, 3], d[:, 4], d[:, 5], d[:, 6]


@pytest.fixture(scope="module")
def demo2():
    """demo2, because its Radware Monte Carlo mean falls off a cliff at the
    lowest calibration line -- the case that exposed the writer defect."""
    N, dN, E, I, dI = _load("demo2.txt")
    fit = fit_efficiency(E, N, dN, I, dI)
    mc = run_monte_carlo(fit, N, dN, I, dI, iterations=1000)
    return fit, mc


def _save(tmp_path, fit, mc, model, name="x"):
    result = EfficiencyResult(fit, mc, model=model, calibration=CAL,
                              source="demo2.txt")
    stem = os.path.join(str(tmp_path), name)
    eio.write_per_peak(stem + "_peaks.txt", result,
                       np.full(len(fit.E), 0.05))
    eio.write_per_bin(stem + "_bins.txt", result, CAL, CHANNELS)
    return result, stem


def _energies():
    e = CAL.apply(np.arange(CHANNELS, dtype=float))
    return e[e >= 50.0]


# --- the curve comes back as it was applied ------------------------------


@pytest.mark.parametrize("model", ("krf", "rw"))
def test_a_saved_efficiency_reads_back_as_it_was_applied(tmp_path, demo2,
                                                         model):
    """At every channel above the 50 keV floor, the loaded curve must equal
    the Monte Carlo mean the program applies. Radware on demo2 is the hard
    case: its best fit sits 10% away from that mean inside the fitted range,
    which is why the loader reads the saved curve and not the parameters."""
    fit, mc = demo2
    result, stem = _save(tmp_path, fit, mc, model)
    saved = read_saved_efficiency(stem + "_bins.txt")
    e = _energies()
    want = np.asarray(result.curve(e, model))
    got = saved.curve(e)
    ok = np.isfinite(want) & np.isfinite(got) & (want > 0)
    assert ok.sum() > 0.9 * len(e)
    worst = np.max(np.abs(got[ok] - want[ok]) / want[ok])
    assert worst < 1e-4, (
        "%s read back %.2e away from the applied curve" % (model, worst))


def test_the_parameters_would_not_have_been_good_enough(demo2):
    """Control for the test above, and the reason for the design. If the
    Radware best fit were close to the applied mean, reading the saved
    parameters would have done, and the round trip above would prove less
    than it seems to."""
    fit, mc = demo2
    result = EfficiencyResult(fit, mc, model="rw", calibration=CAL)
    g = np.linspace(fit.E.min(), fit.E.max(), 400)
    applied = np.asarray(result.curve(g, "rw"))
    best = np.asarray(result.best_fit_raw(g, "rw")) * result.normalisation
    ok = np.isfinite(applied) & np.isfinite(best) & (applied > 0)
    assert np.max(np.abs(best[ok] - applied[ok]) / applied[ok]) > 1e-2


def test_the_saved_file_itself_is_accurate_at_every_channel(tmp_path, demo2):
    """The writer's knots were interpolated straight across Radware's cliff
    at the lowest line, writing values up to 527% wrong under a header that
    claimed 2e-6. Pin the checked-and-refined file to the exact curve."""
    fit, mc = demo2
    result, stem = _save(tmp_path, fit, mc, "rw")
    table = np.loadtxt(stem + "_bins.txt", comments="#")
    E, rw = table[:, 0], table[:, 3]
    exact = (np.asarray(result._mc_mean_raw(E, "rw"))
             * result.normalisation)
    ok = np.isfinite(exact) & (exact > 0) & (E >= 50.0) & np.isfinite(rw)
    worst = np.max(np.abs(rw[ok] - exact[ok]) / exact[ok])
    assert worst < 1e-4, "the saved Radware curve is %.2e off" % worst


def test_plain_knot_interpolation_really_did_fail_there(demo2):
    """Control for the test above: without the refinement the same file
    would be badly wrong, so the refinement is doing real work."""
    fit, mc = demo2
    result = EfficiencyResult(fit, mc, model="rw", calibration=CAL)
    E = CAL.apply(np.arange(CHANNELS, dtype=float))
    knots = np.linspace(E[0], E[-1], eio.FILE_KNOTS)
    plain = np.interp(E, knots, np.asarray(result._mc_mean_raw(knots, "rw"))
                      * result.normalisation)
    exact = np.asarray(result._mc_mean_raw(E, "rw")) * result.normalisation
    ok = np.isfinite(exact) & (exact > 0) & (E >= 50.0)
    assert np.max(np.abs(plain[ok] - exact[ok]) / exact[ok]) > 1e-2


def test_nothing_is_invented_outside_what_was_saved(tmp_path, demo2):
    """NaN below the 50 keV floor and beyond the last saved energy -- the
    apply path zeroes a non-finite efficiency, which is the honest result
    for an energy this curve was never evaluated at."""
    fit, mc = demo2
    _result, stem = _save(tmp_path, fit, mc, "krf")
    saved = read_saved_efficiency(stem + "_bins.txt")
    lo, hi = saved.energy_range
    assert lo >= 50.0
    out = saved.curve(np.array([10.0, 49.9, hi + 50.0, np.nan]))
    assert np.all(np.isnan(out))


# --- which file, and which spelling ---------------------------------------


def test_picking_the_peaks_file_reads_its_bins_file(tmp_path, demo2):
    fit, mc = demo2
    _result, stem = _save(tmp_path, fit, mc, "rw")
    via_peaks = read_saved_efficiency(stem + "_peaks.txt")
    assert via_peaks.path.endswith("_bins.txt")
    assert via_peaks.model == "rw"


def test_a_peaks_file_alone_is_refused_with_a_reason(tmp_path, demo2):
    fit, mc = demo2
    _result, stem = _save(tmp_path, fit, mc, "krf")
    os.remove(stem + "_bins.txt")
    with pytest.raises(SavedEfficiencyError, match="_bins"):
        read_saved_efficiency(stem + "_peaks.txt")


def test_a_file_saved_before_the_rename_still_reads(tmp_path, demo2):
    """Files from 6.1.0 and earlier say KFR and eff_kfr, and those before
    6.1.0 have no Radware scale line. Reading old files back is the point
    of the loader, so an old one must give the identical curve."""
    fit, mc = demo2
    _result, stem = _save(tmp_path, fit, mc, "krf")
    new = read_saved_efficiency(stem + "_bins.txt")
    text = open(stem + "_bins.txt", encoding="utf-8").read()
    old_text = "".join(
        line for line in text.replace("KRF", "KFR").replace("krf", "kfr")
        .splitlines(keepends=True) if "Radware scale" not in line)
    assert "eff_kfr" in old_text and "KFR  params" in old_text
    old_path = os.path.join(str(tmp_path), "old_bins.txt")
    with open(old_path, "w", encoding="utf-8") as fh:
        fh.write(old_text)
    old = read_saved_efficiency(old_path)
    e = _energies()
    assert old.model == "krf"
    np.testing.assert_array_equal(old.curve(e), new.curve(e))
    assert old.calibration == CAL


def test_files_that_are_not_saved_efficiencies_are_refused(tmp_path):
    caleneff = os.path.join(FIXTURES, "demo1.txt")
    assert not is_saved_efficiency(caleneff)
    with pytest.raises(SavedEfficiencyError, match="not an efficiency"):
        read_saved_efficiency(caleneff)


def test_a_non_path_is_never_opened():
    """open(False) opens standard input. A Qt clicked() signal wired
    straight to a slot with an optional path hands it exactly False."""
    assert is_saved_efficiency(False) is False
    assert is_saved_efficiency(None) is False
    with pytest.raises(SavedEfficiencyError):
        read_saved_efficiency(False)


# --- the energy calibration ------------------------------------------------


@pytest.mark.parametrize("calibration", [
    Calibration(kind="linear", a=0.1, b=0.5),
    Calibration(kind="quadratic", a=-0.35, b=0.5, c=2e-08),
    from_points([100.0, 500.0, 1200.0, 3000.0],
                [50.3, 250.1, 600.7, 1500.2], quadratic=True),
])
def test_the_energy_calibration_reads_back_exactly(calibration):
    """The header line is the dataclass repr, which round-trips a float
    exactly -- including a fitted calibration that carries its errors."""
    line = "# energy calibration : %s\n" % (calibration,)
    got = parse_energy_calibration(line)
    assert got == calibration
    assert (got.a, got.b, got.c) == (calibration.a, calibration.b,
                                     calibration.c)


def test_a_numpy_scalar_repr_is_still_read():
    """NumPy 2 prints its scalars as np.float64(...). Today's fields are
    plain floats, but one numpy value reaching the dataclass must not make
    a correct file unreadable."""
    line = ("# energy calibration : Calibration(kind='linear', "
            "a=np.float64(0.25), b=np.float64(0.5), c=0.0, "
            "coefficient_errors=())")
    assert parse_energy_calibration(line) == Calibration("linear", 0.25, 0.5)


def test_a_missing_or_broken_calibration_line():
    assert parse_energy_calibration("# source : x\n") is None
    with pytest.raises(SavedEfficiencyError):
        parse_energy_calibration("# energy calibration : something else\n")


def test_read_energy_calibration_from_a_saved_file(tmp_path, demo2):
    fit, mc = demo2
    _result, stem = _save(tmp_path, fit, mc, "krf")
    for suffix in ("_bins.txt", "_peaks.txt"):
        assert read_energy_calibration(stem + suffix) == CAL


# --- the interface -----------------------------------------------------------


def _window(qapp, calibration=None):
    from main_window import MainWindow
    from spectrum import LoadedSpectrum

    window = MainWindow()
    window._calibration = calibration
    window._calibration_active = calibration is not None
    spectrum = LoadedSpectrum("target.txt",
                              np.random.default_rng(4).poisson(
                                  900.0, CHANNELS).astype(float), "#FF0000")
    spectrum.active = True
    window.spectra.append(spectrum)
    return window, spectrum


def test_the_menu_offers_it(qapp):
    """Computation tests are not proof a feature shipped -- the user has to
    be able to reach it."""
    window, _ = _window(qapp)
    labels = [a.text() for a in window.operations_menu.actions()]
    assert "Load Efficiency..." in labels
    assert (labels.index("Load Efficiency...")
            == labels.index("Show Efficiency...") + 1)
    window.close()


@pytest.mark.parametrize("model", ("krf", "rw"))
def test_applying_a_loaded_efficiency_matches_applying_the_original(
        qapp, tmp_path, demo2, model):
    """The whole point, end to end: correcting a spectrum through a file
    loaded from disk must give the same counts as correcting it with the
    efficiency that was saved."""
    fit, mc = demo2
    original, stem = _save(tmp_path, fit, mc, model)

    window, spectrum = _window(qapp, calibration=CAL)
    window.apply_efficiency(spectrum, original)
    direct = window.spectra[-1].data.copy()

    dialog = window.load_efficiency(stem + "_bins.txt")
    assert dialog is not None and dialog.result.model == model
    dialog.target_combo.setCurrentIndex(0)          # the original spectrum
    dialog._on_apply()
    loaded = window.spectra[-1].data

    both = (direct != 0) & (loaded != 0)
    assert both.sum() > 0.9 * np.count_nonzero(direct)
    rel = np.abs(loaded[both] - direct[both]) / np.abs(direct[both])
    assert rel.max() < 1e-4, "loaded correction differs by %.2e" % rel.max()
    dialog.close()
    window.close()


def test_its_energy_calibration_can_be_made_active(qapp, tmp_path, demo2):
    fit, mc = demo2
    _result, stem = _save(tmp_path, fit, mc, "krf")
    window, _ = _window(qapp, calibration=None)
    dialog = window.load_efficiency(stem + "_bins.txt")
    assert dialog.calibration_button.isEnabled()
    dialog.use_its_calibration()
    assert window._calibration == CAL and window._calibration_active
    assert not dialog.calibration_button.isEnabled()
    dialog.close()
    window.close()


def test_a_bad_file_is_reported_not_raised(qapp, monkeypatch):
    import main_window as mw

    shown = []
    monkeypatch.setattr(mw.QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[2]))
    window, _ = _window(qapp)
    assert window.load_efficiency(os.path.join(FIXTURES, "demo1.txt")) is None
    assert shown and "not an efficiency" in shown[0]
    window.close()


def test_the_calibration_dialog_reads_a_saved_efficiency(qapp, tmp_path,
                                                         demo2):
    """The energy side of the request, where people look for it: the
    existing Load Calibration dialog now takes a saved efficiency and sets
    the kind as well as the numbers."""
    from calibration_dialog import CalibrationDialog

    fit, mc = demo2
    _result, stem = _save(tmp_path, fit, mc, "krf")
    dialog = CalibrationDialog(None, initial=Calibration("linear", 1.0, 2.0))
    assert not dialog._quadratic_radio.isChecked()
    dialog._on_load_file(stem + "_peaks.txt")
    assert dialog._quadratic_radio.isChecked()
    assert float(dialog._a_field.text()) == CAL.a
    assert float(dialog._b_field.text()) == CAL.b
    assert float(dialog._c_field.text()) == CAL.c
    dialog.close()


def test_the_calibration_dialog_still_reads_plain_coefficient_files(
        qapp, tmp_path):
    from calibration_dialog import CalibrationDialog

    path = tmp_path / "coeffs.txt"
    path.write_text("0.25\n0.5\n", encoding="utf-8")
    dialog = CalibrationDialog(None)
    dialog._linear_radio.setChecked(True)
    dialog._on_load_file(str(path))
    assert float(dialog._a_field.text()) == 0.25
    assert float(dialog._b_field.text()) == 0.5
    dialog.close()


def test_the_load_button_does_not_pass_its_checked_flag(qapp, monkeypatch):
    """clicked() delivers a bool; wired straight to _on_load_file it would
    arrive as path=False and open(False) reads standard input. The button
    must go through the file dialog instead."""
    import calibration_dialog as cd

    asked = []
    monkeypatch.setattr(cd.QFileDialog, "getOpenFileName",
                        lambda *a, **k: asked.append(True) or ("", ""))
    # If the lambda ever regresses, the False path ends in a modal warning
    # box -- and a modal nobody can click hangs a headless run instead of
    # failing it. Neutralised so a regression FAILS here.
    monkeypatch.setattr(cd.QMessageBox, "warning", lambda *a, **k: None)
    dialog = cd.CalibrationDialog(None)
    dialog._load_button.click()
    assert asked, "the button did not open the file dialog"
    dialog.close()
