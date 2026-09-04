"""End-to-end: assignments survive, Clear empties them, and calibrating
opens the plot window."""

import numpy as np
import pytest

from energy_assign_dialog import EnergyAssignDialog
from energy_assignments import EnergyAssignments
from main_window import MainWindow
from peak_fit import fit_peaks

ENERGY = EnergyAssignDialog.ENERGY_COLUMN


def _window(tmp_path, centres=(150.0, 420.0)):
    x = np.arange(600, dtype=float)
    clean = np.full(600, 30.0)
    for centre in centres:
        clean = clean + 900.0 * np.exp(-((x - centre) ** 2) / (2 * 4.0 ** 2))
    y = np.random.default_rng(5).poisson(clean).astype(float)
    path = tmp_path / "s.txt"
    path.write_text("\n".join(str(int(v)) for v in y), encoding="utf-8")

    window = MainWindow()
    window._load_files([str(path)])
    active = window.spectra[0]
    axis = np.arange(len(active.data), dtype=float)
    for centre in centres:
        active.fits.append(fit_peaks(
            axis, active.data, (centre - 90, centre - 50),
            (centre + 50, centre + 90), (centre - 30, centre + 30), [centre],
        ))
    return window, active


def test_accepting_stores_the_assignments_on_the_spectrum(qapp, tmp_path):
    window, active = _window(tmp_path)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    dialog._on_accept()
    window._store_energy_assignments(active, dialog)

    assert active.energy_assignments is not None
    assert len(active.energy_assignments.pairs) == 2


def test_reopening_restores_what_was_typed(qapp, tmp_path):
    window, active = _window(tmp_path)
    choices = window.fitted_peak_choices()
    active.energy_assignments = EnergyAssignments(
        None, ((choices[0][1], 300.0), (choices[1][1], 840.0))
    )
    dialog = EnergyAssignDialog(
        None, choices, assignments=active.energy_assignments
    )
    assert float(dialog.table.item(0, ENERGY).text()) == pytest.approx(300.0)
    assert float(dialog.table.item(1, ENERGY).text()) == pytest.approx(840.0)


def test_clear_empties_the_table_and_the_record(qapp, tmp_path):
    window, active = _window(tmp_path)
    choices = window.fitted_peak_choices()
    active.energy_assignments = EnergyAssignments(
        "eu152.sou", ((choices[0][1], 300.0),)
    )
    dialog = EnergyAssignDialog(
        None, choices, assignments=active.energy_assignments
    )
    assert dialog.table.item(0, ENERGY).text() != ""

    dialog.clear_button.click()
    assert dialog.table.item(0, ENERGY).text() == ""
    assert dialog.source_lines is None
    assert dialog.cleared is True


def test_clear_does_not_touch_the_active_calibration(qapp, tmp_path):
    from calibration import Calibration

    window, active = _window(tmp_path)
    window._apply_calibration_change(Calibration(kind="linear", a=1.0, b=2.0), True)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.clear_button.click()
    assert window._calibration is not None
    assert window._calibration_active is True


def test_calibrating_opens_the_plot_window(qapp, tmp_path, monkeypatch):
    import main_window as module

    window, active = _window(tmp_path)
    opened = {}

    class _Spy:
        def __init__(self, *args, **kwargs):
            opened["args"] = args
        def show(self):
            opened["shown"] = True

    monkeypatch.setattr(module, "CalibrationPlotDialog", _Spy)

    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    dialog._on_accept()
    window._show_calibration_plot(active, dialog)

    assert opened.get("shown") is True
