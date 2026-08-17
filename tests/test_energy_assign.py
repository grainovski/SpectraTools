"""X3: assign known energies to fitted peaks, then calibrate from them."""

import numpy as np
import pytest

from energy_assign_dialog import EnergyAssignDialog
from main_window import MainWindow
from peak_fit import fit_peaks


def _spectrum_with_two_peaks(tmp_path):
    x = np.arange(600, dtype=float)
    clean = np.full(600, 30.0)
    for centre, amplitude in ((150.0, 900.0), (420.0, 700.0)):
        clean = clean + amplitude * np.exp(-((x - centre) ** 2) / (2 * 4.0 ** 2))
    y = np.random.default_rng(9).poisson(clean).astype(float)
    path = tmp_path / "src.txt"
    path.write_text("\n".join(str(int(v)) for v in y), encoding="utf-8")
    return path, x, y


def _window_with_two_fits(tmp_path):
    path, x, y = _spectrum_with_two_peaks(tmp_path)
    main_window = MainWindow()
    main_window._load_files([str(path)])
    active = main_window.spectra[0]
    data = active.data
    axis = np.arange(len(data), dtype=float)
    for centre in (150.0, 420.0):
        active.fits.append(
            fit_peaks(axis, data, (centre - 90, centre - 50),
                      (centre + 50, centre + 90), (centre - 30, centre + 30), [centre])
        )
    return main_window, active


def test_peak_choices_are_listed_in_channels_even_when_calibrated(qapp, tmp_path):
    """The user is assigning energies in order to DETERMINE the
    calibration, so offering positions a previous calibration already
    converted would be circular."""
    from calibration import Calibration

    main_window, _ = _window_with_two_fits(tmp_path)
    main_window._apply_calibration_change(Calibration(kind="linear", a=5.0, b=3.0), True)

    choices = main_window.fitted_peak_choices()
    assert len(choices) == 2
    assert choices[0][1] == pytest.approx(150.0, abs=2.0)
    assert choices[1][1] == pytest.approx(420.0, abs=2.0)


def test_the_action_needs_at_least_two_fitted_peaks(qapp, tmp_path):
    main_window = MainWindow()
    main_window._update_operations_availability()
    assert main_window.calibrate_from_peaks_action.isEnabled() is False

    main_window, active = _window_with_two_fits(tmp_path)
    main_window._update_operations_availability()
    assert main_window.calibrate_from_peaks_action.isEnabled() is True

    del active.fits[1]
    main_window._update_operations_availability()
    assert main_window.calibrate_from_peaks_action.isEnabled() is False


def test_dialog_builds_a_calibration_from_the_typed_energies(qapp):
    dialog = EnergyAssignDialog(None, [("Fit 1, peak 1", 100.0), ("Fit 2, peak 1", 500.0)])
    dialog.table.item(0, EnergyAssignDialog.ENERGY_COLUMN).setText("60")
    dialog.table.item(1, EnergyAssignDialog.ENERGY_COLUMN).setText("260")
    dialog._on_accept()

    assert dialog.result_calibration is not None
    assert dialog.result_calibration.a == pytest.approx(10.0)
    assert dialog.result_calibration.b == pytest.approx(0.5)


def test_blank_rows_are_skipped_but_unparseable_ones_are_reported(qapp):
    """A typo silently dropped would produce a calibration quietly fitted
    to fewer points than the user believes."""
    dialog = EnergyAssignDialog(
        None,
        [("a", 100.0), ("b", 300.0), ("c", 500.0)],
    )
    dialog.table.item(0, EnergyAssignDialog.ENERGY_COLUMN).setText("60")
    dialog.table.item(1, EnergyAssignDialog.ENERGY_COLUMN).setText("   ")
    dialog.table.item(2, EnergyAssignDialog.ENERGY_COLUMN).setText("260")
    channels, energies = dialog.assignments()
    assert channels == [100.0, 500.0]
    assert energies == [60.0, 260.0]

    dialog.table.item(1, EnergyAssignDialog.ENERGY_COLUMN).setText("66l")
    dialog._on_accept()
    assert dialog.result_calibration is None
    assert "not a number" in dialog.status.text()
    assert "b:" in dialog.status.text()


def test_too_few_assignments_is_reported_rather_than_accepted(qapp):
    dialog = EnergyAssignDialog(None, [("a", 100.0), ("b", 300.0)])
    dialog.table.item(0, EnergyAssignDialog.ENERGY_COLUMN).setText("60")
    dialog._on_accept()
    assert dialog.result_calibration is None
    assert "at least 2" in dialog.status.text()


def test_the_dialog_records_the_worst_residual(qapp):
    """The number that says whether to believe the calibration; the
    coefficients alone look equally plausible either way."""
    dialog = EnergyAssignDialog(
        None, [("a", 100.0), ("b", 300.0), ("c", 700.0)]
    )
    for row, energy in enumerate(("60", "160", "400")):  # 400 should be 360
        dialog.table.item(row, EnergyAssignDialog.ENERGY_COLUMN).setText(energy)
    dialog._on_accept()

    assert dialog.result_calibration is not None
    assert dialog._worst_residual > 5.0


def test_calibrating_from_peaks_applies_the_result(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    main_window, active = _window_with_two_fits(tmp_path)
    positions = [p.position for f in active.fits for p in f.peaks]

    def fake_exec(self):
        # Assign energies consistent with E = 20 + 1.5*channel.
        for row, channel in enumerate(positions):
            self.table.item(row, EnergyAssignDialog.ENERGY_COLUMN).setText(
                str(20.0 + 1.5 * channel)
            )
        self._on_accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(EnergyAssignDialog, "exec", fake_exec)
    main_window._open_calibrate_from_peaks_dialog()

    assert main_window._calibration_active is True
    assert main_window._calibration.a == pytest.approx(20.0, abs=1e-6)
    assert main_window._calibration.b == pytest.approx(1.5, abs=1e-9)
