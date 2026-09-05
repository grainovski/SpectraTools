"""End-to-end through the main window: the Operations action, fits
appended to whatever is there, and the Calibrate from Fitted Peaks
dialog opening prefilled -- or unassigned with the reason on show when
the identification was refused.

Both dialogs are driven without a human, the way test_energy_assign.py
drives its one: exec() is replaced so the automatic dialog loads a
source and runs, and the assign dialog is captured and cancelled.
"""

import os

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QDialog, QMenu, QPushButton

import auto_calibrate_dialog as auto_module
import energy_assign_dialog as assign_module
import fit_mode
from energy_assign_dialog import EnergyAssignDialog
from energy_assignments import EnergyAssignments
from main_window import MainWindow
from peak_fit import fit_peaks
from synthetic_calibration import fixture_sou, spectrum_from_source

ENERGY = EnergyAssignDialog.ENERGY_COLUMN
SHORTCUT = "Ctrl+Shift+L"

#: Eight well-separated Eu-152 lines: enough to identify, and far enough
#: apart that no fitted peak can have a twin, so every assignment has
#: exactly one row to land on.
EIGHT_LINES = [
    (121.7817, 2859.0), (244.6974, 760.0), (344.2785, 2659.0),
    (443.965, 313.0), (778.9045, 1293.0), (964.057, 1459.0),
    (1112.076, 1354.0), (1408.013, 2085.0),
]


def _write_sou(tmp_path, lines, name):
    path = tmp_path / name
    path.write_text(
        "".join(f"{e:12.4f}  .0100  {i:8.1f}  10.\n" for e, i in lines),
        encoding="utf-8",
    )
    return str(path)


def _window(tmp_path, sou_path, offset=12.5, gain=0.40):
    counts = spectrum_from_source(sou_path, offset, gain, channels=4096)
    path = tmp_path / "cal.txt"
    path.write_text("\n".join(str(int(v)) for v in counts), encoding="utf-8")
    window = MainWindow()
    window._load_files([str(path)])
    return window, window.spectra[0]


def _hand_fit(active, centre):
    axis = np.arange(len(active.data), dtype=float)
    return fit_peaks(
        axis, active.data, (centre - 60, centre - 35), (centre + 35, centre + 60),
        (centre - 20, centre + 20), [centre],
    )


def _run(window, monkeypatch, source_path=None):
    """Drive both dialogs. Returns (automatic dialogs, assign dialogs)
    that were constructed."""
    auto_dialogs, assign_dialogs = [], []

    def fake_exec(self):
        auto_dialogs.append(self)
        if source_path is not None:
            self.load_source(source_path)
        self._on_run()
        return (QDialog.DialogCode.Accepted if self.outcome is not None
                else QDialog.DialogCode.Rejected)

    monkeypatch.setattr(auto_module.AutoCalibrateDialog, "exec", fake_exec)

    class _Spy(assign_module.EnergyAssignDialog):
        def __init__(self, parent, peaks, **kwargs):
            super().__init__(parent, peaks, **kwargs)
            assign_dialogs.append(self)

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(assign_module, "EnergyAssignDialog", _Spy)
    window._open_auto_calibrate_dialog()
    return auto_dialogs, assign_dialogs


def _menu_named(window, title):
    # findChildren rather than QAction.menu(): see test_operations_menu.py
    # for the PySide6 lifetime quirk that makes the latter unreliable.
    for menu in window.findChildren(QMenu):
        if menu.title() == title:
            return menu
    return None


def _energies_by_channel(dialog):
    out = {}
    for row in range(dialog.table.rowCount()):
        channel = float(dialog.table.item(row, 1).text())
        text = dialog.table.item(row, ENERGY).text().strip()
        out[channel] = float(text) if text else None
    return out


# --- the action --------------------------------------------------------


def test_the_action_follows_calibrate_from_fitted_peaks_in_the_menu(qapp):
    window = MainWindow()
    menu = _menu_named(window, "&Operations")
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    position = texts.index("Automatic Calibration...")
    assert texts[position - 1] == "Calibrate from Fitted Peaks..."
    assert window.auto_calibrate_action.shortcut() == QKeySequence(SHORTCUT)


def test_the_shortcut_is_free_and_is_not_a_marking_key(qapp):
    """Every other shortcut in the window, on actions and on the toolbar
    buttons alike, must differ -- and the letter must not be one the
    canvas treats as a held marking key (B/R/P)."""
    window = MainWindow()
    wanted = QKeySequence(SHORTCUT)
    on_actions = [a for a in window.findChildren(QAction) if a.shortcut() == wanted]
    assert on_actions == [window.auto_calibrate_action]
    on_buttons = [b for b in window.findChildren(QPushButton) if b.shortcut() == wanted]
    assert on_buttons == []
    assert Qt.Key.Key_L not in fit_mode._KEY_TO_MARK_TYPE


def test_the_action_needs_a_spectrum_and_no_fits(qapp, tmp_path):
    window = MainWindow()
    window._update_operations_availability()
    assert window.auto_calibrate_action.isEnabled() is False
    window, active = _window(tmp_path, _write_sou(tmp_path, EIGHT_LINES, "eight.sou"))
    assert active.fits == []
    assert window.auto_calibrate_action.isEnabled() is True


# --- what a run does to the spectrum -----------------------------------


def test_fits_are_appended_and_a_hand_fit_is_kept(qapp, tmp_path, monkeypatch):
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    hand = _hand_fit(active, (121.7817 - 12.5) / 0.40)
    active.fits.append(hand)
    window._plot_data()

    auto_dialogs, assign_dialogs = _run(window, monkeypatch, source)
    outcome = auto_dialogs[0].outcome
    assert outcome.match.ok, outcome.match.reason

    assert active.fits[0] is hand
    assert len(active.fits) == 1 + len(outcome.results)
    assert active.fits[1:] == outcome.results
    assert all(r.timestamp for r in outcome.results)
    # The panel shows one row per fitted peak, hand and automatic alike.
    rows = window.fit_controller.results_table.rowCount()
    assert rows == len(window.fitted_peak_choices()) == 1 + len(outcome.peaks)
    assert "1 fit already on the spectrum was kept" in window.statusBar().currentMessage()
    # Committed through the same path as the Fit button, so every one is
    # in the automatic fit log next to the spectrum.
    log = tmp_path / "cal_fits.jsonl"
    assert log.exists()
    assert len(log.read_text(encoding="utf-8").splitlines()) == len(outcome.results)


def test_the_energy_goes_to_the_automatic_copy_of_a_peak_fitted_by_hand(
    qapp, tmp_path, monkeypatch
):
    """The hand fit and the automatic fit of the same peak are two rows a
    hundredth of a channel apart. The identification is stored on the
    automatic centroid, and lands there rather than being refused as a
    coin flip between the two."""
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    hand = _hand_fit(active, (121.7817 - 12.5) / 0.40)
    active.fits.append(hand)
    _auto, assign_dialogs = _run(window, monkeypatch, source)
    dialog = assign_dialogs[0]
    hand_row = dialog.table.item(0, ENERGY).text().strip()
    assert hand_row == ""
    filled = {c: e for c, e in _energies_by_channel(dialog).items() if e is not None}
    assert any(abs(e - 121.7817) < 1e-6 for e in filled.values())


# --- the dialog that opens ---------------------------------------------


def test_the_assign_dialog_opens_prefilled_with_the_source_and_plot(
    qapp, tmp_path, monkeypatch
):
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    auto_dialogs, assign_dialogs = _run(window, monkeypatch, source)
    outcome = auto_dialogs[0].outcome
    assert outcome.match.ok, outcome.match.reason
    assert len(outcome.pairs) >= 6

    assert len(assign_dialogs) == 1
    dialog = assign_dialogs[0]
    assert dialog.source_lines is not None
    assert len(dialog.source_lines) == len(EIGHT_LINES)
    assert "eight.sou" in dialog.source_label.text()
    # Every identified peak's row carries its energy -- these are the very
    # centroids just fitted, so each lands on its own row exactly.
    by_channel = _energies_by_channel(dialog)
    for channel, energy in outcome.pairs:
        assert by_channel[round(channel, 3)] == pytest.approx(energy), channel
    assert sum(1 for e in by_channel.values() if e is not None) == len(outcome.pairs)
    # The live plot is up: that is the calibration plot the user reviews.
    assert dialog._live_plot is not None
    # The summary is the first thing the status line says.
    assert f"{len(outcome.pairs)} identified in eight.sou" in dialog.status.text()
    # Stored on the spectrum already, so cancelling loses nothing.
    assert active.energy_assignments.pairs == tuple(outcome.pairs)
    assert active.energy_assignments.source_path == source


def test_a_refused_identification_opens_the_dialog_unassigned_with_the_reason(
    qapp, tmp_path, monkeypatch
):
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    wrong = fixture_sou("am241.sou")
    auto_dialogs, assign_dialogs = _run(window, monkeypatch, wrong)
    outcome = auto_dialogs[0].outcome
    assert not outcome.match.ok
    assert outcome.match.reason

    # The fits are committed all the same.
    assert len(active.fits) == len(outcome.results) > 0
    dialog = assign_dialogs[0]
    assert all(e is None for e in _energies_by_channel(dialog).values())
    assert dialog.source_lines is not None
    assert "am241.sou" in dialog.source_label.text()
    assert "none identified" in dialog.status.text()
    assert outcome.match.reason in dialog.status.text()
    assert outcome.match.reason in window.statusBar().currentMessage()
    # Nothing to restore next time, but the source is remembered.
    assert active.energy_assignments.pairs == ()
    assert active.energy_assignments.source_path == wrong


def test_cancelling_the_automatic_dialog_changes_nothing(qapp, tmp_path, monkeypatch):
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    opened = []

    monkeypatch.setattr(auto_module.AutoCalibrateDialog, "exec",
                        lambda self: QDialog.DialogCode.Rejected)
    monkeypatch.setattr(assign_module, "EnergyAssignDialog",
                        lambda *a, **k: opened.append(a))
    window._open_auto_calibrate_dialog()
    assert active.fits == []
    assert active.energy_assignments is None
    assert opened == []


def test_a_source_already_chosen_for_the_spectrum_is_offered(qapp, tmp_path, monkeypatch):
    """The manual dialog remembers the source per spectrum; the automatic
    one starts from the same memory, so Run works without picking it
    again."""
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    active.energy_assignments = EnergyAssignments(source_path=source, pairs=())
    auto_dialogs, assign_dialogs = _run(window, monkeypatch, source_path=None)
    assert auto_dialogs[0].source_name() == "eight.sou"
    assert auto_dialogs[0].outcome.match.ok
    assert len(assign_dialogs) == 1


def test_the_variance_of_a_derived_spectrum_reaches_the_fits(qapp, tmp_path, monkeypatch):
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    active.variance = 2.0 * np.maximum(active.data, 1.0)
    seen = []
    real = auto_module.auto_calibrate.calibrate

    def spy(x, y, lines, **kwargs):
        seen.append(kwargs.get("variance"))
        return real(x, y, lines, **kwargs)

    monkeypatch.setattr(auto_module.auto_calibrate, "calibrate", spy)
    _run(window, monkeypatch, source)
    assert len(seen) == 1
    assert seen[0] is active.variance


# --- after OK: the refit for CalEnEff ----------------------------------


def _run_and_accept(window, monkeypatch, source_path, before_accept=None):
    """Drive both dialogs and ACCEPT the assign dialog, as a user happy
    with the plot does. `before_accept` is handed the assign dialog first,
    for a user who unticks a row before pressing OK. Returns (automatic
    dialog, assign dialog)."""
    auto_dialogs, assign_dialogs = [], []

    def fake_exec(self):
        auto_dialogs.append(self)
        self.load_source(source_path)
        self._on_run()
        return (QDialog.DialogCode.Accepted if self.outcome is not None
                else QDialog.DialogCode.Rejected)

    monkeypatch.setattr(auto_module.AutoCalibrateDialog, "exec", fake_exec)

    class _Accept(assign_module.EnergyAssignDialog):
        def __init__(self, parent, peaks, **kwargs):
            super().__init__(parent, peaks, **kwargs)
            assign_dialogs.append(self)

        def exec(self):
            if before_accept is not None:
                before_accept(self)
            self._on_accept()
            return (QDialog.DialogCode.Accepted if self.result_calibration is not None
                    else QDialog.DialogCode.Rejected)

    monkeypatch.setattr(assign_module, "EnergyAssignDialog", _Accept)
    window._open_auto_calibrate_dialog()
    return auto_dialogs[0], assign_dialogs[0]


def test_ok_refits_every_visible_line_and_replaces_the_first_pass(qapp, tmp_path, monkeypatch):
    """What CalEnEff needs is one clean fit per source line. After OK the
    identification pass's fits are replaced by a refit at every visible
    line; a fit made by hand beforehand is untouched."""
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    hand = _hand_fit(active, (121.7817 - 12.5) / 0.40)
    active.fits.append(hand)

    auto, assign = _run_and_accept(window, monkeypatch, source)
    outcome = auto.outcome
    assert outcome.match.ok, outcome.match.reason
    assert assign.result_calibration is not None

    assert active.fits[0] is hand
    first_pass = {id(result) for result in outcome.results}
    assert not any(id(f) in first_pass for f in active.fits), "first-pass fits were kept"
    refit = active.fits[1:]
    assert len(refit) >= 6
    assert all(len(r.peaks) == 1 and r.timestamp for r in refit)

    stored = active.energy_assignments
    assert stored.source_path == source
    assert len(stored.pairs) == len(refit)
    energies = sorted(e for _c, e in stored.pairs)
    assert all(any(abs(e - line) < 1e-6 for line, _i in EIGHT_LINES) for e in energies)
    assert len(set(energies)) == len(energies)

    assert window._calibration is not None and window._calibration_active
    plot = window._calibration_plot
    assert plot is not None
    assert len(plot._points) == len(refit)
    assert "refitted for CalEnEff" in window.statusBar().currentMessage()


def test_cancelling_the_review_keeps_the_first_pass_and_refits_nothing(qapp, tmp_path, monkeypatch):
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    auto_dialogs, _assign = _run(window, monkeypatch, source)   # _run REJECTS the review
    outcome = auto_dialogs[0].outcome
    assert active.fits == outcome.results
    assert window._calibration is None
    assert active.energy_assignments.pairs == tuple(outcome.pairs)


def test_a_row_unticked_before_ok_is_left_out_of_the_refit(qapp, tmp_path, monkeypatch):
    """The tick decides two things now: whether the point anchors the
    energy fit, and whether its area is measured for CalEnEff."""
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)

    dropped = []

    def untick_the_third(dialog):
        for row in range(dialog.table.rowCount()):
            text = dialog.table.item(row, ENERGY).text().strip()
            if text and abs(float(text) - 344.2785) < 1e-3:
                dialog.table.item(row, EnergyAssignDialog.INCLUDE_COLUMN)\
                    .setCheckState(Qt.CheckState.Unchecked)
                dropped.append(float(text))
                return

    _auto, assign = _run_and_accept(window, monkeypatch, source,
                                    before_accept=untick_the_third)
    assert dropped == [344.2785], "the row to untick was never found"
    assert assign.result_calibration is not None

    energies = [e for _c, e in active.energy_assignments.pairs]
    assert not any(abs(e - 344.2785) < 1e-6 for e in energies), \
        "the unticked line was refitted anyway"
    assert any(abs(e - 121.7817) < 1e-6 for e in energies), "nothing was refitted"
    assert 344.2785 in tuple(active.energy_assignments.excluded)
    assert "left unticked in the calibration" in window.statusBar().currentMessage()
    plot = window._calibration_plot
    assert all(abs(p[4] - 344.2785) > 1e-6 for p in plot._points)


def test_with_nothing_unticked_that_line_is_refitted(qapp, tmp_path, monkeypatch):
    """CONTROL for the test above."""
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    _auto, _assign = _run_and_accept(window, monkeypatch, source)
    energies = [e for _c, e in active.energy_assignments.pairs]
    assert any(abs(e - 344.2785) < 1e-6 for e in energies)


# --- the fit log gets one record per fit the user keeps ------------------


def _log_records(active):
    import fit_export

    path = fit_export.auto_log_path(active.path)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [line for line in handle.read().splitlines() if line.strip()]


def test_the_fit_log_records_each_refitted_line_once(qapp, tmp_path, monkeypatch):
    """The identification pass is replaced wholesale by the refit, so
    logging both would put two records of every line in the file the
    HowTo tells people to process with other tools -- the same double
    count the in-memory list goes to some trouble to avoid."""
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    _auto, _assign = _run_and_accept(window, monkeypatch, source)

    records = _log_records(active)
    assert records, "nothing was logged at all"
    assert len(records) == len(active.fits)


def test_cancelling_the_review_still_logs_the_fits_that_are_kept(qapp, tmp_path, monkeypatch):
    """The other half: with no refit coming, the identification pass IS
    what the user keeps, so it must reach the log -- once."""
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    _auto, _assign = _run(window, monkeypatch, source)      # cancels the review

    assert active.fits, "the fits should have been kept"
    assert len(_log_records(active)) == len(active.fits)


def test_a_hand_fit_made_before_the_run_is_logged_once_and_kept(qapp, tmp_path, monkeypatch):
    """CONTROL: the run must not disturb what was already there. A fit
    committed by hand goes through the ordinary path and is logged by
    it, so the count includes it exactly once."""
    source = _write_sou(tmp_path, EIGHT_LINES, "eight.sou")
    window, active = _window(tmp_path, source)
    hand = _hand_fit(active, (121.7817 - 12.5) / 0.40)
    active.fits.append(hand)
    window.fit_controller.append_auto_log(active, hand)
    before = len(_log_records(active))
    assert before == 1

    _auto, _assign = _run_and_accept(window, monkeypatch, source)
    assert active.fits[0] is hand
    assert len(_log_records(active)) == len(active.fits)
