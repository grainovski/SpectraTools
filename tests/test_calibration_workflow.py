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


def test_clear_forgets_the_source_path_too(qapp, tmp_path):
    """Clear promises to forget the loaded source. Leaving the path behind
    meant the discarded nuclide came back on the next visit."""
    window, active = _window(tmp_path)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    sou = tmp_path / "wrong.sou"
    sou.write_text("100.0 .01 1000. 10.\n200.0 .01 900. 9.\n", encoding="utf-8")
    assert dialog.load_source(str(sou))
    assert dialog.source_path() is not None

    dialog.clear_button.click()
    assert dialog.source_path() is None


def test_calibrating_opens_the_plot_window(qapp, tmp_path, monkeypatch):
    import main_window as module

    window, active = _window(tmp_path)
    opened = {}

    class _Spy:
        def __init__(self, *args, **kwargs):
            opened["args"] = args
        def show(self):
            opened["shown"] = True
        def setAttribute(self, *args, **kwargs):
            pass

    monkeypatch.setattr(module, "CalibrationPlotDialog", _Spy)

    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    dialog._on_accept()
    window._show_calibration_plot(active, dialog)

    assert opened.get("shown") is True


def test_clearing_then_retyping_keeps_the_new_assignments(qapp, tmp_path):
    """Clear exists so the user can start over. An earlier version latched
    a `cleared` flag on and never reset it, so the freshly typed energies
    were computed into a calibration and then thrown away.
    """
    window, active = _window(tmp_path)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    dialog._on_accept()
    window._store_energy_assignments(active, dialog)
    assert active.energy_assignments is not None

    dialog.clear_button.click()
    assert dialog.cleared is True
    dialog.table.item(0, ENERGY).setText("310")
    dialog.table.item(1, ENERGY).setText("850")
    dialog._on_accept()
    window._store_energy_assignments(active, dialog)

    assert active.energy_assignments is not None
    assert [e for _c, e in active.energy_assignments.pairs] == [310.0, 850.0]


# --- the live preview, refreshed after every recalculation --------------


def _live_dialog(window, tmp_path, **kwargs):
    """A dialog wired for the live preview, as main_window builds it."""
    return EnergyAssignDialog(
        None, window.fitted_peak_choices(),
        max_channel=len(window.spectra[0].data) - 1,
        export_default_path=str(tmp_path / "s_En_Area.txt"),
        **kwargs
    )


def test_the_plot_appears_as_soon_as_the_points_make_a_calibration(qapp, tmp_path):
    """Not only on OK: the residual strip is what shows a misidentified
    line, and it is worth nothing after the calibration is already
    applied."""
    window, _ = _window(tmp_path)
    dialog = _live_dialog(window, tmp_path)
    assert dialog._live_plot is None, "nothing assigned yet"

    dialog.table.item(0, ENERGY).setText("300")
    assert dialog._live_plot is None, "one point cannot fix a line"

    dialog.table.item(1, ENERGY).setText("840")
    assert dialog._live_plot is not None
    assert dialog._live_plot.isVisible()


def test_editing_a_point_redraws_the_same_window(qapp, tmp_path):
    """Recreating it would raise it to the front on every keystroke and
    take focus off the table being typed into."""
    window, _ = _window(tmp_path)
    dialog = _live_dialog(window, tmp_path)
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    first = dialog._live_plot

    dialog.table.item(1, ENERGY).setText("850")
    assert dialog._live_plot is first, "the window was rebuilt, not redrawn"


def test_editing_a_point_actually_changes_the_curve(qapp, tmp_path):
    """The control for the test above: proving the window is reused is
    worthless unless the redraw really uses the new numbers."""
    window, _ = _window(tmp_path)
    dialog = _live_dialog(window, tmp_path)
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    before = dialog._live_plot._calibration.b

    dialog.table.item(1, ENERGY).setText("1680")
    after = dialog._live_plot._calibration.b
    assert after != pytest.approx(before), "the slope did not follow the edit"


def test_clear_takes_the_plot_down(qapp, tmp_path):
    """A curve left on screen after Clear would claim to describe a table
    that is now empty."""
    window, _ = _window(tmp_path)
    dialog = _live_dialog(window, tmp_path)
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    assert dialog._live_plot is not None

    dialog._on_clear()
    assert dialog._live_plot is None


def test_restored_assignments_bring_the_plot_straight_back(qapp, tmp_path):
    window, active = _window(tmp_path)
    choices = window.fitted_peak_choices()
    dialog = _live_dialog(
        window, tmp_path,
        assignments=EnergyAssignments(
            None, ((choices[0][1], 300.0), (choices[1][1], 840.0))
        ),
    )
    assert dialog._live_plot is not None


def test_no_live_plot_without_a_channel_range(qapp, tmp_path):
    """Every caller that does not ask for one is unaffected."""
    window, _ = _window(tmp_path)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    assert dialog._live_plot is None


def test_ok_closes_the_preview(qapp, tmp_path):
    """main_window opens its own window once the calibration is applied;
    two plots of the same fit on screen would be one too many."""
    window, _ = _window(tmp_path)
    dialog = _live_dialog(window, tmp_path)
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    assert dialog._live_plot is not None

    dialog._on_accept()
    assert dialog._live_plot is None
    assert dialog.result_calibration is not None


# --- v5.0.2: including / excluding a point from the fit ------------------

INCLUDE = EnergyAssignDialog.INCLUDE_COLUMN


def _untick(dialog, row):
    from PySide6.QtCore import Qt
    dialog.table.item(row, INCLUDE).setCheckState(Qt.CheckState.Unchecked)


def _is_ticked(dialog, row):
    from PySide6.QtCore import Qt
    return dialog.table.item(row, INCLUDE).checkState() == Qt.CheckState.Checked


def test_every_point_starts_included(qapp, tmp_path):
    window, _ = _window(tmp_path)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    assert all(_is_ticked(dialog, r) for r in range(dialog.table.rowCount()))


def test_unticking_drops_the_point_from_the_fit_but_keeps_its_energy(qapp, tmp_path):
    """The energy is what the user typed; the tick only says whether the
    calibration is fitted through it."""
    window, _ = _window(tmp_path)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    _untick(dialog, 1)

    channels, energies, _ = dialog.fit_assignments()
    assert energies == [300.0], "the unticked point still reached the fit"

    all_channels, all_energies, _ = dialog.assignments()
    assert all_energies == [300.0, 840.0], "the unticked point lost its energy"
    assert dialog.excluded_energies() == (840.0,)


def test_unticking_actually_changes_the_calibration(qapp, tmp_path):
    """The control for the test above: proving the point is filtered out
    is worth nothing unless the fit that results is different."""
    window, active = _window(tmp_path, centres=(150.0, 300.0, 420.0))
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("600")
    dialog.table.item(2, ENERGY).setText("2000")      # a deliberate outlier
    with_outlier, _ = dialog._compute_calibration()

    _untick(dialog, 2)
    without, _ = dialog._compute_calibration()

    assert with_outlier is not None and without is not None
    assert without.b != pytest.approx(with_outlier.b), \
        "excluding the outlier left the slope unchanged"


def test_an_exclusion_survives_reopening_the_dialog(qapp, tmp_path):
    window, active = _window(tmp_path)
    choices = window.fitted_peak_choices()
    dialog = EnergyAssignDialog(None, choices)
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    _untick(dialog, 1)
    dialog._on_accept()
    window._store_energy_assignments(active, dialog)

    assert active.energy_assignments.excluded == (840.0,)

    reopened = EnergyAssignDialog(None, choices,
                                  assignments=active.energy_assignments)
    assert _is_ticked(reopened, 0)
    assert not _is_ticked(reopened, 1), "the outlier quietly rejoined the fit"
    assert float(reopened.table.item(1, ENERGY).text()) == pytest.approx(840.0)


def test_clear_puts_every_tick_back(qapp, tmp_path):
    window, _ = _window(tmp_path)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    _untick(dialog, 0)
    dialog._on_clear()
    assert all(_is_ticked(dialog, r) for r in range(dialog.table.rowCount()))


def test_an_excluded_point_is_still_exported_for_efficiency(qapp, tmp_path):
    """Deliberate: the tick is about the ENERGY fit. A point left out of
    the calibration still carries a usable area and intensity."""
    window, _ = _window(tmp_path)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    _untick(dialog, 1)
    exported = [p[4] for p in dialog.export_points()]
    assert sorted(exported) == [300.0, 840.0]


def test_unticking_refreshes_the_live_plot(qapp, tmp_path):
    window, _ = _window(tmp_path)
    dialog = _live_dialog(window, tmp_path)
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    assert dialog._live_plot is not None
    before = dialog._live_plot._calibration.b

    _untick(dialog, 1)
    # One point cannot fit a line, so the preview comes down rather than
    # showing a curve that no longer describes the table.
    assert dialog._live_plot is None or dialog._live_plot._calibration.b != before


def test_suggest_will_not_anchor_on_an_excluded_point(qapp, tmp_path):
    """Suggesting is itself an energy fit, so a point the user called
    suspect must not be allowed to place every other line."""
    window, _ = _window(tmp_path)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    assert dialog._anchor_count() == 2
    _untick(dialog, 1)
    assert dialog._anchor_count() == 1
    assert not dialog.suggest_button.isEnabled()
