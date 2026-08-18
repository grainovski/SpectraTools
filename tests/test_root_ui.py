"""ROOT open flow: the picker dialog, and loading its choices into the
main window."""

import numpy as np
import pytest

from main_window import MainWindow
from root_dialog import RootObjectDialog

uproot = pytest.importorskip("uproot")


@pytest.fixture
def root_file(tmp_path):
    path = tmp_path / "acq.root"
    counts = np.zeros(80)
    counts[38:43] = [5, 25, 60, 22, 4]
    matrix = np.zeros((12, 20))
    matrix[3:6, :] = 9.0
    with uproot.recreate(path) as f:
        f["Energy/chan0"] = (counts, np.linspace(0.0, 80.0, 81))
        f["Energy/chan1"] = (counts * 2, np.linspace(0.0, 80.0, 81))
        # A genuinely calibrated axis: 0.5 keV per channel from 100 keV.
        f["Energy/kev"] = (counts, np.linspace(100.0, 140.0, 81))
        f["EE/matrix"] = (matrix.T, np.linspace(0.0, 20.0, 21),
                          np.linspace(0.0, 12.0, 13))
    return path


def _objects(path):
    import root_io
    return root_io.list_objects(path)


# --- the picker --------------------------------------------------------


def test_dialog_lists_every_histogram_with_what_it_opens_as(qapp, root_file):
    dialog = RootObjectDialog(None, str(root_file), _objects(root_file))
    rows = [
        (dialog.tree.topLevelItem(i).text(0), dialog.tree.topLevelItem(i).text(2))
        for i in range(dialog.tree.topLevelItemCount())
    ]
    labels = dict(rows)
    assert labels["Energy/chan0"] == "Spectrum"
    assert labels["EE/matrix"] == "Matrix"
    # ROOT's cycle suffix is part of the real key but noise to a user who
    # has never heard of cycles, so it is stripped from the display only.
    assert all(";" not in name for name, _ in rows)


def test_dialog_refuses_a_mixed_selection(qapp, root_file):
    """A matrix opens its own panel while spectra join the spectrum list,
    so one OK cannot do both."""
    dialog = RootObjectDialog(None, str(root_file), _objects(root_file))
    dialog.tree.selectAll()
    dialog._on_accept()

    assert dialog.result() != RootObjectDialog.DialogCode.Accepted
    assert "not both" in dialog.status.text()
    assert dialog.result_matrix is None
    assert dialog.result_spectra == []


def test_dialog_refuses_two_matrices(qapp, tmp_path):
    path = tmp_path / "two.root"
    m = np.ones((4, 5))
    with uproot.recreate(path) as f:
        f["a"] = (m.T, np.linspace(0, 5, 6), np.linspace(0, 4, 5))
        f["b"] = (m.T, np.linspace(0, 5, 6), np.linspace(0, 4, 5))
    dialog = RootObjectDialog(None, str(path), _objects(path))
    dialog.tree.selectAll()
    dialog._on_accept()
    assert "one matrix" in dialog.status.text()


def test_dialog_requires_a_selection(qapp, root_file):
    dialog = RootObjectDialog(None, str(root_file), _objects(root_file))
    dialog.tree.clearSelection()
    dialog._on_accept()
    assert "at least one" in dialog.status.text()


def test_dialog_returns_the_full_key_including_the_cycle(qapp, root_file):
    """The display strips ';1' but the LOOKUP needs it -- returning the
    stripped name would fail to find the object again."""
    dialog = RootObjectDialog(None, str(root_file), _objects(root_file))
    item = next(
        dialog.tree.topLevelItem(i) for i in range(dialog.tree.topLevelItemCount())
        if dialog.tree.topLevelItem(i).text(0) == "Energy/chan0"
    )
    dialog.tree.setCurrentItem(item)
    item.setSelected(True)
    dialog._on_accept()
    assert dialog.result_spectra == ["Energy/chan0;1"]


# --- loading into the window -------------------------------------------


def test_loading_several_spectra_from_one_file(qapp, root_file):
    main_window = MainWindow()
    main_window._load_root_spectra(str(root_file), ["Energy/chan0;1", "Energy/chan1;1"])

    assert len(main_window.spectra) == 2
    # One file yields several spectra, so the path alone cannot identify
    # them -- the object name has to be part of it or the second would be
    # treated as a duplicate of the first.
    assert main_window.spectra[0].path != main_window.spectra[1].path
    assert all("acq.root::" in s.path for s in main_window.spectra)
    assert main_window.spectra[0].active is True
    assert int(main_window.spectra[1].data.sum()) == 2 * int(
        main_window.spectra[0].data.sum()
    )


def test_loading_the_same_object_twice_is_ignored(qapp, root_file):
    main_window = MainWindow()
    main_window._load_root_spectra(str(root_file), ["Energy/chan0;1"])
    main_window._load_root_spectra(str(root_file), ["Energy/chan0;1"])
    assert len(main_window.spectra) == 1


def test_an_embedded_axis_calibration_is_applied_once(qapp, root_file):
    """Same rule as the N42 reader: the first calibration found wins, and
    it is applied after every spectrum is in the list rather than
    mid-load."""
    main_window = MainWindow()
    assert main_window._calibration_active is False

    main_window._load_root_spectra(str(root_file), ["Energy/kev;1"])
    assert main_window._calibration_active is True
    assert main_window._calibration.kind == "linear"
    assert main_window._calibration.b == pytest.approx(0.5)
    assert main_window._calibration.a == pytest.approx(100.25)


def test_a_plain_channel_axis_does_not_activate_a_calibration(qapp, root_file):
    main_window = MainWindow()
    main_window._load_root_spectra(str(root_file), ["Energy/chan0;1"])
    assert main_window._calibration_active is False


def test_opening_a_root_matrix_creates_a_panel(qapp, root_file):
    main_window = MainWindow()
    before = len(main_window._matrix_panels)
    main_window._open_root_matrix(str(root_file), "EE/matrix;1")
    assert len(main_window._matrix_panels) == before + 1

    panel = main_window._matrix_panels[-1]
    assert panel.matrix.shape == (12, 20)
    # Rows 3-5 hold 9 counts each; a projection on x sums over rows, so
    # every column totals 27. Wrong-way-round loading would give a
    # different shape here, which is the point of a non-square fixture.
    from matrix_cut import compute_projection
    assert compute_projection(panel.matrix, "x") == pytest.approx(np.full(20, 27.0))


def test_a_failed_object_is_reported_not_silently_skipped(qapp, root_file, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **kwargs: warned.append(args[2]))

    main_window = MainWindow()
    main_window._load_root_spectra(str(root_file), ["Energy/absent;1"])
    assert len(main_window.spectra) == 0
    assert warned and "absent" in warned[0]


# --- v4.0.1: the selected row is actually visible -----------------------


def test_the_first_row_is_selected_and_the_list_holds_focus(qapp, root_file):
    """The row was always selected -- OK would have loaded it -- but Qt
    draws a selection in its INACTIVE palette when the widget has no
    focus, a pale grey that reads as nothing being selected at all."""
    dialog = RootObjectDialog(None, str(root_file), _objects(root_file))

    first = dialog.tree.topLevelItem(0)
    assert dialog.tree.currentItem() is first
    assert first.isSelected()
    # focusWidget() rather than hasFocus(): a dialog that has not been
    # shown holds no real focus offscreen, but it still records which
    # widget WILL take it. Shown, hasFocus() is True too -- asserted below
    # so this cannot pass on intent alone.
    assert dialog.focusWidget() is dialog.tree
    dialog.show()
    qapp.processEvents()
    assert dialog.tree.hasFocus(), "an unfocused list greys out its own selection"


def test_the_dark_theme_styles_tree_widgets_at_all(qapp):
    """The ROOT picker is the app's only QTreeWidget, and the dark
    stylesheet covered QTableWidget and QListWidget but not QTreeWidget --
    so the picker fell back to unstyled defaults and its highlight was not
    the theme's highlight colour.
    """
    from theme import qt_stylesheet

    dark = qt_stylesheet("dark")
    assert "QTreeWidget {" in dark or "QTreeWidget," in dark or ", QTreeWidget" in dark, \
        "QTreeWidget is not styled at all in the dark theme"
    assert "QTreeWidget::item:selected" in dark
    # And the inactive state, for when the list does not hold focus.
    assert ":selected:!active" in dark


def test_light_theme_is_still_qt_default(qapp):
    """The light theme deliberately ships no stylesheet -- Qt's own
    defaults. The tree fix must not have introduced one."""
    from theme import qt_stylesheet

    assert qt_stylesheet("light") == ""
