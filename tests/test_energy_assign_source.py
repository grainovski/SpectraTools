"""Source-assisted assignment in the Calibrate from Fitted Peaks dialog:
load a .sou nuclide, anchor two peaks by hand, let it suggest the rest.

Each test drives the dialog the way test_energy_assign.py does -- set
cell text, call the handlers directly -- rather than through the file
picker, which is the one piece that needs a human.
"""

import pytest

from energy_assign_dialog import EnergyAssignDialog

ENERGY = EnergyAssignDialog.ENERGY_COLUMN


def _sou(tmp_path, energies, name="src.sou"):
    path = tmp_path / name
    path.write_text(
        "".join(f"{e:12.4f}  .0100  1000.  10.\n" for e in energies),
        encoding="utf-8",
    )
    return str(path)


def _dialog(channels=(100.0, 200.0, 300.0, 400.0), settings=None):
    peaks = [(f"peak {i + 1}", ch) for i, ch in enumerate(channels)]
    return EnergyAssignDialog(None, peaks, settings=settings)


def _set(dialog, row, text):
    dialog.table.item(row, ENERGY).setText(text)


def _text(dialog, row):
    return dialog.table.item(row, ENERGY).text()


class _Settings:
    def __init__(self, folder=""):
        self.folder = folder
        self.remembered = []

    def last_folder(self):
        return self.folder

    def set_last_folder(self, folder):
        self.remembered.append(folder)


# --- loading a source --------------------------------------------------


def test_loading_a_source_names_it_and_counts_its_lines(qapp, tmp_path):
    dialog = _dialog()
    assert dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0], "eu152.sou"))
    assert dialog.source_lines is not None
    assert len(dialog.source_lines) == 3
    assert "eu152.sou" in dialog.source_label.text()
    assert "3 lines" in dialog.source_label.text()


def test_a_bad_source_is_reported_and_leaves_the_previous_one(qapp, tmp_path):
    dialog = _dialog()
    assert dialog.load_source(_sou(tmp_path, [60.0, 160.0], "good.sou"))
    bad = tmp_path / "bad.sou"
    bad.write_text("60 .1 1\n", encoding="utf-8")
    assert dialog.load_source(str(bad)) is False
    assert "bad.sou" in dialog.status.text()
    assert "found 3" in dialog.status.text()
    assert len(dialog.source_lines) == 2
    assert "good.sou" in dialog.source_label.text()


def test_loading_remembers_the_folder_like_the_spectrum_dialogs(qapp, tmp_path):
    settings = _Settings()
    dialog = _dialog(settings=settings)
    dialog.load_source(_sou(tmp_path, [60.0, 160.0]))
    assert settings.remembered == [str(tmp_path)]


# --- the two-anchor gate -----------------------------------------------


def test_suggest_needs_a_source_and_two_typed_energies(qapp, tmp_path):
    dialog = _dialog()
    assert dialog.suggest_button.isEnabled() is False

    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    # Two anchors but no source: still nothing to suggest from.
    assert dialog.suggest_button.isEnabled() is False

    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0]))
    assert dialog.suggest_button.isEnabled() is True

    _set(dialog, 1, "")
    assert dialog.suggest_button.isEnabled() is False

    # A non-number is not an anchor either.
    _set(dialog, 1, "abc")
    assert dialog.suggest_button.isEnabled() is False


def test_pressing_suggest_with_only_one_usable_anchor_says_so(qapp, tmp_path):
    """The button can be enabled by a value that later becomes unusable
    (e.g. the user edits it to a non-number after enabling), so the
    handler re-checks rather than trusting the button state."""
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0]))
    _set(dialog, 0, "60")
    dialog.suggest_button.setEnabled(True)
    dialog._on_suggest()
    assert "two" in dialog.status.text().lower()
    assert _text(dialog, 1) == ""


# --- suggestions -------------------------------------------------------


def test_unambiguous_rows_are_filled_and_marked(qapp, tmp_path):
    """Anchors 100->60 and 200->160 give E = ch - 40, so rows at 300 and
    400 predict 260 and 360. The source's 259.5 and 360.2 are each far
    from any other line, so both are suggested."""
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 259.5, 360.2, 800.0]))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()

    assert float(_text(dialog, 2)) == pytest.approx(259.5)
    assert float(_text(dialog, 3)) == pytest.approx(360.2)
    assert dialog.suggested_rows() == {2, 3}
    # A suggestion is marked in a way that does not depend on colour
    # alone -- the tooltip names where it came from; the anchors, being
    # the user's own, carry no such mark.
    assert "src.sou" in dialog.table.item(2, ENERGY).toolTip()
    assert dialog.table.item(0, ENERGY).toolTip() == ""
    assert "2" in dialog.status.text()
    assert "review" in dialog.status.text().lower()


def test_ambiguous_rows_stay_blank(qapp, tmp_path):
    """Row at 300 predicts 260; the source has 255 and 265, neither
    clearly nearer. Guessing here is exactly what the rule forbids."""
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 255.0, 265.0, 360.0]))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()

    assert _text(dialog, 2) == ""
    assert float(_text(dialog, 3)) == pytest.approx(360.0)
    assert dialog.suggested_rows() == {3}


def test_a_line_already_used_as_an_anchor_is_not_suggested_again(qapp, tmp_path):
    """Channel 110 predicts 70, nearest line 60 -- but 60 is an anchor.
    Two peaks cannot both be the 60 keV line, so the row stays blank."""
    dialog = _dialog(channels=(100.0, 200.0, 110.0))
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 500.0]))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    assert _text(dialog, 2) == ""


def test_two_rows_wanting_the_same_line_both_stay_blank(qapp, tmp_path):
    """Rows at 300 and 302 both predict ~260, both unambiguously nearest
    to 260. Handing it to one of them would be a coin toss."""
    dialog = _dialog(channels=(100.0, 200.0, 300.0, 302.0))
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0, 900.0]))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    assert _text(dialog, 2) == ""
    assert _text(dialog, 3) == ""
    assert dialog.suggested_rows() == set()


def test_a_second_suggest_replaces_untouched_suggestions(qapp, tmp_path):
    """Suggestions must not become anchors for the next round: with the
    anchors changed, the earlier guesses are cleared and recomputed from
    the user's values alone."""
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0, 360.0, 560.0, 760.0]))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    assert float(_text(dialog, 2)) == pytest.approx(260.0)

    # Re-anchor to E = 2*ch - 140: 300 -> 460 (ambiguous between 360 and
    # 560), 400 -> 660 (ambiguous between 560 and 760).
    _set(dialog, 1, "260")
    dialog._on_suggest()
    assert _text(dialog, 2) == ""
    assert _text(dialog, 3) == ""


def test_editing_a_suggested_cell_makes_it_the_users_own(qapp, tmp_path):
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0, 360.0]))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    assert 2 in dialog.suggested_rows()

    _set(dialog, 2, "261")
    assert 2 not in dialog.suggested_rows()
    # ...and it survives the next round as an anchor, not a guess.
    dialog._on_suggest()
    assert _text(dialog, 2) == "261"


def test_degenerate_anchors_are_reported_not_crashed(qapp, tmp_path):
    """Two anchors with the same energy describe no line."""
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0]))
    _set(dialog, 0, "60")
    _set(dialog, 1, "60")
    dialog._on_suggest()
    assert _text(dialog, 2) == ""
    assert dialog.status.text() != ""


# --- end to end --------------------------------------------------------


def test_source_assisted_calibration_uses_every_assigned_row(qapp, tmp_path):
    """Anchors plus suggestions all feed the final fit: four points on
    E = ch - 40 give exactly that line."""
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0, 360.0]))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    dialog._on_accept()

    calibration = dialog.result_calibration
    assert calibration is not None
    assert calibration.kind == "linear"
    assert calibration.a == pytest.approx(-40.0)
    assert calibration.b == pytest.approx(1.0)
    # Four points, not two: a fitted line with more than the minimum
    # reports its coefficient uncertainties.
    assert calibration.coefficient_errors != ()


# --- the app reaches it ------------------------------------------------


def test_main_window_gives_the_dialog_its_settings(qapp, tmp_path, monkeypatch):
    """A dialog that can load sources but is never handed the app's
    Settings would silently lose the shared last-folder, and the feature
    would be worse in the app than in its own tests. See the standing
    lesson that a computation-layer test is not proof a feature shipped.
    """
    import numpy as np
    from PySide6.QtWidgets import QDialog

    import energy_assign_dialog as module
    from main_window import MainWindow
    from peak_fit import fit_peaks

    x = np.arange(600, dtype=float)
    clean = np.full(600, 30.0)
    for centre, amplitude in ((150.0, 900.0), (420.0, 700.0)):
        clean = clean + amplitude * np.exp(-((x - centre) ** 2) / (2 * 4.0 ** 2))
    y = np.random.default_rng(9).poisson(clean).astype(float)
    path = tmp_path / "src.txt"
    path.write_text("\n".join(str(int(v)) for v in y), encoding="utf-8")

    window = MainWindow()
    window._load_files([str(path)])
    active = window.spectra[0]
    axis = np.arange(len(active.data), dtype=float)
    for centre in (150.0, 420.0):
        active.fits.append(
            fit_peaks(axis, active.data, (centre - 90, centre - 50),
                      (centre + 50, centre + 90), (centre - 30, centre + 30),
                      [centre])
        )

    seen = {}
    real = module.EnergyAssignDialog

    class _Spy(real):
        def __init__(self, parent, peaks, quadratic=False, settings=None):
            seen["settings"] = settings
            super().__init__(parent, peaks, quadratic=quadratic, settings=settings)

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(module, "EnergyAssignDialog", _Spy)
    window._open_calibrate_from_peaks_dialog()

    assert seen["settings"] is window.settings


# --- what the status line says -----------------------------------------


def test_status_distinguishes_all_from_some_from_none(qapp, tmp_path):
    """Three different outcomes must read differently. An earlier version
    said "the rest had no unambiguous match" even when there was no rest.
    """
    # All remaining suggested.
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0, 360.0], "a.sou"))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    assert "All 2 unassigned peaks suggested" in dialog.status.text()
    assert "no unambiguous match" not in dialog.status.text()

    # Some suggested, some not: 300 -> 260 is ambiguous (255/265),
    # 400 -> 360 is clean.
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 255.0, 265.0, 360.0], "b.sou"))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    assert "1 of 2 unassigned peaks suggested" in dialog.status.text()
    assert "The other 1 had no unambiguous match" in dialog.status.text()

    # None suggested.
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 255.0, 265.0, 355.0, 365.0], "c.sou"))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    assert "No unambiguous match" in dialog.status.text()
    assert "2 unassigned peaks" in dialog.status.text()


def test_status_when_nothing_is_left_to_suggest(qapp, tmp_path):
    dialog = _dialog(channels=(100.0, 200.0))
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0]))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    assert dialog.status.text() == "Every peak already has an energy."


def test_the_suggestion_tint_is_visible_in_both_themes(qapp, tmp_path):
    """Rendered pixels, not the brush we set.

    A previous feature shipped a colour that was invisible on the dark
    theme, and both stylesheets set a background on QTableWidget itself --
    exactly the situation where a per-item brush can be painted over.

    Two things this test had to get right, both of which caught me:
    `visualItemRect` is in VIEWPORT coordinates, so grabbing the whole
    table (which includes its header) samples one row off; and the sample
    point must avoid the digits or it reads the text colour. The CONTROL
    is the pair of unsuggested cells -- if they differ from each other,
    the measurement is picking up something other than the tint and
    proves nothing.
    """
    from PySide6.QtCore import QPoint

    import theme

    for name in ("light", "dark"):
        dialog = _dialog()
        dialog.setStyleSheet(theme.qt_stylesheet(name))
        dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0, 360.0]))
        _set(dialog, 0, "60")
        _set(dialog, 1, "160")
        dialog._on_suggest()
        assert dialog.suggested_rows() == {2, 3}

        dialog.resize(600, 300)
        dialog.show()
        qapp.processEvents()
        table = dialog.table
        image = table.viewport().grab().toImage()

        def sample(row):
            rect = table.visualItemRect(table.item(row, ENERGY))
            return image.pixelColor(QPoint(rect.right() - 4, rect.center().y()))

        plain_first, plain_second = sample(0), sample(1)
        tinted_first, tinted_second = sample(2), sample(3)

        assert plain_first == plain_second, (
            f"[{name}] control failed: two unsuggested cells rendered "
            f"differently ({plain_first.name()} vs {plain_second.name()}), so "
            f"this is not measuring the tint"
        )
        assert tinted_first == tinted_second, (
            f"[{name}] two suggested cells rendered differently: "
            f"{tinted_first.name()} vs {tinted_second.name()}"
        )
        assert tinted_first != plain_first, (
            f"[{name}] suggestion tint is invisible: a suggested cell "
            f"rendered {tinted_first.name()}, the same as an unsuggested one"
        )
        dialog.close()


def test_loading_a_different_source_drops_the_previous_guesses(qapp, tmp_path):
    """Switching nuclide is what a user does after realising they had the
    wrong one. Leaving the old guesses in place would let OK calibrate
    against the source they just rejected, with a tooltip still naming
    the old file.
    """
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0, 360.0], "first.sou"))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    assert dialog.suggested_rows() == {2, 3}

    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 900.0], "second.sou"))
    assert dialog.suggested_rows() == set()
    assert _text(dialog, 2) == ""
    assert _text(dialog, 3) == ""
    assert "Cleared 2 suggestion" in dialog.status.text()
    # The user's own anchors are untouched.
    assert _text(dialog, 0) == "60"
    assert _text(dialog, 1) == "160"


def test_an_edited_suggestion_survives_a_source_change(qapp, tmp_path):
    """Once edited it is the user's value, not a guess, so a new source
    has no business removing it."""
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 260.0, 360.0], "first.sou"))
    _set(dialog, 0, "60")
    _set(dialog, 1, "160")
    dialog._on_suggest()
    _set(dialog, 2, "261")

    dialog.load_source(_sou(tmp_path, [60.0, 160.0, 900.0], "second.sou"))
    assert _text(dialog, 2) == "261"
    assert _text(dialog, 3) == ""
    assert "Cleared 1 suggestion" in dialog.status.text()


def test_the_first_source_load_reports_nothing_to_clear(qapp, tmp_path):
    dialog = _dialog()
    dialog.load_source(_sou(tmp_path, [60.0, 160.0]))
    assert dialog.status.text() == ""
