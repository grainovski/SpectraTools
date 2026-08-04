import numpy as np

from combine_dialog import CombineDialog
from spectrum import LoadedSpectrum


def _spectrum(path, length):
    return LoadedSpectrum(path, np.arange(length, dtype=np.int64), "#000000")


def test_combo_boxes_are_populated_with_every_spectrum_by_basename(qapp):
    a = _spectrum("C:/data/a.txt", 10)
    b = _spectrum("C:/data/b.txt", 10)
    dialog = CombineDialog(None, "Add Spectra", [a, b], a)
    assert [dialog._combo_a.itemText(i) for i in range(dialog._combo_a.count())] == ["a.txt", "b.txt"]
    assert [dialog._combo_b.itemText(i) for i in range(dialog._combo_b.count())] == ["a.txt", "b.txt"]


def test_spectrum_a_defaults_to_the_active_spectrum(qapp):
    a = _spectrum("a.txt", 10)
    b = _spectrum("b.txt", 10)
    c = _spectrum("c.txt", 10)
    b.active = True
    dialog = CombineDialog(None, "Add Spectra", [a, b, c], b)
    assert dialog._combo_a.currentIndex() == 1


def test_spectrum_b_defaults_to_the_first_other_spectrum(qapp):
    a = _spectrum("a.txt", 10)
    b = _spectrum("b.txt", 10)
    a.active = True
    dialog = CombineDialog(None, "Add Spectra", [a, b], a)
    assert dialog._combo_a.currentIndex() == 0
    assert dialog._combo_b.currentIndex() == 1


def test_spectrum_b_default_skips_ahead_when_a_defaults_to_index_zero_and_there_is_no_active(qapp):
    a = _spectrum("a.txt", 10)
    b = _spectrum("b.txt", 10)
    # No spectrum is active -- A falls back to index 0 (the same fallback
    # FactorDialog-adjacent code elsewhere uses); B must not also default
    # to 0, or the dialog would open with both dropdowns pointing at the
    # same spectrum with no user action.
    dialog = CombineDialog(None, "Add Spectra", [a, b], None)
    assert dialog._combo_a.currentIndex() == 0
    assert dialog._combo_b.currentIndex() == 1


def test_factor_field_defaults_to_one(qapp):
    a = _spectrum("a.txt", 10)
    b = _spectrum("b.txt", 10)
    dialog = CombineDialog(None, "Add Spectra", [a, b], a)
    assert dialog._factor_field.text() == "1"


def test_accepts_a_valid_pair_and_factor(qapp):
    a = _spectrum("a.txt", 10)
    b = _spectrum("b.txt", 10)
    dialog = CombineDialog(None, "Add Spectra", [a, b], a)
    dialog._combo_a.setCurrentIndex(0)
    dialog._combo_b.setCurrentIndex(1)
    dialog._factor_field.setText("0.5")
    dialog._on_accept()
    assert dialog.result_spectrum_a is a
    assert dialog.result_spectrum_b is b
    assert dialog.result_factor == 0.5


def test_rejects_unparseable_factor(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(a)))
    a = _spectrum("a.txt", 10)
    b = _spectrum("b.txt", 10)
    dialog = CombineDialog(None, "Add Spectra", [a, b], a)
    dialog._factor_field.setText("not a number")
    dialog._on_accept()
    assert dialog.result_factor is None
    assert len(warned) == 1


def test_rejects_zero_factor(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(a)))
    a = _spectrum("a.txt", 10)
    b = _spectrum("b.txt", 10)
    dialog = CombineDialog(None, "Add Spectra", [a, b], a)
    dialog._factor_field.setText("0")
    dialog._on_accept()
    assert dialog.result_factor is None
    assert len(warned) == 1


def test_rejects_negative_factor(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(a)))
    a = _spectrum("a.txt", 10)
    b = _spectrum("b.txt", 10)
    dialog = CombineDialog(None, "Add Spectra", [a, b], a)
    dialog._factor_field.setText("-1")
    dialog._on_accept()
    assert dialog.result_factor is None
    assert len(warned) == 1


def test_rejects_mismatched_lengths_and_names_both_channel_counts(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(a)))
    a = _spectrum("a.txt", 4096)
    b = _spectrum("b.txt", 2048)
    dialog = CombineDialog(None, "Add Spectra", [a, b], a)
    dialog._on_accept()
    assert dialog.result_spectrum_a is None
    assert len(warned) == 1
    message = warned[0][2]
    assert "4096" in message
    assert "2048" in message


def test_window_title_is_set(qapp):
    a = _spectrum("a.txt", 10)
    b = _spectrum("b.txt", 10)
    dialog = CombineDialog(None, "Subtract Spectra", [a, b], a)
    assert dialog.windowTitle() == "Subtract Spectra"


def test_picking_the_same_spectrum_for_both_a_and_b_is_allowed(qapp):
    # Per the design spec's Error Handling Summary: mathematically
    # well-defined (A x (1+factor) for Add), deliberately NOT blocked --
    # this test locks that decision in so it isn't accidentally
    # "fixed" by a future validation check. Two spectra are loaded (the
    # realistic case, since the caller in main_window.py never opens
    # this dialog with fewer than 2 loaded) but both dropdowns are
    # explicitly pointed at the same one.
    a = _spectrum("a.txt", 10)
    b = _spectrum("b.txt", 10)
    dialog = CombineDialog(None, "Add Spectra", [a, b], a)
    dialog._combo_a.setCurrentIndex(0)
    dialog._combo_b.setCurrentIndex(0)
    dialog._on_accept()
    assert dialog.result_spectrum_a is a
    assert dialog.result_spectrum_b is a
    assert dialog.result_factor == 1.0
