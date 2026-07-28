from factor_dialog import FactorDialog


def _positive_factor_dialog(parent=None):
    return FactorDialog(
        parent, "Multiply by Factor", "Factor:",
        parse=float,
        validate=lambda v: None if v > 0 else "Factor must be greater than zero.",
    )


def _rebin_factor_dialog(parent=None):
    return FactorDialog(
        parent, "Rebin by Factor", "Factor:",
        parse=int,
        validate=lambda v: None if v >= 2 else "Rebin factor must be an integer of at least 2.",
    )


def test_accepts_a_valid_factor(qapp):
    dialog = _positive_factor_dialog()
    dialog._field.setText("2.5")
    dialog._on_accept()
    assert dialog.result_factor == 2.5


def test_rejects_unparseable_text(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(a))
    )
    dialog = _positive_factor_dialog()
    dialog._field.setText("not a number")
    dialog._on_accept()
    assert dialog.result_factor is None
    assert len(warned) == 1


def test_rejects_a_value_that_fails_validation(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(a))
    )
    dialog = _positive_factor_dialog()
    dialog._field.setText("-3")
    dialog._on_accept()
    assert dialog.result_factor is None
    assert len(warned) == 1


def test_rejects_zero(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(a))
    )
    dialog = _positive_factor_dialog()
    dialog._field.setText("0")
    dialog._on_accept()
    assert dialog.result_factor is None
    assert len(warned) == 1


def test_integer_parser_rejects_fractional_text(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(a))
    )
    dialog = _rebin_factor_dialog()
    dialog._field.setText("2.5")
    dialog._on_accept()
    assert dialog.result_factor is None
    assert len(warned) == 1


def test_integer_parser_accepts_a_valid_factor(qapp):
    dialog = _rebin_factor_dialog()
    dialog._field.setText("4")
    dialog._on_accept()
    assert dialog.result_factor == 4


def test_integer_parser_rejects_factor_of_one(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(a))
    )
    dialog = _rebin_factor_dialog()
    dialog._field.setText("1")
    dialog._on_accept()
    assert dialog.result_factor is None
    assert len(warned) == 1


def test_window_title_is_set(qapp):
    dialog = _positive_factor_dialog()
    assert dialog.windowTitle() == "Multiply by Factor"
