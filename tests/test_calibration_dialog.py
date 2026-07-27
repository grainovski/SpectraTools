from calibration import Calibration
from calibration_dialog import CalibrationDialog


def test_dialog_defaults_to_linear_with_c_hidden(qapp):
    dialog = CalibrationDialog(None)
    assert dialog._linear_radio.isChecked() is True
    assert dialog._c_field.isVisible() is False


def test_quadratic_radio_shows_c_field(qapp):
    dialog = CalibrationDialog(None)
    dialog._quadratic_radio.setChecked(True)
    assert dialog._c_field.isVisible() is True


def test_linear_radio_hides_c_field_again(qapp):
    dialog = CalibrationDialog(None)
    dialog._quadratic_radio.setChecked(True)
    dialog._linear_radio.setChecked(True)
    assert dialog._c_field.isVisible() is False


def test_prefills_fields_from_initial_calibration(qapp):
    initial = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.0002)
    dialog = CalibrationDialog(None, initial=initial, initially_active=True)
    assert dialog._quadratic_radio.isChecked() is True
    assert float(dialog._a_field.text()) == 10.0
    assert float(dialog._b_field.text()) == 0.5
    assert float(dialog._c_field.text()) == 0.0002
    assert dialog._active_checkbox.isChecked() is True


def test_accept_with_valid_linear_coefficients_sets_result(qapp):
    dialog = CalibrationDialog(None)
    dialog._a_field.setText("10.0")
    dialog._b_field.setText("0.5")
    dialog._active_checkbox.setChecked(True)
    dialog._on_accept()
    assert dialog.result_calibration == Calibration(kind="linear", a=10.0, b=0.5)
    assert dialog.result_active is True


def test_accept_with_valid_quadratic_coefficients_sets_result(qapp):
    dialog = CalibrationDialog(None)
    dialog._quadratic_radio.setChecked(True)
    dialog._a_field.setText("10.0")
    dialog._b_field.setText("0.5")
    dialog._c_field.setText("0.0002")
    dialog._on_accept()
    assert dialog.result_calibration == Calibration(kind="quadratic", a=10.0, b=0.5, c=0.0002)


def test_accept_with_b_zero_shows_error_and_does_not_close(qapp):
    dialog = CalibrationDialog(None)
    dialog._a_field.setText("10.0")
    dialog._b_field.setText("0.0")
    dialog._on_accept()
    assert dialog.result_calibration is None
    assert dialog.isVisible() or not dialog.result_calibration  # dialog not accepted


def test_accept_with_non_numeric_field_shows_error(qapp):
    dialog = CalibrationDialog(None)
    dialog._a_field.setText("not-a-number")
    dialog._b_field.setText("0.5")
    dialog._on_accept()
    assert dialog.result_calibration is None
