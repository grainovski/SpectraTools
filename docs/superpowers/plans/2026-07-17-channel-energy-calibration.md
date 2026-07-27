# Channel/Energy Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add linear/quadratic channel-to-keV calibration, set via a new dialog (type, coefficients from a file or manual entry, an Active switch), that redraws the plot axis in keV and shows fit/integration results in both channels and keV, while every internal marking/fitting/storage structure stays channel-based unchanged.

**Architecture:** A new pure-Python `calibration.py` module (`Calibration.apply`/`derivative`/`invert`, ported from TV's `vsCal.c` `CalP`/`CalC`) and a thin `calibration_dialog.py` Qt layer are the only new files. Everything else is small, surgical edits at exactly the points where channel values cross into "what's drawn on the x-axis" or "what's shown in a result" — `MainWindow.channel_to_display`/`display_to_channel` are the two conversion points every other change routes through.

**Tech Stack:** PySide6 (`QDialog`, `QRadioButton`, `QLineEdit`, `QFileDialog`), matplotlib (existing plotting), pytest + the project's existing Qt test helpers (`qapp` fixture, `_click`/`_held_key_click`/`_make_active_spectrum` in `tests/test_fit_mode_ui.py`).

**Reference:** [docs/superpowers/specs/2026-07-17-channel-energy-calibration-design.md](../specs/2026-07-17-channel-energy-calibration-design.md) — read this first for the *why* behind each decision below (channels-canonical rationale, TV source citations, out-of-scope list). This plan does not repeat that reasoning, only the concrete steps.

---

## Task 1: `Calibration` dataclass and forward evaluation (`apply`)

**Files:**
- Create: `calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration.py`:

```python
import pytest

from calibration import Calibration, CalibrationError


def test_linear_apply_matches_hand_computation():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    # E = a + b*channel
    assert cal.apply(0) == pytest.approx(10.0)
    assert cal.apply(100) == pytest.approx(10.0 + 0.5 * 100)
    assert cal.apply(200) == pytest.approx(10.0 + 0.5 * 200)


def test_quadratic_apply_matches_hand_computation():
    cal = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.001)
    # E = a + b*channel + c*channel**2
    assert cal.apply(0) == pytest.approx(10.0)
    assert cal.apply(100) == pytest.approx(10.0 + 0.5 * 100 + 0.001 * 100 ** 2)


def test_apply_broadcasts_over_a_numpy_array():
    import numpy as np

    cal = Calibration(kind="linear", a=10.0, b=0.5)
    channels = np.array([0.0, 100.0, 200.0])
    result = cal.apply(channels)
    expected = 10.0 + 0.5 * channels
    np.testing.assert_allclose(result, expected)


def test_b_zero_raises_calibration_error():
    with pytest.raises(CalibrationError):
        Calibration(kind="linear", a=10.0, b=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'calibration'`

- [ ] **Step 3: Write the implementation**

Create `calibration.py`:

```python
"""Channel-to-energy calibration: linear or quadratic, ported from TV's
own position calibration (tv-1.9.13/lib/tv/vsCal.c). Pure Python, no Qt
dependency -- calibration_dialog.py is the thin Qt layer on top of this."""

from dataclasses import dataclass


class CalibrationError(Exception):
    """Raised when a Calibration's coefficients are invalid."""


class CalibrationFileError(Exception):
    """Raised when an ASCII calibration coefficient file can't be parsed."""


@dataclass
class Calibration:
    kind: str  # "linear" or "quadratic"
    a: float
    b: float
    c: float = 0.0  # unused for "linear"

    def __post_init__(self):
        # TV's own inversion (CalC, vsCal.c:1047-1082) divides by b for
        # its initial guess with no guard, silently producing NaN if
        # b == 0. This is a new UI entry point, not a raw port of an
        # existing TV-driven flow, so it's validated here instead of
        # replicating that silent failure mode.
        if self.b == 0.0:
            raise CalibrationError("Calibration coefficient 'b' must not be zero")

    def apply(self, channel):
        """channel -> energy (keV). E = a + b*channel + c*channel**2,
        Horner's method. Matches TV's CalP (vsCal.c:981-988). Works on
        a scalar or a numpy array via broadcasting."""
        return self.a + channel * (self.b + channel * self.c)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add calibration.py tests/test_calibration.py
git commit -m "feat: add Calibration dataclass with forward evaluation"
```

---

## Task 2: `derivative()`

**Files:**
- Modify: `calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration.py`:

```python
def test_derivative_linear_is_constant_b():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    assert cal.derivative(0) == pytest.approx(0.5)
    assert cal.derivative(500) == pytest.approx(0.5)


def test_derivative_quadratic_matches_hand_computation():
    cal = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.001)
    # dE/dchannel = b + 2*c*channel
    assert cal.derivative(100) == pytest.approx(0.5 + 2 * 0.001 * 100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py -v -k derivative`
Expected: FAIL with `AttributeError: 'Calibration' object has no attribute 'derivative'`

- [ ] **Step 3: Write the implementation**

In `calibration.py`, add after `apply`:

```python
    def derivative(self, channel):
        """dE/dchannel at `channel` -- b + 2*c*channel. Matches the
        gradient computation inside TV's CalC (vsCal.c:1066-1068), used
        there for Newton's-method inversion and here for both that and
        keV-uncertainty propagation in the results display."""
        return self.b + 2.0 * self.c * channel
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration.py -v -k derivative`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add calibration.py tests/test_calibration.py
git commit -m "feat: add Calibration.derivative for Newton's-method inversion"
```

---

## Task 3: `invert()` (Newton-Raphson, ported from TV's `CalC`)

**Files:**
- Modify: `calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration.py`:

```python
def test_invert_linear_round_trips_apply():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    for channel in (0.0, 1.0, 100.0, 500.0, 4095.0):
        energy = cal.apply(channel)
        assert cal.invert(energy) == pytest.approx(channel, abs=1e-6)


def test_invert_quadratic_round_trips_apply():
    cal = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.0002)
    for channel in (0.0, 1.0, 100.0, 500.0, 4095.0):
        energy = cal.apply(channel)
        assert cal.invert(energy) == pytest.approx(channel, abs=1e-6)


def test_invert_matches_hand_worked_newton_example():
    # Hand-worked: E = 10 + 0.5*ch + 0.0002*ch**2, solve for E=120.
    # Linear initial guess: x0 = (120-10)/0.5 = 220.
    # apply(220) = 10 + 110 + 0.0002*48400 = 10 + 110 + 9.68 = 129.68
    # de = 129.68 - 120 = 9.68; gradient = 0.5 + 2*0.0002*220 = 0.588
    # x1 = 220 - 9.68/0.588 = 220 - 16.462... = 203.537...
    # Iterate to convergence (|de| < 0.01) and confirm apply(x) == 120.
    cal = Calibration(kind="quadratic", a=10.0, b=0.5, c=0.0002)
    channel = cal.invert(120.0)
    assert cal.apply(channel) == pytest.approx(120.0, abs=0.01)
    # Confirms the iteration actually moved from the linear-only guess
    # of 220 rather than returning it unrefined.
    assert channel != pytest.approx(220.0, abs=1.0)


def test_invert_negative_b_still_round_trips():
    # b < 0 is a valid (if unusual) calibration -- energy decreasing
    # with channel. Newton's method must still converge.
    cal = Calibration(kind="linear", a=1000.0, b=-0.5)
    energy = cal.apply(300.0)
    assert cal.invert(energy) == pytest.approx(300.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py -v -k invert`
Expected: FAIL with `AttributeError: 'Calibration' object has no attribute 'invert'`

- [ ] **Step 3: Write the implementation**

In `calibration.py`, add module-level constants near the top (after the imports) and the `invert` method after `derivative`:

```python
# TV's own Newton's-method precision/iteration-cap constants
# (vsCal.c:15-16), reused verbatim rather than re-derived, for exact
# parity with an already source-verified reference.
_NEWTON_PRECISION = 0.01
_NEWTON_MAXITER = 10000
```

```python
    def invert(self, energy):
        """energy (keV) -> channel, via Newton-Raphson. Ported from TV's
        CalC (vsCal.c:1047-1082, the non-start-channel branch, since this
        app has no "start channel" concept). Initial guess is the linear
        approximation x0 = (energy - a) / b regardless of `kind`,
        matching TV exactly; refines against the polynomial's own
        derivative until the residual is within _NEWTON_PRECISION or
        _NEWTON_MAXITER iterations are exhausted. Scalar input only (this
        app only ever calls it with a single click/mouse-move
        coordinate)."""
        x = (energy - self.a) / self.b
        de = self.apply(x) - energy
        iterations = 0
        while abs(de) > _NEWTON_PRECISION and iterations < _NEWTON_MAXITER:
            x -= de / self.derivative(x)
            de = self.apply(x) - energy
            iterations += 1
        return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration.py -v -k invert`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add calibration.py tests/test_calibration.py
git commit -m "feat: add Calibration.invert via Newton-Raphson, ported from TV's CalC"
```

---

## Task 4: ASCII coefficient file reading

**Files:**
- Modify: `calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration.py`:

```python
def test_read_coefficients_file_linear(tmp_path):
    path = tmp_path / "cal.txt"
    path.write_text("10.5\n0.487\n")
    from calibration import read_coefficients_file
    assert read_coefficients_file(str(path), quadratic=False) == [10.5, 0.487]


def test_read_coefficients_file_quadratic(tmp_path):
    path = tmp_path / "cal.txt"
    path.write_text("10.5\n0.487\n0.0002\n")
    from calibration import read_coefficients_file
    assert read_coefficients_file(str(path), quadratic=True) == [10.5, 0.487, 0.0002]


def test_read_coefficients_file_ignores_blank_lines(tmp_path):
    path = tmp_path / "cal.txt"
    path.write_text("10.5\n\n0.487\n\n")
    from calibration import read_coefficients_file
    assert read_coefficients_file(str(path), quadratic=False) == [10.5, 0.487]


def test_read_coefficients_file_wrong_count_raises(tmp_path):
    from calibration import CalibrationFileError, read_coefficients_file
    path = tmp_path / "cal.txt"
    path.write_text("10.5\n")  # only 1 line, linear needs 2
    with pytest.raises(CalibrationFileError):
        read_coefficients_file(str(path), quadratic=False)


def test_read_coefficients_file_non_numeric_raises(tmp_path):
    from calibration import CalibrationFileError, read_coefficients_file
    path = tmp_path / "cal.txt"
    path.write_text("10.5\nnot-a-number\n")
    with pytest.raises(CalibrationFileError):
        read_coefficients_file(str(path), quadratic=False)


def test_read_coefficients_file_missing_file_raises(tmp_path):
    from calibration import CalibrationFileError, read_coefficients_file
    with pytest.raises(CalibrationFileError):
        read_coefficients_file(str(tmp_path / "does_not_exist.txt"), quadratic=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py -v -k read_coefficients`
Expected: FAIL with `ImportError: cannot import name 'read_coefficients_file'`

- [ ] **Step 3: Write the implementation**

In `calibration.py`, add at the end of the file:

```python
def read_coefficients_file(path, quadratic):
    """Reads a and b (and c if `quadratic`) from `path`, one coefficient
    per line, blank lines ignored. Raises CalibrationFileError on a
    wrong line count, non-numeric content, or an unreadable file."""
    expected = 3 if quadratic else 2
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError as exc:
        raise CalibrationFileError(f"Could not read {path}: {exc}") from exc
    if len(lines) != expected:
        kind_name = "quadratic" if quadratic else "linear"
        raise CalibrationFileError(
            f"Expected {expected} coefficient(s) (one per line) for "
            f"{kind_name} calibration, found {len(lines)} in {path}"
        )
    try:
        return [float(line) for line in lines]
    except ValueError as exc:
        raise CalibrationFileError(f"Non-numeric value in {path}: {exc}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS (all tests in the file, 16 total)

- [ ] **Step 5: Commit**

```bash
git add calibration.py tests/test_calibration.py
git commit -m "feat: add read_coefficients_file for ASCII calibration files"
```

---

## Task 5: `CalibrationDialog` — type selector, fields, OK/Cancel validation

**Files:**
- Create: `calibration_dialog.py`
- Test: `tests/test_calibration_dialog.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration_dialog.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration_dialog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'calibration_dialog'`

- [ ] **Step 3: Write the implementation**

Create `calibration_dialog.py`:

```python
"""Qt dialog for setting a channel/energy Calibration -- the thin UI
layer on top of calibration.py's pure logic."""

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from calibration import Calibration, CalibrationError, CalibrationFileError, read_coefficients_file


class CalibrationDialog(QDialog):
    """`result_calibration`/`result_active` are set only after a
    successful OK (`_on_accept` calls `self.accept()`); both stay at
    their initial `None`/`False` if the dialog is cancelled or closed
    without a valid OK."""

    def __init__(self, parent, initial=None, initially_active=False):
        super().__init__(parent)
        self.setWindowTitle("Calibration")
        self.result_calibration = None
        self.result_active = False

        self._linear_radio = QRadioButton("Linear")
        self._quadratic_radio = QRadioButton("Quadratic")
        self._linear_radio.setChecked(True)
        self._quadratic_radio.toggled.connect(self._update_c_field_visibility)

        self._a_field = QLineEdit()
        self._b_field = QLineEdit()
        self._c_field = QLineEdit()
        self._c_label = QLabel("c:")

        self._load_button = QPushButton("Load from file...")
        self._load_button.clicked.connect(self._on_load_file)

        self._active_checkbox = QCheckBox("Active")

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        type_row = QHBoxLayout()
        type_row.addWidget(self._linear_radio)
        type_row.addWidget(self._quadratic_radio)

        form = QFormLayout()
        form.addRow("a:", self._a_field)
        form.addRow("b:", self._b_field)
        form.addRow(self._c_label, self._c_field)

        layout = QVBoxLayout(self)
        layout.addLayout(type_row)
        layout.addLayout(form)
        layout.addWidget(self._load_button)
        layout.addWidget(self._active_checkbox)
        layout.addWidget(button_box)

        if initial is not None:
            self._quadratic_radio.setChecked(initial.kind == "quadratic")
            self._linear_radio.setChecked(initial.kind == "linear")
            self._a_field.setText(repr(initial.a))
            self._b_field.setText(repr(initial.b))
            self._c_field.setText(repr(initial.c))
        self._active_checkbox.setChecked(initially_active)
        self._update_c_field_visibility()

    def _update_c_field_visibility(self):
        is_quadratic = self._quadratic_radio.isChecked()
        self._c_field.setVisible(is_quadratic)
        self._c_label.setVisible(is_quadratic)

    def _on_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration Coefficients", "", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            values = read_coefficients_file(path, quadratic=self._quadratic_radio.isChecked())
        except CalibrationFileError as exc:
            QMessageBox.warning(self, "Calibration", str(exc))
            return
        self._a_field.setText(repr(values[0]))
        self._b_field.setText(repr(values[1]))
        if len(values) > 2:
            self._c_field.setText(repr(values[2]))

    def _on_accept(self):
        try:
            a = float(self._a_field.text())
            b = float(self._b_field.text())
            c = float(self._c_field.text()) if self._quadratic_radio.isChecked() else 0.0
        except ValueError:
            QMessageBox.warning(self, "Calibration", "Coefficients must be valid numbers.")
            return
        kind = "quadratic" if self._quadratic_radio.isChecked() else "linear"
        try:
            calibration = Calibration(kind=kind, a=a, b=b, c=c)
        except CalibrationError as exc:
            QMessageBox.warning(self, "Calibration", str(exc))
            return
        self.result_calibration = calibration
        self.result_active = self._active_checkbox.isChecked()
        self.accept()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration_dialog.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add calibration_dialog.py tests/test_calibration_dialog.py
git commit -m "feat: add CalibrationDialog with type/coefficient fields and validation"
```

---

## Task 6: `CalibrationDialog` — "Load from file..." integration test

**Files:**
- Test: `tests/test_calibration_dialog.py`

Task 5 already implements `_on_load_file`; this task adds the test coverage that was deferred to keep Task 5 focused on the dialog's own fields.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration_dialog.py`:

```python
def test_load_from_file_populates_linear_fields(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    path = tmp_path / "cal.txt"
    path.write_text("10.5\n0.487\n")
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **kw: (str(path), ""))
    )

    dialog = CalibrationDialog(None)
    dialog._on_load_file()

    assert float(dialog._a_field.text()) == 10.5
    assert float(dialog._b_field.text()) == 0.487


def test_load_from_file_populates_quadratic_fields(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    path = tmp_path / "cal.txt"
    path.write_text("10.5\n0.487\n0.0002\n")
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **kw: (str(path), ""))
    )

    dialog = CalibrationDialog(None)
    dialog._quadratic_radio.setChecked(True)
    dialog._on_load_file()

    assert float(dialog._a_field.text()) == 10.5
    assert float(dialog._b_field.text()) == 0.487
    assert float(dialog._c_field.text()) == 0.0002


def test_load_from_malformed_file_shows_warning_and_leaves_fields(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    path = tmp_path / "cal.txt"
    path.write_text("only-one-bad-line\n")
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **kw: (str(path), ""))
    )
    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(a))
    )

    dialog = CalibrationDialog(None)
    dialog._a_field.setText("unchanged")
    dialog._on_load_file()

    assert len(warned) == 1
    assert dialog._a_field.text() == "unchanged"


def test_cancelled_file_dialog_leaves_fields_unchanged(qapp, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **kw: ("", ""))
    )
    dialog = CalibrationDialog(None)
    dialog._a_field.setText("unchanged")
    dialog._on_load_file()
    assert dialog._a_field.text() == "unchanged"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration_dialog.py -v -k load_from`
Expected: These should mostly PASS already since `_on_load_file` was implemented in Task 5 — run first to confirm; if any fail, they're catching a real gap in Task 5's implementation to fix now.

- [ ] **Step 3: Fix implementation if needed**

If `test_load_from_malformed_file_shows_warning_and_leaves_fields` fails because `_a_field` was already blank rather than "unchanged" some other way, adjust the test setup, not the implementation — `_on_load_file` in Task 5 already returns early on `CalibrationFileError` without touching the fields, which is correct behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration_dialog.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_calibration_dialog.py
git commit -m "test: cover CalibrationDialog's Load from file button"
```

---

## Task 7: `MainWindow` calibration state and `channel_to_display`/`display_to_channel`

**Files:**
- Modify: `main_window.py`
- Test: `tests/test_fit_mode_ui.py`

**Files:**
- Modify: `main_window.py:191-193` (inside `__init__`, right after `self._apply_theme(self._theme)`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py` (near the other `MainWindow`-level tests, e.g. after the imports section — this test only needs `MainWindow`, already imported):

```python
def test_channel_to_display_is_identity_with_no_calibration(qapp):
    main_window = MainWindow()
    assert main_window.channel_to_display(100) == 100
    assert main_window.display_to_channel(100) == 100


def test_channel_to_display_applies_calibration_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    assert main_window.channel_to_display(100) == pytest.approx(60.0)
    assert main_window.display_to_channel(60.0) == pytest.approx(100.0, abs=1e-6)


def test_channel_to_display_is_identity_when_calibration_set_but_inactive(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = False
    assert main_window.channel_to_display(100) == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k channel_to_display`
Expected: FAIL with `AttributeError: 'MainWindow' object has no attribute 'channel_to_display'`

- [ ] **Step 3: Write the implementation**

In `main_window.py`, add the import near the top (alongside the other project-local imports):

```python
from calibration import Calibration
```

In `__init__`, right after `self._apply_theme(self._theme)` (currently `main_window.py:193`):

```python
        self._calibration = None  # Calibration | None -- last-set coefficients, persist across on/off toggles
        self._calibration_active = False
```

Add two new methods on `MainWindow` (a reasonable place is right after `_apply_theme`, since they're closely related state):

```python
    def channel_to_display(self, channel):
        """Converts a channel number (or numpy array of channel numbers)
        to whatever's on the x-axis right now: the same value if no
        calibration is active, or its calibrated keV equivalent if one
        is. Every place that draws an x-coordinate routes through this."""
        if not self._calibration_active or self._calibration is None:
            return channel
        return self._calibration.apply(channel)

    def display_to_channel(self, display_x):
        """Inverse of channel_to_display -- converts an x-axis
        coordinate (channel or keV, whichever is currently displayed)
        back to a channel number. Every place that reads a click/hover
        x-coordinate routes through this."""
        if not self._calibration_active or self._calibration is None:
            return display_x
        return self._calibration.invert(display_x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k channel_to_display`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add main_window.py tests/test_fit_mode_ui.py
git commit -m "feat: add MainWindow calibration state and channel/display conversion"
```

---

## Task 8: Menu wiring and the dialog open/apply handler

**Files:**
- Modify: `main_window.py:224-253` (`_build_menu`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_calibration_menu_action_exists(qapp):
    main_window = MainWindow()
    assert hasattr(main_window, "calibration_action")
    assert main_window.calibration_action.text() == "Calibration..."


def test_applying_calibration_from_dialog_updates_state_and_replots(qapp, monkeypatch):
    from calibration import Calibration
    from calibration_dialog import CalibrationDialog

    main_window = MainWindow()
    _make_active_spectrum(main_window)

    applied = Calibration(kind="linear", a=10.0, b=0.5)

    def fake_exec(self):
        self.result_calibration = applied
        self.result_active = True
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec)
    main_window._open_calibration_dialog()

    assert main_window._calibration == applied
    assert main_window._calibration_active is True
    assert main_window.axes.get_xlabel() == "Energy (keV)"


def test_cancelling_calibration_dialog_leaves_state_untouched(qapp, monkeypatch):
    from calibration_dialog import CalibrationDialog

    main_window = MainWindow()
    _make_active_spectrum(main_window)

    def fake_exec(self):
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec)
    main_window._open_calibration_dialog()

    assert main_window._calibration is None
    assert main_window._calibration_active is False
    assert main_window.axes.get_xlabel() == "Channel"
```

This test file needs `QDialog` importable — check the existing `from PySide6.QtWidgets import ...` line at the top of `tests/test_fit_mode_ui.py` and add `QDialog` to it if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k calibration_menu or calibration_dialog`
Expected: FAIL with `AttributeError: 'MainWindow' object has no attribute 'calibration_action'`

- [ ] **Step 3: Write the implementation**

In `main_window.py`, add the import near the top:

```python
from calibration_dialog import CalibrationDialog
```

In `_build_menu`, right after the `dark_theme_action` block (currently ending at `main_window.py:253`):

```python
        view_menu.addSeparator()
        self.calibration_action = QAction("Calibration...", self)
        self.calibration_action.triggered.connect(self._open_calibration_dialog)
        view_menu.addAction(self.calibration_action)
```

Add a new method, e.g. right after `_apply_theme`/the two conversion helpers added in Task 7:

```python
    def _open_calibration_dialog(self):
        dialog = CalibrationDialog(
            self, initial=self._calibration, initially_active=self._calibration_active
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._calibration = dialog.result_calibration
            self._calibration_active = dialog.result_active
            if self.spectra:
                # preserve_view intentionally omitted (defaults to
                # False): the previous xlim was in the other unit
                # (channels vs keV) and carrying it over would show a
                # nonsensical view.
                self._plot_data()
```

`QDialog` needs to be importable in `main_window.py` — check the existing `from PySide6.QtWidgets import (...)` block and add `QDialog` to it if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k calibration_menu or calibration_dialog`
Expected: PASS (3 passed) — note `test_applying_calibration_from_dialog_updates_state_and_replots`'s axis-label assertion will only pass once Task 9 (below) implements the label swap; if running Task 8 in isolation before Task 9, expect that one assertion to fail and note it as a known gap closed by the next task, not a regression.

- [ ] **Step 5: Commit**

```bash
git add main_window.py tests/test_fit_mode_ui.py
git commit -m "feat: wire Calibration... menu action to open the dialog and apply results"
```

---

## Task 9: `_plot_data` axis transform and label

**Files:**
- Modify: `main_window.py:363-397` (`_plot_data`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_plot_data_draws_calibrated_x_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window._plot_data()

    line = main_window.axes.lines[0]
    xdata = line.get_xdata()
    assert xdata[0] == pytest.approx(10.0)  # channel 0 -> a
    assert xdata[-1] == pytest.approx(10.0 + 0.5 * (len(spectrum.data) - 1))
    assert main_window.axes.get_xlabel() == "Energy (keV)"


def test_plot_data_draws_raw_channels_when_inactive(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._plot_data()
    line = main_window.axes.lines[0]
    xdata = line.get_xdata()
    assert xdata[0] == 0
    assert main_window.axes.get_xlabel() == "Channel"


def test_plot_data_xlim_spans_calibrated_range_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window._plot_data()

    xlim = main_window.axes.get_xlim()
    max_channel = len(spectrum.data) - 1
    assert xlim[0] == pytest.approx(10.0, abs=1.0)
    assert xlim[1] == pytest.approx(10.0 + 0.5 * max_channel, abs=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k plot_data_draws or plot_data_xlim`
Expected: FAIL — `xdata[0]` is `0`, not `10.0` (axis transform not implemented yet)

- [ ] **Step 3: Write the implementation**

Replace `main_window.py`'s `_plot_data` (currently lines 363-397):

```python
    def _plot_data(self, preserve_view=False):
        # Captured before axes.clear() (which resets limits) -- used to
        # keep the user's current zoom when a replot is triggered by
        # something unrelated to loading/showing a spectrum, e.g.
        # committing or removing a peak fit.
        saved_xlim = self.axes.get_xlim() if preserve_view else None
        self.axes.clear()
        style_axes(self.axes, self._theme)
        visible = [s for s in self.spectra if s.visible]
        for spectrum in visible:
            channels = np.arange(len(spectrum.data))
            x = self.channel_to_display(channels)
            self.axes.plot(x, spectrum.data, drawstyle="steps-mid", color=spectrum.color)
            self.fit_controller.draw_committed_fits(spectrum)
        self.axes.set_xlabel("Energy (keV)" if self._calibration_active else "Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.axes.set_yscale("log" if self.log_scale_action.isChecked() else "linear")
        if visible:
            # Channel numbers can't be negative; override Matplotlib's
            # default 5% autoscale margin, which would otherwise show them
            # as such. Span the widest currently-visible spectrum, since
            # loaded files can have different channel counts.
            max_channel = max(len(s.data) for s in visible) - 1
            if saved_xlim is not None:
                xlim = saved_xlim
            else:
                xlim = (self.channel_to_display(0), self.channel_to_display(max_channel))
            self.axes.set_xlim(xlim)
            self._autoscale_y(xlim)
        self.canvas.draw()
        # Our custom zoom bypasses the toolbar's usual box-zoom/pan path, so
        # without this the Home/Back/Forward buttons wouldn't know about
        # this view. Reset the navigation history and record this full view
        # as the new "home" baseline.
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()
        self._update_fit_mode_availability()
        self.fit_controller.update_results_list()
```

(Only the `channels`/`x` line, the `xlabel` line, and the `xlim` computation changed from the original — everything else is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k plot_data_draws or plot_data_xlim or calibration_dialog`
Expected: PASS, including `test_applying_calibration_from_dialog_updates_state_and_replots` from Task 8, now that the axis label swap is implemented.

- [ ] **Step 5: Commit**

```bash
git add main_window.py tests/test_fit_mode_ui.py
git commit -m "feat: draw the plot x-axis in keV when calibration is active"
```

---

## Task 10: Fix `_autoscale_y`, `_zoom_x`, `_show_full_spectrum`, `_on_mouse_move` for a calibrated axis

**Files:**
- Modify: `main_window.py:399-413` (`_on_mouse_move`), `:595-655` (`_zoom_x`, `_show_full_spectrum`, `_autoscale_y`) — line numbers as of before this task's edits; re-check with `grep -n "def _on_mouse_move\|def _zoom_x\|def _show_full_spectrum\|def _autoscale_y" main_window.py` since Task 9 may have shifted them slightly.
- Test: `tests/test_fit_mode_ui.py`

This is the task that closes the gap the design spec didn't enumerate: these four functions currently treat the x-axis value as a raw channel number usable directly as an array index or clamp bound. Once `_plot_data` (Task 9) can draw a keV axis, all four need to convert through `display_to_channel`/`channel_to_display` at the right points, or zooming/autoscale/the status-bar readout silently misbehave under an active calibration.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_autoscale_y_uses_correct_channel_range_when_calibrated(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)  # 200 channels -> calibrated range [10, 109.5] keV
    spectrum.data[50] = 99999  # channel 50 -> keV 35, OUTSIDE the range queried below
    spectrum.data[150] = 500  # channel 150 -> keV 85, INSIDE the range queried below
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    # Query the y-autoscale over keV [70, 100] -> channel [120, 180]:
    # excludes channel 50's spike, includes channel 150's smaller bump.
    main_window._autoscale_y((70.0, 100.0))
    ylim = main_window.axes.get_ylim()
    assert ylim[1] < 99999  # the out-of-range spike must not be included
    assert ylim[1] >= 500  # the in-range value must be


def test_show_full_spectrum_spans_calibrated_range(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    main_window._show_full_spectrum()
    xlim = main_window.axes.get_xlim()
    max_channel = len(spectrum.data) - 1
    assert xlim[0] == pytest.approx(10.0, abs=1.0)
    assert xlim[1] == pytest.approx(10.0 + 0.5 * max_channel, abs=1.0)


def test_zoom_x_stays_within_calibrated_bounds(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    # Zoom out repeatedly -- the view must clamp to the calibrated full
    # range, not the raw channel range (10..109.5 for this spectrum),
    # which for a 200-channel spectrum would be far narrower than
    # letting it clamp to [0, 199].
    for _ in range(10):
        main_window._zoom_x(2.0)
    xlim = main_window.axes.get_xlim()
    max_channel = len(spectrum.data) - 1
    assert xlim[0] == pytest.approx(10.0, abs=1.0)
    assert xlim[1] == pytest.approx(10.0 + 0.5 * max_channel, abs=1.0)


def test_on_mouse_move_reports_correct_channel_when_calibrated(qapp):
    from calibration import Calibration
    from matplotlib.backend_bases import MouseEvent

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data[100] = 12345
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    # Channel 100 -> keV 10 + 0.5*100 = 60.
    px, py = main_window.axes.transData.transform((60.0, 10.0))
    event = MouseEvent("motion_notify_event", main_window.canvas, px, py)
    main_window._on_mouse_move(event)

    message = main_window.statusBar().currentMessage()
    assert "12345" in message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k autoscale_y_uses or show_full_spectrum_spans or zoom_x_stays or on_mouse_move_reports`
Expected: FAIL — e.g. `test_on_mouse_move_reports_correct_channel_when_calibrated` fails because `_on_mouse_move` currently does `channel = int(round(event.xdata))` directly (60, not 100), so it reports spectrum.data[60] instead of spectrum.data[100].

- [ ] **Step 3: Write the implementation**

Find current line numbers first:

Run: `grep -n "def _on_mouse_move\|def _zoom_x\|def _show_full_spectrum\|def _autoscale_y" main_window.py`

Replace `_on_mouse_move`:

```python
    def _on_mouse_move(self, event):
        if time.monotonic() < self.fit_controller._status_message_until:
            return
        visible = [s for s in self.spectra if s.visible]
        if not visible or event.inaxes != self.axes or event.xdata is None:
            self.statusBar().clearMessage()
            return
        channel = int(round(self.display_to_channel(event.xdata)))
        if self._calibration_active:
            parts = [f"Channel: {channel}  Energy: {event.xdata:.2f} keV"]
        else:
            parts = [f"Channel: {channel}"]
        for spectrum in visible:
            if 0 <= channel < len(spectrum.data):
                parts.append(f"{os.path.basename(spectrum.path)}: {spectrum.data[channel]}")
            else:
                parts.append(f"{os.path.basename(spectrum.path)}: -")
        self.statusBar().showMessage("  |  ".join(parts))
```

Replace `_zoom_x`:

```python
    def _zoom_x(self, factor, center=None):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        xlim = self.axes.get_xlim()
        if center is None:
            center = (xlim[0] + xlim[1]) / 2
        half_width = (xlim[1] - xlim[0]) / 2 * factor
        max_channel = max(len(s.data) for s in visible) - 1
        display_lo = self.channel_to_display(0)
        display_hi = self.channel_to_display(max_channel)
        display_lo, display_hi = min(display_lo, display_hi), max(display_lo, display_hi)
        # Clamp to the valid displayed range -- zooming/scrolling must
        # never show channels outside the data, whether the axis is
        # currently in raw channels or calibrated keV.
        new_lo = max(display_lo, center - half_width)
        new_hi = min(display_hi, center + half_width)
        if new_hi <= new_lo:
            new_hi = min(display_hi, new_lo + 1)
        new_xlim = (new_lo, new_hi)
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()
```

Replace `_show_full_spectrum`:

```python
    def _show_full_spectrum(self):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        max_channel = max(len(s.data) for s in visible) - 1
        full_xlim = (self.channel_to_display(0), self.channel_to_display(max_channel))
        self.axes.set_xlim(full_xlim)
        self._autoscale_y(full_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()
```

Replace `_autoscale_y`:

```python
    def _autoscale_y(self, xlim):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        channel_lo = self.display_to_channel(xlim[0])
        channel_hi = self.display_to_channel(xlim[1])
        channel_lo, channel_hi = min(channel_lo, channel_hi), max(channel_lo, channel_hi)
        lo_bound = max(0, int(np.floor(channel_lo)))
        hi_bound = int(np.ceil(channel_hi)) + 1
        slices = []
        for spectrum in visible:
            lo = min(lo_bound, len(spectrum.data))
            hi = min(hi_bound, len(spectrum.data))
            if lo < hi:
                slices.append(spectrum.data[lo:hi])
        if not slices:
            return
        combined = np.concatenate(slices)
        y_min = float(np.min(combined))
        y_max = float(np.max(combined))
        if self.log_scale_action.isChecked():
            # A log-scaled axis silently ignores set_ylim() with a
            # non-positive bound, so a plain "y_min - margin" (common
            # here since spectra often have zero-count channels) would
            # leave the Y-axis un-rescaled. Floor to a small positive
            # value and use a multiplicative margin instead.
            y_min = max(y_min, 1.0)
            y_max = max(y_max, y_min * 1.1)
            self.axes.set_ylim(y_min / 1.1, y_max * 1.1)
        else:
            margin = (y_max - y_min) * 0.05 or 1.0
            self.axes.set_ylim(y_min - margin, y_max + margin)
```

(Only the first three lines of `_autoscale_y`'s body changed — `lo_bound`/`hi_bound` are now derived from `display_to_channel(xlim[...])` instead of `xlim[...]` directly.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k autoscale_y_uses or show_full_spectrum_spans or zoom_x_stays or on_mouse_move_reports`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `pytest tests/ -q`
Expected: PASS, same count as before this task plus the new tests — these four functions are exercised heavily by existing (uncalibrated) tests, so this is the step most likely to reveal an accidental behavior change for the inactive-calibration case. `channel_to_display`/`display_to_channel` are identity functions when inactive, so uncalibrated behavior should be byte-for-byte unchanged, but confirm.

- [ ] **Step 6: Commit**

```bash
git add main_window.py tests/test_fit_mode_ui.py
git commit -m "fix: make zoom/autoscale/mouse-position readout calibration-aware"
```

---

## Task 11: Click → channel inversion in `on_click`

**Files:**
- Modify: `fit_mode.py:366-392` (`on_click`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_marking_click_resolves_to_correct_channel_when_calibrated(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    # Click at keV 60 (= channel 100 under a=10, b=0.5) while holding 'r'.
    _held_key_click(main_window, "r", 60.0)
    assert main_window.fit_controller.state.pending_fit_click == pytest.approx(100.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k marking_click_resolves`
Expected: FAIL — `pending_fit_click` is `60.0` (the raw click position), not `100.0`

- [ ] **Step 3: Write the implementation**

Replace `fit_mode.py`'s `on_click` (currently lines 366-392):

```python
    def on_click(self, event):
        if event.inaxes != self.main_window.axes or event.xdata is None:
            return
        if event.button != 1:
            return
        key = self._held_key
        channel_x = self.main_window.display_to_channel(event.xdata)
        if key == "b":
            self.state.add_bg_click(channel_x)
            self._redraw_progress()
        elif key == "r":
            completed = self.state.add_fit_click(channel_x)
            if completed is not None:
                self._reset_parameters_panel()
            self._redraw_progress()
        elif key == "p":
            proximity = self._pixel_proximity_to_data(event)
            # Convert the proximity *distance* to channel units by
            # inverting both endpoints and taking their difference,
            # rather than scaling by the calibration's derivative --
            # reuses the exact same, already-tested invert() with no
            # extra approximation math.
            proximity_channels = abs(
                self.main_window.display_to_channel(event.xdata + proximity) - channel_x
            )
            result = self.state.toggle_peak(channel_x, proximity_channels)
            if result is None:
                self._show_status_message(
                    "Mark the fit region (hold R and click twice) before marking peaks", 3000
                )
                return
            self._reset_parameters_panel()
            self._redraw_progress()
        else:
            return
        self.main_window._update_fit_mode_availability()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k marking_click_resolves`
Expected: PASS

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `pytest tests/ -q`
Expected: PASS, same count plus 1. All existing marking tests run with no calibration active, where `display_to_channel` is identity, so behavior should be unchanged.

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: convert marking clicks from keV back to channel when calibrated"
```

---

## Task 12: Calibrated redraw of in-progress marks (`_redraw_progress`)

**Files:**
- Modify: `fit_mode.py:297-337` (`_redraw_progress`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_redraw_progress_draws_marks_at_calibrated_position(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window._plot_data()

    fc = main_window.fit_controller
    fc.state.peak_positions = [100.0]  # channel 100
    fc._redraw_progress()

    # The peak-position progress line should be drawn at keV 60, not
    # channel 100.
    peak_lines = [
        line for line in main_window.axes.lines
        if line.get_linestyle() == ":" and line.get_color() == "red"
    ]
    assert len(peak_lines) == 1
    assert peak_lines[0].get_xdata()[0] == pytest.approx(60.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k redraw_progress_draws`
Expected: FAIL — the line is drawn at `100.0`, not `60.0`

- [ ] **Step 3: Write the implementation**

Replace `fit_mode.py`'s `_redraw_progress` (currently lines 297-337):

```python
    def _redraw_progress(self):
        """Clears and fully rebuilds every in-progress marking artist
        from the current FitModeState fields. Trades a little redundant
        redraw work (never more than a handful of artists) for avoiding
        any incremental per-artist bookkeeping -- no risk of a stale
        artist left behind by an evicted background region or a
        removed peak. FitModeState itself always stays channel-based;
        to_display converts to keV for drawing only, when calibrated."""
        for artist in self._progress_artists:
            try:
                artist.remove()
            except NotImplementedError:
                pass
        self._progress_artists = []

        axes = self.main_window.axes
        state = self.state
        to_display = self.main_window.channel_to_display

        if state.pending_bg_click is not None:
            self._progress_artists.append(
                axes.axvline(
                    to_display(state.pending_bg_click), color="gray", linestyle="--", linewidth=1
                )
            )
        for lo, hi in state.bg_regions:
            self._progress_artists.append(
                axes.axvspan(
                    to_display(lo), to_display(hi), color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA
                )
            )

        if state.pending_fit_click is not None:
            self._progress_artists.append(
                axes.axvline(
                    to_display(state.pending_fit_click), color="tab:blue", linestyle="--", linewidth=1
                )
            )
        if state.fit_region is not None:
            lo, hi = state.fit_region
            self._progress_artists.append(
                axes.axvspan(
                    to_display(lo), to_display(hi), color=FIT_REGION_COLOR, alpha=FIT_REGION_ALPHA
                )
            )

        for x in state.peak_positions:
            self._progress_artists.append(
                axes.axvline(to_display(x), color="red", linestyle=":", linewidth=1)
            )

        self.main_window.canvas.draw_idle()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k redraw_progress_draws`
Expected: PASS

- [ ] **Step 5: Add a test proving marks survive toggling calibration mid-session**

This is the concrete guarantee behind the design spec's "toggling mid-session is always safe" — `FitModeState` never changes representation, so marks made before calibration is turned on must still redraw correctly (at their newly-calibrated position) afterward, and back at their original channel position if turned off again. Add to `tests/test_fit_mode_ui.py`:

```python
def test_marks_survive_toggling_calibration_mid_session(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    _make_active_spectrum(main_window)
    fc = main_window.fit_controller

    # Mark a peak position while uncalibrated.
    fc.state.fit_region = (85.0, 115.0)
    fc.state.peak_positions = [100.0]
    fc._redraw_progress()
    peak_line = [
        line for line in main_window.axes.lines
        if line.get_linestyle() == ":" and line.get_color() == "red"
    ][0]
    assert peak_line.get_xdata()[0] == pytest.approx(100.0)

    # Turn calibration on -- the mark's stored value is untouched, only
    # where it's drawn changes.
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    fc._redraw_progress()
    assert fc.state.peak_positions == [100.0]  # stored value unchanged
    peak_line = [
        line for line in main_window.axes.lines
        if line.get_linestyle() == ":" and line.get_color() == "red"
    ][0]
    assert peak_line.get_xdata()[0] == pytest.approx(60.0)  # drawn at keV now

    # Turn it back off -- redraws at the original channel position again.
    main_window._calibration_active = False
    fc._redraw_progress()
    assert fc.state.peak_positions == [100.0]
    peak_line = [
        line for line in main_window.axes.lines
        if line.get_linestyle() == ":" and line.get_color() == "red"
    ][0]
    assert peak_line.get_xdata()[0] == pytest.approx(100.0)
```

Run: `pytest tests/test_fit_mode_ui.py -v -k marks_survive_toggling`
Expected: PASS without any implementation change -- this test exercises only what Steps 1-4 already built, confirming the mid-session-toggle guarantee holds rather than adding new behavior.

- [ ] **Step 6: Run the full existing test suite to confirm no regression**

Run: `pytest tests/ -q`
Expected: PASS, same count plus 2 (1 from Step 1, 1 from Step 5).

- [ ] **Step 7: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: draw in-progress marks at their calibrated position"
```

---

## Task 13: Calibrated redraw of committed fits (`draw_committed_fits`)

**Files:**
- Modify: `fit_mode.py:394-473` (`draw_committed_fits`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_draw_committed_fits_draws_peak_position_at_calibrated_x(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window.fit_controller.draw_committed_fits(spectrum)

    peak_label = main_window.axes.texts[-1]
    # Position 100.0 -> keV 60.0
    assert peak_label.get_position()[0] == pytest.approx(60.0)
    assert peak_label.get_text() == "60.0"


def test_draw_committed_fits_model_curve_unaffected_by_calibration(qapp):
    """The fitted curve's Y-values (counts) must be identical whether or
    not calibration is active -- only where it's drawn on X changes, the
    model itself is always evaluated in channel space."""
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )

    main_window.fit_controller.draw_committed_fits(spectrum)
    uncalibrated_curve = [
        line.get_ydata().copy() for line in main_window.axes.lines if line.get_linewidth() == 1.5
    ][0]

    main_window.axes.clear()
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window.fit_controller.draw_committed_fits(spectrum)
    calibrated_curve = [
        line.get_ydata().copy() for line in main_window.axes.lines if line.get_linewidth() == 1.5
    ][0]

    np.testing.assert_allclose(uncalibrated_curve, calibrated_curve)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k draw_committed_fits_draws_peak_position_at_calibrated_x`
Expected: FAIL — the label is drawn at `100.0` with text `"100.0"`, not `60.0`

- [ ] **Step 3: Write the implementation**

Replace `fit_mode.py`'s `draw_committed_fits` (currently lines 394-473):

```python
    def draw_committed_fits(self, spectrum):
        axes = self.main_window.axes
        # x in data coordinates, y in axes-fraction -- keeps peak labels
        # pinned near the top of the visible plot regardless of the
        # current y-axis scale (linear or log) or zoom level.
        label_transform = axes.get_xaxis_transform()
        theme = getattr(self.main_window, "_theme", "light")
        fit_color, bg_line_color = fit_drawing_colors(spectrum.color, theme)
        to_display = self.main_window.channel_to_display
        for result in spectrum.fits:
            if not result.visible:
                continue
            axes.axvspan(
                to_display(result.left_bg_region[0]), to_display(result.left_bg_region[1]),
                color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA,
            )
            axes.axvspan(
                to_display(result.right_bg_region[0]), to_display(result.right_bg_region[1]),
                color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA,
            )
            axes.axvspan(
                to_display(result.fit_region[0]), to_display(result.fit_region[1]),
                color=FIT_REGION_COLOR, alpha=FIT_REGION_ALPHA,
            )

            if isinstance(result, IntegrationResult):
                lo, hi = result.fit_region
                axes.plot(
                    [to_display(lo), to_display(hi)],
                    [result.background_density, result.background_density],
                    color=bg_line_color, linestyle="--", linewidth=1,
                )
                axes.annotate(
                    f"centroid={result.net_centroid:.1f}\n"
                    f"FWHM={result.net_fwhm:.1f}\n"
                    f"full={result.gross_area:.0f}\nnet={result.net_area:.0f}",
                    xy=(to_display(result.net_centroid), 0.95),
                    xycoords=label_transform,
                    ha="center", va="top",
                    fontsize=7, color=fit_color,
                )
                continue

            lo, hi = result.fit_region
            background_lo = result.background_slope * lo + result.background_intercept
            background_hi = result.background_slope * hi + result.background_intercept
            axes.plot(
                [to_display(lo), to_display(hi)], [background_lo, background_hi],
                color=bg_line_color, linestyle="--", linewidth=1,
            )

            # x_dense stays in channel space -- the model below (linear
            # background + Gaussian/hypermet peaks) is defined in terms
            # of the fitted channel-space parameters (peak.position,
            # peak.sigma, background_slope). Only the final plotted
            # x-coordinates are converted, via to_display(x_dense),
            # never the values used in the model math itself.
            x_dense = np.linspace(lo, hi, 200)
            total = result.background_slope * x_dense + result.background_intercept
            for peak in result.peaks:
                if result.tail_fraction is not None:
                    total = total + peak.amplitude * hypermet_left_tail(
                        x_dense, peak.position, peak.sigma,
                        result.tail_fraction, result.tail_beta,
                    )
                else:
                    total = total + peak.amplitude * np.exp(
                        -((x_dense - peak.position) ** 2) / (2 * peak.sigma ** 2)
                    )
            axes.plot(to_display(x_dense), total, color=fit_color, linewidth=1.5)

            # Peak decomposition: each peak's own contribution (background
            # + that single peak), so a multi-peak fit visually shows how
            # the total curve above decomposes into its components.
            background_dense = result.background_slope * x_dense + result.background_intercept
            for peak in result.peaks:
                if result.tail_fraction is not None:
                    component = peak.amplitude * hypermet_left_tail(
                        x_dense, peak.position, peak.sigma,
                        result.tail_fraction, result.tail_beta,
                    )
                else:
                    component = peak.amplitude * np.exp(
                        -((x_dense - peak.position) ** 2) / (2 * peak.sigma ** 2)
                    )
                axes.plot(
                    to_display(x_dense), background_dense + component,
                    color=fit_color, linewidth=0.75, linestyle="--", alpha=0.6,
                )

            for peak in result.peaks:
                label_x = to_display(peak.position)
                axes.axvline(label_x, color=fit_color, linestyle=":", linewidth=1)
                axes.annotate(
                    f"{label_x:.1f}",
                    xy=(label_x, 0.95),
                    xycoords=label_transform,
                    ha="center", va="top",
                    fontsize=7, color=fit_color,
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k draw_committed_fits_draws_peak_position_at_calibrated_x or draw_committed_fits_model_curve_unaffected`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `pytest tests/ -q`
Expected: PASS, same count plus 2. `draw_committed_fits` already has substantial existing coverage (region shading, decomposition lines, hidden results) — confirm none of it broke.

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: draw committed fits at their calibrated position, keep model math in channels"
```

---

## Task 14: Dual-unit Fit Results table and tooltip

**Files:**
- Modify: `fit_mode.py:633-706` (`update_results_list`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_results_table_shows_channels_only_when_inactive(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window.fit_controller.update_results_list()
    table = main_window.fit_controller.results_table
    position_text = table.item(0, 1).text()
    assert "keV" not in position_text
    assert position_text == "100.00 ± 0.10"


def test_results_table_shows_dual_units_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    position_text = table.item(0, 1).text()
    # position 100.0 +/- 0.1 -> keV 60.0 +/- 0.05 (err scaled by |b|=0.5)
    assert position_text == "100.00 ± 0.10 ch (60.00 ± 0.05 keV)"

    fwhm_text = table.item(0, 2).text()
    # fwhm 5.0 +/- 0.2 -> keV 2.50 +/- 0.10
    assert fwhm_text == "5.00 ± 0.20 ch (2.50 ± 0.10 keV)"

    volume_text = table.item(0, 3).text()
    assert volume_text == "1000.0 ± 50.0"  # unchanged -- area has no keV equivalent


def test_results_table_integration_row_shows_dual_units_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_density=20.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=200.0, background_area_err=10.0,
            background_centroid=100.0, background_centroid_err=1.0,
            background_fwhm=9.0, background_fwhm_err=0.5,
            background_skewness=0.0, background_skewness_err=0.1,
            net_area=800.0, net_area_err=32.0,
            net_centroid=100.0, net_centroid_err=0.6,
            net_fwhm=7.5, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    centroid_text = table.item(0, 1).text()
    assert "keV" in centroid_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k results_table_shows`
Expected: `test_results_table_shows_channels_only_when_inactive` PASSES already (no behavior change needed for the inactive case); `test_results_table_shows_dual_units_when_active` and the integration-row test FAIL, showing plain `"100.00 ± 0.10"` with no keV parenthetical.

- [ ] **Step 3: Write the implementation**

Add a small formatting helper near the top of `fit_mode.py` (after the existing module-level helpers like `_parameter_label`, before the `FitModeController` class):

```python
def _dual_unit_value(main_window, channel_value, channel_err, is_width):
    """Formats a channel-space value+error as "X.XX ± Y.YY ch (E.EE ±
    F.FF keV)" when calibration is active, or plain "X.XX ± Y.YY"
    otherwise. `is_width=False` (a position, e.g. peak centroid)
    converts through the full calibration (cal.apply, including the
    offset `a`); `is_width=True` (e.g. FWHM) scales by the local
    derivative only -- a width has no absolute position of its own, and
    running it through apply() would incorrectly add `a`. keV
    uncertainty is first-order error propagation through the
    calibration's local derivative in both cases -- exact for linear
    calibration, a good approximation for quadratic given realistic
    peak-width uncertainties are small relative to the calibration's
    curvature scale."""
    if not main_window._calibration_active or main_window._calibration is None:
        return f"{channel_value:.2f} ± {channel_err:.2f}"
    cal = main_window._calibration
    slope = abs(cal.derivative(channel_value))
    if is_width:
        energy = slope * channel_value
    else:
        energy = cal.apply(channel_value)
    energy_err = slope * channel_err
    return f"{channel_value:.2f} ± {channel_err:.2f} ch ({energy:.2f} ± {energy_err:.2f} keV)"
```

In `update_results_list`, replace the `values` list construction for the `IntegrationResult` branch (currently):

```python
                values = [
                    fit_label,
                    f"{result.net_centroid:.2f} ± {result.net_centroid_err:.2f}",
                    f"{result.net_fwhm:.2f} ± {result.net_fwhm_err:.2f}",
                    f"{result.net_area:.1f} ± {result.net_area_err:.1f}",
                    "—",  # no chi^2 concept for a direct-sum Integration result
                ]
```

with:

```python
                values = [
                    fit_label,
                    _dual_unit_value(
                        self.main_window, result.net_centroid, result.net_centroid_err, is_width=False
                    ),
                    _dual_unit_value(
                        self.main_window, result.net_fwhm, result.net_fwhm_err, is_width=True
                    ),
                    f"{result.net_area:.1f} ± {result.net_area_err:.1f}",
                    "—",  # no chi^2 concept for a direct-sum Integration result
                ]
```

And the `values` list for the Gaussian-fit peak rows (currently):

```python
                values = [
                    fit_label,
                    f"{peak.position:.2f} ± {peak.position_err:.2f}",
                    f"{peak.fwhm:.2f} ± {peak.fwhm_err:.2f}",
                    f"{peak.area:.1f} ± {peak.area_err:.1f}",
                    f"{result.reduced_chi2:.3g}" if result.reduced_chi2 is not None else "—",
                ]
```

with:

```python
                values = [
                    fit_label,
                    _dual_unit_value(
                        self.main_window, peak.position, peak.position_err, is_width=False
                    ),
                    _dual_unit_value(
                        self.main_window, peak.fwhm, peak.fwhm_err, is_width=True
                    ),
                    f"{peak.area:.1f} ± {peak.area_err:.1f}",
                    f"{result.reduced_chi2:.3g}" if result.reduced_chi2 is not None else "—",
                ]
```

(Volume and chi^2 columns are untouched — area/chi^2 have no energy-axis equivalent, per the design spec.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k results_table_shows`
Expected: PASS (3 passed)

- [ ] **Step 5: Update the tooltip lines the same way**

Add the failing test first:

```python
def test_tooltip_shows_dual_units_for_centroid_and_fwhm_when_active(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_density=20.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=200.0, background_area_err=10.0,
            background_centroid=100.0, background_centroid_err=1.0,
            background_fwhm=9.0, background_fwhm_err=0.5,
            background_skewness=0.0, background_skewness_err=0.1,
            net_area=800.0, net_area_err=32.0,
            net_centroid=100.0, net_centroid_err=0.6,
            net_fwhm=7.5, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True
    main_window.fit_controller.update_results_list()

    tooltip = main_window.fit_controller.results_table.item(0, 0).toolTip()
    assert "keV" in tooltip
```

Run: `pytest tests/test_fit_mode_ui.py -v -k tooltip_shows_dual_units`
Expected: FAIL — `_integration_tooltip` doesn't reference `self.main_window` at all currently (check its signature with `grep -n "_integration_tooltip" fit_mode.py`).

Check `_integration_tooltip`'s current signature and callers:

Run: `grep -n "_integration_tooltip" fit_mode.py`

Update `_integration_tooltip(result)` to `_integration_tooltip(main_window, result)`, and inside it change every `f"{label}: area={area:.1f}±{area_err:.1f}, centroid={centroid:.2f}±{centroid_err:.2f}, ..."` line to use `_dual_unit_value` for the centroid/FWHM parts while keeping area as a plain value:

```python
def _integration_tooltip(main_window, result):
    """Full gross/background/net breakdown for an IntegrationResult's Fit
    Results row tooltip -- mirrors the level of detail the existing
    left-tail-info tooltip gives for a Gaussian fit. Centroid/FWHM show
    dual units when calibration is active; area has no energy-axis
    equivalent and stays channel-only, matching the results table."""
    lines = []
    for label, prefix in (("Gross", "gross"), ("Background", "background"), ("Net", "net")):
        area = getattr(result, f"{prefix}_area")
        area_err = getattr(result, f"{prefix}_area_err")
        centroid = getattr(result, f"{prefix}_centroid")
        centroid_err = getattr(result, f"{prefix}_centroid_err")
        fwhm = getattr(result, f"{prefix}_fwhm")
        fwhm_err = getattr(result, f"{prefix}_fwhm_err")
        skewness = getattr(result, f"{prefix}_skewness")
        skewness_err = getattr(result, f"{prefix}_skewness_err")
        lines.append(
            f"{label}: area={area:.1f}±{area_err:.1f}, "
            f"centroid={_dual_unit_value(main_window, centroid, centroid_err, is_width=False)}, "
            f"FWHM={_dual_unit_value(main_window, fwhm, fwhm_err, is_width=True)}, "
            f"skewness={skewness:.3g}±{skewness_err:.3g}"
        )
    return "\n".join(lines)
```

Update its one call site in `update_results_list` from `tooltip = _integration_tooltip(result)` to `tooltip = _integration_tooltip(self.main_window, result)`.

Do the same for the peak-row tooltip lines (`peak_tooltip_lines`/`shared_tooltip_lines` in `update_results_list`) — find the lines currently reading:

```python
            shared_tooltip_lines = [
                f"region full (no bg subtracted): {result.gross_area:.1f} ± {result.gross_area_err:.1f}",
                f"region net (bg subtracted): {result.net_area:.1f} ± {result.net_area_err:.1f}",
                f"reduced chi^2: {result.reduced_chi2:.3g}" if result.reduced_chi2 is not None
                else "reduced chi^2: undefined (zero degrees of freedom)",
            ]
```

Area lines stay unchanged (no keV equivalent) — this block needs no edit. The per-peak tooltip lines also only cover full/net area (no centroid/FWHM shown there today), so no edit needed there either. Only `_integration_tooltip` actually has centroid/FWHM lines to convert.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k tooltip_shows_dual_units`
Expected: PASS

- [ ] **Step 7: Run the full existing test suite to confirm no regression**

Run: `pytest tests/ -q`
Expected: PASS, same count plus 4 (3 from Step 4 + 1 from Step 6). Existing table/tooltip tests run with no calibration active, where `_dual_unit_value` falls back to the original plain format — confirm none of the exact-string assertions in existing tests broke.

- [ ] **Step 8: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: show dual channel/keV units in Fit Results table and tooltip when calibrated"
```

---

## Task 15: Dual-unit JSON and text export

**Files:**
- Modify: `fit_export.py` (all functions)
- Modify: `fit_mode.py` (call sites: `run_fit`, `run_integration`, `_export_fits`)
- Test: `tests/test_fit_export.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_export.py` (check the file's existing imports at the top first — it already imports `FitResult`, `PeakResult`, etc. from `peak_fit`, and the export functions from `fit_export`):

```python
def test_json_record_omits_kev_fields_without_calibration():
    result = _make_result()
    record = fit_result_to_json_record(result, "eu.spe")
    assert record["peaks"][0].get("position_keV") is None


def test_json_record_includes_kev_fields_with_calibration():
    from calibration import Calibration

    result = _make_result()
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    record = fit_result_to_json_record(result, "eu.spe", calibration=cal)
    peak_record = record["peaks"][0]
    # _make_result()'s peak has position=100.0, position_err=0.1
    assert peak_record["position_keV"] == pytest.approx(cal.apply(100.0))
    assert peak_record["position_err_keV"] == pytest.approx(abs(cal.derivative(100.0)) * 0.1)
    assert peak_record["fwhm_keV"] == pytest.approx(cal.apply(7.0) - cal.a)  # b*7.0, since apply subtracts a implicitly below -- see note
```

Note on that last assertion: FWHM is a *width*, not a position — converting it to keV should scale by the calibration's slope only (`abs(cal.derivative(...)) * fwhm`), not run it through `apply()` (which would incorrectly add the offset `a`). Fix the test to match the correct implementation before running it:

```python
def test_json_record_includes_kev_fields_with_calibration():
    from calibration import Calibration

    result = _make_result()
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    record = fit_result_to_json_record(result, "eu.spe", calibration=cal)
    peak_record = record["peaks"][0]
    # _make_result()'s peak: position=100.0, position_err=0.1, fwhm=7.0, fwhm_err=0.2
    assert peak_record["position_keV"] == pytest.approx(cal.apply(100.0))
    assert peak_record["position_err_keV"] == pytest.approx(abs(cal.derivative(100.0)) * 0.1)
    assert peak_record["fwhm_keV"] == pytest.approx(abs(cal.derivative(100.0)) * 7.0)
    assert peak_record["fwhm_err_keV"] == pytest.approx(abs(cal.derivative(100.0)) * 0.2)


def test_text_report_includes_kev_parenthetical_with_calibration():
    from calibration import Calibration

    result = _make_result()
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    report = fit_result_to_text_report(result, "eu.spe", calibration=cal)
    assert "keV" in report


def test_text_report_omits_kev_parenthetical_without_calibration():
    result = _make_result()
    report = fit_result_to_text_report(result, "eu.spe")
    assert "keV" not in report
```

(Remove the first, incorrect version of `test_json_record_includes_kev_fields_with_calibration` from Step 1 before running — only the corrected version above should remain in the file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_export.py -v -k kev`
Expected: FAIL — `fit_result_to_json_record()` doesn't accept a `calibration` keyword argument yet (`TypeError`).

- [ ] **Step 3: Write the implementation**

In `fit_export.py`, update `_peak_record`:

```python
def _peak_record(peak, calibration):
    record = {
        "position": peak.position, "position_err": peak.position_err,
        "fwhm": peak.fwhm, "fwhm_err": peak.fwhm_err,
        "amplitude": peak.amplitude, "amplitude_err": peak.amplitude_err,
        "sigma": peak.sigma, "sigma_err": peak.sigma_err,
        "area": peak.area, "area_err": peak.area_err,
        "full_area": peak.full_area, "full_area_err": peak.full_area_err,
    }
    if calibration is not None:
        record["position_keV"] = calibration.apply(peak.position)
        record["position_err_keV"] = abs(calibration.derivative(peak.position)) * peak.position_err
        record["fwhm_keV"] = abs(calibration.derivative(peak.position)) * peak.fwhm
        record["fwhm_err_keV"] = abs(calibration.derivative(peak.position)) * peak.fwhm_err
    return record
```

(FWHM's keV conversion uses the derivative at the *peak's own position*, consistent with the results-table formula in Task 14 -- a width converted via the local slope near where it's measured, not evaluated at some other channel.)

Update `_integration_layer_record`:

```python
def _integration_layer_record(result, prefix, calibration):
    record = {
        "area": getattr(result, f"{prefix}_area"),
        "area_err": getattr(result, f"{prefix}_area_err"),
        "centroid": getattr(result, f"{prefix}_centroid"),
        "centroid_err": getattr(result, f"{prefix}_centroid_err"),
        "fwhm": getattr(result, f"{prefix}_fwhm"),
        "fwhm_err": getattr(result, f"{prefix}_fwhm_err"),
        "skewness": getattr(result, f"{prefix}_skewness"),
        "skewness_err": getattr(result, f"{prefix}_skewness_err"),
    }
    if calibration is not None:
        centroid = record["centroid"]
        record["centroid_keV"] = calibration.apply(centroid)
        record["centroid_err_keV"] = abs(calibration.derivative(centroid)) * record["centroid_err"]
        record["fwhm_keV"] = abs(calibration.derivative(centroid)) * record["fwhm"]
        record["fwhm_err_keV"] = abs(calibration.derivative(centroid)) * record["fwhm_err"]
    return record
```

Update `fit_result_to_json_record`:

```python
def fit_result_to_json_record(result, spectrum_path, calibration=None):
    """Converts one FitResult into a plain dict covering every
    parameter and uncertainty, ready for json.dumps() -- used by the
    automatic per-fit log. `spectrum_path` is recorded so a later
    re-read of the log can be traced back to its source spectrum.
    `calibration` (a calibration.Calibration or None) adds *_keV fields
    to each peak record when given; omitted entirely otherwise, so
    existing consumers of the auto-log aren't broken by new
    always-required fields."""
    return {
        "type": "fit",
        "timestamp": result.timestamp,
        "spectrum_path": spectrum_path,
        "left_bg_region": list(result.left_bg_region),
        "right_bg_region": list(result.right_bg_region),
        "fit_region": list(result.fit_region),
        "background_slope": result.background_slope,
        "background_intercept": result.background_intercept,
        "link_widths": result.link_widths,
        "tail_fraction": result.tail_fraction,
        "tail_fraction_err": result.tail_fraction_err,
        "tail_beta": result.tail_beta,
        "tail_beta_err": result.tail_beta_err,
        "fixed_params": dict(result.fixed_params),
        "peaks": [_peak_record(peak, calibration) for peak in result.peaks],
        "gross_area": result.gross_area, "gross_area_err": result.gross_area_err,
        "net_area": result.net_area, "net_area_err": result.net_area_err,
        "reduced_chi2": result.reduced_chi2,
    }
```

Update `integration_result_to_json_record`:

```python
def integration_result_to_json_record(result, spectrum_path, calibration=None):
    """Converts one IntegrationResult into a plain dict covering the
    full gross/background/net breakdown, ready for json.dumps() -- the
    Integration-mode analog of fit_result_to_json_record."""
    return {
        "type": "integration",
        "timestamp": result.timestamp,
        "spectrum_path": spectrum_path,
        "left_bg_region": list(result.left_bg_region),
        "right_bg_region": list(result.right_bg_region),
        "fit_region": list(result.fit_region),
        "background_density": result.background_density,
        "gross": _integration_layer_record(result, "gross", calibration),
        "background": _integration_layer_record(result, "background", calibration),
        "net": _integration_layer_record(result, "net", calibration),
    }
```

Update `_to_json_record` and `append_auto_log`:

```python
def _to_json_record(result, spectrum_path, calibration=None):
    if isinstance(result, IntegrationResult):
        return integration_result_to_json_record(result, spectrum_path, calibration)
    return fit_result_to_json_record(result, spectrum_path, calibration)


def append_auto_log(spectrum_path, result, calibration=None):
    """Appends one JSON-Lines record for `result` to
    `<spectrum_stem>_fits.jsonl`, next to the spectrum file. Raises
    OSError on failure (e.g. read-only directory) -- the caller must
    turn that into a non-blocking status message, since a failed log
    write must never invalidate an already-successful fit."""
    record = _to_json_record(result, spectrum_path, calibration)
    path = auto_log_path(spectrum_path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

Add a small text-formatting helper near `_format_err`:

```python
def _format_err(value, err):
    return f"{value:.6g} ± {err:.6g}"


def _format_dual(value, err, calibration, is_width):
    """Channel value+error, plus a keV parenthetical when `calibration`
    is given. `is_width=True` (e.g. FWHM) scales by the calibration's
    local derivative only -- a width has no absolute position of its
    own, and running it through apply() would incorrectly add the
    offset `a`. `is_width=False` (a position, e.g. peak centroid)
    converts through the full calibration."""
    text = _format_err(value, err)
    if calibration is None:
        return text
    slope = abs(calibration.derivative(value))
    energy_value = slope * value if is_width else calibration.apply(value)
    energy_err = slope * err
    return f"{text} ch ({_format_err(energy_value, energy_err)} keV)"
```

Update `fit_result_to_text_report`:

```python
def fit_result_to_text_report(result, spectrum_path, fit_number=1, calibration=None):
    """Human-readable report for one fit -- one block, used for both a
    single-fit export and (joined by write_text_report) an all-fits
    export. `fit_number` is the fit's 1-based index in the Fit Results
    list, for the block's heading only. `calibration` adds a keV
    parenthetical to Position/FWHM lines when given."""
    lines = [
        f"Fit {fit_number}",
        f"Spectrum: {spectrum_path}",
        f"Timestamp: {result.timestamp}",
        f"Fit region: [{result.fit_region[0]:.2f}, {result.fit_region[1]:.2f}]",
        f"Left background region: [{result.left_bg_region[0]:.2f}, {result.left_bg_region[1]:.2f}]",
        f"Right background region: [{result.right_bg_region[0]:.2f}, {result.right_bg_region[1]:.2f}]",
        f"Background: slope={result.background_slope:.6g}, intercept={result.background_intercept:.6g}",
        f"Independent widths: {not result.link_widths}",
        f"Reduced chi^2: {result.reduced_chi2:.4g}" if result.reduced_chi2 is not None
        else "Reduced chi^2: undefined (zero degrees of freedom)",
    ]
    if result.tail_fraction is not None:
        lines.append(
            f"Left tail: r={_format_err(result.tail_fraction, result.tail_fraction_err)}, "
            f"beta={_format_err(result.tail_beta, result.tail_beta_err)}"
        )
    if result.fixed_params:
        fixed_text = ", ".join(
            f"{name}={value:.6g}" for name, value in sorted(result.fixed_params.items())
        )
        lines.append(f"Fixed parameters: {fixed_text}")
    else:
        lines.append("Fixed parameters: none")
    lines.append(f"Region full area (no background subtracted): {_format_err(result.gross_area, result.gross_area_err)}")
    lines.append(f"Region net area (background subtracted):     {_format_err(result.net_area, result.net_area_err)}")
    lines.append("")
    for i, peak in enumerate(result.peaks):
        lines.append(f"  Peak {i + 1}:")
        lines.append(f"    Position:  {_format_dual(peak.position, peak.position_err, calibration, is_width=False)}")
        lines.append(f"    FWHM:      {_format_dual(peak.fwhm, peak.fwhm_err, calibration, is_width=True)}")
        lines.append(f"    Amplitude: {_format_err(peak.amplitude, peak.amplitude_err)}")
        lines.append(f"    Sigma:     {_format_err(peak.sigma, peak.sigma_err)}")
        lines.append(f"    Full area (no background subtracted): {_format_err(peak.full_area, peak.full_area_err)}")
        lines.append(f"    Net area (background subtracted):     {_format_err(peak.area, peak.area_err)}")
    return "\n".join(lines)
```

Update `integration_result_to_text_report`:

```python
def integration_result_to_text_report(result, spectrum_path, fit_number=1, calibration=None):
    """Human-readable report for one Integration result -- the
    Integration-mode analog of fit_result_to_text_report."""
    lines = [
        f"Fit {fit_number} (Integration)",
        f"Spectrum: {spectrum_path}",
        f"Timestamp: {result.timestamp}",
        f"Fit region: [{result.fit_region[0]:.2f}, {result.fit_region[1]:.2f}]",
        f"Left background region: [{result.left_bg_region[0]:.2f}, {result.left_bg_region[1]:.2f}]",
        f"Right background region: [{result.right_bg_region[0]:.2f}, {result.right_bg_region[1]:.2f}]",
        f"Background density: {result.background_density:.6g}",
        "",
    ]
    for label, prefix in (("Gross", "gross"), ("Background", "background"), ("Net", "net")):
        lines.append(f"  {label}:")
        lines.append(
            f"    Area:      {_format_err(getattr(result, f'{prefix}_area'), getattr(result, f'{prefix}_area_err'))}"
        )
        lines.append(
            f"    Centroid:  {_format_dual(getattr(result, f'{prefix}_centroid'), getattr(result, f'{prefix}_centroid_err'), calibration, is_width=False)}"
        )
        lines.append(
            f"    FWHM:      {_format_dual(getattr(result, f'{prefix}_fwhm'), getattr(result, f'{prefix}_fwhm_err'), calibration, is_width=True)}"
        )
        lines.append(
            f"    Skewness:  {_format_err(getattr(result, f'{prefix}_skewness'), getattr(result, f'{prefix}_skewness_err'))}"
        )
    return "\n".join(lines)
```

Update `_to_text_report` and `write_text_report`:

```python
def _to_text_report(result, spectrum_path, fit_number, calibration=None):
    if isinstance(result, IntegrationResult):
        return integration_result_to_text_report(result, spectrum_path, fit_number, calibration)
    return fit_result_to_text_report(result, spectrum_path, fit_number, calibration)


def write_text_report(path, results, spectrum_path, calibration=None):
    """Writes a plain-text report for one or more fits/integrations to
    `path`, overwriting any existing file. `results` is a list of
    (fit_number, result) pairs, in the order they should appear in the
    report."""
    blocks = [
        _to_text_report(result, spectrum_path, fit_number, calibration)
        for fit_number, result in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_export.py -v -k kev`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `pytest tests/ -q`
Expected: PASS, same count plus 4. Every export function's `calibration` parameter defaults to `None`, so every existing call site (none of which pass it yet) is unaffected.

- [ ] **Step 6: Commit**

```bash
git add fit_export.py tests/test_fit_export.py
git commit -m "feat: add optional dual-unit keV fields to JSON and text export"
```

---

## Task 16: Thread calibration from `MainWindow` into export calls

**Files:**
- Modify: `fit_mode.py` (`run_fit`, `run_integration`, `_export_fits` — find current line numbers with `grep -n "def run_fit\|def run_integration\|def _export_fits\|append_auto_log\|write_text_report" fit_mode.py`)
- Test: `tests/test_fit_mode_ui.py`

Tasks 1-15 made every export function accept an optional `calibration` parameter; this task is the last wiring step — passing `self.main_window._calibration` (only when active) from the three call sites in `fit_mode.py` that currently call `fit_export.append_auto_log`/`self._export_fits` without it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_run_fit_auto_log_includes_kev_when_calibrated(qapp, tmp_path):
    import json

    from calibration import Calibration

    main_window = MainWindow()
    path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=path)
    main_window._calibration = Calibration(kind="linear", a=10.0, b=0.5)
    main_window._calibration_active = True

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    from fit_export import auto_log_path
    with open(auto_log_path(path), encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["peaks"][0]["position_keV"] is not None


def test_run_fit_auto_log_omits_kev_when_not_calibrated(qapp, tmp_path):
    import json

    main_window = MainWindow()
    path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    from fit_export import auto_log_path
    with open(auto_log_path(path), encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["peaks"][0].get("position_keV") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k auto_log_includes_kev`
Expected: FAIL — `position_keV` is absent even with calibration active, since `run_fit` doesn't pass `calibration` to `append_auto_log` yet.

- [ ] **Step 3: Write the implementation**

Find the three call sites:

Run: `grep -n "fit_export.append_auto_log\|self._export_fits(" fit_mode.py`

In `run_fit`, change:

```python
        try:
            fit_export.append_auto_log(active.path, result)
        except OSError as exc:
```

to:

```python
        calibration = self.main_window._calibration if self.main_window._calibration_active else None
        try:
            fit_export.append_auto_log(active.path, result, calibration)
        except OSError as exc:
```

In `run_integration`, apply the same change to its `fit_export.append_auto_log(active.path, result)` call.

In `_export_fits`, find where it calls `fit_export.write_text_report(...)` (or however it invokes the export — check the exact current call) and thread `calibration` through the same way:

```python
        calibration = self.main_window._calibration if self.main_window._calibration_active else None
```

added before the `fit_export.write_text_report(path, [...], active.path)` call, then passed as the new final positional (or keyword) argument.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k auto_log_includes_kev`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full test suite one final time**

Run: `pytest tests/ -q`
Expected: PASS, all tests including every one added across Tasks 1-16.

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: pass active calibration into auto-log and manual export calls"
```

---

## Final Verification

- [ ] Run the full test suite: `pytest tests/ -q` — expect all green, no `xfail`/skip changes from before this plan.
- [ ] Manually smoke-test in the running app (per this project's `verify`/`run` conventions): load a spectrum, open `View > Calibration...`, enter a linear calibration (e.g. a=0, b=0.5), check Active, click OK — confirm the x-axis relabels to "Energy (keV)" and the trace redraws. Mark and run a fit — confirm the Fit Results table shows a `ch (... keV)` parenthetical for Position/FWHM. Toggle calibration off from the dialog — confirm the plot and table both revert to channels-only. Try a quadratic calibration and a "Load from file..." round trip with a hand-written 3-line file.
