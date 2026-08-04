# Spectrum Add/Subtract with Scaling Factor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new Operations-menu commands, "Add Spectra..." and "Subtract Spectra...", that combine two loaded spectra of equal length into a new third spectrum, with the second spectrum scaled by a user-provided factor (default 1) first. First feature of version 2.1.0.

**Architecture:** Follows the exact pattern already used three times in this codebase (Multiply/Rebin/Normalize): a pure-logic function in `spectrum_operations.py`, a small dedicated Qt dialog, and thin wiring in `main_window.py`. Documented in the HowTo page like every other Operations-menu entry.

**Tech Stack:** Python, PySide6/Qt, numpy, pytest.

**Design spec:** `docs/superpowers/specs/2026-08-04-spectrum-add-subtract-design.md` — read for the full rationale (including the TV source research this was grounded in) if anything below is unclear.

---

## Task 1: `spectrum_operations.py` — `add()` and `subtract()`

**Files:**
- Modify: `spectrum_operations.py`
- Test: `tests/test_spectrum_operations.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_spectrum_operations.py`:

```python
def test_add_sums_with_second_spectrum_scaled_by_factor():
    a = np.array([10, 20, 30], dtype=np.int64)
    b = np.array([1, 2, 3], dtype=np.int64)
    result = add(a, b, 2.0)
    assert list(result) == [12, 24, 36]
    assert result.dtype == np.int64


def test_add_rounds_fractional_results_to_nearest_integer():
    a = np.array([10, 10], dtype=np.int64)
    b = np.array([1, 3], dtype=np.int64)
    result = add(a, b, 0.5)
    # 10+0.5=10.5 -> 10 (round-half-to-even), 10+1.5=11.5 -> 12 (round-half-to-even)
    assert list(result) == [10, 12]


def test_add_does_not_clamp_negative_results():
    # add() itself never produces negatives from positive inputs and a
    # positive factor, but it must not clamp regardless -- this pins
    # down that no clamping code exists, using a factor large enough
    # that the caller could reasonably combine it with subtract() and
    # expect negatives to survive unchanged through add() too if ever
    # composed. Direct negative-result coverage is on subtract() below,
    # which is the realistic way this app produces negative results.
    a = np.array([0, 0], dtype=np.int64)
    b = np.array([5, -5], dtype=np.int64)
    result = add(a, b, 1.0)
    assert list(result) == [5, -5]


def test_subtract_scales_second_spectrum_before_subtracting():
    a = np.array([10, 20, 30], dtype=np.int64)
    b = np.array([1, 2, 3], dtype=np.int64)
    result = subtract(a, b, 2.0)
    assert list(result) == [8, 16, 24]
    assert result.dtype == np.int64


def test_subtract_allows_negative_results():
    a = np.array([5, 10], dtype=np.int64)
    b = np.array([10, 5], dtype=np.int64)
    result = subtract(a, b, 1.0)
    assert list(result) == [-5, 5]


def test_subtract_by_factor_one_is_plain_subtraction():
    a = np.array([10, 20], dtype=np.int64)
    b = np.array([3, 4], dtype=np.int64)
    result = subtract(a, b, 1.0)
    assert list(result) == [7, 16]
```

And update the import line at the top of the file:

```python
from spectrum_operations import add, multiply, normalize_factors, rebin, reference_value, subtract
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_spectrum_operations.py -v` (Windows) or `.venv/bin/python -m pytest tests/test_spectrum_operations.py -v` (Linux)

Expected: FAIL with `ImportError: cannot import name 'add' from 'spectrum_operations'` (or similar for `subtract`).

- [ ] **Step 3: Implement `add()` and `subtract()`**

Add to the end of `spectrum_operations.py`:

```python
def add(data_a, data_b, factor):
    """result[i] = A[i] + factor * B[i], rounded to nearest integer
    (matches multiply()'s rounding convention). A and B are assumed
    already validated as equal length by the caller. Unlike multiply()/
    rebin(), negative results are NOT clamped -- matches TV's own
    SpcAdd, which never clamps (tv-1.9.13/lib/tv/vsSpectra.c)."""
    return np.round(data_a + factor * data_b).astype(np.int64)


def subtract(data_a, data_b, factor):
    """result[i] = A[i] - factor * B[i], rounded to nearest integer.
    Same assumptions and TV-parity notes as add() above."""
    return np.round(data_a - factor * data_b).astype(np.int64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_spectrum_operations.py -v` (Windows) or `.venv/bin/python -m pytest tests/test_spectrum_operations.py -v` (Linux)
Expected: PASS, all tests in the file.

- [ ] **Step 5: Commit**

```bash
git add spectrum_operations.py tests/test_spectrum_operations.py
git commit -m "feat: add spectrum_operations.add() and subtract()"
```

---

## Task 2: `combine_dialog.py` — the new dialog

**Files:**
- Create: `combine_dialog.py`
- Test: `tests/test_combine_dialog.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_combine_dialog.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_combine_dialog.py -v` (Windows) or `.venv/bin/python -m pytest tests/test_combine_dialog.py -v` (Linux)
Expected: FAIL with `ModuleNotFoundError: No module named 'combine_dialog'`.

- [ ] **Step 3: Implement `CombineDialog`**

Create `combine_dialog.py`:

```python
"""Qt dialog for combining two loaded spectra (Add or Subtract), with
the second scaled by a user-provided factor first -- the two-spectrum-
picker analog of factor_dialog.py's single generic field. Not an
extension of FactorDialog, which is deliberately kept simple and shared
verbatim by Multiply/Rebin; bolting spectrum-pickers onto it would
compromise that reusability."""

import os

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)


class CombineDialog(QDialog):
    """`result_spectrum_a`, `result_spectrum_b`, `result_factor` are set
    only after a successful OK; stay `None` if cancelled or if every OK
    attempt failed to parse/validate. `spectra` is the full list of
    currently loaded LoadedSpectrum objects -- both dropdowns show all
    of them, unfiltered. `active_spectrum` (may be None) is used only to
    pick Spectrum A's default selection."""

    def __init__(self, parent, title, spectra, active_spectrum):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.result_spectrum_a = None
        self.result_spectrum_b = None
        self.result_factor = None
        self._spectra = spectra

        self._combo_a = QComboBox()
        self._combo_b = QComboBox()
        for spectrum in spectra:
            label = os.path.basename(spectrum.path)
            self._combo_a.addItem(label)
            self._combo_b.addItem(label)

        default_a_index = spectra.index(active_spectrum) if active_spectrum in spectra else 0
        self._combo_a.setCurrentIndex(default_a_index)
        # B defaults to the first spectrum that ISN'T A's default, so the
        # two dropdowns never start pointing at the same spectrum.
        default_b_index = 0 if default_a_index != 0 else min(1, len(spectra) - 1)
        self._combo_b.setCurrentIndex(default_b_index)

        self._factor_field = QLineEdit("1")

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Spectrum A:", self._combo_a)
        form.addRow("Spectrum B:", self._combo_b)
        form.addRow("Factor:", self._factor_field)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(button_box)

    def _on_accept(self):
        try:
            factor = float(self._factor_field.text())
        except ValueError:
            QMessageBox.warning(self, self.windowTitle(), "Please enter a valid number.")
            return
        if factor <= 0:
            QMessageBox.warning(self, self.windowTitle(), "Factor must be greater than zero.")
            return
        spectrum_a = self._spectra[self._combo_a.currentIndex()]
        spectrum_b = self._spectra[self._combo_b.currentIndex()]
        if len(spectrum_a.data) != len(spectrum_b.data):
            QMessageBox.warning(
                self, self.windowTitle(),
                f"Spectrum A has {len(spectrum_a.data)} channels; "
                f"Spectrum B has {len(spectrum_b.data)} channels. "
                "Add/Subtract requires both to have the same length.",
            )
            return
        self.result_spectrum_a = spectrum_a
        self.result_spectrum_b = spectrum_b
        self.result_factor = factor
        self.accept()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_combine_dialog.py -v` (Windows) or `.venv/bin/python -m pytest tests/test_combine_dialog.py -v` (Linux)
Expected: PASS, all tests in the file.

- [ ] **Step 5: Commit**

```bash
git add combine_dialog.py tests/test_combine_dialog.py
git commit -m "feat: add CombineDialog for spectrum add/subtract"
```

---

## Task 3: Wire "Add Spectra..." / "Subtract Spectra..." into `main_window.py`

**Files:**
- Modify: `main_window.py`

No dedicated unit tests for this file (matching this codebase's existing convention — `main_window.py`'s Qt wiring has no `tests/test_main_window.py`; it's verified by running the app, per Task 5 below). Run the full test suite after this task to confirm nothing existing broke.

- [ ] **Step 1: Update imports**

In `main_window.py`, change:
```python
from factor_dialog import FactorDialog
```
to:
```python
from combine_dialog import CombineDialog
from factor_dialog import FactorDialog
```

And change:
```python
from spectrum_operations import multiply, normalize_factors, rebin, reference_value
```
to:
```python
from spectrum_operations import add, multiply, normalize_factors, rebin, reference_value, subtract
```

- [ ] **Step 2: Add the two menu actions**

In `main_window.py`, find this existing block (currently ending the Operations menu, right before `self.help_menu = self.menuBar().addMenu("&Help")`):

```python
        self.normalize_action = QAction("Normalize Spectra", self)
        self.normalize_action.setShortcut("Ctrl+N")
        self.normalize_action.setEnabled(False)
        self.normalize_action.triggered.connect(self._normalize_spectra)
        self.operations_menu.addAction(self.normalize_action)

        self.help_menu = self.menuBar().addMenu("&Help")
```

Replace it with:

```python
        self.normalize_action = QAction("Normalize Spectra", self)
        self.normalize_action.setShortcut("Ctrl+N")
        self.normalize_action.setEnabled(False)
        self.normalize_action.triggered.connect(self._normalize_spectra)
        self.operations_menu.addAction(self.normalize_action)

        self.add_action = QAction("Add Spectra...", self)
        self.add_action.setShortcut("Ctrl+A")
        self.add_action.setEnabled(False)
        self.add_action.triggered.connect(self._open_add_dialog)
        self.operations_menu.addAction(self.add_action)

        self.subtract_action = QAction("Subtract Spectra...", self)
        self.subtract_action.setShortcut("Ctrl+Shift+A")
        self.subtract_action.setEnabled(False)
        self.subtract_action.triggered.connect(self._open_subtract_dialog)
        self.operations_menu.addAction(self.subtract_action)

        self.help_menu = self.menuBar().addMenu("&Help")
```

- [ ] **Step 3: Add the handler methods**

In `main_window.py`, find the existing `_normalize_spectra` method's closing (it ends with `self._plot_data(preserve_view=True)` — the last line of that method, per the code around what's currently line 588). Immediately after that method's end (and its blank line), insert:

```python
    def _open_add_dialog(self):
        if len(self.spectra) < 2:
            return
        active = next((s for s in self.spectra if s.active), None)
        dialog = CombineDialog(self, "Add Spectra", self.spectra, active)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_add(dialog.result_spectrum_a, dialog.result_spectrum_b, dialog.result_factor)

    def _apply_add(self, spectrum_a, spectrum_b, factor):
        data = add(spectrum_a.data, spectrum_b.data, factor)
        name_a = os.path.basename(spectrum_a.path)
        name_b = os.path.basename(spectrum_b.path)
        path = f"{name_a} + {name_b}" if factor == 1 else f"{name_a} + {factor}x{name_b}"
        self._add_combined_spectrum(path, data)

    def _open_subtract_dialog(self):
        if len(self.spectra) < 2:
            return
        active = next((s for s in self.spectra if s.active), None)
        dialog = CombineDialog(self, "Subtract Spectra", self.spectra, active)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_subtract(dialog.result_spectrum_a, dialog.result_spectrum_b, dialog.result_factor)

    def _apply_subtract(self, spectrum_a, spectrum_b, factor):
        data = subtract(spectrum_a.data, spectrum_b.data, factor)
        name_a = os.path.basename(spectrum_a.path)
        name_b = os.path.basename(spectrum_b.path)
        path = f"{name_a} - {name_b}" if factor == 1 else f"{name_a} - {factor}x{name_b}"
        self._add_combined_spectrum(path, data)

    def _add_combined_spectrum(self, path, data):
        # Shared by _apply_add/_apply_subtract -- inserting a newly
        # computed spectrum into the loaded list and refreshing every
        # UI surface that depends on it is identical bookkeeping either
        # way; only the operation and naming above differ.
        color_index = self._next_color_index
        self._next_color_index += 1
        spectrum = LoadedSpectrum(path, data, next_color(color_index, self._theme))
        spectrum.color_index = color_index
        for s in self.spectra:
            s.active = False
        spectrum.active = True
        self.spectra.append(spectrum)
        self._update_spectrum_list()
        self._update_fit_mode_availability()
        self._update_operations_availability()
        self.fit_controller.update_results_list()
        self._plot_data()
```

- [ ] **Step 4: Wire the enable/disable state**

In `main_window.py`, find `_update_operations_availability` (currently):

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
        self.save_spectrum_action.setEnabled(active is not None)
        self.close_spectrum_action.setEnabled(active is not None)
        visible_count = sum(1 for s in self.spectra if s.visible)
        self.normalize_action.setEnabled(visible_count >= 2)
```

Replace it with:

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
        self.save_spectrum_action.setEnabled(active is not None)
        self.close_spectrum_action.setEnabled(active is not None)
        visible_count = sum(1 for s in self.spectra if s.visible)
        self.normalize_action.setEnabled(visible_count >= 2)
        self.add_action.setEnabled(len(self.spectra) >= 2)
        self.subtract_action.setEnabled(len(self.spectra) >= 2)
```

- [ ] **Step 5: Run the full test suite to confirm nothing broke**

Run: `.venv/Scripts/python.exe -m pytest -q` (Windows) or `.venv/bin/python -m pytest -q` (Linux)
Expected: PASS, same count as before this task plus Tasks 1-2's new tests (no failures, no new errors).

- [ ] **Step 6: Commit**

```bash
git add main_window.py
git commit -m "feat: wire Add Spectra.../Subtract Spectra... into the Operations menu"
```

---

## Task 4: Document in the HowTo page

**Files:**
- Modify: `help_content.py`
- Modify: `tests/test_help_content.py`

- [ ] **Step 1: Add the two shortcut-table rows**

In `help_content.py`, find (in the Operations menu shortcut table):

```python
<tr><td><kbd>Ctrl+N</kbd></td><td>Normalize Spectra</td></tr>
</table>
```

Replace with:

```python
<tr><td><kbd>Ctrl+N</kbd></td><td>Normalize Spectra</td></tr>
<tr><td><kbd>Ctrl+A</kbd></td><td>Add Spectra...</td></tr>
<tr><td><kbd>Ctrl+Shift+A</kbd></td><td>Subtract Spectra...</td></tr>
</table>
```

- [ ] **Step 2: Insert the new numbered section and renumber what follows**

In `help_content.py`, find this exact block:

```python
<h3>5. Multiply, Rebin, and Normalize</h3>
<p><b>Multiply by Factor...</b> (<kbd>Ctrl+M</kbd>) scales the active
spectrum's counts by a factor you enter. <b>Rebin by Factor...</b>
(<kbd>Ctrl+R</kbd>) combines that many adjacent channels into one,
reducing the active spectrum's channel count -- note this only rescales
that one spectrum's calibration, so other loaded spectra can end up on a
different channel scale than the one you just rebinned. <b>Normalize
Spectra</b> (<kbd>Ctrl+N</kbd>) needs a reference point marked first --
hold <kbd>R</kbd> and click once for a single channel, or twice for a
region -- then scales every <i>visible</i> spectrum (at least two must
be visible) so they all read the same value there, useful for visually
comparing spectra taken with different live times.</p>

<h3>6. Performing a fit</h3>
```

Replace with:

```python
<h3>5. Multiply, Rebin, and Normalize</h3>
<p><b>Multiply by Factor...</b> (<kbd>Ctrl+M</kbd>) scales the active
spectrum's counts by a factor you enter. <b>Rebin by Factor...</b>
(<kbd>Ctrl+R</kbd>) combines that many adjacent channels into one,
reducing the active spectrum's channel count -- note this only rescales
that one spectrum's calibration, so other loaded spectra can end up on a
different channel scale than the one you just rebinned. <b>Normalize
Spectra</b> (<kbd>Ctrl+N</kbd>) needs a reference point marked first --
hold <kbd>R</kbd> and click once for a single channel, or twice for a
region -- then scales every <i>visible</i> spectrum (at least two must
be visible) so they all read the same value there, useful for visually
comparing spectra taken with different live times.</p>

<h3>6. Add and Subtract Spectra</h3>
<p><b>Add Spectra...</b> (<kbd>Ctrl+A</kbd>) and <b>Subtract
Spectra...</b> (<kbd>Ctrl+Shift+A</kbd>) each open a dialog to pick two
loaded spectra, Spectrum A and Spectrum B, plus a factor (defaulting to
1). The result is a new spectrum -- Spectrum A, plus or minus Spectrum B
scaled by the factor -- added alongside the originals, which are left
untouched. Both spectra must have the same number of channels; picking a
mismatched pair shows an error naming both channel counts instead of
proceeding. Subtracting can produce negative channel counts in the
result -- this is expected, not an error.</p>

<h3>7. Performing a fit</h3>
```

- [ ] **Step 3: Renumber the two remaining sections**

In `help_content.py`, find:
```python
<h3>7. Integration</h3>
```
Replace with:
```python
<h3>8. Integration</h3>
```

Then find:
```python
<h3>8. View options</h3>
```
Replace with:
```python
<h3>9. View options</h3>
```

- [ ] **Step 4: Update the two existing tests that enumerate shortcuts/topics**

In `tests/test_help_content.py`, find `test_howto_html_contains_every_shortcut`'s list and change:
```python
        "Ctrl+L", "Ctrl+T", "Ctrl+M", "Ctrl+R", "Ctrl+N",
```
to:
```python
        "Ctrl+L", "Ctrl+T", "Ctrl+M", "Ctrl+R", "Ctrl+N", "Ctrl+A", "Ctrl+Shift+A",
```

Then in `test_howto_html_covers_every_operation`'s list, change:
```python
        "Multiply",
        "Rebin",
        "Normalize",
        "Performing a fit",
```
to:
```python
        "Multiply",
        "Rebin",
        "Normalize",
        "Add and Subtract Spectra",
        "Performing a fit",
```

- [ ] **Step 5: Run the HowTo tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_help_content.py -v` (Windows) or `.venv/bin/python -m pytest tests/test_help_content.py -v` (Linux)
Expected: PASS, all tests (including `test_howto_html_has_balanced_tags`, which would catch a malformed `<h3>`/`<p>` edit).

- [ ] **Step 6: Commit**

```bash
git add help_content.py tests/test_help_content.py
git commit -m "docs: document Add/Subtract Spectra in the HowTo page"
```

---

## Task 5: Windows verification (primary focus)

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass, including everything added in Tasks 1-4.

- [ ] **Step 2: Rebuild and manually exercise the app**

```bash
powershell -File packaging\windows\build.ps1
```

Then run `dist\SpectraTools.exe` directly (matching this project's existing smoke-test pattern) and manually verify:
- Load two spectra of the same length. Confirm "Add Spectra..." and "Subtract Spectra..." are disabled with only one loaded, and become enabled once a second is loaded.
- Run Add Spectra with the default factor (1) — confirm a new spectrum appears in the list, named `"<a> + <b>"`, becomes the active spectrum, and its plotted values are the channel-by-channel sum of the two inputs.
- Run Add Spectra again with a non-1 factor (e.g. 0.5) — confirm the new spectrum's name includes the factor (`"<a> + 0.5x<b>"`) and its values match `A + 0.5*B` (spot-check a few channels).
- Run Subtract Spectra with a factor large enough to produce negative values in some channels — confirm the new spectrum is created (not blocked) and negative values are visible on the plot (e.g. as bars dropping below the zero line), not silently clamped to 0.
- Load a third spectrum with a *different* channel count. Attempt Add Spectra picking the mismatched pair — confirm the warning message names both actual channel counts and the dialog stays open (doesn't create a spectrum or close).
- Confirm the newly created combined spectrum can be fit normally (mark B/R/P, Ctrl+F) and, if a calibration is active, displays in keV correctly.
- Open the HowTo page (F1) and confirm the new shortcut rows and "6. Add and Subtract Spectra" section render correctly, and the renumbered sections (7/8/9) read correctly with no duplicate or skipped numbers.

- [ ] **Step 3: No commit** (verification-only task; fix and re-verify if anything above fails).

---

## Task 6: Linux verification (proportionate)

**Files:** none (verification only)

- [ ] **Step 1: Rebuild via the existing scripts**

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d AlmaLinux-8 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/build.sh"
```
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/build_deb.sh"
```

- [ ] **Step 2: Smoke-test on Ubuntu WSL** (the one Linux environment already confirmed working end-to-end by the user)

Write a verification script to `packaging/linux/output/_task6_verify.sh` and run it via `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u rig -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/_task6_verify.sh"` (per this project's established WSL invocation rules: `MSYS_NO_PATHCONV=1`, a real script file rather than inline `bash -c`, foreground execution):

```bash
#!/usr/bin/env bash
set -uo pipefail

echo "=== reinstalling the freshly built DEB ==="
apt-get install -y --reinstall /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/spectratools_*.deb > /tmp/task6_install.log 2>&1
echo "install exit: $?"
tail -5 /tmp/task6_install.log

echo "=== confirming the app still launches correctly (unrelated to this feature, but a cheap regression check) ==="
rm -f /tmp/task6_stdout.log /tmp/task6_stderr.log
timeout -k 1 8 spectratools >/tmp/task6_stdout.log 2>/tmp/task6_stderr.log
echo "EXIT CODE: $?"
cat /tmp/task6_stderr.log
```

Expected: install succeeds, `EXIT CODE: 124` (genuinely alive, not crashed), stderr only the known-benign Fontconfig/Qt warnings already documented from prior verification (no new errors). This confirms the rebuilt package (with this feature's code now bundled) still installs and launches correctly.

This is a launch-level regression check, not a full manual UI exercise of Add/Subtract itself on Linux — per the design spec, that depth of Linux-specific verification isn't warranted for a pure application-code change, and Task 5 already covers the feature's actual behavior thoroughly.

Delete the throwaway script when done:
```bash
rm packaging/linux/output/_task6_verify.sh
```

- [ ] **Step 3: No commit** (verification-only task).

---

## Final Step: Hand back to the user

Once all 6 tasks are complete, report back: what was built, the Windows and Linux verification results, and that this is the first piece of v2.1.0 (not yet frozen/released — that's a separate, later step following the established release process, including the CHANGELOG.md update).
