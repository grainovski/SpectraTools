# Operations Menu, Keyboard Shortcuts, and New Spectrum Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add keyboard shortcuts to every currently-unbound command, add a new **Operations** menu (housing Calibration, moved from View, plus three new spectrum operations), implement Multiply by Factor / Rebin by Factor / Normalize Spectra, add a brand-new Save Spectrum capability (`.txt`/`.spe`/`.spk` writers), and change `Ctrl+C` Clear so it deletes fits instead of merely hiding them.

**Architecture:** Pure-logic helpers (`spectrum_operations.py`, `Calibration.rescaled()`, the three format writers) stay Qt-free and independently testable, mirroring the existing `calibration.py`/`calibration_dialog.py` split. A new generic `FactorDialog` (mirroring `CalibrationDialog`) handles both Multiply's and Rebin's "ask for one validated number" UI. Each MainWindow-facing operation splits into a thin `_open_*_dialog` wrapper (shows the modal, hard to unit test) and a directly-testable `_apply_*` core, matching the codebase's existing `_open_calibration_dialog`/`_apply_calibration_change` split. `.spk` writing ports `lc2_compress` from the vendored `libmfile-1.0.7` C reference verbatim (verified below against real compressed-byte fixtures already in this repo).

**Tech Stack:** PySide6 (QAction/QMenu/QDialog/QFileDialog), numpy, matplotlib (FigureCanvasQTAgg) — same stack as the rest of the app. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-28-operations-menu-and-shortcuts-design.md` — read this first for full rationale; this plan implements it task-by-task and does not repeat its reasoning.

---

## Task-ordering note

Tasks 1-7 build and verify pure logic / independent behavior changes with no menu dependency. Task 8 builds the Operations menu **skeleton** (only items whose handlers already exist: Calibration, Toggle Calibration Active) so the menu itself is never left pointing at a non-existent method between tasks. Tasks 9-11 and 17 each add exactly one new menu item at the same time as its handler. This ordering means the app is fully working and every test passes after every single task — never mid-task.

---

### Task 1: `Calibration.rescaled()` for Rebin's coefficient adjustment

**Files:**
- Modify: `calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calibration.py`:

```python
def test_rescaled_linear_calibration():
    cal = Calibration(kind="linear", a=5.0, b=2.0)
    rescaled = cal.rescaled(4)
    assert rescaled.kind == "linear"
    assert rescaled.a == 5.0
    assert rescaled.b == 8.0


def test_rescaled_quadratic_calibration():
    cal = Calibration(kind="quadratic", a=1.0, b=2.0, c=3.0)
    rescaled = cal.rescaled(2)
    assert rescaled.a == 1.0
    assert rescaled.b == 4.0
    assert rescaled.c == 12.0


def test_rescaled_is_algebraically_equivalent_at_the_new_channel():
    # Defining property of the transform: E_old(new_channel * factor)
    # must equal E_new(new_channel), for arbitrary coefficients/factor/
    # channel -- not just the hand-verified numbers above.
    cal = Calibration(kind="quadratic", a=3.0, b=1.5, c=0.02)
    factor = 5
    rescaled = cal.rescaled(factor)
    new_channel = 37
    assert rescaled.apply(new_channel) == pytest.approx(cal.apply(new_channel * factor))
```

Check the top of `tests/test_calibration.py` already imports `pytest` and `Calibration`; if `pytest` isn't imported, add `import pytest` at the top.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_calibration.py -k rescaled -v`
Expected: FAIL with `AttributeError: 'Calibration' object has no attribute 'rescaled'`

- [ ] **Step 3: Implement `rescaled`**

In `calibration.py`, add this method to the `Calibration` class, right after `invert`:

```python
    def rescaled(self, factor):
        """A new Calibration equivalent to this one after every channel
        number is divided by `factor` (e.g. after rebinning by `factor`,
        where new channel k represents old channel k*factor) -- exact
        algebraic substitution ch_old = ch_new*factor into
        E = a + b*ch_old + c*ch_old**2, giving a'=a, b'=b*factor,
        c'=c*factor**2."""
        return Calibration(kind=self.kind, a=self.a, b=self.b * factor, c=self.c * factor ** 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_calibration.py -k rescaled -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add calibration.py tests/test_calibration.py
git commit -m "feat: add Calibration.rescaled for rebin coefficient adjustment"
```

---

### Task 2: `spectrum_operations.py` — `multiply()`

**Files:**
- Create: `spectrum_operations.py`
- Test: Create `tests/test_spectrum_operations.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spectrum_operations.py`:

```python
import numpy as np

from spectrum_operations import multiply


def test_multiply_scales_and_rounds():
    data = np.array([10, 20, 30], dtype=np.int64)
    result = multiply(data, 2.0)
    assert list(result) == [20, 40, 60]
    assert result.dtype == np.int64


def test_multiply_rounds_fractional_results_to_nearest_integer():
    data = np.array([10, 14], dtype=np.int64)
    result = multiply(data, 1.5)
    # 10*1.5=15.0 and 14*1.5=21.0 are both exact, so this test isn't
    # sensitive to numpy's round-half-to-even tie-breaking (see the
    # dedicated banker's-rounding test below).
    assert list(result) == [15, 21]


def test_multiply_uses_bankers_rounding_on_an_exact_half():
    data = np.array([11], dtype=np.int64)
    result = multiply(data, 1.5)
    # 11*1.5 = 16.5 -- np.round uses round-half-to-even, so this rounds
    # DOWN to 16 (nearest even), not up to 17.
    assert list(result) == [16]


def test_multiply_by_one_is_a_no_op():
    data = np.array([1, 2, 3], dtype=np.int64)
    result = multiply(data, 1.0)
    assert list(result) == [1, 2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spectrum_operations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spectrum_operations'`

- [ ] **Step 3: Implement `multiply`**

Create `spectrum_operations.py`:

```python
"""Pure-logic spectrum-processing operations (multiply/rebin/normalize)
-- no Qt dependency, mirroring calibration.py's split from its own Qt
dialog layer. main_window.py is the thin UI layer on top of this."""

import numpy as np


def multiply(data, factor):
    """Every channel's count scaled by `factor` and rounded to the
    nearest integer (numpy's round-half-to-even). `factor` is assumed
    already validated (> 0) by the caller."""
    return np.round(data * factor).astype(np.int64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spectrum_operations.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add spectrum_operations.py tests/test_spectrum_operations.py
git commit -m "feat: add multiply() spectrum operation"
```

---

### Task 3: `spectrum_operations.py` — `rebin()`

**Files:**
- Modify: `spectrum_operations.py`
- Test: `tests/test_spectrum_operations.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spectrum_operations.py`:

```python
from spectrum_operations import rebin


def test_rebin_sums_evenly_divisible_groups():
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    result = rebin(data, 2)
    assert list(result) == [3, 7, 11]


def test_rebin_zero_pads_the_final_group_when_not_evenly_divisible():
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = rebin(data, 2)
    assert list(result) == [3, 7, 5]


def test_rebin_channel_count_is_ceiling_division():
    data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = rebin(data, 3)
    assert len(result) == 2
    assert list(result) == [6, 9]


def test_rebin_by_factor_larger_than_length_is_a_single_bin():
    data = np.array([1, 2, 3], dtype=np.int64)
    result = rebin(data, 10)
    assert list(result) == [6]


def test_rebin_preserves_int64_dtype():
    data = np.array([1, 2, 3, 4], dtype=np.int64)
    result = rebin(data, 2)
    assert result.dtype == np.int64
```

Update the `from spectrum_operations import multiply` line at the top of the file to `from spectrum_operations import multiply, rebin`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spectrum_operations.py -k rebin -v`
Expected: FAIL with `ImportError: cannot import name 'rebin'`

- [ ] **Step 3: Implement `rebin`**

Append to `spectrum_operations.py`:

```python


def rebin(data, factor):
    """Sums every `factor` adjacent channels into one. Zero-pads the
    end of `data` up to the next multiple of `factor` first if it
    doesn't divide evenly, so the final output bin sums fewer real
    channels than the others (the padding contributes zero) rather
    than dropping real counts. `factor` is assumed already validated
    (integer >= 2) by the caller."""
    remainder = len(data) % factor
    if remainder:
        pad = np.zeros(factor - remainder, dtype=data.dtype)
        data = np.concatenate([data, pad])
    return data.reshape(-1, factor).sum(axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spectrum_operations.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add spectrum_operations.py tests/test_spectrum_operations.py
git commit -m "feat: add rebin() spectrum operation"
```

---

### Task 4: `spectrum_operations.py` — `reference_value()` and `normalize_factors()`

**Files:**
- Modify: `spectrum_operations.py`
- Test: `tests/test_spectrum_operations.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spectrum_operations.py`:

```python
from spectrum_operations import normalize_factors, reference_value


def test_reference_value_single_channel():
    data = np.array([10, 20, 30, 40])
    assert reference_value(data, channel=2) == 30


def test_reference_value_channel_out_of_range_is_zero():
    data = np.array([10, 20, 30])
    assert reference_value(data, channel=10) == 0
    assert reference_value(data, channel=-1) == 0


def test_reference_value_region_sums_inclusive():
    data = np.array([10, 20, 30, 40, 50])
    assert reference_value(data, region=(1, 3)) == 90


def test_reference_value_region_clamped_to_bounds():
    data = np.array([10, 20, 30])
    assert reference_value(data, region=(-5, 100)) == 60


def test_normalize_factors_scales_up_to_the_maximum():
    factors = normalize_factors([50, 100, 25])
    assert factors == [2.0, 1.0, 4.0]


def test_normalize_factors_skips_zero_values():
    factors = normalize_factors([0, 100, 50])
    assert factors == [None, 1.0, 2.0]


def test_normalize_factors_all_zero_is_a_no_op():
    factors = normalize_factors([0, 0, 0])
    assert factors == [1.0, 1.0, 1.0]
```

Update the import line to `from spectrum_operations import multiply, normalize_factors, rebin, reference_value`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spectrum_operations.py -k "reference_value or normalize_factors" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement both functions**

Append to `spectrum_operations.py`:

```python


def reference_value(data, channel=None, region=None):
    """The value to normalize a spectrum against: the count at a single
    channel, or the summed counts across a (lo, hi) region (inclusive,
    clamped to data's own bounds -- spectra being normalized against
    each other may have different lengths). Exactly one of
    channel/region should be given. Returns 0 for an out-of-range
    channel or an empty clamped region, rather than raising."""
    if region is not None:
        lo, hi = region
        lo = max(0, int(round(lo)))
        hi = min(len(data) - 1, int(round(hi)))
        if lo > hi:
            return 0
        return int(data[lo:hi + 1].sum())
    index = int(round(channel))
    if index < 0 or index >= len(data):
        return 0
    return int(data[index])


def normalize_factors(values):
    """Given each visible spectrum's reference value, returns a
    parallel list of scale factors: 1.0 for the maximum, max/value for
    every other positive entry, or None where the value is 0 (can't be
    scaled up to a nonzero max by multiplication -- the caller should
    skip that spectrum). If every value is 0, every factor is 1.0 (all
    already "at the max" trivially)."""
    peak = max(values)
    factors = []
    for value in values:
        if value == peak:
            factors.append(1.0)
        elif value == 0:
            factors.append(None)
        else:
            factors.append(peak / value)
    return factors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spectrum_operations.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add spectrum_operations.py tests/test_spectrum_operations.py
git commit -m "feat: add reference_value() and normalize_factors() for spectrum normalization"
```

---

### Task 5: `factor_dialog.py` — `FactorDialog`

**Files:**
- Create: `factor_dialog.py`
- Test: Create `tests/test_factor_dialog.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_factor_dialog.py`:

```python
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


def test_rejects_unparseable_text(qapp):
    dialog = _positive_factor_dialog()
    dialog._field.setText("not a number")
    dialog._on_accept()
    assert dialog.result_factor is None


def test_rejects_a_value_that_fails_validation(qapp):
    dialog = _positive_factor_dialog()
    dialog._field.setText("-3")
    dialog._on_accept()
    assert dialog.result_factor is None


def test_rejects_zero(qapp):
    dialog = _positive_factor_dialog()
    dialog._field.setText("0")
    dialog._on_accept()
    assert dialog.result_factor is None


def test_integer_parser_rejects_fractional_text(qapp):
    dialog = _rebin_factor_dialog()
    dialog._field.setText("2.5")
    dialog._on_accept()
    assert dialog.result_factor is None


def test_integer_parser_accepts_a_valid_factor(qapp):
    dialog = _rebin_factor_dialog()
    dialog._field.setText("4")
    dialog._on_accept()
    assert dialog.result_factor == 4


def test_integer_parser_rejects_factor_of_one(qapp):
    dialog = _rebin_factor_dialog()
    dialog._field.setText("1")
    dialog._on_accept()
    assert dialog.result_factor is None


def test_window_title_is_set(qapp):
    dialog = _positive_factor_dialog()
    assert dialog.windowTitle() == "Multiply by Factor"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_factor_dialog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'factor_dialog'`

- [ ] **Step 3: Implement `FactorDialog`**

Create `factor_dialog.py`:

```python
"""Qt dialog for a single validated numeric factor -- shared by
Multiply by Factor and Rebin by Factor, the thin UI layer with no
pure-logic counterpart of its own (there's no logic here beyond
parsing/validating one field), mirroring calibration_dialog.py's
QMessageBox-on-error style."""

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)


class FactorDialog(QDialog):
    """`result_factor` is set only after a successful OK; stays `None`
    if cancelled or if every OK attempt failed to parse/validate.
    `parse(text) -> value` should raise ValueError on unparseable text.
    `validate(value) -> str | None` returns an error message for an
    unacceptable value, or None if it's fine."""

    def __init__(self, parent, title, label, parse, validate):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.result_factor = None
        self._parse = parse
        self._validate = validate

        self._field = QLineEdit()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow(label, self._field)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(button_box)

    def _on_accept(self):
        try:
            value = self._parse(self._field.text())
        except ValueError:
            QMessageBox.warning(self, self.windowTitle(), "Please enter a valid number.")
            return
        error = self._validate(value)
        if error is not None:
            QMessageBox.warning(self, self.windowTitle(), error)
            return
        self.result_factor = value
        self.accept()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_factor_dialog.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add factor_dialog.py tests/test_factor_dialog.py
git commit -m "feat: add FactorDialog, shared by Multiply and Rebin"
```

---

### Task 6: `Ctrl+C` Clear now deletes fits instead of hiding them

**Files:**
- Modify: `fit_mode.py:376-382` (`FitModeController.clear`)
- Test: `tests/test_fit_mode_ui.py`

This is an explicit, spec-approved change to *existing* behavior (see the design spec's "Fits/Marks Clearing Semantics" section) — not new-feature-only scoping.

- [ ] **Step 1: Rewrite the test that locks in the old hide behavior**

In `tests/test_fit_mode_ui.py`, find `test_clear_hides_every_fit_for_the_active_spectrum_without_deleting_it` (currently around line 1558) and replace it entirely with:

```python
def test_clear_deletes_every_fit_for_the_active_spectrum(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()
    assert len(spectrum.fits) == 1

    main_window.fit_controller.clear()

    assert spectrum.fits == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_fit_mode_ui.py -k test_clear_deletes_every_fit_for_the_active_spectrum -v`
Expected: FAIL — `assert spectrum.fits == []` fails because the current code only hides (`len(spectrum.fits)` is still 1).

- [ ] **Step 3: Change `clear()` to delete, and extract a reusable `reset_marks()`**

In `fit_mode.py`, replace the current `clear` method (lines 376-382):

```python
    def clear(self):
        self._clear_progress()
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is not None:
            for result in active.fits:
                result.visible = False
        self.main_window._plot_data(preserve_view=True)
```

with:

```python
    def clear(self):
        self.reset_marks()
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is not None:
            active.fits.clear()
        self.main_window._plot_data(preserve_view=True)

    def reset_marks(self):
        """Public entry point for resetting in-progress marking state
        (in-progress background/fit-region/peak clicks and their drawn
        artists) without touching committed fits or replotting -- used
        by clear() (Ctrl+C) and by Multiply/Rebin/Normalize
        (main_window.py) after they change a spectrum's data, per the
        design spec's "Fits/Marks Clearing Semantics"."""
        self._clear_progress()
```

Note: this does **not** touch the *different* `result.visible = False` usage inside `run_fit()`/`run_integration()` (graying a superseded fit when a new one is committed over the same region) -- that mechanism is explicitly out of scope and stays as-is.

- [ ] **Step 4: Run the full fit-mode-ui test suite**

Run: `python -m pytest tests/test_fit_mode_ui.py -v`
Expected: PASS, all tests including:
  - `test_clear_deletes_every_fit_for_the_active_spectrum` (rewritten above)
  - `test_clear_discards_in_progress_marks_without_committing` (unchanged -- confirms the marks-clearing half still works)
  - `test_clear_forces_a_replot_so_the_canvas_actually_goes_blank` (unchanged)
  - `test_clear_with_no_active_spectrum_does_not_crash` (unchanged)
  - the two superseded-fit-graying tests that assert on `.visible` inside `run_fit()`/`run_integration()` (unchanged code path -- confirms this change didn't touch it)

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: Ctrl+C Clear now deletes fits instead of hiding them"
```

---

### Task 7: Persistent shortcuts for Clear All Fits / Export All Fits

**Files:**
- Modify: `fit_mode.py` (`build_results_panel`, `_on_results_context_menu`, plus two new methods near `_export_fits`)
- Test: `tests/test_fit_mode_ui.py`

The existing "Clear All Fits"/"Export All Fits..." context-menu items are built fresh every time the menu opens (`menu.addAction("Clear All Fits")`), so they can't carry a working keyboard shortcut. This task extracts their logic into named methods and adds persistent `QAction`s that trigger the same methods.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fit_mode_ui.py`:

```python
def test_clear_all_fits_shortcut_deletes_fits(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(_fit_result_with_one_peak())

    main_window.fit_controller.clear_all_fits_action.trigger()

    assert spectrum.fits == []


def test_clear_all_fits_action_has_shortcut(qapp):
    from PySide6.QtGui import QKeySequence
    main_window = MainWindow()
    assert main_window.fit_controller.clear_all_fits_action.shortcut() == QKeySequence("Ctrl+Shift+C")


def test_export_all_fits_action_has_shortcut(qapp):
    from PySide6.QtGui import QKeySequence
    main_window = MainWindow()
    assert main_window.fit_controller.export_all_fits_action.shortcut() == QKeySequence("Ctrl+E")


def test_export_all_fits_shortcut_does_nothing_with_no_fits(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    main_window.fit_controller.export_all_fits_action.trigger()  # must not raise/open a dialog
```

Check whether `_fit_result_with_one_peak` already exists in this file (it's referenced by the pre-existing `test_clear_all_fits_empties_panel_and_spectrum`); if it does, reuse it as-is.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_fit_mode_ui.py -k "clear_all_fits_shortcut or clear_all_fits_action_has_shortcut or export_all_fits" -v`
Expected: FAIL with `AttributeError: 'FitModeController' object has no attribute 'clear_all_fits_action'`

- [ ] **Step 3: Extract shared methods and add persistent actions**

In `fit_mode.py`, add two new methods right before `_export_fits` (around line 943):

```python
    def _clear_all_fits(self):
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        active.fits.clear()
        self.main_window._plot_data(preserve_view=True)

    def _export_all_fits(self):
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None or not active.fits:
            return
        self._export_fits(active, list(range(len(active.fits))))

```

Update `_on_results_context_menu` (around line 916) to call these instead of inlining the logic. Change:

```python
        elif export_all_action is not None and chosen == export_all_action:
            self._export_fits(active, list(range(len(active.fits))))
        elif chosen == clear_action:
            active.fits.clear()
            mw._plot_data(preserve_view=True)
```

to:

```python
        elif export_all_action is not None and chosen == export_all_action:
            self._export_all_fits()
        elif chosen == clear_action:
            self._clear_all_fits()
```

Now add the two persistent actions in `build_results_panel` (around line 626), right after `self.results_table.itemDoubleClicked.connect(...)` and before `self.results_dock = QDockWidget(...)`:

```python
        self.clear_all_fits_action = QAction("Clear All Fits", mw)
        self.clear_all_fits_action.setShortcut("Ctrl+Shift+C")
        self.clear_all_fits_action.triggered.connect(self._clear_all_fits)
        mw.addAction(self.clear_all_fits_action)

        self.export_all_fits_action = QAction("Export All Fits...", mw)
        self.export_all_fits_action.setShortcut("Ctrl+E")
        self.export_all_fits_action.triggered.connect(self._export_all_fits)
        mw.addAction(self.export_all_fits_action)

```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fit_mode_ui.py -v`
Expected: PASS, all tests (including the pre-existing `test_clear_all_fits_empties_panel_and_spectrum`, which exercises the same underlying deletion behavior and must be unaffected by this refactor)

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: add Ctrl+Shift+C/Ctrl+E shortcuts for Clear All Fits/Export All Fits"
```

---

### Task 8: Shortcuts audit + Operations menu skeleton

**Files:**
- Modify: `main_window.py` (`_build_menu`, `_apply_calibration_change`, `_build_zoom_buttons`)
- Modify: `fit_mode.py` (`build_results_panel`, `build_parameters_panel` — one-line shortcut additions)
- Test: Create `tests/test_operations_menu.py`

This task adds every shortcut whose handler **already exists** (Exit, Dark theme, Log scale, panel toggles, zoom/full-spectrum, Calibration, Toggle Calibration Active) and creates the Operations menu itself. Multiply/Rebin/Normalize/Save Spectrum menu items are deliberately *not* added here — each is added in its own task alongside its handler, so the menu never references a nonexistent method.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_operations_menu.py`:

```python
import os
import tempfile

import numpy as np
from matplotlib.backend_bases import MouseEvent
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QDialog

from main_window import MainWindow
from spectrum import LoadedSpectrum


def _click(main_window, xdata, ydata=10.0):
    ax = main_window.axes
    px, py = ax.transData.transform((xdata, ydata))
    event = MouseEvent("button_press_event", main_window.canvas, px, py, button=1)
    main_window.canvas.callbacks.process("button_press_event", event)


def _held_key_click(main_window, key_char, xdata, ydata=10.0):
    main_window.fit_controller._held_key = key_char
    _click(main_window, xdata, ydata)
    main_window.fit_controller._held_key = None


def _make_active_spectrum(main_window, path=None):
    if path is None:
        path = os.path.join(tempfile.mkdtemp(), "synthetic.txt")
    y = np.full(200, 20, dtype=np.int64)
    y[97:104] += (
        500 * np.exp(-((np.arange(97, 104) - 100.0) ** 2) / (2 * 3.0 ** 2))
    ).astype(np.int64)
    spectrum = LoadedSpectrum(path, y, "#1f77b4")
    spectrum.active = True
    main_window.spectra.append(spectrum)
    main_window._plot_data()
    return spectrum


def _commit_a_fit(main_window):
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()


def _menu_named(main_window, title):
    return next(a.menu() for a in main_window.menuBar().actions() if a.text() == title)


def test_operations_menu_exists_with_calibration_items(qapp):
    main_window = MainWindow()
    titles = [a.text() for a in main_window.menuBar().actions()]
    assert "&Operations" in titles

    operations_menu = _menu_named(main_window, "&Operations")
    item_texts = [a.text() for a in operations_menu.actions() if not a.isSeparator()]
    assert item_texts == ["Calibration...", "Toggle Calibration Active"]


def test_calibration_no_longer_under_view(qapp):
    main_window = MainWindow()
    view_menu = _menu_named(main_window, "&View")
    view_texts = [a.text() for a in view_menu.actions() if not a.isSeparator()]
    assert "Calibration..." not in view_texts


def test_view_menu_still_has_its_other_items(qapp):
    main_window = MainWindow()
    view_menu = _menu_named(main_window, "&View")
    view_texts = [a.text() for a in view_menu.actions() if not a.isSeparator()]
    assert view_texts == ["Log scale Y", "Spectra", "Dark theme"]


def test_new_shortcuts_are_set(qapp):
    main_window = MainWindow()
    assert main_window.exit_action.shortcut() == QKeySequence("Ctrl+Q")
    assert main_window.calibration_action.shortcut() == QKeySequence("Ctrl+L")
    assert main_window.calibration_active_menu_action.shortcut() == QKeySequence("Ctrl+T")
    assert main_window.dark_theme_action.shortcut() == QKeySequence("Ctrl+D")
    assert main_window.log_scale_action.shortcut() == QKeySequence("Ctrl+G")
    assert main_window.toggle_spectrum_panel_action.shortcut() == QKeySequence("Ctrl+1")
    assert main_window.zoom_in_action.shortcut() == QKeySequence("Ctrl+=")
    assert main_window.zoom_out_action.shortcut() == QKeySequence("Ctrl+-")
    assert main_window.full_spectrum_action.shortcut() == QKeySequence("Ctrl+0")
    assert main_window.fit_controller.toggle_results_panel_action.shortcut() == QKeySequence("Ctrl+2")
    assert main_window.fit_controller.toggle_parameters_panel_action.shortcut() == QKeySequence("Ctrl+3")


def test_toggle_calibration_active_menu_item_syncs_with_toolbar_button(qapp):
    from calibration import Calibration
    main_window = MainWindow()
    main_window._apply_calibration_change(Calibration(kind="linear", a=1.0, b=1.0), True)

    assert main_window.calibration_active_menu_action.isChecked() is True
    assert main_window.calibration_toggle_action.isChecked() is True

    main_window.calibration_active_menu_action.setChecked(False)

    assert main_window.calibration_toggle_action.isChecked() is False
    assert main_window._calibration_active is False


def test_toggle_calibration_active_menu_item_disabled_with_no_calibration(qapp):
    main_window = MainWindow()
    assert main_window.calibration_active_menu_action.isEnabled() is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_operations_menu.py -v`
Expected: FAIL — `AttributeError: 'MainWindow' object has no attribute 'exit_action'` (and similar for the Operations-menu tests)

- [ ] **Step 3: Rewrite `_build_menu`**

In `main_window.py`, replace the entire `_build_menu` method (currently lines 291-325) with:

```python
    def _build_menu(self):
        self.file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file_dialog)
        self.file_menu.addAction(open_action)

        self.recent_menu = self.file_menu.addMenu("Recent Files")

        self.file_menu.addSeparator()

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("&View")
        self.log_scale_action = QAction("Log scale Y", self)
        self.log_scale_action.setShortcut("Ctrl+G")
        self.log_scale_action.setCheckable(True)
        self.log_scale_action.toggled.connect(self._on_log_scale_toggled)
        view_menu.addAction(self.log_scale_action)

        self.toggle_spectrum_panel_action.setShortcut("Ctrl+1")
        view_menu.addAction(self.toggle_spectrum_panel_action)

        view_menu.addSeparator()
        self.dark_theme_action = QAction("Dark theme", self)
        self.dark_theme_action.setShortcut("Ctrl+D")
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.setChecked(self._theme == "dark")
        self.dark_theme_action.toggled.connect(self._on_theme_toggled)
        view_menu.addAction(self.dark_theme_action)

        self.operations_menu = self.menuBar().addMenu("&Operations")
        self.calibration_action = QAction("Calibration...", self)
        self.calibration_action.setShortcut("Ctrl+L")
        self.calibration_action.triggered.connect(self._open_calibration_dialog)
        self.operations_menu.addAction(self.calibration_action)

        self.calibration_active_menu_action = QAction("Toggle Calibration Active", self)
        self.calibration_active_menu_action.setShortcut("Ctrl+T")
        self.calibration_active_menu_action.setCheckable(True)
        self.calibration_active_menu_action.setEnabled(self._calibration is not None)
        self.calibration_active_menu_action.toggled.connect(self._on_calibration_toggle_action)
        self.operations_menu.addAction(self.calibration_active_menu_action)

        self.operations_menu.addSeparator()
```

Note `self.toggle_spectrum_panel_action` is built in `_build_spectrum_panel()`, which already runs before `_build_menu()` in `__init__` — no reordering needed.

- [ ] **Step 4: Sync the new menu action in `_apply_calibration_change`**

In `main_window.py`, in `_apply_calibration_change` (around line 388-391), change:

```python
        self._calibration = new_calibration
        self._calibration_active = new_active
        self.calibration_toggle_action.setEnabled(new_calibration is not None)
        self.calibration_toggle_action.setChecked(new_active)
```

to:

```python
        self._calibration = new_calibration
        self._calibration_active = new_active
        self.calibration_toggle_action.setEnabled(new_calibration is not None)
        self.calibration_toggle_action.setChecked(new_active)
        self.calibration_active_menu_action.setEnabled(new_calibration is not None)
        self.calibration_active_menu_action.setChecked(new_active)
```

- [ ] **Step 5: Add zoom/full-spectrum shortcuts**

In `main_window.py`, in `_build_zoom_buttons` (currently lines 670-684), add a `setShortcut` call to each of the three actions. Change:

```python
        self.zoom_in_action = QAction(_magnifier_icon("+", dark), "Zoom In X", self)
        self.zoom_in_action.triggered.connect(lambda: self._zoom_x(1 / ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction(_magnifier_icon("-", dark), "Zoom Out X", self)
        self.zoom_out_action.triggered.connect(lambda: self._zoom_x(ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_out_action)

        self.full_spectrum_action = QAction(_full_spectrum_icon(dark), "Show Full Spectrum", self)
        self.full_spectrum_action.triggered.connect(self._show_full_spectrum)
        self.nav_toolbar.addAction(self.full_spectrum_action)
```

to:

```python
        self.zoom_in_action = QAction(_magnifier_icon("+", dark), "Zoom In X", self)
        self.zoom_in_action.setShortcut("Ctrl+=")
        self.zoom_in_action.triggered.connect(lambda: self._zoom_x(1 / ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction(_magnifier_icon("-", dark), "Zoom Out X", self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_out_action.triggered.connect(lambda: self._zoom_x(ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_out_action)

        self.full_spectrum_action = QAction(_full_spectrum_icon(dark), "Show Full Spectrum", self)
        self.full_spectrum_action.setShortcut("Ctrl+0")
        self.full_spectrum_action.triggered.connect(self._show_full_spectrum)
        self.nav_toolbar.addAction(self.full_spectrum_action)
```

- [ ] **Step 6: Add Fit Results/Fit Parameters panel-toggle shortcuts**

In `fit_mode.py`, in `build_results_panel` (around line 640), change:

```python
        self.toggle_results_panel_action = QAction("Fit Results", mw)
        self.toggle_results_panel_action.setCheckable(True)
```

to:

```python
        self.toggle_results_panel_action = QAction("Fit Results", mw)
        self.toggle_results_panel_action.setShortcut("Ctrl+2")
        self.toggle_results_panel_action.setCheckable(True)
```

In `build_parameters_panel` (around line 680), change:

```python
        self.toggle_parameters_panel_action = QAction("Fit Parameters", mw)
        self.toggle_parameters_panel_action.setCheckable(True)
```

to:

```python
        self.toggle_parameters_panel_action = QAction("Fit Parameters", mw)
        self.toggle_parameters_panel_action.setShortcut("Ctrl+3")
        self.toggle_parameters_panel_action.setCheckable(True)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_operations_menu.py -v`
Expected: PASS (8 tests)

Then run the full suite to check for regressions from the `_build_menu` rewrite and the toolbar-icon dark/light generation not being affected:

Run: `python -m pytest tests/ -v`
Expected: PASS, all tests

- [ ] **Step 8: Commit**

```bash
git add main_window.py fit_mode.py tests/test_operations_menu.py
git commit -m "feat: add Operations menu skeleton and shortcuts for existing commands"
```

---

### Task 9: Multiply by Factor wiring

**Files:**
- Modify: `main_window.py`
- Test: `tests/test_operations_menu.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_operations_menu.py` (note: `QDialog` is already imported at the top of this file from Task 8):

```python
def test_apply_multiply_scales_data(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window._apply_multiply(spectrum, 2.0)

    assert spectrum.data[0] == 40  # baseline 20 * 2


def test_apply_multiply_deletes_fits_and_resets_marks(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    _commit_a_fit(main_window)
    assert len(spectrum.fits) == 1
    _held_key_click(main_window, "b", 70)

    main_window._apply_multiply(spectrum, 2.0)

    assert spectrum.fits == []
    assert main_window.fit_controller.state.pending_bg_click is None


def test_apply_multiply_preserves_the_current_view(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    spectrum = main_window.spectra[0]
    main_window.axes.set_xlim(10, 50)

    main_window._apply_multiply(spectrum, 2.0)

    assert main_window.axes.get_xlim() == (10.0, 50.0)


def test_apply_multiply_leaves_other_spectra_untouched(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    original_b = spectrum_b.data.copy()

    main_window._apply_multiply(spectrum_a, 2.0)

    assert list(spectrum_b.data) == list(original_b)


def test_open_multiply_dialog_applies_the_entered_factor(qapp, monkeypatch):
    from factor_dialog import FactorDialog
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    def fake_exec(self):
        self.result_factor = 3.0
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FactorDialog, "exec", fake_exec)
    main_window._open_multiply_dialog()

    assert spectrum.data[0] == 60


def test_open_multiply_dialog_does_nothing_when_cancelled(qapp, monkeypatch):
    from factor_dialog import FactorDialog
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    original = spectrum.data.copy()

    def fake_exec(self):
        self.result_factor = None
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(FactorDialog, "exec", fake_exec)
    main_window._open_multiply_dialog()

    assert list(spectrum.data) == list(original)


def test_open_multiply_dialog_does_nothing_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    main_window._open_multiply_dialog()  # must not raise


def test_multiply_action_disabled_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    assert main_window.multiply_action.isEnabled() is False


def test_multiply_action_enabled_with_active_spectrum(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    assert main_window.multiply_action.isEnabled() is True


def test_multiply_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.multiply_action.shortcut() == QKeySequence("Ctrl+M")


def test_multiply_action_in_operations_menu(qapp):
    main_window = MainWindow()
    assert main_window.multiply_action in main_window.operations_menu.actions()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_operations_menu.py -k multiply -v`
Expected: FAIL with `AttributeError: 'MainWindow' object has no attribute '_apply_multiply'`

- [ ] **Step 3: Add the import**

In `main_window.py`, the local imports currently read (in this order):

```python
from calibration_dialog import CalibrationDialog
from fit_mode import FitModeController
from histogram_io import ParseError, load_histogram
from settings import Settings
from spe_io import load_spe
from spectrum import LoadedSpectrum, next_color
from spk_io import load_spk
from theme import qt_stylesheet, style_axes
```

Replace that whole block with (two new lines inserted, alphabetically among the rest — `spectrum_operations` will gain `rebin`/`reference_value`/`normalize_factors` in later tasks):

```python
from calibration_dialog import CalibrationDialog
from factor_dialog import FactorDialog
from fit_mode import FitModeController
from histogram_io import ParseError, load_histogram
from settings import Settings
from spe_io import load_spe
from spectrum import LoadedSpectrum, next_color
from spectrum_operations import multiply
from spk_io import load_spk
from theme import qt_stylesheet, style_axes
```

- [ ] **Step 4: Add the menu action, availability hook, and handlers**

In `_build_menu`, right after the `self.operations_menu.addSeparator()` line added in Task 8, add:

```python
        self.multiply_action = QAction("Multiply by Factor...", self)
        self.multiply_action.setShortcut("Ctrl+M")
        self.multiply_action.setEnabled(False)
        self.multiply_action.triggered.connect(self._open_multiply_dialog)
        self.operations_menu.addAction(self.multiply_action)
```

Add a new method `_update_operations_availability`, right after `_update_fit_mode_availability` (currently lines 757-761):

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
```

Call it alongside every existing call to `_update_fit_mode_availability()` in `main_window.py`. There are three: at the end of `__init__` (line 276), at the end of `_plot_data` (line 530), and at the end of `_on_active_toggled` (line 651). At each of those three lines, add `self._update_operations_availability()` on the line immediately after `self._update_fit_mode_availability()`.

Add the dialog wrapper and testable core, right after `_open_calibration_dialog`/`_apply_calibration_change` (after line 398, before `_style_nav_toolbar_palette`):

```python
    def _open_multiply_dialog(self):
        active = next((s for s in self.spectra if s.active), None)
        if active is None:
            return
        dialog = FactorDialog(
            self, "Multiply by Factor", "Factor:",
            parse=float,
            validate=lambda v: None if v > 0 else "Factor must be greater than zero.",
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_multiply(active, dialog.result_factor)

    def _apply_multiply(self, spectrum, factor):
        spectrum.data = multiply(spectrum.data, factor)
        self.fit_controller.reset_marks()
        spectrum.fits.clear()
        self._plot_data(preserve_view=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_operations_menu.py -v`
Expected: PASS, all tests

Run: `python -m pytest tests/ -v`
Expected: PASS, all tests (checks `_update_operations_availability` didn't break the three call sites it was added to)

- [ ] **Step 6: Commit**

```bash
git add main_window.py tests/test_operations_menu.py
git commit -m "feat: wire up Multiply by Factor"
```

---

### Task 10: Rebin by Factor wiring

**Files:**
- Modify: `main_window.py`
- Test: `tests/test_operations_menu.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_operations_menu.py`:

```python
def test_apply_rebin_reduces_channel_count_and_sums(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)

    main_window._apply_rebin(spectrum, 2)

    assert list(spectrum.data) == [3, 7, 11]


def test_apply_rebin_adjusts_linear_calibration(qapp):
    from calibration import Calibration
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data = np.array([1, 2, 3, 4], dtype=np.int64)
    main_window._calibration = Calibration(kind="linear", a=1.0, b=2.0)

    main_window._apply_rebin(spectrum, 2)

    assert main_window._calibration.a == 1.0
    assert main_window._calibration.b == 4.0


def test_apply_rebin_adjusts_quadratic_calibration(qapp):
    from calibration import Calibration
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data = np.array([1, 2, 3, 4], dtype=np.int64)
    main_window._calibration = Calibration(kind="quadratic", a=1.0, b=2.0, c=0.5)

    main_window._apply_rebin(spectrum, 3)

    assert main_window._calibration.a == 1.0
    assert main_window._calibration.b == 6.0
    assert main_window._calibration.c == 4.5


def test_apply_rebin_with_no_calibration_leaves_it_none(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window._apply_rebin(spectrum, 2)

    assert main_window._calibration is None


def test_apply_rebin_deletes_fits_and_resets_marks(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    _commit_a_fit(main_window)
    assert len(spectrum.fits) == 1
    _held_key_click(main_window, "b", 70)

    main_window._apply_rebin(spectrum, 2)

    assert spectrum.fits == []
    assert main_window.fit_controller.state.pending_bg_click is None


def test_apply_rebin_does_not_preserve_the_current_view(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    main_window.axes.set_xlim(10, 50)

    main_window._apply_rebin(spectrum, 2)

    assert main_window.axes.get_xlim() != (10.0, 50.0)


def test_open_rebin_dialog_applies_the_entered_factor(qapp, monkeypatch):
    from factor_dialog import FactorDialog
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.data = np.array([1, 2, 3, 4], dtype=np.int64)

    def fake_exec(self):
        self.result_factor = 2
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(FactorDialog, "exec", fake_exec)
    main_window._open_rebin_dialog()

    assert list(spectrum.data) == [3, 7]


def test_open_rebin_dialog_does_nothing_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    main_window._open_rebin_dialog()  # must not raise


def test_rebin_action_disabled_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    assert main_window.rebin_action.isEnabled() is False


def test_rebin_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.rebin_action.shortcut() == QKeySequence("Ctrl+R")


def test_rebin_action_in_operations_menu(qapp):
    main_window = MainWindow()
    assert main_window.rebin_action in main_window.operations_menu.actions()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_operations_menu.py -k rebin -v`
Expected: FAIL with `AttributeError: 'MainWindow' object has no attribute '_apply_rebin'`

- [ ] **Step 3: Update the import**

In `main_window.py`, change:

```python
from spectrum_operations import multiply
```

to:

```python
from spectrum_operations import multiply, rebin
```

- [ ] **Step 4: Add the menu action and handlers**

In `_build_menu`, right after the `multiply_action` block added in Task 9, add:

```python
        self.rebin_action = QAction("Rebin by Factor...", self)
        self.rebin_action.setShortcut("Ctrl+R")
        self.rebin_action.setEnabled(False)
        self.rebin_action.triggered.connect(self._open_rebin_dialog)
        self.operations_menu.addAction(self.rebin_action)
```

In `_update_operations_availability`, change:

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
```

to:

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
```

Add the dialog wrapper and testable core, right after `_apply_multiply` (added in Task 9):

```python
    def _open_rebin_dialog(self):
        active = next((s for s in self.spectra if s.active), None)
        if active is None:
            return
        dialog = FactorDialog(
            self, "Rebin by Factor", "Factor:",
            parse=int,
            validate=lambda v: None if v >= 2 else "Rebin factor must be an integer of at least 2.",
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_rebin(active, dialog.result_factor)

    def _apply_rebin(self, spectrum, factor):
        spectrum.data = rebin(spectrum.data, factor)
        if self._calibration is not None:
            # Calibration is a single MainWindow-level object shared by
            # every loaded spectrum (see __init__), but rebinning changes
            # only THIS spectrum's channel count -- rescaling it here keeps
            # the just-rebinned spectrum's keV axis correct at the cost of
            # desyncing it for any OTHER already-loaded spectrum, which
            # still has its original channel scale. Rescaling is the right
            # default (not rescaling would immediately break the spectrum
            # that was just rebinned), so this is flagged to the user via
            # a status message rather than blocked or silently skipped.
            self._calibration = self._calibration.rescaled(factor)
            if len(self.spectra) > 1:
                # fit_controller._show_status_message (not statusBar()
                # directly) -- it arms _status_message_until, which
                # _on_mouse_move checks before overwriting the status bar
                # with the ordinary hover readout. Without this, the
                # warning gets wiped by the very next mouse move over the
                # canvas, which is nearly guaranteed to happen right after
                # rebinning (found during code review, verified empirically).
                self.fit_controller._show_status_message(
                    "Rebinned. Calibration was rescaled for this spectrum -- "
                    "it may no longer be correct for other loaded spectra.",
                    8000,
                )
        self.fit_controller.reset_marks()
        spectrum.fits.clear()
        self._plot_data()
```

**Post-review addition (discovered during Task 10's code-quality review, not in the original design spec):** `self._calibration` is global (applies to every loaded spectrum, per the original calibration feature's own design), but Rebin is deliberately per-spectrum-scoped. Rescaling the shared calibration for one rebinned spectrum silently desyncs it for every other currently-loaded spectrum (their displayed keV axis, hover readout, and existing fits' keV columns become wrong with no error). Rescaling is still the right default (the alternative breaks the just-rebinned spectrum immediately), but it's now surfaced via a non-blocking status message when more than one spectrum is loaded, rather than shipped silent. Two tests pin this: `test_apply_rebin_warns_when_other_spectra_are_loaded` and `test_apply_rebin_no_warning_with_only_one_spectrum_loaded`.

**Second post-review fix**: the warning must be shown via `self.fit_controller._show_status_message(...)`, not `self.statusBar().showMessage(...)` directly — the latter gets silently overwritten by `_on_mouse_move`'s ordinary hover readout on the very next mouse movement over the canvas (verified empirically: the warning was gone well before its nominal 8-second duration after a single synthetic mouse-move event). `_show_status_message` arms `_status_message_until`, which `_on_mouse_move` already checks before overwriting. `test_apply_rebin_warning_survives_a_mouse_move` pins this.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_operations_menu.py -v`
Expected: PASS, all tests

- [ ] **Step 6: Commit**

```bash
git add main_window.py tests/test_operations_menu.py
git commit -m "feat: wire up Rebin by Factor"
```

---

### Task 11: Normalize Spectra wiring

**Files:**
- Modify: `main_window.py`
- Test: `tests/test_operations_menu.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_operations_menu.py`:

```python
def test_normalize_disabled_with_fewer_than_two_visible_spectra(qapp):
    main_window = MainWindow()
    assert main_window.normalize_action.isEnabled() is False
    _make_active_spectrum(main_window)
    assert main_window.normalize_action.isEnabled() is False


def test_normalize_enabled_with_two_visible_spectra(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)
    assert main_window.normalize_action.isEnabled() is True


def test_normalize_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.normalize_action.shortcut() == QKeySequence("Ctrl+N")


def test_normalize_action_in_operations_menu(qapp):
    main_window = MainWindow()
    assert main_window.normalize_action in main_window.operations_menu.actions()


def test_normalize_with_no_marks_shows_a_status_message(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)

    main_window._normalize_spectra()

    assert "mark" in main_window.statusBar().currentMessage().lower()


def test_normalize_no_marks_message_survives_a_mouse_move(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)

    main_window._normalize_spectra()

    ax = main_window.axes
    px, py = ax.transData.transform((10.0, 10.0))
    event = MouseEvent("motion_notify_event", main_window.canvas, px, py)
    main_window.canvas.callbacks.process("motion_notify_event", event)

    assert "mark" in main_window.statusBar().currentMessage().lower()


def test_normalize_single_marker_scales_by_bin_count(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20, 30], dtype=np.int64)
    spectrum_b.data = np.array([5, 40, 15], dtype=np.int64)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert list(spectrum_a.data) == [20, 40, 60]
    assert list(spectrum_b.data) == [5, 40, 15]


def test_normalize_region_marker_scales_by_area(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20, 30, 5], dtype=np.int64)
    spectrum_b.data = np.array([5, 10, 15, 5], dtype=np.int64)
    main_window.fit_controller.state.fit_region = (1, 2)

    main_window._normalize_spectra()

    assert list(spectrum_a.data) == [10, 20, 30, 5]
    assert list(spectrum_b.data) == [10, 20, 30, 10]


def test_normalize_skips_a_zero_reference_spectrum_with_a_message(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20], dtype=np.int64)
    spectrum_b.data = np.array([0, 0], dtype=np.int64)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert list(spectrum_b.data) == [0, 0]
    message = main_window.statusBar().currentMessage()
    assert os.path.basename(spectrum_b.path) in message


def test_normalize_skipped_message_survives_a_mouse_move(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20], dtype=np.int64)
    spectrum_b.data = np.array([0, 0], dtype=np.int64)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    ax = main_window.axes
    px, py = ax.transData.transform((10.0, 10.0))
    event = MouseEvent("motion_notify_event", main_window.canvas, px, py)
    main_window.canvas.callbacks.process("motion_notify_event", event)

    message = main_window.statusBar().currentMessage()
    assert os.path.basename(spectrum_b.path) in message


class _FakeFit:
    """Stand-in for a FitResult/IntegrationResult, used where a test only
    cares whether normalize's fit-clearing touches a given spectrum's
    .fits list -- not what a real fit looks like. Every real fit result
    always has `.visible` (peak_fit.FitResult/IntegrationResult both
    default it to True), and draw_committed_fits (fit_mode.py) reads
    that attribute first, before anything else, for every spectrum
    _plot_data redraws -- including spectra a given operation didn't
    touch. `visible = False` here makes that redraw skip this fake
    entry immediately, instead of crashing on the fit-region/background
    fields a bare placeholder (e.g. a plain string) doesn't have."""
    visible = False


def test_normalize_clears_fits_only_on_rescaled_spectra(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20], dtype=np.int64)
    spectrum_b.data = np.array([5, 40], dtype=np.int64)
    fake_fit_b = _FakeFit()
    spectrum_a.fits.append(_FakeFit())
    spectrum_b.fits.append(fake_fit_b)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert spectrum_a.fits == []
    assert spectrum_b.fits == [fake_fit_b]
```

**Post-review fix**: the original draft of this test used bare strings (`"fake-fit-a"`/`"fake-fit-b"`) as fake fit placeholders. That crashes: `_normalize_spectra`'s closing `_plot_data(preserve_view=True)` call redraws every visible spectrum's fits via `draw_committed_fits`, which reads `.visible` on every entry first, before anything else -- including `spectrum_b`'s un-rescaled (and therefore un-cleared) fake fit. A bare string has no `.visible` attribute. The `_FakeFit` class above (added during Task 11's implementation) fixes this without weakening the test -- it still verifies exactly the same thing (selective clearing based on rescale factor), just with a placeholder shaped enough not to crash the unrelated redraw side-effect.

```python
def test_normalize_resets_in_progress_marks(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    _make_active_spectrum(main_window)
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert main_window.fit_controller.state.pending_fit_click is None


def test_normalize_ignores_hidden_spectra(qapp):
    main_window = MainWindow()
    spectrum_a = _make_active_spectrum(main_window)
    spectrum_b = _make_active_spectrum(main_window)
    spectrum_c = _make_active_spectrum(main_window)
    spectrum_a.data = np.array([10, 20], dtype=np.int64)
    spectrum_b.data = np.array([5, 40], dtype=np.int64)
    spectrum_c.data = np.array([1, 1000], dtype=np.int64)
    spectrum_c.visible = False
    main_window.fit_controller.state.pending_fit_click = 1

    main_window._normalize_spectra()

    assert list(spectrum_c.data) == [1, 1000]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_operations_menu.py -k normalize -v`
Expected: FAIL with `AttributeError: 'MainWindow' object has no attribute 'normalize_action'`

- [ ] **Step 3: Update the import**

In `main_window.py`, change:

```python
from spectrum_operations import multiply, rebin
```

to:

```python
from spectrum_operations import multiply, normalize_factors, rebin, reference_value
```

- [ ] **Step 4: Add the menu action, availability hook, and handler**

In `_build_menu`, right after the `rebin_action` block added in Task 10, add:

```python
        self.normalize_action = QAction("Normalize Spectra", self)
        self.normalize_action.setShortcut("Ctrl+N")
        self.normalize_action.setEnabled(False)
        self.normalize_action.triggered.connect(self._normalize_spectra)
        self.operations_menu.addAction(self.normalize_action)
```

In `_update_operations_availability`, change:

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
```

to:

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
        visible_count = sum(1 for s in self.spectra if s.visible)
        self.normalize_action.setEnabled(visible_count >= 2)
```

Add the handler, right after `_apply_rebin` (added in Task 10):

```python
    def _normalize_spectra(self):
        # Both status messages below use fit_controller._show_status_message
        # (not statusBar() directly) -- it arms _status_message_until, which
        # _on_mouse_move checks before overwriting the status bar with the
        # ordinary hover readout. Without this, either message gets wiped by
        # the very next mouse move over the canvas -- and for the first
        # message in particular, moving the mouse onto the canvas to make a
        # mark is the literal next thing the message tells the user to do.
        # (Same class of bug found and fixed for Rebin's warning in Task 10;
        # applied here proactively rather than waiting to rediscover it.)
        visible = [s for s in self.spectra if s.visible]
        if len(visible) < 2:
            return
        state = self.fit_controller.state
        if state.fit_region is not None:
            region, channel = state.fit_region, None
        elif state.pending_fit_click is not None:
            region, channel = None, state.pending_fit_click
        else:
            self.fit_controller._show_status_message(
                "Mark a channel or region (hold r, click) to normalize against.", 5000
            )
            return

        values = [reference_value(s.data, channel=channel, region=region) for s in visible]
        if all(v == 0 for v in values):
            self.fit_controller._show_status_message(
                "Nothing to normalize -- every visible spectrum reads zero at the marked channel/region.",
                5000,
            )
            self.fit_controller.reset_marks()
            self._plot_data(preserve_view=True)
            return
        factors = normalize_factors(values)

        skipped = []
        for spectrum, factor in zip(visible, factors):
            if factor is None:
                skipped.append(os.path.basename(spectrum.path))
                continue
            if factor == 1.0:
                continue
            spectrum.data = multiply(spectrum.data, factor)
            spectrum.fits.clear()

        self.fit_controller.reset_marks()
        self._plot_data(preserve_view=True)

        if skipped:
            self.fit_controller._show_status_message(
                f"Skipped (zero reference value): {', '.join(skipped)}", 5000
            )
```

`os` is already imported at the top of `main_window.py` (used by `_try_load_spectrum` and others) — no new import needed.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_operations_menu.py -v`
Expected: PASS, all tests

- [ ] **Step 6: Commit**

```bash
git add main_window.py tests/test_operations_menu.py
git commit -m "feat: wire up Normalize Spectra"
```

---

### Task 12: `histogram_io.save_histogram()`

**Files:**
- Modify: `histogram_io.py`
- Test: `tests/test_histogram_io.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_histogram_io.py`:

```python
def test_save_histogram_round_trips_through_load_histogram(tmp_path):
    data = np.array([4, 0, 1, 0, 1, 7, 200])
    path = tmp_path / "out.txt"

    save_histogram(str(path), data)
    result = load_histogram(str(path))

    assert list(result[:len(data)]) == list(data)


def test_save_histogram_writes_one_value_per_line(tmp_path):
    data = [1, 2, 3]
    path = tmp_path / "out.txt"

    save_histogram(str(path), data)

    lines = path.read_text().splitlines()
    assert lines == ["1", "2", "3"]
```

Add `import numpy as np` at the top of `tests/test_histogram_io.py` if not already present, and update the import line to `from histogram_io import ParseError, load_histogram, save_histogram`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_histogram_io.py -k save_histogram -v`
Expected: FAIL with `ImportError: cannot import name 'save_histogram'`

- [ ] **Step 3: Implement `save_histogram`**

Append to `histogram_io.py`:

```python


def save_histogram(path: str, data) -> None:
    """Writes `data` as one integer count per line -- the exact inverse
    of load_histogram (which ignores blank lines and pads to the next
    bucket size on read; re-loading a saved file naturally re-pads)."""
    with open(path, "w") as f:
        for value in data:
            f.write(f"{int(value)}\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_histogram_io.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add histogram_io.py tests/test_histogram_io.py
git commit -m "feat: add save_histogram writer for .txt spectra"
```

---

### Task 13: `spe_io.save_spe()`

**Files:**
- Modify: `spe_io.py`
- Test: `tests/test_spe_io.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spe_io.py`:

```python
def test_save_spe_round_trips_through_load_spe(tmp_path):
    data = np.array([4, 0, 1, 0, 1, 7, 200], dtype=np.int64)
    path = tmp_path / "out.spe"

    save_spe(str(path), data)
    result = load_spe(str(path))

    assert list(result) == list(data)


def test_save_spe_round_trips_large_values(tmp_path):
    data = np.array([0, 1000000, 41580], dtype=np.int64)
    path = tmp_path / "out.spe"

    save_spe(str(path), data)
    result = load_spe(str(path))

    assert list(result) == list(data)


def test_save_spe_round_trips_all_zero_spectrum(tmp_path):
    data = np.zeros(10, dtype=np.int64)
    path = tmp_path / "out.spe"

    save_spe(str(path), data)
    result = load_spe(str(path))

    assert list(result) == [0] * 10
```

Add `import numpy as np` at the top of `tests/test_spe_io.py` if not already present, and update the import line to `from spe_io import load_spe, save_spe`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spe_io.py -k save_spe -v`
Expected: FAIL with `ImportError: cannot import name 'save_spe'`

- [ ] **Step 3: Implement `save_spe`**

Append to `spe_io.py`:

```python


def save_spe(path: str, data) -> None:
    """Writes `data` as a little-endian Fortran-unformatted-record .spe
    file -- the exact inverse of load_spe's parsing. Not a native TV
    format (see design spec); idim2/ired1/ired2 are placeholder values,
    matched by load_spe's own disregard of them (name is also never
    read back)."""
    idim1 = len(data)
    record1_payload = struct.pack("<8s4i", b"SPECTRUM", idim1, 1, 0, 0)
    record1 = (
        struct.pack("<i", RECORD1_PAYLOAD_SIZE) + record1_payload
        + struct.pack("<i", RECORD1_PAYLOAD_SIZE)
    )

    payload_size = idim1 * 4
    values = [float(v) for v in data]
    record2_payload = struct.pack(f"<{idim1}f", *values)
    record2 = (
        struct.pack("<i", payload_size) + record2_payload + struct.pack("<i", payload_size)
    )

    with open(path, "wb") as f:
        f.write(record1)
        f.write(record2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spe_io.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add spe_io.py tests/test_spe_io.py
git commit -m "feat: add save_spe writer for .spe spectra"
```

---

### Task 14: `spk_io.py` — zigzag encode + tag-packing primitives

**Files:**
- Modify: `spk_io.py`
- Test: `tests/test_spk_io.py`

These are the byte-level building blocks for `_lc2_compress` (Task 15), ported verbatim from `libmfile-1.0.7/src/lc_c2.c`'s `encode`/`put_tag_n`/`fitsinto` macros. Each primitive here is independently verified against real encoded bytes already exercised by this repo's existing decoder tests (`test_lc2_extended_single_value_tag` etc. in this same file).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spk_io.py`:

```python
def test_zigzag_encode_matches_decode_for_various_values():
    for value in [-1000, -3, -2, -1, 0, 1, 2, 3, 1000, 158]:
        assert _zigzag_decode(_zigzag_encode(value)) == value


def test_zigzag_encode_known_values():
    assert _zigzag_encode(0) == 0
    assert _zigzag_encode(-1) == 1
    assert _zigzag_encode(1) == 2
    assert _zigzag_encode(-2) == 3
    assert _zigzag_encode(158) == 316


def test_put_tag_n_single_byte_for_small_values():
    assert _put_tag_n(0x80, 5) == bytes([0x85])
    assert _put_tag_n(0x80, 0) == bytes([0x80])
    assert _put_tag_n(0x80, 59) == bytes([0xBB])


def test_put_tag_n_extended_encoding_matches_known_fixture():
    # Matches test_lc2_extended_single_value_tag above: the zigzag code
    # for 158 is 316, which must encode as tag 0xBD + [0x00, 0x00].
    assert _put_tag_n(0x80, 316) == bytes([0xBD, 0x00, 0x00])


def test_put_tag_n_same_diff_base():
    assert _put_tag_n(0xC0, 7) == bytes([0xC7])
```

Update the import line at the top of `tests/test_spk_io.py` to:

```python
from spk_io import (
    LC_HEADER_SIZE, LC_MAGIC, MAT_COLMAX, _put_tag_n, _zigzag_decode, _zigzag_encode, load_spk,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spk_io.py -k "zigzag_encode or put_tag_n" -v`
Expected: FAIL with `ImportError: cannot import name '_zigzag_encode'`

- [ ] **Step 3: Implement the primitives**

In `spk_io.py`, add these right after `_zigzag_decode` (currently lines 44-47):

```python
def _zigzag_encode(value: int) -> int:
    """Inverse of _zigzag_decode above. value can be any integer
    (positive, negative, or zero)."""
    return 2 * value if value >= 0 else -2 * value - 1


def _put_tag_n(tag_base: int, value: int) -> bytes:
    """Ported from libmfile's put_tag_n macro (lc_c2.c:39-53): encodes a
    non-negative `value` as a single tag byte (tag_base + value) when
    value <= 59, or a tag byte (tag_base + 60 + extra_byte_count) plus
    1-4 little-endian-ish extension bytes for larger values. Shared by
    single-value tags (tag_base=0x80) and same-run tags (tag_base=0xC0)."""
    if value <= 59:
        return bytes([tag_base + value])
    t = value - 60
    extension = [t & 0xFF]
    extra = 0
    t >>= 8
    while t:
        t -= 1
        extension.append(t & 0xFF)
        extra += 1
        t >>= 8
    return bytes([tag_base + 60 + extra]) + bytes(extension)
```

**Post-review fix (discovered during Task 15's code-quality review):** the C reference relies on 32-bit `int` wraparound to implicitly bound how large a value `put_tag_n` ever has to encode; Python ints don't wrap, so nothing bounded `extra` here. Past `extra == 3` (5 total extension bytes), the tag byte `tag_base + 60 + extra` walks outside the format's valid `[60, 63]` low-6-bits range, corrupting the output silently (or crashing confusingly). Fixed with a guard that raises `ValueError` before the invalid 5th extension byte would ever be appended:

```python
    while t:
        if extra == 3:
            raise ValueError(
                f"value {value} is too large to encode in LC2's tag format "
                "(supports at most 4 extension bytes)"
            )
        t -= 1
        extension.append(t & 0xFF)
        extra += 1
        t >>= 8
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spk_io.py -k "zigzag_encode or put_tag_n" -v`
Expected: PASS (5 tests)

Run the full spk test file to make sure the import-line change didn't break anything:

Run: `python -m pytest tests/test_spk_io.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add spk_io.py tests/test_spk_io.py
git commit -m "feat: add zigzag_encode and put_tag_n primitives for LC2 compression"
```

---

### Task 15: `spk_io.py` — `_lc2_compress()`

**Files:**
- Modify: `spk_io.py`
- Test: `tests/test_spk_io.py`

Ported verbatim from `libmfile-1.0.7/src/lc_c2.c`'s `lc2_compress` (lines 63-126). Verified by hand against the file's own known test vectors before writing this task: `[0, -1, 1]` -> `0x24` (3-value tag), `[-2, 1]` -> `0x53` (2-value tag), `[-3]` -> `0x85` (1-value tag), `[1, 0, 0, 0, 0, 0, 0]` -> `0xC7` (same-run tag), `[158]` -> `0xBD 0x00 0x00` (extended 1-value tag) — each matching this repo's existing decoder tests exactly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spk_io.py`:

```python
def test_lc2_compress_3value_tag():
    assert _lc2_compress([0, -1, 1]) == bytes([0x24])


def test_lc2_compress_2value_tag():
    assert _lc2_compress([-2, 1]) == bytes([0x53])


def test_lc2_compress_1value_tag():
    assert _lc2_compress([-3]) == bytes([0x85])


def test_lc2_compress_same_run_tag():
    assert _lc2_compress([1, 0, 0, 0, 0, 0, 0]) == bytes([0xC7])


def test_lc2_compress_same_run_boundary_exactly_4_uses_run_tag():
    assert _lc2_compress([0, 0, 0, 0]) == bytes([0xC0])


def test_lc2_compress_same_run_boundary_exactly_3_falls_through_to_pack():
    assert _lc2_compress([0, 0, 0]) == bytes([0x00])


def test_lc2_compress_extended_single_value_tag():
    assert _lc2_compress([158]) == bytes([0xBD, 0x00, 0x00])


def test_lc2_compress_empty_input():
    assert _lc2_compress([]) == b""
```

**Post-review additions (discovered during Task 15's code-quality review):** the `same == 3` boundary (the exact line between "falls through to normal value-packing" and "uses the same-run tag") had zero test coverage, and a plausible one-character regression there (`if same > 3:` instead of `if same >= 3:`) was proven to silently corrupt output that still round-trips correctly through the unmodified decoder — undetected by every other test, since `demo.spk`'s real data happens not to exercise `same == 3` at all. The two boundary tests above close that gap. Separately, `_put_tag_n`'s new overflow guard (see Task 14's "Post-review fix" note) needs its own boundary tests too:

```python
def test_lc2_compress_raises_on_a_value_too_large_to_encode():
    with pytest.raises(ValueError):
        _lc2_compress([2 ** 40])


def test_lc2_compress_recovers_wrong_data_no_longer_corrupts_silently():
    # Exact repro from the review: previously compressed and decompressed
    # without error but produced silently wrong decoded values.
    with pytest.raises(ValueError):
        _lc2_compress([2200000000, 5, 6, 7, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 20, 30])
```

(`pytest` must already be imported at the top of `tests/test_spk_io.py` for the existing `pytest.raises` usages in that file's oldmat/LC1 rejection tests -- confirm before adding, add `import pytest` if genuinely missing.)

```python
def test_lc2_compress_round_trips_through_uncompress_for_varied_data():
    samples = [
        [0] * 20,
        list(range(50)),
        [5, 5, 5, 5, 5, 5, 5, 5, -100, 200, 0, 0, 0],
        [1000, -1000, 500, -500, 0, 0, 0, 0, 0, 1],
        [7],
    ]
    for values in samples:
        compressed = _lc2_compress(values)
        decoded = _lc2_uncompress(compressed, len(values), "test")
        assert decoded == values


def test_lc2_compress_recompresses_demo_spk_to_the_identical_bytes():
    # The strongest possible check: decode demo.spk's real compressed
    # payload, recompress the decoded values, and confirm the output
    # matches the original file's bytes exactly -- not just that it
    # round-trips through our own decoder.
    with open(FIXTURES / "demo.spk", "rb") as f:
        raw = f.read()
    fields = struct.unpack_from("<11I", raw, 0)
    _magic, _version, _levels, _lines, columns, poslentablepos = fields[:6]
    pos, length = struct.unpack_from("<2I", raw, poslentablepos)
    original_compressed = raw[pos:pos + length]

    decoded = _lc2_uncompress(original_compressed, columns, "demo.spk")
    recompressed = _lc2_compress(decoded)

    assert recompressed == original_compressed
```

Update the import line to add `_lc2_compress` and `_lc2_uncompress`:

```python
from spk_io import (
    LC_HEADER_SIZE, LC_MAGIC, MAT_COLMAX, _lc2_compress, _lc2_uncompress, _put_tag_n,
    _zigzag_decode, _zigzag_encode, load_spk,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spk_io.py -k lc2_compress -v`
Expected: FAIL with `ImportError: cannot import name '_lc2_compress'`

- [ ] **Step 3: Implement `_lc2_compress`**

In `spk_io.py`, add this right after `_put_tag_n` (added in Task 14):

```python
def _fits_in_bits(value: int, bits: int) -> bool:
    return (value >> bits) == 0


def _lc2_compress(values) -> bytes:
    """Ported from libmfile's lc2_compress (lc_c2.c:63-126). `values` is
    a sequence of ints (one spectrum's channel counts). Returns the
    LC2-compressed bytes, choosing the most compact of: a run of the
    previous value (>=4 elements), a 3-value pack (2 bits each,
    anchor-relative to the same running `last`), a 2-value pack (3 bits
    each), or a single value (with extension bytes for large deltas)."""
    out = bytearray()
    last = 0
    n = len(values)
    idx = 0

    while idx < n:
        remaining = n - idx
        d = values[idx] - last
        i = 1
        if 0 <= d < 2:
            while i < remaining and values[idx + i] == last:
                i += 1
        same = i - 1

        if same >= 3:
            out += _put_tag_n(0xC0, ((same - 3) << 1) + d)
            idx += i
            continue

        s0 = values[idx]
        a = _zigzag_encode(s0 - last)

        if _fits_in_bits(a, 3) and remaining >= 2:
            s1 = values[idx + 1]
            b = _zigzag_encode(s1 - last)

            if _fits_in_bits(a | b, 2) and remaining >= 3:
                s2 = values[idx + 2]
                c = _zigzag_encode(s2 - last)
                if _fits_in_bits(c, 2):
                    out.append(a + (b << 2) + (c << 4))
                    idx += 3
                    last = s2
                    continue

            if _fits_in_bits(b, 3):
                out.append(0x40 + a + (b << 3))
                idx += 2
                last = s1
                continue

        out += _put_tag_n(0x80, a)
        idx += 1
        last = s0

    return bytes(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spk_io.py -k lc2_compress -v`
Expected: PASS (8 tests)

Run the full spk test file:

Run: `python -m pytest tests/test_spk_io.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add spk_io.py tests/test_spk_io.py
git commit -m "feat: port lc2_compress from libmfile for .spk writing"
```

---

### Task 16: `spk_io.save_spk()`

**Files:**
- Modify: `spk_io.py`
- Test: `tests/test_spk_io.py`

Container layout ported from `libmfile-1.0.7/src/lc_minfo.c`'s `init_lci`/`lc_flush` and `lc_getput.c`'s `writeline`, for the specific case this app only ever needs: a brand-new, single-spectrum (levels=1, lines=1), LC2-compressed file. Verified by hand-tracing those functions: for a fresh file, `poslentablepos = sizeof(lc_header) = 44`, the single poslentable entry is written at `(pos=44+8=52, len=<compressed length>)`, `freepos` ends up at `52 + <compressed length>`, and `used`/`free`/`status` are always written as 0 (`lc_minfo.c:204-206`, "not yet implemented").

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spk_io.py`:

```python
def test_save_spk_round_trips_through_load_spk(tmp_path):
    data = np.array([0, 0, 1, 5, 100, 0, 41580, 2])
    path = tmp_path / "out.spk"

    save_spk(str(path), data)
    result = load_spk(str(path))

    assert list(result) == list(data)


def test_save_spk_round_trips_the_real_demo_spk_data(tmp_path):
    original = load_spk(str(FIXTURES / "demo.spk"))
    path = tmp_path / "roundtrip.spk"

    save_spk(str(path), original)
    result = load_spk(str(path))

    assert list(result) == list(original)


def test_save_spk_writes_version_2_header(tmp_path):
    data = np.array([1, 2, 3])
    path = tmp_path / "out.spk"

    save_spk(str(path), data)

    with open(path, "rb") as f:
        raw = f.read()
    fields = struct.unpack_from("<11I", raw, 0)
    magic, version, levels, lines, columns, poslentablepos = fields[:6]
    assert magic == LC_MAGIC
    assert version == 2
    assert levels == 1
    assert lines == 1
    assert columns == 3
    assert poslentablepos == LC_HEADER_SIZE


def test_save_spk_round_trips_all_zero_spectrum(tmp_path):
    data = np.zeros(20, dtype=np.int64)
    path = tmp_path / "out.spk"

    save_spk(str(path), data)
    result = load_spk(str(path))

    assert list(result) == [0] * 20


def test_save_spk_round_trips_negative_values(tmp_path):
    # Real spectra are non-negative, but the codec itself is signed --
    # this exercises that the round-trip holds regardless.
    data = np.array([0, 1000000, 41580, -5])
    path = tmp_path / "out.spk"

    save_spk(str(path), data)
    result = load_spk(str(path))

    assert list(result) == list(data)


def test_save_spk_rejects_a_channel_count_too_large_for_the_format():
    with pytest.raises(ValueError):
        save_spk("unused.spk", [0] * (MAT_COLMAX + 1))
```

Update the import line to add `save_spk`:

```python
from spk_io import (
    LC_HEADER_SIZE, LC_MAGIC, MAT_COLMAX, _lc2_compress, _lc2_uncompress, _put_tag_n,
    _zigzag_decode, _zigzag_encode, load_spk, save_spk,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spk_io.py -k save_spk -v`
Expected: FAIL with `ImportError: cannot import name 'save_spk'`

- [ ] **Step 3: Implement `save_spk`**

In `spk_io.py`, add this at the very end of the file:

```python


def save_spk(path: str, data) -> None:
    """Writes `data` as a single-spectrum, LC2-compressed .spk file --
    the modern MAT_LC format, version 2, the only writable .spk variant
    (see design spec). Layout ported from libmfile's own new-file
    behavior (lc_minfo.c's init_lci/lc_flush, lc_getput.c's writeline):
    a 44-byte header, one 8-byte position/length table entry right
    after it, then the LC2-compressed payload."""
    values = [int(v) for v in data]
    columns = len(values)
    if not (1 <= columns <= MAT_COLMAX):
        raise ValueError(
            f"Channel count {columns} is out of range for a .spk file (must be 1-{MAT_COLMAX})"
        )
    compressed = _lc2_compress(values)

    poslentablepos = LC_HEADER_SIZE
    data_pos = poslentablepos + LC_POSLEN_SIZE
    freepos = data_pos + len(compressed)

    header = struct.pack(
        "<11I", LC_MAGIC, 2, 1, 1, columns, poslentablepos, freepos, 0, 0, 0, 0,
    )
    poslen = struct.pack("<2I", data_pos, len(compressed))

    with open(path, "wb") as f:
        f.write(header)
        f.write(poslen)
        f.write(compressed)
```

**Post-review addition (discovered during Task 16's code-quality review):** without this check, `save_spk` could write a file with `columns > MAT_COLMAX` (confirmed empirically: writes cleanly, no error) that this app's own `load_spk` would then permanently refuse to open (`_load_lc` already enforces the same `1..MAT_COLMAX` range on read). `_lc2_compress`'s own overflow guard (Task 15) protects against a different failure mode (individual delta magnitudes), not channel count -- this is a separate bound. `test_save_spk_rejects_a_channel_count_too_large_for_the_format` pins it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spk_io.py -v`
Expected: PASS, all tests

- [ ] **Step 5: Commit**

```bash
git add spk_io.py tests/test_spk_io.py
git commit -m "feat: add save_spk writer for .spk spectra"
```

---

### Task 17: Save Spectrum dialog wiring

**Files:**
- Modify: `main_window.py`
- Test: `tests/test_operations_menu.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_operations_menu.py`:

```python
def test_save_spectrum_action_disabled_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    assert main_window.save_spectrum_action.isEnabled() is False


def test_save_spectrum_action_enabled_with_active_spectrum(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    assert main_window.save_spectrum_action.isEnabled() is True


def test_save_spectrum_action_has_shortcut(qapp):
    main_window = MainWindow()
    assert main_window.save_spectrum_action.shortcut() == QKeySequence("Ctrl+S")


def test_save_spectrum_action_in_file_menu_before_recent_files(qapp):
    main_window = MainWindow()
    actions = main_window.file_menu.actions()
    assert main_window.save_spectrum_action in actions
    save_index = actions.index(main_window.save_spectrum_action)
    recent_index = actions.index(main_window.recent_menu.menuAction())
    assert save_index < recent_index


def test_write_spectrum_extension_takes_priority_over_chosen_filter(qapp, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    txt_path = tmp_path / "out.txt"
    main_window._write_spectrum(spectrum, str(txt_path), "SPE files (*.spe)")

    from histogram_io import load_histogram
    assert list(load_histogram(str(txt_path))[:len(spectrum.data)]) == list(spectrum.data)


def test_write_spectrum_falls_back_to_the_chosen_filter_with_no_recognized_extension(qapp, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    path = tmp_path / "out"
    main_window._write_spectrum(spectrum, str(path), "SPK files (*.spk)")

    from spk_io import load_spk
    assert list(load_spk(str(path))) == list(spectrum.data)


def test_write_spectrum_defaults_to_text_with_no_extension_or_recognized_filter(qapp, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    path = tmp_path / "out"
    main_window._write_spectrum(spectrum, str(path), "All files (*)")

    from histogram_io import load_histogram
    assert list(load_histogram(str(path))[:len(spectrum.data)]) == list(spectrum.data)


def test_open_save_spectrum_dialog_writes_the_chosen_file(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    target = tmp_path / "chosen.spk"

    monkeypatch.setattr(
        "main_window.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(target), "SPK files (*.spk)"),
    )
    main_window._open_save_spectrum_dialog()

    from spk_io import load_spk
    assert list(load_spk(str(target))) == list(spectrum.data)


def test_open_save_spectrum_dialog_does_nothing_when_cancelled(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    monkeypatch.setattr("main_window.QFileDialog.getSaveFileName", lambda *a, **k: ("", ""))
    main_window._open_save_spectrum_dialog()  # must not raise


def test_open_save_spectrum_dialog_does_nothing_with_no_active_spectrum(qapp):
    main_window = MainWindow()
    main_window._open_save_spectrum_dialog()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_operations_menu.py -k save_spectrum -v`
Expected: FAIL with `AttributeError: 'MainWindow' object has no attribute 'save_spectrum_action'`

- [ ] **Step 3: Update the import**

In `main_window.py`, change:

```python
from histogram_io import ParseError, load_histogram
```

to:

```python
from histogram_io import ParseError, load_histogram, save_histogram
```

Change:

```python
from spe_io import load_spe
```

to:

```python
from spe_io import load_spe, save_spe
```

Change:

```python
from spk_io import load_spk
```

to:

```python
from spk_io import load_spk, save_spk
```

- [ ] **Step 4: Add the menu action, availability hook, and handlers**

In `_build_menu`, insert the Save Spectrum action between Open and Recent Files. Change:

```python
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file_dialog)
        self.file_menu.addAction(open_action)

        self.recent_menu = self.file_menu.addMenu("Recent Files")
```

to:

```python
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file_dialog)
        self.file_menu.addAction(open_action)

        self.save_spectrum_action = QAction("Save Spectrum...", self)
        self.save_spectrum_action.setShortcut("Ctrl+S")
        self.save_spectrum_action.setEnabled(False)
        self.save_spectrum_action.triggered.connect(self._open_save_spectrum_dialog)
        self.file_menu.addAction(self.save_spectrum_action)

        self.recent_menu = self.file_menu.addMenu("Recent Files")
```

In `_update_operations_availability`, change:

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
        visible_count = sum(1 for s in self.spectra if s.visible)
        self.normalize_action.setEnabled(visible_count >= 2)
```

to:

```python
    def _update_operations_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        self.multiply_action.setEnabled(active is not None)
        self.rebin_action.setEnabled(active is not None)
        self.save_spectrum_action.setEnabled(active is not None)
        visible_count = sum(1 for s in self.spectra if s.visible)
        self.normalize_action.setEnabled(visible_count >= 2)
```

Add the dialog wrapper and testable core, right after `_normalize_spectra` (added in Task 11):

```python
    def _open_save_spectrum_dialog(self):
        active = next((s for s in self.spectra if s.active), None)
        if active is None:
            return
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "Save Spectrum", os.path.dirname(active.path),
            "SPE files (*.spe);;SPK files (*.spk);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        self._write_spectrum(active, path, chosen_filter)

    def _write_spectrum(self, spectrum, path, chosen_filter):
        lower = path.lower()
        if lower.endswith(".spe"):
            writer = save_spe
        elif lower.endswith(".spk"):
            writer = save_spk
        elif lower.endswith(".txt"):
            writer = save_histogram
        elif chosen_filter.startswith("SPE"):
            writer = save_spe
        elif chosen_filter.startswith("SPK"):
            writer = save_spk
        else:
            writer = save_histogram
        try:
            writer(path, spectrum.data)
        except OSError as exc:
            QMessageBox.warning(self, "Save Spectrum", f"Could not save: {exc}")
        except ValueError as exc:
            QMessageBox.warning(
                self, "Save Spectrum",
                f"Could not save in this format: {exc}\n\n"
                "Try a different format (e.g. Text), or Multiply by a smaller factor first.",
            )
```

**Post-review addition (discovered during Task 16's code-quality review, applied here proactively before this task was ever dispatched):** `save_spk` (via `_lc2_compress`, Task 15's overflow guard) can raise `ValueError` for channel-to-channel deltas outside LC2's encodable range, and `save_spk` itself now also raises `ValueError` for an out-of-range channel count (Task 16's post-review fix). Catching only `OSError` would let either of those escape as an unhandled exception inside a Qt slot instead of a clean warning dialog.

**Second post-review addition (discovered during Task 17's own code-quality review):** a single combined `except (OSError, ValueError)` catches both failure modes, but `_lc2_compress`'s raw message ("value N is too large to encode in LC2's tag format (supports at most 4 extension bytes)") is internal compression-scheme jargon, confusing to an end user, and realistically reachable -- `FactorDialog`'s Multiply validator only checks `v > 0` (no upper bound), so an extreme-but-"valid" Multiply factor can genuinely drive channel deltas past this limit before a Save attempt. Split into two `except` clauses so `ValueError` (a data/input problem) gets an actionable suggestion, while `OSError` (a filesystem problem) keeps its original message. Three tests cover this area, all added post-review: `test_write_spectrum_spe_extension_dispatches_to_save_spe` and `test_write_spectrum_spe_filter_fallback_dispatches_to_save_spe` (the `.spe` dispatch branches had no coverage at all before this), `test_write_spectrum_shows_a_warning_instead_of_crashing_on_failure` (the except branch itself had no coverage), and one pinning the friendlier `ValueError` wording specifically.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_operations_menu.py -v`
Expected: PASS, all tests

- [ ] **Step 6: Run the entire test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS, every test in the repository

- [ ] **Step 7: Commit**

```bash
git add main_window.py tests/test_operations_menu.py
git commit -m "feat: wire up Save Spectrum with format-choice dialog"
```

---

## Final check

After Task 17, do a manual smoke test before considering the feature done:

1. Launch the app: `python main.py` (or whatever the existing entry point is — check `README.md`/existing launch docs if unsure).
2. Load a spectrum (`.txt`, `.spe`, or `.spk`).
3. Open the **Operations** menu — confirm Calibration, Toggle Calibration Active, Multiply by Factor, Rebin by Factor, and Normalize Spectra are all present with their shortcuts shown.
4. Try `Ctrl+M` (Multiply), enter a factor, confirm the plot rescales.
5. Try `Ctrl+R` (Rebin), enter a factor, confirm the channel count drops and the plot redraws full-width.
6. Mark a single channel (hold `r`, click once) with two spectra loaded, press `Ctrl+N`, confirm the smaller one scales up.
7. Press `Ctrl+S`, save as each of `.txt`/`.spe`/`.spk`, then re-open each saved file and confirm the data matches.
8. Press `Ctrl+C` after committing a fit — confirm the fit disappears from the Fit Results table entirely (not just grayed out).
9. Confirm every shortcut in the design spec's table works via keyboard alone.
