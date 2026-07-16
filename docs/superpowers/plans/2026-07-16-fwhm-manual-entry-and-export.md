# FWHM Parameter, Manual Value Entry, and Disk Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show FWHM (not sigma) as the fixable width parameter in the Fit Parameters panel, let every row's value be typed in by hand (not just fixed rows), and write detailed, timestamped fit results to disk both automatically (a JSON Lines log next to the spectrum file) and on demand (a plain-text report via a save dialog).

**Architecture:** `peak_fit.py` (pure numeric core, no Qt) gains an `initial_guess_overrides` parameter and a `timestamp` field on `FitResult` — no change to its canonical `sigma`-based parameter naming. All FWHM↔sigma conversion lives in `fit_mode.py` at the UI boundary. A new `fit_export.py` module (no Qt dependency) holds pure JSON/text serialization functions; `fit_mode.py` wires it to `run_fit()` (auto-log) and two new Fit Results context-menu actions (explicit export).

**Tech Stack:** Python, PySide6 (Qt widgets), numpy, pytest.

**Design spec:** `docs/superpowers/specs/2026-07-16-fwhm-manual-entry-and-export-design.md`

---

### Task 1: `peak_fit.py` — initial guess overrides and a timestamp field

**Files:**
- Modify: `peak_fit.py:76-90` (`FitResult` dataclass), `peak_fit.py:169-188` (`_initial_guess`), `peak_fit.py:217-220` (`fit_peaks` signature), `peak_fit.py:260-262` (`_initial_guess` call site)
- Test: `tests/test_peak_fit.py`

- [ ] **Step 1: Write the failing tests**

Add this test right after `test_fit_result_visible_defaults_to_true` (ends at `tests/test_peak_fit.py:160`), before `test_fit_error_is_an_exception`:

```python
def test_fit_result_timestamp_defaults_to_none():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    result = FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[peak],
    )
    assert result.timestamp is None
```

Add these tests right after `test_fit_with_all_parameters_fixed_skips_optimization` (ends at `tests/test_peak_fit.py:507`), before `test_fit_recovers_from_a_deliberately_bad_initial_width_guess`:

```python
from peak_fit import _initial_guess


def test_initial_guess_overrides_replace_only_the_named_parameters():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    lo, hi = 85.0, 115.0
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_sub = y[mask] - 20.0
    free_names = ["amp_0", "pos_0", "sigma"]

    p0_default = _initial_guess(
        free_names, x_fit, y_sub, (lo, hi), [100.0], link_widths=True, enable_left_tail=False,
    )
    p0_overridden = _initial_guess(
        free_names, x_fit, y_sub, (lo, hi), [100.0], link_widths=True, enable_left_tail=False,
        initial_guess_overrides={"sigma": 7.5},
    )

    assert p0_overridden[free_names.index("sigma")] == 7.5
    assert p0_overridden[free_names.index("amp_0")] == p0_default[free_names.index("amp_0")]
    assert p0_overridden[free_names.index("pos_0")] == p0_default[free_names.index("pos_0")]


def test_fit_peaks_accepts_initial_guess_overrides_without_disturbing_a_fixed_value():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
        fixed_params={"pos_0": 100.0},
        initial_guess_overrides={"pos_0": 999.0, "sigma": 6.0},
    )
    peak = result.peaks[0]
    # pos_0 is fixed, so its override is never consulted -- the fixed
    # value wins.
    assert peak.position == 100.0
    # sigma is free and started far (6.0) from its true value (3.0) via
    # the override -- it must still converge to the true value, proving
    # the override is only a *starting point*, not a held constant.
    assert peak.sigma == pytest.approx(3.0, rel=0.2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_peak_fit.py -k "timestamp_defaults_to_none or initial_guess_overrides or accepts_initial_guess_overrides" -v`
Expected: FAIL — `test_fit_result_timestamp_defaults_to_none` with `AttributeError: 'FitResult' object has no attribute 'timestamp'`; the other two with `TypeError: _initial_guess() got an unexpected keyword argument 'initial_guess_overrides'` / `fit_peaks() got an unexpected keyword argument 'initial_guess_overrides'`.

- [ ] **Step 3: Implement the changes**

In `peak_fit.py`, add `timestamp` to `FitResult`:

```python
@dataclass
class FitResult:
    left_bg_region: tuple
    right_bg_region: tuple
    fit_region: tuple
    background_slope: float
    background_intercept: float
    peaks: list
    link_widths: bool = True
    tail_fraction: float = None
    tail_fraction_err: float = None
    tail_beta: float = None
    tail_beta_err: float = None
    fixed_params: dict = field(default_factory=dict)
    visible: bool = True
    timestamp: str = None
```

Replace `_initial_guess`:

```python
def _initial_guess(
    free_names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail,
    initial_guess_overrides=None,
):
    lo, hi = fit_region
    region_width = hi - lo
    n_peaks = len(peak_positions)
    fallback_sigma0 = max(region_width / (4 * n_peaks), 1e-6)
    sigma0 = _measure_width(x_fit, y_sub, peak_positions, fallback_sigma0)

    guess_by_name = {}
    for i, pos in enumerate(peak_positions):
        idx = int(np.argmin(np.abs(x_fit - pos)))
        guess_by_name[f"amp_{i}"] = float(y_sub[idx])
        guess_by_name[f"pos_{i}"] = float(pos)
        if not link_widths:
            guess_by_name[f"sigma_{i}"] = sigma0
    if link_widths:
        guess_by_name["sigma"] = sigma0
    if enable_left_tail:
        guess_by_name["tail_fraction"] = 0.05
        guess_by_name["tail_beta"] = max(sigma0, TAIL_BETA_MIN)
    if initial_guess_overrides:
        guess_by_name.update(initial_guess_overrides)
    return [guess_by_name[name] for name in free_names]
```

Update `fit_peaks`'s signature:

```python
def fit_peaks(
    x, y, left_bg_region, right_bg_region, fit_region, peak_positions,
    link_widths=True, enable_left_tail=False, fixed_params=None,
    initial_guess_overrides=None,
):
```

Update its `_initial_guess` call site (inside the `else:` branch, currently reading `p0 = _initial_guess(free_names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail)`):

```python
        p0 = _initial_guess(
            free_names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail,
            initial_guess_overrides=initial_guess_overrides,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_peak_fit.py -v`
Expected: PASS (all tests, including the full pre-existing suite — this change is purely additive with a `None`-defaulting new parameter).

- [ ] **Step 5: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "feat: accept an initial-guess override per free parameter, add FitResult.timestamp"
```

---

### Task 2: Fit Parameters panel — FWHM display and manual entry for every row

**Files:**
- Modify: `fit_mode.py:1-14` (imports), `fit_mode.py:142-154` (`_parameter_label`), `fit_mode.py:451-510` (`update_parameters_panel`, `_on_fix_toggled`, `fixed_params_from_panel`), `fit_mode.py:603-654` (`run_fit`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing tests**

Add `FWHM_FACTOR` to the existing import line at `tests/test_fit_mode_ui.py:10`:

```python
from peak_fit import FWHM_FACTOR, FitResult, PeakResult, hypermet_left_tail
```

Update the label assertion in `test_parameters_panel_populates_after_a_fit` (`tests/test_fit_mode_ui.py:437`):

```python
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Shared FWHM"]
```

Update the label assertion in `test_parameters_panel_rebuilds_when_row_set_changes` (`tests/test_fit_mode_ui.py:461`):

```python
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Peak 1 FWHM"]
```

Replace `test_fixing_a_parameter_in_the_panel_holds_it_for_the_next_fit` (`tests/test_fit_mode_ui.py:538-561`) entirely:

```python
def test_fixing_a_parameter_in_the_panel_holds_it_for_the_next_fit(qapp):
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
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared FWHM"
    table.item(2, 1).setText("7.0")  # FWHM, not sigma -- the panel now displays/accepts FWHM

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    second = spectrum.fits[1]
    assert second.peaks[0].fwhm == pytest.approx(7.0)
    assert second.peaks[0].sigma_err == 0.0
    assert second.fixed_params.keys() == {"sigma"}
    assert second.fixed_params["sigma"] == pytest.approx(7.0 / FWHM_FACTOR)
```

Add these new tests after the (just-replaced) `test_fixing_a_parameter_in_the_panel_holds_it_for_the_next_fit`, before `test_invalid_fixed_value_shows_a_status_message_instead_of_crashing`:

```python
def test_parameters_panel_displays_fwhm_not_sigma(qapp):
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

    table = main_window.fit_controller.parameters_table
    fwhm = spectrum.fits[0].peaks[0].fwhm
    assert table.item(2, 0).text() == "Shared FWHM"
    assert table.item(2, 1).text() == f"{fwhm:.6g}"


def test_parameters_panel_displays_fwhm_per_peak_when_independent(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    main_window.independent_widths_action.setChecked(True)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Peak 1 FWHM"]


def test_every_row_value_cell_is_editable_regardless_of_fix_state(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    for row in range(table.rowCount()):  # every row unchecked at this point
        assert table.item(row, 1).flags() & Qt.ItemFlag.ItemIsEditable

    table.cellWidget(0, 2).setChecked(True)
    assert table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable  # still editable once checked


def test_run_fit_passes_an_edited_unchecked_row_as_an_initial_guess_override(qapp, monkeypatch):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()  # first fit populates the panel

    table = main_window.fit_controller.parameters_table
    table.item(0, 1).setText("777")  # edit "Peak 1 amplitude", left unchecked

    captured = {}
    original_fit_peaks = fit_mode.fit_peaks

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return original_fit_peaks(*args, **kwargs)

    monkeypatch.setattr(fit_mode, "fit_peaks", _spy)
    main_window.fit_controller.run_fit()

    assert captured["initial_guess_overrides"]["amp_0"] == 777.0


def test_run_fit_sets_a_timestamp_on_the_result(qapp):
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

    assert spectrum.fits[0].timestamp is not None
    assert spectrum.fits[0].timestamp[:4].isdigit()  # sane ISO-8601-ish prefix, not an exact match
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fit_mode_ui.py -k "parameters_panel or fixing_a_parameter or editable or initial_guess_override or sets_a_timestamp" -v`
Expected: FAIL — label assertions fail (`AssertionError`, still shows "Shared sigma"/"Peak 1 sigma"); the always-editable test fails (rows are read-only when unchecked); the override-wiring test fails with `KeyError: 'initial_guess_overrides'` (not yet passed to `fit_peaks`); the timestamp test fails with `AssertionError` (`timestamp` is `None`).

- [ ] **Step 3: Implement the changes**

In `fit_mode.py`, update the top-of-file imports (currently lines 1-14):

```python
import sys
import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QMenu, QTableWidget, QTableWidgetItem, QToolBar,
    QVBoxLayout, QWidget,
)

from peak_fit import (
    FWHM_FACTOR, FitError, fit_peaks, fit_result_values_by_name, hypermet_left_tail,
    parameter_names,
)
```

Replace `_parameter_label` (currently `fit_mode.py:142-154`) and add the three new conversion helpers right after it:

```python
def _parameter_label(name):
    """Human-readable row label for a canonical parameter name from
    peak_fit.parameter_names() -- e.g. "amp_0" -> "Peak 1 amplitude".
    Sigma-family names are labeled as FWHM: the panel displays and
    accepts FWHM, converting to/from the fitting engine's internal
    sigma at the UI boundary (see _display_value/_panel_value_to_internal
    below) -- peak_fit.py's own parameter naming and optimizer are
    untouched."""
    if name == "sigma":
        return "Shared FWHM"
    if name == "tail_fraction":
        return "Tail fraction (r)"
    if name == "tail_beta":
        return "Tail beta (β)"
    prefix, index = name.rsplit("_", 1)
    peak_num = int(index) + 1
    kind = {"amp": "amplitude", "pos": "position", "sigma": "FWHM"}[prefix]
    return f"Peak {peak_num} {kind}"


def _is_sigma_name(name):
    return name == "sigma" or name.startswith("sigma_")


def _display_value(name, value):
    """Converts a canonical parameter's internal value to what the Fit
    Parameters panel displays -- sigma-family values are shown as FWHM,
    everything else unchanged."""
    return value * FWHM_FACTOR if _is_sigma_name(name) else value


def _panel_value_to_internal(name, value):
    """Inverse of _display_value -- converts a value read back from the
    panel (FWHM for sigma-family rows) to the sigma-space value
    fit_peaks() expects."""
    return value / FWHM_FACTOR if _is_sigma_name(name) else value
```

Replace `update_parameters_panel`, `_on_fix_toggled`, and `fixed_params_from_panel` (currently `fit_mode.py:451-510`) with:

```python
    def update_parameters_panel(self, names, values_by_name):
        """Rebuilds the Fit Parameters table (resetting every Fix
        checkbox) when the set of parameter names has changed since
        the last fit for these marks; otherwise updates displayed
        values in place, preserving Fix checkbox state and any
        user-edited value. Every row's Value cell is always editable:
        a checked row's edited value is read as a fixed value for the
        next fit (fixed_params_from_panel); an unchecked row's edited
        value is read as that parameter's starting guess for the next
        fit (initial_guess_overrides_from_panel) -- the parameter stays
        free, the optimizer can still move it."""
        if names != self._parameter_names_shown:
            self.parameters_table.setRowCount(0)
            self._parameter_names_shown = list(names)
            for name in names:
                row = self.parameters_table.rowCount()
                self.parameters_table.insertRow(row)

                label_item = QTableWidgetItem(_parameter_label(name))
                label_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.parameters_table.setItem(row, 0, label_item)

                value_item = QTableWidgetItem(f"{_display_value(name, values_by_name[name]):.6g}")
                value_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
                )
                self.parameters_table.setItem(row, 1, value_item)

                fix_checkbox = QCheckBox()
                self.parameters_table.setCellWidget(row, 2, fix_checkbox)
        else:
            for row, name in enumerate(names):
                fix_checkbox = self.parameters_table.cellWidget(row, 2)
                if not fix_checkbox.isChecked():
                    self.parameters_table.item(row, 1).setText(
                        f"{_display_value(name, values_by_name[name]):.6g}"
                    )

    def fixed_params_from_panel(self):
        """Reads the current Fix checkboxes/values from the Fit
        Parameters panel into a {name: value} dict for the next
        fit_peaks() call. Empty when nothing is fixed (including the
        first fit for a fresh set of marks, before the panel has ever
        been populated). A sigma-family row's Value cell holds FWHM;
        this converts it back to sigma before returning. Raises
        FitError if a checked row's Value cell isn't a valid number."""
        fixed = {}
        for row, name in enumerate(self._parameter_names_shown):
            fix_checkbox = self.parameters_table.cellWidget(row, 2)
            if fix_checkbox is not None and fix_checkbox.isChecked():
                text = self.parameters_table.item(row, 1).text()
                try:
                    value = float(text)
                except ValueError:
                    raise FitError(
                        f"Fixed value for '{_parameter_label(name)}' is not a valid number: {text!r}"
                    )
                fixed[name] = _panel_value_to_internal(name, value)
        return fixed

    def initial_guess_overrides_from_panel(self):
        """Mirrors fixed_params_from_panel for unchecked rows: reads
        each unfixed row's current Value cell as the starting guess for
        that parameter in the next fit (the parameter stays free -- the
        optimizer can still move it). Converts a sigma-family row's
        displayed FWHM back to sigma, same as fixed_params_from_panel.
        Raises FitError if an unchecked row's Value cell isn't a valid
        number."""
        overrides = {}
        for row, name in enumerate(self._parameter_names_shown):
            fix_checkbox = self.parameters_table.cellWidget(row, 2)
            if fix_checkbox is not None and not fix_checkbox.isChecked():
                text = self.parameters_table.item(row, 1).text()
                try:
                    value = float(text)
                except ValueError:
                    raise FitError(
                        f"Value for '{_parameter_label(name)}' is not a valid number: {text!r}"
                    )
                overrides[name] = _panel_value_to_internal(name, value)
        return overrides
```

Replace `run_fit` (currently `fit_mode.py:603-654`) with:

```python
    def run_fit(self):
        if not self.state.ready_to_fit():
            return
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        left, right = self.state.ordered_bg_regions()
        x = np.arange(len(active.data), dtype=float)
        y = active.data
        link_widths = not self.main_window.independent_widths_action.isChecked()
        enable_left_tail = self.main_window.left_tail_action.isChecked()
        try:
            # A Fix checkbox (or an edited value) from a since-changed
            # row set (e.g. independent widths or left tail toggled
            # since the last fit) names a parameter that no longer
            # exists under the current configuration -- drop it rather
            # than let fit_peaks() reject the whole fit, since
            # update_parameters_panel() would reset that row anyway
            # once this fit succeeds and rebuilds the row set.
            current_names = set(
                parameter_names(len(self.state.peak_positions), link_widths, enable_left_tail)
            )
            fixed_params = {
                name: value for name, value in self.fixed_params_from_panel().items()
                if name in current_names
            }
            initial_guess_overrides = {
                name: value for name, value in self.initial_guess_overrides_from_panel().items()
                if name in current_names
            }
            result = fit_peaks(
                x, y, left, right, self.state.fit_region, list(self.state.peak_positions),
                link_widths=link_widths, enable_left_tail=enable_left_tail,
                fixed_params=fixed_params, initial_guess_overrides=initial_guess_overrides,
            )
        except FitError as exc:
            self._show_status_message(f"Fit failed: {exc}", 5000)
            return
        result.timestamp = datetime.now().isoformat(timespec="seconds")
        # Re-fitting the exact same marks (e.g. after toggling a
        # checkbox) is a supported workflow, not a mistake -- but
        # drawing every attempt at the identical region on top of the
        # others is just visual clutter. Only the latest attempt at a
        # given region is drawn; every attempt stays listed in Fit
        # Results.
        for earlier in active.fits:
            if (
                earlier.left_bg_region == result.left_bg_region
                and earlier.right_bg_region == result.right_bg_region
                and earlier.fit_region == result.fit_region
            ):
                earlier.visible = False
        active.fits.append(result)
        names = parameter_names(len(result.peaks), result.link_widths, result.tail_fraction is not None)
        self.update_parameters_panel(names, fit_result_values_by_name(result))
        self.main_window._plot_data(preserve_view=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (the full file — this also re-verifies every pre-existing test in it still passes with the renamed labels/always-editable behavior).

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: show FWHM (not sigma) in Fit Parameters panel, make every row editable"
```

---

### Task 3: New `fit_export.py` module

**Files:**
- Create: `fit_export.py`
- Test: `tests/test_fit_export.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fit_export.py`:

```python
import json

from fit_export import (
    append_auto_log, auto_log_path, fit_result_to_json_record,
    fit_result_to_text_report, write_text_report,
)
from peak_fit import FitResult, PeakResult


def _make_result(timestamp="2026-07-16T12:00:00"):
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=7.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=3.0,
        amplitude_err=5.0, sigma_err=0.05,
    )
    return FitResult(
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        background_slope=0.1,
        background_intercept=20.0,
        peaks=[peak],
        fixed_params={"sigma": 3.0},
        timestamp=timestamp,
    )


def test_auto_log_path_derives_from_spectrum_stem(tmp_path):
    spectrum_path = str(tmp_path / "eu.spe")
    assert auto_log_path(spectrum_path) == str(tmp_path / "eu_fits.jsonl")


def test_fit_result_to_json_record_includes_every_field_and_uncertainty():
    result = _make_result()
    record = fit_result_to_json_record(result, "eu.spe")
    assert record["timestamp"] == "2026-07-16T12:00:00"
    assert record["spectrum_path"] == "eu.spe"
    assert record["left_bg_region"] == [70.0, 85.0]
    assert record["right_bg_region"] == [115.0, 130.0]
    assert record["fit_region"] == [85.0, 115.0]
    assert record["background_slope"] == 0.1
    assert record["background_intercept"] == 20.0
    assert record["link_widths"] is True
    assert record["fixed_params"] == {"sigma": 3.0}
    assert record["peaks"] == [
        {
            "position": 100.0, "position_err": 0.1,
            "fwhm": 7.0, "fwhm_err": 0.2,
            "amplitude": 200.0, "amplitude_err": 5.0,
            "sigma": 3.0, "sigma_err": 0.05,
            "area": 1000.0, "area_err": 50.0,
        }
    ]


def test_append_auto_log_writes_one_valid_json_line(tmp_path):
    spectrum_path = str(tmp_path / "eu.spe")
    result = _make_result()

    append_auto_log(spectrum_path, result)

    log_path = auto_log_path(spectrum_path)
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["timestamp"] == "2026-07-16T12:00:00"


def test_append_auto_log_appends_without_disturbing_earlier_lines(tmp_path):
    spectrum_path = str(tmp_path / "eu.spe")
    first = _make_result(timestamp="2026-07-16T12:00:00")
    second = _make_result(timestamp="2026-07-16T12:05:00")

    append_auto_log(spectrum_path, first)
    append_auto_log(spectrum_path, second)

    log_path = auto_log_path(spectrum_path)
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["timestamp"] == "2026-07-16T12:00:00"
    assert json.loads(lines[1])["timestamp"] == "2026-07-16T12:05:00"


def test_fit_result_to_text_report_includes_every_parameter_and_uncertainty():
    result = _make_result()
    report = fit_result_to_text_report(result, "eu.spe", fit_number=1)
    assert "Fit 1" in report
    assert "eu.spe" in report
    assert "2026-07-16T12:00:00" in report
    assert "Position:" in report and "100" in report
    assert "FWHM:" in report and "7" in report
    assert "Amplitude:" in report and "200" in report
    assert "Sigma:" in report
    assert "Area:" in report and "1000" in report
    assert "Fixed parameters: sigma=3" in report


def test_write_text_report_joins_multiple_fit_blocks_in_order(tmp_path):
    results = [(1, _make_result("2026-07-16T12:00:00")), (2, _make_result("2026-07-16T12:05:00"))]
    out_path = tmp_path / "report.txt"

    write_text_report(str(out_path), results, "eu.spe")

    text = out_path.read_text(encoding="utf-8")
    assert text.index("Fit 1") < text.index("Fit 2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fit_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fit_export'`.

- [ ] **Step 3: Implement `fit_export.py`**

Create `fit_export.py`:

```python
import json
import os


def auto_log_path(spectrum_path):
    """Derives the auto-log path from a spectrum's file path -- e.g.
    ".../eu.spe" -> ".../eu_fits.jsonl", next to the spectrum file."""
    directory = os.path.dirname(spectrum_path)
    stem = os.path.splitext(os.path.basename(spectrum_path))[0]
    return os.path.join(directory, f"{stem}_fits.jsonl")


def _peak_record(peak):
    return {
        "position": peak.position, "position_err": peak.position_err,
        "fwhm": peak.fwhm, "fwhm_err": peak.fwhm_err,
        "amplitude": peak.amplitude, "amplitude_err": peak.amplitude_err,
        "sigma": peak.sigma, "sigma_err": peak.sigma_err,
        "area": peak.area, "area_err": peak.area_err,
    }


def fit_result_to_json_record(result, spectrum_path):
    """Converts one FitResult into a plain dict covering every
    parameter and uncertainty, ready for json.dumps() -- used by the
    automatic per-fit log. `spectrum_path` is recorded so a later
    re-read of the log can be traced back to its source spectrum."""
    return {
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
        "peaks": [_peak_record(peak) for peak in result.peaks],
    }


def append_auto_log(spectrum_path, result):
    """Appends one JSON-Lines record for `result` to
    `<spectrum_stem>_fits.jsonl`, next to the spectrum file. Raises
    OSError on failure (e.g. read-only directory) -- the caller must
    turn that into a non-blocking status message, since a failed log
    write must never invalidate an already-successful fit."""
    record = fit_result_to_json_record(result, spectrum_path)
    path = auto_log_path(spectrum_path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _format_err(value, err):
    return f"{value:.6g} ± {err:.6g}"


def fit_result_to_text_report(result, spectrum_path, fit_number=1):
    """Human-readable report for one fit -- one block, used for both a
    single-fit export and (joined by write_text_report) an all-fits
    export. `fit_number` is the fit's 1-based index in the Fit Results
    list, for the block's heading only."""
    lines = [
        f"Fit {fit_number}",
        f"Spectrum: {spectrum_path}",
        f"Timestamp: {result.timestamp}",
        f"Fit region: [{result.fit_region[0]:.2f}, {result.fit_region[1]:.2f}]",
        f"Left background region: [{result.left_bg_region[0]:.2f}, {result.left_bg_region[1]:.2f}]",
        f"Right background region: [{result.right_bg_region[0]:.2f}, {result.right_bg_region[1]:.2f}]",
        f"Background: slope={result.background_slope:.6g}, intercept={result.background_intercept:.6g}",
        f"Independent widths: {not result.link_widths}",
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
    lines.append("")
    for i, peak in enumerate(result.peaks):
        lines.append(f"  Peak {i + 1}:")
        lines.append(f"    Position:  {_format_err(peak.position, peak.position_err)}")
        lines.append(f"    FWHM:      {_format_err(peak.fwhm, peak.fwhm_err)}")
        lines.append(f"    Amplitude: {_format_err(peak.amplitude, peak.amplitude_err)}")
        lines.append(f"    Sigma:     {_format_err(peak.sigma, peak.sigma_err)}")
        lines.append(f"    Area:      {_format_err(peak.area, peak.area_err)}")
    return "\n".join(lines)


def write_text_report(path, results, spectrum_path):
    """Writes a plain-text report for one or more fits to `path`,
    overwriting any existing file. `results` is a list of (fit_number,
    FitResult) pairs, in the order they should appear in the report."""
    blocks = [
        fit_result_to_text_report(result, spectrum_path, fit_number=fit_number)
        for fit_number, result in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fit_export.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fit_export.py tests/test_fit_export.py
git commit -m "feat: add fit_export module for JSON/text fit-result serialization"
```

---

### Task 4: Wire the automatic JSON Lines log into `run_fit()`

**Files:**
- Modify: `fit_mode.py:1-14` (imports), `fit_mode.py:603-654` (`run_fit`, as left by Task 2)
- Modify: `tests/test_fit_mode_ui.py:1-11` (imports), `tests/test_fit_mode_ui.py:41-50` (`_make_active_spectrum`)

- [ ] **Step 1: Write the failing tests**

First, update the test helper so tests that exercise `run_fit()` don't write into the repo directory once auto-logging lands. Add `import os` and `import tempfile` to the top of `tests/test_fit_mode_ui.py` (currently line 1 starts with `import numpy as np`):

```python
import os
import tempfile

import numpy as np
```

Replace `_make_active_spectrum` (currently `tests/test_fit_mode_ui.py:41-50`):

```python
def _make_active_spectrum(main_window, path=None):
    """`path` should point at a real (or plausible) location on disk --
    run_fit() writes an auto-log file next to it on every successful
    fit (see fit_export.append_auto_log). Defaults to a fresh temp
    directory per call so the many tests that don't care about export
    behavior never write into the repo; tests that DO care pass an
    explicit tmp_path-based path."""
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
```

Add these new tests at the end of `tests/test_fit_mode_ui.py`:

```python
def test_run_fit_appends_to_the_auto_log_next_to_the_spectrum_file(qapp, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    log_path = tmp_path / "eu_fits.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_run_fit_appends_a_second_line_for_a_second_fit(qapp, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    log_path = tmp_path / "eu_fits.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_run_fit_shows_a_status_message_when_the_auto_log_write_fails(qapp, tmp_path, monkeypatch):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    spectrum = _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(fit_mode.fit_export, "append_auto_log", _raise)
    main_window.fit_controller.run_fit()  # must not raise

    assert len(spectrum.fits) == 1  # the fit still committed
    assert main_window.statusBar().currentMessage() != ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fit_mode_ui.py -k "auto_log" -v`
Expected: FAIL — the first two with `AssertionError` (no `eu_fits.jsonl` file exists yet); the third with `AttributeError: module 'fit_mode' has no attribute 'fit_export'`.

- [ ] **Step 3: Implement the change**

In `fit_mode.py`, add `import fit_export` right before the `from peak_fit import (...)` block (as left by Task 2):

```python
import fit_export
from peak_fit import (
    FWHM_FACTOR, FitError, fit_peaks, fit_result_values_by_name, hypermet_left_tail,
    parameter_names,
)
```

In `run_fit()` (as left by Task 2), insert the auto-log call right after `active.fits.append(result)`:

```python
        active.fits.append(result)
        try:
            fit_export.append_auto_log(active.path, result)
        except OSError as exc:
            self._show_status_message(f"Could not write fit log: {exc}", 5000)
        names = parameter_names(len(result.peaks), result.link_widths, result.tail_fraction is not None)
        self.update_parameters_panel(names, fit_result_values_by_name(result))
        self.main_window._plot_data(preserve_view=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (the full file — confirms the `_make_active_spectrum` default change didn't break any of the other 39 call sites).

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: auto-log every successful fit to a JSON Lines file next to the spectrum"
```

---

### Task 5: Explicit "Export This Fit..." / "Export All Fits..." actions

**Files:**
- Modify: `fit_mode.py:1-14` (imports), `fit_mode.py:553-569` (`_on_results_context_menu`, as left by prior tasks)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing tests**

Add these tests at the end of `tests/test_fit_mode_ui.py`:

```python
def test_export_actions_appear_when_right_clicking_an_existing_fit(qapp, monkeypatch):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    from PySide6.QtWidgets import QMenu

    captured = {}

    class CapturingMenu(QMenu):
        def exec(self, *args, **kwargs):
            captured["actions"] = [a.text() for a in self.actions()]
            return None

    monkeypatch.setattr(fit_mode, "QMenu", CapturingMenu)

    table = main_window.fit_controller.results_table
    position = table.visualItemRect(table.item(0, 0)).center()
    main_window.fit_controller._on_results_context_menu(position)

    assert captured["actions"] == [
        "Remove Fit", "Export This Fit...", "Export All Fits...", "Clear All Fits",
    ]


def test_export_all_fits_action_absent_when_there_are_no_fits(qapp, monkeypatch):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    from PySide6.QtWidgets import QMenu

    captured = {}

    class CapturingMenu(QMenu):
        def exec(self, *args, **kwargs):
            captured["actions"] = [a.text() for a in self.actions()]
            return None

    monkeypatch.setattr(fit_mode, "QMenu", CapturingMenu)

    table = main_window.fit_controller.results_table
    main_window.fit_controller._on_results_context_menu(table.rect().center())

    assert captured["actions"] == ["Clear All Fits"]


def test_export_this_fit_writes_a_report_to_the_chosen_path(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    from PySide6.QtWidgets import QMenu

    class ImmediateExportMenu(QMenu):
        def exec(self, *args, **kwargs):
            for action in self.actions():
                if action.text() == "Export This Fit...":
                    return action
            return None

    monkeypatch.setattr(fit_mode, "QMenu", ImmediateExportMenu)
    out_path = tmp_path / "chosen_report.txt"
    monkeypatch.setattr(fit_mode.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_path), ""))

    table = main_window.fit_controller.results_table
    position = table.visualItemRect(table.item(0, 0)).center()
    main_window.fit_controller._on_results_context_menu(position)

    text = out_path.read_text(encoding="utf-8")
    assert "Fit 1" in text
    assert spectrum_path in text


def test_export_all_fits_writes_every_fit_in_order(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()
    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    from PySide6.QtWidgets import QMenu

    class ImmediateExportAllMenu(QMenu):
        def exec(self, *args, **kwargs):
            for action in self.actions():
                if action.text() == "Export All Fits...":
                    return action
            return None

    monkeypatch.setattr(fit_mode, "QMenu", ImmediateExportAllMenu)
    out_path = tmp_path / "all_report.txt"
    monkeypatch.setattr(fit_mode.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_path), ""))

    table = main_window.fit_controller.results_table
    position = table.visualItemRect(table.item(0, 0)).center()
    main_window.fit_controller._on_results_context_menu(position)

    text = out_path.read_text(encoding="utf-8")
    assert text.index("Fit 1") < text.index("Fit 2")


def test_export_cancelled_dialog_does_not_write_a_report_file(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()  # writes eu_fits.jsonl (the auto-log)

    monkeypatch.setattr(fit_mode.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))

    controller = main_window.fit_controller
    controller._export_fits(main_window.spectra[0], [0])  # must not raise

    assert [p.name for p in tmp_path.iterdir()] == ["eu_fits.jsonl"]


def test_export_shows_a_status_message_when_the_write_fails(qapp, monkeypatch, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    bad_path = str(tmp_path / "missing_dir" / "report.txt")
    monkeypatch.setattr(fit_mode.QFileDialog, "getSaveFileName", lambda *a, **k: (bad_path, ""))

    controller = main_window.fit_controller
    controller._export_fits(main_window.spectra[0], [0])  # must not raise

    assert main_window.statusBar().currentMessage() != ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fit_mode_ui.py -k "export" -v`
Expected: FAIL — the menu-contents tests with `AssertionError` (actions list doesn't yet include the new entries); the write/cancel/error tests with `AttributeError: 'FitModeController' object has no attribute '_export_fits'` (and `fit_mode.QFileDialog` not yet imported).

- [ ] **Step 3: Implement the change**

In `fit_mode.py`, the complete import block (as left by Task 4) is:

```python
import sys
import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QMenu, QTableWidget, QTableWidgetItem, QToolBar,
    QVBoxLayout, QWidget,
)

import fit_export
from peak_fit import (
    FWHM_FACTOR, FitError, fit_peaks, fit_result_values_by_name, hypermet_left_tail,
    parameter_names,
)
```

Add `import os` as the first line, and add `QFileDialog` to the `QtWidgets` import (everything else in the block is unchanged, including the `import fit_export` and `from peak_fit import (...)` lines):

```python
import os
import sys
import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QFileDialog, QMenu, QTableWidget, QTableWidgetItem, QToolBar,
    QVBoxLayout, QWidget,
)

import fit_export
from peak_fit import (
    FWHM_FACTOR, FitError, fit_peaks, fit_result_values_by_name, hypermet_left_tail,
    parameter_names,
)
```

Replace `_on_results_context_menu` (currently `fit_mode.py:553-569`) and add the new `_export_fits` helper right after it:

```python
    def _on_results_context_menu(self, position):
        mw = self.main_window
        item = self.results_table.itemAt(position)
        active = next((s for s in mw.spectra if s.active), None)
        menu = QMenu(mw)
        remove_action = menu.addAction("Remove Fit") if item is not None else None
        export_one_action = menu.addAction("Export This Fit...") if item is not None else None
        export_all_action = (
            menu.addAction("Export All Fits...") if active is not None and active.fits else None
        )
        clear_action = menu.addAction("Clear All Fits")
        chosen = menu.exec(self.results_table.viewport().mapToGlobal(position))
        if active is None:
            return
        if item is not None and chosen == remove_action:
            fit_index = self._results_row_fit_index[item.row()]
            del active.fits[fit_index]
            mw._plot_data(preserve_view=True)
        elif item is not None and chosen == export_one_action:
            fit_index = self._results_row_fit_index[item.row()]
            self._export_fits(active, [fit_index])
        elif chosen == export_all_action:
            self._export_fits(active, list(range(len(active.fits))))
        elif chosen == clear_action:
            active.fits.clear()
            mw._plot_data(preserve_view=True)

    def _export_fits(self, active, fit_indices):
        """Opens a save-file dialog and writes a plain-text report
        covering the given 0-based indices into active.fits -- used by
        both "Export This Fit..." (a single index) and "Export All
        Fits..." (every index, in Fit Results order)."""
        stem = os.path.splitext(os.path.basename(active.path))[0]
        if len(fit_indices) == 1:
            default_name = f"{stem}_fit{fit_indices[0] + 1}_report.txt"
        else:
            default_name = f"{stem}_fits_report.txt"
        directory = os.path.dirname(active.path)
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Export Fit Report", os.path.join(directory, default_name),
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        results = [(i + 1, active.fits[i]) for i in fit_indices]
        try:
            fit_export.write_text_report(path, results, active.path)
        except OSError as exc:
            self._show_status_message(f"Could not write export: {exc}", 5000)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (full file).

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: add Export This Fit / Export All Fits actions to the Fit Results context menu"
```

---

### Task 6: Full regression run and holistic review

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest -v`
Expected: PASS — every test in `tests/`, including `test_peak_fit.py`, `test_fit_mode.py`, `test_fit_mode_ui.py`, `test_fit_export.py`, and the unrelated I/O test files (`test_histogram_io.py`, `test_spe_io.py`, `test_spk_io.py`, `test_qt_setup.py`), all still pass.

- [ ] **Step 2: Review the whole branch diff together, not just per-task**

Run: `git diff master -- peak_fit.py fit_mode.py fit_export.py tests/`

Read through the full diff in one pass and confirm, across files (not just within each task's own diff):
- Every place that used to say "sigma" in a user-facing label now says "FWHM" (`_parameter_label` in `fit_mode.py`, and the five now-stale `# fix "Shared sigma"` comments in `tests/test_fit_mode_ui.py` at the lines that check the "Shared FWHM" checkbox — update any still reading "Shared sigma" to "Shared FWHM").
- `fixed_params_from_panel` and `initial_guess_overrides_from_panel` apply the FWHM→sigma conversion identically (no asymmetry between the two).
- `run_fit()`'s `current_names` filter is applied to both `fixed_params` and `initial_guess_overrides` (not just one of them).
- `fit_export.append_auto_log` and the explicit-export `_export_fits` path both wrap their file I/O in `try/except OSError` and route failures through `_show_status_message` — neither can raise out of `run_fit()` or the context-menu handler.
- No leftover references to the removed `_on_fix_toggled` method.

Fix anything found directly; re-run `python -m pytest -v` after any fix.

- [ ] **Step 3: Report completion**

No commit for this task (verification only). Summarize to the user which of the three original requests (FWHM in the panel, manual entry for every parameter, disk export) are done and ready for manual GUI testing, per this project's established pattern of the user verifying live-app behavior that can't be driven from here.
