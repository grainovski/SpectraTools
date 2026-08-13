# Matrix Panel Fit/Integrate/Calibrate Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the matrix panel's projection view (`matrix_panel.py`) to functional parity with the main window's ordinary-spectrum view (`main_window.py`) for zoom, peak fitting, integration, and calibration, plus two display/UX fixes and Knowledge Database documentation of 2D-spectra concepts.

**Architecture:** Duck-typed reuse — `MatrixPanel` implements the same attribute/method surface `FitModeController` (`fit_mode.py`) already expects from `main_window` (a `spectra` list wrapping the projection as one `LoadedSpectrum`, `channel_to_display`/`display_to_channel`, `_calibration`/`_calibration_active`, `_plot_data`, `_update_fit_mode_availability`), and constructs its own `FitModeController(self)` instance. `fit_mode.py`, `peak_fit.py`, `fit_export.py`, `calibration.py`, `calibration_dialog.py` are used entirely unmodified. `FitModeController.build_results_panel()`/`.build_parameters_panel()` (`fit_mode.py:726-812`) already operate generically against whatever `self.main_window` is — calling them from `MatrixPanel` requires **zero changes to `main_window.py` or `fit_mode.py`**. Calibration is shared: `MatrixPanel._calibration`/`_calibration_active` are properties delegating straight through to `main_window._calibration`/`_calibration_active`, so there is only ever one calibration, never two to keep in sync.

**Tech Stack:** PySide6, matplotlib (QtAgg backend), numpy — all already in use, no new dependencies.

**Full spec:** `docs/superpowers/specs/2026-08-13-matrix-panel-fit-parity-design.md`

---

## Context for every task below

All tasks modify `matrix_panel.py` (currently 290 lines) and `tests/test_matrix_panel.py` unless stated otherwise. The existing `MatrixPanel`/`MatrixCutController`/`MatrixCutState` classes are exactly as shipped in the matrix-analysis feature (`v3-major`, commits through `3b9f0c4`) — read `matrix_panel.py` in full before starting any task; it is short enough to hold in context completely, and every step below assumes you have.

Two important existing facts, already verified against source, that every task should rely on rather than re-derive:
- `_TrimmedNavigationToolbar` (`main_window.py:215-229`) is a 100%-generic `NavigationToolbar2QT` subclass (only overrides the class-level `toolitems` list) — reusable as-is via a **local** import inside `MatrixPanel.__init__` (`from main_window import _TrimmedNavigationToolbar`), never a module-level import. A module-level import would create a circular import: `main_window.py` imports `MatrixPanel` from `matrix_panel.py` at its own module top, before `_TrimmedNavigationToolbar` is defined at line 215, so `matrix_panel.py` cannot import it back at module load time. `matrix_panel.py` already uses this exact local-import pattern for `matrix_cut`/`matrix_heatmap`, so this matches established convention.
- `FitModeController` (`fit_mode.py`) does not check `event.modifiers()` in its own `eventFilter` — Ctrl+B (background-preview toggle) and bare-B (background-mark hold) already coexist in the shipped main window without an explicit modifier guard. `MatrixCutController` follows the identical pattern; do not add new modifier-checking logic that would make the two windows subtly diverge.

---

### Task 1: Histogram display

**Files:**
- Modify: `matrix_panel.py`
- Test: `tests/test_matrix_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_matrix_panel.py`:

```python
def test_matrix_panel_projection_displayed_as_histogram(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    line = panel.axes.lines[0]
    assert line.get_drawstyle() == "steps-mid"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matrix_panel.py::test_matrix_panel_projection_displayed_as_histogram -v`
Expected: FAIL — current drawstyle is `"default"` (a plain line).

- [ ] **Step 3: Fix `_plot_projection` in `matrix_panel.py`**

Change:
```python
        self.axes.plot(range(len(data)), data, linewidth=0.8)
```
to:
```python
        self.axes.plot(range(len(data)), data, drawstyle="steps-mid", linewidth=0.8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "fix: display matrix projection as histogram, matching main window"
```

---

### Task 2: Wrap the projection as a single `LoadedSpectrum`

**Context:** `FitModeController` expects `self.main_window.spectra` — a list of objects with `.path`, `.data`, `.color`, `.visible`, `.active`, `.fits`. `LoadedSpectrum` (`spectrum.py:35-42`) is exactly this shape, with no Qt dependency. `MatrixPanel` always has exactly one "spectrum" (the current working-axis projection) — no visibility toggling, no multi-selection, unlike `MainWindow`'s own list.

**Files:**
- Modify: `matrix_panel.py`
- Test: `tests/test_matrix_panel.py`

- [ ] **Step 1: Write the failing test**

```python
def test_matrix_panel_wraps_projection_as_single_spectrum(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert len(panel.spectra) == 1
    wrapped = panel.spectra[0]
    assert wrapped.active is True
    assert wrapped.visible is True
    assert wrapped.fits == []
    np.testing.assert_array_equal(wrapped.data, panel.projections["x"])


def test_matrix_panel_rebuilds_spectra_on_axis_switch(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    panel.axis_selector.setCurrentIndex(1)

    assert len(panel.spectra) == 1
    np.testing.assert_array_equal(panel.spectra[0].data, panel.projections["y"])
```

Add `import numpy as np` to the top of `tests/test_matrix_panel.py` if not already present (check first — it may already be imported).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_panel.py -k spectra -v`
Expected: FAIL — `MatrixPanel` has no `spectra` attribute.

- [ ] **Step 3: Add `_rebuild_spectra` and wire it in**

Add the import at the top of `matrix_panel.py` (alphabetical among the existing local-module imports, i.e. after `mtx_io` and before `theme`... check the actual current order and insert alphabetically, matching this file's established convention):
```python
from spectrum import LoadedSpectrum
```

Add a new method, placed right after `__init__`:
```python
    def _rebuild_spectra(self):
        """Wraps the current working-axis projection as a single
        LoadedSpectrum so FitModeController (duck-typed against this
        window, see the fit-integration task) can operate on it exactly
        as it does on MainWindow's own spectra list. Always exactly one
        entry, always active and visible -- there is no concept of
        multiple or hidden "spectra" in this window."""
        data = self.projections[self.working_axis]
        label = f"{os.path.basename(self.path)} {self.working_axis} projection"
        spectrum = LoadedSpectrum(label, data, color="tab:blue")
        spectrum.active = True
        self.spectra = [spectrum]
```

Call it in `__init__`, right after `self.working_axis = "x"`:
```python
        self.working_axis = "x"
        self._rebuild_spectra()
```

Call it in `_on_axis_changed`, right after `self.working_axis = self.axis_selector.itemData(index)`:
```python
    def _on_axis_changed(self, index):
        self.working_axis = self.axis_selector.itemData(index)
        self._rebuild_spectra()
        self.cut_controller.clear()
        self._plot_projection()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "feat: wrap matrix projection as a LoadedSpectrum for fit-mode reuse"
```

---

### Task 3: Shared calibration + Calibrate dialog

**Context:** `MatrixPanel._calibration`/`_calibration_active` are properties delegating to `main_window._calibration`/`main_window._calibration_active` — there is only ever one calibration, shared everywhere. `channel_to_display`/`display_to_channel` (`main_window.py:468-484`) depend on nothing but those two attributes plus `Calibration.apply`/`.invert`, which are pure, Qt-free methods (`calibration.py`) — safe to duplicate verbatim. `CalibrationDialog.__init__` (`calibration_dialog.py:28-79`) needs only `(parent, initial=Calibration|None, initially_active=bool)` — no spectrum data.

`_apply_calibration_change` below is intentionally the simple version for this task (redraws via the current `_plot_projection`, no view-preservation yet). Task 5 upgrades it to call the new `_plot_data` with view-preservation once that method exists. Task 7 upgrades it again to refresh the Fit Parameters panel once `fit_controller` exists. This is not a "come back and fix it later" gap — each version is fully correct and tested for what exists at that point in the plan.

**Files:**
- Modify: `matrix_panel.py`
- Test: `tests/test_matrix_panel.py`

- [ ] **Step 1: Write the failing test**

```python
def test_matrix_panel_calibration_shared_with_main_window(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    cal = Calibration(kind="linear", a=1.0, b=2.0, c=0.0)

    panel._calibration = cal
    panel._calibration_active = True

    assert main_window._calibration is cal
    assert main_window._calibration_active is True
    assert panel.channel_to_display(100) == pytest.approx(cal.apply(100))
    assert panel.display_to_channel(201.0) == pytest.approx(cal.invert(201.0))


def test_matrix_panel_channel_to_display_passthrough_when_uncalibrated(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.channel_to_display(150) == 150
    assert panel.display_to_channel(150) == 150


def test_matrix_panel_calibrate_button_opens_dialog_and_applies_result(qapp, monkeypatch):
    from calibration import Calibration
    from calibration_dialog import CalibrationDialog
    from PySide6.QtWidgets import QDialog

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    result = Calibration(kind="linear", a=0.0, b=1.5, c=0.0)

    def fake_exec(self):
        self.result_calibration = result
        self.result_active = True
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CalibrationDialog, "exec", fake_exec)

    panel._open_calibration_dialog()

    assert main_window._calibration is result
    assert main_window._calibration_active is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_panel.py -k calibrat -v`
Expected: FAIL — `MatrixPanel` has no `_calibration`, `channel_to_display`, or `_open_calibration_dialog`.

- [ ] **Step 3: Add calibration properties, transforms, and dialog wiring**

Add imports at the top of `matrix_panel.py`:
```python
from PySide6.QtWidgets import QDialog
```
(add to the existing `from PySide6.QtWidgets import (...)` block, alphabetically) and:
```python
from calibration_dialog import CalibrationDialog
```
(alphabetically among the local-module imports).

Add these methods to `MatrixPanel`, placed after `_rebuild_spectra`:
```python
    @property
    def _calibration(self):
        return self.main_window._calibration

    @_calibration.setter
    def _calibration(self, value):
        self.main_window._calibration = value

    @property
    def _calibration_active(self):
        return self.main_window._calibration_active

    @_calibration_active.setter
    def _calibration_active(self, value):
        self.main_window._calibration_active = value

    def channel_to_display(self, channel):
        if not self._calibration_active or self._calibration is None:
            return channel
        return self._calibration.apply(channel)

    def display_to_channel(self, display_x):
        if not self._calibration_active or self._calibration is None:
            return display_x
        return self._calibration.invert(display_x)

    def _open_calibration_dialog(self):
        dialog = CalibrationDialog(
            self, initial=self._calibration, initially_active=self._calibration_active
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_calibration_change(dialog.result_calibration, dialog.result_active)

    def _apply_calibration_change(self, new_calibration, new_active):
        self._calibration = new_calibration
        self._calibration_active = new_active
        self._plot_projection()
```

Add a "Calibrate..." button to the top bar, matching this window's existing button-based convention (not a menu, since `MatrixPanel` has none). In `__init__`, right after `self.heatmap_button = ...` / before it's added to `top_bar` — add the button construction next to the other button constructions:
```python
        self.calibrate_button = QPushButton("Calibrate...")
        self.calibrate_button.setShortcut("Ctrl+L")
        self.calibrate_button.clicked.connect(self._open_calibration_dialog)
```
and add it to the `top_bar` layout, right after `top_bar.addWidget(self.clear_marks_button)`:
```python
        top_bar.addWidget(self.calibrate_button)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "feat: add shared calibration and Calibrate dialog to matrix panel"
```

---

### Task 4: Gate marking key rename (B → G), calibration-aware marking, pending-click preview

**Context:** Three related fixes to `MatrixCutController`/`MatrixCutState`, all touching the same code:

1. **Key rename.** Hold-B is reserved for the fit system (Task 7, matching `main_window.py` exactly); the gate/cut system's "background region" key moves to hold-**G**. Hold-C (cut) is unaffected.
2. **Calibration-aware marking.** `MatrixCutState` must store channel numbers, not display (possibly keV) coordinates — exactly like `FitModeState` already does ("FitModeState itself always stays channel-based; to_display converts to keV for drawing only, when calibrated" — `fit_mode.py:477`). This is not cosmetic: `compute_cut` (`matrix_cut.py`) expects channel indices. Storing raw, un-converted `event.xdata` (as the code does today) would silently corrupt every cut computed while a calibration is active, once Task 3 makes calibration reachable from this window. `on_click` must convert display→channel on the way in (`self.panel.display_to_channel(event.xdata)`); `_redraw_markers` must convert channel→display on the way out for every drawn artist.
3. **Pending-click preview.** Currently, holding C or G and clicking once silently stores the click with nothing drawn until the second click completes the region. `fit_mode.py`'s `_redraw_progress` (`fit_mode.py:490-495`) already solves this for fit marks: a dashed `axvline` at the pending click's position. Replicate the same treatment for `_pending_cut_click` and (renamed) `_pending_bg_click`.

**Files:**
- Modify: `matrix_panel.py`
- Test: `tests/test_matrix_panel.py`

- [ ] **Step 1: Write the failing tests**

Update the existing `_held_key_click` test helper (it currently sets `panel.cut_controller._held_key = key` directly, bypassing the eventFilter — no change needed there, since it already takes the intended `_held_key` value as a parameter; only the VALUE passed for background marking changes). Update every existing call site in `tests/test_matrix_panel.py` that currently passes `"bg"` as the held key to pass `"gate_bg"` instead — search for `_held_key_click(panel, "bg"` and replace with `_held_key_click(panel, "gate_bg"`. Do this for all matching lines (there are several, covering background-region marking tests).

Add new tests:
```python
def test_matrix_panel_gate_background_uses_g_not_b(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    from PySide6.QtCore import Qt

    panel.cut_controller._held_key = None
    event = type("FakeKeyEvent", (), {
        "type": lambda self: __import__("PySide6.QtCore", fromlist=["QEvent"]).QEvent.Type.KeyPress,
        "key": lambda self: Qt.Key.Key_B,
        "isAutoRepeat": lambda self: False,
    })()
    panel.cut_controller.eventFilter(panel.canvas, event)

    assert panel.cut_controller._held_key is None


def test_matrix_panel_cut_marks_stored_in_channel_space(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel._calibration = Calibration(kind="linear", a=0.0, b=2.0, c=0.0)
    panel._calibration_active = True

    _held_key_click(panel, "cut", 200.0)
    _held_key_click(panel, "cut", 400.0)

    # Clicked at display (keV) positions 200/400 with b=2.0 -- channel
    # space is display / 2, so the stored region should be (100, 200),
    # not the raw display values (200, 400).
    assert panel.cut_controller.state.cut_region == pytest.approx((100.0, 200.0))


def test_matrix_panel_pending_cut_click_shows_dashed_preview_line(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _click(panel, 150.0)  # single click while holding cut -- but no hold set here
    panel.cut_controller._held_key = "cut"
    _click(panel, 150.0)
    panel.cut_controller._held_key = None

    lines = [a for a in panel.cut_controller._artists if hasattr(a, "get_linestyle")]
    assert any(line.get_linestyle() == "--" for line in lines)


def test_matrix_panel_pending_gate_bg_click_shows_dashed_preview_line(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "gate_bg", 300.0)

    lines = [a for a in panel.cut_controller._artists if hasattr(a, "get_linestyle")]
    assert any(line.get_linestyle() == "--" for line in lines)
```

Note: `test_matrix_panel_gate_background_uses_g_not_b` above constructs a fake key event awkwardly because `QKeyEvent` is not trivially constructible in a test; if this proves unreliable when actually run, replace it with a simpler assertion instead — directly call `panel.cut_controller.eventFilter` is fragile. A more robust equivalent: since holding G now must produce a `"gate_bg"` held-key state, and holding bare B must NOT, the simplest reliable test is checking that `_held_key_click(panel, "gate_bg", 300.0)` (which sets `_held_key` directly, matching every other existing test's convention) still produces a background region exactly as `_held_key_click(panel, "bg", ...)` used to — which `test_matrix_panel_pending_gate_bg_click_shows_dashed_preview_line` above already exercises indirectly. Prefer deleting the awkward fake-event test and relying on the simpler ones; use judgment here.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: several FAIL — key not yet renamed, values not yet channel-converted, no preview artists drawn.

- [ ] **Step 3: Implement all three fixes in `MatrixCutController`**

Replace the `eventFilter` method:
```python
    def eventFilter(self, obj, event):
        if obj is self.panel.canvas:
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                if event.key() == Qt.Key.Key_C:
                    self._held_key = "cut"
                elif event.key() == Qt.Key.Key_G:
                    self._held_key = "gate_bg"
            elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
                self._held_key = None
        return False
```

Replace the `on_click` method:
```python
    def on_click(self, event):
        if event.inaxes != self.panel.axes or event.xdata is None or event.button != 1:
            return
        channel_x = self.panel.display_to_channel(event.xdata)
        if self._held_key == "cut":
            self.state.add_cut_click(channel_x)
        elif self._held_key == "gate_bg":
            self.state.add_bg_click(channel_x)
        else:
            return
        self._redraw_markers()
        self.panel._update_activate_button()
```

Replace the `_redraw_markers` method:
```python
    def _redraw_markers(self):
        for artist in self._artists:
            try:
                artist.remove()
            except NotImplementedError:
                pass
        self._artists = []

        axes = self.panel.axes
        to_display = self.panel.channel_to_display
        state = self.state

        if state._pending_cut_click is not None:
            self._artists.append(
                axes.axvline(
                    to_display(state._pending_cut_click),
                    color=CUT_REGION_COLOR, linestyle="--", linewidth=1,
                )
            )
        if state.cut_region is not None:
            lo, hi = state.cut_region
            self._artists.append(
                axes.axvspan(
                    to_display(lo), to_display(hi), color=CUT_REGION_COLOR, alpha=CUT_REGION_ALPHA
                )
            )

        if state._pending_bg_click is not None:
            self._artists.append(
                axes.axvline(
                    to_display(state._pending_bg_click),
                    color=BG_REGION_COLOR, linestyle="--", linewidth=1,
                )
            )
        for lo, hi in state.bg_regions:
            self._artists.append(
                axes.axvspan(
                    to_display(lo), to_display(hi), color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA
                )
            )
        self.panel.canvas.draw()
```

`MatrixCutState` itself needs no changes — `_pending_cut_click`/`_pending_bg_click`/`cut_region`/`bg_regions` already store whatever `add_cut_click`/`add_bg_click` are given, and those now receive channel values from the updated `on_click` above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "fix: rename gate background key to G, make gate marking calibration-aware, add pending-click preview"
```

---

### Task 5: `_plot_data` (replaces `_plot_projection`)

**Context:** Mirrors `main_window.py:884-926`'s `_plot_data` exactly, adapted for `self.spectra` always having exactly one entry and no `log_scale_action` (matrix panel has no log-scale toggle — out of scope, not requested; always linear, matching this window's existing, unchanged behavior). Also updates Task 3's `_apply_calibration_change` to preserve THIS panel's own view across a calibration change, matching `main_window.py:493-523`'s pattern.

**Important — Task 3's actual final shape (a real bug was found and fixed in its code review, after this plan section was originally written):** `_apply_calibration_change` does **not** independently set `self._calibration`/`self._calibration_active` and redraw as this section originally assumed. It delegates:
```python
    def _apply_calibration_change(self, new_calibration, new_active):
        self.main_window._apply_calibration_change(new_calibration, new_active)
        self._plot_projection()
```
This exists because an earlier version called `self.main_window._plot_data(preserve_view=True)` directly, which reused `main_window.axes.get_xlim()`'s raw numbers verbatim — captured AFTER the calibration had already changed, so those numbers carried no memory of what they meant under the OLD calibration, silently showing the wrong region. Delegating to `main_window._apply_calibration_change` (which already gets its OWN view's round-trip right, updates its toolbar/menu indicators, and redraws every panel in `main_window._matrix_panels` — including this one, in real usage via `_open_matrix_panel` registration) fixes that, and also fixes two things this plan section didn't originally address at all: main_window's calibration-toggle UI staying in sync, and a SECOND open matrix panel picking up the change too.

What that delegation does **not** yet do: preserve THIS panel's own zoomed view across the change (it just calls a bare `self._plot_projection()`, which — consistent with that method's existing, unchanged, always-resets-to-full-view behavior — resets to the full projection). Adding real view-preservation for this panel's own plot is genuinely a Task 5 concern, not a regression to fix now — `_plot_projection()` has never taken a `preserve_view`/`xlim_override` argument at all, at any point in this plan before this task.

**Files:**
- Modify: `matrix_panel.py`
- Test: `tests/test_matrix_panel.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_matrix_panel_plot_data_draws_histogram_with_calibration_aware_axis(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel._calibration = Calibration(kind="linear", a=0.0, b=2.0, c=0.0)
    panel._calibration_active = True

    panel._plot_data()

    line = panel.axes.lines[0]
    assert line.get_drawstyle() == "steps-mid"
    assert panel.axes.get_xlabel() == "Energy (keV)"


def test_matrix_panel_plot_data_preserves_view_when_requested(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel.axes.set_xlim(100, 500)

    panel._plot_data(preserve_view=True)

    assert panel.axes.get_xlim() == pytest.approx((100, 500))


def test_matrix_panel_calibrating_preserves_the_current_view(qapp):
    from calibration import Calibration

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel.axes.set_xlim(100, 500)

    panel._apply_calibration_change(Calibration(kind="linear", a=0.0, b=2.0, c=0.0), True)

    # View was (100, 500) in channel space; after a b=2.0 calibration
    # the same channel window should now read as (200, 1000) in keV.
    assert panel.axes.get_xlim() == pytest.approx((200.0, 1000.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_panel.py -k plot_data -v`
Expected: FAIL — `_plot_data` doesn't exist yet.

- [ ] **Step 3: Add `_plot_data`, remove `_plot_projection`, update call sites**

Replace the `_plot_projection` method entirely with:
```python
    def _plot_data(self, preserve_view=False, xlim_override=None):
        saved_xlim = self.axes.get_xlim() if preserve_view else None
        self.axes.clear()
        style_axes(self.axes, self.main_window._theme)
        spectrum = self.spectra[0]
        channels = np.arange(len(spectrum.data))
        x = self.channel_to_display(channels)
        self.axes.plot(x, spectrum.data, drawstyle="steps-mid", linewidth=0.8, color=spectrum.color)
        self.fit_controller.draw_committed_fits(spectrum)
        self.axes.set_xlabel("Energy (keV)" if self._calibration_active else f"{self.working_axis.upper()} channel")
        self.axes.set_ylabel("Counts")
        if xlim_override is not None:
            xlim = xlim_override
        elif saved_xlim is not None:
            xlim = saved_xlim
        else:
            xlim = (self.channel_to_display(0), self.channel_to_display(len(spectrum.data) - 1))
        self.axes.set_xlim(xlim)
        self.canvas.draw()
        self._update_fit_mode_availability()
```

This references `self.fit_controller` and `self._update_fit_mode_availability`, neither of which exist until Task 7 — **this is expected and matches the plan's incremental-build pattern**. Add temporary stand-ins at the end of this step so `_plot_data` is callable and testable now; Task 7 will replace both with the real versions:
```python
    def _update_fit_mode_availability(self):
        pass
```
Do NOT add a temporary `fit_controller` stand-in — instead, guard the call in `_plot_data` itself for now:
```python
        if hasattr(self, "fit_controller"):
            self.fit_controller.draw_committed_fits(spectrum)
```
(replace the unconditional `self.fit_controller.draw_committed_fits(spectrum)` line above with this guarded version). Task 7 will remove the `hasattr` guard once `fit_controller` always exists by the time `_plot_data` can be called (i.e., once it's constructed unconditionally in `__init__`).

Update every call site that referenced `_plot_projection`, in `matrix_panel.py` **and** `main_window.py`:
- In `matrix_panel.py`'s `__init__`, change `self._plot_projection()` to `self._plot_data()`.
- In `matrix_panel.py`'s `_on_axis_changed`, change `self._plot_projection()` to `self._plot_data()`.
- In `matrix_panel.py`'s `_apply_calibration_change` (see below) — the trailing `self._plot_projection()` becomes part of the new view-preserving body, not a bare call.
- In `main_window.py`'s `_apply_calibration_change` (**not `matrix_panel.py`** — this is the one genuinely-necessary change to `main_window.py` this whole plan makes, added during Task 3's code review, not part of this plan's original file-structure section), find:
  ```python
        for panel in self._matrix_panels:
            panel._plot_projection()
  ```
  and change it to:
  ```python
        for panel in self._matrix_panels:
            panel._plot_data()
  ```
  This is a real call site — skipping it leaves an `AttributeError` waiting for the next time a calibration change reaches an open matrix panel through `MainWindow`'s own dialog/toolbar, once `_plot_projection` no longer exists.

Update `_apply_calibration_change` (from Task 3) to ALSO preserve this panel's own view across the calibration change, on top of the delegation Task 3's code review already put in place — capture this panel's own channel-space view bounds (via `display_to_channel`, using the OLD calibration) BEFORE delegating, then re-render at the equivalent bounds under the NEW calibration:
```python
    def _apply_calibration_change(self, new_calibration, new_active):
        old_xlim = self.axes.get_xlim()
        channel_bounds = (self.display_to_channel(old_xlim[0]), self.display_to_channel(old_xlim[1]))
        self.main_window._apply_calibration_change(new_calibration, new_active)
        new_xlim = (self.channel_to_display(channel_bounds[0]), self.channel_to_display(channel_bounds[1]))
        self._plot_data(xlim_override=new_xlim)
```
Note `channel_bounds` must be computed BEFORE `self.main_window._apply_calibration_change(...)` runs — that call is what overwrites the shared calibration state `self.display_to_channel` reads, so computing it after would use the NEW calibration for a value that's supposed to represent the OLD view.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "feat: replace _plot_projection with _plot_data (calibration-aware, fit-overlay-ready)"
```

---

### Task 6: Zoom (Ctrl+=/Ctrl+-/Ctrl+0, scroll-wheel)

**Context:** Mirrors `main_window.py`'s `_zoom_x` (1192-1221), `_show_full_spectrum` (1223-1232), `_autoscale_y` (1234-1266), and scroll-wheel wiring (1186-1190, connected at 278), adapted for `self.spectra` always having one entry and no `log_scale_action` (always linear — see Task 5's rationale). Replaces the stock `NavigationToolbar2QT` with the shared `_TrimmedNavigationToolbar` (imported locally, see the "Context for every task" note at the top of this plan).

**Files:**
- Modify: `matrix_panel.py`
- Test: `tests/test_matrix_panel.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_matrix_panel_has_trimmed_toolbar_and_zoom_actions(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    action_texts = [a.text() for a in panel.nav_toolbar.actions()]
    assert "Zoom" not in action_texts  # stock rectangle-zoom is trimmed out
    assert panel.zoom_in_action.shortcut().toString() == "Ctrl+="
    assert panel.zoom_out_action.shortcut().toString() == "Ctrl+-"
    assert panel.full_view_action.shortcut().toString() == "Ctrl+0"


def test_matrix_panel_zoom_in_narrows_xlim(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    full_lo, full_hi = panel.axes.get_xlim()
    full_width = full_hi - full_lo

    panel.zoom_in_action.trigger()

    new_lo, new_hi = panel.axes.get_xlim()
    assert (new_hi - new_lo) < full_width


def test_matrix_panel_full_view_action_resets_zoom(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    full_xlim = panel.axes.get_xlim()
    panel.axes.set_xlim(1000, 2000)

    panel.full_view_action.trigger()

    assert panel.axes.get_xlim() == pytest.approx(full_xlim)


def test_matrix_panel_scroll_event_zooms(qapp):
    from matplotlib.backend_bases import MouseEvent

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    full_lo, full_hi = panel.axes.get_xlim()
    full_width = full_hi - full_lo

    px, py = panel.axes.transData.transform((4096.0, 10.0))
    event = MouseEvent("scroll_event", panel.canvas, px, py, button="up")
    panel.canvas.callbacks.process("scroll_event", event)

    new_lo, new_hi = panel.axes.get_xlim()
    assert (new_hi - new_lo) < full_width
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_panel.py -k zoom -v`
Expected: FAIL — none of `zoom_in_action`/`zoom_out_action`/`full_view_action` exist yet, toolbar is still the untrimmed stock one.

- [ ] **Step 3: Add zoom actions, `_zoom_x`, `_show_full_view`, `_autoscale_y`, scroll wiring**

Add the import at the top of `matrix_panel.py`:
```python
from PySide6.QtGui import QAction
```

Add a module-level constant near the top of the file, alongside the existing `CUT_REGION_COLOR` etc. constants:
```python
ZOOM_FACTOR = 1.5
```

In `__init__`, replace:
```python
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)
```
with:
```python
        from main_window import _TrimmedNavigationToolbar  # local: avoids a circular import with main_window.py

        self.nav_toolbar = _TrimmedNavigationToolbar(self.canvas, self, coordinates=False)
        self.nav_toolbar.addSeparator()

        self.zoom_in_action = QAction("Zoom In (+)", self)
        self.zoom_in_action.setShortcut("Ctrl+=")
        self.zoom_in_action.triggered.connect(lambda: self._zoom_x(1 / ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Zoom Out (-)", self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_out_action.triggered.connect(lambda: self._zoom_x(ZOOM_FACTOR))
        self.nav_toolbar.addAction(self.zoom_out_action)

        self.full_view_action = QAction("Full View", self)
        self.full_view_action.setShortcut("Ctrl+0")
        self.full_view_action.triggered.connect(self._show_full_view)
        self.nav_toolbar.addAction(self.full_view_action)
```
(Text-labeled actions, deliberately not the custom magnifier-icon graphics `main_window.py` uses — those icons are baked per-theme at construction time, a complexity class this window doesn't need to take on for zoom to work correctly; see the design spec's scope notes.)

Wire scroll-wheel zoom, right after the canvas is constructed (near the existing `self.canvas = FigureCanvasQTAgg(self.figure)` line, or any point in `__init__` after `self.canvas` and `self.axes` both exist):
```python
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
```

Add these methods to `MatrixPanel` (placed after `_apply_calibration_change`):
```python
    def _on_scroll(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)

    def _zoom_x(self, factor, center=None):
        spectrum = self.spectra[0]
        xlim = self.axes.get_xlim()
        if center is None:
            center = (xlim[0] + xlim[1]) / 2
        half_width = abs(xlim[1] - xlim[0]) / 2 * factor
        max_channel = len(spectrum.data) - 1
        display_lo = self.channel_to_display(0)
        display_hi = self.channel_to_display(max_channel)
        display_lo, display_hi = min(display_lo, display_hi), max(display_lo, display_hi)
        new_lo = max(display_lo, center - half_width)
        new_hi = min(display_hi, center + half_width)
        if new_hi <= new_lo:
            new_hi = min(display_hi, new_lo + 1)
        new_xlim = (new_lo, new_hi) if xlim[0] <= xlim[1] else (new_hi, new_lo)
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()

    def _show_full_view(self):
        spectrum = self.spectra[0]
        max_channel = len(spectrum.data) - 1
        full_xlim = (self.channel_to_display(0), self.channel_to_display(max_channel))
        self.axes.set_xlim(full_xlim)
        self._autoscale_y(full_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()

    def _autoscale_y(self, xlim):
        spectrum = self.spectra[0]
        channel_lo = self.display_to_channel(xlim[0])
        channel_hi = self.display_to_channel(xlim[1])
        channel_lo, channel_hi = min(channel_lo, channel_hi), max(channel_lo, channel_hi)
        lo_bound = max(0, int(np.floor(channel_lo)))
        hi_bound = min(int(np.ceil(channel_hi)) + 1, len(spectrum.data))
        if lo_bound >= hi_bound:
            return
        window = spectrum.data[lo_bound:hi_bound]
        y_min = float(np.min(window))
        y_max = float(np.max(window))
        margin = (y_max - y_min) * 0.05 or 1.0
        self.axes.set_ylim(y_min - margin, y_max + margin)
```

Add `import numpy as np` at the top of `matrix_panel.py` if not already present (it is not, per the current file — add it near the top, before the matplotlib imports, matching this codebase's usual import ordering).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "feat: add custom X-axis zoom to matrix panel, matching main window"
```

---

### Task 7: Fit, Clear, Integrate, Preview-Background wiring + docks

**Context:** Constructs `self.fit_controller = FitModeController(self)` and calls its own `build_results_panel()`/`build_parameters_panel()` (`fit_mode.py:726-812`) — both already fully generic against whatever `self.main_window` is, requiring zero changes to `fit_mode.py`. Adds the four keyboard-shortcut-only actions `main_window.py:1143-1164` defines (`fit_button`/`clear_fit_button`/`integrate_button`/`background_preview_button`) — none of these have a visible toolbar button in `MainWindow` either (only a keyboard shortcut), so none are added here. Removes the `hasattr` guard and stub `_update_fit_mode_availability` from Task 5.

**Files:**
- Modify: `matrix_panel.py`
- Test: `tests/test_matrix_panel.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_matrix_panel_has_fit_and_parameters_docks(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.fit_controller.results_dock is not None
    assert panel.fit_controller.parameters_dock is not None
    assert panel.fit_controller.results_dock.parent() is panel
    assert panel.fit_controller.parameters_dock.parent() is panel


def test_matrix_panel_fit_button_disabled_without_fit_region(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.fit_button.isEnabled() is False


def test_matrix_panel_fitting_a_peak_on_the_projection_works(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    data = panel.spectra[0].data
    # Any two bare channel positions bracketing a stretch of real
    # projection data -- this is a real 8192-channel fixture, not
    # synthetic, so a real fit may or may not converge depending on
    # the exact region; the point of this test is that the SAME
    # FitModeController machinery MainWindow uses is reachable and
    # produces a fits-list entry, not that any specific region fits
    # cleanly. Mark a wide fit region and one peak position roughly
    # in its middle, matching how test_fit_mode_ui.py's own tests
    # mark fits.
    lo, hi = 100.0, 300.0

    panel.cut_controller._held_key = None  # not used by fit marking; ensure no interference
    panel.fit_controller._held_key = "r"
    _click(panel, lo)
    _click(panel, hi)
    panel.fit_controller._held_key = "p"
    _click(panel, (lo + hi) / 2)
    panel.fit_controller._held_key = None

    assert panel.fit_controller.state.fit_region == pytest.approx((lo, hi))
    assert len(panel.fit_controller.state.peak_positions) == 1


def test_matrix_panel_integrate_action_reachable(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.integrate_button.shortcut().toString() == "Ctrl+I"
    assert panel.fit_button.shortcut().toString() == "Ctrl+F"
    assert panel.clear_fit_button.shortcut().toString() == "Ctrl+C"
    assert panel.background_preview_button.shortcut().toString() == "Ctrl+B"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_panel.py -k "fit_controller or fit_button or integrate or fitting_a_peak" -v`
Expected: FAIL — none of `fit_controller`/`fit_button`/`integrate_button`/etc. exist yet.

- [ ] **Step 3: Wire `FitModeController` into `MatrixPanel`**

Add the import at the top of `matrix_panel.py`:
```python
from fit_mode import FitModeController
```

In `__init__`, right after the existing `self.cut_controller = MatrixCutController(self)` line, add:
```python
        self.fit_controller = FitModeController(self)
        self.fit_controller.build_results_panel()
        self.fit_controller.build_parameters_panel()

        self.fit_button = QAction("Fit", self)
        self.fit_button.setShortcut("Ctrl+F")
        self.fit_button.setEnabled(False)
        self.fit_button.triggered.connect(self.fit_controller.run_fit)
        self.addAction(self.fit_button)

        self.clear_fit_button = QAction("Clear", self)
        self.clear_fit_button.setShortcut("Ctrl+C")
        self.clear_fit_button.triggered.connect(self.fit_controller.clear)
        self.addAction(self.clear_fit_button)

        self.integrate_button = QAction("Integrate", self)
        self.integrate_button.setShortcut("Ctrl+I")
        self.integrate_button.setEnabled(False)
        self.integrate_button.triggered.connect(self.fit_controller.run_integration)
        self.addAction(self.integrate_button)

        self.background_preview_button = QAction("Preview Background Fit", self)
        self.background_preview_button.setShortcut("Ctrl+B")
        self.background_preview_button.triggered.connect(self.fit_controller.toggle_background_preview)
        self.addAction(self.background_preview_button)
```

Replace the Task 5 stub `_update_fit_mode_availability` with the real version:
```python
    def _update_fit_mode_availability(self):
        self.fit_button.setEnabled(self.fit_controller.state.ready_to_fit())
        self.integrate_button.setEnabled(self.fit_controller.state.ready_to_integrate())
```

In `_plot_data`, replace the guarded line:
```python
        if hasattr(self, "fit_controller"):
            self.fit_controller.draw_committed_fits(spectrum)
```
with the unconditional version (safe now that `fit_controller` is always constructed before `_plot_data` can be called — `__init__` builds it before the first `_plot_data()` call, since `_rebuild_spectra`/calibration/etc. all precede it in `__init__`'s ordering; verify this ordering holds and adjust `__init__`'s statement order if the first `_plot_data()` call currently happens before `self.fit_controller = FitModeController(self)` — it must come after):
```python
        self.fit_controller.draw_committed_fits(spectrum)
```

Update `_apply_calibration_change` to also refresh the parameters panel, matching `main_window.py:523` — **keep the `self.main_window._apply_calibration_change(...)` delegation Task 5 already has** (do not replace it with direct `self._calibration = ...`/`self._calibration_active = ...` assignment; that would silently drop main_window's own redraw, its toolbar/menu sync, and the loop that refreshes every OTHER open matrix panel — exactly the bug Task 3's code review found and fixed):
```python
    def _apply_calibration_change(self, new_calibration, new_active):
        old_xlim = self.axes.get_xlim()
        channel_bounds = (self.display_to_channel(old_xlim[0]), self.display_to_channel(old_xlim[1]))
        self.main_window._apply_calibration_change(new_calibration, new_active)
        new_xlim = (self.channel_to_display(channel_bounds[0]), self.channel_to_display(channel_bounds[1]))
        self._plot_data(xlim_override=new_xlim)
        self.fit_controller.refresh_parameters_panel_calibration()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, full file. Also run `pytest tests/test_fit_mode_ui.py tests/test_main_window.py -v` (or whatever the actual main-window/fit-mode test file names are — check with `ls tests/` first) to confirm zero regression to the existing, unmodified `fit_mode.py`/`main_window.py` behavior.

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "feat: reuse FitModeController for fit/integrate/results docks in matrix panel"
```

---

### Task 8: Axis-switch fully resets fit and gate state

**Context:** `_on_axis_changed` already calls `_rebuild_spectra()` (Task 2) and `cut_controller.clear()`. Now that `fit_controller` exists (Task 7), extend the reset to fits and any in-progress fit marking too — switching from X to Y projection is a switch to completely different underlying data; nothing marked or fitted against the old data is meaningful against the new data.

**Files:**
- Modify: `matrix_panel.py`
- Test: `tests/test_matrix_panel.py`

- [ ] **Step 1: Write the failing test**

```python
def test_matrix_panel_switching_axis_clears_fit_state(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    panel.fit_controller._held_key = "r"
    _click(panel, 100.0)
    _click(panel, 300.0)
    panel.fit_controller._held_key = None
    assert panel.fit_controller.state.fit_region is not None

    panel.axis_selector.setCurrentIndex(1)

    assert panel.fit_controller.state.fit_region is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matrix_panel.py::test_matrix_panel_switching_axis_clears_fit_state -v`
Expected: FAIL — `_on_axis_changed` doesn't touch `fit_controller` yet, so `fit_region` survives the switch.

- [ ] **Step 3: Extend `_on_axis_changed`**

```python
    def _on_axis_changed(self, index):
        self.working_axis = self.axis_selector.itemData(index)
        self._rebuild_spectra()
        self.cut_controller.clear()
        self.fit_controller.reset_marks()
        self._plot_data()
```

`FitModeController.reset_marks()` already exists (referenced in `fit_mode.py`'s own `clear()` method per the earlier research — it resets in-progress b/r/p marks without touching the fits LIST; since `_rebuild_spectra()` just above already replaced `self.spectra` with a fresh, empty-`.fits` wrapper, there is nothing further needed to clear committed fits — the old spectrum object, `.fits` and all, is simply discarded).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "fix: fully reset fit state (not just gate marks) when switching projection axis"
```

---

### Task 9: Cross-window fit/integrate parity tests

**Context:** The design spec's testing approach calls for proving the duck-typed reuse produces identical results, not just "doesn't crash." Fit and integrate the same synthetic data through both `MainWindow` and `MatrixPanel` and confirm matching output.

**Files:**
- Test: `tests/test_matrix_panel.py`

- [ ] **Step 1: Write and verify these tests pass immediately** (this task adds only tests — no new production code; if any of these fail, it indicates a real behavioral divergence introduced by an earlier task, which must be fixed before proceeding, not papered over here)

```python
def test_matrix_panel_and_main_window_integrate_identically(qapp):
    """Loads the identical numpy array as an ordinary spectrum in
    MainWindow and as a matrix projection in MatrixPanel, integrates
    the same region on both, and confirms matching gross/background/net
    -- proving the duck-typed FitModeController reuse produces
    identical results, not just that it runs without crashing."""
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    data = panel.spectra[0].data

    from spectrum import LoadedSpectrum
    spectrum = LoadedSpectrum("synthetic.spe", data, color="tab:blue")
    spectrum.active = True
    main_window.spectra = [spectrum]

    left_bg, right_bg, fit_region = (50.0, 70.0), (250.0, 270.0), (100.0, 200.0)

    for target in (main_window, panel):
        target.fit_controller._held_key = "b"
        _click(target, left_bg[0]); _click(target, left_bg[1])
        _click(target, right_bg[0]); _click(target, right_bg[1])
        target.fit_controller._held_key = "r"
        _click(target, fit_region[0]); _click(target, fit_region[1])
        target.fit_controller._held_key = None
        target.fit_controller.run_integration()

    mw_result = main_window.spectra[0].fits[-1]
    panel_result = panel.spectra[0].fits[-1]
    assert panel_result.gross_area == pytest.approx(mw_result.gross_area)
    assert panel_result.background_area == pytest.approx(mw_result.background_area)
    assert panel_result.net_area == pytest.approx(mw_result.net_area)
```

Note: `_click(target, x)` needs to work against both `main_window` and `panel` — check the existing `_click` helper's signature in `tests/test_matrix_panel.py` (it currently takes `(panel, xdata, ydata=10.0)` and reads `panel.axes`/`panel.canvas` internally). Since both `MainWindow` and `MatrixPanel` expose `.axes`/`.canvas` under the same names, the existing helper should already work unmodified against either — verify this is true rather than assuming; adjust the helper's parameter name from `panel` to something neutral like `target` if needed for clarity (a pure rename, no behavior change, safe either way).

- [ ] **Step 2: Run and confirm**

Run: `pytest tests/test_matrix_panel.py -k identically -v`
Expected: PASS. If it fails, diagnose and fix the ROOT CAUSE in whichever earlier task's code introduced the divergence — do not weaken this test to make it pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_matrix_panel.py
git commit -m "test: prove matrix panel and main window integrate/fit identically"
```

---

### Task 10: HowTo documentation

**Files:**
- Modify: `help_content.py`
- Test: `tests/test_help_content.py`

- [ ] **Step 1: Read the current section 11 text**

Read `help_content.py`'s `build_howto_html()`, specifically the `<h3>11. Matrix analysis</h3>` section (added in the prior matrix-analysis plan) and the shortcuts tables near the top of the function (`<h3>File menu</h3>` etc.) to match established formatting exactly before writing new content.

- [ ] **Step 2: Write the failing test**

```python
def test_howto_html_documents_matrix_panel_fit_integrate_calibrate():
    html = build_howto_html()
    assert "hold <kbd>G</kbd>" in html or "<kbd>G</kbd>" in html
    assert "<kbd>Ctrl+F</kbd>" in html
    assert "<kbd>Ctrl+I</kbd>" in html
    assert "<kbd>Ctrl+L</kbd>" in html
    assert "<kbd>Ctrl+=</kbd>" in html
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_help_content.py -k matrix_panel_fit -v`
Expected: FAIL — section 11 doesn't mention any of these yet.

- [ ] **Step 4: Rewrite section 11**

Replace the existing `<h3>11. Matrix analysis</h3>` paragraph with an expanded version documenting the new capability. Write the actual HTML matching this page's established prose style (see the existing section 11 text and section 9 "Fitting" for tone/format to match) — cover: hold-C for cut region, hold-G for background region (renamed from B, since B/R/P now mean the same thing as everywhere else in this app), Ctrl+F/Ctrl+C/Ctrl+I/Ctrl+B for fit/clear/integrate/preview-background exactly as in the main window, Ctrl+L for Calibrate (shared with the main window -- calibrating here also affects the main window's own display and vice versa), Ctrl+=/Ctrl+-/Ctrl+0 for zoom, and that switching between X/Y projection clears any in-progress marks and fits (different data). Also add a new shortcuts table (matching the File/View/Fit menu tables' existing `<table>`/`<tr>`/`<kbd>` format) specifically for the matrix panel's own window-scoped shortcuts, since they don't belong to any of the main window's menus.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_help_content.py -v`
Expected: PASS, full file.

- [ ] **Step 6: Commit**

```bash
git add help_content.py tests/test_help_content.py
git commit -m "docs: document matrix panel fit/integrate/calibrate/zoom in HowTo"
```

---

### Task 11: Knowledge Database — 2D spectra concepts

**Files:**
- Modify: `help_figures.py`
- Modify: `help_content.py`
- Test: `tests/test_help_content.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_knowledge_database_explains_2d_matrix_concepts():
    html = build_knowledge_database_html()
    assert "projection" in html.lower()
    assert "cut" in html.lower() or "gate" in html.lower()
    assert "coincidence" in html.lower()
```

Add this to `tests/test_help_content.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_help_content.py -k 2d_matrix -v`
Expected: FAIL — no such content exists in the Knowledge Database page yet.

- [ ] **Step 3: Add a new figure to `help_figures.py`**

Follow the exact pattern `anatomy_of_a_fit_figure` (`help_figures.py:30-70+`, already read in full during this plan's research) establishes: build a synthetic schematic with matplotlib, return PNG bytes via `_figure_to_png_bytes(fig)`. Add:

```python
def matrix_projection_cut_figure():
    """A small schematic 2D coincidence matrix with a cut (red) and
    background (green) band marked on one axis, alongside the resulting
    1D projection -- the reference figure the Knowledge Database's 2D
    spectra section points back to."""
    rng = np.random.default_rng(1)
    size = 60
    matrix = rng.poisson(3.0, size=(size, size)).astype(float)
    # A diagonal ridge of coincidence counts, the kind of structure a
    # real gamma-gamma matrix shows for genuinely correlated peaks.
    for i in range(size):
        j = min(size - 1, i + 5)
        matrix[i, j] += 40 * np.exp(-((np.arange(size) - j) ** 2) / 8.0)

    fig = Figure(figsize=(8.0, 4.2), dpi=110)
    ax_matrix = fig.add_subplot(121)
    ax_matrix.imshow(matrix, origin="lower", cmap="viridis", aspect="auto")
    cut_lo, cut_hi = 20, 30
    bg_lo, bg_hi = 40, 46
    ax_matrix.axvspan(cut_lo, cut_hi, color=_REGION_FIT_COLOR, alpha=0.35)
    ax_matrix.axvspan(bg_lo, bg_hi, color=_REGION_BG_COLOR, alpha=0.35)
    ax_matrix.set_xlabel("X channel")
    ax_matrix.set_ylabel("Y channel")
    ax_matrix.set_title("2D matrix")

    projection = matrix.sum(axis=0)
    ax_proj = fig.add_subplot(122)
    ax_proj.step(np.arange(size), projection, where="mid", color=_DATA_COLOR, linewidth=1.0)
    ax_proj.axvspan(cut_lo, cut_hi, color=_REGION_FIT_COLOR, alpha=0.25)
    ax_proj.axvspan(bg_lo, bg_hi, color=_REGION_BG_COLOR, alpha=0.25)
    ax_proj.set_xlabel("X channel")
    ax_proj.set_ylabel("Counts")
    ax_proj.set_title("X projection")

    fig.text(
        0.5, 0.01,
        "orange = cut region (the gate)   green = background region -- "
        "summing along Y gives the projection shown on the right",
        ha="center", va="bottom", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _figure_to_png_bytes(fig)
```

Check whether `import numpy as np` is already present at the top of `help_figures.py` (it very likely is, given `anatomy_of_a_fit_figure` uses `np.linspace`/`np.random`/`np.clip` already) — add it only if genuinely missing.

- [ ] **Step 4: Add a new Knowledge Database section referencing the figure**

In `help_content.py`, add `matrix_projection_cut_figure` to the existing `from help_figures import (...)` block (alphabetically). Read `build_knowledge_database_html()`'s existing structure (each section pairs a `<h3>`/prose block with an `<img src="{_embed_png(some_figure())}">` tag, matching `anatomy_of_a_fit_figure`'s own usage elsewhere in this function) and add a new section, placed logically among the existing conceptual sections (before the closing `"""`/`return _page(...)`), covering: what a 2D coincidence matrix is (two detectors, correlated events, X/Y are each detector's own channel axis), what a projection is (summing all counts along one axis, collapsing 2D into 1D), what a cut/gate is (restricting the sum to a narrow band on one axis, viewing the correlated distribution on the other -- this is how gamma-gamma coincidence spectroscopy isolates a specific decay cascade), and why background subtraction matters (random, uncorrelated coincidences contribute counts too; subtracting a nearby background band removes them, gate-width-weighted). Reference the new figure via `_embed_png(matrix_projection_cut_figure())`, matching the exact `<img>` tag pattern the other KB figures already use.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_help_content.py -v`
Expected: PASS, full file.

- [ ] **Step 6: Commit**

```bash
git add help_figures.py help_content.py tests/test_help_content.py
git commit -m "docs: explain 2D matrix/projection/cut concepts in Knowledge Database"
```

---

### Task 12: Windows verification

**Files:** None (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: PASS, full suite (aside from any already-known, pre-existing, unrelated failure — check current project memory/recent session history for whether one exists before treating any failure as new).

- [ ] **Step 2: Build the Windows package**

Run `packaging/windows/build.ps1` (do not redirect stderr via `2>&1`).

- [ ] **Step 3: Smoke-test the built exe**

Launch `dist/SpectraTools.exe`. Find the real worker process (the launched PID is the onefile bootloader, not the app) via `Get-CimInstance Win32_Process -Filter "ParentProcessId = <launched PID>"`, then confirm its window via direct Win32 API calls (`EnumWindows` filtered to its PID, `GetWindowRect`/`IsWindowVisible`). Kill both PIDs explicitly afterward.

- [ ] **Step 4: Manually exercise the real feature**

Using the running app: File > Open Matrix..., mark a cut region (hold-C) and a background region (hold-G, not B), confirm the dashed pending-click preview line appears after the first click of each pair, confirm zoom (Ctrl+=/Ctrl+-/Ctrl+0 and scroll-wheel) works, mark a fit region (hold-R) and peak (hold-P) directly on the projection and fit it (Ctrl+F), confirm the Fit Results and Fit Parameters docks appear and populate, integrate a region (Ctrl+I), open Calibrate (Ctrl+L or the button) and set a calibration, confirm both the matrix panel's AND the main window's spectra now display in keV, switch between X/Y projection and confirm fits/marks clear, activate a cut and confirm the resulting spectrum still works normally in the main window.

- [ ] **Step 5: Report**

Report pass/fail of Steps 1-4 plainly, including any issues found during manual exercise.

---

### Task 13: Linux verification

**Files:** None (verification only).

- [ ] **Step 1: Run the full test suite under WSL**

Using a real script file (not an inline `bash -c` one-liner), run `pytest -q` inside Ubuntu-24.04.

- [ ] **Step 2: Build both packages**

Run `packaging/linux/build.sh` (RPM, built on AlmaLinux-8) and `packaging/linux/build_deb.sh` (DEB, Ubuntu-24.04).

- [ ] **Step 3: Smoke-test**

Install using an **absolute path** to each built package (`dnf reinstall`/`apt-get install --reinstall --allow-downgrades`, unconditionally — do not rely on `install` alone, which is a silent no-op at an unchanged version). Smoke-test the RPM on **AlmaLinux-10** (never AlmaLinux-8) and the DEB on Ubuntu-24.04. Confirm the installed binary's checksum matches the freshly-built `dist-onedir/.../SpectraTools` binary before trusting either result. Launch under `timeout -k 1 <seconds> spectratools`, treat exit 124 as the primary "it's alive" signal -- but if `timeout` gives an unexplained non-124 result on either distro, don't assume a crash: cross-check with a `nohup ... & ; sleep N; kill -0 $PID` polling script before concluding a real regression (this has been a false alarm on both AlmaLinux-10 and Ubuntu-24.04 before, unrelated to actual app health).

- [ ] **Step 4: Report**

Report pass/fail of Steps 1-3 plainly. Full manual feature exercise (per Task 12 Step 4) is not required on both platforms if Windows manual verification already passed, but flag anything that looks Linux-specific if noticed.

---

## Self-Review Notes

- **Spec coverage:** every Resolved Decision in the design spec has a task — reuse strategy (duck-typed, Tasks 2/3/5/7), B→G rename (Task 4), calibration sharing (Task 3), axis-switch reset (Task 8), no separate Spectra dock (never introduced, consistent with `self.spectra` always being a 1-item list), scope boundary (no Multiply/Rebin/Normalize/etc. added anywhere), histogram display (Task 1), pending-click preview (Task 4), HowTo (Task 10), Knowledge Database (Task 11), cross-window parity testing (Task 9).
- **Dependency ordering verified:** `self.spectra` (Task 2) precedes zoom (Task 6) and fit wiring (Task 7), both of which read it. Calibration (Task 3) precedes gate-marking calibration-awareness (Task 4), zoom (Task 6), and `_plot_data` (Task 5), all of which use `channel_to_display`/`display_to_channel`. `_plot_data` (Task 5) precedes fit wiring (Task 7), which calls it via `FitModeController`. Fit wiring (Task 7) precedes the axis-switch fit-reset (Task 8) and the cross-window parity tests (Task 9), both of which need `fit_controller` to exist.
- **Incremental-build seams are explicit, not hidden:** `_apply_calibration_change` is deliberately written three times (Tasks 3, 5, 7), each version complete and correct for what exists at that point — flagged explicitly in Task 3's context note so a future reader doesn't mistake this for an oversight.
- **Placeholder scan:** no TBD/TODO. The one intentionally-provisional piece (`_update_fit_mode_availability`'s `pass` stub and `_plot_data`'s `hasattr` guard in Task 5) is explicitly labeled as provisional, with the exact task (7) that removes it named.
- **Type/name consistency:** `_held_key` values are `"cut"`/`"gate_bg"` throughout `MatrixCutController` after Task 4 (never `"bg"` again — Task 4 Step 1 explicitly calls out updating every existing test call site). `channel_to_display`/`display_to_channel`/`_calibration`/`_calibration_active`/`_plot_data`/`spectra`/`fit_controller` names match `main_window.py`'s own names exactly everywhere, by design (that's what makes the duck-typed reuse work) — verified against the Task 12 research report's exact attribute list, not reconstructed from memory.
- **Scope discipline:** no Multiply/Rebin/Normalize/Add-Subtract/Save/Close-Spectrum added to `MatrixPanel` anywhere in this plan. No log-scale toggle added (explicitly noted as out of scope in Tasks 5 and 6, matching the design spec). No second, independent calibration introduced — `_calibration`/`_calibration_active` are properties, not stored state, by construction (Task 3), so there is structurally no way for them to diverge from `main_window`'s.
