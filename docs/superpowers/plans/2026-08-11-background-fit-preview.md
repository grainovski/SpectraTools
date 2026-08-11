# Background Line Display Fix + Ctrl+B Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the background line's drawn extent (currently clipped to the fit region; should span the two background regions it's actually calculated from), and add a `Ctrl+B` action that previews the background fit using only the two background regions, independent of the fit region or peaks.

**Architecture:** `peak_fit.py`'s existing private background computation becomes public so `fit_mode.py` can call it directly for the preview (the alternative, `fit_peaks()`/`integrate_region()`, both require a `fit_region` the preview doesn't have). `fit_mode.py` gets a new `FitModeState` boolean flag, a new blocked-reason pair matching the existing `fit_blocked_reason()`/`integrate_blocked_reason()` convention, a new controller toggle method, and a new drawing block in the existing `_redraw_progress()`. `main_window.py` gets a new always-enabled `QAction` bound to `Ctrl+B`, following the exact pattern already used for `Ctrl+F`/`Ctrl+C`/`Ctrl+I` there (not `fit_mode.py`'s own action registrations, which are for a different, results-panel-specific purpose).

**Tech Stack:** Python, PySide6, matplotlib, pytest.

---

### Task 1: `peak_fit.py` — make `_compute_background` public

**Files:**
- Modify: `peak_fit.py`
- Test: `tests/test_peak_fit.py`

- [ ] **Step 1: Rename in `peak_fit.py`**

Change:
```python
def _compute_background(x, y, left_bg_region, right_bg_region):
```
to:
```python
def compute_background(x, y, left_bg_region, right_bg_region):
```

Update its one internal call site, inside `fit_peaks()`:
```python
    slope, intercept = _compute_background(x, y, left_bg_region, right_bg_region)
```
to:
```python
    slope, intercept = compute_background(x, y, left_bg_region, right_bg_region)
```

(`integrate_region()` does NOT call this function — it has its own, separate flat-density background computation. Do not touch `integrate_region()`.)

- [ ] **Step 2: Update `tests/test_peak_fit.py`**

Update the import line:
```python
from peak_fit import FitError, FitResult, PeakResult, _compute_background, hypermet_left_tail
```
to:
```python
from peak_fit import FitError, FitResult, PeakResult, compute_background, hypermet_left_tail
```

Update all four call sites (in `test_compute_background_flat`, `test_compute_background_sloped`, `test_compute_background_rejects_empty_region`, `test_compute_background_rejects_identical_mean_x`) from `_compute_background(...)` to `compute_background(...)`. The test function names themselves (already `test_compute_background_*`, not `test__compute_background_*`) don't need to change.

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS, all tests including the four renamed call sites.

Run: `pytest -q` (full suite, sanity check nothing else references the old private name)
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "refactor: make compute_background public for the new Ctrl+B preview"
```

---

### Task 2: `fit_mode.py` — fix the background line's drawn extent

**Files:**
- Modify: `fit_mode.py`
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write/update the failing tests**

**Update** the existing test that currently asserts the *buggy* behavior. Find `test_draw_committed_fits_integration_result_draws_at_calibrated_x` in `tests/test_fit_mode_ui.py` (its `IntegrationResult` fixture uses `left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0), fit_region=(85.0, 115.0)` and `Calibration(kind="linear", a=10.0, b=0.5)`). Replace:
```python
    bg_line = main_window.axes.lines[-1]
    xdata = bg_line.get_xdata()
    assert xdata[0] == pytest.approx(52.5)  # channel 85 -> keV 10+0.5*85
    assert xdata[1] == pytest.approx(67.5)  # channel 115 -> keV 10+0.5*115
```
with:
```python
    bg_line = main_window.axes.lines[-1]
    xdata = bg_line.get_xdata()
    assert xdata[0] == pytest.approx(45.0)  # channel 70 (left_bg_region[0]) -> keV 10+0.5*70
    assert xdata[1] == pytest.approx(75.0)  # channel 130 (right_bg_region[1]) -> keV 10+0.5*130
```

**Add** a new test for the Fit case (no existing test checks the Fit branch's background-line endpoints at all today — `background_slope=0.0` in every existing Fit-result fixture in this file makes the endpoint bug invisible, since a flat line's Y value doesn't depend on where its X endpoints are). Add near the other `test_draw_committed_fits_*` tests:
```python
def test_draw_committed_fits_fit_background_line_spans_bg_regions(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.1, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )

    main_window.axes.clear()
    main_window.fit_controller.draw_committed_fits(spectrum)

    # The background dashed line (linewidth 1, distinct from the fit
    # curve's 1.5 and each peak component's 0.75) must span the
    # background regions' own outer bounds, not the fit region --
    # background_slope is nonzero here specifically so a wrong span
    # would move both Y endpoints too, not just X.
    bg_lines = [line for line in main_window.axes.lines if line.get_linewidth() == 1]
    assert len(bg_lines) == 1
    xdata = bg_lines[0].get_xdata()
    ydata = bg_lines[0].get_ydata()
    assert xdata[0] == pytest.approx(70.0)   # left_bg_region[0]
    assert xdata[1] == pytest.approx(130.0)  # right_bg_region[1]
    assert ydata[0] == pytest.approx(0.1 * 70.0 + 20.0)
    assert ydata[1] == pytest.approx(0.1 * 130.0 + 20.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -k "draw_committed_fits" -v`
Expected: `test_draw_committed_fits_integration_result_draws_at_calibrated_x` FAILs (still asserts old 52.5/67.5 values against new 45.0/75.0 expectation — wait, this step edits the test to the NEW expected values first, so at this point it should fail against the CURRENT, unfixed `draw_committed_fits()`, which still produces 52.5/67.5). The new `test_draw_committed_fits_fit_background_line_spans_bg_regions` FAILs too (current code draws at fit_region bounds 85/115, not bg-region bounds 70/130).

- [ ] **Step 3: Fix `draw_committed_fits()` in `fit_mode.py`**

In the Integration branch, replace:
```python
            if isinstance(result, IntegrationResult):
                if result.has_background:
                    lo, hi = result.fit_region
                    axes.plot(
                        [to_display(lo), to_display(hi)],
                        [result.background_density, result.background_density],
                        color=bg_line_color, linestyle="--", linewidth=1,
                    )
```
with:
```python
            if isinstance(result, IntegrationResult):
                if result.has_background:
                    bg_line_lo, bg_line_hi = result.left_bg_region[0], result.right_bg_region[1]
                    axes.plot(
                        [to_display(bg_line_lo), to_display(bg_line_hi)],
                        [result.background_density, result.background_density],
                        color=bg_line_color, linestyle="--", linewidth=1,
                    )
```

In the Fit branch (further down the same function), replace:
```python
            lo, hi = result.fit_region
            background_lo = result.background_slope * lo + result.background_intercept
            background_hi = result.background_slope * hi + result.background_intercept
            axes.plot(
                [to_display(lo), to_display(hi)], [background_lo, background_hi],
                color=bg_line_color, linestyle="--", linewidth=1,
            )
```
with:
```python
            lo, hi = result.fit_region
            bg_line_lo, bg_line_hi = result.left_bg_region[0], result.right_bg_region[1]
            background_lo = result.background_slope * bg_line_lo + result.background_intercept
            background_hi = result.background_slope * bg_line_hi + result.background_intercept
            axes.plot(
                [to_display(bg_line_lo), to_display(bg_line_hi)], [background_lo, background_hi],
                color=bg_line_color, linestyle="--", linewidth=1,
            )
```
`lo, hi = result.fit_region` stays exactly where it is — it's still used immediately afterward for `x_dense = np.linspace(lo, hi, 200)`, the peak curve's own span, which must stay clipped to the fit region.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS, full file (confirms the two edited/new tests pass AND nothing else in this large file broke).

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "fix: draw background line across background regions, not fit region"
```

---

### Task 3: `fit_mode.py` — `FitModeState` preview flag and blocked-reason pair

**Files:**
- Modify: `fit_mode.py`
- Test: `tests/test_fit_mode.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fit_mode.py` (matching the file's existing direct-`FitModeState`, no-Qt style):
```python
def test_background_blocked_reason_when_no_bg_regions():
    state = FitModeState()
    assert (
        state.background_blocked_reason()
        == "Mark two background regions (hold B and click twice per region) before previewing the background"
    )


def test_background_blocked_reason_when_one_bg_region():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    assert (
        state.background_blocked_reason()
        == "Mark two background regions (hold B and click twice per region) before previewing the background"
    )


def test_background_blocked_reason_is_none_when_two_bg_regions():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    assert state.background_blocked_reason() is None
    assert state.ready_to_preview_background() is True


def test_add_bg_click_resets_background_preview_flag_when_regions_change():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.show_background_preview = True

    # A third completed pair evicts the oldest region -- a preview
    # computed from the old regions is now stale and must clear.
    state.add_bg_click(200)
    state.add_bg_click(210)

    assert state.show_background_preview is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode.py -k background -v`
Expected: FAIL with `AttributeError: 'FitModeState' object has no attribute 'background_blocked_reason'` (and similarly for `ready_to_preview_background`/`show_background_preview`).

- [ ] **Step 3: Implement in `fit_mode.py`**

In `FitModeState.reset()`, add the new field:
```python
    def reset(self):
        self.bg_regions = []
        self.pending_bg_click = None
        self.fit_region = None
        self.pending_fit_click = None
        self.peak_positions = []
        self.show_background_preview = False
```

In `FitModeState.add_bg_click()`, invalidate a stale preview whenever the background regions actually change. Replace:
```python
        region = (min(self.pending_bg_click, x), max(self.pending_bg_click, x))
        self.pending_bg_click = None
        self.bg_regions.append(region)
        if len(self.bg_regions) > BG_REGION_CAP:
            self.bg_regions.pop(0)
        return region
```
with:
```python
        region = (min(self.pending_bg_click, x), max(self.pending_bg_click, x))
        self.pending_bg_click = None
        self.bg_regions.append(region)
        if len(self.bg_regions) > BG_REGION_CAP:
            self.bg_regions.pop(0)
        self.show_background_preview = False
        return region
```

Add two new methods to `FitModeState`, next to `integrate_blocked_reason()`:
```python
    def background_blocked_reason(self):
        """Same shape as fit_blocked_reason()/integrate_blocked_reason()
        -- the background preview needs only the two background
        regions, independent of fit_region or peaks."""
        if len(self.bg_regions) != BG_REGION_CAP:
            return "Mark two background regions (hold B and click twice per region) before previewing the background"
        return None

    def ready_to_preview_background(self):
        return self.background_blocked_reason() is None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode.py
git commit -m "feat: add background-preview readiness gating to FitModeState"
```

---

### Task 4: `fit_mode.py` + `main_window.py` — Ctrl+B preview action

**Files:**
- Modify: `fit_mode.py`, `main_window.py`
- Test: `tests/test_fit_mode_ui.py`

**Depends on Task 1** (imports `compute_background`) **and Task 3** (`FitModeState`'s new flag/methods).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fit_mode_ui.py`:
```python
def test_toggle_background_preview_draws_line_spanning_bg_regions(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    lines_before = len(main_window.axes.lines)

    main_window.fit_controller.toggle_background_preview()

    assert len(main_window.axes.lines) == lines_before + 1
    assert main_window.statusBar().currentMessage() == ""
    preview_line = main_window.axes.lines[-1]
    xdata = preview_line.get_xdata()
    assert xdata[0] == pytest.approx(70.0)
    assert xdata[1] == pytest.approx(130.0)


def test_toggle_background_preview_hides_on_second_press(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)

    main_window.fit_controller.toggle_background_preview()
    lines_with_preview = len(main_window.axes.lines)

    main_window.fit_controller.toggle_background_preview()

    assert len(main_window.axes.lines) == lines_with_preview - 1


def test_toggle_background_preview_shows_a_status_message_when_marks_are_incomplete(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    # only one background region marked

    main_window.fit_controller.toggle_background_preview()

    assert (
        main_window.statusBar().currentMessage()
        == "Mark two background regions (hold B and click twice per region) before previewing the background"
    )
    assert main_window.fit_controller.state.show_background_preview is False


def test_background_preview_button_has_ctrl_b_shortcut_and_no_toolbar(qapp):
    main_window = MainWindow()

    assert main_window.background_preview_button.shortcut().toString() == "Ctrl+B"
    assert main_window.background_preview_button.isEnabled() is True
    assert main_window.background_preview_button in main_window.actions()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -k background_preview -v`
Expected: FAIL — `toggle_background_preview` and `background_preview_button` don't exist yet.

- [ ] **Step 3: Implement**

In `fit_mode.py`, add `compute_background` to the existing `from peak_fit import (...)` block:
```python
from peak_fit import (
    FWHM_FACTOR, FitError, IntegrationResult, compute_background, fit_peaks,
    fit_result_values_by_name, hypermet_left_tail, integrate_region, parameter_names,
)
```

In `fit_mode.py`'s `FitModeController`, add a new method (near `run_fit`/`run_integration`):
```python
    def toggle_background_preview(self):
        if not self.state.ready_to_preview_background():
            self._show_status_message(self.state.background_blocked_reason(), 5000)
            return
        self.state.show_background_preview = not self.state.show_background_preview
        self._redraw_progress()
```

In `fit_mode.py`'s `_redraw_progress()`, add a new block after the existing peak-position loop (right before `self.main_window.canvas.draw_idle()`):
```python
        if state.show_background_preview and len(state.bg_regions) == BG_REGION_CAP:
            active = next((s for s in self.main_window.spectra if s.active), None)
            if active is not None:
                left, right = state.ordered_bg_regions()
                x = np.arange(len(active.data), dtype=float)
                slope, intercept = compute_background(x, active.data, left, right)
                bg_lo_x, bg_hi_x = left[0], right[1]
                bg_lo_y = slope * bg_lo_x + intercept
                bg_hi_y = slope * bg_hi_x + intercept
                line = axes.plot(
                    [to_display(bg_lo_x), to_display(bg_hi_x)], [bg_lo_y, bg_hi_y],
                    color="black", linestyle="--", linewidth=1.2,
                )[0]
                self._progress_artists.append(line)
```

In `main_window.py`, add a new action after the existing `integrate_button` setup (around line 1064):
```python
        self.background_preview_button = QAction("Preview Background Fit", self)
        self.background_preview_button.setShortcut("Ctrl+B")
        self.background_preview_button.triggered.connect(self.fit_controller.toggle_background_preview)
        self.addAction(self.background_preview_button)
```
No `setEnabled(False)` call — matches `clear_fit_button`'s always-enabled pattern, not `fit_button`/`integrate_button`'s dynamically-gated one. `toggle_background_preview()`'s own internal blocked-reason check is what handles the "not ready" case, exactly like `clear()` needs no external gating either.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS, full file.

Run: `pytest -q` (full suite)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py main_window.py tests/test_fit_mode_ui.py
git commit -m "feat: add Ctrl+B background-fit preview, independent of fit region/peaks"
```

---

### Task 5: `help_content.py` — HowTo documentation

**Files:**
- Modify: `help_content.py`
- Test: `tests/test_help_content.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_help_content.py`, add `"Ctrl+B"` to `test_howto_html_contains_every_shortcut()`'s list (insert alongside the other Fitting-section shortcuts, e.g. right after `"Ctrl+F"`):
```python
        "Ctrl+F", "Ctrl+B", "Ctrl+C", "Ctrl+Shift+C", "Ctrl+E", "Ctrl+I",
```
(replacing the existing `"Ctrl+F", "Ctrl+C", "Ctrl+Shift+C", "Ctrl+E", "Ctrl+I",` line)

Append a new test:
```python
def test_howto_html_documents_ctrl_b_preview():
    html = build_howto_html()
    assert "Preview the background fit from the two background regions alone" in html
    assert "no fit region or peaks needed" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_help_content.py -k "shortcut or ctrl_b" -v`
Expected: FAIL — `Ctrl+B` isn't in the HTML yet.

- [ ] **Step 3: Update `build_howto_html()` in `help_content.py`**

Add a new row to the Fitting shortcuts table, after the `Ctrl+F` row:
```html
<tr><td><kbd>Ctrl+F</kbd></td><td>Fit</td></tr>
<tr><td><kbd>Ctrl+B</kbd></td><td>Preview the background fit from the two background regions alone (no fit region or peaks needed); press again to hide</td></tr>
```

Add a new paragraph to the "7. Performing a fit" section, after the existing `</ol>` and before the "See the Knowledge Database page..." paragraph:
```html
</ol>
<p>Once both background regions are marked, <kbd>Ctrl+B</kbd> previews
just the background line -- no fit region or peaks needed -- useful
for sanity-checking the background before marking the rest. It's a
preview only: nothing is added to Fit Results, logged, or exported.
Press <kbd>Ctrl+B</kbd> again to hide it.</p>
<p>See the Knowledge Database page for exactly what's being computed
here, and what each fit parameter means.</p>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_help_content.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add help_content.py tests/test_help_content.py
git commit -m "docs: document Ctrl+B background preview in HowTo"
```

---

### Task 6: Windows verification

**Files:** None (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: PASS, full suite.

- [ ] **Step 2: Build the Windows package**

Run `packaging/windows/build.ps1` (do not redirect stderr via `2>&1` — PowerShell 5.1 wraps a native command's stderr into a terminating-looking error even on real success; let stderr surface on its own).

- [ ] **Step 3: Smoke-test the built exe**

Launch `dist/SpectraTools.exe`. Since this is a `--onefile` build, the launched PID is the bootloader, not the real app — find the actual worker via `Get-CimInstance Win32_Process -Filter "ParentProcessId = <launched PID>"`, then confirm that child's window via direct Win32 API calls (`EnumWindows` filtered to its PID, then `GetWindowRect`/`IsWindowVisible`/`IsIconic`) rather than trusting `Get-Process`'s `MainWindowHandle` on the parent (which stays 0). Kill both the bootloader and child PIDs explicitly afterward.

- [ ] **Step 4: Report**

Report pass/fail of Steps 1-3 plainly.

---

### Task 7: Linux verification

**Files:** None (verification only).

- [ ] **Step 1: Run the full test suite under WSL**

Using a real script file (not an inline `bash -c` one-liner), run `pytest -q` inside the relevant WSL distro(s).

- [ ] **Step 2: Build both packages**

Run `packaging/linux/build.sh` (RPM) and `packaging/linux/build_deb.sh` (DEB).

- [ ] **Step 3: Smoke-test both packages**

Install using an **absolute path** to the built package file (a relative path can make `dnf install` silently misinterpret the argument as a repo package name instead of a local file). Before trusting the smoke test, confirm the installed binary's checksum matches the freshly-built `dist-onedir/.../SpectraTools` binary (catches a stale prior install left over from an earlier session). Launch under `timeout -k 1 <seconds> spectratools`, treat exit 124 as the only real "it's alive" signal.

- [ ] **Step 4: Report**

Report pass/fail of Steps 1-3 plainly.

---

## Self-Review Notes

- **Spec coverage**: display-extent fix (Task 2, both Fit and Integration branches), `compute_background` public rename (Task 1), `Ctrl+B` preview with no-status-on-success/blocked-message-on-failure/toggle-off/invalidate-on-region-change (Tasks 3-4), HowTo documentation (Task 5) — every requirement in the spec has a task.
- **Corrections made from the spec during planning** (re-verified against real source rather than trusting my own earlier spec text): `compute_background` has only ONE internal call site (`fit_peaks()`), not two as the spec said — `integrate_region()` has an entirely separate background computation. The new `Ctrl+B` action belongs in `main_window.py` (matching where `Ctrl+F`/`Ctrl+C`/`Ctrl+I` actually live), not in `fit_mode.py`'s `build_results_panel()` as the spec suggested — that spot is for results-panel-specific actions (Export/Clear All), a different purpose. Also found and included fixing an existing test (`test_draw_committed_fits_integration_result_draws_at_calibrated_x`) that currently asserts the *buggy* display behavior and would otherwise fail after Task 2's fix with no explanation.
- **Placeholder scan**: no TBD/TODO; every code block is complete, runnable code, not a description of code.
- **Type/name consistency**: `compute_background` (Task 1's new public name) is imported and called identically in Task 4's `_redraw_progress()` addition. `show_background_preview`, `background_blocked_reason()`, `ready_to_preview_background()` (Task 3) are the exact names used in Task 4's `toggle_background_preview()`. `background_preview_button` (Task 4's `main_window.py` addition) is the exact name used in Task 4's own shortcut test.
