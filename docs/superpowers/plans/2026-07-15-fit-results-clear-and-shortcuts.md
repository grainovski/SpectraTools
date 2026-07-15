# Fit Results Table, Clear Fix, and Keyboard Shortcuts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "Clear" visibly reset the plot (without deleting fit history), stop re-fits of the same marks from visually stacking on top of each other, give the Fit Results panel explicit per-peak fields (position/FWHM/volume) instead of a text blob, move the width-link/left-tail checkboxes into the Fit Parameters panel, and replace the "Fit"/"Clear" toolbar buttons with `Ctrl+F`/`Ctrl+C` shortcuts.

**Architecture:** `peak_fit.py`'s `FitResult` gains a `visible: bool = True` field — the same "stays in history, stops being drawn" pattern `LoadedSpectrum.visible` already uses. `fit_mode.py`'s `draw_committed_fits` skips hidden results; `run_fit()` auto-hides any earlier fit for the exact same marks before appending a fresh one (fixes stacking); `clear()` hides every fit for the active spectrum and forces a replot (fixes the visual-reset complaint) without touching `active.fits` itself. The Fit Results panel becomes a `QTableWidget` (`results_table`, one row per peak) instead of a `QListWidget` of text blobs. `independent_widths_action`/`left_tail_action` become `QCheckBox`es inside the Fit Parameters dock instead of toolbar `QAction`s. `fit_button`/`clear_fit_button` stay as `QAction`s (for `.trigger()`/`.isEnabled()` call sites) but move off the toolbar entirely, driven only by `Ctrl+F`/`Ctrl+C`.

**Tech Stack:** Python 3.13, PySide6/matplotlib, `pytest` with the existing `qapp` fixture.

---

## Before you start

Read `docs/superpowers/specs/2026-07-15-fit-results-clear-and-shortcuts-design.md` in full. Run the full suite once to confirm a clean baseline:

Run: `pytest -v` from the repo root (`C:\Users\RIG\Documents\Claude\PeakFinderFitting`).
Expected: all tests pass (123 at last count).

---

### Task 1: `FitResult.visible` + auto-hide a superseded same-region fit

**Files:**
- Modify: `peak_fit.py` (`FitResult` dataclass)
- Modify: `fit_mode.py` (`draw_committed_fits`, `run_fit`)
- Test: `tests/test_peak_fit.py`, `tests/test_fit_mode_ui.py`

Re-fitting identical marks under a changed checkbox (the workflow the
2026-07-14 "marks persist" change enables) currently draws every past
attempt for that region on top of the others forever. A `visible` flag on
`FitResult`, defaulted `True` and never read by `fit_peaks()` itself, lets
`fit_mode.py` hide superseded attempts without deleting them.

- [ ] **Step 1: Write the failing test for the default**

Add to `tests/test_peak_fit.py`, directly after `test_fit_result_fixed_params_defaults_to_empty_dict`:

```python
def test_fit_result_visible_defaults_to_true():
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
    assert result.visible is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_peak_fit.py -v -k visible_defaults_to_true`
Expected: FAIL (`TypeError: FitResult.__init__() got an unexpected keyword argument` is not raised since no new kwarg is passed here — expect `AttributeError: 'FitResult' object has no attribute 'visible'`)

- [ ] **Step 3: Add the field**

In `peak_fit.py`, change the `FitResult` dataclass's last line (`fixed_params: dict = field(default_factory=dict)`) so the full dataclass reads:

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_peak_fit.py -v -k visible_defaults_to_true`
Expected: PASS

- [ ] **Step 5: Write the failing UI-level tests**

Add to `tests/test_fit_mode_ui.py`, directly after `test_double_click_reloads_a_committed_fit_for_editing` (before `test_marking_order_is_free`):

```python
def test_refitting_same_marks_hides_the_earlier_result(qapp):
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
    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    assert spectrum.fits[0].visible is False
    assert spectrum.fits[1].visible is True


def test_refitting_a_different_region_does_not_hide_the_first(qapp):
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

    _held_key_click(main_window, "b", 20)
    _held_key_click(main_window, "b", 35)
    _held_key_click(main_window, "b", 165)
    _held_key_click(main_window, "b", 180)
    _held_key_click(main_window, "r", 35)
    _held_key_click(main_window, "r", 165)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    assert spectrum.fits[0].visible is True
    assert spectrum.fits[1].visible is True


def test_draw_committed_fits_skips_hidden_results(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1,
                    fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0,
                    amplitude=200.0, sigma=2.0,
                )
            ],
            visible=False,
        )
    )

    lines_before = len(main_window.axes.lines)
    patches_before = len(main_window.axes.patches)
    texts_before = len(main_window.axes.texts)
    main_window.fit_controller.draw_committed_fits(spectrum)

    assert len(main_window.axes.lines) == lines_before
    assert len(main_window.axes.patches) == patches_before
    assert len(main_window.axes.texts) == texts_before
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -v -k "hides_the_earlier_result or does_not_hide_the_first or skips_hidden_results"`
Expected: FAIL (`assert spectrum.fits[0].visible is False` fails since nothing sets it yet; the skip test fails since `draw_committed_fits` still draws the hidden result)

- [ ] **Step 7: Skip hidden results in `draw_committed_fits`**

In `fit_mode.py`, in `draw_committed_fits`, change:

```python
        for result in spectrum.fits:
            axes.axvspan(*result.left_bg_region, color="gray", alpha=0.15)
```

to:

```python
        for result in spectrum.fits:
            if not result.visible:
                continue
            axes.axvspan(*result.left_bg_region, color="gray", alpha=0.15)
```

- [ ] **Step 8: Auto-hide a superseded same-region fit in `run_fit`**

In `fit_mode.py`, in `run_fit()`, change:

```python
        except FitError as exc:
            self._show_status_message(f"Fit failed: {exc}", 5000)
            return
        active.fits.append(result)
```

to:

```python
        except FitError as exc:
            self._show_status_message(f"Fit failed: {exc}", 5000)
            return
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
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py tests/test_peak_fit.py -v`
Expected: PASS (all tests)

- [ ] **Step 10: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add peak_fit.py fit_mode.py tests/test_peak_fit.py tests/test_fit_mode_ui.py
git commit -m "feat: hide (not delete) a fit result when superseded by a re-fit of the same marks"
```

---

### Task 2: `Clear` visually resets the plot without deleting history

**Files:**
- Modify: `fit_mode.py` (`clear`)
- Test: `tests/test_fit_mode_ui.py`

Currently `clear()` only removes the in-progress marking overlays and never
triggers a replot, so a just-committed fit's curve/shading/annotation stay
fully visible — the plot looks unchanged, which reads as "Clear didn't
work" (and, per the design spec, is the most likely explanation for reports
that fitting seems impossible afterward, even though the marking state
underneath was already being reset correctly). This task makes Clear hide
every fit for the active spectrum (via the `visible` flag from Task 1) and
forces a redraw, while leaving `active.fits` — and therefore the Fit
Results panel — untouched.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_mode_ui.py`, directly after `test_clear_discards_in_progress_marks_without_committing`:

```python
def test_clear_hides_every_fit_for_the_active_spectrum_without_deleting_it(qapp):
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
    assert spectrum.fits[0].visible is True

    main_window.fit_controller.clear()

    assert len(spectrum.fits) == 1  # nothing deleted
    assert spectrum.fits[0].visible is False  # but no longer drawn


def test_clear_forces_a_replot_so_the_canvas_actually_goes_blank(qapp):
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

    lines_with_fit = len(main_window.axes.lines)
    patches_with_fit = len(main_window.axes.patches)

    main_window.fit_controller.clear()

    # Only the spectrum's own step line should remain -- no fit overlay,
    # no in-progress marking artists.
    assert len(main_window.axes.lines) == 1
    assert len(main_window.axes.patches) == 0
    assert len(main_window.axes.lines) < lines_with_fit
    assert len(main_window.axes.patches) < patches_with_fit


def test_clear_with_no_active_spectrum_does_not_crash(qapp):
    main_window = MainWindow()
    main_window.fit_controller.clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -v -k "clear_hides_every_fit or clear_forces_a_replot or clear_with_no_active"`
Expected: FAIL (`assert spectrum.fits[0].visible is False` fails since `clear()` doesn't touch `active.fits` yet; the replot test fails since the fit's curve/shading are still drawn, `len(main_window.axes.lines) == 1` is false)

- [ ] **Step 3: Update `clear()`**

In `fit_mode.py`, replace:

```python
    def clear(self):
        self._clear_progress()
        self.main_window._update_fit_mode_availability()
        self.main_window.canvas.draw_idle()
```

with:

```python
    def clear(self):
        self._clear_progress()
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is not None:
            for result in active.fits:
                result.visible = False
        self.main_window._plot_data(preserve_view=True)
```

(`_plot_data` already calls `_update_fit_mode_availability()` and
`update_results_list()` and redraws the canvas, so the explicit calls to
those that used to live in `clear()` are now redundant and removed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (all tests -- including `test_clear_progress_survives_an_intervening_full_replot`, which exercises the pre-existing stale-artist `.remove()` guard that `_clear_progress()` still needs even though `clear()` now always replots itself)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "fix: Clear visibly hides the active spectrum's fits and marks instead of leaving them drawn"
```

---

### Task 3: Fit Results panel becomes a per-peak table

**Files:**
- Modify: `fit_mode.py` (`build_results_panel`, `update_results_list`, `_on_results_context_menu`, `_on_result_double_clicked`)
- Test: `tests/test_fit_mode_ui.py`

Replaces the `QListWidget` of multi-line text blobs with a `QTableWidget`
(`results_table`), one row per peak, with explicit Position/FWHM/Volume
columns. `self._results_row_fit_index[row]` maps a table row back to its
index in `active.fits`, since multiple rows can belong to one fit.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_mode_ui.py`, directly after `test_marks_persist_after_a_successful_fit` (before `test_refitting_same_marks_with_changed_checkbox_appends_a_new_entry`):

```python
def test_results_table_shows_one_row_per_peak_with_explicit_columns(qapp):
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

    table = main_window.fit_controller.results_table
    assert table.rowCount() == 1
    assert table.item(0, 1).text() == "1"  # Peak column, 1-based
    assert table.item(0, 2).text().startswith("100.0")  # Position
    assert "±" in table.item(0, 2).text()
    assert "±" in table.item(0, 3).text()  # FWHM
    assert "±" in table.item(0, 4).text()  # Volume


def test_results_table_has_one_row_per_peak_across_a_multi_peak_fit(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 130.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                ),
                PeakResult(
                    position=120.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=800.0, area_err=40.0, amplitude=160.0, sigma=2.0,
                ),
            ],
        )
    )
    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    assert table.rowCount() == 2
    assert table.item(0, 1).text() == "1"
    assert table.item(1, 1).text() == "2"
    assert table.item(0, 0).text() == table.item(1, 0).text()  # same Fit cell


def test_results_table_dims_hidden_fit_rows(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 110.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                )
            ],
            visible=False,
        )
    )
    main_window.fit_controller.update_results_list()

    table = main_window.fit_controller.results_table
    default_color = QTableWidgetItem().foreground()
    assert table.item(0, 0).foreground() != default_color


def test_remove_fit_from_context_menu_removes_the_correct_fit_by_row(qapp, monkeypatch):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
            fit_region=(30.0, 50.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=40.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=500.0, area_err=25.0, amplitude=100.0, sigma=2.0,
                )
            ],
        )
    )
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
    assert table.rowCount() == 2
    second_row_item = table.item(1, 0)

    from PySide6.QtWidgets import QMenu
    remove_action_holder = {}

    def fake_exec(self, *args, **kwargs):
        return remove_action_holder["remove"]

    def fake_add_action(self, text):
        action = QMenu.addAction(self, text)
        if text == "Remove Fit":
            remove_action_holder["remove"] = action
        return action

    monkeypatch.setattr(QMenu, "addAction", fake_add_action)
    monkeypatch.setattr(QMenu, "exec", fake_exec)

    position = table.visualItemRect(second_row_item).center()
    main_window.fit_controller._on_results_context_menu(position)

    assert len(spectrum.fits) == 1
    assert spectrum.fits[0].fit_region == (30.0, 50.0)


def test_double_click_a_peak_row_reloads_its_parent_fit(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
            fit_region=(30.0, 50.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=40.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=500.0, area_err=25.0, amplitude=100.0, sigma=2.0,
                )
            ],
        )
    )
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

    main_window.fit_controller._on_result_double_clicked(table.item(1, 0))

    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
    assert main_window.fit_controller.state.peak_positions == pytest.approx([100.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -v -k "results_table or Remove_Fit or double_click_a_peak_row"`
Expected: FAIL with `AttributeError: 'FitModeController' object has no attribute 'results_table'`

- [ ] **Step 3: Update imports**

In `fit_mode.py`, replace:

```python
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QListWidget, QListWidgetItem, QMenu, QTableWidget,
    QTableWidgetItem, QToolBar,
)
```

with:

```python
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QMenu, QTableWidget, QTableWidgetItem, QToolBar,
)
```

(`QListWidget`/`QListWidgetItem` are no longer used anywhere in this file
once this task is done.)

- [ ] **Step 4: Rebuild `build_results_panel`**

In `fit_mode.py`, replace the body of `build_results_panel` from
`self.results_list = QListWidget()` through
`self.results_list.itemDoubleClicked.connect(self._on_result_double_clicked)`
with:

```python
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ["Fit", "Peak", "Position", "FWHM", "Volume"]
        )
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._on_results_context_menu)
        self.results_table.itemDoubleClicked.connect(self._on_result_double_clicked)
```

and change the next line (`self.results_dock.setWidget(self.results_list)`) to:

```python
        self.results_dock.setWidget(self.results_table)
```

- [ ] **Step 5: Rewrite `update_results_list`**

Replace the entire method with:

```python
    def update_results_list(self):
        self.results_table.setRowCount(0)
        self._results_row_fit_index = []
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        for fit_index, result in enumerate(active.fits):
            fit_label = f"{fit_index + 1} [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]"
            tooltip_lines = []
            if not result.link_widths:
                tooltip_lines.append("independent widths")
            if result.tail_fraction is not None:
                tooltip_lines.append(
                    f"left tail: r={result.tail_fraction:.2f}"
                    f"±{result.tail_fraction_err:.2f}, "
                    f"β={result.tail_beta:.1f}±{result.tail_beta_err:.1f} "
                    f"(volume excludes tail)"
                )
            tooltip = "\n".join(tooltip_lines)

            for peak_index, peak in enumerate(result.peaks):
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self._results_row_fit_index.append(fit_index)

                values = [
                    fit_label,
                    str(peak_index + 1),
                    f"{peak.position:.2f} ± {peak.position_err:.2f}",
                    f"{peak.fwhm:.2f} ± {peak.fwhm_err:.2f}",
                    f"{peak.area:.1f} ± {peak.area_err:.1f}",
                ]
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    if col == 0:
                        item.setToolTip(tooltip)
                    if not result.visible:
                        item.setForeground(QColor("gray"))
                    self.results_table.setItem(row, col, item)
```

- [ ] **Step 6: Update `_on_results_context_menu`**

Replace the method with:

```python
    def _on_results_context_menu(self, position):
        mw = self.main_window
        item = self.results_table.itemAt(position)
        menu = QMenu(mw)
        remove_action = menu.addAction("Remove Fit") if item is not None else None
        clear_action = menu.addAction("Clear All Fits")
        chosen = menu.exec(self.results_table.viewport().mapToGlobal(position))
        active = next((s for s in mw.spectra if s.active), None)
        if active is None:
            return
        if item is not None and chosen == remove_action:
            fit_index = self._results_row_fit_index[item.row()]
            del active.fits[fit_index]
            mw._plot_data(preserve_view=True)
        elif chosen == clear_action:
            active.fits.clear()
            mw._plot_data(preserve_view=True)
```

- [ ] **Step 7: Update `_on_result_double_clicked`**

In `fit_mode.py`, in `_on_result_double_clicked`, change:

```python
        index = self.results_list.row(item)
        result = active.fits[index]
```

to:

```python
        fit_index = self._results_row_fit_index[item.row()]
        result = active.fits[fit_index]
```

- [ ] **Step 8: Initialize `_results_row_fit_index` in `__init__`**

In `FitModeController.__init__`, add a line after `self._parameter_names_shown = []`:

```python
        self._results_row_fit_index = []
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS. Several pre-existing tests reference `results_list` and will now fail to collect/run — fix each by renaming `results_list` → `results_table` and adapting `.item(0).text()`-style assertions on the old text-blob format to check specific `results_table` cells instead (e.g. the old `"volume excludes tail"` tooltip assertion in `test_...` near the end of the file now reads the Fit column's tooltip via `table.item(0, 0).toolTip()` instead of item text). Search the file for `results_list` to find every call site.

- [ ] **Step 10: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: redesign Fit Results panel as a per-peak table with explicit Position/FWHM/Volume columns"
```

---

### Task 4: Move Independent widths / Left tail into the Fit Parameters panel

**Files:**
- Modify: `fit_mode.py` (`build_parameters_panel`)
- Modify: `main_window.py` (`_build_fit_mode_buttons`)
- Test: `tests/test_fit_mode_ui.py`

`independent_widths_action` and `left_tail_action` become `QCheckBox`
widgets living at the top of the Fit Parameters dock, above
`parameters_table`. `QCheckBox` has the same `.isChecked()`/
`.setChecked()`/`.toggled` API the old `QAction`s did, so every other call
site in `fit_mode.py`/`main_window.py`/existing tests is unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`, directly after `test_clear_empties_the_parameters_panel`:

```python
def test_width_link_and_left_tail_checkboxes_live_in_the_parameters_panel(qapp):
    main_window = MainWindow()

    assert isinstance(main_window.independent_widths_action, QCheckBox)
    assert isinstance(main_window.left_tail_action, QCheckBox)
    assert main_window.independent_widths_action.parentWidget() is main_window.fit_controller.parameters_dock.widget()
    assert main_window.left_tail_action.parentWidget() is main_window.fit_controller.parameters_dock.widget()
```

Add `from PySide6.QtWidgets import QCheckBox` to the existing import block at the top of `tests/test_fit_mode_ui.py` if a plain (non-PySide6.QtWidgets-prefixed) `QCheckBox` isn't already imported there — check the existing `from PySide6.QtWidgets import ...` line first, since `fit_mode.py` already imports it and the test file may already too.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k checkboxes_live_in_the_parameters_panel`
Expected: FAIL (`assert isinstance(main_window.independent_widths_action, QCheckBox)` fails -- it's still a `QAction`)

- [ ] **Step 3: Rebuild `build_parameters_panel`'s widget hierarchy**

In `fit_mode.py`, add to the imports:

```python
from PySide6.QtWidgets import QVBoxLayout, QWidget
```

(add `QVBoxLayout, QWidget` to the existing `from PySide6.QtWidgets import (...)` block rather than a second import line).

Replace the start of `build_parameters_panel`, from `def build_parameters_panel(self):` through
`self.parameters_dock.setWidget(self.parameters_table)`, with:

```python
    def build_parameters_panel(self):
        mw = self.main_window
        mw.independent_widths_action = QCheckBox("Independent widths")
        mw.independent_widths_action.setToolTip(
            "Fit each peak's width independently instead of sharing one FWHM"
        )
        mw.left_tail_action = QCheckBox("Left tail")
        mw.left_tail_action.setToolTip(
            "Allow a small low-channel tail contribution to each peak's shape"
        )

        self.parameters_table = QTableWidget(0, 3)
        self.parameters_table.setHorizontalHeaderLabels(["Parameter", "Value", "Fix"])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(mw.independent_widths_action)
        layout.addWidget(mw.left_tail_action)
        layout.addWidget(self.parameters_table)

        self.parameters_dock = QDockWidget("Fit Parameters", mw)
        self.parameters_dock.setWidget(container)
```

- [ ] **Step 4: Remove the checkboxes' old construction from `main_window.py`**

In `main_window.py`, in `_build_fit_mode_buttons`, remove these two blocks:

```python
        self.independent_widths_action = QAction("Independent widths", self)
        self.independent_widths_action.setCheckable(True)
        self.independent_widths_action.setToolTip(
            "Fit each peak's width independently instead of sharing one FWHM"
        )
        self.fit_toolbar.addAction(self.independent_widths_action)

        self.left_tail_action = QAction("Left tail", self)
        self.left_tail_action.setCheckable(True)
        self.left_tail_action.setToolTip(
            "Allow a small low-channel tail contribution to each peak's shape"
        )
        self.fit_toolbar.addAction(self.left_tail_action)
```

`_build_fit_mode_buttons` now starts with `self.fit_toolbar = QToolBar(...)`/`self.fit_toolbar.setMovable(False)` and then goes straight to the `self.fit_button = QAction("Fit", self)` block.

- [ ] **Step 5: Fix construction order**

`build_parameters_panel()` now creates `main_window.independent_widths_action`/`left_tail_action`, but `_build_fit_mode_buttons()` (called before `build_parameters_panel()` in `MainWindow.__init__`, per the existing order) no longer creates them and no longer references them either after Step 4 -- confirm no remaining line in `_build_fit_mode_buttons` reads `self.independent_widths_action`/`self.left_tail_action` before they exist. (It doesn't -- those two blocks were the only place that touched them in that method.) No reordering of the `MainWindow.__init__` call sequence is needed.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (all tests -- every existing call site using `.isChecked()`/`.setChecked()`/`.toggled` on these two attributes keeps working unchanged since `QCheckBox` has the same API surface)

- [ ] **Step 7: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add fit_mode.py main_window.py tests/test_fit_mode_ui.py
git commit -m "feat: move Independent widths / Left tail checkboxes into the Fit Parameters panel"
```

---

### Task 5: Replace the Fit/Clear toolbar buttons with Ctrl+F / Ctrl+C shortcuts

**Files:**
- Modify: `main_window.py` (`_build_fit_mode_buttons`)
- Test: `tests/test_fit_mode_ui.py`

`fit_button`/`clear_fit_button` stay as `QAction`s (existing test call
sites use `.trigger()`/`.isEnabled()`/`.setEnabled()`) but are no longer
added to any toolbar. Each gets a shortcut and is registered directly on
the `QMainWindow` via `addAction()` so the shortcut fires window-wide.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`, directly after `test_width_link_and_left_tail_checkboxes_live_in_the_parameters_panel`:

```python
def test_fit_and_clear_are_shortcut_only_with_no_toolbar(qapp):
    main_window = MainWindow()

    assert main_window.fit_button.shortcut().toString() == "Ctrl+F"
    assert main_window.clear_fit_button.shortcut().toString() == "Ctrl+C"
    assert main_window.fit_button in main_window.actions()
    assert main_window.clear_fit_button in main_window.actions()
    assert not hasattr(main_window, "fit_toolbar")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k fit_and_clear_are_shortcut_only`
Expected: FAIL (`assert main_window.fit_button.shortcut().toString() == "Ctrl+F"` fails -- no shortcut set yet; `fit_toolbar` still exists)

- [ ] **Step 3: Rewrite `_build_fit_mode_buttons`**

In `main_window.py`, replace the entire method with:

```python
    def _build_fit_mode_buttons(self):
        self.fit_button = QAction("Fit", self)
        self.fit_button.setShortcut("Ctrl+F")
        self.fit_button.setEnabled(False)
        self.fit_button.triggered.connect(self.fit_controller.run_fit)
        self.addAction(self.fit_button)

        self.clear_fit_button = QAction("Clear", self)
        self.clear_fit_button.setShortcut("Ctrl+C")
        self.clear_fit_button.triggered.connect(self.fit_controller.clear)
        self.addAction(self.clear_fit_button)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add main_window.py tests/test_fit_mode_ui.py
git commit -m "feat: replace Fit/Clear toolbar buttons with Ctrl+F/Ctrl+C shortcuts"
```

---

## Self-Review Notes

- **Spec coverage:** Volume as an explicit field (Task 3), Clear visually
  resetting without deleting history (Task 2), independent-widths/left-tail
  relocation (Task 4), Fit/Clear shortcut-only (Task 5), per-peak
  Position/FWHM/Volume table (Task 3), re-fit stacking (Task 1) -- all
  covered. Disk export, the Integration mode, and a manual re-show-hidden-fit
  control are explicitly out of scope per the spec, and untouched here.
- **Placeholder scan:** No TBD/TODO; every step has complete, runnable
  code, except Task 3 Step 9's instruction to adapt pre-existing
  `results_list`-referencing tests, which is necessarily open-ended since
  it depends on each test's exact old assertions (there are roughly a
  dozen such call sites across the file) -- the implementer should grep for
  `results_list` and convert each one to the `results_table` API rather
  than guessing at a fixed list up front.
- **Type consistency:** `FitResult.visible` (Task 1) is read in Task 1's
  `draw_committed_fits`/`run_fit` and Task 2's `clear()`, and rendered in
  Task 3's `update_results_list`. `self._results_row_fit_index` (Task 3) is
  initialized in `__init__` and populated only in `update_results_list`,
  consumed by `_on_results_context_menu` and `_on_result_double_clicked`.
  `main_window.independent_widths_action`/`left_tail_action` (Task 4) are
  read via `.isChecked()` in `run_fit()`/`_on_result_double_clicked()`
  (unchanged call sites) and written via `.setChecked()` in
  `_on_result_double_clicked()` (unchanged) -- `QCheckBox` supports both,
  so no call site needs to change.
