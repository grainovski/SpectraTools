# Ctrl+C Non-Destructive Clear + Blocked-Action Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Ctrl+C` stops permanently deleting committed fits — it hides them (reusing the existing `visible` mechanism already used for superseded fits) while `Ctrl+Shift+C` stays as the true delete. `Ctrl+F`/`Ctrl+I` stop silently doing nothing when marks are incomplete — they now show a status-bar message naming what's still missing, mirroring the app's own existing peak-marking hint.

**Architecture:** Both changes are confined to `fit_mode.py`, plus a doc-accuracy fix in `help_content.py`. No new files, no new mechanism — `clear()`'s fits-handling switches from `active.fits.clear()` to a `visible = False` loop over the same list; two new `FitModeState` methods (`fit_blocked_reason()`, `integrate_blocked_reason()`) mirror `ready_to_fit()`/`ready_to_integrate()`'s own conditions to produce a specific message instead of a bare boolean.

**Tech Stack:** Python 3.13 (Windows dev/build), PySide6/matplotlib/numpy (existing), `pytest` with the existing `qapp` fixture.

**Design spec:** `docs/superpowers/specs/2026-08-07-fit-clear-and-blocked-feedback-design.md` — read for the full rationale (including exactly which existing code paths were traced to arrive at this design) if anything below is unclear.

---

## Context for the implementer

This is SpectraTools, a nuclear-spectroscopy peak-fitting desktop app. Both fixes were found during real Windows testing of a prior feature (v2.1.2's Integration-without-background) and are unrelated to each other except that they both live in `fit_mode.py` and were found in the same testing session.

All work happens directly on `master`, committing after each task — this matches every prior feature in this repo's history (`git log --oneline`). Do not create a branch or worktree for this work. (There is a pre-existing, unrelated worktree at `.claude/worktrees/fit-parameter-fixing` — leave it alone.)

Run tests with `.venv/Scripts/python.exe -m pytest <path> -v` on Windows (this repo's dev machine) or `.venv/bin/python -m pytest <path> -v` on Linux.

---

## Task 1: `fit_mode.py` — `Ctrl+C` hides fits instead of deleting them

**Files:**
- Modify: `fit_mode.py:394-399` (`clear()`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_mode_ui.py` (after `_fit_result_with_one_peak()`'s definition, around line 1837 — right before `test_results_panel_lists_committed_fit`):

```python
def test_clear_hides_committed_fits_instead_of_deleting_them(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(_fit_result_with_one_peak())

    main_window.fit_controller.clear()

    assert len(spectrum.fits) == 1  # still present, not deleted
    assert spectrum.fits[0].visible is False

    table = main_window.fit_controller.results_table
    assert table.rowCount() == 1  # still listed
    default_color = QTableWidgetItem().foreground()
    assert table.item(0, 0).foreground() != default_color  # dimmed, same convention as an existing superseded fit

    # The plot itself shows nothing for a hidden fit -- draw_committed_fits()
    # skips it entirely, same as test_clear_forces_a_replot_so_the_canvas_
    # actually_goes_blank already expects for the (still-true) "canvas goes
    # blank" behavior.
    assert len(main_window.axes.lines) == 1  # only the spectrum's own step line
    assert len(main_window.axes.patches) == 0  # no region shading
```

Replace (currently `tests/test_fit_mode_ui.py:1613-1629`):

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

with:

```python
def test_clear_hides_rather_than_deletes_a_fit_made_through_the_real_marking_flow(qapp):
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

    assert len(spectrum.fits) == 1  # still present
    assert spectrum.fits[0].visible is False
```

(`test_clear_forces_a_replot_so_the_canvas_actually_goes_blank`, `test_clear_discards_in_progress_marks_without_committing`, `test_reset_marks_clears_progress_without_touching_fits_or_replotting`, `test_clear_with_no_active_spectrum_does_not_crash`, `test_clear_progress_survives_an_intervening_full_replot`, `test_clear_empties_the_parameters_panel`, `test_clear_all_fits_empties_panel_and_spectrum`, `test_clear_all_fits_shortcut_deletes_fits` — checked against this change already; none of them assert anything that this task's change would break, so none of them need editing. Do not touch them.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode_ui.py -k "clear_hides" -v`

Expected: `test_clear_hides_committed_fits_instead_of_deleting_them` FAILS on `assert len(spectrum.fits) == 1` (currently `0` — `clear()` empties the list). `test_clear_hides_rather_than_deletes_a_fit_made_through_the_real_marking_flow` FAILS the same way.

- [ ] **Step 3: Implement**

Replace (currently `fit_mode.py:394-399`):

```python
    def clear(self):
        self.reset_marks()
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is not None:
            active.fits.clear()
        self.main_window._plot_data(preserve_view=True)
```

with:

```python
    def clear(self):
        self.reset_marks()
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is not None:
            for result in active.fits:
                result.visible = False
        self.main_window._plot_data(preserve_view=True)
```

(`_plot_data()` already calls `self.fit_controller.update_results_list()` as its own last line — see `main_window.py:830` — so the Fit Results panel refresh needed to show the newly-dimmed rows happens automatically; no additional call is needed here.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode_ui.py -v`

Expected: all tests pass, including the new/renamed ones and every pre-existing test — in particular `test_clear_all_fits_shortcut_deletes_fits` (confirms `Ctrl+Shift+C` is completely unchanged) and `test_clear_forces_a_replot_so_the_canvas_actually_goes_blank` (confirms the plot still goes visually blank, since a hidden fit still isn't drawn).

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "fix: Ctrl+C hides committed fits instead of permanently deleting them"
```

---

## Task 2: `fit_mode.py` — status message when Ctrl+F/Ctrl+I are blocked

**Files:**
- Modify: `fit_mode.py:96-104` (add two methods to `FitModeState`, right after `ready_to_integrate()`)
- Modify: `fit_mode.py`, `run_fit()`'s opening guard (currently around line 1075 — Task 1's edit shifts everything below it down by one line, so match by the code shown in Step 4 rather than the exact number)
- Modify: `fit_mode.py`, `run_integration()`'s opening guard (currently around line 1137 — same line-shift note as above)
- Test: `tests/test_fit_mode.py`
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_mode.py` (at the end of the file):

```python
def test_fit_blocked_reason_when_no_fit_region():
    state = FitModeState()
    assert state.fit_blocked_reason() == "Mark the fit region (hold R and click twice) before fitting"


def test_fit_blocked_reason_when_bg_regions_incomplete():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.fit_blocked_reason() == "Mark two background regions (hold B and click twice per region) before fitting"


def test_fit_blocked_reason_when_no_peaks():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.fit_blocked_reason() == "Mark at least one peak (hold P and click) before fitting"


def test_fit_blocked_reason_is_none_when_ready():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.add_fit_click(85)
    state.add_fit_click(115)
    state.toggle_peak(100.0, proximity=1.0)
    assert state.ready_to_fit() is True
    assert state.fit_blocked_reason() is None


def test_integrate_blocked_reason_when_no_fit_region():
    state = FitModeState()
    assert state.integrate_blocked_reason() == "Mark the fit region (hold R and click twice) before integrating"


def test_integrate_blocked_reason_when_exactly_one_bg_region():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.integrate_blocked_reason() == "Mark zero or two background regions (not one) before integrating"


def test_integrate_blocked_reason_is_none_with_zero_bg_regions():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.ready_to_integrate() is True
    assert state.integrate_blocked_reason() is None


def test_integrate_blocked_reason_is_none_with_two_bg_regions():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.ready_to_integrate() is True
    assert state.integrate_blocked_reason() is None
```

Add to `tests/test_fit_mode_ui.py` (at the end of the file):

```python
def test_run_fit_shows_a_status_message_when_marks_are_incomplete(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    # No B marks, no P marks -- bg_regions is the first thing missing
    # after fit_region, per fit_blocked_reason()'s check order.

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 0
    assert (
        main_window.statusBar().currentMessage()
        == "Mark two background regions (hold B and click twice per region) before fitting"
    )


def test_run_fit_shows_no_message_and_commits_when_marks_are_complete(qapp):
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
    assert main_window.statusBar().currentMessage() == ""


def test_run_integration_shows_a_status_message_when_marks_are_incomplete(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    # Exactly 1 bg region -- not 0, not 2.

    main_window.fit_controller.run_integration()

    assert len(spectrum.fits) == 0
    assert (
        main_window.statusBar().currentMessage()
        == "Mark zero or two background regions (not one) before integrating"
    )


def test_run_integration_shows_no_message_and_commits_when_marks_are_complete(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)

    main_window.fit_controller.run_integration()

    assert len(spectrum.fits) == 1
    assert main_window.statusBar().currentMessage() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode.py -k "blocked_reason" -v`

Expected: all 8 new tests FAIL with `AttributeError: 'FitModeState' object has no attribute 'fit_blocked_reason'` (or `integrate_blocked_reason`).

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode_ui.py -k "shows_a_status_message_when_marks_are_incomplete" -v`

Expected: both FAIL on the message assertion — currently `main_window.statusBar().currentMessage() == ""` (the gate silently returns with no message).

- [ ] **Step 3: Implement — `FitModeState`**

Add to `fit_mode.py`, immediately after `ready_to_integrate()` (currently ending at line 104) and before `ordered_bg_regions()`:

```python
    def fit_blocked_reason(self):
        """Human-readable explanation of what's still missing before
        ready_to_fit() would return True, or None if it's already
        ready. Checked in the order a user would naturally complete
        marks in -- toggle_peak() itself refuses to add a peak before
        fit_region exists, so checking fit_region first always points
        at genuinely the next missing thing, never a redundant one.
        Returns None under exactly the condition ready_to_fit()
        returns True (the same three checks, so a caller that already
        confirmed ready_to_fit() is False can call this directly with
        no further guard)."""
        if self.fit_region is None:
            return "Mark the fit region (hold R and click twice) before fitting"
        if len(self.bg_regions) != BG_REGION_CAP:
            return "Mark two background regions (hold B and click twice per region) before fitting"
        if len(self.peak_positions) == 0:
            return "Mark at least one peak (hold P and click) before fitting"
        return None

    def integrate_blocked_reason(self):
        """Same as fit_blocked_reason(), but for ready_to_integrate()
        -- Integration accepts zero or two background regions, never
        exactly one."""
        if self.fit_region is None:
            return "Mark the fit region (hold R and click twice) before integrating"
        if len(self.bg_regions) not in (0, BG_REGION_CAP):
            return "Mark zero or two background regions (not one) before integrating"
        return None
```

- [ ] **Step 4: Implement — `run_fit()` and `run_integration()`**

Replace (currently `fit_mode.py:1075-1077`):

```python
    def run_fit(self):
        if not self.state.ready_to_fit():
            return
```

with:

```python
    def run_fit(self):
        if not self.state.ready_to_fit():
            self._show_status_message(self.state.fit_blocked_reason(), 5000)
            return
```

Replace (currently `fit_mode.py:1137-1139`, locate by content if line numbers have shifted):

```python
    def run_integration(self):
        if not self.state.ready_to_integrate():
            return
```

with:

```python
    def run_integration(self):
        if not self.state.ready_to_integrate():
            self._show_status_message(self.state.integrate_blocked_reason(), 5000)
            return
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode.py tests/test_fit_mode_ui.py -v`

Expected: all tests pass, including the 8 new `test_fit_mode.py` tests, the 4 new `test_fit_mode_ui.py` tests, and every pre-existing test — in particular every other test that calls `run_fit()`/`run_integration()` with complete marks (e.g. `test_run_fit_appends_a_fit_result`-style tests, `test_ready_to_integrate_allows_zero_background_regions`) continues to pass unaffected, since a successful call never reaches the new `_show_status_message` line.

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: show a status message when Ctrl+F/Ctrl+I are blocked by incomplete marks"
```

---

## Task 3: `help_content.py` — fix the now-inaccurate Ctrl+C description

**Files:**
- Modify: `help_content.py:136` (shortcuts table row)
- Modify: `help_content.py:230-234` (prose paragraph)
- Test: `tests/test_help_content.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_help_content.py` (after `test_howto_integration_section_mentions_optional_background`):

```python
def test_howto_ctrl_c_description_says_hide_not_delete():
    html = build_howto_html()
    assert "hides" in html.lower()
    assert "permanently delete" in html.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_help_content.py -k ctrl_c_description -v`

Expected: FAIL — the current text says "clears"/"clear", not "hides"/"permanently delete".

- [ ] **Step 3: Implement**

Replace (currently `help_content.py:136`):

```python
<tr><td><kbd>Ctrl+C</kbd></td><td>Clear the active spectrum's in-progress marks and committed fits</td></tr>
<tr><td><kbd>Ctrl+Shift+C</kbd></td><td>Clear the active spectrum's committed fits only (in-progress marks untouched)</td></tr>
```

with:

```python
<tr><td><kbd>Ctrl+C</kbd></td><td>Clear in-progress marks and hide committed fits (not delete)</td></tr>
<tr><td><kbd>Ctrl+Shift+C</kbd></td><td>Permanently delete the active spectrum's committed fits (in-progress marks untouched)</td></tr>
```

Replace (currently `help_content.py:230-234`):

```python
<p><kbd>Ctrl+C</kbd> clears the active spectrum's in-progress B/R/P
marks as well as its already-committed fits. <kbd>Ctrl+Shift+C</kbd>
clears only the active spectrum's committed fits, leaving any
in-progress marks alone. <kbd>Ctrl+E</kbd> exports the active
spectrum's committed fits.</p>
```

with:

```python
<p><kbd>Ctrl+C</kbd> clears the active spectrum's in-progress B/R/P
marks and hides its already-committed fits (grayed out in Fit
Results, removed from the plot, but not deleted).
<kbd>Ctrl+Shift+C</kbd> permanently deletes the active spectrum's
committed fits, leaving any in-progress marks alone. <kbd>Ctrl+E</kbd>
exports the active spectrum's committed fits.</p>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_help_content.py -v`

Expected: all tests pass, including the new one and the existing balanced-HTML-tags check and `test_howto_html_covers_every_operation`.

- [ ] **Step 5: Commit**

```bash
git add help_content.py tests/test_help_content.py
git commit -m "docs: correct the HowTo page's Ctrl+C description to hide, not delete"
```

---

## Task 4: Windows verification (primary focus)

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass, including everything added in Tasks 1-3.

- [ ] **Step 2: Rebuild and manually exercise the app**

```bash
powershell -File packaging\windows\build.ps1
```

Then run `dist\SpectraTools.exe` directly and manually verify:
- Load a spectrum. Mark B/R/P and run a Fit (Ctrl+F) — confirm it commits as before.
- Press `Ctrl+C`. Confirm the fit is now grayed out in the Fit Results panel (not gone), and gone from the plot (no curve, no region shading).
- Press `Ctrl+Shift+C`. Confirm the (now-hidden) fit is actually gone from the Fit Results panel too.
- Mark only a fit region (R) with nothing else, press `Ctrl+F` — confirm a status-bar message appears naming background regions as missing (not peaks — bg_regions is checked before peaks).
- Mark a fit region and two background regions but no peak, press `Ctrl+F` — confirm the message now specifically asks for a peak.
- Mark a fit region and exactly one background region, press `Ctrl+I` — confirm a status-bar message appears explicitly saying zero or two, not one.
- Complete a normal Fit and a normal zero-background Integration — confirm neither shows any blocked-action message (only the success itself).
- Open the HowTo page (F1) and confirm the Ctrl+C/Ctrl+Shift+C rows and prose read correctly and accurately describe the new behavior.

- [ ] **Step 3: No commit** (verification-only task; fix and re-verify if anything above fails).

---

## Task 5: Linux verification (proportionate)

**Files:** none (verification only)

- [ ] **Step 1: Rebuild via the existing scripts**

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d AlmaLinux-8 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/build.sh"
```
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/build_deb.sh"
```

- [ ] **Step 2: Smoke-test on Ubuntu-24.04 and AlmaLinux-10**

Per this project's established WSL invocation rules (`MSYS_NO_PATHCONV=1`, real script files — never inline `bash -c`, foreground execution with `timeout`, exit code 124 as the only real "it's alive" signal — see project memory on repeated, costly failures otherwise): write install + launch-smoke-test scripts to `packaging/linux/output/`, reinstall the exact freshly-built package on each distro (`apt-get install -y --reinstall <exact .deb filename>` on Ubuntu; `dnf reinstall -y <exact .rpm filename>` on AlmaLinux-10, since the version number is unchanged from before and `dnf install` alone silently no-ops on an already-present same-version package), then run `timeout -k 1 10 spectratools` as the regular (non-root) user on each and confirm `EXIT CODE: 124` with only the already-documented benign Fontconfig warning on stderr — no new errors. Delete the throwaway scripts when done.

This is a launch-level regression check, not a full manual UI exercise of Ctrl+C/blocked-message behavior on Linux — per the design spec, that depth of Linux-specific verification isn't warranted for a pure application-code change with no new packaging surface, and Task 4 already covers the feature's actual behavior thoroughly.

- [ ] **Step 3: No commit** (verification-only task).

---

## Final Step: Hand back to the user

Once all 5 tasks are complete, report back: what was built, the Windows and Linux verification results. Neither of these two fixes has been frozen/released yet — that stays a separate, later step following the established release process, only after the user has manually confirmed both work correctly.
