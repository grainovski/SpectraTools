# Matrix Panel Fit/Integrate/Calibrate Parity — Design Spec

Date: 2026-08-13
Branch: `v3-major`

## Purpose

The matrix panel's projection view (`matrix_panel.py`, shipped as the first `v3-major` feature) currently supports only: picking X or Y projection, marking a cut region and background regions to compute a gated spectrum, and an on-demand heatmap. It does not support zooming, peak fitting, integration, or calibration — every one of the main window's core spectrum-analysis capabilities. The user identified three concrete gaps after testing real builds:

1. The projection panel should support zoom, marking, fitting, integrating, and calibration the same way the main window does for ordinary spectra.
2. The projection should render as a histogram (step plot), not a smooth line.
3. Holding a mark key and clicking once should show a dashed preview line at that click, the same way fit/integrate marking already does in the main window — currently nothing is drawn until the second click completes a region.

Plus one documentation addition: the Knowledge Database page should explain 2D-spectra concepts (matrix, projection, cut/gate) for users unfamiliar with the technique.

## Current State

`MatrixPanel` (`matrix_panel.py`) is a separate, parentless top-level `QMainWindow`. Its own marking system, `MatrixCutController`, binds hold-**C** (cut region) and hold-**B** (background region) via a Qt `eventFilter` on the canvas, drawing completed regions as `axvspan` bands — but nothing for an in-progress first click. Its toolbar is a stock, untrimmed `NavigationToolbar2QT` (Home/Pan/rectangle-Zoom/Save only). The projection is drawn with `axes.plot(range(len(data)), data, linewidth=0.8)` — a smooth line.

The main window's equivalent machinery lives in `fit_mode.py`'s `FitModeController` (~1200 lines), bound tightly to `MainWindow`: it reads `self.main_window.canvas`, `.axes`, `.spectra` (a list of `LoadedSpectrum`), `.statusBar()`, `._calibration`/`._calibration_active`, `.channel_to_display`/`.display_to_channel`, `.independent_widths_action`/`.left_tail_action`, calls `.main_window._plot_data(...)` to redraw, and `.main_window._update_fit_mode_availability()` to enable/disable menu items. `main_window.py` also owns the Fit Results dock, Fit Parameters dock, the custom `_TrimmedNavigationToolbar` + Ctrl+=/Ctrl+-/Ctrl+0 + scroll-wheel zoom, the Calibrate action/dialog, and Integrate (Ctrl+I) wiring. `LoadedSpectrum` itself (`spectrum.py`) is a plain, simple container: `path, data, color, visible, active, fits=[]`.

## Resolved Decisions

**Reuse strategy: duck-typed reuse, not a shared base class.** `MatrixPanel` will implement the same attribute/method surface `FitModeController` (and the integrate/calibrate wiring) already expects from `main_window`, and construct its own `FitModeController` instance pointed at itself. `fit_mode.py`, `peak_fit.py`, `fit_export.py`, `calibration.py`, and `calibration_dialog.py` are used entirely unmodified — this is the actual, hard-won, TV-parity-verified logic, and it is not touched or duplicated. This keeps the risk of regressing the main window's existing, extensively-tested behavior at essentially zero, since none of its code paths change.

The one place this needs `main_window.py` to change at all is extracting the Fit Results / Fit Parameters dock **widget construction** (not their behavior) into small, shared helper functions both windows call, so that Qt boilerplate isn't hand-duplicated between two files. This is a narrow, mechanical "extract method" — if the helper produces the identical widget tree `MainWindow` already builds inline today, its behavior is unchanged and the existing test suite proves it. This is explicitly *not* the "shared base class" approach that was ruled out during brainstorming (that would have touched `FitModeController`'s own coupling, which is where the real regression risk lives).

**Marking key rename: hold-B (gate background) → hold-G.** Cut/gate marking and fit marking will be simultaneously live on the same canvas, and both currently want hold-B (gate's "background region" vs. fit's "background region"). Hold-C (cut) doesn't collide with anything and stays as-is. Hold-B is reserved for the fit system (matching the main window exactly, unchanged); the gate/cut system's background-region key moves to hold-**G**. `MatrixCutController`'s `eventFilter` and `MatrixCutState` are updated accordingly; the visual marking (green `axvspan`) is unchanged, just the trigger key.

**Calibration is shared, not per-window.** `MatrixPanel` reads and writes `main_window._calibration`/`main_window._calibration_active` directly — the same single source of truth already used everywhere else in this app. Calibrating from the matrix panel updates the main window's display too, and vice versa. There is exactly one calibration per session, matching the existing architecture; this feature does not introduce a second one.

**Switching X/Y projection resets fit and gate state.** `_on_axis_changed` already calls `cut_controller.clear()` when the working axis changes, since a cut region marked against X-channel positions is meaningless once the data underneath is the Y projection. The same logic extends to fits: switching axis clears the projection's `LoadedSpectrum.fits` list and any in-progress fit marking. Fits are not preserved per-axis and switchable — they belong to whichever data is currently displayed.

**No separate "Spectra" list dock.** The main window's Spectra dock (with visibility checkboxes, multi-spectrum selection) exists because multiple spectra can be loaded there simultaneously. The matrix panel always has exactly one "spectrum" — the current working-axis projection — so there is nothing for a spectrum-list dock to manage. `MatrixPanel.spectra` is always a one-item list.

**Scope boundary: fit/integrate/calibrate only.** The Operations menu's other capabilities — Multiply by Factor, Rebin by Factor, Normalize Spectra, Add/Subtract, Save Spectrum, Close Spectrum — are not part of this request and are not added to the matrix panel. If a user wants those, the existing path (Activate Cut → work with the resulting spectrum in the main window, which already has all of them) still applies unchanged.

**Ctrl-modified shortcuts coexisting with bare-key holds: match the existing (imperfect but working) pattern, don't "fix" it.** `fit_mode.py`'s own `_mark_type_for_key_event` doesn't check `event.modifiers()` either — Ctrl+B (background-preview toggle) and bare-B (background-mark hold) already coexist in the shipped, tested main window without an explicit modifier guard. `MatrixCutController` will follow the identical pattern for its own Ctrl-shortcuts (Ctrl+I, Ctrl+F, Ctrl+C for the fit list, etc.) rather than introducing new, inconsistent modifier-checking logic that would make the two windows subtly diverge.

## What Gets Built

**Zoom.** Replace the matrix panel's stock `NavigationToolbar2QT` with the same `_TrimmedNavigationToolbar` the main window uses, plus the same Ctrl+=/Ctrl+-/Ctrl+0 actions and scroll-wheel zoom (`main_window.py`'s `_zoom_x`), operating on the projection's X axis.

**Fit marking and fitting.** Hold-R (fit region), hold-B (fit background, capped at 2 regions per the existing `FitModeState` convention), hold-P (peak positions) — identical semantics, visuals, and interaction to the main window. Peak fitting itself (`peak_fit.py`) is invoked exactly as `FitModeController` already does.

**Fit Results / Fit Parameters docks.** New docks in `MatrixPanel`, functionally identical to the main window's: a results list (double-click for detail, Ctrl+C to delete), and an editable parameter table for the selected fit. Built via the shared widget-construction helpers described above.

**Integration.** Ctrl+I integrates a region (0 or 2 background regions, gross/background/net) exactly as `integrate_region`/`IntegrationResult` already compute it.

**Calibration.** A Calibrate action reachable from the matrix panel opens the same `CalibrationDialog`, operating on the shared calibration state described above.

**Export.** Export All Fits works the same way, against the projection's own fit list.

**Histogram display.** `_plot_projection` changes from `axes.plot(range(len(data)), data, linewidth=0.8)` to the same `drawstyle="steps-mid"` style the main window uses for spectra.

**Pending-click preview.** `MatrixCutState` already tracks `_pending_cut_click`/`_pending_bg_click` (soon `_pending_gate_bg_click`, per the key rename) internally but draws nothing for them. `_redraw_markers` gains the same treatment `fit_mode.py`'s `_redraw_progress` already gives `pending_bg_click`/`pending_fit_click`: a dashed `axvline` at the pending click's position, cleared once the second click completes the region (or on `Clear Marks`).

## Documentation

**HowTo.** Section 11 ("Matrix analysis") needs a substantial rewrite to document the full new capability set and shortcut list, matching how thoroughly the rest of the HowTo page documents the main window. The File-menu-style shortcut table convention should extend to cover the matrix panel's own shortcuts (a new table, since these are window-scoped rather than menu-scoped) — mirroring how `Ctrl+Shift+O`'s earlier omission from the shortcut table was treated as a real gap, not a nice-to-have.

**Knowledge Database.** New content explaining: what a 2D coincidence matrix is, what a projection is (summing one axis), what a cut/gate is (selecting a region on one axis to view the correlated distribution on the other), and why background subtraction is needed (random coincidences) — for a reader unfamiliar with gamma-gamma coincidence technique, not just unfamiliar with this app's UI. Matching the Knowledge Database's existing convention of pairing each concept with an annotated figure (`help_figures.py`), at least one new schematic figure (a matrix with cut/background bands marked, alongside its resulting projection) is likely needed — exact figure count and content to be finalized during planning.

## Out of Scope

- Multiply/Rebin/Normalize/Add-Subtract/Save/Close-Spectrum operations on the projection.
- Per-axis independent fit persistence (switching X/Y always clears fit and gate state).
- Any change to `Activate Cut`'s existing computation or behavior.
- Any change to `FitModeController`, `peak_fit.py`, `fit_export.py`, `calibration.py`, or `calibration_dialog.py`'s own logic — used as-is.
- A second, independent calibration state for the matrix panel.

## Testing Approach

Beyond ordinary per-feature tests, two things specifically need proving, not just assuming:

1. **The duck-typed reuse actually produces identical results**, not just "doesn't crash" — tests should fit/integrate/calibrate equivalent data through both `MainWindow` and `MatrixPanel` and confirm matching output, the same rigor this project applied when it wasn't obvious a reused code path was actually being exercised correctly.
2. **Zero regression to the existing main window test suite.** Since the reuse strategy is specifically chosen to avoid touching `fit_mode.py`/`main_window.py`'s existing behavior, the full existing test suite must stay 100% green throughout — any failure there is a signal the "duck-typed, zero-risk" premise has been violated somewhere.
