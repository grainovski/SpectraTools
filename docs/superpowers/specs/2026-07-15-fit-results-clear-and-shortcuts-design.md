# Fit Results Table, Clear Fix, and Keyboard Shortcuts — Design Spec

Date: 2026-07-15

## Purpose

A feedback round after using the fit-repeatability/parameter-fixing feature
(`docs/superpowers/specs/2026-07-14-fit-repeatability-and-parameter-fixing-design.md`)
in practice. This spec is sub-project 3 from that spec's deferral list (the
"Clear" button fix), bundled with several adjacent requests that surfaced
from the same usage session:

1. The peak volume (Gaussian integral area) needs to be a clear, explicit
   field in the Fit Results panel, not just embedded in free text.
2. "Clear" doesn't visually reset the plot the way the user expects.
3. "Independent widths"/"Left tail" move from the toolbar into the Fit
   Parameters panel.
4. The "Fit"/"Clear" toolbar buttons are replaced by `Ctrl+F`/`Ctrl+C`
   keyboard shortcuts.
5. The Fit Results panel needs explicit per-peak fields (position, area,
   FWHM), not a multi-line text blob.
6. Re-fitting the same marks under changed settings visually stacks
   overlapping curves/labels on the plot.

## Root Cause Analysis

**Items 2 and 6 share a root cause.** `FitModeController.draw_committed_fits`
draws every entry in `spectrum.fits` unconditionally, every time. Combined
with the 2026-07-14 change where marks persist and every successful fit
always appends a new entry (never replaces one), re-fitting the same marks
under a changed checkbox produces two (or more) `FitResult`s for the
identical region, and both get drawn fully overlapping — confirmed by
direct reproduction: two fits of one peak at the same marks produces 7 line
artists / 6 patch artists / 2 text annotations on the axes where one fit
alone produces half that. This is what reads as the plot "stacking" when
zoomed in.

Separately, `FitModeController.clear()` (bound to the toolbar's "Clear"
button) only removes the *in-progress marking overlays*
(`self._progress_artists`) via `_clear_progress()`. It never calls
`main_window._plot_data()`, so whatever was last drawn — including a
just-committed fit's background shading, curve, and peak annotation from
`draw_committed_fits` — stays fully visible. The user's experience of
"after Clear no new fit is possible" is very likely this: the plot looks
completely unchanged after clicking Clear, giving the impression that
nothing happened and the app is stuck, even though the underlying marking
state *was* reset correctly (confirmed by direct reproduction: marking a
fresh region and fitting again after `clear()` already works today at the
state-machine level). Making Clear visibly reset the canvas directly
addresses the confusion.

## Fit Visibility Model (`peak_fit.py`, `fit_mode.py`)

`FitResult` gains a `visible: bool = True` field — the same pattern
`LoadedSpectrum.visible` already uses to let history persist without being
drawn. `draw_committed_fits()` skips any result where `visible` is
`False`. Nothing is ever deleted by these mechanisms; `active.fits` (and
therefore the Fit Results panel) always reflects full history.

Two things flip `visible` to `False`, both additive to what's already there:

1. **Auto-supersede on re-fit (fixes item 6).** When `run_fit()` commits a
   new result, any existing fit for the active spectrum whose
   `(left_bg_region, right_bg_region, fit_region)` exactly matches the new
   one's is marked `visible = False` before the new one is appended (fresh,
   `visible = True`). Marks are read from `self.state` unchanged between
   re-fits of the same region (only checkboxes/fixed values differ), so
   exact tuple equality is reliable here — no float-tolerance comparison
   needed. Only the latest attempt at a given region is ever drawn; every
   attempt remains listed in Fit Results.
2. **Clear hides, does not delete (fixes items 2 and 3 from the prior
   spec's deferral, i.e. sub-project 3).** `clear()` sets `visible = False`
   on every fit belonging to the active spectrum (not just ones matching
   the current marks), then triggers a full replot
   (`main_window._plot_data(preserve_view=True)`) so the canvas visibly
   goes blank of fit overlays. Nothing is removed from `active.fits` or the
   Fit Results panel — this is a *visual* reset only, confirmed with the
   user as the intended scope (as opposed to also deleting history, which
   the existing "Clear All Fits" context-menu action already does
   destructively when that's actually wanted).

The Fit Results table (below) marks hidden rows with dimmed text so it's
clear why an entry isn't drawn, rather than silently confusing.

## Fit Results Panel Redesign (`fit_mode.py`)

Replace `results_list` (a `QListWidget` of multi-line text items, one item
per fit) with `results_table`, a `QTableWidget` with one row **per peak**
(not per fit), columns:

| Fit | Peak | Position | FWHM | Volume |
|-----|------|----------|------|--------|

- **Fit**: `f"{fit_number} [{lo:.1f}, {hi:.1f}]"` (1-based index into
  `active.fits`, plus the fit region) — tooltip on this cell shows the
  extra settings the old header line carried (independent widths, left
  tail `r`/`β` with uncertainties, "volume excludes tail" caveat).
- **Peak**: 1-based peak number within that fit.
- **Position**: `f"{pos:.2f} ± {pos_err:.2f}"`.
- **FWHM**: `f"{fwhm:.2f} ± {fwhm_err:.2f}"`.
- **Volume**: `f"{area:.1f} ± {area_err:.1f}"` — this is the Gaussian
  integral area (`amplitude * sigma * sqrt(2*pi)`), already computed today
  but now a first-class column instead of buried in a text blob.

Rows for the same fit appear consecutively, oldest fit first (matching
today's append order). A hidden fit's rows render in gray/italic text.

`FitModeController` tracks `self._results_row_fit_index`, a list parallel
to the table's rows, mapping row → index into `active.fits`, rebuilt every
`update_results_list()` call. The right-click context menu ("Remove Fit" /
"Clear All Fits") and double-click-to-reload both resolve the clicked row
to a fit via this list instead of assuming row index == fit index.
"Remove Fit" from any one of a fit's peak-rows removes that entire fit (all
its peak-rows together) — same granularity as today, since a multi-peak
`FitResult` can't be partially removed.

## Fit Parameters Panel and Toolbar Changes (`main_window.py`, `fit_mode.py`)

- `independent_widths_action` and `left_tail_action` become `QCheckBox`
  widgets (not `QAction`s) living at the top of the Fit Parameters dock
  panel, above `parameters_table`, inside a small `QVBoxLayout` container.
  `QCheckBox` has the same `.isChecked()`/`.setChecked()`/`.toggled` API
  `QAction` did, so every existing call site
  (`main_window.independent_widths_action.setChecked(...)`, etc.) is
  unchanged — only construction and parent panel move.
- The `fit_button`/`clear_fit_button` `QAction`s stay as attributes (so
  `.trigger()`/`.isEnabled()`/`.setEnabled()` call sites are unchanged),
  but are no longer added to any toolbar or menu. Each gets
  `.setShortcut("Ctrl+F")` / `.setShortcut("Ctrl+C")` and is registered via
  `self.addAction(...)` directly on the `QMainWindow` so the shortcut fires
  window-wide with no visible button anywhere.
- `fit_toolbar` (now empty — both checkboxes and both buttons moved out)
  is removed entirely.

## Testing

- `peak_fit.py`: `FitResult.visible` defaults to `True`.
- `fit_mode.py`/`main_window.py`:
  - Re-fitting identical marks under a changed checkbox hides the earlier
    result (`visible == False`) and shows only the latest on the plot
    (`draw_committed_fits` artist counts, or a direct check of which
    results got drawn).
  - `clear()` hides every active-spectrum fit and a subsequent
    `_plot_data()` draws none of them, while `active.fits` and the Fit
    Results table still list them all.
  - A fit for a *different* region is unaffected by either mechanism.
  - Fit Results table renders the expected columns/values per peak, maps
    right-click/double-click actions to the correct fit regardless of
    which peak-row was clicked, and dims hidden rows.
  - `Ctrl+F`/`Ctrl+C` trigger `run_fit`/`clear` with no toolbar button
    present; `independent_widths_action`/`left_tail_action` are
    `QCheckBox`es inside the Fit Parameters dock and toggle exactly as
    before.

## Out of Scope

- Disk export of results (still deferred, sub-project 2 from the prior
  spec).
- The "Integration" mode (still deferred, sub-project 4).
- A UI control to manually re-show a hidden fit (once hidden by Clear or
  superseded by a re-fit, it stays hidden on the plot; it remains fully
  visible/editable via the Fit Results table and double-click-reload,
  which produces a fresh, visible entry when re-fit).
- Any change to the underlying fit math, uncertainty formulas, or the
  `fixed_params` mechanism from the prior spec.
