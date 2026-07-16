# FWHM Parameter, Manual Value Entry, and Disk Export — Design Spec

Date: 2026-07-16

## Purpose

A follow-up feedback round after live-testing the Marquardt fit and
parameter-fixing work
(`docs/superpowers/specs/2026-07-14-fit-repeatability-and-parameter-fixing-design.md`,
`docs/superpowers/specs/2026-07-15-marquardt-fit-and-peak-decomposition-design.md`).
Three requests, bundled because all three touch the same Fit
Parameters / Fit Results workflow:

1. FWHM (shared or per-peak, depending on "Independent widths") should
   appear in the Fit Parameters panel so it can be fixed directly,
   instead of only sigma.
2. Every row's Value cell in the Fit Parameters panel should be
   editable by hand, not just fixed rows.
3. Detailed fit results, timestamped, should be written to disk — both
   automatically and on explicit request. This is sub-project 2
   deferred from the 2026-07-14 spec.

## A. FWHM Instead of Sigma in the Fit Parameters Panel

The fitting engine (`peak_fit.py`) keeps `sigma` as its internal
canonical parameter (`sigma` when linked, `sigma_i` per peak when
independent) — no changes to `_parameter_names`, `_marquardt_fit`,
step damping, or bounds, all of which are tuned around sigma. The
FWHM↔sigma conversion happens only at the UI boundary in
`fit_mode.py`:

- `_parameter_label` renames `"sigma"` → `"Shared FWHM"` and
  `"sigma_{i}"` → `"Peak {i+1} FWHM"`.
- `update_parameters_panel` displays `sigma * FWHM_FACTOR` for any
  sigma-family row instead of the raw sigma.
- Reading the panel back (both `fixed_params_from_panel` and the new
  `initial_guess_overrides_from_panel`, see section B) converts a
  sigma-family row's entered FWHM back to sigma
  (`fwhm_entered / FWHM_FACTOR`) before inserting into the dict handed
  to `fit_peaks()`.

`FWHM_FACTOR` already exists in `peak_fit.py:7`
(`2*sqrt(2*ln(2))`); `fit_mode.py` adds it to its existing `from
peak_fit import (...)` line (`fit_mode.py:12-14`) rather than
redefining it.

## B. Manual Value Entry for Every Parameter

Today `_on_fix_toggled` (`fit_mode.py:483-490`) is the only thing that
makes a Value cell editable, and only while its row is checked. This
changes to:

- Every row's Value cell is always editable (drop the
  `_on_fix_toggled` flag-toggling; the Fix checkbox continues to exist
  and still determines *how* the value is used — see below — but no
  longer gates editability).
- `run_fit()` reads **every** row, splitting into two dicts:
  - Checked rows → `fixed_params[name] = value` (unchanged semantics:
    excluded from the optimizer's free vector, `0.0` uncertainty
    reported).
  - Unchecked rows → `initial_guess_overrides[name] = value`, a new
    dict passed to `fit_peaks()`.
- `fit_peaks()` gains `initial_guess_overrides: dict[str, float] | None
  = None`. Inside `_initial_guess()`, any name present in this dict
  overrides the internally-computed guess for that free parameter
  before `p0` is assembled; names absent from it keep today's
  heuristic (measured width, clicked-position amplitude, etc.)
  unchanged.
- Sigma-family entries in `initial_guess_overrides` go through the
  same FWHM→sigma conversion as `fixed_params` (section A) before
  being passed to `fit_peaks()`.
- Validation: a row's text must parse as a float whether it's checked
  or not (today this is only enforced for checked rows). An invalid
  unchecked row raises the same `FitError` pattern
  `fixed_params_from_panel` already uses, naming the offending row.

**Behavioral note:** the panel already repopulates every unchecked
row's displayed value with the previous fit's own converged result
(`update_parameters_panel`'s existing else-branch). Since that
displayed value now always feeds forward as
`initial_guess_overrides`, a second fit on unchanged marks starts
"warm" from the first fit's result by default, not from a
freshly-recomputed heuristic guess — a user doesn't have to type
anything for this to happen. This is expected to help convergence in
the common re-fit-after-tweaking-a-checkbox workflow, but is a real
change from today's always-recompute behavior. The **first** fit for
a fresh set of marks is unaffected (panel is empty, `fixed_params` and
`initial_guess_overrides` are both `{}`, per the existing "first fit
is always fully free" rule).

## C. Disk Export

### Automatic JSON Lines log

Every successful fit appends one JSON object (one line, newline
terminated) to `<spectrum_stem>_fits.jsonl`, in the same directory as
the spectrum file (`LoadedSpectrum.path`). E.g. loading `eu.spe` logs
to `eu_fits.jsonl` alongside it. This happens silently inside
`run_fit()`, right after `active.fits.append(result)`. A write failure
(read-only directory, missing permissions, etc.) shows a
`_show_status_message` warning and does not affect the in-app fit,
which already succeeded and is already reflected in the UI and
in-memory `active.fits` list.

The log is an independent, append-only audit trail: later removing a
fit from the app ("Remove Fit" / "Clear All Fits" in the Fit Results
context menu) does not rewrite or touch already-written log lines.
Concurrent-write locking across multiple app instances writing the
same log file is out of scope (see below).

### Explicit plain-text export

Two new actions on the Fit Results list's context menu
(`_on_results_context_menu`, `fit_mode.py:553-569`):

- **"Export This Fit..."** — only offered when right-clicking an
  existing entry, resolving the row to a fit the same way "Remove
  Fit" already does (`self._results_row_fit_index[item.row()]` —
  a fit's multiple peaks span multiple rows, and right-clicking any of
  them targets the whole fit, not just that one peak). Opens
  `QFileDialog.getSaveFileName` defaulting to
  `<stem>_fit{N}_report.txt`, where `N` is the fit's 1-based index
  matching its label in the Fit Results list (`fit_index + 1`), in the
  spectrum's directory, then writes a human-readable report for that
  one fit.
- **"Export All Fits..."** — offered whenever the active spectrum has
  at least one fit. Opens the same style of save dialog, defaulting to
  `<stem>_fits_report.txt`, and writes one report block per fit, in
  the same order as the Fit Results list.

### Record content (both formats)

Per fit: timestamp, spectrum path, left/right background regions, fit
region, background slope/intercept, `link_widths`, tail parameters
with uncertainties (when enabled), `fixed_params` (name→value dict),
and every peak's position/FWHM/amplitude/sigma/area, each with its
`_err`. This matches "all parameters + uncertainties" from the
original 2026-07-14 spec's deferred sub-project 2, plus the newly
requested timestamp.

### New `FitResult.timestamp` field

`peak_fit.py` stays free of wall-clock calls (`fit_peaks()` remains
deterministic given its inputs, which existing and future tests rely
on). `FitResult` gains `timestamp: str = None`, left unset by
`fit_peaks()` itself; `fit_mode.py`'s `run_fit()` sets
`result.timestamp = datetime.now().isoformat(timespec="seconds")`
immediately after `fit_peaks()` returns, before appending to
`active.fits`.

### New module `fit_export.py`

No Qt dependency — pure functions, independently testable:

- `fit_result_to_json_record(result, spectrum_path) -> dict`
- `append_auto_log(spectrum_path, result)` — derives the `.jsonl` path
  from `spectrum_path`, opens in append mode, writes one line.
- `fit_result_to_text_report(result, spectrum_path, peak_index=None)
  -> str` — one fit's block; used by both single- and all-fit export
  (the latter joins multiple blocks).

`fit_mode.py` wires this module to `run_fit()` (auto-log) and the two
new context-menu actions (explicit export), including the save-file
dialogs.

## Data Model Changes Summary

- `FitResult` (`peak_fit.py`) gains `timestamp: str = None`.
- `fit_peaks()` gains `initial_guess_overrides: dict[str, float] | None
  = None`.
- New `fit_export.py` module (no existing file changes beyond adding
  the new file).

## UI Changes Summary

- `_parameter_label`: sigma-family labels renamed to FWHM.
- `update_parameters_panel`: sigma-family display values converted to
  FWHM; Value cell's `ItemIsEditable` flag is set unconditionally when
  each row is built, so `_on_fix_toggled` and the checkbox's
  `toggled.connect(...)` wiring are removed entirely — the Fix
  checkbox's state is read directly via `.isChecked()` at fit time
  (`fixed_params_from_panel` / `initial_guess_overrides_from_panel`),
  it no longer needs to react live to being toggled.
- `fixed_params_from_panel`: sigma-family read-back converts FWHM back
  to sigma.
- New `initial_guess_overrides_from_panel`: mirrors
  `fixed_params_from_panel` for unchecked rows.
- `run_fit()`: builds both dicts, passes `initial_guess_overrides` to
  `fit_peaks()`, sets `result.timestamp`, calls
  `fit_export.append_auto_log()` after a successful fit.
- Two new context-menu actions in `_on_results_context_menu`, wired to
  `fit_export.fit_result_to_text_report()` and a save-file dialog.

## Testing

- `peak_fit.py`: `initial_guess_overrides` overrides the internal
  guess for a free parameter (position, amplitude, sigma, tail) and is
  ignored for a name that's also in `fixed_params`; absent names keep
  today's heuristic; existing test suite continues to pass unchanged
  (new parameter defaults to `None`, preserving current behavior
  exactly).
- `fit_mode.py`: FWHM label and sigma↔FWHM round-trip display/fix/edit
  for both shared and independent-width configurations; every row
  (checked or not) is editable and a non-numeric entry in either state
  raises a clear error naming the row; an edited unchecked row's value
  is read into `initial_guess_overrides` and passed through to
  `fit_peaks()`; double-click reload still restores fixed rows
  correctly with FWHM-converted values.
- `fit_export.py`: JSON record round-trips all fields including
  uncertainties and `fixed_params`; `append_auto_log` appends one
  valid JSON line per call without disturbing prior lines; text report
  contains every parameter/uncertainty/timestamp for one fit and for
  multiple fits in order; auto-log path is correctly derived from the
  spectrum path's stem.
- `main_window.py`/`fit_mode.py` integration: a successful fit writes
  an auto-log entry next to the (temp-directory, in tests) spectrum
  file; "Export This Fit"/"Export All Fits" context-menu actions
  appear/behave correctly based on right-click target and fit count.

## Out of Scope

- CSV export format (plain-text report and JSON Lines only, per this
  spec's requirements).
- Typing in initial peak marks (position/amplitude/width) before the
  first fit, without clicking the plot — marking remains click-based;
  only *editing an already-populated* Fit Parameters row is in scope.
- Persisting export file-dialog locations, or any export
  preferences/settings, across sessions.
- Auto-log file rotation, size limits, or concurrent-write locking
  across multiple app instances.
- Any change to the fitting engine's numerical behavior (damping,
  bounds, convergence criteria) beyond accepting an optional starting
  guess per free parameter.
- Configurable/selectable export columns or report templates — content
  is fixed as described above.
- Undo for a manually-edited (fixed or override) parameter value —
  re-fitting with the previous checkbox/value state is the existing
  workflow for reverting.
