# Operations Menu, Keyboard Shortcuts, and New Spectrum Operations — Design Spec

Date: 2026-07-28

## Purpose

User request: add keyboard shortcuts to every command that currently lacks one, add three new spectrum-processing operations (multiply by a constant, rebin by a factor, normalize two or more spectra against each other), add a new "Save spectrum" capability, and introduce a new **Operations** menu that houses calibration (moved from View) alongside the three new operations. All new commands should be reachable both via the Operations menu and via a keyboard shortcut.

## Current State (audited before designing)

Before this feature, the app has exactly four keyboard shortcuts, all `QAction.setShortcut()` calls in `main_window.py`: `Ctrl+O` Open, `Ctrl+F` Fit, `Ctrl+C` Clear, `Ctrl+I` Integrate. Every other menu item, toolbar button, checkbox, and context-menu action has no shortcut at all. There is currently no code anywhere that *writes* a spectrum file — `spe_io.py`, `spk_io.py`, and `histogram_io.py` each contain only a `load_*` function; "Save spectrum" is a wholly new capability, not an existing feature that merely needs a shortcut.

Two shortcut collisions existed in the original request and were resolved with the user before this spec was written (see "Resolved ambiguities" below).

## Resolved Ambiguities

These were established through discussion, not assumed:

- **Ctrl+C already means "Clear"** (clears in-progress marks and hides committed fits — see "Fits/marks clearing semantics" below). The request's "Ctrl+C loads calibration" conflicts with this pre-existing binding. Resolved: Clear keeps `Ctrl+C`; Load Calibration gets `Ctrl+L`. The request's separate "Ctrl+c toggles calibration" was never actually distinguishable from "Ctrl+C loads calibration" (a bare Ctrl+letter combo doesn't distinguish case) — Toggle Calibration Active gets its own key, `Ctrl+T`.
- **Ctrl+n / Ctrl+N is the same collision** between "multiply by factor" and "normalize spectra." Resolved: Normalize keeps `Ctrl+N` (reads more naturally as the mnemonic); Multiply gets `Ctrl+M`.
- **Save format**: since no writer exists for any format today, and the formats differ enormously in implementation cost (`.spe` is a moderately simple binary record format; `.spk` is a compressed binary format requiring a real codec), the user chose to support a **format-choice dialog** covering all three readable formats (`.spe`, `.spk`, plain text) rather than picking just one.
- **Normalize semantics**: the spectrum with the largest reference value (bin count or area, per the marker rule below) stays unchanged; every other spectrum is scaled up to match it.
- **Fits/marks after Multiply/Rebin/Normalize**: cleared automatically on any spectrum whose data actually changes (see "Fits/marks clearing semantics" below) rather than left stale or blocking the operation.
- **Rebin + calibration**: rebinning automatically transforms the active calibration's coefficients so the keV axis stays correct, rather than leaving it silently wrong.

## Fits/Marks Clearing Semantics (a precise reference for all three new operations)

The existing app currently has two different "clear" behaviors, and this feature **removes the distinction, in favor of deletion everywhere**:

- `FitModeController.clear()` (bound to the existing `Ctrl+C`, `fit_mode.py`) currently sets `result.visible = False` on every committed fit for the active spectrum instead of deleting it — a *hide*, not a delete. Hidden fits remain in `active.fits`, still listed (grayed out) in the Fit Results table, still double-clickable and exportable.
- The Fit Results panel's "Clear All Fits" context-menu action (`fit_mode.py`, `_on_results_context_menu`) already calls `active.fits.clear()` — an actual deletion.

**This spec changes `Ctrl+C`'s existing behavior**: it will call `active.fits.clear()` instead of setting `result.visible = False`, so it deletes rather than hides — the "hide but keep listed" behavior is removed from the app entirely, not just avoided in the three new operations. `Ctrl+C` and "Clear All Fits" still remain two distinct actions afterward, since `Ctrl+C` also clears in-progress marking artists (`_clear_progress()`), which "Clear All Fits" does not touch — only the fit-handling half of `Ctrl+C` changes. This is an intentional, in-scope change to existing behavior, not new-feature-only scoping: the existing test `test_clear_hides_every_fit_for_the_active_spectrum_without_deleting_it` (`tests/test_fit_mode_ui.py`) currently locks in the old hide behavior and will need rewriting to assert deletion instead.

Multiply, Rebin, and Normalize use this same **deletion** semantics for the fits they clear: after any of these operations changes a spectrum's data, that spectrum's old fit results are not just stale-looking, their channel-space parameters (position, region bounds) no longer correspond to anything meaningful in the new data — especially after Rebin, which changes the channel count itself. In addition to deleting `active.fits`, any **in-progress** marks in the single, global `FitModeState` (there is one `FitModeState` instance total, not one per spectrum, so in-progress marks are only ever meaningful for whatever was active when they were placed) must also be reset, the same way a completed Fit already resets them.

Out of scope, and deliberately untouched: the *different* `result.visible = False` usage in `run_fit()`/`run_integration()` that grays out a superseded fit when a new one is committed over the same region (preserving fit history rather than clearing on request) — that mechanism serves a different purpose and this spec does not change it.

## Shortcut Table

Unchanged (already existed): `Ctrl+O` Open, `Ctrl+F` Fit, `Ctrl+C` Clear, `Ctrl+I` Integrate.

New:

| Command | Shortcut | Menu |
|---|---|---|
| Load Calibration... | `Ctrl+L` | Operations |
| Toggle Calibration Active | `Ctrl+T` | Operations |
| Multiply by Factor... | `Ctrl+M` | Operations |
| Rebin by Factor... | `Ctrl+R` | Operations |
| Normalize Spectra | `Ctrl+N` | Operations |
| Save Spectrum... | `Ctrl+S` | File |
| Exit | `Ctrl+Q` | File |
| Dark theme | `Ctrl+D` | View |
| Log scale Y | `Ctrl+G` | View |
| Toggle Spectra panel | `Ctrl+1` | View |
| Toggle Fit Results panel | `Ctrl+2` | View |
| Toggle Fit Parameters panel | `Ctrl+3` | View |
| Zoom In X | `Ctrl+=` | new shortcut; stays toolbar-only, not moved into a menu |
| Zoom Out X | `Ctrl+-` | new shortcut; stays toolbar-only, not moved into a menu |
| Show Full Spectrum | `Ctrl+0` | new shortcut; stays toolbar-only, not moved into a menu |
| Clear All Fits | `Ctrl+Shift+C` | new shortcut; stays on the Fit Results context menu, not moved |
| Export All Fits... | `Ctrl+E` | new shortcut; stays on the Fit Results context menu, not moved |

Deliberately left without a shortcut: "Remove Fit" and "Export This Fit..." (both act on whichever Fit Results row was right-clicked — a keyboard shortcut would be ambiguous with no row-selection model), the spectrum panel's "Remove" context action (same reasoning), and the "Independent widths"/"Left tail" checkboxes (persistent fit settings, not one-shot commands).

## Menu Reorganization

**File**: Open... (`Ctrl+O`), Save Spectrum... (`Ctrl+S`, new), Recent Files, Exit (`Ctrl+Q`).

**View**: Log scale Y (`Ctrl+G`), Toggle Spectra panel (`Ctrl+1`), Dark theme (`Ctrl+D`). Calibration... is removed from here (moved below).

**Operations** (new menu): Calibration... (`Ctrl+L`, moved from View — opens the existing `CalibrationDialog`, unchanged behavior), Toggle Calibration Active (`Ctrl+T`, new menu item wrapping the existing toolbar toggle's logic so it's reachable without the toolbar), separator, Multiply by Factor... (`Ctrl+M`), Rebin by Factor... (`Ctrl+R`), Normalize Spectra (`Ctrl+N`).

The Fit Results/Fit Parameters panel toggles (`Ctrl+2`/`Ctrl+3`) stay wherever their existing tab-bar toggle actions live today (`fit_mode.py`'s `build_results_panel`/`build_parameters_panel`); they are not part of the View menu today and this spec doesn't move them into one.

## Multiply by Factor

**Trigger**: `Ctrl+M`, or Operations > Multiply by Factor.... Disabled (like Fit/Integrate) when there's no active spectrum.

**Dialog**: a single numeric input for the factor. Validates factor > 0 (rejects zero or negative with an inline error, matching `CalibrationDialog`'s validation style) — spectrum counts can't be negative, and a zero factor is almost certainly a mistake rather than an intentional "zero out this spectrum."

**Behavior**: `spectrum.data = np.round(spectrum.data * factor).astype(np.int64)`, applied to the active spectrum in place. Deletes that spectrum's committed fits and resets in-progress marks per "Fits/marks clearing semantics" above. Triggers a normal replot (`_plot_data(preserve_view=True)`, matching how other data-affecting operations already preserve the current zoom).

**Not undoable** — this changes the spectrum's in-memory data only; the file on disk is untouched unless the user separately uses Save Spectrum. Reloading the file recovers the original data.

## Rebin by Factor

**Trigger**: `Ctrl+R`, or Operations > Rebin by Factor.... Disabled when there's no active spectrum.

**Dialog**: a single integer input for the factor. Validates factor is an integer ≥ 2 (1 would be a no-op, rejected with an inline error).

**Behavior**: combines every `factor` adjacent channels into one by **summing** their counts (not averaging — total counts are conserved, matching the physical meaning of a count spectrum). If `len(data)` isn't evenly divisible by `factor`, the data is zero-padded at the end up to the next multiple of `factor` before grouping, so the last output bin sums fewer real channels than the others (the padding contributes zero) rather than silently dropping real counts via truncation. Result: `new_channel_count = ceil(old_channel_count / factor)`.

If a calibration is active, its coefficients are transformed to keep the keV axis correct: treating new channel *k* as representing old channel *k × factor* (the start of the group it summarizes), a calibration `E = a + b·ch + c·ch²` becomes `a' = a`, `b' = b × factor`, `c' = c × factor²` — an exact algebraic substitution (`ch_old = ch_new × factor`), not an approximation.

Deletes that spectrum's committed fits and resets in-progress marks per "Fits/marks clearing semantics" above (rebinning changes the channel count itself, so old fit-region channel bounds are not just stale but potentially out of range entirely). Triggers a full replot (not `preserve_view` — the channel/keV range itself has changed, so an old zoom window may no longer make sense; matches how applying a calibration already recomputes the view rather than blindly reusing stale bounds).

## Normalize Spectra

**Trigger**: `Ctrl+N`, or Operations > Normalize Spectra. Disabled when fewer than 2 spectra are currently visible.

**Marker input reuses the existing region-marking mechanism** (hold `R`, click) rather than introducing a new one:
- One pending `R`-click (`FitModeState.pending_fit_click` set, `fit_region` not yet complete): normalize using each visible spectrum's **count at that single channel**.
- A complete region (`FitModeState.fit_region` set, from two `R`-clicks): normalize using each visible spectrum's **summed counts across that channel range** (the area).

If neither is present (no marks placed at all), the action shows a status message asking the user to mark a channel or region first, rather than doing nothing silently.

**Behavior**: for every currently-visible spectrum, compute its reference value (bin count or area, per above). Find the maximum across all of them. For every visible spectrum *except* the one already at the maximum, scale its data by `max_value / this_spectrum's_value` (same rounding rule as Multiply) and update it in place. A spectrum whose reference value is exactly 0 can't be scaled to match a nonzero maximum by multiplication — it is skipped, with a status message naming which spectrum was skipped and why, rather than dividing by zero or silently doing nothing.

Deletes committed fits and resets in-progress marks (per "Fits/marks clearing semantics") only on spectra that were actually rescaled (factor ≠ 1) — the reference spectrum's data and fits are untouched, since nothing about it changed. After the operation, the in-progress marks used as input are also cleared (they've been consumed), matching how completing a Fit already resets marking state.

## Save Spectrum

**Trigger**: `Ctrl+S`, or File > Save Spectrum.... Disabled when there's no active spectrum.

**Dialog**: `QFileDialog.getSaveFileName` with a format filter offering all three writable formats, mirroring the existing Open dialog's filter style: `"SPE files (*.spe);;SPK files (*.spk);;Text files (*.txt);;All files (*)"`. The chosen filter (or the extension actually typed) determines which writer runs.

**Text (.txt)**: one integer count per line — the exact inverse of `histogram_io.load_histogram`.

**SPE (.spe)**: writes the Fortran-unformatted-record binary structure `spe_io.load_spe` already reads — a small header record (an 8-byte name field, `idim1` = channel count, `idim2`/`ired1`/`ired2` left at sensible placeholder values since they're read but never used) followed by a data record of the channel counts as little-endian 32-bit floats, each record framed by matching leading/trailing 4-byte length markers. This is not a native TV format (no reference C implementation exists to match byte-for-byte), so correctness here means round-tripping cleanly through this app's own `load_spe`, not matching an external tool.

**SPK (.spk)**: writes the "LC" container format `spk_io.load_spk` already reads, targeting the modern **LC2** compression variant (the vendored reference library's own default for new files, and what the one real `.spk` sample in this repo actually uses). Layout: a 44-byte little-endian header (magic, version=2, levels=1, lines=1, columns=N, and the position/length bookkeeping fields), an 8-byte position/length table entry (this app only ever writes single-spectrum files, so always exactly one entry), then the LC2-compressed payload. The compressor implements LC2's full tag scheme (single-value, same-run, 2-value, 3-value packing), not a simplified always-single-tag fallback, so file sizes are reasonably close to what the reference tool itself would produce. Calibration cannot be embedded in this or any of the three formats — none of them have a field for it; this is a hard format limitation, not a design choice made here.

Saving does not touch the spectrum's fits — this is about the raw channel data only, matching how Open only ever loads raw data. Exporting fit results is the existing, separate "Export This Fit.../Export All Fits..." mechanism.

## Testing

- **Shortcut/menu wiring**: every new `QAction` has the correct `setShortcut()` and lands in the correct menu; the Operations menu exists and contains exactly the listed items; Calibration... no longer appears under View.
- **`Ctrl+C` behavior change**: `test_clear_hides_every_fit_for_the_active_spectrum_without_deleting_it` is rewritten to assert deletion (`active.fits == []`) rather than every fit's `.visible` becoming `False`; a new or adjusted test confirms in-progress marks are still cleared too (the other half of `Ctrl+C`'s behavior, unchanged); confirm the unrelated "superseded fit" graying in `run_fit()`/`run_integration()` still works and still uses `.visible = False` (not touched by this change).
- **Multiply**: correct rounding for both integer and fractional factors; rejects factor ≤ 0 with the dialog staying open; deletes fits/resets marks on the active spectrum; leaves other loaded spectra untouched.
- **Rebin**: correct channel-count reduction and count-summing for evenly-divisible and non-evenly-divisible channel counts (zero-pad case); correct calibration coefficient transformation for both linear and quadratic calibrations, verified against hand-computed values; rejects non-integer or <2 factors; deletes fits/resets marks.
- **Normalize**: single-marker (bin) and two-marker (area) cases, both hand-verified against manually computed scale factors; correct skip-with-message behavior for a zero-reference spectrum; correct no-op on the already-maximum spectrum (factor stays 1, its fits untouched); disabled with <2 visible spectra.
- **Save**: each of the three writers round-trips through this app's own corresponding reader for a variety of spectra (small, large, all-zero, containing large values); the `.spk` writer's LC2 output is additionally verified against the real `demo.spk` fixture already in the repo (recompressing its known decompressed values and confirming byte-identical output, as already validated during this spec's research phase) and against every existing `.spk` reader test case in `tests/test_spk_io.py`, run in the reverse direction.

## Out of Scope

- Writing the legacy "oldmat" `.spk` sub-format (read-only forever; only the modern LC format gets a writer).
- Writing LC1-compressed `.spk` files (LC2 only, per the "resolved ambiguities" above — the reader keeps supporting both for files written by other tools).
- Embedding calibration coefficients in any saved file (no format has a slot for this).
- Undo for Multiply/Rebin/Normalize (none of the app's other data-affecting operations have undo either; the safety net is that the source file on disk is untouched unless the user separately saves).
- A keyboard shortcut for "Remove Fit," "Export This Fit...," the spectrum panel's "Remove," or the "Independent widths"/"Left tail" checkboxes (see Shortcut Table).
- Any change to how Fit/Integrate/Open already work, or to `Ctrl+C`'s in-progress-mark-clearing half — only its fit-handling half changes, per "Fits/Marks Clearing Semantics" above.
