# Background Line Display Fix + Ctrl+B Background Preview — Design Spec

Date: 2026-08-11

## Purpose

Two related changes to how the background is shown, both found/requested during user testing of the current master build:

1. **Bug: a committed fit's background line is drawn across the fit region, not across the background regions it's actually calculated from.** The line's slope/intercept are correct; only the drawn *extent* is wrong.
2. **New feature: a way to compute and preview the background fit using only the two background regions — no fit region, no peaks needed — bound to `Ctrl+B`, matching TV's own `"B"` key (`fit background-create`).** Confirmed by reading `tv-1.9.13/share/.tvkeys` directly, not assumed.

## Researched Before Designing: Current Behavior

Read directly from source:

- **`fit_mode.py`'s `draw_committed_fits()`, Fit branch (lines 632-638):** `lo, hi = result.fit_region` is used both for the background line's drawn span AND (a few lines later, line 646) for `x_dense = np.linspace(lo, hi, 200)`, the peak-curve's own span. These are two different, legitimately different spans that happen to reuse the same two local variables — the peak curve genuinely *should* stay clipped to the fit region (extrapolating a Gaussian shape into the background regions would look wrong), but the background line should not share that clipping.
- **`fit_mode.py`'s `draw_committed_fits()`, Integration branch (lines 603-610):** identical bug, `lo, hi = result.fit_region` used for the flat `background_density` line's drawn span. Nothing else in that branch depends on `lo`/`hi` afterward (lines 611-619 are just an `axes.annotate()` call), so this branch's fix is a clean, self-contained substitution.
- **`peak_fit.py`'s `_compute_background(x, y, left_bg_region, right_bg_region)`** (private, leading underscore): the actual background computation — mean-centroid of each region, two-point line through them. Takes no `fit_region` argument at all; it's already fully independent of the fit region and of peaks. Currently called only internally by `fit_peaks()` and `integrate_region()`, both of which separately *require* a `fit_region` argument for their own masking — that requirement is those functions' own, not `_compute_background()`'s.
- **`tv-1.9.13/share/.tvkeys:48`**: `"B"	tv> fit param backgr free; ... tv> fit background-create; ... tv> window show fit function/marker;` — TV's `B` key computes the background and shows it in a transient display, not as a persisted, separately-exportable artifact. `tv-1.9.13/doc/tex/Manual/cmd_summary.tex:409-410` confirms `fit background-create` "Creates a background fit" as its own concept, independent of a full peak fit.
- **`fit_mode.py`'s `_redraw_progress()`** (lines 456-507): the single place that draws every in-progress mark (pending clicks, B/R region shading, peak lines), fully rebuilt from `FitModeState` fields on every call. No data-derived drawing happens here today (only marks) — extending it to draw a computed background line is a new kind of thing for this function, but fits its existing "clear all progress artists, rebuild from state" structure directly.
- **`fit_mode.py`'s established blocked-reason pattern** (`fit_blocked_reason()`/`integrate_blocked_reason()`, `ready_to_fit()`/`ready_to_integrate()` derived from them): the precedent this project already uses for "action requires marks that aren't there yet," including the anti-drift fix (`ready_to_X()` just calls `X_blocked_reason() is None`) from a code review earlier this session.

## Resolved Decisions (confirmed with the user)

- **Background line display fix applies to both Fit and Integration** (same root cause, same fix shape) — not just the Fit case the user directly observed.
- **Ctrl+B is a lightweight, non-committed preview (Option 1)**: draws the background line, does not create a Fit Results entry, is not auto-logged, is not exportable. Confirmed over the alternative (a full third result type alongside Fit/Integration) as the better match for both TV's own transient-display behavior and the actual use case (sanity-checking the background before committing to a real fit).
- **No status message on a successful preview** (explicit user instruction: "even without status message"). A blocked-reason status message is still shown when Ctrl+B is pressed without both background regions marked — this reuses the exact pattern already shipped for Ctrl+F/Ctrl+I's own blocked-feedback (added specifically to fix "silently does nothing" confusion); removing it only for this one new action would reintroduce that same confusion for this specific case, which the "no status message" instruction wasn't asking for (it was about *success* feedback, in contrast to the two options being discussed at the time, not about removing the already-established failure feedback).
- **Ctrl+B toggles**: pressing it again while the preview is showing hides it, without needing `Ctrl+C` (which would also clear B/R/P marks). Not explicitly requested, but a minimal, low-risk addition consistent with this app's existing toggle-shortcut conventions (`Ctrl+D`, `Ctrl+G`, `Ctrl+T`).

## Architecture

Two files change: `peak_fit.py` (expose the existing background computation publicly) and `fit_mode.py` (the display fix, the new preview state/action, and the drawing logic). `help_content.py` gets the HowTo documentation update.

### `peak_fit.py` — make `_compute_background` public

Rename `_compute_background` to `compute_background` (drop the leading underscore) — its two existing call sites (inside `fit_peaks()` and `integrate_region()`) update to match. No behavior change; this purely makes it callable from `fit_mode.py`, which needs to call it directly for the new preview (the only other options, `fit_peaks()`/`integrate_region()`, both require a `fit_region` the preview doesn't have).

### `fit_mode.py` — background line display fix

In `draw_committed_fits()`, Fit branch, replace:
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
`lo, hi = result.fit_region` stays — it's still needed immediately after for `x_dense`'s peak-curve span, which must stay clipped to the fit region. Only the background line's own two endpoints change.

In the Integration branch, replace:
```python
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
if result.has_background:
    bg_line_lo, bg_line_hi = result.left_bg_region[0], result.right_bg_region[1]
    axes.plot(
        [to_display(bg_line_lo), to_display(bg_line_hi)],
        [result.background_density, result.background_density],
        color=bg_line_color, linestyle="--", linewidth=1,
    )
```

### `fit_mode.py` — `FitModeState`: new preview flag + blocked-reason pair

```python
def reset(self):
    self.bg_regions = []
    self.pending_bg_click = None
    self.fit_region = None
    self.pending_fit_click = None
    self.peak_positions = []
    self.show_background_preview = False
```
(`__init__` already calls `reset()`, so no separate initialization needed.)

In `add_bg_click()`, invalidate a stale preview whenever the background regions actually change (a newly completed pair, or the eviction of a third one) — insert right after `self.bg_regions.append(region)`:
```python
self.show_background_preview = False
```

New methods, mirroring `fit_blocked_reason()`/`ready_to_fit()`:
```python
def background_blocked_reason(self):
    if len(self.bg_regions) != BG_REGION_CAP:
        return "Mark two background regions (hold B and click twice per region) before previewing the background"
    return None

def ready_to_preview_background(self):
    return self.background_blocked_reason() is None
```

### `fit_mode.py` — `FitModeController`: new action + toggle method

Registered alongside the existing action setup (near `export_all_fits_action`):
```python
self.background_preview_action = QAction("Preview Background Fit", mw)
self.background_preview_action.setShortcut("Ctrl+B")
self.background_preview_action.triggered.connect(self.toggle_background_preview)
mw.addAction(self.background_preview_action)
```

New method:
```python
def toggle_background_preview(self):
    if not self.state.ready_to_preview_background():
        self._show_status_message(self.state.background_blocked_reason(), 5000)
        return
    self.state.show_background_preview = not self.state.show_background_preview
    self._redraw_progress()
```

### `fit_mode.py` — `_redraw_progress()`: draw the preview

New block, added after the existing peak-position loop (drawing order follows the function's existing bottom-to-top mark sequence: background regions, fit region, peaks, then this):
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
`axes.plot(...)` returns a list even for one line — `[0]` extracts the actual `Line2D` artist, required since `_clear_progress()` calls `.remove()` directly on each entry in `_progress_artists`. Black, distinct from the existing green/blue region-shading and gray pending-click markers, is used so the preview clearly reads as "the computed result," not another mark. `compute_background` needs importing from `peak_fit` alongside the module's existing imports from that file.

### `help_content.py` — HowTo documentation

Add a new row to the existing Fitting shortcuts table (after the `Ctrl+F` row):
```html
<tr><td><kbd>Ctrl+B</kbd></td><td>Preview the background fit from the two background regions alone (no fit region or peaks needed); press again to hide</td></tr>
```

Add a new paragraph to the "7. Performing a fit" section, after the existing `</ol>` and before the "See the Knowledge Database page..." paragraph:
```html
<p>Once both background regions are marked, <kbd>Ctrl+B</kbd> previews
just the background line -- no fit region or peaks needed -- useful
for sanity-checking the background before marking the rest. It's a
preview only: nothing is added to Fit Results, logged, or exported.
Press <kbd>Ctrl+B</kbd> again to hide it.</p>
```

## Error Handling Summary

- `Ctrl+B` with fewer than two background regions marked: status-bar message naming what's missing (same 5000ms duration as the existing Ctrl+F/Ctrl+I blocked messages), no crash, no partial preview drawn.
- `Ctrl+B` with two background regions marked but no active spectrum (shouldn't be reachable in practice, since marks require an active spectrum to have been clicked on): the preview silently doesn't draw (matches this function's existing defensive `if active is not None:` style elsewhere in the file) rather than raising.
- Toggling background regions (a third completed pair evicting the oldest) while a preview is showing: preview clears immediately (via the `show_background_preview = False` reset in `add_bg_click()`), rather than showing a stale line computed from regions that no longer exist.
- The background line display fix has no new error paths — same inputs, same computation, only the drawn endpoints change.

## Testing Plan

- **`tests/test_peak_fit.py`**: update references to `_compute_background` to the renamed `compute_background` (a couple of existing tests likely call it directly or indirectly — check during implementation); no new test needed for the rename itself since it's a pure rename with no behavior change, but confirm existing background-computation tests still pass under the new name.
- **`tests/test_fit_mode.py`**: `background_blocked_reason()`/`ready_to_preview_background()` tested directly against `FitModeState`, mirroring the existing `fit_blocked_reason()`/`integrate_blocked_reason()` tests — every combination of 0/1/2 background regions. A test confirming `add_bg_click()` resets `show_background_preview` to `False` when a region is completed while a preview was already showing.
- **`tests/test_fit_mode_ui.py`**: a test confirming `toggle_background_preview()` with two background regions marked draws exactly one new line artist (patch-count assertion, matching this project's established convention for this kind of test) at the correct endpoints (background regions' outer bounds, not the fit region); a test confirming a second call removes it (toggle-off); a test confirming the blocked-reason status message appears when marks are incomplete; a test confirming `draw_committed_fits()` for both a Fit result and an Integration-with-background result draws the background line spanning `left_bg_region[0]` to `right_bg_region[1]`, not `fit_region`.
- **`tests/test_help_content.py`**: extended to confirm the new Ctrl+B shortcut row and its exact wording, following this project's established exact-substring-assertion convention.
- **Windows/Linux verification**: same build+smoke-test pattern as every other change this session; this feature has no packaging implications, so the standard suite-plus-smoke-test is proportionate.

## Out of Scope

- No richer background model (polynomial degree, exponential component) — TV's own `fit background-create` macro toggles free/hold on parameters this app's simpler two-point-linear background doesn't have; not porting that complexity, only the "compute background independent of other markers" capability.
- No Fit Results panel entry, no auto-log, no export for the preview (that's the whole point of choosing Option 1).
- No menu or toolbar entry for the new action — keyboard-shortcut-only, matching `Ctrl+F`/`Ctrl+I`'s own existing precedent.
- No change to how `Ctrl+F`/`Ctrl+I` themselves compute or display backgrounds beyond the display-extent fix already described — their gating/requirements (still need two background regions, still need a fit region, `Ctrl+F` still needs peaks) are unchanged.
