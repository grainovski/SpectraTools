# Integration Without Background Regions — Design Spec

Date: 2026-08-06

## Purpose

First feature of version 2.1.2 ("a few more minor fixes"). Currently, `Ctrl+I` (Integration) requires exactly two background regions (`B` marks) plus a fit region (`R` mark) before it will run at all — `FitModeState.ready_to_integrate()` hard-requires `len(bg_regions) == BG_REGION_CAP` (2). This means a user who only wants a raw region sum and centroid, with no background subtraction, has no way to get one: they either mark two background regions they don't actually want subtracted, or they can't use Integration at all.

This feature makes Integration work with **zero** background regions marked (a fit region alone is enough), in which case it reports the raw (gross) sum and centroid — no background subtraction, since there's nothing to subtract.

## Researched Before Designing: How TV Does This

This project's foundation is behavioral parity with TV (`tv-1.9.13/`), and `integrate_region()` in `peak_fit.py` is explicitly a line-by-line port of TV's `FIIntegrateRegion` (`tv-1.9.13/lib/tv/vsFitInt.c:19-51,194-309`). TV's own source was read directly before designing anything here, not assumed from prose.

**TV already supports this, natively, with no special-casing needed:**

- TV's `FIIntegrateRegion` never requires a specific background-region count. It checks `VecNum(FitBgMrk(fp)) != 0` (`vsFitInt.c:194`) — any nonzero count, not `== 2`. The fit *region* mark is mandatory (`Mrk_Check("region", ...)`, `VsFit.c:504`); background marks are entirely optional, with no equivalent check anywhere in the integration call path (`Fit_IntegrateExec`, `VsFit.c:486-513`).
- The background computation is a real `if (FitBgTime(fp)) {...} else if (VecNum(FitBgMrk(fp)) != 0) {...}` with **no trailing `else`** (`vsFitInt.c:85,194`, confirmed by reading through to the function's closing brace at line 314). When neither condition holds, that entire ~225-line block is skipped outright — background/net integrals are never touched after being zeroed by `FitIntReset` at the top of the function (`vsFitInt.c:32-34`, `vsFit.c:158-172`). This is a skip, not a "loop over zero regions that happens to sum to zero."
- The gross/"total" sum and its moments (`vsFitInt.c:36-83`) run **unconditionally**, regardless of background — this part of the function doesn't change at all between the two cases.
- Critically, **TV's own report format doesn't print background/net as zero when there's no background — it omits those rows entirely.** `ParseIntPeaks` (`vsFitFmt.c:310-327`) guards the `'b'` (background) and `'s'` (subtracted/net) format directives with `if (!VecNum(FitBgMrk(fp))) { return NULL; }` — in this file's convention, `NULL` means "nothing to print here, not an error," and critically this skips not just the numbers but the row's own label (printed from inside the now-never-called `ParseIntPeak`). The `'t'` (total) directive has no such guard and always prints. TV's actual default report templates (`VsFitWrite.c:40-45`, `share/.tvinthd.l:7-10`) unconditionally *attempt* all three rows every time — it's this guard that makes the background/net rows silently vanish rather than show as 0.
- There is no separate TV command for "integrate without background" — it's the same `fit integration-create` command (`VsFace.h:1764`) either way, branching internally exactly as described above.

This directly resolves the two decisions below in favor of matching TV: suppress the background/net breakdown entirely rather than showing it as zero, and skip the background computation block rather than looping over zero regions.

## Resolved Decisions (through discussion with the user)

- **Display**: when there's no background, the result is shown as a **single row/value set** (area, centroid, FWHM, skewness) — no "Gross: X / Background: 0 / Net: X" breakdown. Matches TV's own suppress-the-row behavior, and matches the user's own framing of the request ("it returns the net area... and the centroid," not "returns gross/background/net with background and net-minus-gross both zero").
- **Included fields**: the single-row result still includes **FWHM and skewness**, not just area and centroid — these come from the same already-computed moments at no extra cost, and TV's own "total" row always includes them alongside position and volume.
- **Region-count gating**: Integration is enabled with **exactly 0 or exactly 2** background regions marked — not 1. An incomplete pair has no defined meaning here and wasn't requested; this is a narrower rule than TV's own "any N ≥ 1, pooled," which is not being adopted (out of scope, see below).

## Architecture

No new files, no new architectural pattern — this extends the existing `peak_fit.py` / `fit_mode.py` / `fit_export.py` pipeline that already handles the with-background case.

### `peak_fit.py`

**`IntegrationResult`**: `left_bg_region` and `right_bg_region` become `Optional[tuple] = None` (currently required `tuple` fields). A new read-only helper is added rather than repeating the `is None` check at every call site:

```python
@dataclass
class IntegrationResult:
    left_bg_region: tuple | None
    right_bg_region: tuple | None
    ...  # unchanged otherwise

    @property
    def has_background(self):
        return self.left_bg_region is not None
```

**`integrate_region(x, y, left_bg_region, right_bg_region, fit_region)`**: `left_bg_region`/`right_bg_region` become optional (`None` accepted). The gross-sum/gross-moments block (`vsFitInt.c:36-83`'s port, currently unconditional) stays unconditional — unchanged. The background-pooling block (currently a bare `for region in (left_bg_region, right_bg_region):` loop) is wrapped in `if left_bg_region is not None and right_bg_region is not None:` — mirroring TV's real `if`/`elif`-with-no-`else` branch structure, not relying on "loop happens to do nothing over an empty/None pair" (which would need different, more fragile guarding anyway since the current loop iterates a fixed 2-tuple of regions, not a variable-length list). When skipped, `background_area`/`background_area_err`/`background_density` stay at their pre-loop-declared zero defaults, and the subsequent background/net moments block (guarded by its own existing `if net_sum != 0.0 or background_area != 0.0:`) computes net moments from `background_density = 0` — no code changes needed to that block itself.

  For a region with a positive gross sum (the overwhelmingly normal case — a real count region being integrated), this makes `net_area`/`net_centroid`/`net_fwhm`/`net_skewness` come out bit-for-bit equal to their `gross_*` counterparts, since `net_sum` reduces to exactly `gross_sum` and the moment formulas' only asymmetry between gross and net is `abs()` on the divisor, which is a no-op for a positive value. **This does NOT hold in general** for a region whose gross sum is negative or zero — the gross moments divide by `gross_sum` directly (no `abs()`, `peak_fit.py`'s existing, pre-existing-and-unchanged "plain sum, not abs" choice) while net's divide by `abs(net_sum)`, so a negative-sum region would see gross and net moments differ in sign. This asymmetry already exists today in the with-background path (it's a confirmed, deliberate TV-parity quirk, not something introduced by this feature) and is called out here only because it's newly *reachable* in a no-background integration of a spectrum that itself has negative values — e.g. one produced by v2.1.0's Subtract Spectra feature, whose negative results are deliberately unclamped. Not fixing this pre-existing quirk is in scope for "unchanged," but the single-row display (see below) reads from `gross_*` specifically, sidestepping the question of which of the two disagreeing values would even be "more correct" to show.

### `fit_mode.py`

**`FitModeState.ready_to_integrate()`**:
```python
def ready_to_integrate(self):
    return self.fit_region is not None and len(self.bg_regions) in (0, BG_REGION_CAP)
```

**`FitModeState.ordered_bg_regions()`**: returns `(None, None)` when `len(self.bg_regions) == 0` (currently unconditionally unpacks `a, b = self.bg_regions`, which would raise `ValueError` on an empty list).

**`run_integration()`**: no logic change needed beyond what `ready_to_integrate()`/`ordered_bg_regions()` already provide — `integrate_region(x, y, left, right, self.state.fit_region)` is called the same way, just potentially with `left=right=None` now.

**`main_window.py`**: no changes — `self.integrate_button.setEnabled(available and self.fit_controller.state.ready_to_integrate())` already delegates entirely to `ready_to_integrate()` (confirmed directly, not assumed), so fixing that one method is sufficient to correctly enable the button for the 0-region case too.

**Plot drawing** (the `isinstance(result, IntegrationResult)` branch that draws region spans and an annotation): currently draws `axvspan` for `result.left_bg_region`/`right_bg_region` unconditionally — wrapped in `if result.has_background:` (also skips the background-density dashed line in that case, since there's no meaningful density to show). The annotation text currently reads `centroid=X\nFWHM=X\nfull=X\nnet=X` — when `not result.has_background`, collapses to `centroid=X\nFWHM=X\narea=X` (one number, not two identical ones).

**`_integration_tooltip()`**: currently unconditionally loops over `("Gross", "gross"), ("Background", "background"), ("Net", "net")` and joins three lines. When `not result.has_background`, returns a single line using the `gross_*` fields (equal to `net_*`, but reading from `gross_*` avoids relying on the "net happens to equal gross" invariant at a display layer, matching TV's own choice to report the *total*/gross fields — not the also-zero-background-corrected net fields — as the single row in this case) with no "Gross"/"Net" qualifier — just e.g. `f"Area: {area:.1f}±{area_err:.1f}, centroid=..., FWHM=..., skewness=..."`.

**Fit Results table** (`update_results_list()`'s `IntegrationResult` branch): **no change**. It already reads `result.net_centroid`/`result.net_fwhm`/`result.net_area` under generic column headers ("Position"/"FWHM"/"Volume", not "Net Position" etc.) — these are mathematically equal to gross when there's no background, so the existing display is already correct without modification. (The tooltip attached to this row's first cell does change, per above.)

### `fit_export.py`

Both the JSONL auto-log writer and the human-readable report writer currently do unconditional things like `list(result.left_bg_region)` and `f"...{result.left_bg_region[0]:.2f}..."` — these raise `TypeError`/`AttributeError` on `None` today and must be guarded regardless of the display-style decision above. Matching the same suppress-don't-zero choice: when `not result.has_background`, the background-region lines, background density line, and the Background/Net breakdown lines are omitted from the human-readable report (only a Gross/Area-equivalent line remains), and the JSONL record omits (rather than nulls) `left_bg_region`/`right_bg_region`/`background_*`/`net_*` keys, keeping only `gross_*` plus the fields that don't depend on background (`fit_region`, `timestamp`, etc.). Exact key list to be pinned down in the implementation plan against the current JSONL schema.

### `help_content.py`

The existing "7. Integration" HowTo section (renumbered from "6." during v2.1.0's Add/Subtract work) gets a sentence added explaining that background marks are now optional: marking only a fit region and pressing `Ctrl+I` reports the raw area and centroid with no background subtraction, while marking two background regions first still works exactly as before.

## Error Handling Summary

- **No fit region marked**: `Ctrl+I` stays disabled, same as today — unchanged.
- **Exactly 1 background region marked**: `Ctrl+I` stays disabled (not a valid state) — this is a *new* rule (today, 0 or 1 regions both leave it disabled identically; after this change, 0 becomes valid but 1 alone still doesn't).
- **0 or 2 background regions marked, fit region marked**: `Ctrl+I` enabled either way.
- **Export/report with no background**: no crash (the current unconditional indexing would otherwise raise) — background-specific lines/keys are omitted, not written as null/zero.

## Testing Plan

- **`tests/test_peak_fit.py`**: `integrate_region()` called with `left_bg_region=None, right_bg_region=None` on a realistic positive-count region — result's `background_area`/`background_density` are exactly `0.0`, and `net_area == gross_area`, `net_centroid == gross_centroid`, `net_fwhm == gross_fwhm` (bit-for-bit, not just approximately, since the reduction is algebraic for a positive gross sum — see the architecture section's note on the negative-sum edge case, which is pre-existing behavior this feature doesn't need to newly test) — plus the existing with-background tests continue passing unchanged (no regression). A test for `has_background` being `False`/`True` in each case.
- **`tests/test_fit_mode_ui.py`**: `ready_to_integrate()` returns `True` for 0 background regions + a fit region, `False` for exactly 1 background region + a fit region (even though 0 and 2 are both valid, 1 must not be), `ordered_bg_regions()` returns `(None, None)` for the zero case. `run_integration()` end-to-end with zero background regions produces a result with `has_background is False`. Plot-annotation and tooltip content tested for both branches (single-line vs three-line).
- **`tests/test_fit_export.py`**: JSONL and human-readable report generation for a no-background `IntegrationResult` — confirm no crash (the concrete regression this whole error-handling section exists to prevent) and confirm the expected keys/lines are the ones actually omitted, not just "doesn't crash."
- **HowTo page**: existing `tests/test_help_content.py` balanced-tags/content checks extended to cover the updated Integration section text, matching this project's standing convention for every Operations/Fit-menu feature.
- **Windows** (primary focus, real manual verification): rebuild, load a real spectrum, mark only a fit region (no background), run Integration, confirm a single-row result with sensible area/centroid/FWHM/skewness and no background clutter in the tooltip/plot annotation/export; then mark two background regions and confirm the existing with-background path is unchanged.
- **Linux** (proportionate, matching this project's established practice for pure application-code changes): rebuild via existing `build.sh`/`build_deb.sh`, launch-alive smoke test on Ubuntu-24.04 and AlmaLinux-10 — no new packaging surface here, so no deeper Linux-specific verification is warranted.

## Out of Scope

- TV's more permissive "any N ≥ 1 background regions, pooled together" model — not adopted. This project's own `BG_REGION_CAP = 2` and paired-region UI (mark two regions to define a background line/level) stays exactly as-is for the with-background case; this feature only adds a 0-region path, not a 1-region or N-region one.
- Any change to `fit_peaks()` (the peak-shape-fitting path) or its own background handling (`_compute_background()`'s sloped-line model) — that function already has its own, unrelated background requirement and isn't touched by this feature at all.
- Any change to the with-background Integration path's numbers, display, or export format — every with-background test and behavior stays byte-for-byte identical; this is purely additive for the zero-background case.
