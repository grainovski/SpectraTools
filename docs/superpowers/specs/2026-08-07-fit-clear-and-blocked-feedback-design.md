# Ctrl+C Non-Destructive Clear + Blocked-Action Feedback — Design Spec

Date: 2026-08-07

## Purpose

Two small, related fixes to Fit mode, found during Windows verification of v2.1.2's Integration-without-background feature:

1. **`Ctrl+C` ("Clear") currently permanently deletes all of the active spectrum's committed fits** (`active.fits.clear()`), removing them from both the plot and the Fit Results panel with no way to get them back short of re-fitting from scratch. This is more destructive than a "clear the workspace" shortcut should be, given the app already has a non-destructive "hide" mechanism it uses internally for a different case (see below).
2. **Pressing `Ctrl+F` (Fit) or `Ctrl+I` (Integrate) with incomplete marks does nothing at all, silently.** This directly caused a support round-trip during Windows verification: the user marked B/R but forgot P, pressed Ctrl+F, and got no feedback of any kind — indistinguishable from the app being broken. A related case (pressing P before a fit region is marked) already shows a helpful status-bar message; the analogous case for the action shortcuts themselves does not.

## Researched Before Designing: Current Behavior

Read directly from `fit_mode.py`, not assumed:

- **`FitModeController.clear()`** (lines 394-399, triggered by `Ctrl+C`): calls `self.reset_marks()` (clears in-progress B/R/P marks — unaffected by this spec) then `active.fits.clear()` (empties the committed-fits list entirely) then `self.main_window._plot_data(preserve_view=True)`.
- **`FitModeController._clear_all_fits()`** (lines 992-997, triggered by `Ctrl+Shift+C` / the results panel's "Clear All Fits" context-menu action): also calls `active.fits.clear()`, but does not touch in-progress marks. Structurally near-identical to `clear()`'s fits-clearing half.
- **The `visible` field already exists on both `FitResult` and `IntegrationResult`** (`peak_fit.py`) and already has an established, working "hide, don't delete" behavior: when a new fit/integration is committed at the exact same marked region as an earlier one, the earlier one gets `visible = False` instead of being removed (`run_fit()` lines 1120-1126, `run_integration()`'s equivalent block). Downstream, `draw_committed_fits()` (`fit_mode.py`) skips drawing any result with `visible == False` entirely, while `update_results_list()` still lists it in the Fit Results panel, rendered in gray (`item.setForeground(QColor("gray"))`). `_export_all_fits()` includes hidden results in its export, same as visible ones. Double-clicking a hidden result's row to reload it for editing works identically to a visible one (`_on_result_double_clicked()` doesn't check `visible` at all).
- **`main_window.py`'s `_plot_data()`** (lines 789-830) already calls `self.fit_controller.update_results_list()` as its own last line. Every caller of `_plot_data()` — including `clear()` — therefore already gets the Fit Results panel refreshed for free; no separate explicit call is needed.
- **`FitModeState.ready_to_fit()`** (`fit_mode.py:96-101`): `len(bg_regions) == BG_REGION_CAP and fit_region is not None and len(peak_positions) > 0`. **`ready_to_integrate()`** (`fit_mode.py:103-104`): `fit_region is not None and len(bg_regions) in (0, BG_REGION_CAP)`.
- **`run_fit()`** (line 1075) and **`run_integration()`** (line 1137) both start with `if not self.state.ready_to_XXX(): return` — a bare, silent return with no user feedback.
- **Existing precedent for this exact kind of message already exists**, one level down: `on_click()`'s `"p"` branch (lines 525-...) shows `self._show_status_message("Mark the fit region (hold R and click twice) before marking peaks", 3000)` when a peak-mark click happens before a fit region exists. `run_fit()`'s own `except FitError` branch shows `self._show_status_message(f"Fit failed: {exc}", 5000)` when marks are complete but the fit itself fails. Only the "marks are incomplete, action shortcut pressed anyway" case has no message today.

## Resolved Decisions (confirmed with the user)

- **`Ctrl+C` hides instead of deletes**, reusing the existing `visible` mechanism exactly as-is (same panel styling, same plot-skipping, same export inclusion, same double-click-restore) — no new rendering code, no new mechanism.
- **`Ctrl+Shift+C` ("Clear All Fits") is unchanged** — it remains the true, permanent delete. This is the answer to "keep the option" to actually delete fits.
- **Hiding via `Ctrl+C` is one-way**, matching the existing superseded-fit precedent exactly (which also has no "restore" affordance). No new UI is added to un-hide a fit. `Ctrl+Shift+C` remains the way to actually get rid of hidden fits for good.
- **Blocked-action feedback is added for both `Ctrl+F` and `Ctrl+I`**, mirroring the existing peak-marking hint's style and tone, naming specifically what's still missing rather than a generic "can't fit yet" message.

## Architecture

No new files. Both changes are confined to `fit_mode.py`, plus one doc-accuracy fix in `help_content.py`.

### `fit_mode.py` — `clear()`

Replace the body's fits-clearing line:
```python
active.fits.clear()
```
with:
```python
for result in active.fits:
    result.visible = False
```
Nothing else in `clear()` changes — `reset_marks()` and the closing `_plot_data()` call (which already refreshes the results panel) stay exactly as they are. `_clear_all_fits()` (`Ctrl+Shift+C`) is untouched.

### `fit_mode.py` — `FitModeState`: two new reason-lookup methods

Mirroring `ready_to_fit()`/`ready_to_integrate()`'s own conditions, in the order a user naturally completes marks (a fit region has to exist before peaks can even be marked, per `toggle_peak()`'s own guard — so checking it first always points at genuinely the next missing thing, never a redundant one):

```python
def fit_blocked_reason(self):
    if self.fit_region is None:
        return "Mark the fit region (hold R and click twice) before fitting"
    if len(self.bg_regions) != BG_REGION_CAP:
        return "Mark two background regions (hold B and click twice per region) before fitting"
    if len(self.peak_positions) == 0:
        return "Mark at least one peak (hold P and click) before fitting"
    return None

def integrate_blocked_reason(self):
    if self.fit_region is None:
        return "Mark the fit region (hold R and click twice) before integrating"
    if len(self.bg_regions) not in (0, BG_REGION_CAP):
        return "Mark zero or two background regions (not one) before integrating"
    return None
```

Each returns `None` under exactly the same condition its corresponding `ready_to_XXX()` returns `True` — the two are logically complementary by construction (same underlying checks, inverted). This means `run_fit()`/`run_integration()` can call the reason method directly, with no `is not None` guard, once `ready_to_XXX()` has already been confirmed `False` — there is no reachable state where the gate fails but the reason method returns `None` anyway.

### `fit_mode.py` — `run_fit()` / `run_integration()`

Replace:
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
and the analogous change to `run_integration()`'s opening two lines, calling `integrate_blocked_reason()`. Nothing else in either method changes. `5000` ms matches the existing `FitError`-failure message's duration (both are "the action didn't happen" feedback), rather than the shorter `3000` ms used for the lower-stakes in-progress peak-marking hint.

### `help_content.py` — doc accuracy

Two spots currently describe `Ctrl+C` as deleting fits, which becomes wrong once this ships:
- The shortcuts table row (currently *"Clear the active spectrum's in-progress marks and committed fits"*).
- The prose paragraph in the "Performing a fit" section (currently *"Ctrl+C clears the active spectrum's in-progress B/R/P marks as well as its already-committed fits."*).

Both get reworded to describe the new hide-not-delete behavior, and to state plainly that `Ctrl+Shift+C` is the permanent-delete option — exact replacement text is pinned down in the implementation plan.

## Error Handling Summary

- Pressing `Ctrl+F`/`Ctrl+I` with any marks incomplete: no longer a silent no-op — shows a specific status-bar message naming the next missing mark, same duration/style as the existing `FitError` failure message.
- `Ctrl+C` with zero committed fits: the `visible = False` loop runs over an empty list — a no-op, same as today's `active.fits.clear()` on an empty list.
- Everything about `Ctrl+Shift+C`, the right-click "Remove Fit"/"Export This Fit"/"Export All Fits" actions, and double-click-to-reload stays byte-for-byte unchanged.

## Testing Plan

- **`tests/test_fit_mode.py`**: `fit_blocked_reason()` and `integrate_blocked_reason()` each tested directly against `FitModeState` for every combination that should return a specific string vs. `None`, confirming exact wording and confirming `None` if and only if the corresponding `ready_to_XXX()` is `True`.
- **`tests/test_fit_mode_ui.py`**: `Ctrl+C` (`clear()`) with committed fits present — confirm `active.fits` still has the same length afterward (nothing deleted), every result's `visible` is `False`, the plot shows no fit-region/background shading for them (patch-count assertions, matching this project's established convention for this exact function), and the results-panel rows are still present but grayed out. A companion test confirms `Ctrl+Shift+C` is completely unchanged (still empties `active.fits`). Two new tests confirm `run_fit()`/`run_integration()` show the exact expected status message for a couple of representative incomplete-marks states, and confirm a *complete* set of marks still fits/integrates with no message shown (no regression to the success path).
- **`tests/test_help_content.py`**: extended to confirm the reworded Ctrl+C description no longer claims fits are deleted and does state that Ctrl+Shift+C is the permanent option, following this project's standing convention of testing HowTo content for every behavior change.
- **Windows** (primary focus, real manual verification): commit a fit, press Ctrl+C, confirm it's grayed out in Fit Results (not gone) and gone from the plot; press Ctrl+Shift+C, confirm it's now actually gone from the panel too. Separately, mark an incomplete set of B/R/P and press Ctrl+F (and Ctrl+I), confirm a status-bar message appears naming what's missing.
- **Linux** (proportionate): full automated suite via the existing `build.sh`/`build_deb.sh` + launch-alive smoke test — no new packaging surface, consistent with how every other pure-application-code change in this project has been verified on Linux.

## Out of Scope

- No new UI to restore/un-hide a `Ctrl+C`-hidden fit — matches the existing superseded-fit precedent, which has never had one either.
- No change to the right-click "Remove Fit" (single-fit permanent delete), "Export This Fit...", or "Export All Fits..." actions.
- No change to `FitError`-triggered messages (bad data / non-convergent fit) — those already show feedback today; this spec only adds feedback for the separate, currently-silent "marks incomplete" gate.
