# Fit Repeatability and Parameter Fixing — Design Spec

Date: 2026-07-14

## Purpose

A follow-up feedback round on the peak-fitting feature
(`docs/superpowers/specs/2026-07-10-peak-fitting-refinement-design.md`),
bundling several requests. This spec covers the first and largest of
four sub-projects agreed with the user:

1. **This spec:** a defined fit must be repeatable after changing
   conditions (independent widths, left tail, ...), with an option to
   fix any individual fit parameter, and complete uncertainty
   reporting for every parameter (not just position/FWHM/volume).
2. Export fit results (all parameters + uncertainties) to disk.
3. Fix the "Clear" toolbar button, which doesn't currently reset
   fit-related state the way the user expects.
4. A new "Integration" mode (TV-style direct background-subtracted
   region sum, not a Gaussian fit).

Sub-projects 2-4 are deferred to their own spec/plan cycles.

## Interaction Model Changes

**Marks are no longer cleared after a successful fit.** Background
regions, the fit region, and peaks stay live and visible after
clicking "Fit" — the marking state (`FitModeState`) is not reset.
This lets the user immediately toggle "Independent widths"/"Left
tail" (or fix/free parameters, see below) and click "Fit" again. Each
successful fit **always appends a new entry** to the active
spectrum's fit list (never replaces an existing one) — comparing
several attempts side by side, and discarding the ones you don't want
via the existing "Remove Fit" context-menu action, is the intended
workflow. "Clear" (fixed properly in sub-project 3) becomes the only
way to discard the live marks and start over.

**Double-clicking an entry in the Fit Results panel reloads it for
editing.** This copies that entry's background regions, fit region,
and peak positions back into the live marking state; restores the
"Independent widths"/"Left tail" checkboxes to match; and repopulates
the Fit Parameters panel (below) with that entry's fitted values,
with "Fix" checkboxes restored to match its `fixed_params`. The
reloaded marks redraw as the current live overlay. Re-fitting from
here appends yet another new entry — the original reloaded entry is
untouched unless separately removed.

**First fit is always fully free.** The initial fit for a given set
of marks always runs with no parameters fixed (this matches the
existing default: `link_widths=True`, `enable_left_tail=False`,
nothing fixed) — fixing only becomes available by checking boxes in
the Fit Parameters panel *after* that first fit has produced values
to anchor on.

## Fit Parameters Panel (new)

A new dockable panel, "Fit Parameters" (same `QDockWidget` +
persistent toolbar-tab pattern as the existing "Fit Results" panel —
see `FitModeController.build_results_panel`). It shows one row per
free/fixable parameter of the *last fit run for the current live
marks*:

- Per peak (in `peak_positions` order): `amplitude`, `position`, and
  — only when "Independent widths" is checked — `sigma`.
- One shared `sigma` row instead, when widths are linked (the
  default).
- `tail_fraction` and `tail_beta` rows, only when "Left tail" is
  checked.

Each row has a **Value** cell and a **Fix** checkbox:

- Unchecked (default): Value is a read-only display of the last
  fit's result for that parameter; free for the next fit.
- Checked: Value becomes editable (pre-filled with the last fitted
  value, editable to a different constant); held fixed at that exact
  value for the next fit, excluded entirely from the optimizer's free
  parameters.

**The panel is empty until the first fit runs** for the current
marks (per the confirmed flow: fit free first, then lock/free as
needed). Whenever the *set* of rows changes — a peak is added/removed,
or "Independent widths"/"Left tail" is toggled — the panel rebuilds
and all rows reset to unfixed, since the row identities themselves
changed and any prior fixed values may no longer correspond to
anything meaningful.

## Fitting Model Changes (`peak_fit.py`)

Replace the current fixed 4-combination parameter layout with a
named-parameter system: every parameter slot gets a canonical name
(`amp_0`, `pos_0`, `sigma_0` when unlinked or `sigma` when linked,
`amp_1`, `pos_1`, ..., `tail_fraction`, `tail_beta`), built in the
same order as today. `fit_peaks()` gains a new parameter:

```
fit_peaks(..., link_widths=True, enable_left_tail=False, fixed_params=None)
```

`fixed_params` is a `dict[str, float] | None` mapping any subset of
the canonical names above to a constant value. Names present there
are **excluded entirely** from `curve_fit`'s free-parameter vector
(not bounded to a point — cleanly removed), and substituted as
constants when the model function reconstructs the full parameter set
to evaluate the peak sum. This is a generalization: with
`fixed_params=None` or `{}`, behavior is unchanged from today, so the
existing 4 link/tail combinations still fall out of it as the
"nothing fixed" case.

**Uncertainty for a fixed parameter is exactly `0.0`** — it wasn't
optimized, so it carries no fit-derived error by definition. This
requires no special-casing in the existing volume/area uncertainty
formula (which already combines amplitude/sigma relative errors in
quadrature): a `0.0` term there is simply the mathematically correct
treatment of a value assumed exact.

## Data Model Changes

- `PeakResult` gains `amplitude_err: float` and `sigma_err: float`
  (currently missing — every reported parameter now has an
  uncertainty, including ones that are `0.0` for a fixed parameter).
- `FitResult` gains `fixed_params: dict[str, float]` (empty dict when
  nothing was fixed), recording exactly what was held fixed and at
  what value, needed for both the double-click reload and the later
  disk-export sub-project.

## UI Changes

- Add the "Fit Parameters" `QDockWidget` + toolbar-tab pair (mirrors
  the existing "Fit Results" panel construction).
- `run_fit()` no longer calls `_clear_progress()` after a successful
  fit; it reads the Fit Parameters panel's current Fix/Value state
  into a `fixed_params` dict passed to `fit_peaks()`.
- After every successful fit, the Fit Parameters panel is rebuilt (if
  the row set changed) or updated in place (if not) with the new
  fit's values.
- Double-click handling added to the Fit Results list widget.

## Testing

- `peak_fit.py`: fixing each parameter type individually and in
  combination (a peak's amplitude, position, or sigma; the shared
  sigma when linked; `tail_fraction`/`tail_beta`) — confirming the
  fixed value passes through unchanged (not perturbed by the
  optimizer) and its reported `_err` is `0.0`; confirming volume/area
  uncertainty correctly reflects a fixed sigma or amplitude.
- `fit_mode.py`/`main_window.py`: marks persisting (not cleared) after
  a successful fit; a second "Fit" click with changed
  checkboxes/fixed values appending a new entry rather than replacing
  one; the Fit Parameters panel rebuilding on peak/checkbox changes
  and updating in place otherwise; double-click reload restoring
  marks, checkboxes, and fixed/value state from an older entry.

## Out of Scope

- Disk export of results (sub-project 2, separate spec).
- The "Clear" button's actual bug fix (sub-project 3, separate spec)
  — this spec only changes *when* marks would need clearing, not
  fixes the button's existing broken behavior.
- The "Integration" mode (sub-project 4, separate spec).
- A right-side tail or the step-background term from gf3's full
  Hypermet model.
- Persisting Fix/Value state or checkbox states across app restarts.
- Automatic peak-finding.
- Editing a *free* parameter's initial guess by hand (only fixed
  parameters get an editable value; free parameters continue to use
  the existing internally-computed initial guess).
