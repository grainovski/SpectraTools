# Channel/Energy Calibration — Design Spec

Date: 2026-07-17

> **Note (2026-07-28):** the "Results Display and Export" section below describes the *original* design for the Fit Results table and Fit Parameters panel — a combined "X ch (Y keV)" string. After user testing, the on-screen panels (not the JSON/text export, which still uses the combined format as designed) were changed to switch units entirely — channels-only or keV-only, indicated via column header / row label — because the combined string didn't fit the real table columns. See `fit_mode.py`'s `_unit_switched_value`/`_panel_value_to_internal` and commit `dcc9d9a` for the current behavior.

## Purpose

User request: linear or quadratic channel-to-energy calibration, set via a
new calibration window (type selector, coefficients `a`, `b`, and `c` for
quadratic, entered manually or read from an ASCII file), with a switch
inside that window to apply the calibration to all loaded spectra,
transforming the plot's x-axis from channels to keV.

## Scope and Core Design Decision

Established through discussion, not assumption:

- **Calibration is global** — one active calibration applies to every
  loaded spectrum, not a per-spectrum setting (per the explicit "applies
  to all loaded spectra" requirement).
- **Channels remain the canonical, internally-stored unit everywhere.**
  `FitModeState`, `fit_peaks()`, `integrate_region()`, and every existing
  `FitResult`/`PeakResult`/`IntegrationResult` field never change — they
  stay channel-based exactly as today, regardless of whether a
  calibration is active. keV values are *derived* at two boundaries only:
  the plot (drawing and click interpretation) and the results display
  (table/tooltip/export). This was chosen deliberately over converting
  the fitting pipeline itself to operate in keV, because several
  TV-ported formulas (e.g. `_measure_width`'s
  `_MIN_PLAUSIBLE_FWHM_CHANNELS` threshold) implicitly assume channel
  units; keeping calibration out of `peak_fit.py` entirely avoids
  re-auditing that whole pipeline for unit-correctness.
- **Behavior when inactive**: identical to today — channels only,
  everywhere (plot axis, table, tooltip, export). No new code path is
  exercised at all when no calibration is active.
- **Behavior when active**: the plot's x-axis is redrawn in keV; marking
  clicks are interpreted against that keV axis and converted back to a
  channel number internally; the Fit Results table/tooltip/export show
  results in both channels and keV. This dual-unit display is computed
  at *view* time from each result's stored (always channel-based) values
  and the *current* calibration state, not frozen at fit time — see
  "Results Display and Export" below for what that means for a result
  computed before calibration was turned on.
- **Not in scope**: per-spectrum calibration, calibration persistence
  across app restarts (session-only), cubic-or-higher calibration, TV's
  "start channel" offset concept, and TV's separate width/tail/step/
  efficiency calibration types (`tv-1.9.13/lib/tv/vsCal.c` has all of
  these — this spec only ports the position/energy calibration).

## Calibration Math (`calibration.py`)

New module, mirroring `theme.py`'s role as a small, self-contained,
UI-framework-agnostic module.

```python
@dataclass
class Calibration:
    kind: str       # "linear" or "quadratic"
    a: float
    b: float
    c: float = 0.0  # unused for "linear"
```

- `apply(channel)` — forward evaluation, Horner's method:
  `E = a + b*channel + c*channel**2` (`c` is `0.0` for linear, so the
  same formula covers both without a branch). Matches TV's `CalP`
  (`tv-1.9.13/lib/tv/vsCal.c:981-988`, itself a Horner evaluation via the
  `CAL_COEF`/`HORNER` macros at lines 968-979).
- `derivative(channel)` — `b + 2*c*channel`, used both by Newton's-method
  inversion and by the results-display error propagation (see below).
  Matches the gradient computation inside TV's `CalC`
  (`vsCal.c:1066-1068`).
- `invert(energy)` — channel for a given energy, via Newton-Raphson,
  ported from TV's `CalC` (`vsCal.c:1047-1082`, the non-start-channel
  branch at lines 1072-1081, since this app doesn't have TV's
  "startchannel" concept): initial guess `x0 = (energy - a) / b` (a
  linear approximation regardless of `kind`, matching TV exactly), then
  iterate `x -= (apply(x) - energy) / derivative(x)` until
  `abs(apply(x) - energy) < 0.01` or 10000 iterations — TV's own
  `NEWTON_PRECISION`/`NEWTON_MAXITER` constants (`vsCal.c:15-16`), reused
  verbatim rather than re-derived, for exact parity with an already-
  source-verified reference (consistent with how this app has ported
  every other TV algorithm this project touches).
- **One deliberate deviation from TV**: `CalC` divides by `cfc[1]` (`b`)
  for the initial guess with no guard — if `b == 0`, TV's C would produce
  NaN and silently propagate it. This app validates `b != 0` at the point
  a `Calibration` is *set* (see dialog below) and refuses with a clear
  error instead, since this is a new UI entry point (not a raw port of an
  existing TV-driven flow where matching a silent failure mode would
  matter for parity) and a NaN channel from a bad calibration would
  otherwise corrupt marking silently.

## Calibration Dialog (`calibration_dialog.py`, UI)

A `QDialog`, opened via a new **`Calibration...`** action in the `View`
menu (near `Dark theme`, since this is a display/analysis toggle in the
same family, not a file operation).

- **Type selector**: radio buttons, "Linear" / "Quadratic".
- **Coefficient fields**: `a` and `b` always visible and editable; `c` is
  shown and editable only when "Quadratic" is selected (hidden/disabled
  for "Linear" — same show/hide pattern the Fit Parameters panel already
  uses for mode-dependent rows).
- **"Load from file..." button**: opens a file dialog, reads an ASCII
  file (format below), and populates the coefficient fields. Loading
  only pre-fills — the fields stay directly editable afterward, so
  "load then tweak" and "enter manually" both go through the same
  inputs.
- **"Active" checkbox** (the requested switch): enables/disables applying
  this calibration. Lives inside this dialog, not a separate quick-access
  menu toggle (per explicit user preference) — unchecking it and clicking
  OK deactivates calibration but keeps the coefficients for next time the
  dialog is opened.
- **OK**: validates (`b != 0`; for quadratic, `c` must parse as a number,
  including `0.0`) and applies — updates `MainWindow`'s calibration state
  and triggers a re-plot. Validation failure keeps the dialog open with
  an inline error message, doesn't apply anything.
- **Cancel**: discards any edits made in this dialog session; the
  previously-active calibration (if any) is untouched.
- Reopening the dialog later shows the last-set values (kind,
  coefficients, active state), whether currently active or not.

### ASCII File Format

One coefficient per line, plain numbers, no labels or keywords:

```
20.5
0.487
```

for linear, or

```
20.5
0.487
0.0002
```

for quadratic. A file with the wrong number of lines for the currently-
selected type, or non-numeric content, is rejected with a clear error
(dialog fields are left unchanged, not partially overwritten).

## Plot and Marking Integration

- **Axis transform** (`main_window.py`, `_plot_data()`): when calibration
  is active, each spectrum's x-coordinates become
  `calibration.apply(np.arange(len(data)))` instead of raw channel
  numbers; the x-axis label switches from `"Channel"` to
  `"Energy (keV)"`. Inactive: unchanged from today.
- **Click → channel inversion** (`fit_mode.py`, `on_click()`): matplotlib
  always reports `event.xdata` in whatever's plotted on the axis — keV,
  once calibration is active. When active, `on_click()` converts
  `event.xdata` to a channel number via `calibration.invert(event.xdata)`
  *before* handing it to `FitModeState.add_bg_click`/`add_fit_click`/
  `toggle_peak`, which — like every other channel-space consumer in this
  app — never change and never see keV. This is the only new touchpoint
  in the entire click-handling path.
- **Redrawing marks and committed fits**: `_redraw_progress()`
  (in-progress b/r/p marks) and `draw_committed_fits()` (committed
  `FitResult`/`IntegrationResult` shading, curves, position lines,
  annotations) currently draw directly at their stored channel-based
  x-values. When calibration is active, each of those x-values is passed
  through `calibration.apply()` immediately before the corresponding
  `axes.axvspan`/`axes.plot`/`axes.axvline`/`axes.annotate` call — the
  stored values themselves (`state.fit_region`, `result.peaks[i].position`,
  etc.) are never touched.
- **Toggling mid-session is always safe**: since marks and results stay
  channel-based internally regardless of calibration state, turning
  calibration on/off at any point — mid-marking, or with committed fits
  already on the plot — only changes where things are drawn on the next
  redraw. Nothing needs to be invalidated, recomputed, or blocked.

## Results Display and Export

Dual-unit display is purely a formatting-layer concern, computed at
display time from the existing (always channel-based) stored values —
not a new stored field on `FitResult`/`PeakResult`/`IntegrationResult`.
This means it's also **retroactive**: a result computed while calibration
was inactive still gains a keV column/parenthetical the next time it's
displayed, if calibration happens to be active *then* — "if calibration
is not active, spectra and fits are in channels only" describes what you
see given the *current* calibration state, not a frozen property of when
a given fit was originally run.

- **Fit Results table** (`fit_mode.py`, `update_results_list()`):
  Position and FWHM columns show both units when calibration is active,
  e.g. `245.32 ± 0.10 ch (342.62 ± 0.14 keV)`. Volume and chi^2 stay
  channel-only in both cases — an area/counts quantity has no meaningful
  single energy-axis equivalent, and converting it would misrepresent
  the counts.
- **keV uncertainty**: first-order error propagation through the
  calibration's local derivative —
  `err_keV = abs(calibration.derivative(channel)) * err_channels`.
  Exact for linear calibration; a good approximation for quadratic given
  realistic peak-position uncertainties are small relative to the
  calibration's curvature scale.
- **Tooltip**: the existing region-level/peak-level breakdown lines gain
  the same keV parenthetical for centroid/FWHM-style values; area lines
  stay channel-only, matching the table.
- **Export (JSON)**: JSON records gain `position_keV`/
  `position_err_keV`/`fwhm_keV`/`fwhm_err_keV` fields, present (non-null)
  only when calibration was active at export time — absent/`None`
  otherwise, so existing consumers of the auto-log aren't broken by new
  always-required fields.
- **Export (text report)**: same parenthetical format as the table.

## Testing

- **`calibration.py`**: `apply()` matches hand-computed values for both
  linear and quadratic; `invert(apply(ch)) == ch` round-trips across a
  range of channels (including near the low/high ends of a typical
  spectrum) for both kinds; `invert()` matches a hand-verified
  Newton's-method worked example (same cross-check style used for this
  session's other TV-parity ports); setting `b=0` raises a clear error
  rather than propagating NaN.
- **Calibration dialog**: Linear ↔ Quadratic toggle shows/hides the `c`
  field; "Load from file..." on well-formed 2-line and 3-line files
  populates the right fields; a malformed file (wrong line count,
  non-numeric content) shows an error and leaves the fields unchanged;
  OK with `b=0` is rejected with a message and the dialog stays open;
  Cancel leaves a previously-active calibration untouched.
- **Plot/marking integration**: with calibration active, `_plot_data()`
  draws at `calibration.apply(channel)` x-positions and the axis label
  reads `"Energy (keV)"`; a simulated click at a given pixel position
  resolves to the correct channel through `on_click()` (mirroring this
  session's existing click-simulation test helpers); marks made *before*
  calibration is turned on remain valid and redraw at the correct keV
  position once it's turned on, and back at the correct channel position
  if turned off again.
- **Results display**: table/tooltip/export show dual units only when
  calibration is active, with keV values matching `calibration.apply()`
  on the stored channel-based fields directly, and keV uncertainties
  matching the derivative-scaled propagation formula, hand-verified
  against at least one worked linear and one worked quadratic example.

## Out of Scope

- Per-spectrum calibration (global only, per explicit requirement).
- Calibration persistence across app restarts (session-only).
- Cubic-or-higher-order calibration.
- TV's "start channel" offset concept.
- TV's separate width/tail/step-height/step-width/efficiency calibration
  types (`tv-1.9.13/lib/tv/vsCal.c`) — this spec is position/energy
  calibration only.
- Converting Volume/chi^2/area quantities to an energy-axis equivalent.
