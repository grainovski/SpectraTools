# Peak Fitting Refinement — Design Spec

Date: 2026-07-10

## Purpose

A refinement round for the peak-fitting feature (`docs/superpowers/specs/2026-07-09-peak-fitting-design.md`,
`docs/superpowers/plans/2026-07-09-peak-fitting.md`), driven by hands-on
testing feedback after the v1 implementation shipped. Two genuine bugs
from that testing round were already fixed directly (the plot resetting
to the full spectrum after a fit, and the fitted volume not being
visible enough — see the `9730213` commit). This spec covers the three
remaining, larger changes:

1. Replace drag-based region marking with keyboard-modifier + click
   (b/r/p), matching tv's own hotkey convention, and removing the
   rigid step-by-step marking order entirely.
2. Link all peaks in one fit region to a single shared FWHM by
   default, with an option to free them.
3. Allow an optional, small, left-side tail contribution to the peak
   shape (gf3's Hypermet tail term, left-only).

## Interaction Model

**No more "Fit Peaks" mode toggle.** Since b/r/p+click never collides
with a plain, unmodified click-drag (which continues to pan/zoom
exactly as today, via the existing toolbar), there's no need for an
exclusive mode — marking is simply always available. This also removes
all of the current nav-toolbar-disabling logic and the
scroll-wheel-suspension check in `_on_scroll`.

**Marking actions**, each triggered by holding a key and clicking on the
plot:

- **b+click** — background region. Each click drops one boundary
  point; every two b-clicks complete one region, counting only among
  b-clicks themselves (r-clicks and p-clicks marked in between don't
  interrupt a pending b-click's pairing — this is what "any order"
  means in practice: b/r/p each keep their own independent pending/click
  count, unaffected by the others). Background regions are a **2-slot
  ring buffer**: once 2 regions exist, a third completed pair evicts
  the *oldest* region, leaving the other untouched. This is the
  "auto-redo" behavior — fixing a mis-marked background region is just
  two more b-clicks, no explicit clear step.
- **r+click** — the fit region. Same two-click-per-region mechanic
  (counting only among r-clicks), but only one slot: a newly completed
  pair always replaces the existing fit region outright.
- **p+click** — a peak, but only within the current fit region's
  bounds (if no fit region is set yet, the click is rejected with a
  status-bar hint). This is a **toggle**: clicking inside the fit
  region but not near an existing peak marker adds a new peak; clicking
  within a small pixel-distance threshold (8 px) of an existing peak
  marker removes it instead. Using pixel distance (not a fixed
  data-coordinate distance) keeps the "close enough to hit" tolerance
  consistent regardless of zoom level.
- A single pending click (the first of a not-yet-completed b/r pair)
  gets a light dashed marker, so the user can see it registered before
  completing the pair.
- **"Fit"** enables once 2 background regions, 1 fit region, and ≥1
  peak all exist — in whatever order they were marked. **"Clear"**
  remains as a full reset of everything in-progress (but never touches
  already-committed fits, same as today).

**Technical note on detecting the held key:** matplotlib's own
`MouseEvent.key` attribute (reflecting the last keyboard event on the
canvas) is explicitly documented as unreliable if the canvas didn't
have keyboard focus when the key was pressed — a real risk here, since
the natural gesture is "hover over the plot, press b, click" and there
would be nothing to guarantee the canvas already has Qt focus at that
moment. Instead: `FitModeController` becomes a `QObject` and installs a
Qt event filter directly on the canvas widget (not the whole
application, to avoid any stale-filter accumulation across the
multiple `MainWindow` instances created in the test suite), tracking
its own `_held_key` state from `QEvent.KeyPress`/`KeyRelease` for B/R/P
(ignoring auto-repeat events). The canvas is set to a strong focus
policy and grabs focus automatically whenever the mouse enters the
plot (via matplotlib's `figure_enter_event`), so a key held immediately
after moving the mouse onto the plot is reliably captured regardless of
what had focus beforehand.

## Fitting Model Changes

Both new options are **shared across every peak in the fit region**,
never per-peak — matching how tv/gf3 treat these as region-wide
detector-response characteristics, not per-peak-tunable knobs.

**Linked FWHM (default on).** All peaks in the region fit against a
single shared `sigma`, instead of one independent `sigma` per peak.
Parameter count drops from `3n` to `2n + 1` (n×(amplitude, position) +
1 shared sigma). A checkbox ("Independent widths") lets the user free
them back to one `sigma` per peak for a specific fit.

**Left tail (default off, opt-in checkbox "Left tail").** Adds gf3's
exact Hypermet tail term (ported from `srcRW/gf3_subs.c`'s `eval()`,
confirmed directly against that source rather than the earlier
research summary) in place of the pure Gaussian core, for each peak:

```
peak(x) = amplitude · [ (1-r)·exp(-w²) + r·exp(x/β)·erfc(w+y)/erfc(y) ]
x = channel - position
w = x / (sigma·√2)
y = sigma / (β·√2)
```

`r` (tail fraction) and `β` (tail decay constant, in channel units) are
two new **shared** free parameters across all peaks in the region.
`r` is bounded to `[0, 0.3]` (keeping the contribution genuinely
"small," per the request that motivated this) and `β` is constrained
positive (`β > 1e-6`) — verified numerically during design that a
positive `β` with this exact sign convention produces a tail biased
toward *lower* channels (the left side), not the right: evaluating the
tail term at equal distances on either side of a peak center (σ=3,
β=5) gives roughly 45× more contribution on the left (x=−10) than the
right (x=+10), matching the physical low-energy-tailing behavior this
feature is for.

**All four combinations** of the two options (linked/independent
width × tail on/off) are valid and must be supported: parameter counts
are `2n+1` (linked, no tail), `3n` (independent, no tail — the old v1
default), `2n+3` (linked + tail), `3n+2` (independent + tail).

**Uncertainty propagation note:** when width is linked, every peak's
`PeakResult.sigma`/`.sigma_err`/`.fwhm`/`.fwhm_err` will hold the
identical shared value — this is expected, not a bug. This extends the
existing, already-documented v1 simplification (ignoring the
amplitude/sigma covariance cross-term in the volume-uncertainty
formula) rather than introducing a new one: with a shared sigma, that
same simplification is applied per-peak using each peak's own
amplitude paired with the one shared sigma's uncertainty.

## Data Model Changes

`FitResult` gains: `link_widths: bool`, and (all `None` when the tail
option was off for that fit) `tail_fraction: float | None`,
`tail_fraction_err: float | None`, `tail_beta: float | None`,
`tail_beta_err: float | None`. `PeakResult` is unchanged — its existing
fields already cover the linked-width case (see above), and there's
nothing to add per-peak for the shared tail parameters (they live on
`FitResult`, once per fit, not once per peak).

## UI Changes

- Remove: the "Fit Peaks" toggle action, `_update_fit_mode_availability`'s
  mode-gating role (its "must have an active, visible spectrum" check
  moves to gating the "Fit" button directly instead), the nav-toolbar
  enable/disable dance in `toggle()`, and the `_on_scroll` fit-mode
  guard.
- Keep: "Fit" and "Clear" toolbar actions.
- Add: two checkable toolbar actions, "Independent widths" (unchecked
  by default — i.e., linked is the default) and "Left tail" (unchecked
  by default). Both are read at the moment "Fit" is clicked and passed
  through to `fit_peaks()`.

## Testing

`tests/test_fit_mode.py` (pure state machine) and
`tests/test_fit_mode_ui.py` (Qt-dependent) both need substantial
rework for the new interaction model — this is expected and is a
plan-level concern, not detailed further here. New coverage needed
includes: the background-region ring-buffer eviction behavior, the
fit-region single-slot overwrite, peak add/remove toggling (including
the pixel-proximity threshold), the canvas event-filter's key-tracking
(a real `QKeyEvent` simulation, not just directly setting `_held_key`,
for at least one test — to catch a regression in the filter wiring
itself, not just the logic that consumes `_held_key`), and `fit_peaks()`
tests for all four width-link/tail-option combinations, including a
numeric check that the left-tail term is asymmetric in the expected
direction (matching the verification done during this design).

## Out of scope

- A right-side tail or the step-background term from gf3's full
  Hypermet model — only the left tail was requested.
- Per-peak width/tail overrides — both remain strictly region-wide
  shared parameters, matching tv/gf3.
- Persisting the "Independent widths"/"Left tail" checkbox states
  across app restarts (QSettings) — they reset to their defaults
  (linked, no tail) each time the app starts, same as how the app
  doesn't currently persist the log-scale toggle either.
- Automatic peak-finding — still entirely manual marking, as in v1.
