# Damped Marquardt Fit and Peak Decomposition — Design Spec

Date: 2026-07-15

## Purpose

A user report after live testing: fitting 3+ peaks in a region sometimes
converges badly — a peak's fitted position collapses toward the middle of
the region instead of staying near where it was marked. This is a classic
symptom of an *unconstrained* nonlinear least-squares optimizer
(`scipy.optimize.curve_fit`, MINPACK's plain Levenberg-Marquardt) taking an
oversized step for closely-spaced/overlapping Gaussians and getting stuck.
The user asked to "examine and apply the fit procedure from TV, including
peak decomposition" — TV (`tv-1.9.13/`, vendored into this repo for
reference) is an established gamma-spectroscopy tool whose own Marquardt
implementation applies **per-iteration step damping** that plain
`curve_fit` doesn't. Investigated directly against TV's source
(`tv-1.9.13/lib/tv/vsCurFit.c`, `tv-1.9.13/src/VsFitFct.c`,
`tv-1.9.13/lib/tv/vsFitSetup.c`) — see citations throughout.

## What TV Actually Does (research findings, not proposal)

**Marquardt loop** (`vsCurFit.c:CurFit`, lines 73-156): standard
curvature-matrix Marquardt with one twist — on a successful step, the
damping factor for the *next* iteration is set proportional to how much
chi-square just improved (`CurLambda = .001 * (oldMeasure - tryMeasure)`,
line 124) rather than the classic fixed `lambda *= 0.1`. On a failed step,
`lambda *= 10`; if `lambda` exceeds `1e30` the algorithm tries one
last-resort half-step backward before giving up. Convergence is a
relative-improvement threshold (`(oldMeasure - measure) / measure <=
1e-12`) or `iterations >= 50`, whichever comes first. Constants:
`CUR_RATIO=1e-12`, `CUR_LAMBDA=1e-3` (start), `CUR_MINLAMBDA=1e-20`,
`CUR_MAXLAMBDA=1e30`, `CUR_INCLAMBDA=10`, `CUR_ITERATIONS=50`
(`vsCurFit.h:13-22`).

**Per-parameter step damping** (`VsFitFct.c:CHANGE_TRY` macro, lines
84-121, applied every iteration via `FF_ChangeTry`) — this is the actual
fix for the reported bug:
- **Position**: the raw Newton step is capped at `±` the peak's own current
  width. If the position was inside the marked fit region and the trial
  step would push it outside, the step is replaced with *half* the
  distance to that boundary instead (a soft barrier — a peak can still
  drift out over several iterations, just slowly, not in one jump).
- **Width (sigma)**: capped at `±2×` the current width per iteration
  ("change widths slowly... convergence is very bad" — TV's own comment).
  A trial that would land exactly on `0.0` is nudged instead of accepted
  outright (avoids a `1/sigma²` singularity in the peak formula).
- **Tail parameters**: a trial landing exactly on `0.0` is halved instead.
- **Amplitude**: no damping at all — takes the raw step every time.

**Initial width estimate** (`vsFitSetup.c:FSInitWidth`, lines 296-405) is
smarter than "read the y-value at the clicked position": it walks
outward, pixel by pixel, from the *largest-amplitude* marked peak's
clicked position to the true local maximum, measures the FWHM directly
from the data by walking outward from that maximum until counts drop
below half-amplitude, and uses that one measured sigma as the starting
width for every peak in the region.

**"Peak decomposition" is a visualization, not an algorithm**
(`VdDecomp.c`, 128 lines, read in full) — it draws each already-fitted
peak's *individual* component curve on top of the total fit, rather than
showing only the summed total. There is no automatic clustering/splitting
logic anywhere in TV; decomposition into components is entirely the
user's manual peak-marking, same as in this app today.

**Peak shape models are NOT being adopted.** TV's two selectable peak
models (a hard-spliced Gaussian/exponential-tail model, and a
power-law-multiplicative-tail model, each with a *separate* additive
arctan/erfc step-background term) are architecturally different from this
project's existing gf3-derived hypermet formula
(`hypermet_left_tail` in `peak_fit.py`, ported from `srcRW/gf3_subs.c`,
already tested and in production use). The reported bug is about fit
*robustness*, not an incorrect peak shape — this spec keeps the existing
`hypermet_left_tail` formula entirely unchanged and only replaces the
*procedure* used to fit it.

## Scope

1. A new damped Marquardt-Levenberg solver in `peak_fit.py`, replacing
   `scipy.optimize.curve_fit` inside `fit_peaks()`, replicating TV's
   lambda-adaptation and per-parameter step-damping rules above.
2. TV's local-max/FWHM-based initial width estimate, replacing the
   current `region_width / (4 * n_peaks)` heuristic.
3. Peak decomposition: draw each peak's individual component curve
   (in addition to the existing total curve) when rendering a committed
   fit with more than one peak.

## The Damped Marquardt Solver

New function `_marquardt_fit(model, x, y, y_err, p0, param_types,
fit_region_bounds=None)` in `peak_fit.py`, replacing the `curve_fit(...)`
call sites in `fit_peaks()`. Returns `(popt, pcov)` with the same meaning
`curve_fit(..., absolute_sigma=True)` returns today, so every downstream
consumer (`perr = np.sqrt(np.diag(pcov))`, the uncertainty math already in
`fit_peaks()`) is unchanged.

**Numerical core:** standard Marquardt curvature-matrix update —
`alpha = J^T J` (weighted by `1/y_err`), damped diagonal
`alpha[i,i] *= (1 + lambda)`, solve `alpha @ delta = J^T r` for the raw
step via `numpy.linalg.solve` (falls back to `numpy.linalg.lstsq` if
`alpha` is singular — can happen transiently for a near-degenerate
overlapping-peaks Jacobian). The Jacobian `J` is computed by **central-
difference numerical differentiation** (not hand-derived analytically) —
TV's own engine falls back to numeric differentiation
(`CurNumericDerivation`) when a fit-function module doesn't supply an
analytic one, and hand-deriving partials of the hypermet `erfc` formula
correctly is a real source of subtle bugs for little practical benefit at
this problem size (a handful of parameters, a few hundred data points).

**Lambda adaptation and convergence:** replicate TV's constants and
formulas exactly (`_CUR_RATIO = 1e-12`, `_CUR_LAMBDA_START = 1e-3`,
`_CUR_MIN_LAMBDA = 1e-20`, `_CUR_MAX_LAMBDA = 1e30`, `_CUR_INC_LAMBDA =
10`, `_CUR_MAX_ITERATIONS = 50`). On acceptance,
`lambda = max(0.001 * (old_measure - new_measure), _CUR_MIN_LAMBDA)`; on
rejection, `lambda = min(lambda * 10, ...)`, with the half-step-backward
last resort when `lambda` exceeds `_CUR_MAX_LAMBDA`, then stop.

**Per-parameter step damping**, applied to the raw Newton step before
evaluating a trial point, dispatched by parameter *name* (reusing the
existing `amp_i`/`pos_i`/`sigma`/`sigma_i`/`tail_fraction`/`tail_beta`
naming from `_parameter_names`) — no new parameter-classification
mechanism needed, since the names already encode the category via their
prefix:
- `pos_*`: cap the step at `±` the associated peak's *current* sigma
  value (the shared `sigma` if linked, else that peak's own `sigma_i`).
  If `fit_region_bounds` is given and the parameter's old value was
  inside it, halve the step instead of letting it cross the boundary.
- `sigma` / `sigma_*`: cap the step at `±2×` the current value; if the
  trial would be exactly `0.0`, halve the step instead. After damping,
  hard-clamp the trial to `>= 1e-6` (this project's existing minimum —
  see `_bounds()`'s current `(1e-6, np.inf)` for sigma — kept as an
  additional safety net TV doesn't need because it tolerates sign flips
  differently; this project already takes `abs(sigma)` when assembling
  results, so a step landing slightly negative is fine, only exactly/near
  zero is the hazard).
- `tail_fraction`: if the trial would be exactly `0.0`, halve the step;
  after damping, hard-clamp to `(0.0, TAIL_FRACTION_MAX)` (existing
  constant, existing bound semantics preserved exactly).
- `tail_beta`: if the trial would be exactly `0.0`, halve the step; after
  damping, hard-clamp to `>= TAIL_BETA_MIN` (existing constant, existing
  bound preserved exactly).
- `amp_*`: no damping, matches TV exactly (`FIT_UTVOL` has no case in
  `CHANGE_TRY`).

**Covariance for uncertainties:** at the final accepted parameter point,
recompute the *undamped* curvature matrix (`J^T J`, no `(1+lambda)`
boost) and invert it for `pcov`, matching what `curve_fit(...,
absolute_sigma=True)` does today (uncertainties scale with the actual
`y_err` passed in, not rescaled by reduced chi-square).

**All-parameters-fixed and single-free-parameter edge cases** (already
handled specially in `fit_peaks()` today for the zero-free-parameters
case) are unaffected by this change — the Marquardt loop only replaces
the branch where `curve_fit` is currently called with a non-empty
`free_names`.

## Initial Width Estimate

New helper `_measure_width(x_fit, y_sub, peak_positions, amplitudes)` in
`peak_fit.py`, replacing the current `region_width / (4 * n_peaks)` /
peak-spacing-cap logic in `_initial_guess`:

1. Pick the marked peak with the largest `amplitudes[i]` (background-
   subtracted count at its clicked channel — already computed by the
   existing amplitude initial-guess step).
2. From that peak's clicked channel, walk outward one channel at a time
   in whichever direction `y_sub` keeps increasing, to find the true
   local-maximum channel (bounded by the fit region's edges).
3. From that maximum, walk outward until `y_sub` drops below half the
   maximum's value, to measure a FWHM in channels (bounded by the fit
   region's edges; falls back to the existing heuristic if the data is
   too flat/noisy to find a clear half-max crossing within the region).
4. `sigma0 = fwhm_channels / FWHM_FACTOR` (existing constant), floored at
   `1e-6`.

This single measured sigma becomes the starting width for **every** peak
in the region, both linked and independent-widths modes (independent
widths still each start from this same shared estimate — TV does the
same — and are then free to diverge during fitting).

## Peak Decomposition (visualization)

In `fit_mode.py`'s `draw_committed_fits`, for a visible result with more
than one peak, draw each peak's own component curve (background +
that single peak's contribution, using the same Gaussian/hypermet formula
already used for the total) as a thin dashed line in the peak's own color
family (e.g. a lighter/dashed variant of the existing red), in addition
to the existing solid total curve. A single-peak fit's component curve is
identical to its total curve, so no special-casing is needed for `n == 1`
— drawing it anyway is harmless (exact overlap).

## Testing

- **Solver correctness in isolation**: `_marquardt_fit` fits a simple
  known problem (e.g. a single Gaussian, or a plain quadratic) and
  recovers the true parameters within a sensible tolerance, independent
  of the peak-fitting integration — proves the numerical core before
  wiring it into the more complex multi-peak/hypermet machinery.
- **No regression on existing scenarios**: the full existing
  `tests/test_peak_fit.py` suite (single peak, linked/independent widths,
  left tail, fixed parameters, all-fixed, etc.) continues to pass. Since
  this deliberately changes the underlying algorithm (not a pure
  refactor), a test's `pytest.approx` tolerance may be loosened if the
  new solver's correct-but-different convergence path lands slightly
  outside an overly tight existing tolerance — that's an acceptable,
  expected adjustment here, not a regression to chase down.
- **Demonstrated improvement**: a new test fits a heavily-overlapping,
  unequal-amplitude three-peak scenario (adapted from manual
  investigation during this bug report: peaks at relative spacing
  smaller than their own sigma) and asserts the fitted positions land
  substantially closer to their true/marked values than the old
  `curve_fit`-based approach did on the same synthetic data (a regression
  benchmark, not just "does it run without erroring").
- **Initial width estimate**: `_measure_width` (or the `_initial_guess`
  integration) recovers a sigma close to a known synthetic peak's true
  width from data alone, and produces a sensible fallback when no clean
  half-max crossing exists in the data.
- **Peak decomposition**: `draw_committed_fits` on a multi-peak result
  produces one component line per peak in addition to the total line
  (assert on `axes.lines` count, similar to the existing
  `test_draw_committed_fits_skips_hidden_results` pattern).

## Out of Scope

- Adopting either of TV's own peak shape models (CA or AE) — the existing
  gf3-derived hypermet formula is kept exactly as-is.
- A step/background term separate from the existing linear background
  (TV's arctan/erfc step) — not requested, not part of this bug report.
- Any automatic peak-count/clustering algorithm — "decomposition" here is
  visualization only, matching what TV itself actually does.
- Changing `fit_peaks()`'s public signature/return types — `fixed_params`,
  `link_widths`, `enable_left_tail`, `PeakResult`/`FitResult` fields all
  stay exactly as they are; only the internal fitting procedure changes.
