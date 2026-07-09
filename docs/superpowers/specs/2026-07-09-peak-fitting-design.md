# Peak Fitting — Design Spec

Date: 2026-07-09

## Purpose

Add interactive Gaussian peak fitting to SpectraTools, operating on the
currently **Active** spectrum. Third and final of three approved
sub-projects (`.spe` reader → `.spk` reader → peak fitting). Informed by
`docs/research/2026-07-09-peak-finding-fitting-and-spe-spk-formats.md`'s
study of tv's and gf3's fitting algorithms, but this is a from-scratch
design, not a port — only the background-region technique is deliberately
modeled on tv's actual workflow (see below).

Reported peak positions are in **channel** units, matching the rest of
this app (no energy calibration exists anywhere in SpectraTools yet).

## Architecture

Two new modules, following this project's established split between pure
logic (no Qt dependency, thoroughly unit-tested) and UI wiring
(`main_window.py`):

- **`peak_fit.py`** — pure fitting math: given raw channel data and
  user-marked region bounds, computes the background and fits the
  Gaussian peaks. No Qt dependency; fully testable with synthetic data,
  same spirit as `histogram_io.py`/`spe_io.py`/`spk_io.py`.
- **`fit_mode.py`** — the interactive state machine (tracking which
  region-marking step the user is on) and the results-panel widget.
  `main_window.py` delegates to this rather than growing further itself
  (it's already the largest file in the project at 468 lines, and this
  feature adds a non-trivial amount of new interactive UI logic).

`main_window.py` gains: a "Fit Peaks" toggle button (same visual style as
the existing zoom/full-spectrum toolbar icons), "Fit" and "Clear" buttons
that are only enabled during an in-progress marking sequence, mouse
event wiring that hands off to `fit_mode.py` while fit mode is active,
and a new results-panel side-tab (same closed-by-default
`QDockWidget` + persistent toolbar pattern already used for the
spectrum list panel).

## Interaction Flow

A fixed, linear sequence — each step must complete before the next
begins:

1. **Toggle "Fit Peaks" mode on.** Existing pan/zoom/toolbar behavior is
   suspended while active; toggling off cancels any *in-progress*
   (not-yet-fitted) marking sequence, but never removes already-committed
   fits. Fit mode requires the Active spectrum to currently be visible
   (`Show` checked) — if it isn't, the toggle is disabled with a short
   explanatory message, since there'd be nothing meaningful to click
   against.
2. **Drag-select the first background region** (a channel range).
3. **Drag-select the second background region** (another channel range,
   the other side of the peak(s)). The two background regions can be
   marked in either spatial order — whichever ends up with the lower
   mean x-coordinate is treated as "left" for the background-line
   calculation, regardless of which one was marked first.
4. **Drag-select the fit region** — one region, where the peak(s) live.
5. **Click inside the fit region once per peak** to mark each peak's
   approximate position. Supports multiplets (multiple clicks in the
   same fit region). A click outside the fit region's bounds is ignored
   (with a brief status-bar hint), not clamped.
6. **"Fit"** (enabled once both background regions and the fit region are
   set, and at least one peak is marked) runs the fit. **"Clear"**
   discards the entire in-progress sequence (all regions and peak marks)
   without fitting, resetting to step 2.
7. **On a successful fit**, the two background regions, the fixed
   background line, the fit region, the peak markers, the fitted curve,
   and the numeric results become one persisted result drawn on the plot
   and listed in the results panel. The state resets to step 2 so the
   user can immediately mark another fit. **Multiple fits can coexist**
   on the same spectrum until individually or collectively cleared (see
   "Results display" below).
8. **On a failed fit** (see "Error handling"), a status-bar message
   explains why, and the user stays at step 5/6 with their existing
   marks intact, free to adjust and retry or hit "Clear".

Regions and peak marks are per-attempt scratch state (in `fit_mode.py`)
until a fit succeeds; only successful fits are persisted (as `FitResult`
objects, see below).

## Fitting Model

**Background** is computed directly from the two background regions —
**not** a free parameter in the nonlinear fit (matching tv's actual
workflow, where background-region marking/fitting is its own step,
separate from and prior to peak fitting):

- Left point: `(mean(x) for x in left_bg_region, mean(y) for x in left_bg_region)`.
- Right point: the same computation over `right_bg_region`.
- The background is the straight line through these two points, held
  fixed for the rest of the fit. `fit_peaks` trusts its `left_bg_region`/
  `right_bg_region` arguments as given — it's `fit_mode.py`'s job (per
  the interaction flow above) to work out which of the two marked
  regions is spatially left/right before calling in.

**Peaks** are fit against the fit-region data with the fixed background
subtracted. For `n` marked peaks, there are `3n` free parameters
(amplitude, position, sigma per peak):

```
y_sub(x) = Σᵢ Aᵢ·exp(-(x-x0ᵢ)² / (2σᵢ²))      for x in fit_region
y_sub    = y - (background_slope·x + background_intercept)
```

**Initial guesses:** position = the user's click x-coordinate; amplitude
= `y_sub` at the nearest channel to that click; sigma = a generic
starting width, `(fit_region width) / (4 · n_peaks)` — nonlinear
least-squares (Levenberg-Marquardt via `scipy.optimize.curve_fit`) only
needs a reasonable starting point, not a precise one, for peaks that
aren't extremely tightly overlapping; degenerate cases are handled as
fit failures (see below), not silently produced as bad fits.

**Weighting:** each fit-region data point's standard deviation is taken
as `sqrt(max(y, 1))` (the `sigma=` argument to `curve_fit`, with
`absolute_sigma=True`) — the standard Poisson-counting-error convention,
floored at 1 to avoid a zero/undefined deviation at zero-count channels.
`curve_fit` internally weights the least-squares cost by the inverse
*square* of this (i.e. `1/y`), which is the statistically correct
inverse-variance weighting for Poisson data (de-emphasizing high-count,
high-absolute-variance channels relative to low-count ones) — not simply
`1/sqrt(y)` used directly as a weight, which would weight the wrong way.
This is a lightweight acknowledgment that gamma-spectrum channels are
Poisson counts, without adopting tv/gf3's full maximum-likelihood
machinery (explicitly out of scope, see below).

**Reported per peak**, computed from the fitted `(amplitude, position, sigma)`
and the covariance matrix `curve_fit` returns:
- `position`, with `position_err = sqrt(pcov[i][i])` for that parameter.
- `fwhm = 2.3548 · sigma`, `fwhm_err = 2.3548 · sigma_err`.
- `area = amplitude · sigma · sqrt(2π)` (the analytic Gaussian integral),
  with `area_err` propagated via the standard independent-error product
  formula (`area · sqrt((amplitude_err/amplitude)² + (sigma_err/sigma)²)`)
  — a deliberate simplification that ignores the amplitude/sigma
  covariance cross-term rather than the fully-correct propagation through
  the 2×2 covariance submatrix; acceptable for v1, flagged here as a
  known simplification rather than an oversight.

## Module: `peak_fit.py`

```python
@dataclass
class PeakResult:
    position: float
    position_err: float
    fwhm: float
    fwhm_err: float
    area: float
    area_err: float
    amplitude: float
    sigma: float

@dataclass
class FitResult:
    left_bg_region: tuple[float, float]
    right_bg_region: tuple[float, float]
    fit_region: tuple[float, float]
    background_slope: float
    background_intercept: float
    peaks: list[PeakResult]

class FitError(Exception):
    """Raised when a fit cannot be performed or does not converge."""

def fit_peaks(
    x: np.ndarray,
    y: np.ndarray,
    left_bg_region: tuple[float, float],
    right_bg_region: tuple[float, float],
    fit_region: tuple[float, float],
    peak_positions: list[float],
) -> FitResult:
    ...
```

- `x`/`y` are the Active spectrum's full channel-index and count arrays
  (`peak_fit.py` slices internally by region bounds — call sites don't
  pre-slice).
- Raises `FitError` (not a bare exception from `scipy`) when: `curve_fit`
  fails to converge (catches the underlying `RuntimeError`); either
  background region or the fit region contains zero data points; the fit
  region contains fewer data points than free parameters (`3 * n_peaks`);
  `peak_positions` is empty.
- New exception type — distinct from `histogram_io.ParseError`, since
  this is a runtime fitting failure, not a file-parsing failure.

## Module: `fit_mode.py`

- A small state machine tracking the current step (awaiting left bg
  region → awaiting right bg region → awaiting fit region → marking
  peaks) and the in-progress scratch values for each.
- Owns the matplotlib artists for the *in-progress* marking (shaded spans
  for regions being dragged, marker lines for clicked peaks) and for each
  *committed* `FitResult` (shaded background/fit regions, dashed
  background line, solid fitted-curve overlay, peak position markers).
- Owns the results-panel widget (list of committed fits for the Active
  spectrum, each showing its peaks' position/FWHM/area), with per-fit
  removal and a "clear all" action.
- `main_window.py` wires mouse press/move/release events to this module
  while fit mode is toggled on, and asks it to draw/undraw artists on the
  existing `self.axes`/`self.canvas` — mirroring how the existing
  zoom/pan code already touches those same objects directly.

## Data Model

Committed `FitResult`s are stored on the spectrum they were fit against:
`LoadedSpectrum.fits: list[FitResult]` (new field, default empty list).
This means fits naturally persist, hide, and get removed alongside their
owning spectrum — consistent with how `visible`/`active` already work —
with no separate global registry to keep in sync.

## Results Display

- **On the plot:** each committed fit shows its two background regions
  and fit region as shaded spans, the fixed background as a dashed line,
  the total fitted curve (sum of all peaks in that fit) as a solid
  overlay line, and a small marker at each peak's fitted position.
- **Results panel:** a new closed-by-default side-tab dock (same
  `QDockWidget` + persistent toolbar pattern as the spectrum list panel),
  listing every committed fit for the Active spectrum with its region
  bounds and, per peak, position/FWHM/area (with uncertainties).
  Right-click (on the plot overlay or the panel entry) removes that one
  fit; a "Clear all fits" action removes every fit on the Active
  spectrum.

## Error Handling

`fit_peaks` raising `FitError` (non-convergence, degenerate/too-small
regions, or too few data points for the requested number of peaks) is
caught in `fit_mode.py` and surfaced as a status-bar message (e.g. "Fit
did not converge — adjust regions or peaks and try again"). The user's
in-progress marks are **not** discarded on failure — they stay at the
marking-peaks step, free to add/adjust peaks and retry, or hit "Clear" to
start over.

## Testing

- **`peak_fit.py`** (pure unit tests, no Qt): synthetic single-Gaussian +
  linear-background data (no noise) — recovered parameters match ground
  truth closely. Synthetic 2-3-peak multiplet (overlapping Gaussians,
  shared background) — each peak individually recovered. The same cases
  with added Poisson noise — recovered parameters within a reasonable
  tolerance band. Explicit tests for every `FitError` condition (empty
  peak list, zero-width region, too few data points for the requested
  parameter count, a `curve_fit` non-convergence case constructed from
  genuinely degenerate input).
- **UI integration:** offscreen `QTest`-simulated interaction (same
  pattern already used elsewhere in this project) — toggle fit mode,
  simulate the three region drags and peak click(s), click "Fit", confirm
  a `FitResult` is appended to the Active spectrum and the expected
  overlay artists appear; confirm "Clear" discards in-progress state
  without committing anything; confirm per-fit removal and "clear all"
  both work.

## Out of scope

- **Hypermet-style tail/step background modeling** (both tv's "CA"/"AE"
  shapes and gf3's Hypermet peak) — plain Gaussian + fixed linear
  background only, per the already-agreed v1 scope. A natural future
  enhancement, not attempted here.
- **Automatic peak-finding** (tv's matched-filter search, gf3's
  smoothed-second-difference search) — peaks are always manually marked
  by the user in v1.
- **Position/width linking across peaks in a multiplet** (gf3's
  `irelw`/`irelpos`) — each peak's position/width is independently free
  in the fit; no shared-ratio constraint mechanism.
- **Poisson maximum-likelihood fitting** (tv's `CurPoissonLikelihood`) —
  weighted least-squares (via the Poisson-error weighting described above)
  instead of a true likelihood-based fit.
- **Energy calibration** — peak positions/results are in channel units;
  no channel-to-energy mapping exists anywhere in this app yet.
- **Editing a placed region** (e.g. dragging an edge to resize) — if a
  region is marked wrong, the user clears the whole in-progress sequence
  and starts over; no in-place adjustment in v1.
- **Saving/exporting fit results** to a file — results live only in the
  running app's `LoadedSpectrum.fits`; no persistence/export format is
  defined in this sub-project.
