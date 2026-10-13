# Matrix symmetrization -- design

Date: 2026-10-10. Status: approved in conversation; building a Windows test.

## Goal

Let the user, by choice, turn an asymmetric gamma-gamma matrix into a
symmetric one: a CoMPASS `.json` matrix, a ROOT 2-D histogram, or a TV `.mtx`.
The result opens as a new matrix panel, so a gate catches events through
either detector and the statistics roughly double. The original panel is
untouched.

## Why the axes must be gain-matched first

A two-detector matrix has one detector per axis. On the CoMPASS Co-60 run,
X = CH1 and Y = CH0, and the gains differ: the Co-60 lines sit at X bins
148.8/168.5 but Y bins 151.9/172.3. A plain M + MT folds two energy scales
together. Measured on that run (FWHM in bins, 1173/1332):

| | 1173 | 1332 |
|---|---|---|
| X alone | 2.86 | 3.25 |
| naive M + MT | 4.97 | 5.69 |
| Y matched to X, then M + MT | 3.01 | 3.47 |

So Y is mapped onto X's scale first, with a linear map `x = a + b*y`
derived from the same two lines located on both projections. Single-array
matrices whose axes already share one scale use a = 0, b = 1.

## The diagonal: S = M + MT, doubled diagonal kept

Decided by simulation (an E-E self-coincidence cascade plus an E-E3
reference, sigma = 1.3 bins, gate on E):

- M + MT: E/E3 = 1.9999 against an expected 1.9989. Correct.
- M + MT - diag(M): 1.788 (-10.6%), and the peak shape is distorted (the
  centre bin loses 15%). With finite resolution only about 21% of a
  self-coincidence peak lands exactly on the diagonal, so counting the
  diagonal once undercounts just that part.

Every (gate gamma, partner) pair is counted once, which is what a gate
measures. The one cost is variance: a diagonal cell holds the same events
twice, so its variance is 4*D, not the 2*D that the "variance = counts"
rule gives. **Correction:** keep `D = diag(M')` (M' = the gain-matched
matrix before folding) next to S. In a cut, add `2*D[j]` to the variance
of every channel j that lies on a gated line of the diagonal, and
`factor**2 * 2*D[j]` for background lines. With that, the error bars are
exact.

## Integer counts when rebinning

Mapping Y bins onto X's scale makes each old bin overlap one to three new
bins. Splitting counts by overlap fractions gives fractional cells (99.6%
of the nonzero cells on the Co-60 run), while the app's matrices are
int64. Instead each count is assigned at random, with the overlap
fractions as the probabilities (sequential binomial splitting, vectorised
along a row), from a FIXED seed. This keeps integer counts and exact
Poisson statistics, and the result is reproducible. Counts whose new
position falls outside the matrix are dropped and reported.

## Refused, with the reason given

- Non-square matrix.
- Any negative cell (background-subtracted or difference data, e.g. gpff.mtx).
- Already symmetric (e.g. gg.mtx -- symmetrizing it again would double it).
- A line pair that is too close (< 2 bins apart on Y) or gives a
  non-positive gain.

## UI

- Matrix panel: a new **Symmetrize...** button. It is disabled on a
  symmetrized panel; on a matrix that cannot be symmetrized it explains why.
- Dialog: either "Both axes already share one scale" or "Match Y to X using
  two lines" (recommended for two detectors). For each of the two lines the
  user types an approximate X position and Y position (in the projections'
  own channels). They are refined to the local peak centroid, a
  background-subtracted centroid iterated about the guess (on by default).
  The dialog shows the resulting map `x = a + b*y` live.
- The result opens in a new panel titled "... (symmetrized)", with path
  `<root>_sym<ext>`, keeping the real extension last for fit-log naming.
  X and Y are identical, so the axis selector is fixed to X and disabled,
  with a tooltip explaining why.
- Help, HowTo section 15: when to use it, gain matching, the M + MT
  convention (self-coincidences), the refusals.

## Code

- New `matrix_symmetrize.py`: `SymmetrizeError`, `check_symmetrizable`,
  `refine_centroid`, `map_from_lines`, `rebin_rows`, `symmetrize`, the
  last returning (S, D, lost). No Qt.
- New `symmetrize_dialog.py`: the dialog.
- `matrix_cut.compute_cut_with_variance(..., diagonal_twice=None)`: the
  variance correction.
- `matrix_panel.py`: the button, the `symmetric_diagonal` kwarg, the
  disabled axis selector, passing D to the cut.
- `main_window._open_matrix_panel(path, matrix=None, symmetric_diagonal=None)`.

## Tests

Refusals; the identity map giving exactly M + MT; counts conserved and
integer; determinism by seed; a line mapped to the right place; statistical
correctness of the splitting; the map from the Co-60 fixture; the matched
projection sharper than the naive one; the self-coincidence ratio; the
diagonal variance (gate on a pure-diagonal cell gives variance 4D, with the
background factor applied); and panel wiring.

## Not in scope

Per-axis energy calibrations; folding into a triangular storage format;
saving the symmetrized matrix to disk.
