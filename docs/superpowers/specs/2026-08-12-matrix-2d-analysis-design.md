# 2D Matrix Analysis (mtx / gate / cut) — Design Spec

Date: 2026-08-12
Branch: `v3-major` (first feature of the 3.0.0 cycle)

## Purpose

Add support for TV-style 2D matrix analysis: opening a `.mtx` file (gamma-gamma coincidence data), viewing its X/Y projections, marking a cut (positive gate) region and background regions, and producing a background-subtracted 1D cut spectrum — which then behaves as an ordinary spectrum with full access to this app's existing fitting/calibration/integration tools. A real 2D heatmap viewer is added as a visualization-only enhancement beyond what TV itself ever had.

## Researched Before Designing

Three parallel research passes read TV 1.9.13's C source directly (`tv-1.9.13/`) plus its vendored binary-I/O library (`libmfile-1.0.7/`, kept in this repo for exactly this purpose). Full findings are preserved in the session transcript; the load-bearing facts are summarized here with source citations.

### The `.mtx` file format

TV has no matrix-specific parser of its own — `Cut_MtxOpenP` (`tv-1.9.13/src/VsCut.c:644`) opens every matrix via `SpcOpen(path, mtxFmt, "r")`, the exact same generic spectrum-I/O entry point TV uses for 1D spectra. That function is a thin wrapper around `libmfile`'s `mopen()`, which auto-detects among ~20 registered binary layouts (`libmfile-1.0.7/src/mat_types.c:59-103`). **Matrices are read-only in TV — there is no write path anywhere** (`src/VsFace.h:1338-1343`'s entire matrix command menu is `open`/`close`/`format`/`status`/`attach-projection`, no `save`/`write`/`create`).

Both real sample files provided (`gpff.mtx`, `gg.mtx`, currently untracked at the repo root) use the `lc` (line-compressed) format — libmfile's own default for integer data, and the same *style* of custom delta/run-length/variable-length-integer compression this app's `spk_io.py` already implements for `.spk` files (not a literal port, but an established precedent for this class of format in this codebase).

**`lc` format binary layout**, confirmed against `libmfile-1.0.7/src/lc_minfo.h:32-53` and verified empirically (see "Verified against the real files" below):

```c
#define MAGIC_LC 0x80FFFF10
typedef struct {
  u_int magic;             /* 0x80FFFF10, little-endian */
  u_int version;           /* 2 in both sample files */
  u_int levels, lines, columns;
  u_int poslentablepos;    /* = 44 = sizeof(header) */
  u_int freepos, freelistpos;
  u_int used, free;
  u_int status;
} lc_header;                /* 44 bytes total, all fields u_int32 little-endian */
```

After the header: a table of `{u_int pos, u_int len}` pairs (8 bytes each), one per `(level, line)` — i.e. `levels * lines` entries — giving the byte offset and length of each row's independently-compressed block. Row addressing is `level*lines + line` into this table.

The compression itself (`libmfile-1.0.7/src/lc_c2.c`, the default v2 codec): each channel's value is zig-zag delta-encoded against the previous channel, runs of ≥3 identical values collapse to a single run-length tag, 1-3 small deltas pack into one byte (2/3/6-bit fields), and anything larger falls back to a tagged variable-length byte sequence. This is **not** an LZ/dictionary compressor — no back-references, purely local delta+RLE+varint. The exact bit-level tag encoding must be read directly from `lc_c2.c`/`lc_getput.c` during implementation rather than transcribed here, to avoid a hand-transcription error in something this bit-precise — see "Verification strategy" below for how to do that safely.

### Verified against the real files

`libmfile-1.0.7` was compiled from source under WSL (Ubuntu-24.04, CRLF-stripped, direct `gcc` compilation bypassing the CRLF-broken autotools `configure`) and used to read both real files' actual headers and row data — not just the format spec on paper:

- Both files: `filetype` confirms `lc`, `version=2`, `lines=8192`, `columns=8192` — matches the "8k.8k.lc" TV format string.
- **`levels=2` in both files** — not 1. Each file bundles two independent 8192×8192 matrices. Confirmed via real row reads that level 0 and level 1 are genuinely different data (e.g. `gg.mtx` row 500: level-0 sum 227896 vs level-1 sum 263186, 507/512 sampled columns differ), not duplicates.
- `gpff.mtx` level 0 contains negative values (e.g. row 0: `6 -1 1 0 0 0 0 0 0 0 -1 ...`); level 1 in the same row is non-negative. **Confirmed by the user: this is real random-coincidence-subtracted data, and negative values must be preserved throughout the pipeline** (matrix reading, projections, and the resulting cut spectra) — no clamping anywhere, matching TV's own behavior (traced through every arithmetic primitive in the cut engine, none of which clamp).
- `gg.mtx`'s symmetry (`M[10][2000] == M[2000][10] == 1`) confirms it stores the **full** `lines×columns` grid even though symmetric — `lc` is not one of libmfile's triangular-packed formats (those are a separate `oldmat`-family feature), so no special symmetric-unpacking logic is needed for this format.
- **The app always reads level 0 and never exposes level selection to the user** — confirmed by the user, and matches TV's own command surface, where `Cut_MtxOpen` (`src/VsCut.c:624-651`) takes only `<mtx-idx> <filename>`, no level argument anywhere.

### The gate/cut algorithm (TV's engine, `lib/tv/vsCut.c`)

Traced completely from marker to result spectrum. A "projection" and a "cut" are the same underlying mechanism in TV — there is no separate projection-computation function; `SpcProject` (`lib/tv/vsSpectra.c:962-981`) sums matrix rows `[NINT(x1), NINT(x2)]` (inclusive, clamped to the matrix bounds) into a 1D output, and a "plain projection" is just the degenerate case of one gate spanning the whole axis with no background gates.

The default and only weighting mode this app implements — TV's `CUT_WGAT` (gate-width weighting):

```
pos_width = Σ (channel width of each positive/cut region)      -- this app: exactly one region
bg_width  = Σ (channel width of each background region)         -- any number of regions

pos[ch] = Σ over cut region(s) of rowSum(region, ch)             -- raw, unweighted row-sum
bg[ch]  = Σ over background region(s) of rowSum(region, ch)      -- raw, unweighted row-sum

net[ch] = pos[ch]  −  (pos_width / bg_width) · bg[ch]
```

Traced source: `src/VsCut.c:466-482` (`Cut_CreateFacGate` — sets `fac=1.0` for the positive gate, `fac=-(pos_width/bg_width)` for every background gate) combined with `lib/tv/vsCut.c:80-99,110-136` (`CutCreateSpectrumUpdate`/`CutCreateSpectrum` — sums each region's raw projection, scaled by its `fac`, into one accumulator). With zero background regions this reduces to `net[ch] = pos[ch]` (no subtraction) — well-defined, not an error case (TV's own `CUT_WGAT` division-by-zero-width in this case happens inside an empty loop and is discarded, never actually computed against real data). Negative results are never clamped anywhere in the pipeline (confirmed by tracing every arithmetic primitive: `SpcBufAdd`, `SpcMultiply`, `SpcAdd`).

TV's engine technically supports multiple positive-gate regions (`CutRec.gate` is a vector) and a second weighting mode (`CUT_WFIT`, solving a linear system from fitted-peak volumes) — both are **explicitly out of scope** here per the user's simplification (below).

### UI/display reality check

TV 1.9.13 has **no 2D matrix rendering anywhere in its source** — no heatmap, no image, no intensity map (confirmed by the complete absence of any raster/image primitive — `XPutImage`/`XCreateImage`/`XImage` — anywhere in the codebase; every TV visual is vector line-lists). Users interact with a matrix exclusively through its 1D projection, with gate markers overlaid as brackets (top-anchored = positive gate, bottom-anchored = background gate; color indicates which cut buffer, not marker type). This app's heatmap viewer is therefore a genuine enhancement, not a port — confirmed explicitly with the user as in-scope, visualization-only, opened on demand.

"Slicing" — a term the feature request used alongside "gate/cut markers" — does not exist as a distinct TV concept; it appears exactly once in the manual as a plain-English synonym for "cut." No separate mechanism to account for.

## Resolved Decisions (confirmed with the user)

1. **Heatmap window**: a separate, non-blocking, on-demand top-level window, opened via a button inside the matrix panel. Visualization only — no marking/interaction on it. Gets the same matplotlib navigation toolbar (zoom/pan/reset) already used elsewhere in this app.
2. **Matrix panel window**: a separate top-level window from the main window. The two are mutually exclusive in focus — activating (clicking into) one disables interaction with the other without closing it; either can be reactivated by clicking it. This is a custom-implemented behavior, not a native Qt modal dialog (a true modal would block the parent entirely, which is not what's wanted here — the user needs to switch back and forth freely without losing matrix-panel state).
3. **Projections**: both X and Y projections are computed when a matrix is opened (this app has the whole matrix available via `mtx_io.py`, unlike TV which required opening a second, transposed matrix file to get the other axis). The user picks which one to work on.
4. **Marking**: exactly one positive/cut region, plus any number of background regions — a deliberate simplification of TV's engine (which technically allows multiple positive regions) down to TV's own common-case/simple-hotkey usage.
5. **Weighting**: automatic, gate-width-based only (`CUT_WGAT`'s formula above). No user-facing weighting-mode choice, and TV's `CUT_WFIT` (fit/volume weighting) is not implemented — it requires exactly one positive gate, degrades silently to gate-width weighting in TV itself whenever its preconditions aren't met, and the TV research found what looks like a genuine indexing bug in TV's own implementation of it. Not worth the complexity for this app.
6. **Cut activation → handoff**: pressing "activate cut" computes the cut spectrum (in the axis complementary to whichever projection was marked) and hands it to the main window as an ordinary `LoadedSpectrum` — it joins the existing Spectra list and gets full fitting/calibration/integration for free, no new code needed for any of that.
7. **Negative values**: preserved everywhere — matrix data, projections, and cut spectra. No clamping at any stage.
8. **Matrix level**: always level 0. Not exposed to the user at all, matching the total absence of level-selection anywhere in TV's own command surface.
9. **No write support**: matrices are read-only, matching TV's own architecture exactly (not a limitation being worked around — TV itself never had a matrix writer).
10. **No `.gate`/`.cutdir` persistence**: since the cut spectrum becomes an ordinary spectrum, this app's existing Save Spectrum dialog (`.txt`/`.spe`/`.spk`) already covers persisting the result. TV's own gate-marker save/reload mechanism is not ported.

## Architecture

### `mtx_io.py` (new)

```python
def load_mtx(path: str) -> np.ndarray:
    """Reads an lc-format TV matrix file, returning its level-0 data as
    a square/rectangular int64 array of shape (lines, columns). Read-only;
    no writer. Negative values (e.g. from random-coincidence subtraction)
    are preserved as-is."""
```

Parses the 44-byte header (magic/version/levels/lines/columns/poslentablepos), reads the `{pos,len}` table for level 0's rows, and decodes each row's compressed block. The exact bit-level decode algorithm (delta-zigzag + RLE + variable bit-packing) must be transcribed from `libmfile-1.0.7/src/lc_c2.c` and `lc_getput.c` directly during implementation, not re-derived from this summary.

**Verification strategy for the implementation plan**: `libmfile` has already been successfully compiled from source under WSL (Ubuntu-24.04) this session, bypassing its CRLF-broken `configure` script by compiling the `.c` files directly with `gcc`. That compiled reference decoder is a ground-truth oracle — the plan should use it to generate exact expected values (arbitrary rows, arbitrary channel ranges, from both real sample files) as test fixtures, then verify the pure-Python decoder reproduces them exactly. This removes the risk of a transcription error in anything this bit-precise going unnoticed. Follow the same "verify against source, not prose" discipline already established in this project's memory.

Shares `ParseError` from `histogram_io.py`, matching every other loader in this app.

### `matrix_cut.py` (new)

**Explicit axis convention** (this app's own — TV's file format has no fixed X/Y labeling, it's only ever "the gated axis" vs "the output axis"): `matrix[row, col]` — **rows are Y-channels, columns are X-channels**, matching numpy's natural `(row, column)` indexing to `(Y, X)`. Concretely:

- The **X-projection** is a spectrum indexed by X-channel: sum over every row for each column, i.e. `matrix.sum(axis=0)`.
- The **Y-projection** is a spectrum indexed by Y-channel: sum over every column for each row, i.e. `matrix.sum(axis=1)`.
- Marking a cut/background region **on the X-projection** means selecting an X-channel (column) range; activating the cut gates on those *columns* and produces a result indexed by **Y** (row) — the complementary axis. Symmetrically, marking on the **Y-projection** gates on *rows* and produces a result indexed by **X** (column).

```python
def compute_projection(matrix: np.ndarray, axis: str) -> np.ndarray:
    """axis='x' -> matrix.sum(axis=0), a spectrum indexed by X-channel (column).
    axis='y' -> matrix.sum(axis=1), a spectrum indexed by Y-channel (row)."""

def compute_cut(matrix: np.ndarray, axis: str, cut_region: tuple[float, float],
                 bg_regions: list[tuple[float, float]]) -> np.ndarray:
    """`axis` is which projection was marked (the axis being gated ON, in
    that projection's own channel units). Implements
    net[ch] = pos[ch] - (pos_width/bg_width)*bg[ch] as derived above, where
    pos/bg are raw sums over the marked row-range (axis='y') or
    column-range (axis='x') of `matrix`. Returns a 1D array along the
    COMPLEMENTARY axis (axis='x' input -> Y-indexed output, and vice
    versa) -- matching TV's own "gate on rows, get columns" (or the
    reverse) behavior. Zero bg_regions => net = pos (no subtraction, not
    an error). Never clamps negative results."""
```

Pure functions, no Qt dependency — testable in isolation, matching this app's existing separation of computation (`peak_fit.py`, `spectrum_operations.py`) from UI.

### `matrix_panel.py` (new) — the matrix panel window

A new top-level window class. Opened via a new `File > Open Matrix...` action in `main_window.py`. Responsibilities:
- Load the matrix via `mtx_io.py`, compute both projections via `matrix_cut.py`.
- Display the working projection (X or Y, user-selectable) as an ordinary spectrum plot.
- Marker placement for the cut region and background regions — model this on this app's existing marker-placement interaction pattern (the click-based B/R/P system already built for 1D peak fitting in `fit_mode.py`), adapted to two marker roles instead of three, rather than re-deriving marker UX from TV's C source directly.
- "Activate cut" button: calls `compute_cut`, constructs a `LoadedSpectrum` from the result, and hands it to `main_window.py` to add to the Spectra list — reusing the existing spectrum-loading path as much as possible rather than duplicating it.
- A button to open the heatmap window (below), passing it the loaded matrix data.
- Focus-exclusivity behavior with the main window per Resolved Decision 2.

### `matrix_heatmap.py` (new) — the heatmap window

A separate, non-modal top-level window. matplotlib `imshow`/`pcolormesh` of the level-0 matrix data, log-scale intensity mapping (appropriate for gamma-gamma data spanning orders of magnitude), with the standard matplotlib navigation toolbar for zoom/pan/reset. No markers, no interaction beyond viewing.

## Error Handling Summary

- Malformed/non-`lc`-magic file, truncated header, or corrupt row data → `ParseError`, matching every other loader's contract in this app.
- Zero background regions at cut-activation time → valid, produces an unsubtracted cut (not an error).
- No cut region marked yet → "activate cut" should be disabled/blocked with feedback, matching this app's existing pattern for blocked fit-mode actions (status-bar message, not a silent no-op or a crash).

## Testing Plan

- **`tests/test_mtx_io.py`**: real-file end-to-end tests against `gpff.mtx` and `gg.mtx` (copied into `tests/fixtures/`, committed despite their size — 30-31MB — per explicit user confirmation), using the oracle-verified test vectors described in "Verification strategy" above (exact row/header values generated via the compiled `libmfile` reference decoder, not hand-computed). Small hand-built synthetic `lc`-format files supplement these for header-parsing edge cases and malformed-input error handling that the two real files don't naturally exercise (truncated header, bad magic, etc.).
- **`tests/test_matrix_cut.py`**: pure-function tests for `compute_projection`/`compute_cut` against small synthetic matrices with hand-computed expected results, covering: zero background regions, multiple background regions, negative-value preservation, both axis directions.
- **`tests/test_matrix_panel.py`** / UI-level tests: matching this app's existing convention for window/dialog tests (e.g. `test_operations_menu.py`'s style) — cut-region + background-region marking, cut activation producing a correctly-valued spectrum in the main window's Spectra list, focus-exclusivity behavior between the two windows.

## Out of Scope

- Writing `.mtx` files.
- `CUT_WFIT` (fit/volume weighting) and any user-facing weighting-mode choice.
- Multiple positive/cut gate regions (TV's engine supports this; this app deliberately does not).
- Matrix level selection (always level 0).
- TV's `.gate`/`.cutdir` save/reload mechanism — superseded by this app's existing Save Spectrum dialog once a cut becomes an ordinary spectrum.
- Any binary format other than `lc` (the only format the two real sample files use; libmfile's other ~19 auto-detected formats — `gf2`, `trixi`, the `oldmat` family, etc. — are not implemented unless a real need arises).
- Axis transpose/in-app matrix rotation (not needed — both projections are computed directly from the one loaded matrix).
