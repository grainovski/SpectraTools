# Reference study: peak-finding/fitting algorithms and .spe/.spk formats

Date: 2026-07-09

Study of two existing legacy gamma-ray spectroscopy programs, done to
inform the design of SpectraTools' upcoming peak-finding, peak-fitting,
and additional-file-format features. Not code to port or reuse directly —
this is prior art informing a from-scratch Python design.

- **`tv` (v1.9.13) + `libmfile` (v1.0.7)** — source at `tv-1.9.13/` and
  `libmfile-1.0.7/` in the repo root. Cologne HPGe spectrum viewer/fitter
  (Institute for Nuclear Physics, University of Cologne).
- **`gf3`** — source at `srcRW/` in the repo root. Part of D.C. Radford's
  RadWare package (originally ORNL/HHIRF, 1999).

---

## Peak-finding

### tv: sliding matched-filter fit

`tv-1.9.13/lib/tv/vsSearch.c` — `_peak_search()` (336–480), `_insert_peak()`
(115–310), `gauss_integral()` (57–89), public entry `SeaPeakSearch()`
(614–676).

Requires a pre-existing **width calibration** `FWHM(channel)` (evaluated
via `CalW()`, `vsCal.c:1012`, Horner's method). At every channel `i` with
expected FWHM `W`:

- Window `[i-2W, i+2W]`, template `g(q) = exp(C·q²)` with
  `C = 4·ln(0.5)/W²` (unit-height Gaussian parametrized by FWHM), `q` =
  distance from `i`.
- Fit `y_j = BG + Amp·g_j` via **closed-form 2-parameter ordinary least
  squares** (no iteration).
- Propagate per-channel Poisson variance to `dAmp0`; significance
  `Prob0 = gauss_integral(Amp0/dAmp0)` (two-sided normal probability,
  Bevington's series expansion).
- Scan left→right; a region opens when `Prob0 >= 99.99%` (default) and
  closes when `Prob0` stops improving; `_insert_peak()` then does a
  **coarse-to-fine grid search** (start step = 0.1×bracket width, divide by
  10 each round until step < 0.03 channel) to refine the sub-channel peak
  position — not gradient-based.
- Stored peak: position, volume `= sqrt(-π/C)·Amp` (analytic Gaussian
  integral), width as sigma (`FWHM × 0.424661`).

Only reliable for well-separated peaks; dense multiplets need re-fitting
by the full fit engine (see below).

### gf3: smoothed second-difference (Mariscotti-style)

`srcRW/gf3_subs.c` — `pfind()` (5074–5180), driven by `findpks()` (3048)
and `autofit()` (1313–1502).

Given expected FWHM in channels `ifwhm` (from a calibration curve, see
below): at every channel `nd`, compute a windowed second-difference
statistic —

```
ctr    = Σ spec[nd + j]                                for j in [j1, j2]
wings  = Σ (spec[nd - j + j1] + spec[nd + j + j2])      for j in [1, j2]
S(nd)  = ctr - wings                     # smoothed −(2nd derivative)
noise  = sqrt(ctr + wings + 1)           # Poisson noise estimate
p(nd)  = S(nd) / noise                   # significance in sigma
```

(`j1`, `j2` derived from `ifwhm` so the center window spans `ifwhm`
channels; an extra boundary correction `w4` applies when `ifwhm` is odd.)

- A region locks when `p(nd)` exceeds a threshold (`sigma`, default 4);
  while locked, track the running local maximum of `p`; when `p` falls
  back below threshold, finalize the peak with a **3-point parabolic
  vertex interpolation** through the p-statistic around the local max for
  sub-channel position.
- FWHM calibration curve (channel-based, not energy):
  `FWHM(x) = sqrt(F² + G²·(x/1000) + H²·(x/1000)²)` (defaults F=3, G=2,
  H=0).
- `findpks` discards found peaks below `ipercent`% (default 5%) of the
  largest found peak.
- `autofit`'s multi-peak strategy: search once, fit all found peaks
  together, then iteratively fit the **residuals** `(data-fit)/sqrt(fit)`
  with `pfind` again to catch at most one more blended/missed peak per
  round, re-fit, repeat (capped at 35 peaks) — a residual-driven
  peak-addition loop.
- Separate quick/non-iterative-fit path for single unblended peaks
  (`find_ge_cent`, 2929–3045): linear background via
  minimum-3-channel-average on each side (`find_bg`/`find_bg2`), then
  **moment analysis** (area = Σ(y−bg), centroid = 1st moment,
  FWHM = 2.355·sqrt(2nd moment − centroid²)), re-centering the window at
  ±1.5·FWHM and iterating up to 20 times.

### Comparison

tv's method is a proper (if non-iterative) matched-filter fit with a
statistically principled significance test; gf3's is a simpler, faster
smoothed-derivative filter (classic NIM 100 (1972) Mariscotti algorithm
lineage) with parabolic refinement. gf3's approach is simpler to port to
NumPy as a convolution; tv's gives cleaner statistical footing per-peak
but needs the per-channel weighted-LS machinery.

---

## Peak-fitting

Both programs converge on the same overall shape: **Levenberg-Marquardt
nonlinear least squares** (both cite Bevington's *Data Reduction and Error
Analysis*, CURFIT, page 237) fitting a **Gaussian core + tail + step
background**, with tail/step/background parameters **shared across all
peaks in a fit region** so multiplets are fit simultaneously rather than
peak-by-peak.

### tv: `CurFit()` (`vsCurFit.c:73-156`)

- Builds curvature matrix, solves `(J^T W J + λ·diag)·δ = J^T W r` via
  full-pivoting Gauss-Jordan (`CurGaussJordan`); adaptive λ (note:
  `λ = 0.001·(oldMeasure - tryMeasure)` on success, not the textbook
  `λ /= factor`).
- Two objective modes: chi-square (`CurChiSqr`) or **Poisson maximum
  likelihood** (`CurPoissonLikelihood`, `Σ [y·ln(f(x)) - f(x)]`) — the
  latter correct at low statistics but only valid on raw, unmodified
  (not background-subtracted) spectra.
- Analytic derivatives supplied (not finite-difference) for the peak
  functions.
- Two interchangeable peak shapes documented in `doc/fit-info.txt`
  (132–252):
  - **"CA"**: Gaussian core with **exponential tails on both sides**
    (crossover points `SL = TL·S^EL`, `SR = TR·S^ER`) plus an arctan-based
    step.
  - **"AE"** (preferred — analytically integrable, less
    step/tail-parameter correlation): Gaussian ×
    `(1 + power-law tail)` plus an **erf-based step**:
    ```
    PEAK(x) = V/NORM · [ (1+TAIL(x))·exp(-x²/2σ²) + STEP(x) ]
    STEP(x) = 0.5·SH·(1 - erf(x/(SW·σ·√2)))
    NORM    = σ·(√(2π) + TL + TR)
    ```
  - 7 free parameters per peak: position, width(σ), **volume** (not
    amplitude — deliberately, so area error bars don't need post-hoc
    integration), tail-left, tail-right, step-height, step-width.
  - Peak parameters (tail/step/width) can be **tied to a shared
    calibration curve** instead of fit per-peak (`FSInitTails`/
    `FSInitStep`, `vsFitSetup.c:432+`), the same pattern as the width
    calibration used for peak search.

### gf3: `fitter()` (`gf3_subs.c:3088-3405`), model `eval()` (2716-2820)

- Parameter vector: `3×npeaks + 6` — quadratic background (3) + shared
  tail fraction `R`, shared tail decay `β`, shared step height `S` (3),
  then per-peak position/FWHM/height.
- Peak shape (Hypermet-style, analytically convolved Gaussian ⊗ one-sided
  exponential, normalized to 1 at peak center):
  ```
  peak(x) = h·[ (1-r)·exp(-w²) + r·exp(x/β)·erfc(z)/erfc(y) + (S/200)·erfc(w) ]
  # w = x/(σ√2), y = σ/(β√2), z = w + y, r = R/100
  ```
- If `R` fixed at 0 → fast path, pure Gaussian + step, no tail.
- Analytic Jacobian supplied to the fitter (`eval` mode ≥1 returns
  `derivs[]` directly).
- Standard Marquardt λ adjustment (×10 on rejected step, ÷10 on accepted);
  convergence when all parameter deltas < 1% of formal error; one final
  λ=0 pass purely to get correct covariances.
- **Multiplet handling** — the signature gf3 feature, two mechanisms:
  1. `R`/`β` (tail) are *inherently* single region-wide parameters, never
     per-peak.
  2. Optional explicit linking of **widths** (`irelw`) and/or **positions**
     (`irelpos`) across peaks in a region: all but one reference parameter
     are removed from the free list, and the others are stepped
     proportionally via a fixed ratio (`fixed[]`) — locks relative
     width/spacing while letting the whole cluster move together.
     `autofit` uses this as a two-pass stabilizer: fit once with
     `irelpos=0`, then again with `irelpos=1` once relative spacing is
     known.
- Peak area (not just height×FWHM, since the tail carries volume too):
  ```
  y = FWHM/(β·3.33021838);  d = exp(-y²)/erfc(y)
  Area = h·( r·2·β·d + (1-r)·FWHM·1.06446705 )   # 1.06446705 = sqrt(2π)/2.35482
  ```
  Errors propagated via the parameter covariance matrix through these
  analytic partials. The tail also shifts the true centroid below the
  fitted position parameter; gf3 computes a tail-corrected centroid for
  calibration use.

### Implications for a Python design

- `scipy.optimize.curve_fit`/`least_squares` (both Levenberg-Marquardt)
  map directly onto either model; supplying the analytic Jacobian (both
  programs derive one) will be faster/more robust than finite-difference.
- A **Gaussian + one-sided exponential tail + step background** shape
  (gf3's Hypermet form is the cleaner of the two to implement — closed
  analytic form, no piecewise crossover logic like tv's "CA" variant) is
  the right target for anything beyond a bare Gaussian; start with plain
  Gaussian + linear background for v1 and treat tail/step as a deliberate
  future enhancement, per this project's established YAGNI approach.
  Recommend deciding this scope explicitly in the next design/brainstorm
  round rather than assuming full Hypermet support is in scope immediately.
  See "Out of scope" note below.
- Multiplet handling is the one piece of real "special" logic worth
  redesigning cleanly rather than porting verbatim — e.g. explicit shared
  parameter groups (a Python fitting parametrization can pass shared
  tail/step parameters once across all peaks in a region naturally,
  without gf3's index-juggling `nextp[]`/`fixed[]` machinery).

---

## `.spe` format

**Two different, incompatible formats share this extension** in these
codebases — neither is the ORTEC/Maestro ASCII `.SPE` format (with
`$SPEC_ID:`/`$DATA:` sections) that's arguably the most common `.spe`
variant in the wild today; **that format is not implemented in either
codebase** and would need to be written from scratch/from public spec if
needed.

### tv/libmfile's `.spe` (actually the Radware gf2/gf3 binary format, added 2003 for interop — `libmfile-1.0.7/src/gf2_minfo.c`, `gf2_getput.c`)

36-byte header, no Fortran record brackets:

```
offset  size  field
0       4     magic = 24 (int32) -- used for format/endian detection
4       8     spectrum name (ASCII, not padded/terminated)
12      4     Size1 = channel count (int32)
16      4     Size2 = 1
20      4     ired1 = 1
24      4     ired2 = 1
28      4     rec1b = 1
32      4     rec2a = channels * 4
36      N*4   channel data, float32, one 4-byte word/channel
```

Endianness: read first 4 bytes; `0x00000018` → little-endian data,
`0x18000000` (byte-swapped) → big-endian data. Pure magic-number
detection, no extension check. **No metadata** (no live/real time, date,
or calibration) — just name + dimensions + data.

### gf3's `.spe` (`srcRW/libs/util/rwspec.c` — `read_spe_file`, `wspec`)

Fortran-unformatted sequential file, two records, each bracketed by a
4-byte length marker before and after (`[int32 len][payload][int32 len]`):

```
Record 1 (24-byte payload):
  bytes  0- 7 : name       char[8], space-padded
  bytes  8-11 : idim1      int32, channel count
  bytes 12-15 : idim2      int32, always 1 for 1-D
  bytes 16-19 : ired1      int32, always 1
  bytes 20-23 : ired2      int32, always 1

Record 2 (idim1*idim2*4-byte payload):
  numch × float32          raw channel counts
```

Endianness: no explicit field; heuristic on the record-length marker — if
the leading int32 read natively isn't a sane length (`>= 65536`), assume
byte-swapped and retry; `read_spe_file` specifically checks for
`rl == 24` (native) vs `rl == -24` (swapped-flag convention used
internally) and if swapped, byte-swaps both header ints and every
payload float. **No metadata beyond name + count** — calibration lives in
a separate companion `.cal`/`.aca` file (title + polynomial order + up to
6 double gain coefficients, up to x⁵, same swap-detection convention with
different expected record lengths: 98/-98 for title, 52/-52 for
order+coefficients).

### For reference: this project's existing ASCII format

Both study reports note that this project's current plain-ASCII `.txt`
format (see `docs/superpowers/specs/2026-07-08-histogram-viewer-design.md`)
is conceptually close to `libmfile`'s own `MAT_TXT` handler
(`libmfile-1.0.7/src/txt_minfo.c`): whitespace/comma-separated numbers,
`#`-prefixed comments skipped, optional magic first-line format tag.

---

## `.spk` format

**Only found in gf3/srcRW — not present in tv/libmfile at all.**

Implementation: `srcRW/libs/util/readsp.c` — `spkio()` (139–372),
`spkman()` (385–424), `spkread()` (428–500). Magic string `"HHIRFSPK"`
(ORNL/HHIRF heritage).

Unlike `.spe`, this is a **multi-spectrum indexed container** (up to 254
named spectra per file), addressed in half-words (2-byte units — a
16-bit-minicomputer legacy):

1. **Directory block**, fixed 2048 bytes at file offset 0:
   - bytes 0–7: magic `"HHIRFSPK"`
   - word 2: `nid` = number of spectra currently stored (max 254)
   - word 3: `nxwd` = next free half-word offset for appending
   - words 4+: repeating `(id, half-word offset)` pairs, one per spectrum,
     linear-scanned to locate a given ID

2. **Per-spectrum header** (up to 64 words):
   ```
   word 0     : id number
   words 1-3  : parameter label (12 bytes)
   words 4-6  : reserved (date-time)
   word 7     : bytes/channel = -4 (fixed)
   word 8     : header length (half-words)
   word 11    : histogram length = channel count ("numch")
   words 15-16: min/max non-zero channel #
   words 17-19: up to 3 calibration constants  <- .spk DOES carry per-spectrum calibration
   words 22-31: title (40 bytes)
   ```

3. **Data**: 4-byte (32-bit) integers, one per channel, stored
   **immediately after the header, only for the non-zero sub-range**
   `[minc, maxc]` (sparse-storage optimization) — channels outside that
   range are implicitly zero on read.

4. **No Fortran record-length brackets** (unlike `.spe`) and **no
   byte-swap/endianness detection anywhere** in this code path — a Python
   reader would need to just assume a fixed (almost certainly
   little-endian) byte order, since the format carries no self-describing
   marker for it.

Given `.spk` carries more legacy-specific complexity (indexed container,
sparse storage, half-word addressing) for comparatively little benefit
over `.spe` (only 3 calibration constants beyond what `.spe` lacks), it's
reasonable to deprioritize `.spk` support unless there's a concrete corpus
of legacy `.spk` files that actually need reading.

---

## Out of scope for this document

This is a research reference, not a design or implementation plan. Actual
scope decisions (which peak shape to implement first, whether to support
`.spe`/`.spk` at all and which variant, how multiplets should be
represented in this app's own data model) belong in a proper
brainstorming session against `docs/superpowers/specs/`, not here.
