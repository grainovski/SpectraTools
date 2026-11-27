# v7.0.0 — Sorting CoMPASS list-mode data into a random-subtracted γγ matrix

**Status:** plan, not started. Written 2026-11-27 on NUCLINO-IV.
**Branch:** `v7-major` (cut from `3051e8e`). `master` and tag `v6.3.1` are
frozen and are not touched by this work; per the standing release rule the
branch is merged into `master` only at the 7.0.0 freeze, never tagged from
the branch.
**Execution:** subagent-driven, one fresh implementer per task with a
two-stage review between tasks (standing choice). Nothing is pushed to
GitHub and nothing is written to the Drive payload without asking.

## Goal

A new entry, **Operations → Sort List-Mode γγ Data…**, which opens a
sequence of dialogs and turns one CoMPASS list-mode file (`SData*.root`,
TTree `Data` / `Data_F` / `Data_R`) into:

1. the singles spectra of CH0 and CH1 at full digitizer resolution;
2. gain-matching coefficients that map CH1 onto CH0's ADC channel scale,
   fitted to as many strong peaks across the range as possible;
3. a time-difference spectrum on which the user marks a **prompt** window
   and one or more **random** windows;
4. a prompt γγ matrix and a random γγ matrix, sorted at the digitizer's own
   bin resolution (16384 × 16384 for the DT5780's 14-bit ADC);
5. the random-subtracted matrix `prompt − w · random`, `w` = prompt width /
   random width (TV's gate-width weighting), saved as an **asymmetric**
   matrix (X = CH1, Y = CH0, CoMPASS's own orientation) to `.json` and/or
   `.mtx`. Symmetrization stays a separate step, done later with the
   existing Symmetrize… command.

## Decisions already made (user, 2026-11-27)

| Question | Decision |
|---|---|
| Gain-match target | **CH0's ADC scale.** CH1 is mapped onto CH0 with a fitted polynomial; CH0 is untouched. The result is then ready for Symmetrize with the identity map. |
| Where the gain-matching peaks come from | **The run itself by default, or a separate calibration run** loaded in that step (e.g. Eu-152) for lines across the whole range. |
| Fractional / negative counts after subtraction | **JSON stores the exact floats; `.mtx` (integer-only format) uses seeded stochastic rounding**, so each cell keeps its expected value. Plain rounding would map, e.g., every −1/3 to 0 and systematically under-subtract. |
| Output orientation | **X = CH1, Y = CH0**, the same as CoMPASS's EvsE export ([[reference_compass_json_matrix]]), so the result overlays CoMPASS's own matrices. |
| JSON at full resolution | **Option (c):** a **sparse** JSON layout at full resolution (`{"format": "sparse", "shape", "x", "y", "v", …}`, read by SpectraTools only), plus **dense CoMPASS-style** JSON (`data[x][y]`) when the user picks a compression factor (≤ 4096 per axis). `.mtx` is always written at full resolution. |
| Version | Major bump to **7.0.0**: a new operation, two new file writers and a new input kind (list-mode TTrees). |

## What exists today (checked, not assumed)

- `root_io.py` reads only TH1/TH2 histograms; TTrees are skipped. Nothing in
  the app reads list-mode data yet.
- **No matrix writer exists.** `mtx_io.load_mtx` is read-only (lc format,
  version 2, int32 cells, `DIM_MAX` = 65536). `json_matrix_io` is
  read-only, and it **rejects** negative or non-integer counts, which is
  exactly what a subtracted matrix contains.
- `spk_io._lc2_compress` is already a verified port of libmfile's
  `lc2_compress` (`libmfile-1.0.7/src/lc_c2.c:63`). The `.mtx` writer needs
  libmfile's lc **file layout** on top of it (`lc_getput.c` `writeline`,
  `lc_minfo.c` header and row table), which is not ported yet.
- Peak finding and fitting are reusable: `peak_search.search`,
  `auto_calibrate.fit_found_peaks`, and `auto_calibrate.match`, a
  calibration-free linear matcher of peaks to reference positions with
  inlier and efficiency checks.
- `matrix_symmetrize.rebin_rows` already implements seeded overlap-weighted
  remapping (`DEFAULT_SEED = 20261010`) for whole matrix rows. Gain-matching
  applies the same idea per event, before binning.
- `matrix_cut._cut_with_variance` implements the background weighting used
  for cuts: the factor is the number of lines actually summed in the gate
  over the number in the background (HDTV's rule). That equals TV's rule
  (`VsCut.c:478`: `−MrkRange(gate)/MrkRange(bgGate)`, where `vsMark.c:29`
  `MrkRange` = Σ(x2 − x1) + number of marks, i.e. **inclusive bin counts**)
  whenever no window is clipped.
- `MainWindow._open_matrix_panel(path, matrix=…, axes=…)` opens an
  in-memory matrix, and `LoadedSpectrum(...)` adds an in-memory spectrum.
  Both are the natural hand-off points for results.

## Facts about the data (measured 2026-11-27 on `Na22_261126_ANI_STEF_180`)

These are the numbers the tests and acceptance checks rest on.

- TTree branches: `Channel` u16, `Timestamp` u64 (**picoseconds, 10 ns
  granularity**: no fine timestamp in this run), `Board` u16, `Energy` u16
  (ADC 0…16383), `CalibEnergy` f64 (= C0 + C1·Energy exactly), `Flags` u32.
  Time-sorted, both channels merged (`listTimeSorting = true` in
  `settings.xml`).
- Streams: RAW `Data_R` 656,167 events; UNFILTERED `Data` 655,336; FILTERED
  `Data_F` 264,245.
- **FILTERED is unsuitable as the input for this operation.** CoMPASS's
  software coincidence (COUPLES, window 0.01, reference CH1) kept all
  39,914 exact-time pairs but dropped essentially every other pair: within
  1 µs FILTERED has 39,914 pairs, UNFILTERED 43,672. With no off-prompt
  pairs left there is nothing for a random window to measure. 70% of
  FILTERED's events are unpaired singles that CoMPASS kept by a rule not
  yet identified. **Default input: UNFILTERED**, and a FILTERED file gets a
  warning in step 1.
- UNFILTERED nearest-partner pair counts: dt = 0: 39,924; |dt| ≤ 10 ns:
  43,387; |dt| ≤ 1 µs: 43,672. So the prompt peak spans the −10/0/+10 ns
  ticks, with a thin random floor around it.
- The run's E–E JSON holds exactly 39,914 counts, the same as FILTERED's
  dt = 0 pairs. Per channel, `bincount(Energy)` equals the run's
  `EspectrumF` `.txt` bin for bin.

## Step-by-step flow (what the user sees)

The flow is driven by one controller and is resumable with **Back**: each
dialog reads and writes a single session object, so going back keeps every
choice made. **Cancel** at any step discards the session; nothing is
written to disk until step 5.

1. **Input.** Pick the `SData*.root` file. The dialog shows the tree name,
   events per channel, run start/stop and live time (from
   `<run>_info.txt`, if found beside the stream folders), the timestamp
   granularity it detected, and a table of `Flags` values with counts.
   Each flag value has an *exclude* box; defaults are set only after Task 1
   confirms the bit meanings against CAEN's documentation. It also has the
   detector assignment (X = CH1, Y = CH0 by default) and the maximum |Δt|
   to pair over (default ±2 µs). A FILTERED stream gets the warning above.
2. **Singles.** CH0 and CH1 are sorted at 16384 channels and shown in the
   dialog. **Send to main window** adds both as ordinary spectra, so they
   can be fitted, saved or calibrated with the existing tools.
3. **Gain matching.** Peaks are found and fitted in both singles. The peak
   source is *this run* or *load calibration run…*, which accepts another
   SData file or a pair of 1-D spectra. Optionally a `.sou` source file
   identifies the lines by energy. A table lists the matched CH1↔CH0 pairs
   with an include box per pair. Polynomial order is 1 or 2: 2 needs at
   least 4 pairs, and with Na-22 alone, 511/1275 keV, it is 1. The dialog
   shows a residual plot, an overlay of gain-matched CH1 on CH0, and
   **Save / Load coefficients…** (a small JSON sidecar) for reuse on later
   runs.
4. **Time windows.** The Δt = t(CH1) − t(CH0) spectrum, binned at the
   detected granularity, over the step-1 range. The user marks one prompt
   window and one or more random windows, by dragging (the same hold-key
   marking habit as the matrix panel) or with spin boxes. Shown live: the
   counts in each window, the inclusive bin widths, `w` = N_prompt /
   N_random, the random level expected under the prompt (w × random), and
   the net prompt. Overlapping windows are refused.
5. **Sort and save.** Progress bar while sorting, then a summary: prompt
   pairs, random pairs, `w`, net total, and counts dropped off the matrix
   edge by gain matching. The user ticks `.mtx` and/or `.json`; default
   file names come from the run ID. **Open result** opens the subtracted
   matrix in a Matrix panel (X = CH1, Y = CH0, 1:1 channel axes).

## Computation rules

- **Pairing.** Every CH0 event is paired with **every** CH1 event within the
  maximum |Δt|, not just the nearest one. Each combination is a pair; with
  two detectors this is the standard all-combinations rule. The file is
  read in chunks (`uproot.iterate`) with the last `max|Δt|` of each chunk
  carried into the next, so no pair is lost at a boundary. Input that is
  not time-sorted is detected; it is sorted if it fits in memory and
  refused with a clear message otherwise.
- **Gain mapping.** ch₀′ = ⌊P(E₁ − ½ + u) + ½⌋, u ~ U(0, 1) seeded
  (`DEFAULT_SEED`). This is the centre-coordinate form: P is fitted to
  peak centroids measured on x = arange(n), i.e. bin centres, and the
  naive ⌊P(E + u)⌋ would shift every event by (P′ − 1)/2 channels
  (corrected in Task 3). This is the same
  random-split idea as `rebin_rows`, applied before binning so there is no
  picket-fence pattern from a non-integer gain. Events mapped outside
  0…16383 are dropped and counted. CH0 is not remapped.
- **Weights.** Window widths are **inclusive numbers of Δt bins actually
  summed**, after clipping to the sorted Δt range. That is TV's `MrkRange`
  convention, with HDTV's clipping rule as in `matrix_cut`.
  `w = N_prompt / N_random`, with several random windows summed (as TV
  sums several background marks).
- **Matrices** are accumulated **sparse** (flat index → count via
  `np.unique` / `np.bincount` on pair lists), never as dense 16384² arrays
  while sorting. A dense float64 16384² array is 2 GiB, an int64 one too.
  Dense arrays exist only when a matrix is opened in the panel or written
  as dense JSON.
- **Subtraction.** S = P − w·R, exact in float64 (sparse union of P and R
  cells).
- **`.mtx` rounding.** Each cell of S becomes ⌊S⌋ + Bernoulli(S − ⌊S⌋),
  seeded, so E[cell] = S exactly and the total is preserved to within
  Poisson-like noise. The seed and `w` are recorded in the JSON metadata
  and in the summary.

## Tasks

Every task: tests first, then code; CRLF line endings preserved
([[reference_editing_files_from_python_on_windows]]); `git add -u` plus
explicit new paths, never `git add -A`; one commit per task on `v7-major`.

### Task 1: List-mode reader, `listmode_io.py`
- Open an SData file, find the CoMPASS TTree (`Data`, `Data_F`, `Data_R`),
  validate the six branches and their types, and give a clear `ParseError`
  for anything else. Classify the stream (RAW, UNFILTERED or FILTERED) from
  the tree name.
- Chunked iterator yielding (channel, timestamp, energy, flags); detect the
  timestamp granularity (gcd); check sortedness.
- Parse `<run>_info.txt` (start/stop, real/live time, per-channel
  calibration). It is optional and never required.
- **Flag meanings:** look up the CoMPASS flag bits in CAEN's documentation
  and record them with the source. Defaults are set from that source only,
  not guessed.
- Tests: a small synthetic TTree written with `uproot.recreate`/`mktree`;
  wrong tree / missing branch / non-root errors; real-file cross-check that
  `bincount(Energy)` == `EspectrumF` `.txt` per channel (skipped when the
  untracked run is absent; see Fixtures).

### Task 2: Sorting core, `listmode_sort.py` (no GUI)
- `sort_singles(events, nchan=16384)` per channel.
- `pair_events(events, max_dt, x_channel, y_channel)` → Δt and (E_x, E_y)
  pair arrays, all-combinations, chunk-boundary carry-over.
- `dt_histogram(dt, granularity, range)`.
- `window_weight(prompt, randoms, dt_axis)` → (w, N_prompt, N_random) by
  inclusive clipped bin counts; refuses overlap or empty windows.
- `sort_matrix(pairs, mask, gain_map, shape)` → sparse counts.
- `subtract(P, R, w)` → float sparse; `stochastic_round(S, seed)`.
- Tests: a **synthetic generator** with known peaks, a known CH1 gain, a
  known prompt rate and a flat random rate. Expected `w` exact; the
  recovered prompt total within statistics; a **flat-random-only control**,
  whose subtracted matrix must average 0 (and must *fail* when `w` is
  deliberately wrong); boundary-pair test (a pair straddling a chunk edge
  is counted once); stochastic rounding preserves the mean on 10⁶ cells.
  Real-data cross-check on the Na-22 UNFILTERED run: pairs at |Δt| ≤ 10 ns
  ≥ 43,387 (all-combinations ≥ nearest-partner); FILTERED, Δt = 0 pairs ==
  39,914 == the E–E JSON total; and the FILTERED prompt matrix rebinned to
  CoMPASS's x16 grid has the same X projection as the JSON (orientation
  proof, not just totals).

### Task 3: Gain matching, `gain_match.py` (no GUI)
- Find and fit peaks in both singles (reuse `peak_search` +
  `fit_found_peaks`).
- Pairing without a source: CH0 centroids act as the "energies" of
  `auto_calibrate.match`, which finds the linear CH1→CH0 relation and its
  inliers. With a `.sou`: identify both channels separately and pair by
  energy.
- Weighted polynomial fit (order 1 or 2), residuals, per-pair include
  flags; `apply(E1, seed)` with dithering; JSON sidecar save/load.
- Calibration run: accept another SData file (singles via Task 2) or two
  1-D spectra.
- Tests: synthetic quadratic gain recovered within the centroid errors;
  wrong pairing is reported, not guessed (two-peak ambiguity case); dither
  produces no picket fence (flat input stays flat to χ² expectation); real
  Na-22: the 511 and 1275 keV lines in CH1 land within 1 FWHM of CH0's
  after mapping.

### Task 4: Matrix writers
- `mtx_io.save_mtx(path, int_matrix_or_sparse)`: lc version 2, one level,
  port of libmfile's `lc_getput.c`/`lc_minfo.c` layout on top of
  `spk_io._lc2_compress` (move the shared compressor into `lc_codec` rather
  than duplicating it). Rows are written from sparse input without a dense
  array. Header and row table are byte-compatible with libmfile.
- `json_matrix_io.save_json_matrix(path, matrix, meta)`: CoMPASS layout
  (`data[x][y]`, i.e. the app's [y][x] transposed back, like the reader),
  plus a SpectraTools metadata block (source run, windows, `w`, seed, gain
  coefficients, axes).
- Loader change: accept negative and fractional values **when the file
  carries the SpectraTools metadata block**; CoMPASS files keep today's
  strict check.
- Tests: `.mtx` round-trip through `load_mtx`, including negatives and an
  all-zero row; byte-compare against a matrix written by the vendored
  libmfile C (built in WSL) for a small case; JSON round-trip, with
  orientation pinned by a non-square matrix; the existing CoMPASS strictness
  tests still pass.
- **JSON layouts (decided: option c).** `save_json_matrix` writes the
  sparse layout at full resolution and the dense CoMPASS layout only for a
  compressed matrix (≤ 4096 per axis). It refuses dense output above that
  size with a message pointing at the sparse form. Compression sums
  factor × factor cells, and the dense file records the factor in its
  metadata so the reader can give the panel the right `ChannelAxis`
  ([[feedback_hdtv_axis_convention_for_2d]]). `load_json_matrix` reads
  both layouts and returns [y][x] either way.
- Tests add: sparse round-trip at 16384² without building a dense array in
  the writer; refusal of dense > 4096; compressed dense opens with channel
  axes matching the 16k singles (a peak at the same channel in both).

### Task 5: Dialogs and controller
- `listmode_operation.py`: a controller holding the session object and
  running the five dialogs in order with Back/Next/Cancel. Long work
  (reading, pairing, sorting) runs on a `QThread` with real progress,
  following the `_load_matrix_with_progress` pattern. Slot exceptions are
  caught and shown, never swallowed ([[reference_qt_dialog_reopen_trap]]).
- `listmode_input_dialog.py`, `listmode_singles_dialog.py`,
  `gain_match_dialog.py`, `time_window_dialog.py`,
  `listmode_output_dialog.py`. Plots use the existing matplotlib/theme
  helpers, so dark theme works.
- Carried over from Task 3's review (**required**): on the Na-22 run the
  automatic choice (order 2 at ≥ 4 pairs) takes its curvature from a
  ~150-channel low-energy feature, probably the threshold edge and not a
  line both detectors see. Through the three real lines (511, 1275,
  1461 keV) a straight line fits to ≤ 3.5 ch, but the quadratic differs
  from it by ~380 ch at ADC 16000, beyond the last pair (~6200), where
  nothing constrains it. So the gain dialog must: (a) show the range the
  pairs span and plot the map with the extrapolated part marked; (b) warn
  when order 2 is used and the curvature term moves any channel outside the
  pair range by more than one FWHM against the linear fit; (c) leave peaks
  below a low-channel cutoff unticked by default (threshold/noise region,
  cutoff shown and editable). The order is chosen by the user, with the
  automatic choice only as the starting value.
- Carried over from Task 3's report, small: `peak_search._MAX_FWHM` (60 ch)
  is too narrow for scintillators on a 16k ADC. `gain_match.find_peaks`
  works around it by compressing ×2–×8. `gain_match` also imports private
  helpers from `auto_calibrate` (`_inliers`, `_usable`, …), which should be
  made public. There is no bundled `na22.sou`; adding one (511,
  1274.537 keV) lets Na-22 runs take the source route.
- Carried over from Task 4: `save_mtx` is pure-Python per tag (≈ 1.8 s
  for 10⁵ entries at 16384², ≈ 34 s for 10⁷), so writing runs on the
  worker thread with its `progress` callback. Use one `compress` only:
  `json_matrix_io.compress_matrix` (f must divide both sides) and
  `listmode_sort.compress` overlap. Pick `compress_matrix` for output
  and drop or delegate the other.
- Carried over from Task 2's review: the time-window dialog snaps the
  maximum |Δt| to a multiple of the detected granularity (otherwise the
  outermost Δt bin is only partly populated and a random window reaching
  it under-measures), and passes the granularity to
  `listmode_sort.window_mask` so that the counts summed and the widths
  counted use the same bins.
- Tests (pytest-qt): each dialog's state ↔ session round-trip; Back keeps
  choices; Cancel writes nothing; overlapping windows refused; the output
  dialog writes exactly the ticked formats.

### Task 6: Main-window integration
- **Operations → Sort List-Mode γγ Data…** (after Subtract Spectra, behind
  a separator; shortcut chosen after a check against the shortcut table,
  none if nothing is free and sensible).
- Opening an SData file through File → Open (no histograms inside) offers
  to start the operation instead of reporting "nothing to open".
- Singles go to the main window as `LoadedSpectrum`; the result opens via
  `_open_matrix_panel(..., axes=identity)`.
- **Verify the panel handles a float matrix with negative cells** (cuts,
  projections, heatmap log scale, Symmetrize's refusal of negatives with a
  clear message, since `rebin_rows` refuses negative counts) before relying
  on it ([[feedback_verify_the_ui_reaches_the_feature]]).
- Carried over from Task 4: File → Open of a SpectraTools JSON must use
  `load_json_matrix_with_meta` and pass its axes to the panel (compressed
  files carry a (f−1)/2-offset `ChannelAxis`). `load_json_matrix` can now
  return float64 for such files. A full-resolution sparse JSON still
  densifies to 2 GiB on load: measure it, and refuse with a clear message
  if memory is short.
- Carried over from Task 5's reviews:
  - File → Open of an SData file starts the operation **with that file
    already opened** (today `initial_path` only sets the picker's folder);
    pass the main window as `parent` (theme) and wrap
    `_open_matrix_panel(..., matrix=, axes=)` as `session.open_matrix`.
  - Keep the step-5 output options on the session (folder, base name,
    formats, factor) so a Back visit restores them; replace the step-5
    `_on_next` override with a small `StepDialog` hook.
  - Step 4: warn when the random windows hold (almost) nothing, as happens
    with a FILTERED stream.
  - Public `listmode_io.num_entries(path)` (the gain dialog uses the
    private `_open_tree`). One shared 1-D `load_spectrum_file(path)` for
    main_window and the gain dialog (today copied as `_load_1d`).
    `GainPair.matcher_ok` to keep the matcher's "suspect" flag across
    edits.
  - Cosmetic: the max-|Δt| note in step 1 clips at 700 px.
- Measure the memory of a 16384² result in the panel on this machine
  ([[feedback_measure_the_real_system]]) and state the number in the help.

### Task 7: Help, CHANGELOG, version
- HowTo section for the operation, every claim fact-checked against the
  code ([[feedback_factcheck_docs_against_source]]); the FILTERED-stream
  warning explained; the weighting formula with its TV/HDTV citation.
- CHANGELOG 7.0.0 entry ([[feedback_update_changelog_on_freeze]]); version
  strings bumped only for test builds and the freeze.

### Task 8: End-to-end check on the real run, then the release pipeline
- Drive the whole operation in the running app on
  `Na22_261126_ANI_STEF_180/UNFILTERED/SData_*.root`. Report the
  prompt/random/net totals and `w`, then gate on 511 keV in X and check
  that 511 keV appears in Y (Na-22 back-to-back annihilation) and survives
  subtraction.
- Full suite on Windows (count compared with the baseline plus the new
  tests), then the usual build → install → smoke on Windows, Ubuntu-24.04
  and AlmaLinux-10. **User confirmation before the freeze**; merge
  `v7-major` into `master`, then tag. No push or release without an
  explicit go-ahead.

## Fixtures

Real list-mode data is not in git. Proposal: write a **trimmed UNFILTERED
fixture** (the first ~30 s of the Na-22 run, ≈ 35 k events, well under
1 MB) with `uproot` into `tests/fixtures/compass/listmode_na22_30s.root`,
plus the matching `info.txt`. It needs your OK to commit, as the `.mtx`
fixtures did. Until then the real-data tests skip **by name** when the
file is absent, and the synthetic generator covers the logic.

## Not in scope for 7.0.0

- More than two detectors, or event building beyond pairs (γγγ cubes).
- A variance (error) matrix alongside S. Possible later: Var = P + w²R
  is cheap to sort at the same time.
- Writing the prompt and random matrices separately. Not chosen; trivial to
  add later as a checkbox.
- Explaining CoMPASS's FILTERED selection or the `info.txt` input-count
  mismatch (both noted in memory as unexplained).

## Open points

1. ~~JSON at full resolution~~ — settled 2026-11-27: option (c), see the
   decisions table and Task 4.
2. The flag-bit defaults in step 1 (Task 1 settles them from CAEN's
   documentation).
3. Committing the trimmed real-data fixture (needs the user's OK).
