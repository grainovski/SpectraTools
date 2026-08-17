# v4.0.0 — ROOT support and the HDTV inheritance

**Goal:** add ROOT file support, and act on the findings from the HDTV study that the user adopted.

**Source of the work:** the HDTV study (https://claude.ai/code/artifact/2bf61a67-e21c-41a4-92f6-924e071600f3), walked through finding-by-finding with the user on 2026-08-17. All 26 findings were dispositioned: **19 adopted, 5 skipped, 2 deferred.** Three additional scoping questions about ROOT itself were settled in the same session.

**Scope decisions (user, 2026-08-17):**
- **ROOT reading is histogram objects only** — `TH1` → spectrum, `TH2` / `THnSparse(2D)` → matrix. No `TTree` event-data histogramming in v4.0.0; it has no HDTV precedent and needs a whole new UI surface.
- **The `uproot` prototype gates everything.** If it cannot be frozen into a onefile build, the architecture changes and this plan's Phase 4 is rewritten.
- **The backend interface goes in after the cut fixes, before the ROOT backend.** C1 mostly *removes* clamping logic rather than adding backend-specific code, so extracting an interface around an already-correct algorithm is cheaper than refactoring first.

**Explicitly out of scope** (user decisions, do not re-raise): F7 right tail + step width; F8 RejectPoint masking; X4 efficiency calibration (**skipped, not deferred**); X7 normalise; C5 (nothing to fix — we have no step term at all). **Deferred to a later release:** X2 nuclide databases (blocked on licensing/size), X9 several named cuts per matrix.

**Standing rules that apply throughout:** verify every TV/gf3 parity claim against the vendored C source, never against spec prose. Fact-check every user-facing sentence against real behaviour. Use the `_rendered_text` tag-stripping helper for help-text assertions — raw-HTML substring checks have produced false positives three times. Add a CHANGELOG entry when the version is frozen and copy CHANGELOG.md into `releases/vX.Y.Z/`.

---

## Phase 0 — Gate

### Task 1 — Prototype `uproot`, and settle test fixtures
Nothing else in Phase 4 can be planned in detail until this is answered.

- Read a real `TH1` and a real `TH2` with `uproot`; confirm bin edges, contents and any stored axis titles come through.
- Determine whether `THnSparse` is covered. If it is not, decide then whether sparse support is dropped from v4.0.0 or written by hand.
- **Build a Windows onefile exe containing `uproot` and run it.** This is the actual risk. Also confirm the Linux `.deb`/`.rpm` builds still work with the new dependency, on the AlmaLinux-8 glibc floor.
- Record the installed size delta — `uproot` pulls `awkward` and `numpy` transitively.
- **Test fixtures: we have no `.root` files at all** (verified: zero in the repo). Either the user supplies representative files from their own analysis — strongly preferable, since that is what must actually work — or we generate them with `uproot`'s writing support. Committed fixtures must be small; large sample files follow the existing "intentionally untracked" convention.

**Gate:** if the onefile build fails, stop and re-decide the reader (own minimal `TH1`/`TH2` reader was the user's option 3 and remains the fallback). Phases 1–3 and 5–6 do not depend on this and can proceed regardless.

---

## Phase 1 — Matrix cut correctness

### Task 2 — C1: weight the background by lines actually summed
Replace the unclamped marked width (TV's `MrkRange`) with the count of lines actually summed after clipping, per HDTV's `VMatrix.cxx:105-114` and `histogram.py:505`.

- Measured defect: on a flat matrix where the correct net is exactly zero, a background mark overhanging an edge leaves **250 of 1000 counts per channel unsubtracted (25%)**. HDTV gives 0.
- **This deliberately breaks bit-parity with TV** for edge-overhanging marks. The user accepted that trade.
- The existing randomised TV-parity sweep must be **re-scoped, not loosened**: it currently claims parity for background regions overlapping the matrix, which is exactly the domain this changes. Narrow it to fully-inside regions, where parity still holds, and add a test pinning the new divergence *against* the TV oracle so it cannot silently drift back. A weakened sweep would stop checking parity at all — the same trap avoided in v3.1.4.
- v3.1.4's existing divergence (ignoring wholly-outside regions) becomes a consequence of one uniform rule rather than a special case. Simplify `_overlaps` / `compute_cut` accordingly if it falls out.
- The KB section "How the cut's background is weighted" describes the current rule and **must be rewritten** — it currently explains the `MrkRange` behaviour as correct.

### Task 3 — C2: merge overlapping regions
One shared boundary-list merge helper, used by **both** the matrix cut/background regions and the fit's background regions.

- Measured defect: on a ramped background, nested marks (0–29 plus 25–29) give 8821.4 against a correct 9000 — **−1.98%**; an asymmetric overlap gives −0.40%.
- **A symmetric overlap cancels exactly.** Any test must use an asymmetric or nested overlap and a sloped background, or it will pass while the bug is present.
- Model on HDTV's `VMatrix::AddRegion` (`VMatrix.cxx:25-61`) — a sorted boundary list with a parity walk, merging on insert. `PolyBg::AddRegion` is the same algorithm for the fit side.
- No parity judgment needed: there is no defensible reading in which a twice-marked channel counts twice.

### Task 4 — C4: several foreground gates per cut
**Depends on Task 3** — reuses the merge helper.

- `compute_cut` takes a list of foreground regions rather than one, with the cut-line count accumulating across them (HDTV `histogram.py:485-498`, and `AddCutRegion` on the C++ side).
- Marking UI in the matrix panel needs to support adding, listing and clearing several gates. Ctrl+C must clear all of them, consistent with the v3.1.3 "exact analogy to Clear Marks" requirement.
- Arithmetic generalises for free once merging exists; the work is UI.

### Task 5 — C3: per-channel variance on derived spectra
- Compute the variance as a second accumulator over the lines already being summed: `gross + f²·background`. Near-free — same loop.
- Attach it to cut results and to Add/Subtract results.
- `peak_fit` uses it when present, **falling back to the current Poisson `sqrt(max(y,1))` when absent** so ordinary spectra are unaffected.
- This fixes genuinely wrong weighting: a background-subtracted cut is not Poisson, its variance exceeds its own count, and it can go negative where the 1.0 floor is arbitrary. Expect reported parameter errors and χ² on cut spectra to **grow** — that is the correction, not a regression.
- Display and export of the error array are **deferred** (user decision). Store it; do not surface it yet.

---

## Phase 2 — Peak area and its uncertainty

These three touch the same code and should land as one coherent change. **Every one of them moves numbers users have seen**, so the changelog needs to say so plainly.

### Task 6 — F3: closed-form tail-inclusive area
The highest-priority finding in the whole study.

- `peak_fit.py:526-530` currently integrates the plain Gaussian core only and says so: the tail's contribution "is not included here — a known approximation". With the left tail enabled, **every reported area is systematically low** by roughly what the tail holds.
- Derive the analytic integral of the full gf3 Hypermet shape (`hypermet_left_tail`) and use it in place of `amplitude * sigma * sqrt(2*pi)`. **Verify the derivation against `srcRW/gf3_subs.c`**, not against HDTV — HDTV's `GetNorm` is the Theuerkauf shape, a different parametrisation, and is a model for the *approach* only.
- Sanity-check the closed form numerically against a fine-grained numeric integral of the same fitted shape before trusting it.
- Existing area assertions will move. They are wrong today; update them, and record the before/after magnitude for a representative tailed fit so the changelog can quantify it.
- Add HDTV's memoisation on the shape parameters **only if profiling shows it matters** (user decision).

### Task 7 — F2: propagate the area error through the covariance
- `peak_fit.py:531-536` adds relative errors in quadrature, which assumes amplitude and sigma are uncorrelated. In a peak fit they are strongly anti-correlated, so the true variance is **smaller** than we report — our area errors are inflated.
- Replace with `J·C·Jᵀ` over the relevant covariance block, where `J` is the gradient of the (new, Task 6) area expression with respect to the fitted parameters.
- **Not** reparametrising to volume as HDTV does (user decision): equivalent at first order, far more churn.
- Must handle the degenerate cases we already know about — NaN when the covariance is unavailable or the relevant block is rank-deficient, matching the existing convention rather than inventing a number.

### Task 8 — F6: background uncertainty, in the net area and on the plot
- `peak_fit.py:544` currently asserts `full_area_err = area_err` because the linear background is "uncertainty-free". It is not; we simply never computed it.
- Implement HDTV's `PolyBg::EvalError` as `sqrt(v @ cov @ v)` with `v = [1, x, x², …]` (`PolyBg.cxx:238-262`). We already hold the background fit's covariance.
- Feed it into the net/full area error as `sqrt(e_hist² + e_bg²)`, per `TH1BgsubIntegral::GetBinError2`.
- Draw it as a shaded band around the background line. Must work in both themes — the v2.2.0 background-line work already hit an invisible-in-dark-theme colour bug.
- Return NaN when no covariance exists, as HDTV does, matching our tail-uncertainty convention.

---

## Phase 3 — Fit conditioning

### Task 9 — F4: shared width by default
- Flip `link_widths` to default on, matching TV's actual behaviour.
- Needs a UI affordance to unlink for the cases that genuinely want independent widths.
- Existing multi-peak fits will produce different — generally better conditioned — results. Changelog callout required.
- Do this **before** Task 10 so the seeding sweep is run against the default that actually ships.

### Task 10 — F1: integral-based initial parameters — *conditional*
**Depends on Task 9.**

- Port HDTV's `_Fit()` seeding: total volume over the region minus background, distributed among peaks by amplitude ratio, with one common width backed out as `sumVol / (sumAmp * sqrt(2*pi))`. Honour fixed parameters by holding them out of the sums, as HDTV does with `sumFreeVol`/`sumFreeAmp`.
- Skip the step-height half of HDTV's estimation — we have no step term (C5).
- **This is gated on measurement.** Judge it on the same randomised sweep that validated the current halved seed: convergence and failure rates across 1–4 peaks, sigma 1.2–10 ch, separations 1–4 sigma. The v3.1.0 M1 work set the precedent — the halved seed was kept because it measured better (10 convergence failures → 0), not because it matched TV.
- **If the sweep does not show it better, keep the current seed and record why.** A width from an integral *should* survive overlap better than one measured from the trace, but that is a hypothesis until measured.

### Task 11 — F5: joint background + peak fit
**Depends on Tasks 6, 8, 10.**

- Append background polynomial coefficients to the peak parameter vector and fit in one pass, as HDTV's internal-background mode does.
- Seed the background at the lowest bin in the region (HDTV uses the leftmost bin when a step is present — not applicable to us).
- Fitting the background first and freezing it discards the peak–background correlation, so peak errors come out too small. Expect reported errors to **grow**; that is the correction.
- Largest change in this phase: it grows the parameter vector, interacts with the background-marking workflow, and its interaction with Task 8's error band needs thought (the band's covariance now comes from the joint fit).

---

## Phase 4 — ROOT support
**Depends on Task 1's gate.**

### Task 12 — Extract the matrix backend interface
Model on `VMatrix`/`RMatrix`/`MFMatrix` (`src/mfile-root/VMatrix.h:60-71`) — five methods, the important one being "add one line into an accumulator".

- Move `compute_cut` and `compute_projection` behind the interface. The cut arithmetic is format-independent and stays shared.
- **Mark-to-index conversion belongs in the backend**, not in shared code: ROOT axes are 1-based with under/overflow and use `TAxis::FindBin`, while our `_nint` follows TV's `NINT`. HDTV's own two backends already disagree here — `MFMatrix::FindCutBin` is `ceil(x-0.5)`, which rounds a half-channel *down* where `_nint` rounds away from zero.
- Purely structural: no behaviour change, and the existing matrix tests must pass untouched.

### Task 13 — ROOT reader
- `TH1` → spectrum (histogram + calibration from the axis where present). `TH2` and `THnSparse(2D)` → matrix, via a backend implementing Task 12's interface.
- Follow HDTV's type gate (`rootInterface.py:247`, `:369`) and its defensive asymmetry: **glob patterns allowed for 1D spectra, deliberately refused for matrices**, because it makes accidentally loading enough huge objects to crash the program too easy. That is a real accident class we can also have.
- ROOT files hold a directory tree, so the open dialog needs to let the user navigate it and pick objects — not just choose a file.
- Read-only, consistent with our other readers (n42, spe, spk, mtx).
- Guard the failure modes the N42 work established: unreadable file, wrong object type, dimension mismatch, empty histogram — each with a message naming the file and the object.

### Task 14 — UI integration
- File menu entry, wired through the existing `_try_load_spectrum` / `_load_files` path so calibration application and the single-redraw behaviour (v3.1.0 M18) are preserved.
- Matrix opening goes through `_load_matrix_with_progress` and the existing `_MatrixLoadWorker`, so large ROOT matrices get the same progress window and threaded load.
- Auto-apply an embedded calibration if none is active, matching the N42 reader's behaviour.

---

## Phase 5 — Performance

### Task 15 — P1: cache the decoded matrix
**Deliberately not HDTV's design.** HDTV caches projections (`.prx`/`.pry`) because it never holds the matrix; we do hold it, so projections cost milliseconds and the real expense is the ~4.9 s decode.

- Cache the decoded array as `.npy` **in a cache directory, not beside the user's data**.
- Key on source path + mtime + size so a changed file cannot serve a stale cache.
- Memory-map on reload.
- Needs a size cap or eviction policy — a cache of 500 MB matrices grows without bound otherwise.
- Skip the `.tmtx` transpose: it only pays off under streaming, and we can transpose in memory.
- **Beware the measurement traps from v3.1.1:** benchmark in a fresh process, best-of-three. Repeated large allocations in one process reversed a timing result once already.

### Task 16 — P3: hold matrix-derived caches weakly
- Memoised projections and similar large derived objects held by `weakref`, rebuilt on demand if collected (HDTV `matrix.py:66-86`).
- Structural insurance against exactly the 537 MB-per-open leak class fixed in v3.1.2 — a cache should never be the only thing keeping a large object alive.
- Keep the existing explicit clearing in `closeEvent`; the weakref is a second line of defence, not a replacement.
- The existing leak test is scoped to its own panels' weakrefs (other tests leave panels open) — keep that scoping.

---

## Phase 6 — Features

### Task 17 — X1: save and reload fits
The largest user-facing gap found: analysis currently does not survive closing the app.

- Build on the existing JSONL export. **Add a schema version field from day one** and keep per-version readers as the schema evolves, following HDTV's `fitxml.py` (readers retained from `_v1_3` back to `_v0`).
- Support **both** restoring stored values and re-running the fit from the saved marks — HDTV's `refit` flag. This matters more than usual here, because Tasks 6, 9, 10 and 11 will all change what a refit produces.
- Round-trip test: save, reload, and assert the restored results match, including NaN uncertainties surviving as NaN (the `_json_safe` NaN → `null` convention already exists).

### Task 18 — X3: assign literature energies, then recalibrate
- Attach a known energy to a fitted peak; refit the calibration from the accumulated assignments; repeatable as peaks are added or corrected.
- Uses fitted centroids rather than eyeballed positions — that is the accuracy gain.
- Extends the existing calibration dialog. Calibration *copying* between spectra was **not** included (user decision).

### Task 19 — F9: peak finder units and bad-fit rejection
- Accept the expected peak width in **keV** when a calibration is active, converting to channels (HDTV `peakfinder.py:77-88`), and validate the converted value is positive.
- Reject auto-fits whose fitted width is ≤ 0 or more than 5× the expected sigma (`peakfinder.py:167`).
- The rejection only exists because the keV input gives the search an expectation to test against — do both or neither.

### Task 20 — X5: rebin
- Group *n* channels into one, and **transform the active calibration alongside it** — the detail worth copying from HDTV's `cal.Rebin(ngroup)`. Leaving the calibration stale is the easy mistake.
- `calbin` (rebinning to calibration units) excluded (user decision).
- Interaction to resolve: what happens to existing fits and marks on a rebinned spectrum. Simplest defensible answer is to clear them, since their channel positions no longer mean the same thing.

### Task 21 — X6: reload from disk
- Menu action plus shortcut that re-reads the active spectrum's file in place, **preserving calibration, fits and marks**.
- Manual only — no auto-watch (user decision), so no timer and no cross-platform file-watching differences.
- Guard the case where the file has changed length: a spectrum that grew or shrank invalidates channel-based marks.

### Task 22 — X8: LaTeX and CSV export
- LaTeX table writer with proper ± formatting and the existing `"n/a"` convention for undetermined values.
- CSV alongside it. **First check what the current export already covers** — there is a `*_report.txt` path and JSONL; CSV should not duplicate an existing format.
- Both are formatting layers over the same result objects.

---

## Phase 7 — Documentation and release

### Task 23 — Help and Knowledge Database updates
Every claim verified against real behaviour, and assertions written against `_rendered_text`, not raw HTML.

- **Rewrite** the KB's "How the cut's background is weighted" section — Task 2 invalidates it.
- KB math section: the new tail-inclusive area formula (Task 6), the covariance-based area error (Task 7), and the background uncertainty formula (Task 8). The existing sigma/FWHM and integration formulas are unaffected.
- Note that shared width is now the default and how to unlink (Task 9).
- HowTo: ROOT file opening and directory navigation (Tasks 13–14), several foreground gates (Task 4), fit save/reload (Task 17), energy assignment and recalibration (Task 18), rebin (Task 20), reload (Task 21), export formats (Task 22).
- The "When an uncertainty reads n/a" section may need extending if Tasks 7/8 introduce new NaN paths.

### Task 24 — Freeze v4.0.0
- CHANGELOG entry (standing instruction), copied into `releases/v4.0.0/`.
- Version bump in `installer.iss`, `build_info.py` and wherever else the version is asserted.
- **The changelog must be explicit that peak areas and their uncertainties changed** (Tasks 6, 7, 8, 9, and possibly 10, 11) and that cut background weighting changed (Task 2). Users comparing v4.0.0 numbers against v3.1.4 will see differences in ordinary results, and they deserve to know which are corrections.
- Builds: Windows onefile + installer, `.deb`, `.rpm`. Same-version rebuilds need `dnf reinstall` / `apt-get install --reinstall --allow-downgrades` and a sha256 check against the freshly built binary — `install` reports "nothing to do" when the version string is unchanged. `gh release create` needs `run_in_background` (317 MB exceeds the 10-minute tool timeout).
- Smoke-test the onefile build by checking the bootloader's **child** process window, per the known gotcha.

---

## Sequencing summary

```
Task 1 (uproot gate) ─────────────────────────────► Phase 4 (12 → 13 → 14)

Task 2 (C1) ─┐
Task 3 (C2) ─┼─► Task 4 (C4)
Task 5 (C3) ─┘

Task 6 (F3) ─► Task 7 (F2) ─► Task 8 (F6) ─┐
Task 9 (F4) ─► Task 10 (F1, conditional) ──┴─► Task 11 (F5)

Tasks 15, 16 (perf)        ─ independent
Tasks 17–22 (features)     ─ independent
Tasks 23, 24 (docs, freeze) ─ last
```

Phases 1, 2, 3, 5 and 6 are independent of the ROOT gate and can proceed whatever Task 1 concludes.

## Risks

- **Task 1 is a genuine gate.** A `uproot` that will not freeze into a onefile build invalidates Phase 4's approach; the fallback is a minimal hand-written `TH1`/`TH2` reader.
- **No `.root` test fixtures exist.** Phase 4 cannot be verified without them, and the user's own files are far better evidence than anything we synthesise.
- **Phases 2 and 3 change numbers users have already recorded.** Five separate tasks move peak areas or their errors. This needs to be one clearly-explained release note, not five scattered ones.
- **Task 10 may be discarded.** It is explicitly conditional on measurement, and its dependent (Task 11) must not assume it landed.
- **Task 2 breaks TV bit-parity deliberately.** The parity test suite must be re-scoped rather than loosened, or it silently stops testing parity at all.
