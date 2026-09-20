# Automatic MC Efficiency Calibration — Design (v6.0.0)

**Status:** draft for review. Nothing is implemented.

**Goal.** After an automatic energy calibration, fit a relative detector
efficiency curve ε(E) by Monte Carlo using two independent models, display
and examine it the way CalEnEff does, save it, and let the user apply it to
any spectrum to produce an efficiency-corrected copy.

**Reference implementation.** `C:\Users\RIG\Documents\Claude\efficieny`
(CalEnEff, `ra226_gui.py`). The maths below is a port of it, not a
reinvention. Where this document and that source disagree, the source wins
and this document is wrong — see §12.

---

## 1. Why this is cheap to build

Everything the fit needs already reaches the window it will launch from.

`CalibrationPlotDialog.__init__` receives `points` — documented in that file
as `(channel, channel_err, area, area_err, energy)` per point — and
`source_lines`, whose `SourceLine` dataclass carries `energy`, `energy_err`,
`intensity`, `intensity_err`.

CalEnEff's input is a 7-column file: `ch Δch N ΔN E[keV] I[%] ΔI[%]`. Every
one of those seven is already in the dialog. No plumbing change is needed to
obtain the data; the work is the fit, the window, the files and the apply
step.

**One scale difference, which cancels.** `sou_io`'s own docstring states the
intensity is relative, normalised so the strongest line reads 10000 (na24 at
1000) — not a percentage. Since ε = N/I is itself relative and is then
normalised again (§4), a constant factor common to every line of one source
divides out exactly. It must NOT be "converted to %": that would be a no-op
at best, and wrong if the assumed normalisation were ever different.

---

## 2. The two models

Ported verbatim from the reference.

**KFR — 4 parameters**

    ε(E) = (a·E + b/E) · exp(c·E + d/E)

Fitted by `scipy.optimize.curve_fit`, `method="trf"`, bounds
`([0, 0, -inf, -inf], [inf, inf, 0, inf])`, multi-start over five seeds: the
data-driven seed (from the geometric means of E and ε), three perturbations
of it, and the fixed fallback `[1.0, 1e3, -1e-3, 0.0]`. Lowest weighted χ²
wins.

**Radware — 5 free parameters** (Radford, `effit.c`, EFFIT v4.0)

    ln ε(E) = f · (1 + r^g)^(-1/g)
    f1 = a1 + a2·x + a3·x²      x = ln(E/100)
    f2 = a4 + a5·y + a6·y²      y = ln(E/1000)
    f  = min(f1, f2),  F = max(f1, f2),  r = f/F

with **a3 = 0 and g = 15 held fixed**, which is Radford's own default
(`freepars[2] = 0`, `freepars[6] = 0`). Fitted by `method="lm"`, no bounds,
seeded first by Radford's `parset()` logic and then by an independent
2-degree polyfit of ln ε; lowest χ² wins.

Two details in the reference are load-bearing and must survive the port:

- `f1` and `f2` are **log**-efficiencies and are normally negative. An
  earlier version of the reference clipped them positive with
  `np.maximum(..., 1e-30)`, which made the loss landscape pathological and
  stalled the MC. Do not clip.
- The `(1 + r^g)^(-1/g)` evaluation goes through `logaddexp` with the
  exponent clipped to ±700, or it overflows.

---

## 3. The Monte Carlo

`N_MC_EFFICIENCY = 10000`, fixed RNG seed so a repeated run reproduces its
numbers.

Per iteration: resample `N ~ Normal(N, ΔN)` and `I ~ Normal(I, ΔI)`; reject
the sample outright if any `N ≤ 0`, any `I ≤ 0`, or any resulting ε is
non-finite or ≤ 0; otherwise refit both models to the resampled ε and store
the parameter vectors.

- KFR refits with `method="lm"`, warm-started from the best fit.
- Radware refits with a **fresh `parset()` seed computed from the resampled
  ε**, not a rolling warm start. This mirrors `effit.c` calling `parset()`
  then `fitter()` for each new data set, and avoids chain bias.
- Both use `maxfev=1500` and `ftol = xtol = gtol = 1e-5`. The reference
  documents these as deliberate: σ-level accuracy is all that is needed, and
  the scipy default of 1e-8 is far too slow for 10 000 × 2 fits.

Failed or non-finite fits are skipped, not retried. The counts of accepted
KFR fits, accepted Radware fits and rejected samples are reported in the
window and written into the file headers — a band computed from 200
surviving samples means something different from one computed from 9 900,
and hiding that would be dishonest.

**Which curve is "the efficiency".** The Monte Carlo is how the uncertainty
is obtained; the curve itself is the **best fit**. That is the reference's
own division — `_draw_efficiency` plots `f_kfr(E_g, *eff_popt)`, the best
fit, and draws the MC family as a band around it, while
`predict_efficiency` returns the best-fit value and the MC mean ± σ
side by side. So: the curve written to file, drawn in the window and divided
into a spectrum is the best fit, and every `deff` comes from the MC. The two
differ slightly but really — on the reference's own Ra-226 data at 1155 keV
the best fit gives 534.562 against an MC mean of 534.762 — so this is a
choice to state rather than leave to whichever the implementer reaches for.

**The MC refits against the ORIGINAL `deff`,** not one recomputed from each
resampled `N_s`/`I_s`. The reference passes `sigma=deff` unchanged inside the
loop. Recomputing it per sample looks like a correction and is not one: the
weights would then vary with the noise draw, which changes what the spread
of fitted parameters measures. Port it as it is.

Uncertainty bands are the 15.87 / 84.13 percentiles of the MC curve family,
**Birge-scaled** about the best-fit curve, exactly as the reference does.
Two details of that, both checked against the source rather than assumed:

- The band is evaluated from the first `N_BAND = 4000` stored samples, not
  from all 10 000. Evaluating every sample on the plotting grid is what makes
  redrawing slow, and 4 000 is already far more than a percentile needs.
- `B = sqrt(chi2/ndf)`, and the reference **returns 1.0 when ndf ≤ 0** rather
  than dividing by zero: an exactly-determined fit carries no scatter
  information, so the stated σ are used unscaled. With
  `MIN_EFFICIENCY_POINTS = 6` and Radware's 5 free parameters, ndf can be as
  low as 1, so this branch is reachable and must be ported.

**Runtime.** About one minute in the reference. It therefore runs on a
worker thread behind a cancellable progress dialog. Blocking the UI for a
minute is not acceptable, and the project already has threaded-load
precedent.

---

## 4. Normalisation

**Decision (user, 2026-09-20): scale by the peak of the model the user has
selected to apply.**

`norm = 1 / max(selected_curve)`, evaluated on a fine grid over the fitted
energy range. **Both** curves are multiplied by that one factor, so the
selected curve peaks at exactly 1.0 and the other is drawn on the same scale
and stays directly comparable to it.

Consequences, all intended:

- The applied efficiency is ≤ 1 across the fitted range, which is what
  "in %, always less than 1" asks for.
- The unselected curve may exceed 1 where it runs above the selected one.
  That is real information about how far the models disagree, not an error
  to clip away.
- **Switching the selected model rescales both curves**, so the numbers
  written to the files depend on which model was selected when they were
  saved. The file header therefore records the selected model and the
  numeric normalisation constant.

This differs from CalEnEff, which always normalises by the KFR peak
(`eff_norm = 100 / peak_KFR`). The deviation is deliberate and was chosen
over matching the reference.

---

## 5. User interface

### 5.1 Entry point

`CalibrationPlotDialog` (title "Energy Calibration") gains a button:
**Auto MC Efficiency Calibration**.

It is enabled only when every precondition holds, and when disabled its
tooltip names the one that failed:

- at least `MIN_EFFICIENCY_POINTS` matched peaks (§9),
- every matched peak has `area > 0` and `area_err > 0`,
- every matched source line has `intensity > 0` and `intensity_err > 0`,
- every matched energy is > 0 — both models evaluate `ln E` and `b/E`.

A disabled button with a stated reason is better than an enabled one that
fails inside scipy with a message pointing at the fitter rather than at the
data.

### 5.2 The efficiency window

A new modeless dialog, themed like the rest of the app in both light and
dark. CalEnEff's palette is its own and is not copied.

- **Main axes** — ε vs E with Δε error bars; KFR solid with its 1σ band;
  Radware dashed with its 1σ band; legend carries each model's Birge factor.
- **Residual strip** below, sharing the x axis — Δε for both models with
  ±RMS lines, matching how `CalibrationPlotDialog` already presents
  residuals.
- **Examine** — an energy entry; on submit, report for each model the MC
  mean ± σ and the best-fit value at that energy. Fewer than 10 surviving
  finite MC samples reports NaN rather than a meaningless σ, as the
  reference does.
- **Model selector** — KFR / Radware radio buttons. Drives the normalisation
  (§4), what the files record, and what Apply uses.
- **Save** — writes both files (§6).
- **Summary** — per model: χ², ndf, χ²/ndf, Birge factor, RMS; plus the MC
  accepted and rejected counts and the fitted energy range.

---

## 6. Output files

Two files per save. **Channels are not stored**: the energy calibration is
available wherever the file is used, so a channel column would duplicate a
derived value on disk where it can go stale (user decision, 2026-09-20).

**Per-peak** — one row per matched peak:

    E  dE  eff_kfr  deff_kfr  eff_rw  deff_rw

`dE` is the source line's literature energy uncertainty.

**Per-bin** — the curves sampled at every channel's energy:

    E  eff_kfr  deff_kfr  eff_rw  deff_rw

No `dE` column: a sampled curve point has no energy uncertainty, and writing
a zero column would be writing a meaningless number.

Both files carry a comment header recording the selected model, the numeric
normalisation constant, the fitted energy range, each model's parameters,
χ²/ndf and Birge factor, the MC accepted and rejected counts, the source
file, and the energy calibration coefficients in force.

`deff` is the Birge-scaled 1σ MC width at that energy. It is written for
inspection only; §7 does not use it.

---

## 7. Applying an efficiency to a spectrum

Available once an efficiency exists, for **any** loaded spectrum.

    corrected[i] = counts[i] / ε(E(i))

where `E(i)` is the bin's energy under the **active** energy calibration and
ε is the **selected** model's normalised curve.

- **Extrapolation** — the curve is evaluated outside the fitted energy range
  without restriction (user decision, 2026-09-20, taken with the divergence
  risk stated).
- **Non-positive efficiency — zero the bin** (user decision, 2026-09-20).
  Where the evaluated ε is `≤ 0` the corrected bin is set to 0 rather than
  divided. Everywhere ε is positive the user gets the extrapolation they
  asked for, however extreme.
- **Non-finite efficiency — also zero the bin.** This is an extension of the
  rule above, not a separate instruction, and is called out because the rule
  as stated does not cover it: under IEEE comparison `NaN ≤ 0` is **false**,
  so a NaN efficiency would slip past a literal `eff <= 0` test and leave a
  NaN count behind. NaN counts then propagate into plotting, autoscaling,
  fitting and integration, where they are far more damaging than a zero.
  `ε = ±inf` already divides to ≈0, so zeroing it changes nothing and keeps
  the rule uniform: **a bin is corrected only where ε is finite and > 0, and
  is zeroed otherwise.**
- **Reported** — the dialog reports how many bins were zeroed, split by
  reason (ε ≤ 0 versus non-finite). A silently truncated spectrum would be
  indistinguishable from a genuinely empty region.
- **Uncertainty** — Δε is deliberately NOT propagated (user instruction:
  "only eff is taken into account"). The spectrum's own variance, where it
  has one, is scaled by `1/ε²`, which is ordinary propagation for a per-bin
  scale factor. Leaving it alone would produce a spectrum whose stored
  variance no longer matches its counts, and would silently corrupt any later
  fit. Zeroed bins get variance 0; that is safe because `peak_fit.py:882`
  already floors the fit weight with `np.sqrt(np.maximum(variance, 1.0))`, so
  a zero-variance bin cannot divide by zero in the weighting. Verified in the
  source, not assumed.
- **Result** — a new `LoadedSpectrum` added through
  `MainWindow._add_combined_spectrum`, so it picks up the next colour on the
  5.2.6 ramp, collision-safe naming and the correct auto-log path. Named
  `<original> [eff-corrected KFR]` or `[eff-corrected RW]`. The original is
  untouched and both are displayed together.

**Precondition: an active energy calibration.** Without one, bins have no
energies and the correction is undefined. The action is disabled with that
reason.

**Calibration drift.** The efficiency is stored together with the
calibration it was derived under. If the active calibration differs when
Apply is used, the user is warned before proceeding: the same ε(E) applied
through a different channel→energy map is a different correction, and
nothing else would reveal it.

---

## 8. Module structure

| module | contents | Qt? |
|---|---|---|
| `efficiency.py` | `f_kfr`, `f_radware`, `f_radware_5p`, seeds, `_multistart`, `EfficiencyResult`, `calibrate_efficiency`, `predict_efficiency`, normalisation | no |
| `efficiency_dialog.py` | the results window of §5.2 | yes |
| `efficiency_io.py` | the two writers of §6 | no |
| `efficiency_apply.py` | the correction of §7, as a pure function over counts, calibration and curve | no |

This mirrors the existing `calibration.py` / `calibration_plot_dialog.py` /
`calibration_quality.py` / `caleneff_export.py` split: maths without Qt, one
dialog, one I/O module.

---

## 9. Constants

| name | value | why |
|---|---|---|
| `N_MC_EFFICIENCY` | 10000 | reference `N_MC_EFF` |
| `N_BAND` | 4000 | reference; samples used for the plotted band |
| `EFFICIENCY_SEED` | 42 | reference `SEED`; fixed so a rerun reproduces |
| `MC_MAXFEV` | 1500 | reference |
| `MC_TOL` | 1e-5 | reference |
| `RADWARE_G` | 15.0 | Radford default, held fixed |
| `RADWARE_C` | 0.0 | Radford default, held fixed |
| `MIN_EFFICIENCY_POINTS` | 6 | Radware has 5 free parameters, so ndf = n − 5 must exceed 0; 6 is the smallest n leaving any redundancy at all. **Ours, not the reference's** — CalEnEff's `MIN_POINTS = 4` guards its *energy* fit (3 free parameters), a different constraint. |

---

## 10. Decisions taken, and by whom

No decision in this document is still open.

| decision | choice | source |
|---|---|---|
| Normalisation | peak of the **selected** model = 1.0 | user, 2026-09-20 |
| Extrapolation | unrestricted, outside the fitted range | user, 2026-09-20 |
| Files | two — per-peak and per-bin, each carrying both curves | user, 2026-09-20 |
| Channel columns | not stored; the calibration is available | user, 2026-09-20 |
| ε ≤ 0 when applying | zero the bin | user, 2026-09-20 |
| ε non-finite when applying | zero the bin | §7, as the uniform form of the rule above — `NaN ≤ 0` is false, so the literal rule would not catch it |
| Zeroed bins' variance | 0, relying on the existing fit-weight floor | §7, verified at `peak_fit.py:882` |

Everything else follows the reference implementation.

---

## 11. Out of scope

- **Absolute efficiency.** ε = N/I is relative. The corrected spectrum's
  absolute scale is arbitrary and is set by §4's normalisation. This is
  correct for comparing relative intensities and is not an activity
  calibration. Nothing here may be described to the user as absolute.
- Reading an efficiency back from a saved file. The files are output; an
  efficiency is produced by running the calibration.
- Efficiency in the matrix panel's projection view.
- Any change to the energy calibration itself.

---

## 12. Testing

**The external oracle, and the reason to trust any of this.** CalEnEff ships
`226Ra_En_Area.txt`, `demo1.txt` (Ba-133) and `demo2.txt` (Eu-152). Running
our port and the reference on the same input must give the same best-fit
parameters for both models, to a stated tolerance. That is a genuine
external check rather than a self-consistency one, and it is the single most
valuable test in this feature. The inputs are copied into `tests/fixtures/`
so the suite never depends on the reference directory existing.

Alongside it:

- Model shape — known analytic values of `f_kfr` and `f_radware_5p`.
- Normalisation — the selected curve peaks at exactly 1.0; switching the
  selection rescales both curves; the unselected one is allowed to exceed 1.
- MC — a fixed seed reproduces; rejected non-physical samples are counted
  rather than silently dropped; a band from too few survivors reports NaN.
- Files — column counts and header contents; no channel column anywhere; a
  round-trip read of the numbers.
- Apply — counts divided bin by bin; variance scaled by 1/ε²; a new spectrum
  appears and the original is unmodified; the action is disabled without an
  active calibration.
- Apply, the zeroing rule — a bin whose ε is negative is zeroed, and so is
  one whose ε is `NaN`, `+inf` or `-inf`. **The NaN case gets its own test**:
  it is the one an implementation using a literal `eff <= 0` would fail,
  since that comparison is false for NaN, and the resulting NaN count would
  then spread through every later operation on the spectrum. The counts of
  zeroed bins are reported and split by reason.
- Apply, no NaN escapes — assert the corrected counts array is entirely
  finite for an efficiency curve deliberately built to go negative, zero and
  NaN across the range.
- **Controls that must fail.** Per the project's standing rule every one of
  the above gets a control: a deliberately wrong curve must fail the oracle,
  an unscaled curve must fail the normalisation test, and a correction that
  leaves counts untouched must fail the apply test.

---

## 13. Risks

1. **Radware convergence.** The reference documents real stalls and an
   earlier clipping bug. If MC acceptance for Radware is poor on our data,
   that is reported honestly in the window rather than papered over.
2. **Runtime.** About a minute on the reference's 23 points, and more points
   cost more. Threaded and cancellable, with iteration counts shown.
3. **Extrapolation.** §7 divides by a curve outside the range where it was
   fitted, at the user's explicit instruction. A corrected spectrum can be
   dominated by its extrapolated tails. The fitted range is recorded in the
   file header and shown in the window so the effect is visible rather than
   silent.
