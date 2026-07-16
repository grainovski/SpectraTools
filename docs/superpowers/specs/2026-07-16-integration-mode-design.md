# Integration Mode — Design Spec

Date: 2026-07-16

## Purpose

Sub-project 4, deferred from
`docs/superpowers/specs/2026-07-14-fit-repeatability-and-parameter-fixing-design.md`:
"A new 'Integration' mode (TV-style direct background-subtracted region
sum, not a Gaussian fit)." TV (`tv-1.9.13/`, vendored into this repo)
implements exactly this in `FIIntegrateRegion`
(`tv-1.9.13/lib/tv/vsFitInt.c`), already used as a reference for other
features in this app. This spec ports TV's own algorithm exactly — per
explicit request to "compute like in TV" — rather than designing a new
one, including several arithmetic quirks in TV's 40-year-old C that look
unintentional but are deliberately preserved here for exact parity (each
called out below, with the alternative considered and rejected).

## Marking and Trigger

Reuses `FitModeState`'s existing marks unchanged: two background regions
(`bg_regions`, the `b` key) and one region (`fit_region`, the `r` key).
No peak marks (`p`) are needed or read — if any are already marked, they
are simply ignored by Integration.

A new `ready_to_integrate()` method on `FitModeState` mirrors
`ready_to_fit()` minus the peak-count check: `len(bg_regions) ==
BG_REGION_CAP and fit_region is not None`.

A new `QAction`, `integrate_button`, mirrors `fit_button`/`clear_fit_button`
exactly (`main_window.py:439-448`): label "Integrate", shortcut `Ctrl+I`,
shortcut-only (no toolbar button, `self.addAction(...)`), initially
disabled, `triggered` connected to a new `FitModeController.run_integration()`.
Its enabled state is updated in `_update_fit_mode_availability()` alongside
`fit_button`, gated on `ready_to_integrate()` instead of `ready_to_fit()`.

Marks persist after a successful integration (same philosophy as a
successful fit — repeat attempts on tweaked regions without re-marking).

## Computation (`peak_fit.py`)

New function `integrate_region(x, y, left_bg_region, right_bg_region,
fit_region)` — no `peak_positions`, `link_widths`, `enable_left_tail`, or
`fixed_params`; none of that applies to a direct sum. Ported as a
**literal, line-by-line translation** of `FIIntegrateRegion`'s
no-fitted-background branch (`vsFitInt.c` lines 19-51, 45-83, 194-309 —
TV's OTHER branch, `FitBgTime(fp)` true, relies on a pre-fit background
*function* this app has no equivalent of for Integration, and is out of
scope). A literal translation (explicit loops mirroring the C structure,
not a vectorized/numpy rewrite) is deliberate: the moment calculations are
dense, order-dependent, and cross-reference earlier-computed values
(mom2 depends on mom1; mom3's uncertainty depends on the already-normalized
mom2; see below) — a vectorized rewrite risks silently changing semantics.
This matches how `_marquardt_fit`/`_measure_width` were already ported
from the same source tree.

Per-channel Poisson variance is the channel's own count: `ds[i] = y[i]`
(TV's separate "variance spectrum" concept collapses to this for a plain
counting spectrum). Channel index and this app's `x` coordinate are
identical (`x = np.arange(len(active.data))`), so TV's integer channel
index `i` maps directly to this app's `x[i]`.

**Gross** (`vsFitInt.c:41-43`, region channels only, no background
subtracted yet):
- `gross_area = Σ y[i]`, `gross_area_err = sqrt(Σ y[i])` (Poisson: variance
  of a sum of independent counts is the sum of the counts).

**Gross moments** (`vsFitInt.c:45-83`, computed on raw `y[i]`, independent
of any background estimate):
- `gross_centroid (mom1) = Σ(i·y[i]) / gross_area`.
- `gross_centroid_err = sqrt(Σ (i−mom1)²·ds[i]) / |gross_area|`, computed
  in the same pass as `gross_variance (mom2) = Σ(i−mom1)²·y[i] /
  gross_area` — note: mom2 normalizes by **plain** `gross_area`, not its
  absolute value.
- `gross_fwhm = FWHM_FACTOR * sqrt(gross_variance)` (reusing the existing
  constant; `sqrt(mom2)` is a "sigma"-like quantity of the channel
  distribution, not a fitted Gaussian sigma).
- `gross_fwhm_err = FWHM_FACTOR * sqrt(Σ (dDlt−mom2)²·ds[i]) /
  |gross_area|` where `dDlt = (i−mom1)²` per channel (`dMom2` in source),
  computed alongside `gross_skewness_raw (mom3) = Σ(i−mom1)³·y[i]`, which
  is then normalized by `|gross_area|` — **absolute value this time**,
  the asymmetric normalization (plain sum for mom2, `abs(sum)` for mom3
  and all `_err` terms) is exactly how TV does it and is preserved as-is.
- `gross_skewness_err = sqrt(Σ (dlt·(dlt²−3·mom2)−mom3)²·ds[i]) /
  |gross_area|` where `dlt = (i−mom1)`, using the already-normalized
  `mom2`/`mom3` from the prior passes.

**Background** (`vsFitInt.c:194-221`, the no-pre-fit-function branch):
pool BOTH marked background regions' channels together into one flat
density estimate:
- `bg_channel_count = (channels in left_bg_region) + (channels in
  right_bg_region)`, `bg_count = Σ y[j]` over those pooled channels.
- `bg_density = bg_count / bg_channel_count`,
  `bg_density_var = bg_count / bg_channel_count²` (variance of a mean).
- `n_region = number of channels in fit_region` (`x_fit.size`).
- `background_area = bg_density * n_region`.
- `background_area_err² = bg_density_var * n_region` — **linear, not
  quadratic, scaling in `n_region`** (strict error propagation for a
  scaled mean would use `n_region²`). This is a deliberate TV-parity
  choice (confirmed): TV's own `dBgSum = dBgF * (r-l)` is reproduced
  exactly, not "fixed" to the statistically correct form.

**Net**: `net_area = gross_area - background_area`, `net_area_err² =
gross_area_err² + background_area_err²` (independent variances add).

**Background and net moments** (`vsFitInt.c:223-309`): the same
mom1→mom2→mom3 moment machinery as the gross block, but now over
`(y[i] − bg_density)` for net and treating the background as the flat
level `bg_density` at every channel for its own moments. One quirk
confirmed for exact replication (not a fix): **the background's own
`gross_fwhm_err`/`skewness_err`-equivalent terms use the *net*
distribution's 2nd moment (`mom2`) rather than the background's own
(`bgMom2`)** — `vsFitInt.c:281,303`, `dDltb -= mom2` where `bgMom2` would
be locally consistent. This looks like a copy-paste slip in the original
C but is replicated exactly per explicit decision, alongside the
background-scaling quirk above.

All of gross/background/net guard against `sum == 0.0` (an empty or
exactly-zero-net region) the same way TV does (`if (sum != 0.0)` /
`if (bgSum != 0.0)`) — a zero-area layer's centroid/width/skewness and
their uncertainties are left at `0.0` rather than dividing by zero.

New `IntegrationResult` dataclass (sibling to `FitResult`/`PeakResult`,
not shoehorned into either): `left_bg_region`, `right_bg_region`,
`fit_region`, `timestamp: str = None`, `visible: bool = True`, and for
each of `gross_`/`background_`/`net_`: `area`, `area_err`, `centroid`,
`centroid_err`, `fwhm`, `fwhm_err`, `skewness`, `skewness_err` (flat
field naming, matching this project's existing flat dataclass style
rather than nesting).

## Storage

`LoadedSpectrum.fits` (attribute name kept as-is, not renamed) becomes a
mixed list of `FitResult` and `IntegrationResult` objects, appended by
`run_fit()`/`run_integration()` respectively.

## Results Table, Plot, and Export

**Table**: Integration rows share the existing 5-column Fit Results table
(`Fit / Peak / Position / FWHM / Volume`). `update_results_list()`
branches on `isinstance(result, IntegrationResult)`: one row per result
(no per-peak loop, since there's exactly one region), "Peak" column
shows `"region"`, Position/FWHM/Volume show `net_centroid`/`net_fwhm`/
`net_area` each `±` its `_err`, same formatting as today. The row's
tooltip (reusing the existing tail-info tooltip mechanism) lists the full
gross/background/net breakdown for area/centroid/width, plus skewness for
all three layers.

**Plot** (`draw_committed_fits`): for a visible `IntegrationResult`, draws
the same bg/fit region shading as a Gaussian fit, plus a horizontal line
at the flat `background_density` level across the fit region (in place of
the diagonal two-point background line Gaussian fits draw — Integration's
background is flat, not sloped), plus one annotation (in place of the
per-peak annotations) showing net area `±` err and net centroid `±` err.
No peak curve or decomposition lines (there is no fitted curve).

**Double-click reload**: for an `IntegrationResult`, restores `bg_regions`/
`fit_region` marks only — no peaks, no Fit Parameters panel state (there
are no free/fixed parameters for a direct sum).

**Export** (`fit_export.py`): new `integration_result_to_json_record`/
`integration_result_to_text_report`, dispatched by `isinstance` wherever
a result is serialized — `append_auto_log` (called from both
`run_fit()`/`run_integration()`), and the existing "Export This Fit..."/
"Export All Fits..." context-menu actions, which already iterate
`active.fits` and now need to type-dispatch per entry. No new UI actions
needed; the existing ones pick up Integration results automatically.

## Testing

- `peak_fit.py`: `integrate_region()` on a synthetic flat background +
  single Gaussian peak recovers a net area close to the peak's analytic
  area; a synthetic *uniform* background (no real peak) gives a
  background centroid equal to the fit region's mean channel index
  (sanity check on the flat-density moment math); a small (3-5 channel),
  hand-computable synthetic region where every gross/background/net
  moment and uncertainty is independently computed by hand and compared
  exactly (catches translation bugs in the dense moment machinery before
  they reach production); a region with zero net counts or zero gross
  counts doesn't raise or divide by zero (matches TV's `sum != 0.0`
  guards); the background-uncertainty linear-scaling quirk and the
  background's own moment-uncertainty terms using the net `mom2` are each
  covered by a dedicated test asserting the *exact* (quirky) formula, not
  the "corrected" one, so a future refactor can't silently "fix" them.
- `fit_mode.py`: `ready_to_integrate()` gating (enabled/disabled
  transitions independent of `ready_to_fit()`); `Ctrl+I` triggers
  `run_integration()`; marks persist after a successful integration; the
  Fit Results table shows a `"region"` row with net values and the full
  breakdown in its tooltip; the plot draws region shading + flat
  background line + annotation for a visible `IntegrationResult` and
  skips it when hidden (mirrors the existing
  `test_draw_committed_fits_skips_hidden_results` pattern); double-click
  reload restores marks only, no panel state.
- `fit_export.py`: JSON/text serialization of `IntegrationResult`
  round-trips all fields; the auto-log and both export actions correctly
  handle a spectrum whose `fits` list mixes `FitResult` and
  `IntegrationResult` entries, in order.

## Out of Scope

- Any UI mode-toggle/checkbox for Integration — a plain button/shortcut
  only, coexisting with "Fit", per explicit decision.
- TV's pre-fit-background-function branch (`FitBgTime` true in
  `vsFitInt.c`) — this app has no equivalent background-function concept
  for Integration; only the "marked regions only" branch is ported.
- Editing or fixing any integration quantity by hand — there are no
  free/fixed parameters for a direct sum (unlike the Fit Parameters
  panel's manual entry for Gaussian fits).
- Renaming `LoadedSpectrum.fits` — kept as-is per explicit decision, even
  though it now also holds `IntegrationResult` objects.
- Any change to the existing Gaussian fit path, its parameters, or its
  panel.
