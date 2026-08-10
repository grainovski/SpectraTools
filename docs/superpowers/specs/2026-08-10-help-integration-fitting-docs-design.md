# Help Pages: Integration/Fitting Math + Export Docs — Design Spec

Date: 2026-08-10

## Purpose

The Knowledge Database page explains the fit model conceptually but never shows the actual formulas for how Integration computes its numbers, how a fit's position/volume uncertainties are derived, or how sigma relates to FWHM. The HowTo page never explains that fit results are saved anywhere beyond the Ctrl+E export. Three additions, all documentation-only (no computation code changes):

1. **How Integration computes gross/background/net** — the real moment formulas (centroid, sigma→FWHM, skewness, area), covering both the with-background and without-background cases.
2. **Position, volume, and uncertainties for fitting** — expand the existing "Volume: full vs. net" section with the actual position/area/uncertainty formulas.
3. **Sigma ↔ FWHM** — the exact conversion (`FWHM_FACTOR = 2.3548200450309493`), shared by both paths.
4. **HowTo: saving and exporting fit results** — document the two existing persistence mechanisms (automatic JSONL log, manual text-report export) and their file formats, which aren't documented anywhere today.

## Researched Before Designing: Current Behavior

Read directly from source, not assumed:

**`peak_fit.py`:**
- `FWHM_FACTOR = 2.3548200450309493  # 2*sqrt(2*ln(2))` (line 7) — used identically in both paths: `fwhm = FWHM_FACTOR * sigma` (`fit_peaks`, line 389) and `fwhm = sigma * FWHM_FACTOR` (`integrate_region`'s `_to_reported`, line 629).
- `fit_peaks()` (lines 268-452): position and sigma are direct nonlinear-least-squares parameters (in-house Levenberg-Marquardt, `_marquardt_fit`); `position_err`/`sigma_err` come from `perr = np.sqrt(np.diag(pcov))` (line 354) — the fit's covariance matrix diagonal. Net area: `area = amplitude * sigma * np.sqrt(2 * np.pi)` (line 395, Gaussian core only — tail excluded, an explicit documented approximation). `area_err = abs(area) * np.sqrt(rel_err_sq)` where `rel_err_sq = (amplitude_err/amplitude)**2 + (sigma_err/sigma)**2` (lines 396-401, quadrature sum of relative uncertainties, no covariance term). Full area: `background_under_peak = (slope * position + intercept) * fwhm; full_area = area + background_under_peak` (lines 411-412), `full_area_err = area_err` exactly (line 413 — the two-point background line is deterministic, carries no uncertainty of its own). Region totals: `gross_area = sum(y_fit)`, `gross_area_err = sqrt(gross_area)` (Poisson); `net_area = sum(peak areas)`, `net_area_err = sqrt(sum(area_err**2))` (lines 434-437, quadrature across peaks).
- `integrate_region()` (lines 484-659): direct-summation moment analysis, no peak-shape fit. Gross layer (always computed): `gross_area = sum(s)`, `gross_area_err = sqrt(sum(s))` (Poisson, lines 512-515); `M1 = sum(idx*s)/sum(s)` = centroid (line 520); `M2 = sum((idx-M1)**2 * s) / sum(s)` → `sigma = sqrt(M2)` → `fwhm = sigma * FWHM_FACTOR` (lines 524-527, 625-631); `M3` = third central moment = skewness (lines 529-534). Background layer (only when both bg regions given, lines 547-565): flat density pooled across both regions (`bg_density = bg_count/bg_chn`), scaled by fit-region width `n` to get `background_area`. Net layer: `net_sum = gross_sum - background_area`, `net_dsum = gross_dsum + background_area_var` (line 567-570, variances added — independence assumption), same M1/M2/M3 moment machinery applied to background-subtracted counts. **Not surfaced in the new docs**: several TV-ported uncertainty terms deliberately don't follow textbook error propagation (e.g. the background/net M2-M3 uncertainty terms reuse net's own M2 in places a naive implementation would use each layer's own — explicitly flagged in the source as "CONFIRMED TV QUIRK", lines 603, 617) — these are implementation trivia verified against `tv-1.9.13/` source in an earlier design, not something a user needs to operate the tool; the new docs describe the moment formulas at the level a spectroscopist needs, not a literal transcription of every TV-parity arithmetic choice.

**`fit_export.py`** — two independent persistence paths:
- **Automatic**: `append_auto_log()` (line 115), called from `fit_mode.py:1157` and `:1190` on every successful commit of a fit or integration (no user action). Appends one JSON object per line to `auto_log_path()`'s result (line 7-12): `<spectrum-dir>/<spectrum-stem>_fits.jsonl`. `fit_result_to_json_record()`/`integration_result_to_json_record()` cover every parameter and its uncertainty, plus `*_keV` fields when a calibration is passed.
- **Manual**: `fit_mode.py`'s `_export_fits()` (line 1031), reached via `Ctrl+E` (bound to `export_all_fits_action`, `fit_mode.py:705-708`, all committed fits) or the Fit Results panel's right-click "Export This Fit..." (`fit_mode.py:998`, one fit). Opens `QFileDialog.getSaveFileName` defaulting to `<stem>_fits_report.txt` (all) or `<stem>_fit<N>_report.txt` (one, 1-based index, `fit_mode.py:1036-1040`), writes via `write_text_report()` → `fit_result_to_text_report()`/`integration_result_to_text_report()` (lines 153-259): a human-readable block per fit, every parameter with its `±` uncertainty and optional keV parenthetical.

**`help_content.py`/`help_figures.py`** current structure: `build_knowledge_database_html()` has sections "Why a tailed Gaussian?" → "The fit shape" (Fig 1, anatomy) → "What the tail parameters mean" (Fig 2, tail effect) → "Multiplets" (Fig 3, multiplet) → "Volume: full vs. net" (no figure, prose only) → "Integration vs. fitting" (no figure, conceptual only) → "Calibration" (Fig 4). `build_howto_html()` step-by-step guide runs 1-9 ("Loading a spectrum" through "View options"); step 8 is "Integration", step 7 is "Performing a fit" and already ends with "See the Knowledge Database page for exactly what's being computed here" — no mention anywhere of the auto-log or what the exported report contains. Figures are matplotlib `Figure` objects rendered to PNG bytes via `_figure_to_png_bytes()` and embedded as base64 data URIs via `help_content._embed_png()`. `tests/test_help_content.py` asserts exact substrings (not loose lowercased checks, per this project's own established discipline) and `html.count("data:image/png;base64,") == 4`; `tests/test_help_figures.py` asserts each figure function returns bytes starting with the PNG signature and above a rough size floor.

## Resolved Decisions (confirmed with the user)

- **Page split**: Knowledge Database gets all new formulas and figures; HowTo stays procedural, gaining only the new export/save-formats section (approved this turn) — no formulas duplicated into HowTo.
- **Two new figures**: "Sigma and FWHM" (single annotated Gaussian) and "Integration: with vs. without background" (two-panel comparison) — both in `help_figures.py`, following the existing `<topic>_figure()` naming and `_figure_to_png_bytes()` convention.
- **No new figure for the fitting position/volume section** — Figure 1 already shows the FWHM span visually; the new prose there just attaches formulas to what's already pictured.
- **Uncertainty formulas documented at the level a user needs**, not a literal transcription of every TV-ported arithmetic quirk in `integrate_region()` (see "Researched" above).
- **HowTo gets a new step documenting both persistence paths** (automatic JSONL log, manual text-report export), not just a reference to "Ctrl+E exports" as today.

## Architecture

No new files. Two existing files change: `help_figures.py` (two new figure functions) and `help_content.py` (import list, `build_knowledge_database_html()`, `build_howto_html()`).

### `help_figures.py` — two new figure functions

Insert `sigma_fwhm_figure()` after `tail_effect_figure()`:

```python
def sigma_fwhm_figure():
    """A single Gaussian core with sigma and the FWHM (full width at
    half maximum) both marked -- the reference figure for the "Sigma
    and FWHM" section, which every width this program reports (fitted
    or integrated) converts through."""
    x = np.linspace(70.0, 130.0, 400)
    position, sigma = 100.0, 8.0
    y = np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
    fwhm = sigma * FWHM_FACTOR

    fig = Figure(figsize=(7.0, 4.5), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(x, y, color=_DATA_COLOR, linewidth=1.8)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.annotate(
        "", xy=(position - fwhm / 2, 0.5), xytext=(position + fwhm / 2, 0.5),
        arrowprops=dict(arrowstyle="<->", color=_FIT_COLOR),
    )
    ax.text(position, 0.54, "FWHM", color=_FIT_COLOR, ha="center", fontsize=9)

    ax.annotate(
        "", xy=(position, 0.03), xytext=(position + sigma, 0.03),
        arrowprops=dict(arrowstyle="<->", color=_BG_COLOR),
    )
    ax.text(position + sigma / 2, 0.07, "sigma", color=_BG_COLOR, ha="center", fontsize=10)

    ax.text(
        0.5, -0.16,
        "FWHM = 2 * sqrt(2 * ln2) * sigma  ~=  2.3548 * sigma",
        transform=ax.transAxes, ha="center", va="top", fontsize=10,
    )
    ax.set_xlabel("Channel")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Sigma and FWHM")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return _figure_to_png_bytes(fig)
```

Insert `integration_background_figure()` after `multiplet_figure()`, before `calibration_curve_figure()`:

```python
def integration_background_figure():
    """The same simulated peak integrated two ways side by side: with
    two background regions marked (background line drawn, net shaded
    separately from gross) and with none (gross only, no subtraction)
    -- the reference figure for "How integration computes gross,
    background, and net"."""
    x = np.linspace(60.0, 140.0, 400)
    position, sigma, amplitude = 100.0, 4.0, 400.0
    bg_slope, bg_intercept = 0.05, 15.0
    background = bg_intercept + bg_slope * x
    gross = background + amplitude * np.exp(-((x - position) ** 2) / (2 * sigma ** 2))

    rng = np.random.default_rng(1)
    gross_noisy = rng.poisson(np.clip(gross, 1, None)).astype(float)

    fit_region = (80.0, 120.0)
    left_bg = (65.0, 75.0)
    right_bg = (125.0, 135.0)
    mask = (x >= fit_region[0]) & (x <= fit_region[1])

    fig = Figure(figsize=(9.5, 4.5), dpi=110)

    ax1 = fig.add_subplot(121)
    ax1.step(x, gross_noisy, where="mid", color=_DATA_COLOR, linewidth=1.0, label="Gross (data)")
    ax1.plot(x, background, color=_BG_COLOR, linestyle="--", linewidth=1.3, label="Background")
    ax1.fill_between(
        x[mask], background[mask], gross_noisy[mask], step="mid",
        color=_FIT_COLOR, alpha=0.25, label="Net",
    )
    ax1.axvspan(*left_bg, color=_REGION_BG_COLOR, alpha=0.25)
    ax1.axvspan(*right_bg, color=_REGION_BG_COLOR, alpha=0.25)
    ax1.axvspan(*fit_region, color=_REGION_FIT_COLOR, alpha=0.10)
    ax1.set_title("With background regions")
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("Counts")
    ax1.legend(loc="upper left", fontsize=7)

    ax2 = fig.add_subplot(122, sharey=ax1)
    ax2.step(x, gross_noisy, where="mid", color=_DATA_COLOR, linewidth=1.0, label="Gross (data)")
    ax2.fill_between(
        x[mask], 0, gross_noisy[mask], step="mid",
        color=_FIT_COLOR, alpha=0.25, label="Gross area",
    )
    ax2.axvspan(*fit_region, color=_REGION_FIT_COLOR, alpha=0.10)
    ax2.set_title("Without background regions")
    ax2.set_xlabel("Channel")
    ax2.legend(loc="upper left", fontsize=7)

    fig.suptitle("Integration: with vs. without background subtraction")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _figure_to_png_bytes(fig)
```

Both reuse the module's existing color constants (`_DATA_COLOR`, `_FIT_COLOR`, `_BG_COLOR`, `_REGION_BG_COLOR`, `_REGION_FIT_COLOR`) and `_figure_to_png_bytes()` — no new helpers needed.

### `help_content.py` — import list

```python
from help_figures import (
    anatomy_of_a_fit_figure,
    calibration_curve_figure,
    integration_background_figure,
    multiplet_figure,
    sigma_fwhm_figure,
    tail_effect_figure,
)
```

### `help_content.py` — `build_knowledge_database_html()`

Add two new `_embed_png()` calls, reordered to match the page's new figure order:

```python
    anatomy_src = _embed_png(anatomy_of_a_fit_figure())
    tail_src = _embed_png(tail_effect_figure())
    sigma_fwhm_src = _embed_png(sigma_fwhm_figure())
    multiplet_src = _embed_png(multiplet_figure())
    integration_bg_src = _embed_png(integration_background_figure())
    calibration_src = _embed_png(calibration_curve_figure())
```

**Insert new section** after the tail-parameters table (after the existing `</table>` that closes "What the tail parameters mean"), before `<h2>Multiplets</h2>`:

```html
<h2>Sigma and FWHM</h2>
<figure>
<img src="{sigma_fwhm_src}" alt="Sigma and FWHM">
<figcaption>Figure 3. The same Gaussian core, with &sigma; and the FWHM
(the width at half the peak's own maximum height) both marked -- FWHM
is always the wider of the two.</figcaption>
</figure>
<p>Every width this program reports -- FWHM, whether it came from a
fit or from Integration (see below) -- is derived from &sigma;, the
Gaussian core's own scale parameter, through one fixed conversion:</p>
<p style="text-align:center"><code>FWHM = 2&radic;(2&middot;ln2) &middot; &sigma; &asymp; 2.3548 &middot; &sigma;</code></p>
<p>The same constant (2.3548200450309493) applies everywhere in this
program a width is converted, and uncertainties convert the same way:
<code>FWHM_err = 2.3548 &middot; &sigma;_err</code>.</p>
```

Existing "Multiplets" section's figure caption changes from "Figure 3." to "Figure 4." (its content is otherwise unchanged).

**Replace** the entire existing "Volume: full vs. net" section with:

```html
<h2>Position, volume, and uncertainties (fitting)</h2>
<p>Position and &sigma; are fit directly by the optimizer, alongside
amplitude and (if enabled) the tail parameters. Their uncertainties --
<code>position_err</code>, <code>sigma_err</code> -- come straight from
the diagonal of the fit's covariance matrix, the standard uncertainty
estimate for a nonlinear least-squares fit. FWHM's own uncertainty
follows the same conversion as FWHM itself (see "Sigma and FWHM"
above): <code>fwhm_err = 2.3548 &middot; sigma_err</code>.</p>
<p>The Fit Results panel's Volume column shows one number per peak: its
<b>net</b> volume -- the analytic integral of just that peak's Gaussian
core (background excluded):</p>
<p style="text-align:center"><code>area = amplitude &middot; &sigma; &middot; &radic;(2&pi;)</code></p>
<p>with its own uncertainty propagated from amplitude's and &sigma;'s
(their correlation is not included -- a documented simplification):</p>
<p style="text-align:center"><code>area_err = |area| &middot; &radic;[(amplitude_err/amplitude)&sup2; + (&sigma;_err/&sigma;)&sup2;]</code></p>
<p>If the fit has a tail (tail fraction r &gt; 0), the tail's own
contribution isn't included in this number -- a known simplification,
flagged directly in the panel's own tooltip as "volume excludes tail."
Hover over a row for the full breakdown: that same peak's <b>full</b>
volume --</p>
<p style="text-align:center"><code>full_area = area + (background_slope &middot; position + background_intercept) &middot; FWHM</code></p>
<p>-- net volume plus the background level at the peak's own center
(the straight line fixed by the two background regions, Figure 1),
times its FWHM. That background line is a fixed two-point line rather
than a separately-fit quantity, so it carries no uncertainty of its
own: <code>full_area_err</code> equals <code>area_err</code> exactly.
Region-level full/net totals for the whole fit work the same way -- the
region's full total is the sum of the raw, observed counts across the
whole fit region (not the fitted model curve), and its net total sums
just the peaks' own net volumes, with their uncertainties combined in
quadrature. Net volume is almost always the number you actually want
(for example, when computing activity or a branching ratio).</p>
```

**Insert new section** after the existing "Integration vs. fitting" section's closing `</p>`, before `<h2>Calibration</h2>`:

```html
<h2>How integration computes gross, background, and net</h2>
<figure>
<img src="{integration_bg_src}" alt="Integration with and without background">
<figcaption>Figure 5. The same peak integrated two ways: with two
background regions marked (left -- background line drawn, net shaded
separately from gross) and with only a fit region marked (right --
gross only, no subtraction).</figcaption>
</figure>
<p>Integration works directly on the raw counts in the marked fit
region -- channel positions <code>x</code> and their counts
<code>y</code> -- with no peak shape fit at all. Area, centroid, and
width all come from the same three statistical moments of that data:</p>
<p style="text-align:center"><code>area = &Sigma;y</code> &nbsp;&nbsp;
<code>centroid = &Sigma;(x&middot;y) / &Sigma;y</code> &nbsp;&nbsp;
<code>&sigma;&sup2; = &Sigma;((x&minus;centroid)&sup2;&middot;y) / &Sigma;y</code></p>
<p>Centroid is the count-weighted mean channel (the first moment).
&sigma; comes from the second moment -- the spread of counts around the
centroid -- and converts to FWHM exactly as in "Sigma and FWHM" above.
Skewness (a third moment) measures asymmetry the same way. Every one of
these carries its own uncertainty, propagated from Poisson counting
statistics on each individual channel's count.</p>
<p>This <b>gross</b> layer -- the raw, un-subtracted data -- is always
computed, and is all <kbd>Ctrl+I</kbd> reports when no background
regions are marked. When two background regions are marked instead, a
flat count density is pooled across both of them (total background
counts divided by total background channels) and scaled by the fit
region's width to get the <b>background</b> layer's own area; the same
moment formulas, applied to that flat density, give its centroid,
FWHM, and skewness too. The <b>net</b> layer is gross minus background
at every step -- net area, and a centroid/FWHM/skewness computed from
the background-subtracted counts -- with the two layers' uncertainties
combined accordingly.</p>
```

Existing "Calibration" section's figure caption changes from "Figure 4." to "Figure 6." (its content is otherwise unchanged).

Final section order: Why tailed Gaussian → The fit shape (Fig 1) → What the tail parameters mean (Fig 2) → **Sigma and FWHM (Fig 3, new)** → Multiplets (Fig 4) → **Position, volume, and uncertainties (fitting) (expanded, no fig)** → Integration vs. fitting (unchanged, no fig) → **How integration computes gross, background, and net (Fig 5, new)** → Calibration (Fig 6).

### `help_content.py` — `build_howto_html()`

**Insert new step** after the existing step 8 ("Integration"), before step 9 ("View options"), renumbering the old step 9 to step 10:

```html
<h3>9. Saving and exporting fit results</h3>
<p>Every fit or integration you commit (<kbd>Ctrl+F</kbd> /
<kbd>Ctrl+I</kbd>) is automatically logged to a file next to the
spectrum: the spectrum's own filename with <code>_fits.jsonl</code>
appended (e.g. <code>eu.spe</code> logs to <code>eu_fits.jsonl</code>).
Each line is one self-contained JSON record covering every parameter
and its uncertainty, plus keV equivalents when calibration is active.
This happens automatically every time -- there's nothing to trigger and
nothing to configure.</p>
<p>To save a human-readable report instead, press <kbd>Ctrl+E</kbd>
(<b>Export All Fits...</b>): a save dialog opens, defaulting to the
spectrum's filename with <code>_fits_report.txt</code> appended, and
writes every committed fit and integration on the active spectrum as
one plain-text report. To export just one instead, right-click its row
in the Fit Results panel and choose <b>Export This Fit...</b> (defaults
to the filename with <code>_fit</code> and its number appended, e.g.
<code>_fit2_report.txt</code>). Either way, the report lists every
parameter with its uncertainty in plain text -- see the Knowledge
Database page for what each one means.</p>

<h3>10. View options</h3>
```

(The old `<h3>9. View options</h3>` heading text becomes `<h3>10. View options</h3>`; its body content is unchanged.)

## Error Handling Summary

Not applicable — this is a documentation-only change. No new user-facing behavior, no new failure modes. The two new figure functions follow the exact same shape (pure function, deterministic inputs via a fixed RNG seed, returns PNG bytes) as the four existing ones, so they carry the same (already-accepted) risk profile: none, since they're called at HTML-build time with no external inputs.

## Testing Plan

- **`tests/test_help_figures.py`**: add `test_sigma_fwhm_figure_returns_valid_png` and `test_integration_background_figure_returns_valid_png`, matching the existing pattern exactly (PNG signature check + a size floor) — the implementer should render each once locally to measure its real size before picking the floor, the same way the existing four tests' comments record real measured ranges.
- **`tests/test_help_content.py`**:
  - `test_knowledge_database_html_embeds_four_figures` → rename to `test_knowledge_database_html_embeds_six_figures`, assert `== 6`.
  - New `test_knowledge_database_html_explains_sigma_and_fwhm`: asserts the exact constant text and the `FWHM_err` formula substring are present.
  - New `test_knowledge_database_html_explains_position_volume_and_uncertainties`: asserts the `area = amplitude` formula substring, the `area_err` formula substring, and the `full_area` formula substring are present (three independent assertions, one per formula, so a partial revert of any single formula is caught — matching this project's established convention of exact, location-specific assertions rather than one loose check, per the mutation-testing lesson from the Ctrl+C feature).
  - New `test_knowledge_database_html_explains_integration_moments`: asserts the centroid/sigma-squared moment formula substrings and the "How integration computes gross, background, and net" heading text are present.
  - `test_howto_html_covers_every_operation`: add `"Saving and exporting"` to the topic list.
  - New `test_howto_html_documents_export_formats`: asserts `_fits.jsonl`, `Export All Fits`, `_fits_report.txt`, and `Export This Fit` are all present in the HowTo HTML.
  - Existing `test_howto_html_contains_every_shortcut`, `test_knowledge_database_html_contains_parameter_names`, and both tag-balance tests need no changes but must still pass — confirms nothing existing broke.
- **Manual**: open both generated pages (via the app's Help menu or `open_help_page`) and visually confirm the two new figures render sensibly and the formula text displays correctly (HTML entities resolve, no stray literal `&amp;`-escaped double-entities).

## Out of Scope

- No changes to `peak_fit.py`/`fit_export.py`/`fit_mode.py` computation or export code — purely documentation.
- No LaTeX/MathML rendering — formulas stay as inline `<code>` + HTML entities, matching the existing Hypermet-formula convention on this same page.
- No new export format or save feature — the HowTo addition documents what already exists (the automatic JSONL log and the Ctrl+E text report), nothing new is being built.
- No itemization of `integrate_region()`'s TV-ported uncertainty-propagation quirks (see "Researched" above) in user-facing text — those are implementation trivia already verified against source in an earlier design, not something a user needs.
- No third new figure for the fitting position/volume section — reuses the existing Figure 1.
