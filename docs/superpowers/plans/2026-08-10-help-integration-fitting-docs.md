# Help Pages: Integration/Fitting Math + Export Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real formulas and two new figures to the Knowledge Database page (Sigma/FWHM, fitting position/volume/uncertainties, integration gross/background/net), and document the existing fit-export file formats in the HowTo page.

**Architecture:** Two new pure-function figures in `help_figures.py`; `help_content.py`'s `build_knowledge_database_html()` and `build_howto_html()` each get targeted section insertions/rewrites. No computation code changes anywhere — this is a documentation-only feature. Full formulas and exact HTML/figure code are pinned down in `docs/superpowers/specs/2026-08-10-help-integration-fitting-docs-design.md`; this plan's code blocks are the same content, already rendered and visually verified once during planning (see Task 1).

**Tech Stack:** Python, matplotlib (`Figure`/`FigureCanvasAgg`), PySide6 (unaffected), pytest.

---

### Task 1: `help_figures.py` — two new figure functions

**Files:**
- Modify: `help_figures.py`
- Test: `tests/test_help_figures.py`

- [ ] **Step 1: Write the failing tests**

Update the import block at the top of `tests/test_help_figures.py`:

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

Append these two tests at the end of `tests/test_help_figures.py`:

```python
def test_sigma_fwhm_figure_returns_valid_png():
    data = sigma_fwhm_figure()
    assert data.startswith(_PNG_SIGNATURE)
    assert len(data) > 20000  # measured ~42KB


def test_integration_background_figure_returns_valid_png():
    data = integration_background_figure()
    assert data.startswith(_PNG_SIGNATURE)
    assert len(data) > 20000  # measured ~42KB
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_help_figures.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'sigma_fwhm_figure'` (the two new names don't exist in `help_figures.py` yet).

- [ ] **Step 3: Add the two figure functions to `help_figures.py`**

Insert `sigma_fwhm_figure()` immediately before `def multiplet_figure():` (i.e. right after `tail_effect_figure()`'s closing `return _figure_to_png_bytes(fig)`):

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

Insert `integration_background_figure()` immediately before `def calibration_curve_figure():` (i.e. right after `multiplet_figure()`'s closing `return _figure_to_png_bytes(fig)`):

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

Both functions were already rendered and visually verified during planning (real output: `sigma_fwhm_figure` = 42114 bytes, `integration_background_figure` = 42001 bytes, both valid PNGs, both visually correct) — this is the final code, not a draft.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_help_figures.py -v`
Expected: PASS, all 6 tests (4 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add help_figures.py tests/test_help_figures.py
git commit -m "feat: add sigma/FWHM and integration-background help figures"
```

---

### Task 2: `help_content.py` — Knowledge Database formulas and figures

**Files:**
- Modify: `help_content.py`
- Test: `tests/test_help_content.py`

**Depends on Task 1** (imports the two new figure functions).

- [ ] **Step 1: Write the failing tests**

In `tests/test_help_content.py`, replace:

```python
def test_knowledge_database_html_embeds_four_figures():
    html = build_knowledge_database_html()
    assert html.count("data:image/png;base64,") == 4
```

with:

```python
def test_knowledge_database_html_embeds_six_figures():
    html = build_knowledge_database_html()
    assert html.count("data:image/png;base64,") == 6
```

Append these new tests to `tests/test_help_content.py`:

```python
def test_knowledge_database_html_explains_sigma_and_fwhm():
    html = _strip_base64_images(build_knowledge_database_html())
    assert "2.3548" in html
    assert "FWHM_err = 2.3548" in html


def test_knowledge_database_html_explains_position_volume_and_uncertainties():
    html = _strip_base64_images(build_knowledge_database_html())
    assert "area = amplitude &middot; &sigma; &middot; &radic;(2&pi;)" in html
    assert "area_err = |area|" in html
    assert "full_area = area + (background_slope" in html


def test_knowledge_database_html_explains_integration_moments():
    html = _strip_base64_images(build_knowledge_database_html())
    assert "How integration computes gross, background, and net" in html
    assert "centroid = &Sigma;(x&middot;y) / &Sigma;y" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_help_content.py -k knowledge_database -v`
Expected: `test_knowledge_database_html_embeds_six_figures` FAILs (still 4, not 6); the three new `explains_*` tests FAIL (text not present yet).

- [ ] **Step 3: Update `help_content.py`**

**3a.** Replace the import block:

```python
from help_figures import (
    anatomy_of_a_fit_figure,
    calibration_curve_figure,
    multiplet_figure,
    tail_effect_figure,
)
```

with:

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

**3b.** In `build_knowledge_database_html()`, replace:

```python
    anatomy_src = _embed_png(anatomy_of_a_fit_figure())
    tail_src = _embed_png(tail_effect_figure())
    multiplet_src = _embed_png(multiplet_figure())
    calibration_src = _embed_png(calibration_curve_figure())
```

with:

```python
    anatomy_src = _embed_png(anatomy_of_a_fit_figure())
    tail_src = _embed_png(tail_effect_figure())
    sigma_fwhm_src = _embed_png(sigma_fwhm_figure())
    multiplet_src = _embed_png(multiplet_figure())
    integration_bg_src = _embed_png(integration_background_figure())
    calibration_src = _embed_png(calibration_curve_figure())
```

**3c.** Fix the stale in-table cross-reference to the multiplet figure (it becomes Figure 4, not 3) and add a pointer to the new section. Replace:

```html
<tr><td>FWHM (&sigma;)</td><td>The Gaussian core's width. Peaks in the
same multiplet share one FWHM by default -- see Figure 3 -- unless you
turn on the Fit Parameters panel's "Independent widths" option.</td></tr>
```

with:

```html
<tr><td>FWHM (&sigma;)</td><td>The Gaussian core's width. Peaks in the
same multiplet share one FWHM by default -- see Figure 4 -- unless you
turn on the Fit Parameters panel's "Independent widths" option. See
"Sigma and FWHM" below for exactly how FWHM relates to
&sigma;.</td></tr>
```

**3d.** Insert the new "Sigma and FWHM" section and renumber the Multiplets figure caption. Replace:

```html
<tr><td>background slope / intercept</td><td>The straight line fixed by
the two background regions (Figure 1).</td></tr>
</table>

<h2>Multiplets</h2>
<figure>
<img src="{multiplet_src}" alt="Multiplet fit">
<figcaption>Figure 3. Three peaks fit together; two of them overlap
```

with:

```html
<tr><td>background slope / intercept</td><td>The straight line fixed by
the two background regions (Figure 1).</td></tr>
</table>

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

<h2>Multiplets</h2>
<figure>
<img src="{multiplet_src}" alt="Multiplet fit">
<figcaption>Figure 4. Three peaks fit together; two of them overlap
```

**3e.** Replace the entire "Volume: full vs. net" section:

```html
<h2>Volume: full vs. net</h2>
<p>The Fit Results panel's Volume column shows one number per peak: its
<b>net</b> volume -- the analytic integral of just that peak's Gaussian
core (background excluded). If the fit has a tail (tail fraction r > 0),
the tail's own contribution isn't included in this number -- a known
simplification, flagged directly in the panel's own tooltip as "volume
excludes tail." Hover over a row for the full breakdown: that same
peak's <b>full</b> volume (net volume plus the background level at the
peak's own center, times its FWHM), plus region-level full/net totals
for the whole fit -- the region's full total is the sum of the raw,
observed counts across the whole fit region (not the fitted model
curve), and its net total sums just the peaks' own net volumes. All of
these come with propagated uncertainties. Net volume is almost always
the number you actually want (for example, when computing activity or
a branching ratio).</p>
```

with:

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

**3f.** Insert the new "How integration computes gross, background, and net" section and renumber the Calibration figure caption. Replace:

```html
<h2>Integration vs. fitting</h2>
<p><kbd>Ctrl+I</kbd> (Integrate) computes a full/background/net split
directly from the data in the marked regions -- centroid, FWHM,
skewness, and area for each -- without fitting a peak shape at all. It
still reports into the same Volume column as a committed fit, but its
own hover tooltip labels the split "Gross"/"Background"/"Net" rather
than "full"/"net" -- the same background-included-vs-excluded idea as
above, just worded differently between the two features. Integration is
faster, doesn't depend on an optimizer converging, and works on peaks
too irregular or blended to fit cleanly. The trade-off is that it can't
separate overlapping peaks the way a multiplet fit can, and it reports
one combined result for the whole region rather than per-peak
parameters.</p>

<h2>Calibration</h2>
<figure>
<img src="{calibration_src}" alt="Calibration curve">
<figcaption>Figure 4. Linear vs. quadratic calibration through the same
```

with:

```html
<h2>Integration vs. fitting</h2>
<p><kbd>Ctrl+I</kbd> (Integrate) computes a full/background/net split
directly from the data in the marked regions -- centroid, FWHM,
skewness, and area for each -- without fitting a peak shape at all. It
still reports into the same Volume column as a committed fit, but its
own hover tooltip labels the split "Gross"/"Background"/"Net" rather
than "full"/"net" -- the same background-included-vs-excluded idea as
above, just worded differently between the two features. Integration is
faster, doesn't depend on an optimizer converging, and works on peaks
too irregular or blended to fit cleanly. The trade-off is that it can't
separate overlapping peaks the way a multiplet fit can, and it reports
one combined result for the whole region rather than per-peak
parameters.</p>

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

<h2>Calibration</h2>
<figure>
<img src="{calibration_src}" alt="Calibration curve">
<figcaption>Figure 6. Linear vs. quadratic calibration through the same
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_help_content.py -v`
Expected: PASS, all tests including the 4 modified/new ones. In particular confirm `test_knowledge_database_html_contains_parameter_names`, `test_knowledge_database_html_has_balanced_tags`, and `test_knowledge_database_html_has_no_external_links` still pass unchanged (nothing existing broke).

- [ ] **Step 5: Commit**

```bash
git add help_content.py tests/test_help_content.py
git commit -m "docs: add sigma/FWHM, position/volume, and integration-moment formulas to Knowledge Database"
```

---

### Task 3: `help_content.py` — HowTo export/save-format documentation

**Files:**
- Modify: `help_content.py`
- Test: `tests/test_help_content.py`

Independent of Task 2 (different function, `build_howto_html()`), but sequenced after it to keep `help_content.py` edits in one continuous pass.

- [ ] **Step 1: Write the failing tests**

In `tests/test_help_content.py`, in `test_howto_html_covers_every_operation()`, add `"Saving and exporting"` to the topic list:

```python
def test_howto_html_covers_every_operation():
    html = build_howto_html()
    for topic in [
        "Loading a spectrum",
        "Working with multiple spectra",
        "Saving a spectrum",
        "Calibrating the energy axis",
        "Multiply",
        "Rebin",
        "Normalize",
        "Add and Subtract Spectra",
        "Performing a fit",
        "Integration",
        "Saving and exporting",
        "View options",
        "Knowledge Database",
    ]:
        assert topic in html, f"missing topic {topic!r}"
```

Append this new test:

```python
def test_howto_html_documents_export_formats():
    html = build_howto_html()
    for term in ["_fits.jsonl", "Export All Fits", "_fits_report.txt", "Export This Fit"]:
        assert term in html, f"missing term {term!r}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_help_content.py -k howto -v`
Expected: `test_howto_html_covers_every_operation` FAILs (missing "Saving and exporting"); `test_howto_html_documents_export_formats` FAILs (none of the terms present yet).

- [ ] **Step 3: Update `build_howto_html()`**

Replace:

```html
<h3>8. Integration</h3>
<p><kbd>Ctrl+I</kbd> computes gross/background/net counts across the
marked regions directly (background centroid, FWHM, skewness, and area,
all with uncertainties) <i>without</i> fitting a peak shape -- faster,
and useful when a peak is too irregular to fit well, or when you only
need a total count rather than individual peak parameters.
Background regions are optional: marking only a fit region and
pressing <kbd>Ctrl+I</kbd> reports the raw area and centroid (plus
FWHM and skewness) with no background subtraction; marking two
background regions first still works exactly as before.</p>

<h3>9. View options</h3>
```

with:

```html
<h3>8. Integration</h3>
<p><kbd>Ctrl+I</kbd> computes gross/background/net counts across the
marked regions directly (background centroid, FWHM, skewness, and area,
all with uncertainties) <i>without</i> fitting a peak shape -- faster,
and useful when a peak is too irregular to fit well, or when you only
need a total count rather than individual peak parameters.
Background regions are optional: marking only a fit region and
pressing <kbd>Ctrl+I</kbd> reports the raw area and centroid (plus
FWHM and skewness) with no background subtraction; marking two
background regions first still works exactly as before.</p>

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_help_content.py -v`
Expected: PASS, full file. Also run `pytest tests/test_help_figures.py tests/test_help_content.py tests/test_help_menu.py -v` to confirm nothing in the Help feature area regressed.

- [ ] **Step 5: Commit**

```bash
git add help_content.py tests/test_help_content.py
git commit -m "docs: document auto-log and export report formats in HowTo"
```

---

### Task 4: Windows verification

**Files:** None (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: PASS, full suite (558+ tests from before this plan, plus the 8 new/modified tests added across Tasks 1-3).

- [ ] **Step 2: Build the Windows package**

Run `packaging/windows/build.ps1` from a fresh or existing `.venv` (per [[project_installer_uac_blocked]], this step is reliable and non-interactive).

- [ ] **Step 3: Smoke-test the built exe**

Launch `dist/SpectraTools.exe` (or the equivalent onedir path `build.ps1` produces), confirm it stays alive a few seconds via `Get-Process`, kill by the specific PID launched. Per the standing environment limitation ([[project_installer_uac_blocked]]), full interactive click-through of the Help menu is not available via computer-use in this environment — this smoke test is the reliable ceiling. The two new figures and all new formula text were already visually verified during planning (see Task 1's Step 3 note and this plan's own creation) via direct PNG rendering, which substitutes for an in-app visual check to the extent this environment allows.

- [ ] **Step 4: Report**

Report pass/fail of Steps 1-3 plainly; do not claim interactive verification that didn't happen.

---

### Task 5: Linux verification

**Files:** None (verification only).

- [ ] **Step 1: Run the full test suite under WSL**

Using a real script file (not an inline `bash -c` one-liner, per [[reference_wsl_invocation_from_git_bash]]), run `pytest -q` inside the relevant WSL distro(s) used for this project's Linux builds.

- [ ] **Step 2: Build both packages**

Run `packaging/linux/build.sh` (RPM) and `packaging/linux/build_deb.sh` (DEB) per this project's existing convention.

- [ ] **Step 3: Smoke-test both packages**

Install/launch on the distros this project already uses for verification (AlmaLinux, Ubuntu), confirm the process stays alive under `timeout` (exit 124, not an early exit), clean stderr.

- [ ] **Step 4: Report**

Report pass/fail of Steps 1-3 plainly.

---

## Self-Review Notes

- **Spec coverage**: every requirement in the design spec has a task — Sigma/FWHM (Task 2, 3d), fitting position/volume/uncertainties (Task 2, 3e), integration gross/background/net with and without background (Task 2, 3f), export formats (Task 3). Figure renumbering (3→4, 4→6) and the one cross-reference fix found while writing this plan (the "see Figure 3" table row) are both in Task 2, 3c/3d.
- **Placeholder scan**: no TBD/TODO; the only "measure and adjust" language from the original spec draft was replaced with real measured values (42114 / 42001 bytes) after actually rendering both figures during planning.
- **Type/name consistency**: `sigma_fwhm_src`/`integration_bg_src` variable names match between the `_embed_png()` calls (3b) and their `{...}` references in the new HTML blocks (3d, 3f). Function names (`sigma_fwhm_figure`, `integration_background_figure`) match between Task 1's definitions, the import block (3a), and their call sites (3b).
