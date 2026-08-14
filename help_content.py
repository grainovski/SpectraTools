"""Builds the three Help-menu pages (HowTo, Knowledge Database, About) as
self-contained HTML strings, and opens a generated page in the system's
default web browser. No external files, no external links -- Knowledge
Database's figures (help_figures.py) are embedded as base64 data URIs, so
a generated page has zero dependencies once written to disk."""

import base64
import functools
import os
import shutil
import subprocess
import sys
import tempfile

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from help_figures import (
    anatomy_of_a_fit_figure,
    calibration_curve_figure,
    integration_background_figure,
    matrix_projection_cut_figure,
    multiplet_figure,
    sigma_fwhm_figure,
    tail_effect_figure,
)


def _embed_png(png_bytes):
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_PAGE_CSS = """
body {
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    max-width: 900px;
    margin: 2rem auto;
    padding: 0 1.5rem;
    line-height: 1.5;
    color: #1a1a1a;
}
h1 { border-bottom: 2px solid #ccc; padding-bottom: 0.3rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #ddd; padding-bottom: 0.2rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; }
th { background: #f0f0f0; }
kbd {
    background: #eee;
    border: 1px solid #bbb;
    border-radius: 3px;
    padding: 0.1rem 0.4rem;
    font-family: monospace;
    font-size: 0.9em;
}
figure { margin: 1.5rem 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #ddd; }
figcaption { font-size: 0.9em; color: #555; margin-top: 0.4rem; }
"""


def _page(title, body_html):
    """Wraps body_html in the shared doctype/head/style shell every Help
    page uses. `title` and `body_html` are always this module's own
    literal/generated content, never user input -- not escaped."""
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
{body_html}
</body>
</html>"""


def build_howto_html():
    """Every keyboard shortcut, grouped by menu, plus a short step-by-step
    guide for every operation -- content verified against main_window.py/
    fit_mode.py's actual behavior, not just the UI's stated labels (see
    commits c320353/2ce5fb2 for the factual corrections that came out of
    that verification)."""
    body = """
<h1>SpectraTools -- HowTo</h1>
<p>This page covers every keyboard shortcut in SpectraTools, grouped by
menu, followed by short step-by-step guides for every operation the
program supports.</p>

<h2>Keyboard shortcuts</h2>

<h3>File menu</h3>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>Ctrl+O</kbd></td><td>Open a spectrum file</td></tr>
<tr><td><kbd>Ctrl+Shift+O</kbd></td><td>Open Matrix...</td></tr>
<tr><td><kbd>Ctrl+S</kbd></td><td>Save Spectrum...</td></tr>
<tr><td><kbd>Ctrl+W</kbd></td><td>Close Spectrum (remove the active spectrum from the program)</td></tr>
<tr><td><kbd>Ctrl+Q</kbd></td><td>Exit</td></tr>
</table>

<h3>View menu</h3>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>Ctrl+G</kbd></td><td>Toggle log-scale Y axis</td></tr>
<tr><td><kbd>Ctrl+1</kbd></td><td>Toggle the Spectra panel</td></tr>
<tr><td><kbd>Ctrl+D</kbd></td><td>Toggle dark theme</td></tr>
</table>

<h3>Operations menu</h3>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>Ctrl+L</kbd></td><td>Calibration...</td></tr>
<tr><td><kbd>Ctrl+T</kbd></td><td>Toggle Calibration Active</td></tr>
<tr><td><kbd>Ctrl+M</kbd></td><td>Multiply by Factor...</td></tr>
<tr><td><kbd>Ctrl+R</kbd></td><td>Rebin by Factor...</td></tr>
<tr><td><kbd>Ctrl+N</kbd></td><td>Normalize Spectra</td></tr>
<tr><td><kbd>Ctrl+A</kbd></td><td>Add Spectra...</td></tr>
<tr><td><kbd>Ctrl+Shift+A</kbd></td><td>Subtract Spectra...</td></tr>
</table>

<h3>Plot toolbar</h3>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>Ctrl+=</kbd></td><td>Zoom in (X axis)</td></tr>
<tr><td><kbd>Ctrl+-</kbd></td><td>Zoom out (X axis)</td></tr>
<tr><td><kbd>Ctrl+0</kbd></td><td>Show full spectrum</td></tr>
</table>

<h3>Fitting</h3>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>B</kbd></td><td>Hold and click twice per region (two regions needed, four clicks total) to mark the two background regions</td></tr>
<tr><td><kbd>R</kbd></td><td>Hold and click twice (once on each end) to mark the fit region</td></tr>
<tr><td><kbd>P</kbd></td><td>Hold and click once per peak to mark a peak position</td></tr>
<tr><td><kbd>Ctrl+F</kbd></td><td>Fit</td></tr>
<tr><td><kbd>Ctrl+B</kbd></td><td>Preview the background fit from the two background regions alone (no fit region or peaks needed); press again to hide</td></tr>
<tr><td><kbd>Ctrl+2</kbd></td><td>Toggle the Fit Results panel</td></tr>
<tr><td><kbd>Ctrl+3</kbd></td><td>Toggle the Fit Parameters panel</td></tr>
<tr><td><kbd>Ctrl+C</kbd></td><td>Clear in-progress marks and hide committed fits (not delete)</td></tr>
<tr><td><kbd>Ctrl+Shift+C</kbd></td><td>Permanently delete the active spectrum's committed fits (in-progress marks untouched)</td></tr>
<tr><td><kbd>Ctrl+E</kbd></td><td>Export the active spectrum's fits</td></tr>
<tr><td><kbd>Ctrl+I</kbd></td><td>Integrate</td></tr>
</table>

<h3>Matrix panel</h3>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>C</kbd></td><td>Hold and click twice to mark the cut (signal) region</td></tr>
<tr><td><kbd>G</kbd></td><td>Hold and click twice per region to mark a background region for the cut -- any number of regions allowed</td></tr>
<tr><td><kbd>Ctrl+Alt+C</kbd></td><td>Activate Cut (same as clicking the button; needs a cut region marked first)</td></tr>
<tr><td><kbd>B</kbd></td><td>Hold and click twice per region (two regions needed, four clicks total) to mark the two background regions for fitting the working projection</td></tr>
<tr><td><kbd>R</kbd></td><td>Hold and click twice (once on each end) to mark the fit region for the working projection</td></tr>
<tr><td><kbd>P</kbd></td><td>Hold and click once per peak to mark a peak position on the working projection</td></tr>
<tr><td><kbd>Ctrl+F</kbd></td><td>Fit</td></tr>
<tr><td><kbd>Ctrl+I</kbd></td><td>Integrate</td></tr>
<tr><td><kbd>Ctrl+B</kbd></td><td>Preview the background fit from the two background regions alone (no fit region or peaks needed); press again to hide</td></tr>
<tr><td><kbd>Ctrl+C</kbd></td><td>Clear in-progress marks and hide committed fits (not delete)</td></tr>
<tr><td><kbd>Ctrl+Shift+C</kbd></td><td>Permanently delete the working projection's committed fits (in-progress marks untouched)</td></tr>
<tr><td><kbd>Ctrl+E</kbd></td><td>Export the working projection's fits</td></tr>
<tr><td><kbd>Ctrl+2</kbd></td><td>Toggle the Fit Results panel</td></tr>
<tr><td><kbd>Ctrl+3</kbd></td><td>Toggle the Fit Parameters panel</td></tr>
<tr><td><kbd>Ctrl+L</kbd></td><td>Calibrate... (same calibration as the main window's Operations &gt; Calibration... -- see "11. Matrix analysis" below)</td></tr>
<tr><td><kbd>Ctrl+=</kbd></td><td>Zoom in (X axis)</td></tr>
<tr><td><kbd>Ctrl+-</kbd></td><td>Zoom out (X axis)</td></tr>
<tr><td><kbd>Ctrl+0</kbd></td><td>Show full projection</td></tr>
</table>

<h3>Help menu</h3>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>F1</kbd></td><td>HowTo (this page)</td></tr>
<tr><td>&mdash;</td><td>Knowledge Database (no shortcut) -- how fits actually work, parameter meanings, calibration math</td></tr>
<tr><td>&mdash;</td><td>About (no shortcut) -- version and build date</td></tr>
</table>

<h2>Step-by-step guides</h2>

<h3>1. Loading a spectrum</h3>
<p><b>File &gt; Open...</b> (<kbd>Ctrl+O</kbd>) opens a file picker for the
four supported formats: <b>.txt</b> (one integer count per line, channel
number implied by line position), <b>.spe</b>, <b>.spk</b>, and
<b>.n42</b> (ANSI/IEEE N42.42; only the raw histogram and, if present,
the embedded energy calibration are read -- everything else in the file
is ignored, and an N42 file's calibration is applied automatically only
when no calibration is already active). Recently opened files also
appear under <b>File &gt; Recent Files</b> for one-click reopening.
Every spectrum you open stays loaded until you close it -- opening a
new one adds it alongside the others rather than replacing what's
already there.</p>

<h3>2. Working with multiple spectra</h3>
<p>Loaded spectra appear in the Spectra panel, one row each. Exactly one
spectrum is <b>active</b> at a time (selected via its radio button) --
every operation (Calibration, Multiply, Rebin, fitting, and so on) acts
on whichever spectrum is currently active. To remove a spectrum from the
program (the file on disk is never touched): right-click its row and
choose <b>Remove</b>, or make it active and press <kbd>Ctrl+W</kbd>
(<b>File &gt; Close Spectrum</b>), which always acts on the active
spectrum without needing to right-click a specific row.</p>

<h3>3. Saving a spectrum</h3>
<p><b>File &gt; Save Spectrum...</b> (<kbd>Ctrl+S</kbd>) saves the active
spectrum, prompting you to choose one of three formats (.txt/.spe/.spk);
<b>.n42</b> is read-only and isn't offered here.</p>

<h3>4. Calibrating the energy axis</h3>
<p><b>Operations &gt; Calibration...</b> (<kbd>Ctrl+L</kbd>) opens a
dialog to enter or load a linear (<i>E = a + b&middot;channel</i>) or
quadratic (<i>E = a + b&middot;channel + c&middot;channel<sup>2</sup></i>)
calibration. Once set, <kbd>Ctrl+T</kbd> (or the toolbar button) toggles
whether it's actually applied -- with it active, the plot's X axis and
every fit parameter that has units (position, FWHM) switch from raw
channels to keV. Calibration is shared across every loaded spectrum.</p>

<h3>5. Multiply, Rebin, and Normalize</h3>
<p><b>Multiply by Factor...</b> (<kbd>Ctrl+M</kbd>) scales the active
spectrum's counts by a factor you enter. <b>Rebin by Factor...</b>
(<kbd>Ctrl+R</kbd>) combines that many adjacent channels into one,
reducing the active spectrum's channel count -- note this only rescales
that one spectrum's calibration, so other loaded spectra can end up on a
different channel scale than the one you just rebinned. <b>Normalize
Spectra</b> (<kbd>Ctrl+N</kbd>) needs a reference point marked first --
hold <kbd>R</kbd> and click once for a single channel, or twice for a
region -- then scales every <i>visible</i> spectrum (at least two must
be visible) so they all read the same value there, useful for visually
comparing spectra taken with different live times.</p>

<h3>6. Add and Subtract Spectra</h3>
<p><b>Add Spectra...</b> (<kbd>Ctrl+A</kbd>) and <b>Subtract
Spectra...</b> (<kbd>Ctrl+Shift+A</kbd>) each open a dialog to pick two
loaded spectra, Spectrum A and Spectrum B, plus a factor (defaulting to
1). The result is a new spectrum -- Spectrum A, plus or minus Spectrum B
scaled by the factor -- added alongside the originals, which are left
untouched. Both spectra must have the same number of channels; picking a
mismatched pair shows an error naming both channel counts instead of
proceeding. Subtracting can produce negative channel counts in the
result -- this is expected, not an error.</p>

<h3>7. Performing a fit</h3>
<p>This is the core workflow, and it's entirely mouse-plus-keyboard on
the plot itself:</p>
<ol>
<li>Hold <kbd>B</kbd> and click twice on one side of the peak(s), then
twice on the other side -- four clicks total, marking the two
background regions used to compute a linear background under the
peak. A third completed pair evicts the oldest region, so only the
two most recent stick.</li>
<li>Hold <kbd>R</kbd> and click twice (once on each end) to mark the
fit region (the span that gets fit).</li>
<li>Hold <kbd>P</kbd> and click once per peak, at each peak's
approximate position -- mark more than one for a multiplet (peaks that
share one FWHM by default; see the Knowledge Database page for the
"Independent widths" option).</li>
<li>Press <kbd>Ctrl+F</kbd> to fit. Results appear in the Fit Results
panel; per-parameter values (with uncertainties) appear in the Fit
Parameters panel.</li>
</ol>
<p>Once both background regions are marked, <kbd>Ctrl+B</kbd> previews
just the background line -- no fit region or peaks needed -- useful
for sanity-checking the background before marking the rest. It's a
preview only: nothing is added to Fit Results, logged, or exported.
Press <kbd>Ctrl+B</kbd> again to hide it.</p>
<p>See the Knowledge Database page for exactly what's being computed
here, and what each fit parameter means.</p>
<p><kbd>Ctrl+C</kbd> clears the active spectrum's in-progress B/R/P
marks and hides its already-committed fits (grayed out in Fit
Results, removed from the plot, but not deleted).
<kbd>Ctrl+Shift+C</kbd> permanently deletes the active spectrum's
committed fits, leaving any in-progress marks alone. <kbd>Ctrl+E</kbd>
exports the active spectrum's committed fits.</p>

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
<p><kbd>Ctrl+G</kbd> toggles a logarithmic Y axis. <kbd>Ctrl+D</kbd>
toggles dark theme. <kbd>Ctrl+1</kbd>/<kbd>Ctrl+2</kbd>/<kbd>Ctrl+3</kbd>
show or hide the Spectra, Fit Results, and Fit Parameters panels.
<kbd>Ctrl+=</kbd>/<kbd>Ctrl+-</kbd> zoom the X axis in/out around the
current view, and <kbd>Ctrl+0</kbd> resets to the full spectrum.</p>

<h3>11. Matrix analysis</h3>
<p><b>File &gt; Open Matrix...</b> (<kbd>Ctrl+Shift+O</kbd>) opens a
2D coincidence matrix (<b>.mtx</b>) in its own window. Only the raw
histogram is read -- there's no way to save a matrix back out. The
matrix panel computes both its X and Y projections up front; pick
which one to work on from the dropdown. Hold <kbd>C</kbd> and click
twice to mark the cut (signal) region, and hold <kbd>G</kbd> and click
twice for each background region -- any number of background regions
are allowed, and more background generally means better statistics.
The <b>Clear Marks</b> button resets the cut region and every
background region mark together, so you can start over without
switching projections -- fit marks, committed fits, and any
already-activated cut are all left untouched. <b>Activate Cut</b>
(<kbd>Ctrl+Alt+C</kbd>) computes a background-subtracted
spectrum (weighted automatically by region width, same convention TV uses) and
adds it to the main window like any other loaded spectrum -- you can
fit, calibrate, or export it exactly the same way. Negative counts can
appear in the result and are expected, not an error. <b>Show
Heatmap...</b> opens a separate, view-only 2D intensity map of the
matrix with its own zoom/pan controls -- purely for visual reference;
marking and cutting always happens on the projection, not the
heatmap.</p>
<p>The working projection can also be fit or integrated directly, with
no need to activate a cut first -- the same <kbd>B</kbd>/<kbd>R</kbd>/<kbd>P</kbd>
marking and <kbd>Ctrl+F</kbd>/<kbd>Ctrl+I</kbd>/<kbd>Ctrl+B</kbd>/<kbd>Ctrl+C</kbd>
shortcuts described in "7. Performing a fit" and "8. Integration"
above, run by the exact same fitting engine as the main window's, just
pointed at the projection instead of a loaded spectrum. A Fit Results
and Fit Parameters panel pair -- including the "Independent
widths"/"Left tail" options and their own
<kbd>Ctrl+Shift+C</kbd>/<kbd>Ctrl+E</kbd>/<kbd>Ctrl+2</kbd>/<kbd>Ctrl+3</kbd>
shortcuts -- appears in this window too, working exactly like the main
window's. Cut/background marks (<kbd>C</kbd>/<kbd>G</kbd>) and fit
marks (<kbd>B</kbd>/<kbd>R</kbd>/<kbd>P</kbd>) don't interfere with
each other -- both can be in progress at once.</p>
<p><kbd>Ctrl+L</kbd> opens the same Calibrate dialog as the main
window's own Calibration..., because it <i>is</i> the main window's
calibration -- there's only one calibration per session, not a
separate one per window. Calibrating from a matrix panel updates the
main window's plot (and any other open matrix panel) immediately, and
calibrating from the main window updates every open matrix panel the
same way. <kbd>Ctrl+=</kbd>/<kbd>Ctrl+-</kbd>/<kbd>Ctrl+0</kbd> (or the
scroll wheel) zoom the projection's X axis, matching the main window's
own zoom.</p>
<p>Switching the dropdown between X and Y projection clears any
in-progress cut/background marks, and discards any in-progress or
committed fit marks for the projection you're leaving -- it's
different underlying data, so nothing carries over, even switching
back to an axis you'd already marked or fit before.</p>
"""
    return _page("SpectraTools -- HowTo", body)


@functools.lru_cache(maxsize=1)
def build_knowledge_database_html():
    """The fit model, parameter meanings, and calibration math, each
    cross-referenced to one of help_figures.py's seven annotated figures
    -- content verified against peak_fit.py/fit_mode.py's actual
    behavior through two rounds of correction after the first draft
    shipped wrong claims about area terminology and multiplet parameter
    sharing (see commits f9c73cd and ce57765)."""
    anatomy_src = _embed_png(anatomy_of_a_fit_figure())
    tail_src = _embed_png(tail_effect_figure())
    sigma_fwhm_src = _embed_png(sigma_fwhm_figure())
    multiplet_src = _embed_png(multiplet_figure())
    integration_bg_src = _embed_png(integration_background_figure())
    calibration_src = _embed_png(calibration_curve_figure())
    matrix_src = _embed_png(matrix_projection_cut_figure())

    body = f"""
<h1>SpectraTools -- Knowledge Database</h1>
<p>This page explains how SpectraTools actually performs a fit, what
each fit parameter means, and how calibration and integration work
underneath the HowTo page's step-by-step instructions.</p>

<h2>Why a tailed Gaussian?</h2>
<p>A germanium or scintillator detector doesn't record a perfectly sharp
line at a gamma ray's true energy -- charge-collection losses and
incomplete charge trapping skew a fraction of events to slightly lower
apparent energy. The result is a peak that's very close to Gaussian
near its center, but with a low-energy shoulder a pure Gaussian can't
reproduce. SpectraTools fits this shape directly rather than ignoring
the shoulder or fitting it as a separate background component.</p>

<h2>The fit shape</h2>
<figure>
<img src="{anatomy_src}" alt="Anatomy of a fit">
<figcaption>Figure 1. A single peak: the two background regions (B), the
fit region (R), the peak position (P), and the resulting fitted
curve.</figcaption>
</figure>
<p>Fitting proceeds in the same order you mark it in: the two background
regions fix a straight line (slope and intercept) under the peak; the
fit region defines the span actually fit; each peak mark seeds one
peak's starting position. The peak shape itself, following the
<b>Hypermet</b> function this app ports from the <code>gf3</code>
peak-fitting tool, is:</p>
<p style="text-align:center"><code>(1&minus;r)&middot;exp(&minus;w&sup2;) + r&middot;exp(dx/&beta;)&middot;erfc(w+y)/erfc(y)</code></p>
<p>where <code>dx = channel &minus; position</code>,
<code>w = dx / (&sigma;&radic;2)</code>, and
<code>y = &sigma; / (&beta;&radic;2)</code> -- a Gaussian core
(the <code>(1&minus;r)&middot;exp(&minus;w&sup2;)</code> term) blended
with an exponential tail on the low-energy side (the
<code>r&middot;...</code> term).</p>

<h2>What the tail parameters mean</h2>
<figure>
<img src="{tail_src}" alt="The tail effect">
<figcaption>Figure 2. The same peak with tail fraction r = 0 (pure
Gaussian) vs. r = 0.25 (a clearly visible low-energy tail).</figcaption>
</figure>
<table>
<tr><th>Parameter</th><th>Meaning</th></tr>
<tr><td>position</td><td>The peak's centroid channel (or keV, with
calibration active).</td></tr>
<tr><td>FWHM (&sigma;)</td><td>The Gaussian core's width. Peaks in the
same multiplet share one FWHM by default -- see Figure 4 -- unless you
turn on the Fit Parameters panel's "Independent widths" option. See
"Sigma and FWHM" below for exactly how FWHM relates to
&sigma;.</td></tr>
<tr><td>amplitude</td><td>The peak's height above background.</td></tr>
<tr><td>tail fraction (r)</td><td>What fraction of the peak's area sits
in the tail rather than the Gaussian core. r = 0 is a pure
Gaussian.</td></tr>
<tr><td>tail beta (&beta;)</td><td>How far the tail extends below the
peak position -- a larger &beta; stretches the tail further to lower
energy.</td></tr>
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
closely enough that fitting each independently wouldn't separate them
reliably, so all three share one FWHM (the default) as well as one
tail fraction and tail beta (never optional).</figcaption>
</figure>
<p>Mark more than one peak (<kbd>P</kbd>) within the same fit region and
SpectraTools fits them together as a multiplet: every peak gets its own
position and amplitude, and by default they all share one FWHM too --
turn on the Fit Parameters panel's "Independent widths" option to fit
each peak's width separately instead. Tail fraction and tail beta are
always shared across the multiplet; there's no equivalent option for
those. Sharing width (by default) is what makes it possible to separate
overlapping peaks that a single-peak fit couldn't
resolve.</p>

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
points, with residuals -- the systematic curve in the linear residuals
is exactly what a quadratic term corrects for.</figcaption>
</figure>
<p>Calibration converts channel numbers to energy: linear
(<i>E = a + b&middot;channel</i>) or quadratic
(<i>E = a + b&middot;channel + c&middot;channel<sup>2</sup></i>). Once
active, every displayed position and FWHM converts through it
automatically -- FWHM (a width, not a position) scales by the
calibration's local derivative evaluated at the peak's own position,
since a width has no location on the calibration curve of its own.
Uncertainties propagate through the same derivative.</p>

<h2>2D matrices, projections, and cuts</h2>
<figure>
<img src="{matrix_src}" alt="2D matrix, cut, and projection">
<figcaption>Figure 7. A schematic 2D coincidence matrix with a diagonal
ridge of correlated counts (left) and its full X projection (right),
with a cut region (orange) and background region (purple) marked on
the X axis of both. Activating the cut sums only the gated rows to
build a new spectrum on the <i>other</i> axis -- not shown
here.</figcaption>
</figure>
<p>A <b>2D coincidence matrix</b> (<b>File &gt; Open Matrix...</b>, see
the HowTo page's "11. Matrix analysis") records pairs of gamma rays
detected close together in time -- typically one in each of two
detectors watching the same source. Each coincident pair increments one
cell of the matrix; X and Y are the two detectors' own channel axes, so
a cell's (X, Y) position records which channel each detector saw for that
event. A cascade of two genuinely correlated gamma rays -- the same two
energies, detected together, over and over across many decays -- builds
up as a streak of counts at a fixed (X, Y), the diagonal ridge shown in
Figure 7. Uncorrelated counts, from unrelated gamma rays that merely
arrived close together by chance, spread out across the matrix
instead.</p>
<p>A <b>projection</b> collapses the matrix into an ordinary 1D
spectrum by summing counts along one axis: the X projection sums each
column over its full Y range, and the Y projection sums each row over
its full X range. The matrix panel computes both projections up front
when a matrix is opened, and lets you pick which one to work on.</p>
<p>A <b>cut</b> -- called a <b>gate</b> in the wider gamma-gamma
coincidence literature -- restricts that sum to a narrow band on one
axis instead of its full range (hold <kbd>C</kbd> and click twice to
mark it). Summing only the rows or columns inside that band, instead of
all of them, shows the distribution on the <i>other</i> axis among
events specifically correlated with the gated band. This is how
gamma-gamma coincidence spectroscopy isolates one decay cascade out of
a whole matrix: gate on one member of the cascade, and the resulting
spectrum comes out enriched in the other member(s), with unrelated
gamma rays suppressed.</p>
<p>A cut region alone still includes <b>random, uncorrelated</b>
coincidences -- unrelated gamma rays that happened to land in that band
anyway. Marking one or more background regions elsewhere on the same
axis (hold <kbd>G</kbd>) and pressing "Activate Cut" subtract them
out: the background regions' counts are pooled and scaled by the ratio
of the cut region's width to the total background width, then
subtracted channel by channel from the cut -- the same
region-width-weighted convention Integration applies for its own
background subtraction (see "How integration computes gross,
background, and net" above). Zero background regions means no
subtraction happens at all, and the result can legitimately go negative
where the scaled background outweighs the gated counts -- expected, not
an error.</p>
"""
    return _page("SpectraTools -- Knowledge Database", body)


def build_about_html():
    """Program name, version, build date, and copyright -- version/date
    come from a build_info.py generated by build.ps1 at packaging time,
    falling back to "dev"/"development build" when running from source
    (no build_info.py) rather than raising. Also falls back on
    AttributeError/SyntaxError, not just ImportError: build_info.py is
    a generated file, and a malformed one (e.g. a stray quote in
    installer.iss's AppVersion producing invalid Python, or a stale
    file missing VERSION/BUILD_DATE) should degrade to the same
    fallback rather than raise out of a Qt slot -- especially since the
    packaged exe is built --windowed, where stdout/stderr are None and
    an uncaught exception has nowhere to report itself."""
    try:
        import build_info
        version = build_info.VERSION
        build_date = build_info.BUILD_DATE
    except (ImportError, AttributeError, SyntaxError):
        version = "dev"
        build_date = "development build"

    body = f"""
<div style="text-align:center; margin-top:4rem;">
<h1>SpectraTools</h1>
<p>Version {version}</p>
<p>Built: {build_date}</p>
<p style="margin-top:2rem;">This application is created using AI Claude Code.</p>
<p>Copyright &copy; Georgi Rainovski</p>
</div>
"""
    return _page("About SpectraTools", body)


# Tried in order on Linux -- xdg-open first since it's the "correct"
# freedesktop-standard opener (and correctly respects a real default-browser
# association when one exists), then the most common browsers directly by
# binary name as a fallback for systems with no such association configured
# at all (confirmed common, not hypothetical -- see open_help_page's
# docstring). Not used on Windows/macOS, where QDesktopServices.openUrl()
# already works correctly and natively.
_FALLBACK_BROWSERS = (
    "xdg-open",
    "firefox",
    "firefox-esr",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "epiphany",
)


def _sanitized_subprocess_env():
    """A copy of the environment with LD_LIBRARY_PATH restored to its
    pre-bundle value (or removed if there wasn't one). PyInstaller's Linux
    bootloader points LD_LIBRARY_PATH at the bundled library directory so
    this frozen app's own dependencies resolve correctly, saving whatever
    the real original value was (if any) under LD_LIBRARY_PATH_ORIG for
    exactly this situation -- a child process like a system browser
    inheriting the bundle's path unmodified can end up loading mismatched
    bundled libraries instead of its own and crashing outright. Confirmed
    concretely, not just in theory: with LD_LIBRARY_PATH left pointing at
    the bundle, a spawned Epiphany fails immediately with `libstdc++.so.6:
    version 'GLIBCXX_3.4.30' not found`, since the bundled libstdc++ is
    older than what a real system WebKitGTK build needs; with it removed,
    the exact same launch works cleanly."""
    env = os.environ.copy()
    orig = env.pop("LD_LIBRARY_PATH_ORIG", None)
    if orig is not None:
        env["LD_LIBRARY_PATH"] = orig
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def open_help_page(html):
    """Writes html to a fresh temp file and opens it in the system's
    default browser, returning whether that succeeded. `delete=False`
    is required -- the file must still exist after this function
    returns for the browser to open it -- and cleanup is left to the OS
    temp directory rather than this code, matching this feature's
    no-caching, no-persistence design. The write (a locked-down TEMP
    directory) can fail -- every other I/O boundary in this codebase
    surfaces that kind of failure to the user instead of silently doing
    nothing (see e.g. main_window.py's _write_spectrum,
    _normalize_spectra), so this returns a bool for the caller to do
    the same rather than raising or swallowing it.

    On Windows/macOS, opens via QDesktopServices.openUrl(), which
    already works correctly there. On Linux, that same call is
    deliberately NOT used at all -- confirmed two independent, both
    real ways for it to fail silently or worse:
    (1) many real Linux installs have no xdg-open and no
        ~/.config/mimeapps.list / /etc/xdg/mimeapps.list configured at
        all, so it just returns False with nothing to show for it; and
    (2) even when xdg-open IS present and DOES find a browser, Qt's own
        internal invocation of it inherits this frozen process's own
        (PyInstaller-set) environment unmodified, including
        LD_LIBRARY_PATH -- which crashes the spawned browser (see
        _sanitized_subprocess_env's docstring for the exact confirmed
        failure). There is no way to pass Qt's internal call a
        sanitized environment, since QDesktopServices.openUrl() takes
        no such parameter -- the only fix is to never let it make that
        call in the first place and always launch the browser
        ourselves, with an environment we control, via
        _FALLBACK_BROWSERS (which still tries xdg-open first, so a
        properly configured system's real default browser choice is
        still respected -- just invoked by us, sanitized, instead of
        by Qt)."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", prefix="spectratools_help_",
            encoding="utf-8", delete=False,
        ) as f:
            f.write(html)
            path = f.name
    except OSError:
        return False
    if not sys.platform.startswith("linux"):
        return QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    env = _sanitized_subprocess_env()
    for candidate in _FALLBACK_BROWSERS:
        binary = shutil.which(candidate)
        if binary is None:
            continue
        try:
            subprocess.Popen([binary, path], env=env)
        except OSError:
            continue
        return True
    return False
