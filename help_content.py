"""Builds the three Help-menu pages (HowTo, Knowledge Database, About) as
self-contained HTML strings, and opens a generated page in the system's
default web browser. No external files, no external links -- Knowledge
Database's figures (help_figures.py) are embedded as base64 data URIs, so
a generated page has zero dependencies once written to disk."""

import base64
import tempfile

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from help_figures import (
    anatomy_of_a_fit_figure,
    calibration_curve_figure,
    multiplet_figure,
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
<tr><td><kbd>Ctrl+2</kbd></td><td>Toggle the Fit Results panel</td></tr>
<tr><td><kbd>Ctrl+3</kbd></td><td>Toggle the Fit Parameters panel</td></tr>
<tr><td><kbd>Ctrl+C</kbd></td><td>Clear the active spectrum's in-progress marks and committed fits</td></tr>
<tr><td><kbd>Ctrl+Shift+C</kbd></td><td>Clear the active spectrum's committed fits only (in-progress marks untouched)</td></tr>
<tr><td><kbd>Ctrl+E</kbd></td><td>Export the active spectrum's fits</td></tr>
<tr><td><kbd>Ctrl+I</kbd></td><td>Integrate</td></tr>
</table>

<h2>Step-by-step guides</h2>

<h3>1. Loading a spectrum</h3>
<p><b>File &gt; Open...</b> (<kbd>Ctrl+O</kbd>) opens a file picker for the
three supported formats: <b>.txt</b> (one integer count per line, channel
number implied by line position), <b>.spe</b>,
and <b>.spk</b>. Recently opened files also appear under
<b>File &gt; Recent Files</b> for one-click reopening. Every spectrum you
open stays loaded until you close it -- opening a new one adds it
alongside the others rather than replacing what's already there.</p>

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
spectrum, prompting you to choose one of the same three formats
(.txt/.spe/.spk) it can read.</p>

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

<h3>6. Performing a fit</h3>
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
share one FWHM).</li>
<li>Press <kbd>Ctrl+F</kbd> to fit. Results appear in the Fit Results
panel; per-parameter values (with uncertainties) appear in the Fit
Parameters panel.</li>
</ol>
<p>See the Knowledge Database page for exactly what's being computed
here, and what each fit parameter means.</p>
<p><kbd>Ctrl+C</kbd> clears the active spectrum's in-progress B/R/P
marks as well as its already-committed fits. <kbd>Ctrl+Shift+C</kbd>
clears only the active spectrum's committed fits, leaving any
in-progress marks alone. <kbd>Ctrl+E</kbd> exports the active
spectrum's committed fits.</p>

<h3>7. Integration</h3>
<p><kbd>Ctrl+I</kbd> computes gross/background/net counts across the
marked regions directly (background centroid, FWHM, skewness, and area,
all with uncertainties) <i>without</i> fitting a peak shape -- faster,
and useful when a peak is too irregular to fit well, or when you only
need a total count rather than individual peak parameters.</p>

<h3>8. View options</h3>
<p><kbd>Ctrl+G</kbd> toggles a logarithmic Y axis. <kbd>Ctrl+D</kbd>
toggles dark theme. <kbd>Ctrl+1</kbd>/<kbd>Ctrl+2</kbd>/<kbd>Ctrl+3</kbd>
show or hide the Spectra, Fit Results, and Fit Parameters panels.
<kbd>Ctrl+=</kbd>/<kbd>Ctrl+-</kbd> zoom the X axis in/out around the
current view, and <kbd>Ctrl+0</kbd> resets to the full spectrum.</p>
"""
    return _page("SpectraTools -- HowTo", body)


def build_knowledge_database_html():
    """The fit model, parameter meanings, and calibration math, each
    cross-referenced to one of help_figures.py's four annotated figures
    -- content verified against peak_fit.py/fit_mode.py's actual
    behavior through two rounds of correction after the first draft
    shipped wrong claims about area terminology and multiplet parameter
    sharing (see commits f9c73cd and ce57765)."""
    anatomy_src = _embed_png(anatomy_of_a_fit_figure())
    tail_src = _embed_png(tail_effect_figure())
    multiplet_src = _embed_png(multiplet_figure())
    calibration_src = _embed_png(calibration_curve_figure())

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
same multiplet share one FWHM by default -- see Figure 3 -- unless you
turn on the Fit Parameters panel's "Independent widths" option.</td></tr>
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

<h2>Multiplets</h2>
<figure>
<img src="{multiplet_src}" alt="Multiplet fit">
<figcaption>Figure 3. Three peaks fit together; two of them overlap
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
"""
    return _page("SpectraTools -- Knowledge Database", body)


def build_about_html():
    """Program name, version, build date, and copyright -- version/date
    come from a build_info.py generated by build.ps1 at packaging time,
    falling back to "dev"/"development build" when running from source
    (no build_info.py) rather than raising."""
    try:
        import build_info
        version = build_info.VERSION
        build_date = build_info.BUILD_DATE
    except ImportError:
        version = "dev"
        build_date = "development build"

    body = f"""
<div style="text-align:center; margin-top:4rem;">
<h1>SpectraTools</h1>
<p>Version {version}</p>
<p>Built: {build_date}</p>
<p style="margin-top:2rem;">Copyright &copy; Georgi Rainovski</p>
</div>
"""
    return _page("About SpectraTools", body)


def open_help_page(html):
    """Writes html to a fresh temp file and opens it in the system's
    default browser. `delete=False` is required -- the file must still
    exist after this function returns for the browser to open it --
    and cleanup is left to the OS temp directory rather than this code,
    matching this feature's no-caching, no-persistence design."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", prefix="spectratools_help_",
        encoding="utf-8", delete=False,
    ) as f:
        f.write(html)
        path = f.name
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))
