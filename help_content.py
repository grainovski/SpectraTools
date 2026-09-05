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
<tr><td><kbd>F5</kbd></td><td>Reload Spectrum (re-read the active spectrum from disk)</td></tr>
<tr><td><kbd>Ctrl+S</kbd></td><td>Save Spectrum...</td></tr>
<tr><td><kbd>Ctrl+W</kbd></td><td>Close Spectrum (remove the active spectrum from the program)</td></tr>
<tr><td><kbd>Ctrl+Q</kbd></td><td>Exit</td></tr>
</table>

<h3>View menu</h3>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>Ctrl+G</kbd></td><td>Go To an energy or channel</td></tr>
<tr><td><kbd>Ctrl+Y</kbd></td><td>Toggle log-scale Y axis</td></tr>
<tr><td><kbd>Ctrl+1</kbd></td><td>Toggle the Spectra panel</td></tr>
<tr><td><kbd>Ctrl+D</kbd></td><td>Toggle dark theme</td></tr>
</table>

<h3>Operations menu</h3>
<table>
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td><kbd>Ctrl+L</kbd></td><td>Calibration...</td></tr>
<tr><td><kbd>Ctrl+T</kbd></td><td>Toggle Calibration Active</td></tr>
<tr><td><kbd>Ctrl+Shift+L</kbd></td><td>Automatic Calibration...</td></tr>
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
<p>The mouse wheel zooms the X axis about the cursor. <b>Dragging with
either mouse button held</b> slides the spectrum left and right at the
current zoom: the X axis keeps its width, and the Y axis rescales as you
go to fit whatever is now on screen, so a small peak is not left
flattened by a tall one that has scrolled out of view. The view stops at
the ends of the data.</p>
<p>Marking always takes precedence over panning. While you hold a marking
key -- <kbd>B</kbd>, <kbd>R</kbd> or <kbd>P</kbd> -- the left button
places marks and does not pan, so a slightly unsteady hand cannot nudge
the view mid-mark. The right button pans regardless, which is useful for
scrolling along while keeping a marking key held. Left-drag also stands
aside while the plot toolbar's own Pan or Zoom tool is switched on, since
that tool drives the left button itself.</p>

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
<tr><td><kbd>C</kbd></td><td>Hold and click twice per region to mark a cut (signal) region -- any number of regions allowed</td></tr>
<tr><td><kbd>G</kbd></td><td>Hold and click twice per region to mark a background region for the cut -- any number of regions allowed</td></tr>
<tr><td><kbd>Ctrl+Alt+C</kbd></td><td>Activate Cut (same as clicking the button; needs a cut region marked first)</td></tr>
<tr><td><kbd>B</kbd></td><td>Hold and click twice per region (two regions needed, four clicks total) to mark the two background regions for fitting the working projection</td></tr>
<tr><td><kbd>R</kbd></td><td>Hold and click twice (once on each end) to mark the fit region for the working projection</td></tr>
<tr><td><kbd>P</kbd></td><td>Hold and click once per peak to mark a peak position on the working projection</td></tr>
<tr><td><kbd>Ctrl+F</kbd></td><td>Fit</td></tr>
<tr><td><kbd>Ctrl+I</kbd></td><td>Integrate</td></tr>
<tr><td><kbd>Ctrl+B</kbd></td><td>Preview the background fit from the two background regions alone (no fit region or peaks needed); press again to hide</td></tr>
<tr><td><kbd>Ctrl+C</kbd></td><td>Clear the cut and background marks (exactly as <b>Clear Marks</b> does), clear in-progress fit marks, and hide committed fits (not delete)</td></tr>
<tr><td><kbd>Ctrl+Shift+C</kbd></td><td>Permanently delete the working projection's committed fits (in-progress marks untouched)</td></tr>
<tr><td><kbd>Ctrl+E</kbd></td><td>Export the working projection's fits</td></tr>
<tr><td><kbd>Ctrl+2</kbd></td><td>Toggle the Fit Results panel</td></tr>
<tr><td><kbd>Ctrl+3</kbd></td><td>Toggle the Fit Parameters panel</td></tr>
<tr><td><kbd>Ctrl+L</kbd></td><td>Calibrate... (same calibration as the main window's Operations &gt; Calibration... -- see "15. Matrix analysis" below)</td></tr>
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
five supported formats: <b>.txt</b> (one integer count per line, channel
number implied by line position), <b>.spe</b>, <b>.spk</b>,
<b>.n42</b> (ANSI/IEEE N42.42; only the raw histogram and, if present,
the embedded energy calibration are read -- everything else in the file
is ignored, and an N42 file's calibration is applied automatically only
when no calibration is already active), and <b>.lzs</b> (labZY /
nanoMCA; the histogram plus the instrument's own two-point energy
calibration, which is read only when the file marks it enabled and is
applied on the same terms as an N42 file's -- acquisition times,
hardware registers and firmware details are ignored). Recently opened files also
appear under <b>File &gt; Recent Files</b> for one-click reopening.
Every spectrum you open stays loaded until you close it -- opening a
new one adds it alongside the others rather than replacing what's
already there.</p>
<p><b>ROOT files</b> (<b>.root</b>) are opened separately, via
<b>File &gt; Open ROOT File...</b>, because one ROOT file is not one
spectrum -- it is a directory tree that can hold dozens of named
objects. Picking the file opens a second dialog listing every 1D and 2D
histogram in it; select one or more 1D histograms to load them as
spectra, or a single 2D histogram to open it as a matrix. The two
cannot be mixed in one go, since a matrix opens its own window while
spectra join the list. If the histogram's axis is calibrated, that
calibration is read and applied automatically, the same way an N42
file's is. See the Knowledge Database page for which histograms are
refused and why.</p>
<p><b>File &gt; Reload Spectrum</b> (<kbd>F5</kbd>) re-reads the active
spectrum's file from disk while keeping its calibration, fits and
marks -- useful for watching a measurement that is still counting. If
the file has changed length, the fits and marks are cleared and the
status bar says so: they are anchored to channel numbers, which after a
length change may no longer point at the same thing.</p>

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
<b>.n42</b> and <b>.lzs</b> are read-only and aren't offered here.</p>

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
<p>Every row's Value cell in the Fit Parameters panel stays editable
after a fit -- edit one and press <kbd>Ctrl+F</kbd> again to re-fit
from the new number. Check a row's "Fix" box to hold that parameter
at its current Value for the next fit instead of letting the
optimizer adjust it. Leave a row unchecked and its Value is used only
as that parameter's starting guess -- the fit can still move it.</p>
<p>The panel also carries three checkboxes. <b>Independent widths</b>
fits each peak its own width instead of one shared FWHM. <b>Left tail</b>
adds a low-channel tail to the peak shape. <b>Fit background</b> fits the
background line together with the peaks rather than subtracting it
first -- peak uncertainties grow when it is on, because they then
include how well the background itself is known, and two extra rows
(Background level and Background slope) appear in the panel. See the
Knowledge Database page for when each is worth using.</p>
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

<p>Two other formats are available from the same dialog, chosen by the
extension you type (the filter only decides when the name has none):
<code>.csv</code> writes one row per fitted peak for a spreadsheet, and
<code>.tex</code> writes a LaTeX <code>tabular</code>, one row per peak,
ready to paste into a paper. When a calibration is active, positions and
widths are exported in keV and the column headers say so; areas are
never converted, since they are counts. An uncertainty the fit could not
determine is left as an empty cell in CSV rather than the text "n/a", so
the column stays numeric.</p>

<h3>10. Saving and reloading fits</h3>
<p><b>File &gt; Save Fits...</b> writes every fit on the active spectrum
to a <code>.json</code> file, and <b>Load Fits...</b> brings them back,
so an analysis survives closing the program. Integration results are not
saved -- they report a region's totals rather than fitted peaks, and the
status bar says how many were left out.</p>
<p>Loading asks how you want them back. <b>Yes</b> restores the saved
numbers exactly as they were reported when the file was written.
<b>No</b> re-runs each fit from its saved marks using the current
version of the program -- use this to bring an older analysis up to
date, since v4.0.0 changed how peak areas and their uncertainties are
computed. Fits saved against a different spectrum can be loaded too, and
you are told when that is what is happening.</p>

<h3>11. Calibrating from fitted peaks</h3>
<p><b>Operations &gt; Calibrate from Fitted Peaks...</b> lists every peak
you have already fitted. Type the known energy beside the ones you can
identify, leave the rest blank, and press OK: the calibration is fitted
by least squares through those points. Because the channel positions
come from fitted centroids rather than from where you clicked, this is
more accurate than entering coefficients by hand.</p>
<p>The status bar reports the <b>worst residual</b> afterwards. That is
the number to look at: a mistyped energy, or a peak matched to the wrong
line, shifts the whole calibration while leaving the coefficients
looking perfectly reasonable. A residual much larger than your peaks'
own position uncertainties means one of the assignments is wrong. Two
assignments determine a line and three a quadratic; assign more than the
minimum whenever you can.</p>
<p><b>Using a source file.</b> Instead of typing every energy from a
table, press <b>Load source...</b> and pick a <code>.sou</code> file: a
plain-text list of one nuclide's lines, one per row, as four numbers --
energy in keV, its error, relative intensity, its error -- with no
header. Type the energies of <b>two</b> peaks you recognise, then press
<b>Suggest remaining</b>. A straight line through those two anchors
predicts where every other peak falls, and each blank row is filled with
the source line nearest that prediction -- but only when that line is
clearly nearest, closer than half the distance to the runner-up. Rows
where two lines compete stay blank rather than guessed, as do rows whose
nearest line is already taken by an anchor. Suggested cells are tinted
and carry a tooltip; they are ordinary cells you can edit or clear, and
OK fits through exactly what the table shows. Pressing Suggest again
clears the untouched suggestions and recomputes from your own values, and
so does loading a different source file -- guesses made from one nuclide
are never carried into another.</p>
<p><b>Each peak has a tick, and unticking it leaves that point out of
the fit.</b> The energy you typed is kept -- the tick only decides
whether the calibration is fitted through that point. It is what the
residual strip is for: when one point sits far off the line, untick it
and watch the fit and the residuals redraw without it. The excluded
point is still drawn, as a hollow marker, and its residual against the
new fit is still shown, so you can see what you rejected and change your
mind. <b>Suggest remaining</b> ignores unticked points too, since
suggesting is itself an energy fit and a suspect anchor would place every
other line wrongly. The CalEnEff export is the exception: an excluded
point is still written there, because its area and intensity are
unaffected by a doubt about the energy fit.</p>
<p><b>Your assignments are remembered.</b> Everything you type stays with
that spectrum for the session, so refitting and reopening the dialog
brings it back rather than making you retype it -- and so does the
tick, so a point you judged an outlier does not quietly rejoin the fit
after a refit. A peak whose centroid moved further than
its own width comes back blank, because past that distance it is a
different peak.
<b>Clear</b> forgets every assignment and the loaded source, and puts
every tick back. It deliberately leaves the active calibration
alone: discarding your identifications and un-calibrating the spectrum
are separate actions.</p>
<p><b>The fit opens in its own window</b> showing the assigned points
with their uncertainties, the calibration curve across the whole
spectrum, a residual strip, the coefficients with their errors, and the
reduced chi-squared. It appears as soon as there are enough assignments
to fit a calibration and <b>redraws after every change</b> -- a typed
energy, a loaded source, a suggestion accepted -- so a misidentified
line shows in the residuals while you can still correct it, rather than
only after the calibration has been applied. Unticking a point redraws
it too. Clearing the assignments takes it down again. <b>Finish and save for CalEnEff...</b> writes a
seven-column file for the efficiency-calibration program: channel and
its error, net area and its error, energy, and relative intensity with
its error.</p>

<h3>12. Automatic calibration</h3>
<p><b>Operations &gt; Automatic Calibration...</b> (<kbd>Ctrl+Shift+L</kbd>)
calibrates a spectrum of a known source with no fits and no calibration
to start from. It finds the peaks, fits them, works out which line of
the source each one is, and opens the result in the same Calibrate from
Fitted Peaks dialog as "11. Calibrating from fitted peaks" for you to
check.</p>
<ol>
<li>Load the calibration spectrum and make it the active one.</li>
<li>Open the dialog and press <b>Load source...</b> to pick the
<code>.sou</code> file of the nuclide the spectrum was taken with. A
source already loaded for this spectrum in the Calibrate from Fitted
Peaks dialog is offered without asking.</li>
<li>Set the <b>sensitivity</b>: how many standard deviations a peak must
stand above the continuum around it. The count of peaks found updates
as you change it -- lower it if a line you can see is missing, raise it
if noise is being counted. The default of 5 finds the lines a person
would point at.</li>
<li>Press <b>Run</b>. Every group of found peaks is fitted -- peaks
closer than three widths are fitted together as one multiplet -- and
the fits are added to Fit Results. <b>Fits already on the spectrum are
kept</b>: the new ones are appended after them, and the status bar says
how many were there before.</li>
<li>The Calibrate from Fitted Peaks dialog then opens with the source
loaded and every identified peak's energy already filled in. Check the
residual strip in the live plot, untick any point that sits off the
line, and press OK to apply the calibration exactly as in "11.
Calibrating from fitted peaks".</li>
</ol>
<p>The identification is either confident or refused; it never guesses.
When no confident assignment exists -- the spectrum is not this nuclide,
too few of its lines were found, or two different calibrations explain
it equally well -- the status bar and the dialog say why, and the dialog
opens with the peaks fitted but unassigned. Type the energies of two
peaks you recognise and press <b>Suggest remaining</b>, exactly as you
would with hand-fitted peaks. The status line also reports peaks that
were found but could not be fitted, groups skipped for lying too close
to the spectrum edge for a background region, and fits set aside for an
implausible width or a negative area.</p>
<p>A peak fitted twice -- once by hand before the run, once by the
automatic pass -- appears twice in the table, and the energy goes to the
automatic copy. Running again appends a second set of fits;
<kbd>Ctrl+Shift+C</kbd> deletes every committed fit on the spectrum if
you would rather start clean.</p>

<h3>13. Go To an energy or channel</h3>
<p><kbd>Ctrl+G</kbd>, or <b>View &gt; Go To...</b>, jumps the view to one
place in the spectrum. Type an <b>energy in keV</b> when a calibration is
active, or a <b>channel</b> when one is not -- the dialog asks for whichever
the x-axis is currently showing, and states the range the spectrum covers so
you know what is available before typing.</p>
<p>The view centres on what you entered, zoomed to a 100-channel window, and
the Y axis rescales to whatever is now on screen. A dotted purple line marks
the spot. <kbd>Ctrl+=</kbd> and <kbd>Ctrl+-</kbd> widen or narrow the view
from there.</p>
<p>The window is 100 <i>channels</i> rather than a fixed keV span on purpose:
it means the same thing whether or not a calibration is active, and stays
sensible across coarse and fine binning where a fixed energy span would be
far too wide on one and far too narrow on the other. Near either end of the
spectrum the window slides inward rather than being cut in half, so a line at
channel 5 is still shown with context around it.</p>
<p>The mark is stored as a channel, so toggling or replacing the calibration
moves it to the matching energy rather than stranding it at a stale
coordinate. <kbd>Ctrl+C</kbd> (Clear) removes it along with the fit marks.
Go To works the same way in a matrix panel's projection window.</p>

<h3>14. View options</h3>
<p><kbd>Ctrl+Y</kbd> toggles a logarithmic Y axis. <kbd>Ctrl+D</kbd>
toggles dark theme. <kbd>Ctrl+1</kbd>/<kbd>Ctrl+2</kbd>/<kbd>Ctrl+3</kbd>
show or hide the Spectra, Fit Results, and Fit Parameters panels.
<kbd>Ctrl+=</kbd>/<kbd>Ctrl+-</kbd> zoom the X axis in/out around the
current view, and <kbd>Ctrl+0</kbd> resets to the full spectrum.</p>

<h3>15. Matrix analysis</h3>
<p><b>File &gt; Open Matrix...</b> (<kbd>Ctrl+Shift+O</kbd>) opens a
2D coincidence matrix (<b>.mtx</b>) in its own window. Only the raw
histogram is read -- there's no way to save a matrix back out.
Decoding a full-size matrix takes several seconds, so a
<b>Reading matrix...</b> progress window appears while it works; the
application stays responsive throughout, and the panel opens by itself
when the read finishes. Loading is fastest from a local disk -- reading
a matrix across a network share, or from a Windows drive inside WSL, is
noticeably slower.
The matrix panel computes both its X and Y projections up front; pick
which one to work on from the dropdown. Hold <kbd>C</kbd> and click
twice to mark a cut (signal) region -- repeat for as many gates as
you want, and they are summed together -- and hold <kbd>G</kbd> and click
twice for each background region -- any number of background regions
are allowed, and more background generally means better statistics.
<b>Clear Marks</b> resets the cut region and every
background region mark together -- fit marks, committed fits, and any
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
own zoom, and dragging with either mouse button held slides the
projection along at the current zoom with the Y axis rescaling to suit,
exactly as it does in the main window -- including that a held marking
key (<kbd>C</kbd> or <kbd>G</kbd> for cut marks, <kbd>B</kbd>/<kbd>R</kbd>/
<kbd>P</kbd> for fit marks) keeps the left button on marking rather than
panning.</p>
<p><kbd>Ctrl+C</kbd> here clears everything you have marked on the
projection: the cut and background marks (the same thing the
<b>Clear Marks</b> button does), any in-progress fit marks, and it hides
the committed fits. An already-activated cut spectrum is a separate
loaded spectrum and is not affected.</p>
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
energy. Bounded to at most 20&times;&sigma; -- see below.</td></tr>
<tr><td>background slope / intercept</td><td>The straight line fixed by
the two background regions (Figure 1).</td></tr>
</table>
<p><b>Why &beta; is capped.</b> If you enable the left tail on a peak that
does not really have one, <i>r</i> settles at nearly zero -- and once it
does, the tail contributes nothing to the shape, so the fit has no way to
tell one &beta; from another. Left free, &beta; then drifts to whatever
value it happens to reach, which can be astronomically large. That does no
harm to the fitted curve, but the peak's <i>volume</i> includes the whole
tail integrated out to infinity, so it would be reported as an absurd
number -- billions of times the peak's real content -- while the fit
itself still looked healthy. Capping &beta; at 20&times;&sigma; keeps that
from happening. A real detector tail has &beta; of the order of &sigma;,
so the cap is far away from anything physical and does not affect a
genuine tail fit.</p>
<p>If &beta; is reported at exactly 20&times;&sigma;, read it as "there is
no tail here to measure" rather than as a measurement. Its uncertainty
will usually read "n/a" in that case, for the same reason -- see "When an
uncertainty reads n/a" below.</p>

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
<p>A parameter checked "Fix" in the Fit Parameters panel (see the
HowTo page's "7. Performing a fit") is excluded from this
optimization entirely -- it's held at its Value cell's number rather
than fit, so its own uncertainty is reported as exactly zero instead
of coming from the covariance matrix.</p>
<p>The Fit Results panel's Volume column shows one number per peak: its
<b>net</b> volume -- the exact integral of that peak's fitted shape,
background excluded. Without a tail that is the familiar Gaussian
integral:</p>
<p style="text-align:center"><code>area = amplitude &middot; &sigma; &middot; &radic;(2&pi;)</code></p>
<p>and with a tail enabled the tail's own counts are included too, via
the closed-form integral of the full Hypermet shape:</p>
<p style="text-align:center"><code>area = amplitude &middot; [ (1&minus;r) &middot; &sigma;&radic;(2&pi;) + 2r&beta; / erfcx(y) ]</code></p>
<p>where <code>y = &sigma;/(&beta;&radic;2)</code> as elsewhere and
<code>erfcx</code> is the scaled complementary error function. Setting
r = 0 recovers the line above exactly, so untailed fits are unaffected.
The tail can hold a substantial share of a peak: for r = 0.3 with
&beta; = &sigma; it is about 14% of the volume, and for a long tail
(&beta; = 10&sigma;/3) over 40%.</p>
<p>The volume's uncertainty is propagated through the fit's full
covariance matrix, including the <i>correlations</i> between amplitude,
&sigma; and the tail parameters. That matters because amplitude and
&sigma; are strongly anti-correlated in a peak fit -- a wider peak with
a lower amplitude fits nearly as well -- so treating them as
independent overstates the uncertainty.</p>
<p>Hover over a row for the full breakdown: that same peak's <b>full</b>
volume --</p>
<p style="text-align:center"><code>full_area = area + (background_slope &middot; position + background_intercept) &middot; FWHM</code></p>
<p>-- net volume plus the background level at the peak's own center
(the straight line fixed by the two background regions, Figure 1),
times its FWHM. The background line carries its own uncertainty, since
both ends are averages of real counts, and
<code>full_area_err</code> combines the two in quadrature. See "How
certain is the background?" below.</p>
<p>Region-level full/net totals for the whole fit work the same way --
the region's full total is the sum of the raw, observed counts across
the whole fit region (not the fitted model curve), and its net total
sums just the peaks' own net volumes. Net volume is almost always the
number you actually want (for example, when computing activity or a
branching ratio).</p>
<p>The net total's <i>uncertainty</i> is propagated through the fit's
whole covariance matrix in one step, exactly as a single peak's volume
is. It is deliberately not the peaks' individual uncertainties added in
quadrature, because those are not independent of one another. In an
overlapping multiplet the amplitudes are strongly <i>anti</i>-correlated:
the data pins down how many counts the group holds far better than it
pins down how they divide between the peaks. So the total is better
determined than any one component, and adding in quadrature can overstate
it by a wide margin -- for a doublet one &sigma; apart, by a factor of
ten. Well-separated peaks and single-peak fits are unaffected.</p>
<p>A practical consequence worth knowing: the net total's uncertainty
barely changes as two peaks are moved closer together, even though each
individual peak's uncertainty grows quickly. That is real, not a
rounding artefact -- the blend is still the same number of counts.</p>

<h2>How the background is determined, and what <kbd>Ctrl+B</kbd> shows</h2>
<p>The background under a fit is a straight line, and it is fixed by the
two marked background regions alone. Each region's counts are averaged to
give one level, and the line is drawn through those two points -- it is
not a least-squares fit to the background channels, which is why its
uncertainty can be written down exactly (see "How certain is the
background?" above).</p>
<p><kbd>Ctrl+B</kbd> draws that line before you commit to anything. It
needs both background regions marked and nothing else -- no fit region, no
peaks -- so it is the quickest way to check that your two regions actually
sit on background before spending a fit on them. Press it again to hide
the line. It is a preview only: nothing is fitted, computed or stored, and
the line disappears the moment the marks change.</p>
<p><b><kbd>Ctrl+B</kbd> is not affected by the "Fit background"
checkbox.</b> The preview always shows this same two-point line, whether
that box is ticked or not. The checkbox changes what happens when you
<i>fit</i>, not what the preview draws -- so the preview tells you where
the background regions put the line, and nothing about which fitting mode
you are in.</p>

<h2>How certain is the background?</h2>
<p>The background under a fit is the straight line through the mean
level of each of the two marked background regions. Those means are
averages of real counts, so each has a counting uncertainty of its own,
and the line inherits it. Writing <code>t</code> for how far along you
are between the two regions' centres:</p>
<p style="text-align:center"><code>background(x) = (1&minus;t) &middot; level&#8321; + t &middot; level&#8322;</code></p>
<p style="text-align:center"><code>background_err(x) = &radic;[ (1&minus;t)&sup2; &middot; var(level&#8321;) + t&sup2; &middot; var(level&#8322;) ]</code></p>
<p>where <code>var(level)</code> is the variance of that region's mean
-- the summed counts divided by the channel count squared. The two
regions are separate stretches of spectrum, so there is no cross term
between them.</p>
<p>This is drawn as a faint shaded band around the dashed background
line. It is narrowest <i>between</i> the two background regions and
widens outward from there, which is worth watching: it shows directly
that a background <i>interpolated</i> between two regions is far better
determined than one <i>extrapolated</i> beyond them. If the band is wide
where your peak sits, moving the background regions closer to the peak
will tighten it.</p>
<p>Note that the narrowest point is between the regions rather than at
either one. Averaging two independent measurements beats either on its
own, so the line is best known somewhere in the middle: for two regions
whose means are known to &plusmn;2.1 and &plusmn;2.4 counts, the band
closes to about &plusmn;1.6 counts between them.</p>
<p>Widening a background region also tightens the band, since averaging
more channels measures the level more precisely -- but only while the
region stays on genuine background. A region stretched over the
shoulder of a neighbouring peak measures something that is not
background at all, and no amount of averaging fixes that.</p>

<h2>Fitting the background with the peaks</h2>
<p>By default the background is fixed <i>before</i> the peaks are fitted:
the line through the two regions is subtracted, and the peaks are then
fitted to what remains. That is what TV does, and it is fast and
predictable.</p>
<p>It also quietly overstates how well the peaks are known. The data
cannot really distinguish a slightly taller peak sitting on a slightly
lower background from the reverse, so treating the background as exact
claims information nobody has, and every peak uncertainty comes out
smaller than it should be.</p>
<p>Ticking <b>Fit background</b> in the Fit Parameters panel fits the
background's level and slope <i>together with</i> the peaks, as two more
free parameters. The marked background regions then only provide the
starting guess. Two things follow:</p>
<ul>
<li>Peak uncertainties <b>grow</b>. That is the correction, not a
problem -- they now include how well the background itself is known. The
increase is modest on a well-defined peak, around 3%, and grows with how
much the background contributes relative to the peak's own counts.</li>
<li>The background gets its own fitted uncertainty, and the shaded band
is computed from that instead of from the two regions.</li>
</ul>
<p>Mechanically, the difference is what the optimiser is handed. The
two-point line is computed either way -- it is what seeds the peak
amplitudes and widths in both modes, since those estimates want the
background out of the way whichever way it is later treated. What changes
is the target:</p>
<ul>
<li><b>Unticked</b>, the line is subtracted once and the optimiser is
fitted against the <i>residual</i>. The model contains only peaks; the
background has no free parameters and cannot move.</li>
<li><b>Ticked</b>, the optimiser is fitted against the <i>raw counts</i>,
and the model is peaks <i>plus</i> a line whose level and slope are two
more free parameters. The two-point line is demoted to their starting
guess.</li>
</ul>
<p>That is why the reported uncertainties differ: fitting a residual as
though it were the measurement asserts the subtraction was exact, while
fitting the raw counts lets the data say how much of what it sees is peak
and how much is background.</p>
<p>It is not automatically the better choice. Two extra free parameters
cost two degrees of freedom, and on a short fit region the background
slope and the peak width start to describe the same thing, which can
make the fit less stable rather than more honest. Use it when the
background matters to your answer -- a weak peak on a steeply sloping
continuum -- and leave it off for a strong, well-isolated peak on a flat
background, where it buys nothing.</p>
<p>Rows for <b>Background level</b> and <b>Background slope</b> appear in
the Fit Parameters panel while it is on, and can be fixed like any other
parameter. "Level" is the height of the line at the middle of the fit
region, not at channel zero -- and that is a deliberate choice, not a
display convention. Fitting the level at channel zero would make it an
extrapolated number: for a peak near channel 3000 it is enormous and
almost perfectly anti-correlated with the slope, so the two parameters
stop being independently determinable and the fit becomes unstable for a
reason that has nothing to do with the data. Measuring the level at the
middle of the region removes that. The value you see is converted back to
an ordinary line before anything is drawn or reported.</p>

<h2>What you see on the plot after fitting</h2>
<p>A committed fit draws the same four things in both modes:</p>
<ul>
<li>The <b>dashed background line</b>, spanning from the start of the left
background region to the end of the right one -- so it is drawn across the
ground it was actually measured from, not just under the peak.</li>
<li>A <b>faint shaded band</b> around that line, its half-width being the
background's own uncertainty at each channel.</li>
<li>The <b>total model curve</b> -- background plus every peak.</li>
<li>Each <b>peak's own contribution</b>, and its position label.</li>
</ul>
<p>The band is drawn either way; what differs is where its width comes
from. Unticked, it comes from the counting statistics of the two region
means. Ticked, it comes from the fit's own covariance for the two
background parameters -- so the band you see is what the fit concluded
about the line, not what the two regions implied on their own.</p>
<p>That makes the ticked band <b>noticeably wider away from the fit
region</b>. On one worked example the unticked band ran from about
&plusmn;1.6 counts in the middle to &plusmn;2.7 at the far edge, while the
ticked one ran from &plusmn;1.9 to &plusmn;8.1 over the same span. Both
are narrowest near the middle of the fit region and flare outward; the
ticked one flares much harder, which is the fit being honest that it
constrained the line where the data is and not beyond it.</p>

<h2>ROOT files</h2>
<p>A ROOT file (<code>.root</code>) is not one spectrum -- it is a
directory tree that can hold dozens of named objects. <b>File &rarr; Open
ROOT File...</b> therefore asks twice: first for the file, then for what
to take out of it. Select one or more 1D histograms to load them as
spectra, or a single 2D histogram to open it as a matrix. The two cannot
be mixed in one go, because a matrix opens its own window while spectra
join the list.</p>
<p>Only 1D and 2D histograms are listed. Trees, graphs and
instrument-specific objects are left out rather than offered and then
refused.</p>
<p><b>Calibration comes free when the file has one.</b> A ROOT axis
carries real coordinates, so a histogram binned in keV already describes
its own calibration -- it is read and applied automatically, exactly as
an N42 file's is. A histogram binned in plain channels reports no
calibration, since an identity transform would be noise.</p>
<p>Two things are refused rather than guessed at, because guessing would
produce numbers that look fine and are wrong: a histogram whose contents
are fractional (one that has been scaled or weighted, so its bins are no
longer counts), and one with non-uniform bin widths, which no polynomial
channel calibration can express.</p>
<p>Reading is one-way. Nothing is written back to a ROOT file.</p>

<h2>Calibrating from fitted peaks</h2>
<p><b>Operations &rarr; Calibrate from Fitted Peaks...</b> lists every
peak you have already fitted. Type the known energy beside the ones you
can identify, leave the rest blank, and the calibration is fitted by
least squares through those points.</p>
<p>This is the more accurate way to calibrate. The channel positions come
from fitted centroids, so the calibration inherits the precision of the
fits rather than of your aim with a cursor -- and because the fits are
already there, identifying two known lines is all it takes.</p>
<p>Peak positions are always listed in <b>channels</b>, even when a
calibration is already active. You are assigning energies in order to
determine the calibration, so showing positions that an earlier
calibration had already converted would be circular.</p>
<p>The status bar reports the <b>worst residual</b> -- the largest
disagreement between an energy you typed and what the fitted calibration
predicts there. This is the number that says whether to believe the
result: a single mistyped energy, or a peak assigned to the wrong line,
shifts the whole fit while leaving the coefficients looking perfectly
reasonable. A residual much larger than your peaks' own position
uncertainties means one of the assignments is wrong.</p>
<p>Two assignments determine a line and three a quadratic. Supplying
exactly the minimum works but is an interpolation rather than a fit: it
passes through the points exactly and so cannot tell you anything about
how good it is. Assign more than the minimum whenever you can.</p>
<p>The <b>&plusmn; ch</b> column shows each centroid's own uncertainty,
straight from the fit that produced it, and the calibration is
<i>weighted</i> by it -- a peak the fit pinned down to a hundredth of a
channel counts for more than one it could only place to within a
channel, which is often the difference between a strong line and a weak
one in the same spectrum. Without that weighting the weakest peak you
assign would pull on the answer exactly as hard as the strongest.</p>
<p>A peak whose position was held fixed, or that the fit could not
determine, shows a dash instead of a number. There is no usable weight
for such a peak, and rather than invent one the calibration falls back to
weighting every point equally.</p>
<p>When you assign more than the minimum, the status bar also reports the
fitted <b>slope and its uncertainty</b>. The slope is the coefficient
that matters for anything read far from the lines you assigned, so its
uncertainty is a direct statement of how far you can trust the
calibration away from your reference points. At exactly the minimum
number of points there is no scatter to estimate it from, and none is
reported.</p>

<h3>Assigning energies from a source file</h3>
<p>A <code>.sou</code> file describes one calibration nuclide's known
lines. It is plain text with no header: one line per gamma, four
whitespace-separated numbers each -- <b>energy in keV</b>, the
uncertainty on that energy, the <b>relative intensity</b>, and its
uncertainty. The intensity scale is per-file and not comparable between
files: most sources here normalise the strongest line to 10000, but not
all do. Only the energies are used for calibration; the other columns
are read and checked, but nothing is inferred from them.</p>
<p>Press <b>Load source...</b>, choose the file, then type the energies
of <b>two</b> peaks you recognise and press <b>Suggest remaining</b>.
The two anchors define a provisional straight line, that line predicts an
energy for every other fitted peak, and each blank row is offered the
source line nearest its prediction.</p>
<p>Two anchors are needed because of an ordering problem that has no way
around it: deciding which line a peak <i>is</i> requires a channel-to-
energy mapping, which is the very thing being determined. Your two
identifications break that circle. The suggestion line is always a
straight one even when the Quadratic box is ticked -- two points cannot
define a curve, and this line only has to be good enough to tell
neighbouring lines apart, not to be the final answer.</p>
<p>A suggestion is only offered when the nearest source line is
<b>unambiguously</b> nearest: closer than half the distance to the
next-nearest line. Where two lines compete for one peak, the row is left
blank rather than guessed. This rule has no tunable tolerance and scales
itself to the source -- a nuclide with two lines a megaelectronvolt apart
is matched freely, while a dense spectrum such as Eu-152 is matched only
where the answer is not in doubt. Two rules follow from the same
principle: a line already claimed by one of your anchors is never
suggested for a second peak, and when two peaks are both nearest to the
same line, neither receives it.</p>
<p>Suggested cells are tinted and carry a tooltip naming the file they
came from. They are ordinary editable cells -- correct one, clear one,
or add energies of your own -- and OK fits through exactly what the table
shows, never through a hidden list. Editing a suggestion makes it yours:
the tint disappears and it counts as an anchor if you press Suggest
again, which first clears every untouched suggestion so that a new round
is computed from your values alone and never from the previous round's
guesses.</p>
<p>Loading a <b>different source file</b> clears the untouched
suggestions too. Switching nuclide is what you do on realising the first
choice was wrong, and leaving those rows filled would let OK calibrate
against the source you had just rejected. Anything you typed or corrected
yourself stays -- it is your value, not a guess -- and the status line
reports how many guesses were dropped.</p>
<p>The suggestions are a labour-saving device, not an identification.
Check the <b>worst residual</b> afterwards exactly as you would for
energies typed by hand: an unambiguously nearest line can still be the
wrong line if the provisional anchors were themselves misidentified.</p>

<h3>How automatic calibration identifies the lines</h3>
<p><b>Operations &rarr; Automatic Calibration...</b> has to break the
same circle as Suggest -- deciding which line a peak is needs a
calibration, and the calibration needs the lines -- with no anchors
typed by anyone. It does so by trying them all. Two peaks paired with two
source lines fix a straight line exactly; every pairing of the ten
strongest peaks with the fifteen strongest lines is tried, each one
predicts an energy for every other fitted peak, and a peak whose
prediction lands on a real line within half a peak width counts as
explained. The pairing that explains the most peaks wins, ties broken by
the smaller residual. Only the strongest peaks and lines propose
pairings, because the anchors need only be right and a strong peak in a
calibration spectrum is far more likely to be a strong line of the
source than a contaminant; the checking then runs against every peak
and every line.</p>
<p>The width behind "within half a peak width" is not each peak's own
fitted width but the width expected at its channel, read off a robust
straight line through all the fitted widths. A fit component that has
run away to absorb background comes back ten times too wide, and judged
against its own width it was being matched to a line ten or twenty keV
away; judged against the trend it is set aside once it exceeds three
times the expected width, as is any component whose fitted area is
negative. The status line counts these separately from peaks that were
simply not identified.</p>
<p>The winner is <b>refused</b>, and the dialog opens unassigned with
the reason on show, when fewer than four peaks are explained; when
fewer than two peaks beyond the two anchors confirm it, since the
anchors fit exactly by construction and are evidence of nothing; when
another pairing with a gain differing by more than five percent
explains as many peaks, because the evidence then does not choose
between two genuinely different calibrations; or when fewer than three
of the five strongest peaks were identified, because the strongest
peaks of a calibration spectrum <i>are</i> the source's lines, and an
assignment that leaves most of them unexplained has found a coincidence
among the weak ones. The last two guards were each added after watching
the matcher produce a confident, self-consistent and entirely wrong
calibration without them.</p>
<p>The peaks themselves come from a search on a lightly smoothed copy
of the counts: a candidate is anything standing more than the chosen
number of standard deviations above the continuum around it, measured
as its prominence divided by the square root of that continuum, so the
same sensitivity means the same thing in a weak spectrum and a strong
one. Peaks closer than three widths are fitted together, since a
doublet fitted as two single peaks gets both centroids wrong, and a
group too close to the spectrum edge for a background region beside it
is skipped rather than fitted against nothing.</p>

<h3>What the dialog remembers</h3>
<p>Assignments are stored per spectrum for the session. They are not
written to disk and do not travel with Save Fits. When the dialog
reopens, each row takes the stored energy whose channel is nearest,
provided it lies <b>within that peak&rsquo;s own FWHM</b>. A peak that
shifted by more than its width between fits is treated as a different
peak and opens blank. A peak whose FWHM the fit could not determine
falls back to a much narrower tolerance of one channel, so only a
centroid that barely moved is still treated as the same peak. Restoring
an energy onto the wrong peak would produce a calibration whose
coefficients and residuals both look reasonable while being wrong, which
is the failure this rule exists to prevent. No two rows can claim the
same stored assignment: the nearer one takes it.</p>

<h3>Reduced chi-squared, and when there is none</h3>
<p>The plot window divides each point&rsquo;s residual by that
point&rsquo;s energy uncertainty, obtained from its channel uncertainty
through the calibration&rsquo;s local slope dE/dch, squares and sums
them, and divides by n&minus;p. <i>p</i> is 2 for a line and 3 for a
quadratic.</p>
<p>It is reported as <b>undefined</b> in four cases, three of them
naming their reason. When any centroid uncertainty is unusable the
calibration was fitted <b>unweighted</b>, so the weights a chi-squared
would divide by are arbitrary and the value would look like a goodness
of fit while meaning nothing. When there are exactly as many points as
parameters there are <b>no degrees of freedom</b> and the fit passes
through them exactly. When a point sits on a quadratic&rsquo;s turning
point, dE/dch is zero there, so its channel uncertainty maps to no
energy uncertainty at all. The fourth case gives no reason at all: if
the computed value itself is not a finite number, a bare
<b>undefined</b> is reported on its own.</p>

<h3>The CalEnEff export</h3>
<p><b>Finish and save for CalEnEff...</b> writes seven
whitespace-separated columns with no header, all uncertainties
<b>absolute</b>: channel, its error, net area, its error, energy in keV,
relative intensity in percent, and its error.</p>
<p>The first four come from the fit and the last three from the source
file. <b>N is the net area</b>, background subtracted, because CalEnEff
computes efficiency as N divided by I: a gross area would fold the
background into the efficiency curve.</p>
<p>Source files carry intensities on no common scale, so the strongest
line in the loaded source is normalised to <b>100</b> and every other
line scaled by the same factor. Because efficiency is a ratio to I, a
constant factor rescales the whole curve without changing its shape, and
the relative uncertainties are unaffected.</p>
<p>A peak whose energy you typed by hand has no intensity, so it cannot
be exported and is <b>skipped</b>, with the count reported. Writing a
zero instead would be worse: CalEnEff rejects rows whose net-area and
intensity uncertainties are both zero, because the efficiency
uncertainty would come out zero.</p>

<h3>Reading the results columns</h3>
<p>The Fit Results panel shows <b>Position</b>, <b>Volume</b>,
<b>FWHM</b> and <b>chi^2</b>. Each column is sized to what is in it, so
none is padded while another is clipped.</p>
<p>All three numeric columns use the compact notation of nuclear data
tables, where the parenthesised digits are the uncertainty in the last
digits shown: <code>1332.49(12)</code> means 1332.49 with an
uncertainty of 0.12. A value with no usable uncertainty, such as a
position held fixed, is printed alone with no parentheses.</p>
<p><b>Position and FWHM are capped at two decimals.</b> Where the
uncertainty is too small to be written in them it is dropped rather
than shown as <code>(0)</code>, so a position known to better than
0.01 reads simply as <code>352.72</code>. Where it does fit, its digits
are read at that same place: <code>661.66(3)</code>. Volume is not
capped -- it counts events, where the uncertainty is routinely larger
than one and the parenthesised digits are the whole of it.</p>
<p>Neither the fit region nor a row number is a column. The region
has moved into the tooltip, along with the gross and net areas, and
hovering anywhere on a row shows it.</p>

<h2>Saving and reloading your work</h2>
<p><b>File &rarr; Save Fits...</b> writes every fit on the active
spectrum to a <code>.json</code> file, and <b>Load Fits...</b> brings
them back -- so an analysis survives closing the program.</p>
<p>Loading asks how you want them back:</p>
<ul>
<li><b>Restore</b> reproduces the saved numbers exactly as they were
reported when the file was written.</li>
<li><b>Re-run</b> fits again from the saved marks using the current
version of the program. Use this to bring an older analysis up to date --
v4.0.0 changed how peak areas and their uncertainties are computed, so a
fit saved before it will not agree with one made today.</li>
</ul>
<p>Fits saved against a different spectrum can be loaded too -- comparing
one run's fits against another's is a normal thing to want -- and you are
told when that is what is happening, because the marks land wherever
those channel numbers fall in the spectrum you are looking at.</p>
<p>Every file records the schema version it was written with, and the
program keeps a reader for each one, so files written by older versions
keep opening.</p>
<p><b>File &rarr; Reload Spectrum</b> (<kbd>F5</kbd>) re-reads the active
spectrum from disk while keeping its calibration, fits and marks -- for
watching a measurement that is still running. If the file has changed
length, the fits and marks are cleared and you are told: they are
anchored to channel numbers, which after a length change may no longer
point at the same thing.</p>

<h2>Exporting results</h2>
<p>"Export Fit..." and "Export All Fits..." write in whichever format the
filename's extension asks for:</p>
<ul>
<li><code>.txt</code> -- the readable report, with the full breakdown per
fit.</li>
<li><code>.csv</code> -- one row per peak, for a spreadsheet. An
uncertainty the fit could not determine is left as an empty cell rather
than the text "n/a", so a spreadsheet still treats the column as
numbers.</li>
<li><code>.tex</code> -- a LaTeX <code>tabular</code>, one row per peak,
ready to paste into a paper.</li>
</ul>
<p>When a calibration is active, positions and widths are exported in keV
and the column headers say so. Areas are never converted -- they are
counts, and counts have no energy equivalent.</p>

<h2>Why peaks in one fit share a width</h2>
<p>Every peak in a single fit is fitted with <i>one</i> width by
default. That is TV's behaviour, and it is deliberate: peaks close
enough in energy to be fitted together came from the same detector at
essentially the same resolution, so their widths genuinely should agree.
Tying them together also stabilises the fit, because a width is what a
multiplet is worst at determining -- given a blended hump, a fit with
free widths can trade width against amplitude between neighbours almost
without penalty.</p>
<p>Tick <b>Independent widths</b> when you have reason to expect
genuinely different widths -- a doublet where one component is a sum
peak, say, or peaks far enough apart that detector resolution really has
changed between them. Expect larger uncertainties on all of them, since
each width is then determined by its own peak alone.</p>

<h2>When an uncertainty reads "n/a"</h2>
<p>Occasionally a fit succeeds but one of its numbers is shown as
<code>&plusmn; n/a</code> in the Fit Results panel, in a row's tooltip,
or in an exported report. That is not a formatting glitch and not a
failed fit: it means the data did not constrain that particular
parameter, so no honest uncertainty can be quoted for it. The
<i>value</i> beside it is still the fitted result; only its error bar is
unknown.</p>
<p>The usual cause is the <b>left tail</b> on a peak that doesn't really
have one. With nothing in the data pulling the tail fraction away from
zero, the optimizer drives it to zero -- and once the tail contributes
nothing, the tail's decay length can take any value at all without
changing the fitted curve. It is genuinely undetermined, and the fit
says so rather than inventing a number. If you see this on the tail
parameters, the fit itself is fine; unchecking <b>Left tail</b> and
re-fitting removes them from the fit entirely.</p>
<p>Earlier versions rejected the whole fit in this situation, discarding
a perfectly good position, width and volume because a tail parameter
could not be pinned down. The fit is now reported, with "n/a" marking
only what is actually unknown. In an exported text report the same
quantity reads <code>&plusmn; n/a</code>; in the automatic
<code>_fits.jsonl</code> log it is written as JSON <code>null</code>, so
the file stays valid for other tools.</p>
<p>A fit is still refused outright when the <i>peaks themselves</i>
cannot be separated -- two peaks marked at (or very near) the same
position, for instance, where any split of the counts between them fits
the data equally well. There the message names the peaks concerned, and
the fix is to re-mark them: remove the duplicate, or widen the fit
region so the peaks are genuinely resolvable.</p>

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
these carries its own uncertainty, propagated from the counting
statistics of each individual channel.</p>
<p>For a spectrum read from a file those statistics are Poisson -- a
channel holding <i>N</i> counts has variance <i>N</i>. A spectrum this
program <i>derived</i> is different: a matrix cut, or an Add/Subtract
result, has a variance larger than its own counts (see "Why a cut's
uncertainty is not &radic;N" below), and Integration uses that real
variance instead. This is also why such a spectrum can be integrated
even where its counts have gone negative, while a file-backed one
cannot -- there is no Poisson variance to read off a negative count, but
a propagated one is perfectly well defined.</p>
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
the X axis of both. Activating the cut sums only the gated columns to
build a new spectrum on the <i>other</i> axis -- not shown
here.</figcaption>
</figure>
<p>A <b>2D coincidence matrix</b> (<b>File &gt; Open Matrix...</b>, see
the HowTo page's "15. Matrix analysis") records pairs of gamma rays
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

<h2>Gating on more than one peak</h2>
<p>You can mark more than one cut region. Every marked gate is summed
into the one cut spectrum, and the channel count used to weight the
background (<code>N<sub>cut</sub></code> above) totals across all of
them, so the subtraction stays correct however many you use.</p>
<p>This is what you want when a cascade has several members you trust:
gating on two or three of them at once collects the coincidences from
all of them, which builds statistics faster than gating on one. It is
only worthwhile while every gate really is the same cascade -- a gate
placed on a contaminant adds that contaminant's coincidences to the
result along with everything else.</p>
<p>Overlapping gates are counted once, not twice, so a channel covered
by two of them contributes what it actually holds.</p>

<h2>How the cut's background is weighted</h2>
<p>Written out, activating a cut computes, for every channel of the
<i>other</i> axis:</p>
<p style="text-align:center"><code>net[ch] = cut[ch] &minus; (N<sub>cut</sub> / N<sub>bg</sub>) &middot; &Sigma;bg[ch]</code></p>
<p>where <code>cut[ch]</code> is the raw sum over the gated band,
<code>&Sigma;bg[ch]</code> is the raw sum over every background band
pooled together, and the <b>N</b> terms are how many channels each of
those actually contributed. This is <b>gate-width weighting</b>: the
background bands are almost never the same width as the cut, so their
pooled counts have to be rescaled to the cut's own width before they can
be subtracted. Two background bands of 10 channels each stand in for 20
channels' worth of background; if the cut is 40 channels wide, that
pooled background is multiplied by 40/20 = 2 before subtraction.</p>
<p>Because it is a ratio of widths rather than an average, adding more
background regions does not weaken the subtraction -- it improves its
statistics, since a wider total background is measured from more counts
and the ratio compensates for the width exactly. Channels are counted
inclusively: a band marked from channel 100 to 150 is 51 channels, not
50.</p>
<p><b>Both halves of the ratio count the same channels.</b> A band
hanging over the edge of the matrix contributes only the part that is
really there -- to its counts <i>and</i> to <code>N<sub>bg</sub></code>
alike -- so the ratio still describes background per channel and the
subtraction stays correct. A band marked <i>completely</i> outside the
matrix contributes nothing at all, and if every background band is
outside then no subtraction is performed. Marking bands well inside the
data is still the clearer thing to do, but doing otherwise no longer
quietly weakens the result.</p>
<p><b>Overlapping bands are counted once.</b> If two background bands
overlap -- easily done by marking a wide one and then a narrower one
inside it -- the shared channels are summed once and counted once, rather
than being given double weight in the background estimate.</p>
<p>For background bands that sit fully inside the matrix and do not
overlap each other, this reproduces TV's own cut exactly, verified
against TV's source across a wide range of marked regions. The two
differences are the ones just described, and both are deliberate. TV
takes the ratio from where you marked rather than from what was summed,
so an overhanging band under-subtracts there; and TV counts overlapping
channels twice. The rule used here comes from HDTV, TV's ROOT-based
successor, which normalises by the channels actually summed and merges
overlapping bands before using them.</p>

<h2>Why a cut's uncertainty is not &radic;N</h2>
<p>A spectrum read from a file is a set of Poisson counts: a channel
holding <i>N</i> counts has variance <i>N</i>, and its uncertainty is
&radic;<i>N</i>. A spectrum this program <i>derived</i> is not, and
treating it as though it were would claim precision that was never
measured.</p>
<p>A background-subtracted cut is a difference of two measured sums, so
its variance is the sum of theirs:</p>
<p style="text-align:center"><code>var(net[ch]) = cut[ch] + (N<sub>cut</sub> / N<sub>bg</sub>)&sup2; &middot; &Sigma;bg[ch]</code></p>
<p>Note what this says. The variance is <i>larger</i> than the net counts
it accompanies -- subtracting a background removes counts but adds
uncertainty -- and it stays positive where the counts cancel to zero or
go negative, which is exactly where &radic;N has nothing to offer. Add and
Subtract Spectra follow the same rule, <code>var(A &plusmn; f&middot;B) =
var(A) + f&sup2;&middot;var(B)</code>: the factor is squared either way, so
two spectra that subtract to nothing still produce a result with a real
uncertainty attached.</p>
<p>This propagated variance travels with the spectrum. Fitting weights
each channel by it instead of by &radic;N, Integration uses it in place of
Poisson statistics, and the operations that change a spectrum carry it
along -- Multiply and Normalize scale it by the square of the factor,
and Rebin adds together the variances of the channels it merges, just as
it adds their counts.</p>
<p>The practical effect is that fitted uncertainties on a cut are
<i>larger</i> than a naive &radic;N treatment would report, and the
reduced chi-square is more honest. If you compare a fit on a cut against
one on a raw spectrum with similar counts, expect the cut's error bars to
be wider. That is the correct answer, not a defect.</p>
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
