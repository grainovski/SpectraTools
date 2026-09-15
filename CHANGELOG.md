# Changelog

All notable changes to SpectraTools are documented here, starting from
version 2.0.0. Dates are when the version was frozen and released, not
when individual pieces of work happened.

## [5.2.4] - 2026-09-15

### Fixed

- **The reduced chi-squared could report a wrong number instead of
  refusing.** The calculation checked that the channel uncertainties
  matched the number of points but never checked the energies. A shorter
  energy list made the sum stop early while the divisor still used the
  full count, so the result was not an error but a plausible value that
  was simply too small. Mismatched inputs are now refused, which is what
  the rest of the module already promised.

- **The CalEnEff export could write a row CalEnEff refuses to load.** The
  export documents that a row whose area error and intensity error are
  both zero is useless -- the efficiency would carry no uncertainty at
  all -- and gave that as its reason for skipping peaks with no matching
  source line. It never applied the rule to the rows it did write. Such
  rows are now skipped and counted like any other.

- **A peak with a non-finite area was exported as the text "nan".**
  CalEnEff reads the file with `np.loadtxt`, which turns that back into a
  number and computes with it. The export now refuses and names the peak
  and the field instead.

- **The calibration plot's residual strip drew no error bars.** The strip
  exists to show whether a point sits further off the line than its own
  uncertainty allows, and without bars a 2-sigma outlier and a 0.2-sigma
  one looked identical. Each point now carries its own uncertainty: the
  centroid uncertainty converted to keV through the calibration's slope,
  combined with the energy uncertainty the source file states -- the same
  combination the reduced chi-squared is weighted with.

- **The fitted coefficients could not be selected for copying.** The call
  meant to make them selectable handed the label its own current setting
  back, so it changed nothing.

### Internal

- The test suite no longer writes into the real per-user settings. Every
  test that opened a dialog was writing to the installed application's
  own stored settings, which put temporary test paths into the remembered
  folders and the recent-files list. On this machine that sent the
  installed application's **Load source...** dialog into a test directory
  instead of the calibration sources shipped with it.

  Fixing it also made the suite **eight times faster** -- a full run falls
  from 16m15s to 2m08s on the machine this was measured on, 11.4x on a
  controlled subset -- because each of those writes was committed to the
  Windows registry one at a time. That accounts for a long-standing and
  previously unexplained difference in suite duration between machines.

- Nine defects parked during the v5.0.0 build are closed, including the
  five user-visible ones above. A developer's absolute path is out of the
  shipped source, and three tests that could not fail now can.

## [5.2.3] - 2026-09-14

### Added

- **The calibration source files now ship with the program.** Both
  calibration dialogs ask for a `.sou` file naming a nuclide's known gamma
  lines, and until now the packages contained none: the feature was
  installed but arrived with no data to work on, so a new installation
  could not calibrate at all until the user found source files elsewhere.
  Ten are now included -- Am-241, Am-243, Ba-133, Co-56, Eu-152, Na-24,
  Ra-226, Se-75, Ta-182 and Y-88 -- rebuilt from evaluated ENSDF data in
  September 2026.

  Packaging them is only half of it, so **Load source...** now opens on
  them. Previously it opened wherever a spectrum was last loaded from,
  which is never where these live; bundled files nobody can find would
  have left the feature exactly as unusable as before.

### Fixed

- **Loading a source file no longer changes where Open Spectrum starts.**
  The two dialogs shared one remembered folder, so picking a `.sou` sent
  the next spectrum browse to the nuclide data instead of the spectra. The
  source folder is now remembered separately, and once you load a source
  of your own the dialog returns there rather than to the shipped files.

### Internal

- The `.sou` files have one tracked home, `sources/`, instead of being
  split between `tests/fixtures/` and ten untracked copies at the
  repository root. Untracked files do not travel in git, which had left
  one machine running the application against the pristine 2018 originals
  while its tests validated against the 2026 ENSDF rebuild, with nothing
  to indicate the two disagreed. The 2018 originals are tracked too, in
  `assets/`, and `*.sou` is pinned to LF so a byte comparison between a
  checkout and an archive answers the question actually being asked.

## [5.2.2] - 2026-09-14

### Fixed

- **Automatic calibration reported an efficiency scatter that included
  the points it had just rejected.** A peak whose area does not sit on
  the efficiency curve is flagged suspect, left out of the calibration
  fit and out of the CalEnEff export, and opens unticked in the dialog.
  The scatter printed beside it, however, was computed before suspects
  were known and never recomputed, so the number described a curve
  nothing downstream uses. It now describes the points that survive.

  This is not cosmetic. On a real Eu-152 spectrum a single peak in the
  crowded 674.64 / 678.62 / 688.67 keV triplet is flagged, correctly,
  and on its own carried the reported scatter from 0.045 to 0.078. That
  inflated figure was the whole evidence against fitting the automatic
  pass with the full peak shape; measured again with the suspect
  excluded it reads 0.049 against 0.045, while identifying 36 lines
  instead of 33 and cutting median reduced chi-squared from 6.6 to 1.6.
  The shape itself is unchanged and still off by default -- that choice
  now needs deciding on evidence rather than on an artifact.

  The hard rejection threshold still sees the uncleaned scatter, which
  is the conservative direction: a genuinely mispaired assignment should
  fail on all of its points, not on the ones left after a trim.

### Internal

- **The test suite could no longer finish on one of the development
  machines**, dying at 52% of a run with a Windows access violation and
  the process holding 3.4 GB. No leak in the application: a matrix
  panel releases its decoded matrix when it closes, and closing the main
  window closes its panels first. Nothing ran those handlers, because
  one Qt application serves the whole session and the tests never closed
  what they opened -- one file alone opens 75 panels and closes none,
  each keeping an 8192x8192 matrix reachable. Tests now close their
  windows at teardown; the suite completes, 1521 passed.

## [5.2.1] - 2026-09-05

### Fixed

- **A half-typed energy in the Calibrate dialog stopped the live plot
  following the table.** `export_points` runs from the table's
  `itemChanged` signal, on every keystroke, and raised on a cell holding
  something that is not a number. Raising there reports nothing: the
  traceback goes to a stderr the packaged executable does not have, and
  the plot silently stops updating until the typo is fixed. Rows that do
  not parse are now simply not plotted, and the typo is still named --
  the calibration itself refuses it, and the reason reaches the status
  line and the plot's summary, which is where it can be acted on.

- **The `_fits.jsonl` log recorded every automatically calibrated line
  twice.** An automatic run commits the identification pass, then
  replaces all of it with a refit of the same peaks. Both passes were
  logged, so the file the HowTo tells people to process with other tools
  carried two records of every line -- exactly the double count the
  in-memory fit list goes to some trouble to avoid. The identification
  pass is now logged only if the review is cancelled and those fits are
  the ones being kept; otherwise the refit is logged, once. Measured on
  a synthetic Eu-152 run: 66 records for 33 fits before, 33 after.

- **The refit judged a fit against a looser rule than the identification
  pass did**, though both describe themselves as applying "the same
  judgement". The refit measured each fitted width and shift against the
  trend through the SEARCH's widths, which are read off a smoothed copy
  and come out about 1.8 times wider -- 5.03 channels against 2.83 at
  channel 551 of a real Eu-152 spectrum. Both now use the trend through
  the fitted widths. Nothing wrong had reached an export, but the export
  is the more consequential of the two paths and had the weaker gate.

- **An unexpected failure during a run had nowhere to go.** Every
  failure the automatic pipeline is designed to have comes back as a
  reason; anything else raised out of a Qt slot, where it printed to a
  stderr the windowed executable does not have and left Run looking as
  though it had done nothing. It now reaches the dialog's status line,
  and the wait cursor is restored either way.

- **"None of the peaks found could be fitted" now says which way they
  went.** It reads as a solver failure and usually is not: on a crowded
  spectrum the fits succeed and are then dropped for landing somewhere
  other than the peak they were seeded on. The reason names the counts.

- **`restore` returned numpy integers as row numbers**, where Qt and
  `range()` comparisons want a Python `int`.

### Changed

- **The matcher's inlier search computes its candidate lines for the
  whole array at once** instead of one scalar `searchsorted` per peak
  inside a loop that runs thousands of times per match. Verified to
  produce byte-identical results -- same pairs, same gains, same
  residuals -- across the real Eu-152 spectrum and forty synthetic ones
  covering five nuclides. Worth about a tenth of the matching time,
  0.377 s to 0.334 s; the profiler had suggested rather more, which the
  wall clock did not bear out.

## [5.2.0] - 2026-09-05

### Added

- **A `Step` option in the Fit Parameters panel: the smoothed shelf that
  sits under every real peak.** Photons that scatter in the detector and
  carry part of their energy away leave a step, not a bump, under the
  photopeak -- anything from nothing to everything can be lost, so the
  deficit fills the whole range below. `gf3` and TV both model it and
  this app did not; `peak_fit.py` said so, "the step-background term is
  out of scope, per the design spec".

  Ported from `srcRW/gf3_subs.c`'s `eval()`, which computes
  `h * STEP * erfc(w) / 200` with STEP in percent of peak height. Here
  it is a fraction, so `amplitude * step * erfc(w) / 2`: the shelf is
  `step` of the peak's height far below it, half that at the centroid,
  and nothing above. It is **background, not peak**, which is `gf3`'s
  own reading -- its background-only branch evaluates this term and
  nothing else from the peak -- so it is excluded from the reported
  volume exactly as the fitted background line is.

  **The option only does its job alongside `Fit background`.** A step
  and a background line are the same thing to a fit that sees one peak:
  both raise the low side. They separate only when fitted together,
  which is how `gf3` does it, having no pre-subtracted background at
  all. With the line subtracted first it was fitted to data that already
  contained the shelf, so it has absorbed most of it -- on a synthetic
  peak with a 1.00% step, the step term recovers 1.03% when the
  background is fitted with it and only 0.34% when it is not.

  **What it is really for is stopping the tail running away.** The shelf
  is in the data whether or not it is modelled, and without a step the
  tail is the only component that can reach it -- and the tail, unlike
  the step, is counted in the volume. On a synthetic peak carrying both
  a 2% tail and a 3% step, the tail alone overstated the volume by
  31.5%; tail and step together gave +0.04%.

  The background-included `full_area` counts the step, since `area` does
  not: the step is background, and a term excluded from both would be
  counted nowhere. Sampled at the centroid, where the step is exactly
  half its asymptotic height, matching the convention that term already
  used for the background line.

### Changed

- **The calibration's reduced chi-squared now counts the source lines'
  own energy uncertainties**, added in quadrature with the centroid's.
  The residual has two sources and the test was only counting one, which
  made it far harsher than the data warranted: on a real Eu-152
  calibration it halved the reported value, 995 to 482. The lines it
  matters for are the ones quoted to a keV rather than a thousandth of
  one -- eu152's 1084.0(10) keV carries an uncertainty eight hundred
  times its centroid's. A hand-typed energy has no stated uncertainty
  and contributes nothing extra, which also means every calibration made
  without a source file reports exactly what it did before.

  A point sitting on a quadratic's turning point is no longer
  automatically undefined either: dE/dch is zero there, but the line's
  own uncertainty still weighs the point.

### Documentation

- **What a large reduced chi-squared means, in both places it is
  reported.** Neither page said, and the number is alarming without it.

  For a **fit**: about 1 is what a good fit gives and a weak peak does
  give it, while a strong peak routinely gives tens or hundreds and is
  not thereby a failed fit. A peak with a million counts is known to
  about a tenth of a percent per channel, and no peak-shape model
  describes a real detector that well, so a mismatch a weak peak hides
  inside its own noise stands out by tens of standard deviations in a
  strong one. On a real Eu-152 spectrum the ten weakest peaks gave a
  median of 1.3 and the ten strongest 66.8. Two consequences are worth
  knowing and are now stated: the reported uncertainties on a strong
  peak are optimistic, because they assume the model is right; and a
  large value on a WEAK peak is the opposite situation and worth
  chasing, since only a real problem can lift it above the noise.

  For a **calibration**: a large value does not mean the fit failed, it
  means a polynomial describes the points less well than the points are
  known. Centroids from strong peaks are good to a hundredth of a
  channel and no detector is exactly linear at that precision, so
  residuals of a tenth of a keV against uncertainties of a hundredth
  give a value in the hundreds while the calibration is perfectly
  usable. The page now says to judge it by the numbers that are in keV,
  and how to read the residual strip: a smooth sweep across zero is
  non-linearity, which the quadratic will partly absorb; scatter with no
  pattern means the fit is already as good as the shape allows.

- **How the three shape options work together.** `Left tail`, `Step` and
  `Fit background` are the full Hypermet description and are what brings
  a strong peak near 1. Singly they can each make matters worse. On the
  real Eu-152 spectrum the median reduced chi-squared over 66 peaks was
  5.83 as fitted by default and 1.54 with all three, and over the strong
  peaks alone 46.1 against 2.0. All three stay OFF by default: turning
  them on changes every reported volume, which is the user's decision.

## [5.1.0] - 2026-09-05

### Added

- **Automatic calibration.** `Operations > Automatic Calibration...`
  (`Ctrl+Shift+L`) calibrates a spectrum of a known source with no fits
  and no calibration to start from. It finds the peaks, fits them, works
  out which line of the `.sou` source each one is, and opens the result
  in the Calibrate from Fitted Peaks dialog with the energies filled in
  and the live plot showing, so a misidentified line is a residual you
  can see and untick before anything is applied.

  Identifying a peak needs a calibration and the calibration needs the
  identifications. The matcher breaks that circle the way a star tracker
  does: every pairing of the strongest peaks with the strongest source
  lines fixes a provisional straight line, each line predicts where
  every other peak should fall, and the pairing that explains the most
  peaks wins. It is confident or it refuses -- when fewer than four
  peaks are explained, when a rival calibration with a clearly different
  gain explains as many, or when most of the strongest peaks stay
  unidentified. A refusal is not a failure of the feature: the peaks are
  still fitted and the dialog still opens, unassigned, with the reason
  in its status line, so two energies typed by hand and *Suggest
  remaining* finish the job as before. On synthetic spectra built from
  the sample sources it recovered every calibration tried -- Eu-152,
  Ba-133, Ra-226 with its daughters, Co-56, Ta-182 -- and refused every
  wrong source it was offered.

  The tolerance for "explained" is half a peak width, but the width of
  the trend through all the fitted widths rather than each peak's own.
  Judged against its own width, a fit component that had run away to a
  hundred channels bought itself a tolerance of twenty keV and was
  matched to a line that far away -- one to three such assignments per
  spectrum, before this was found. Such components are now set aside
  and counted in the status line.

  Each peak is fitted on its own, with its own markers. Nearby peaks
  were first fitted together as a multiplet, and on a real Eu-152
  spectrum that chained the strongest line in it -- 121.78 keV -- to a
  broad Compton feature eight channels away and lost it entirely.
  Candidates far wider than the spectrum's own peaks are now set aside
  before fitting, the fit window stops halfway to any neighbour, and the
  two background windows are searched for on flat stretches clear of
  every found peak, relaxing only in a spectrum too crowded to offer one.

  Three things about that had to be measured on a real spectrum rather
  than reasoned about, because between them they cost five lines of a
  real Eu-152 run: 344.28, 367.79, 411.12, 416.02 and 443.97 keV, the
  first of them the cleanest strong line in the spectrum.

  A broad feature was kept clear of by its own measured width, and a
  Compton structure at channel 643 measured 56 channels wide. That
  blanked out 169 channels: six peaks below it could find no background
  on their right and three above it none on their left, and nine
  photopeaks went unfitted. A broad feature is now kept clear of by the
  width a photopeak has at that channel, and the flatness test -- which
  is what actually detects a window sitting on structure -- decides the
  rest.

  The fit window had a floor under its clipping, measured in a width
  read off a smoothed copy that comes out up to twice the fitted width.
  A close neighbour therefore did not narrow the window but widened it
  back over the neighbour, and the fit ran there: five peaks were lost
  that way, 416.02 keV among them, eight channels from a line twenty
  times stronger. The clip now always wins, and a window clipped to
  nothing gives a fit that fails and is dropped rather than a confident
  wrong answer.

  And a fit that ends up somewhere other than the peak it started from,
  or on a peak another fit already caught, is now dropped instead of
  committed. Ten of that spectrum's 67 fits did one or the other --
  components up to eighteen times the width of its real peaks, three of
  them with a negative area -- and kept, they are peaks in Fit Results
  that are not in the spectrum, which the matcher is then free to name.

  Together: peaks skipped for want of a clear background fell from 11 to
  1, lines identified from 26 to 33, and lines exported for CalEnEff
  from 26 to 33, at the same quality -- 0.11 keV rms about the fitted
  line against 0.12 before, and areas scattering by a factor of 1.11
  about the efficiency curve against 1.10 before.

  The areas must trace one efficiency curve as well. Area over intensity
  is the detector's efficiency times a constant, and an HPGe efficiency
  curve is smooth; under a wrong pairing the ratios scatter by orders of
  magnitude -- a factor 1.3 against 5.7 on the real Eu-152 spectrum. The
  winner is refused if they scatter too widely, a tie between two
  pairings is broken by it, and a point more than three scatters off the
  curve opens UNTICKED: assigned and hollow on the plot, but out of the
  fit until you decide.

  On OK, every source line that is visibly present is refitted, one at a
  time, and those fits replace the identification pass's -- one clean
  area per line, which is the input CalEnEff needs. The plot opens on
  the refitted points. Lines outside the spectrum, with no visible peak,
  or blended into a stronger neighbour are counted, not fitted.

  Fits already on the spectrum are always kept; the automatic fits are
  appended after them and the status bar says how many were there. The
  peak search's sensitivity, in standard deviations above the local
  continuum, has a sensible default and a control in the dialog whose
  count of peaks found updates as it is changed.

- **Click a point on the calibration plot to find out which one it is.**
  Clicking a point on the curve or on the residual strip rings it on
  both, names it under the plot -- channel, energy, and its residual --
  and selects its row in the Calibrate dialog's table. The residual
  strip is where a misidentified line shows up, and until now it did not
  say which line each point was, so the outlier furthest from zero had
  to be found in the table by counting. Clicking empty space clears the
  ring and the selection.

- **A point unticked in the calibration dialog is now left out of the
  CalEnEff refit too, and the residual strip is scaled to the points the
  fit was made through.** In an automatic run a point is unticked
  because its area does not sit on the efficiency curve the others
  trace, and the refit exists to measure areas for that curve; refitting
  it anyway fed CalEnEff the one number the run had already judged
  wrong. It is counted in the status line rather than silently dropped,
  and it still claims its own peak, so a weaker line blended into it
  cannot inherit the peak's whole area.

  The residual strip used to scale to every point drawn, excluded ones
  included. A point excluded for sitting far off the line therefore set
  the scale and squashed every remaining residual onto zero -- the one
  thing the strip exists to show. The excluded point is still drawn and
  still named when clicked; it is simply allowed to fall outside the
  strip's view.

### Changed

- **The calibration plot follows the kind of fit and stays open.**
  Switching linear to quadratic now redraws the live plot in place. It
  used to do nothing until OK, which applied the fit and closed every
  window, so the only way to see the quadratic drawn was to lose the
  dialog. And when the points do not make a fit at the moment -- two
  points under a quadratic, a typo mid-edit -- the plot shows them
  without a curve and says why, instead of closing.

### Fixed

- **A remembered energy sitting exactly on a peak's centroid could still
  be dropped as ambiguous.** The rule that refuses a coin flip between
  two nearby peaks now recognises a distance of exactly zero as the
  stored peak itself. Without this, a peak fitted twice -- once by hand
  and once by the automatic pass -- left both copies blank.

## [5.0.2] - 2026-09-05

### Added

- **Each peak in the calibration dialog now has a tick, and unticking it
  leaves that point out of the fit.** The energy you typed is kept; the
  tick only decides whether the calibration is fitted through that
  point. It is what the residual strip has been asking for since the plot
  arrived in 5.0.0: when one point sits visibly off the line, untick it
  and watch the fit and its residuals redraw without it.

  The excluded point stays on the plot as a hollow marker, with its
  residual against the new fit still shown, so what you rejected remains
  visible and reversible rather than disappearing.

  **Suggest remaining** ignores unticked points as well. Suggesting is
  itself an energy fit, so allowing a point you have called suspect to
  anchor it would push the error into every line proposed from it.

  The tick is remembered with the energies, per spectrum, so a point
  judged an outlier does not quietly rejoin the fit after a refit or a
  reopened dialog. **Clear** puts every tick back along with emptying the
  table.

  One deliberate exception: an excluded point is **still written to the
  CalEnEff export**. The tick expresses a doubt about the energy fit,
  which leaves the peak's area and the intensity matched to it
  untouched.

### Notes

- The `.sou` source files used here during development were rebuilt from
  the evaluated ENSDF decay data that NNDC's NuDat serves. They are not
  part of the application and are not installed with it, so nothing in
  this release changes for anyone using their own -- but the errors found
  are worth checking for in any set of the same vintage. In ours:
  `eu152` was missing the 1085.837 keV line entirely and had 964.057 keV
  recorded 0.074 keV too high; `ba133` had 223.237 keV out by 0.121 keV;
  `co56`'s lines above 3 MeV were low by up to 0.09 keV and it carried no
  511 keV annihilation peak despite a 19.6% positron branch; `ta182`'s
  intensities were uniformly about 7% off relative to its strongest line.
  A new `ra226` set covers Ra-226 in secular equilibrium with its Pb-214
  and Bi-214 daughters, 45 lines from 53 to 2448 keV, whose strongest
  line is the 609.312 keV one CalEnEff's own 226Ra sample normalises
  to 100.

## [5.0.1] - 2026-09-04

### Fixed

- **A remembered energy could land on the wrong peak.** When the
  calibration dialog reopened, each stored assignment was matched to a
  peak by walking the possible pairings nearest-first and taking each as
  it came. In a doublet that is the wrong order to decide in: the first
  pairing considered claims its line, and the peak next to it is left
  with whatever remains, even when swapping the two would have moved
  both less. The pairing is now the one that minimises total
  displacement across all the peaks at once.

  A match whose runner-up is nearly as good is now left blank instead of
  guessed. The two outcomes are not equally bad -- a blank costs one
  retyped energy, while a wrong energy produces a calibration that
  passes through every assigned point, so its coefficients and residuals
  both look reasonable and nothing says otherwise.

  Measured over 154,000 generated peaks with a different width on each,
  as real fits produce: peaks separated by 1.5 x FWHM or more are now
  never restored wrongly, where the old pass erred on some. Closer than
  that the result is improved rather than guaranteed -- 16,453
  misassignments became 1,523 across all separations -- because below
  one width the two peaks are not separable and no rule can settle which
  line belongs to which.

### Internal

- The channel/keV conversion methods and the calibration dialog opener
  were written twice, once in the main window and once in the matrix
  panel, character for character. They now come from one
  `CalibrationViewMixin`, alongside the `GoToMixin` added for the same
  reason in 4.1.1. The two windows keep their own
  `_apply_calibration_change`, which genuinely differs between them.
  The copies had not drifted yet; this same pair had already done so
  once, which is what the 4.1.0 audit found.

- Removed two `MatrixCutState` methods that nothing called and that each
  cleared half of what `reset()` clears -- a field added to `reset()`
  would not have reached them.

- Dropped an unused import and a comprehension that rebuilt each tuple
  unchanged before passing it on.

## [5.0.0] - 2026-09-04

### Added

- **Peak energies can be assigned from a `.sou` source file.** A `.sou`
  file lists the known gamma lines of one calibration nuclide as four
  numbers per line -- energy, energy error, intensity, intensity error --
  and loading one lets a fitted peak be matched to a line instead of
  having its energy typed by hand. The reader is deliberately strict:
  any line that is not four finite numbers is refused, naming the line,
  rather than skipped, because a silently dropped line would leave you
  calibrating against a source quietly missing a peak. The intensity
  column is shown but never relied on -- the sample files normalise
  their strongest line to anything from 1000 to 100000, so there is no
  common scale to trust.

- **A calibration plot window.** Calibrating now opens a window showing
  the fit rather than reporting coefficients in a status line: the
  assigned points with their channel error bars, the fitted curve drawn
  across the whole channel range so extrapolation beyond the assigned
  points is visible, a residual strip beneath it, the coefficients with
  uncertainties, and a reduced chi-squared. Coefficients look equally
  plausible whether or not one point was misidentified; the residual
  strip is where that shows. The window opens as soon as the
  assignments can be fitted and redraws after every change to them --
  an energy typed, a source loaded, a suggestion taken -- so a
  misidentified line is visible while it can still be corrected, rather
  than only once the calibration has been applied.

- **Reduced chi-squared for a calibration.** Channel uncertainties are
  converted to energy through the calibration's own local slope. It is
  reported only when it means something: the underlying fit falls back
  to unweighted if *any* centroid error is unusable, and a chi-squared
  computed from the usable subset would be judging a fit that was never
  performed. In that case, and three others, the window says why instead
  of showing a number.

- **Export to CalEnEff.** A `Finish` button writes the seven
  whitespace-separated columns CalEnEff reads -- channel, channel error,
  net area, area error, energy, intensity in percent, intensity error --
  with absolute uncertainties, as its own source documents. The area
  written is the net area, since CalEnEff computes efficiency as N / I,
  and a gross area would fold the background into the efficiency curve.
  Because `.sou` intensities have no common scale, the strongest line in
  the loaded file is normalised to 100, matching CalEnEff's own sample.
  A constant factor rescales the efficiency curve without changing its
  shape.

- **Energy assignments are remembered per spectrum.** They survive
  closing and reopening the calibration dialog, and are restored after a
  refit by matching each assignment to a peak within that peak's own
  FWHM. A `Clear` button forgets them -- along with the loaded source
  file -- without disturbing the calibration currently in effect.

### Changed

- **The Fit Results panel has been re-columned** to `Position`,
  `Volume`, `FWHM` and `chi^2`. The fit region has moved into a tooltip
  that now covers the whole row, and the row-number column is gone
  entirely -- both were spending width to show something you rarely
  read. Each remaining column is sized to its own contents rather than
  to an equal share of the dock.

- **Values carry their uncertainty in compact notation**, as nuclear
  data tables quote them: `1332.49(12)` is 1332.49 with an uncertainty
  of 0.12, the parenthesised digits being in units of the last decimal
  shown. A parameter held fixed, or one the data does not constrain,
  shows no parenthesis rather than a misleading `(0)` or `(nan)`.
  Position and FWHM are capped at two decimals -- an uncertainty too
  small to write in them is dropped rather than rendered as `(0)`, so a
  well-determined line reads as `352.72` instead of `352.7217(14)`.
  Volume is not capped: it counts events, where the uncertainty is
  routinely larger than one and the parenthesised digits are all of it.

## [4.2.2] - 2026-08-27

### Added

- **A warning when a calibration folds back on itself.** A quadratic
  calibration reverses direction at one channel, and if that channel
  falls inside the loaded spectrum, two different channels end up
  sharing the same energy -- so converting an energy back to a channel
  has two answers, and Go To or an energy typed into the Fit Parameters
  panel may resolve to the wrong one. Nothing else about such a
  calibration looks wrong: it still passes through every assigned point,
  so the coefficients and residuals can both look reasonable. Setting
  one now says so, and suggests the usual causes (a mistyped energy, a
  peak assigned to the wrong line, or points that simply do not support
  a quadratic). A calibration that only bends beyond the last channel is
  ordinary and stays silent, as does a straight line.

### Internal

- The `.mtx` decode-speed guard no longer times one decoder to
  completion before the other. It alternates them, swapping the order
  each round, so background load reaches both sides equally. Timing them
  in sequence put the two measurements minutes apart inside a full suite
  run, and load covering only the first window was enough to fail the
  test with nothing wrong in the code -- which happened once during an
  overnight run, reporting a ratio twice its normal value while three
  immediate re-runs on an idle machine passed. The sample count also
  rose from three to six per side: the control test compares a decoder
  against itself with only 5% of headroom, and three samples were too
  few for a best-of estimate to stay inside that. Both now hold under
  six competing CPU-bound processes, and the guard still fails as it
  should when the optimization is reverted.

## [4.2.1] - 2026-08-26

Support for a new spectrum format, plus a Windows packaging and
documentation overhaul.

### Added

- **`.lzs` spectrum files (labZY / nanoMCA) can now be opened**, from
  **File > Open...** like any other spectrum. The histogram is read
  together with the instrument's own two-point energy calibration,
  which is applied automatically when the file marks it enabled and no
  calibration is already active -- the same rule an N42 file's
  calibration follows. Acquisition times, hardware registers and
  firmware details are ignored, and the format is read-only.

  Two details of the format are worth knowing, because both would
  quietly corrupt a reading if handled naively. A file's `softsize`
  field is **not** the number of valid channels: one of the sample
  files declares 8192 against a hardware size of 16384 while holding
  99.97% of its counts above channel 8192, so the full histogram is
  always read. And the files are not strictly valid XML -- every one
  written by the current firmware closes a tag in its status section
  with a mismatched name, which a conventional XML reader rejects
  outright. Each section is therefore read independently, so a defect
  in a part of the file this app ignores cannot stop the spectrum
  loading.

- **Windows install instructions** (`packaging/windows/INSTALL.md`),
  shipped with each release. They explain the "unknown publisher"
  warning Windows shows — what it does and does not mean, and how to get
  past it — how to install without an administrator prompt, and why the
  very first launch is slower than every launch after it (Windows
  Defender's initial scan and a one-time font-cache build, neither of
  which recurs).
- **A `SHA256SUMS.txt` with every release**, so a download can be checked
  against the file that was actually published. Since the installer is
  not code-signed, this is the meaningful integrity check — and unlike a
  signature prompt, it detects a corrupted or altered download.

### Changed

- **The Windows build ships as a directory rather than a single-file
  executable.** The old `--onefile` build unpacked its entire ~108 MB
  bundle into a temporary folder on *every* launch, with nothing cached
  between runs — a cost the Linux packages never paid, because they were
  already built this way. Repeat launches are now quicker, and the app no
  longer looks like a self-extracting archive to antivirus software,
  which is a shape scanners treat with suspicion. The installer hides the
  difference: it installs and uninstalls exactly as before.
- **The Windows installer download is substantially smaller**, now that
  it compresses its contents. The previous single-file build was already
  compressed internally, so compressing the installer too would have
  gained nothing; a directory of loose runtime files compresses well.

### Internal

- The Windows bundle is no longer UPX-packed. Packing every executable and
  DLL re-creates the self-extracting shape that antivirus heuristics treat
  with suspicion -- which is the same shape leaving the single-file build
  had just removed, and it matters for an unsigned application, where
  scanners have little else to judge by. The setting was inert (UPX is not
  installed on the build machine, so it was silently skipped and no
  released binary was ever packed), but a setting that only waits for a
  tool to appear is a trap rather than a preference.

## [4.2.0] - 2026-08-26

Everything in this release comes out of a full audit of the codebase —
13 findings across correctness, stability and polish, each fixed with its
own regression test, plus randomised sweeps (including the first-ever
sweep of the fitted-background mode: 200 cases, no violations) and a
leak-regression check confirming repeated matrix-panel open/close stays
flat.

### Fixed

- **Multiply and Normalize now propagate uncertainties correctly for
  file-loaded spectra.** Scaling a spectrum's counts by a factor *f* left
  later fits and integrations reading the scaled counts as plain Poisson
  data, which misstates the variance by exactly *f* — every error bar and
  reduced χ² on a multiplied or normalized spectrum was off by √f, with no
  warning. The uncertainty is now derived from the pre-scale counts and
  scaled with the data. Multiplying by exactly 1 changes nothing, as it
  should.
- **Double-clicking a stored fit restores its "Fit background" mode.** It
  used to restore the other two mode flags but not this one, so the
  parameters panel came back without its background rows and a refit
  silently ran in the wrong background mode — producing different areas
  and uncertainties than the fit being revisited.
- **A ROOT histogram bin holding a value too large for a 64-bit count is
  refused with a clear message** instead of silently wrapping to a
  meaningless number on import. (Non-finite bins were already refused.)
- Three crashes from ordinary input are gone: picking a binary file in
  **Load Fits…**, typing `nan`/`inf` — or, with a quadratic calibration
  active, an energy the calibration cannot reach — into the **Fit
  Parameters** panel, and opening an N42 file whose calibration reference
  contains an apostrophe or bracket (legal XML).
- **The heatmap's axes now tick in real channel numbers.** On large
  matrices the display is downsampled, and the axis labels used to show
  the downsampled indices while saying "channel" — off by 8× on an
  8192-channel matrix.
- The Energy Assignment dialog names the offending row when an energy is
  entered as `nan`/`inf`, instead of failing later with a message about
  the whole calibration.
- Activating the same multi-gate cut always produces the same auto-log
  file name regardless of the order the gates were marked in; previously
  each marking order got its own file, splitting one cut's fit log.

### Added

- **Log scale Y in the matrix panel**, on the same Ctrl+Y as the main
  window — the one view control the projection view was missing.
- **Saving a derived spectrum now says what was saved.** No spectrum file
  format stores propagated uncertainties, so saving a matrix cut or an
  Add/Subtract result keeps only the counts — and reloading the file
  re-assumes Poisson uncertainties. A note after such a save says so;
  saving an ordinary file-loaded spectrum is unchanged.

### Internal

- The decoded-matrix disk cache marks an entry as recently used on every
  hit, so its least-recently-used eviction works even where the
  filesystem never updates access times (common on Windows/NTFS).
- LaTeX export escaping is a single pass and can no longer mangle
  backslashes; the fit-plausibility, calibration round-trip, codec
  round-trip and integration self-consistency sweeps were all extended
  and ran clean.

## [4.1.2] - 2026-08-24

### Changed

- Reading a `.mtx` matrix file is a few percent faster. `lc_codec.decode_row`
  built its output through a pre-sized list, on the assumption that this
  avoided per-`append()` growth overhead; a direct A/B — same code, same
  rows, only that one difference — measured it 5–6% *slower* than plain
  `append()`/`extend()` on Python 3.13, so it was removed. Decoded values
  are unchanged, verified byte-for-byte across all 16384 rows of both real
  matrix fixtures and on every decode error path.

### Internal

- The `.mtx` decode-speed regression test no longer asserts an absolute
  wall-clock bound. That bound was calibrated on one machine, which made
  the test a property of the hardware: it failed outright on slower
  hardware with no regression present, and cleared the bar by only ~9%
  even where it passed. It now times the decoder against a frozen copy of
  the pre-optimization implementation in the same process and asserts a
  ratio, which holds on any machine, alongside a control test proving the
  guard can still fail.

## [4.1.1] - 2026-08-22

### Added

- Knowledge Database sections on **how the background line is determined**,
  what <kbd>Ctrl+B</kbd> previews (and that it is unaffected by the "Fit
  background" checkbox — it always shows the same two-point line), and what
  a committed fit draws: the dashed line, the shaded uncertainty band, the
  total model curve and each peak's contribution. The band appears in both
  fitting modes; what differs is whether its width comes from the two region
  means or from the joint fit's own covariance.
- **Go To** (<kbd>Ctrl+G</kbd>, or View &gt; Go To...) jumps the view to one
  place in the spectrum. Type an energy in keV when a calibration is active
  or a channel when one is not — the dialog asks for whichever the x-axis is
  showing, and states the range the spectrum covers. The view centres on the
  target in a 100-channel window with the Y axis rescaled to what is on
  screen, and a dotted marker is left at the spot. Near either end of the
  spectrum the window slides inward rather than being halved, so a line at
  channel 5 still gets context around it. The mark is stored as a channel,
  so toggling the calibration moves it to the matching energy instead of
  stranding it. <kbd>Ctrl+C</kbd> clears it with the other marks. Works the
  same way in a matrix panel's projection.

### Changed

- **Log scale Y moved from <kbd>Ctrl+G</kbd> to <kbd>Ctrl+Y</kbd>**, freeing
  the key every editor and browser uses for "go to". Ctrl+Y is the better
  mnemonic for it in any case. Calibration keeps <kbd>Ctrl+L</kbd>.

### Fixed

- **The dialog flicker returned on AlmaLinux, and the Linux packages were
  the reason.** v4.1.0 stopped forcing the X11 platform so Qt could use
  Wayland, which is what removed the flicker — but neither the .rpm nor
  the .deb declared the Wayland client libraries, because those dependency
  lists dated from when the app always forced X11. AlmaLinux 10 ships
  neither `libwayland-cursor` nor `libwayland-egl` by default, so Qt found
  its own Wayland plugin, failed to load it, silently fell back to
  XWayland, and the flicker came back. Ubuntu happened to have both
  preinstalled, which is why only one distro showed it. Both packages now
  require them.
- **A shortcut sharing a letter with a marking key no longer arms that
  marking.** <kbd>Ctrl+B</kbd> (Preview Background Fit) armed background
  marking, <kbd>Ctrl+R</kbd> (Rebin) armed the fit region, and in a matrix
  panel <kbd>Ctrl+C</kbd> (Clear) armed cut marking — the key filter only
  ever looked at which letter arrived, never at the modifiers. It mattered
  more than a stray flag suggests: the shortcut usually opens a dialog,
  which takes focus, so the key release that would disarm it never arrived
  and the next ordinary click placed a mark nobody asked for. Found while
  adding Go To, which would have been a fourth instance.
- Two HowTo cross-references pointed at "11. Matrix analysis" when that
  section had drifted to 13. The section numbering and every numbered
  cross-reference are now checked by a test.
- The Knowledge Database claimed the background's uncertainty band is
  narrowest **at** the two background regions. It is narrowest *between*
  them — averaging two independent measurements beats either alone, so for
  regions known to ±2.1 and ±2.4 counts the band closes to about ±1.6
  between them.

## [4.1.0] - 2026-08-19

A full audit of the code for scientific correctness, stability and
performance, and the fixes for everything it found. **Several reported
uncertainties change**, and one of them changes by a large factor. None of
the reported *values* change -- areas, positions and widths are unaffected
except where noted under Changed.

### Fixed

- **A multiplet's total area uncertainty was overstated, by up to a factor
  of ten.** The net total added each peak's uncertainty in quadrature,
  which is only valid if the peaks are independent. Overlapping peaks are
  strongly anti-correlated: the data pins down how many counts the group
  holds far better than how they divide between the peaks. The total is
  now propagated through the fit's whole covariance matrix in one step.
  Validated against a 400-realisation Monte Carlo -- refitting the same
  spectrum under fresh noise and measuring the actual spread of the
  fitted total. For a doublet one sigma apart the old figure was 2324
  against a true spread of 233; the new one is 238. Single-peak fits and
  well-separated peaks are unaffected.
- **A fit's region gross-area uncertainty ignored a supplied variance**,
  reporting the Poisson square root even for a matrix cut, whose variance
  exceeds its own counts. Measured a factor of two low, and it left two
  numbers in the same results panel disagreeing about how well one
  spectrum was known.
- **Multiply and Normalize left a derived spectrum's variance untouched**,
  so every later fit on it reported uncertainties too small by the scale
  factor -- silently. Scaling counts by f scales their variance by f squared.
- **Rebin made a derived spectrum impossible to fit.** Its variance kept
  the pre-rebin channel count, which the fitter then rejected outright.
  The variance is now rebinned with the counts.
- **A peak fitted with "Left tail" on could report an absurd area** -- 6e18
  counts for a peak of amplitude 3800. The tail decay length was bounded
  below but not above, so on data with no real tail the tail fraction went
  to nearly zero, the tail stopped contributing to the model, and the decay
  length was left unconstrained and wandered as far as 7e18. The fitted
  curve stayed perfectly good, which is what made this easy to miss, but
  the area formula's tail term grows with that length and the product is
  enormous even for a negligible tail fraction. In a randomised sweep, 16
  of about 150 tailed fits reported an area more than a hundred times their
  own Gaussian core; now none do. The decay length is bounded at 20 times
  the widest peak width, which is permissive enough not to touch a genuine
  tail -- fitting data generated *from* the tailed shape recovers a true
  decay length of 10 sigma as 10.1, with the area correct to better than
  1%. The trade-off: the unbounded fit reached a marginally lower
  chi-square in about 12% of tailed fits (worst case, 0.05 in reduced
  chi-square), because a decay length of 1e18 acts as a flat pedestal that
  absorbs background mismatch. That is the background's job, not the
  tail's.

- **Dialogs no longer vanish and reappear on WSL.** Opening a file or
  matrix dialog made it appear, disappear after about a second, and come
  back a few seconds later. The cause was this app's own workaround: it
  forced the X11 platform under WSL to dodge an older WSLg bug that
  rendered the main window at zero size, and going through XWayland is
  where the flicker came from. It now runs on Wayland, which removes that
  layer, and keeps X11 as an automatic fallback for compositors that still
  need it — detected by measuring the window at startup rather than by
  guessing at versions. The flicker leaves no trace in the X protocol (one
  map, one expose, one unmap, no repeated exposes), which is why it was
  previously documented as unfixable.
- The matrix panel drew its projection with a thinner line (0.8) than the
  main window draws a spectrum (1.5), so the same data looked fainter in
  one view than the other.

### Added

- **Integration can use a derived spectrum's propagated variance**, as
  fitting already did, so the two stop disagreeing about the same data. It
  also means a cut can now be integrated where its counts have gone
  negative -- there is no Poisson variance to read off a negative count,
  but a propagated one is perfectly well defined. Spectra read from a file
  are unaffected and keep TV's exact behaviour.
- **Calibrating from fitted peaks is weighted by each centroid's own
  uncertainty.** The feature's whole claim is that fitted centroids beat a
  cursor position, and an unweighted fit threw that away -- the weakest
  line you assigned pulled as hard as the strongest. The assign dialog
  gained a "± ch" column showing each peak's uncertainty, and the status
  bar now reports the fitted slope with its own uncertainty when more than
  the minimum number of points is assigned.
- Knowledge Database sections on why a derived spectrum's uncertainty is
  not the square root of its counts, and on how the net total's
  uncertainty is propagated.

### Internal

- The drag-pan rule is shared between the main window and the matrix panel
  rather than duplicated, so the panel's second marking controller cannot
  be forgotten in one copy.
- The two drawing hot paths use the cached channel axis that already
  existed for them.

## [4.0.1] - 2026-08-18

Two interaction fixes.

### Fixed

- **The selected entry in the ROOT object list is now visible.** Opening a
  ROOT file preselects the first histogram, and pressing OK always loaded
  it -- but nothing looked selected. Two causes: the dark theme styled
  tables and lists but never tree views, which is what that dialog uses,
  so it fell back to unstyled defaults; and the list did not hold focus,
  and an unfocused list draws its selection in a pale grey in either
  theme.

### Changed

- **Dragging with the left mouse button now slides the spectrum
  sideways**, the same gesture the right button already had: the X axis
  keeps its width and the Y axis rescales to fit what is on screen. Works
  in the main window and in a matrix panel's projection.

  Holding a marking key still marks and never pans -- <kbd>B</kbd>,
  <kbd>R</kbd> or <kbd>P</kbd> in the main window, <kbd>C</kbd> or
  <kbd>G</kbd> in a matrix panel -- so marking is unaffected. Left-drag
  also stands aside while the plot toolbar's own Pan or Zoom tool is
  active. Right-drag is unchanged and still pans even with a marking key
  held.

## [4.0.0] - 2026-08-18

ROOT file support, and a set of corrections to how peak areas and
uncertainties are computed. Both came out of a study of HDTV, the
ROOT-based successor to the TV program this app ports.

**Read this before comparing results against 3.1.4.** Several numbers
this release reports are different, and in every case the old one was
wrong. If you have results from an earlier version that matter, re-run
those fits rather than assuming they still agree.

### Numbers that changed

- **Peak volume now includes the tail.** It was the integral of the
  Gaussian core alone, so any fit with a left tail reported an area
  short by whatever the tail held -- 5% for a small tail, 14% at a
  typical one, and over 40% for a long tail. Volumes with no tail are
  unaffected.
- **Volume uncertainty now accounts for correlations.** It previously
  added relative errors in quadrature, which assumes amplitude and width
  are independent. In a peak fit they are strongly anti-correlated, so
  the reported uncertainties were too large -- by about 14% in a typical
  case.
- **The background now carries its own uncertainty.** It was treated as
  exact, which it is not: both ends of the background line are averages
  of real counts. Background-included ("full") areas therefore have
  slightly larger, and honest, uncertainties.
- **Peak widths are seeded from the region's integral.** The old seed
  was systematically about half the true width. Fits that previously
  failed to converge on crowded multiplets now generally succeed --
  measured over 1200 randomised hard cases, failures fell from 41 to 12.
  Well-behaved fits land in the same place as before.
- **Matrix cuts weight the background by the channels actually summed.**
  A background band left hanging over the edge of a matrix used to
  under-subtract, leaving up to a quarter of the gross counts behind.
  Overlapping background bands also used to count their shared channels
  twice.
- **Rebinning now anchors the calibration on the centre of each group of
  channels** rather than its first channel, correcting an energy offset
  of half a channel at factor 2 and 3.5 channels at factor 8.

### Added

- **ROOT files.** File > Open ROOT File... reads 1D histograms as spectra
  and 2D histograms as matrices, from anywhere in the file's directory
  tree. A calibrated axis is read and applied automatically. Reading only;
  nothing is written back.
- **Save and reload fits.** File > Save Fits... and Load Fits... keep an
  analysis across sessions. Loading offers either restoring the saved
  numbers or re-running each fit from its saved marks with the current
  code -- worth using given the changes above.
- **Calibrate from fitted peaks.** Assign known energies to peaks you have
  already fitted and calibrate from their fitted centroids, which is more
  accurate than positioning a cursor. The worst residual is reported, so a
  mistyped energy is visible rather than silently absorbed.
- **Reload Spectrum (F5)** re-reads the active spectrum from disk while
  keeping its calibration, fits and marks -- for watching a measurement
  that is still running.
- **CSV and LaTeX export** alongside the existing text report, one row per
  peak.
- **Fit background** option, fitting the background line together with the
  peaks instead of subtracting it first. Off by default. Peak
  uncertainties grow when it is on, because they then include how well the
  background itself is known.
- **Several gates per matrix cut**, for gating on more than one member of
  a cascade at once. Hold <kbd>C</kbd> and mark as many as you want; they
  are summed, overlapping gates count their shared channels once, and the
  background weighting totals across all of them.
- **A shaded uncertainty band** around the background line, narrow between
  the background regions and wider outside them.

### Changed

- Matrix cuts and Add/Subtract results now carry their propagated
  per-channel uncertainties, and fits to them are weighted by those rather
  than by assuming Poisson counts. Reported uncertainties on such fits
  grow accordingly.
- Re-opening a matrix is roughly 17x faster, from a cache of the decoded
  data kept outside your data directory.

### Fixed

- Overlapping cut or background regions no longer count their shared
  channels twice.
- A background region marked entirely outside a matrix contributes
  nothing, as a consequence of the weighting rule rather than a special
  case.

## [3.1.4] - 2026-08-17

A refinement to the matrix cut, plus documentation of how its background
subtraction actually works.

### Changed

- **A background region marked entirely outside the matrix is now
  ignored** when activating a cut, contributing neither counts nor
  width. Previously such a region still counted toward the background
  width, which quietly weakened the subtraction even though the region
  had no data under it. If every background region is outside the
  matrix, no subtraction is performed.
- A background region that merely *overlaps* the edge of the matrix is
  unaffected and still counts its full marked width — that is the
  behaviour corrected in 3.1.3 and it remains in place.

### Documentation

- The Knowledge Database now explains the cut's **gate-width weighting**
  in full: the formula used, a worked example, the fact that widths are
  counted inclusively, and why adding more background regions improves
  the statistics rather than weakening the subtraction.
- It also spells out the **matrix-edge caveat**: region widths come from
  where you marked, while counts can only come from data that exists, so
  a background band left hanging over the edge removes less background
  than intended. Mark background bands well inside the data.

## [3.1.3] - 2026-08-17

Corrects two faults in the matrix cut that removed too much background,
and adds a way to slide along a spectrum with the mouse. If you have
activated cuts whose regions reached beyond the edge of the matrix, those
results were wrong and are worth recomputing.

### Added

- **Drag with the right mouse button to slide the spectrum sideways.**
  The X axis keeps its current width and the Y axis rescales as you go to
  fit whatever is on screen, so a small peak is no longer flattened by a
  tall one that has scrolled out of view. Panning stops at the ends of
  the data. Works in the main window and in matrix panels. The left
  button is unchanged — it still places marks.

### Fixed

- **Ctrl+C in a matrix panel now clears the cut and background marks**
  as well, exactly as the Clear Marks button does. It previously cleared
  only fit marks, so cut marks survived it — and since redrawing the plot
  removes their markers from view without forgetting them, it was
  possible to be left with an active cut region you could no longer see.
  An already-activated cut spectrum is separate and is not affected.
- **Activating a cut could subtract far too much background.** Two
  separate causes, both requiring a cut or background region marked
  partly beyond the edge of the matrix — easily done by dragging past
  the end of a projection:
  - The background scale factor was computed from the part of each
    region that overlapped the matrix, rather than from the region as
    marked. That inflated the factor and removed too much: a single
    background region reaching past the low edge left about 7% of the
    correct net counts.
  - A background region lying entirely below channel 0 was summed as
    though it covered nearly the whole matrix, injecting a large
    spurious background that was then subtracted. A region past the
    upper edge was never affected.

  Cuts whose regions lay wholly inside the matrix were already correct
  and are unchanged.

### Notes

- Both faults were found by checking this code against the original TV
  implementation line by line, and the corrected version now reproduces
  TV's results exactly across a wide randomised comparison.
- One consequence, matching TV: a background region marked entirely
  outside the matrix still counts toward the background width, so it
  weakens the subtraction rather than being ignored.

## [3.1.2] - 2026-08-17

Results of a full audit of performance, stability and usability. The
memory fix is the one that matters in a long session.

### Fixed

- **Closing a matrix window now frees its memory.** A closed matrix
  window was still holding its entire decoded matrix — around 500 MB for
  a full-size one — so opening and closing several matrices in one
  session steadily consumed memory until the application was restarted.
- **A failed fit now names the peaks involved**, rather than the
  program's internal parameter names. Where it previously read "could
  not determine amp_0, amp_1, pos_0, ..." it now says "could not
  determine peaks 1, 2, 3, 4", and explains that peaks marked at nearly
  the same position, or too narrow a fit region, are the usual cause.

### Performance

- **Redrawing is faster on spectra carrying many fits.** Fits lying
  entirely outside the visible range are no longer redrawn: with 100
  committed fits, a redraw while zoomed in on part of the spectrum drops
  from about half a second to under a tenth. Viewing the whole spectrum,
  where every fit really is on screen, is unchanged.

### Documentation

- The Knowledge Database now explains **what "± n/a" means** when a fit
  reports it — that the value is real and only its uncertainty could not
  be determined, that a left tail on a peak without one is the usual
  cause, and that unchecking Left tail removes it. It also covers how
  that appears in exported reports and logs.
- The HowTo now describes the progress window shown while a matrix
  loads, and notes that loading is slower from a network share.

## [3.1.1] - 2026-08-16

A performance and responsiveness release, addressing matrix loading and
zooming feeling sluggish — most visibly on Linux.

### Changed

- **Opening a matrix no longer freezes the application.** The decode now
  runs in the background behind a progress dialog, so the window keeps
  responding and shows how far along it is. The decode itself takes
  slightly longer in absolute terms as a result; it no longer blocks
  everything while it runs.
- **Zooming is smoother.** Rapid mouse-wheel zooming previously redrew
  the plot once per wheel click; those redraws are now combined, so a
  burst of ten clicks costs one redraw instead of ten. The difference is
  most noticeable where drawing is slow — a remote desktop, a virtual
  machine, or any system without graphics acceleration.

### Performance

- **Matrices load roughly twice as fast on Linux** and about 10% faster
  on Windows. Three separate causes: the file is now read in one
  operation instead of one per matrix line (which dominated when the
  file lived on a network or virtual filesystem, such as a Windows drive
  seen from WSL), decoded rows are assembled more efficiently, and the
  Linux packages are now built against a newer Python whose loop
  performance is markedly better.

### Notes

- The Linux `.deb` and `.rpm` are built against Python 3.12 rather than
  3.9. The minimum supported system is unchanged — they still run on the
  same RHEL- and Debian-family releases as before.

## [3.1.0] - 2026-08-16

A correctness and robustness release. It fixes everything found by a
full audit of v3.0.0, including five crashes or silent-corruption bugs
reachable in ordinary use, and restores exact agreement with TV in two
places where the fitting maths had drifted.

### Fixed - crashes and data corruption

- **Integration on a spectrum containing negative counts** (most easily
  produced by Subtract Spectra) crashed the application. It now reports
  a clear error instead.
- **Opening a file the app could not read as text** — an image, a PDF,
  anything picked through the "All files" filter — crashed the
  application. Such files are now rejected with a message.
- **A corrupt or truncated .n42 file** could exhaust memory and take the
  application down. Run lengths are now bounded, as they already were
  for .spk and .mtx.
- **A very large or infinite Multiply/Add/Subtract factor** silently
  corrupted the spectrum's counts in place, with no visible error. Such
  factors are now rejected.
- **The matrix panel's Fit Results table never filled in**, so fits made
  on a projection could not be selected, removed or exported from it,
  despite the documentation saying otherwise.
- **A calibration file or .n42 containing "nan" or "inf"** was accepted
  and made every displayed energy blank. Coefficients must now be finite.

### Changed - fitting behaviour

- **Peak fits now start from the same initial width TV uses.** The
  previous starting guess was twice TV's. This changes fitted results
  for closely-spaced multiplets and makes fits converge more reliably:
  across 200 randomised test fits the old guess failed to converge 10
  times where the corrected one never did, and average position error
  roughly halved.
- **A fit whose tail parameters the data cannot pin down is no longer
  thrown away.** Previously the whole fit was rejected. It now reports
  position, width and area normally, showing "n/a" for only the
  uncertainties that could not be determined. Fits where the peak
  parameters themselves are undetermined are still rejected.
- The solver's last-resort recovery step now re-applies its own step
  limits, so a fitted peak can no longer land outside the marked region.
- Background uncertainty for Integration used the wrong statistical
  moment, making the reported background skewness error incorrect
  whenever background regions were marked.

### Changed - other

- The zoom level is now preserved when toggling log scale, toggling a
  spectrum's visibility, or removing a spectrum.
- Save Spectrum always writes a real file extension, instead of an
  extensionless file on Linux.
- Activate Cut labels are now path-safe, so automatic fit logs are no
  longer written to truncated filenames.

### Performance

- Multi-peak fits are substantially faster: the solver no longer
  computes a full derivative matrix for trial steps it then rejects.
- Loading a large 2D matrix is faster.
- Adding or removing spectra no longer rebuilds the whole spectra list;
  with many spectra loaded this is 10x faster at 20 and around 57x at
  200.

### Notes

- Automatic fit logs (`*_fits.jsonl`) now write `null` rather than a
  bare `NaN` for an undetermined uncertainty, so the files are valid
  JSON for other tools.
- matplotlib is now pinned below 3.12; see requirements.txt.

## [3.0.0] - 2026-08-13

### Added

- **Matrix analysis**: `File > Open Matrix...` (`Ctrl+Shift+O`) opens a 2D
  gamma-gamma coincidence matrix (`.mtx`) in its own window. Both the X and Y
  projections are computed up front; pick which one to work on from the
  dropdown, displayed as a histogram like any ordinary spectrum. Hold `C` to
  mark a cut (signal) region and `G` to mark one or more background regions,
  then **Activate Cut** (`Ctrl+Alt+C`) to compute a background-subtracted
  spectrum (region-width-weighted, same convention as Integration) and add
  it to the main window like any other loaded spectrum. **Show
  Heatmap...** opens a separate, view-only 2D intensity map for visual
  reference. The Knowledge Database gained a new section explaining 2D
  matrices, projections, and cuts/gates for anyone unfamiliar with the
  technique.
- **Full fit/integrate/calibrate/zoom parity for the matrix panel**: the
  projection view now supports everything the main window's ordinary-spectrum
  view does. Mark background/fit region/peaks (`B`/`R`/`P`) and fit
  (`Ctrl+F`), integrate (`Ctrl+I`), or preview the background alone
  (`Ctrl+B`) directly on the projection, with its own Fit Results and Fit
  Parameters docks. Calibration (`Ctrl+L`) is shared with the main window --
  calibrating from either window updates every open window's display.
  `Ctrl+=`/`Ctrl+-`/`Ctrl+0` and the scroll wheel zoom the projection's X
  axis. Switching between X and Y projection clears in-progress marks and
  fits, since it's different underlying data.

### Fixed

- **Fits and integrations on a matrix projection, an activated cut, or an
  Add/Subtract result could auto-log to the wrong location** (and default
  "Export Fit Report" to the same wrong place). These spectra only ever had
  a display label as their path (e.g. "gg.mtx x projection", "a.spe +
  b.spe"), not a real one, so both features silently fell back to the
  process's working directory instead of somewhere sensible. All three now
  anchor to a real directory -- the source `.mtx` file's own directory for
  projections and cuts, the first operand's directory for Add/Subtract --
  while the spectrum's displayed name in the Loaded Spectra list is
  unchanged. This bug affected Add/Subtract Spectra since its v2.1.0
  introduction, not just the matrix features new in this release.

## [2.2.2] - 2026-08-13

### Fixed

- **The built-in Save/Home/Pan toolbar icons could get stuck on the wrong
  color after switching theme**, most reliably visible on Windows.
  matplotlib only colors those particular icons once, when the toolbar
  is first built, so toggling dark theme afterward updated everything
  else but silently left those three icons showing whichever color
  matched the theme active at startup. They're now explicitly
  re-rendered on every theme change, matching this app's own
  zoom/calibration toolbar icons, which already did this correctly.

## [2.2.1] - 2026-08-12

### Added

- **N42 file support (read-only)**: `File > Open...` now also accepts
  `.n42` files (ANSI/IEEE N42.42-2011). Only the raw histogram and, if
  present, the embedded energy calibration are read — everything else in
  the file is ignored, and there's no way to save back to `.n42`. If no
  calibration is currently active, an N42 file's own calibration is
  applied automatically; if one is already active, it's left alone. Files
  containing anything other than exactly one spectrum are rejected with a
  clear error rather than guessed at.

## [2.2.0] - 2026-08-11

### Added

- **Integration without background regions**: `Ctrl+I` now works with zero
  background regions marked, not just two — reports the raw gross
  area/centroid/FWHM/skewness with no background subtraction. Marking two
  background regions first still works exactly as before.
- **`Ctrl+B`**: previews the background fit computed from the two marked
  background regions alone — no fit region or peaks needed. Useful for
  sanity-checking the background before marking the rest. It's a preview
  only: nothing is added to Fit Results, logged, or exported. Press
  `Ctrl+B` again to hide it.
- **Status-bar feedback when `Ctrl+F`/`Ctrl+I`/`Ctrl+B` are pressed with
  incomplete marks** — previously silent no-ops, these now name
  specifically what's still missing.
- **Knowledge Database page** gained real formulas: how sigma relates to
  FWHM, how a fit's position/volume/uncertainties are computed (including
  the full-vs-net volume distinction), and how Integration computes
  gross/background/net directly from the data — with two new annotated
  figures.
- **HowTo page** gained documentation of the automatic per-fit JSON-Lines
  log and the `Ctrl+E` text-report export format (neither was documented
  before), and of the new `Ctrl+B` shortcut.

### Changed

- **`Ctrl+C`** ("Clear") now hides the active spectrum's committed fits
  instead of permanently deleting them — grayed out in Fit Results,
  removed from the plot, but recoverable by re-fitting the same marks.
  In-progress B/R/P marks are still cleared as before. `Ctrl+Shift+C`
  remains the permanent-delete option, unchanged.

### Fixed

- **The background line for a committed fit or integration was drawn
  across the fit region instead of the two background regions it's
  actually calculated from** — visually misleading whenever those two
  spans differ, which is the normal case. Now spans the background
  regions correctly, for both a peak fit and an integration.

## [2.1.1] - 2026-08-05

### Fixed

- **Help pages (HowTo, Knowledge Database, About) failed to open on Linux**,
  showing either no browser at all or a crash in the launched browser,
  depending on what was installed on the system. Root cause: the app
  relied entirely on the OS to know how to open an HTML file, which many
  real Linux installs don't have configured, and even when they do, the
  frozen app's own bundled libraries could interfere with the browser it
  launched. The Help menu now always launches a browser directly itself
  on Linux, sidestepping both problems. Confirmed working by the user on
  a real Linux machine after two fix iterations.

## [2.1.0] - 2026-08-05

### Added

- **Add Spectra... / Subtract Spectra...** (`Ctrl+A` / `Ctrl+Shift+A`):
  combine two loaded spectra of equal length into a new spectrum, with
  the second scaled by a user-provided factor (defaulting to 1) first.
  The originals are left untouched. Subtracting can produce negative
  channel counts in the result, which is expected and not an error.
  Picking spectra of different lengths shows an error naming both
  channel counts instead of proceeding.
- Documented in the HowTo page alongside every other Operations-menu
  command.
- The About page now credits Claude Code alongside the copyright line.

### Fixed

- A new spectrum from Add/Subtract could collide in name with an
  existing one if the same operation was repeated with the same
  inputs — since spectra are otherwise identified by that name
  internally, this could cause removing one to silently remove the
  other too. Fixed by auto-disambiguating with a `(2)`, `(3)`, ...
  suffix on collision.
- A `NaN` factor typed into the Add/Subtract dialog was not rejected
  by the "must be greater than zero" check the way it should have
  been, due to a comparison that doesn't behave as expected for `NaN`
  under IEEE-754 rules.

### Known issues

- On WSLg (WSL's GUI subsystem) specifically, a secondary window (a
  dialog such as Multiply by Factor or the file-open dialog) can
  briefly flicker — disappear and reappear — right after opening. This
  is the same underlying WSLg compositor behavior already noted below
  for the main window, just triggered by dialog creation instead of
  app launch. Purely cosmetic and self-resolving; not observed on a
  native Linux desktop, and no application-side fix exists. See
  `packaging/linux/INSTALL.md` for details.

## [2.0.0] - 2026-08-04

### Added

- **Help menu**: HowTo (full keyboard-shortcut reference and
  step-by-step guides for every operation), Knowledge Database (fit
  model, parameter meanings, calibration math, with annotated figures),
  and About (version/build-date display).
- **Native Linux packages**: `.deb` and `.rpm` installers, replacing
  the earlier AppImage. Both automatically install any missing runtime
  libraries as part of the same install command. Built on AlmaLinux 8
  for broad glibc compatibility; verified to also install and run
  correctly on current releases (AlmaLinux 10.2).
- **End-user install documentation**: a real "Installing" section in
  `README.md` (Windows and Linux requirements and commands) and
  `packaging/linux/INSTALL.md` with full Linux install/uninstall
  instructions per distro family.

### Fixed

- The original Linux AppImage release failed to run at all: missing
  FUSE on some systems, a GLIBC version mismatch on others. Replaced
  entirely rather than patched, since AppImage's FUSE requirement made
  it unreliable as a no-install option regardless.
- A packaging bug where `rpmbuild`'s default file-stripping silently
  corrupted a bundled math library, crashing the app on launch despite
  the package installing and its metadata looking correct.
- Two WSLg (Windows Subsystem for Linux GUI) display bugs that could
  leave the app's window invisible (a taskbar icon with no visible
  window) when run under WSL specifically: a Wayland window-sizing
  issue (worked around automatically, no user action needed) and a
  WSLg compositor issue (worked around by restarting WSL).
- Various minor documentation and defensive-coding issues caught during
  code review (see git history for specifics).

### Requirements

- **Windows**: Windows 10 or later.
- **Linux**: RHEL/CentOS/AlmaLinux/Rocky 8 or later, or a current
  Debian/Ubuntu release. RHEL/CentOS 7 and older are not supported —
  PySide6 (the Qt6 binding this app uses) and current numpy/scipy no
  longer publish builds compatible with that old a system, an upstream
  constraint no packaging choice can work around.
