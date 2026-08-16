# Changelog

All notable changes to SpectraTools are documented here, starting from
version 2.0.0. Dates are when the version was frozen and released, not
when individual pieces of work happened.

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
