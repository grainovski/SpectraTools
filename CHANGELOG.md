# Changelog

All notable changes to SpectraTools are documented here, starting from
version 2.0.0. Dates are when the version was frozen and released, not
when individual pieces of work happened.

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
