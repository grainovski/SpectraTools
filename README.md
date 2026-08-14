# SpectraTools

A cross-platform desktop app (Windows + Linux) for opening ASCII
histogram `.txt` files, plotting them, and fitting peaks — the
foundation of the "PeakFinderFitting" project.

See `CHANGELOG.md` for what's new in each release.

## Installing

Pre-built releases are under `releases/`, one subdirectory per version
(e.g. `releases/v3.0.0/`). Check that directory for the latest, or see
`CHANGELOG.md` for the release history. Pick your platform:

### Windows

Requires **Windows 10 or later** (Qt6, which this app is built on,
doesn't support Windows 7/8). Run the installer inside the latest
release directory (e.g. `SpectraTools-v3.0.0-Setup.exe` in
`releases/v3.0.0/`) and follow the installer. No other prerequisites —
everything the app needs is bundled.

### Linux

Requires **RHEL/CentOS/AlmaLinux/Rocky 8 or later** (RPM) or a **current
Debian/Ubuntu release** (DEB). Older RHEL/CentOS (7 and before) isn't
supported — PySide6, the Qt6 binding this app uses, stopped publishing
wheels compatible with that old a glibc after version 6.2.4 (2021), and
that's an upstream constraint no packaging choice here can work around.

Install the `.deb` or `.rpm` from the latest release directory — the
commands below show v3.0.0 as an example; substitute whatever version
you actually have:

```
sudo apt install ./spectratools_3.0.0_amd64.deb      # Debian/Ubuntu
sudo dnf install ./spectratools-3.0.0-1.x86_64.rpm    # RHEL/CentOS/AlmaLinux/Rocky
```

Both automatically install any missing runtime libraries as part of the
same command. See `packaging/linux/INSTALL.md` for the full instructions
(including a one-time EPEL prerequisite on RHEL-family **8** specifically)
and uninstall steps.

**Running under WSL:** if the app's window doesn't appear (a taskbar
icon shows up but clicking or Alt+Tab-selecting it does nothing), that's
a WSLg display bug, not an app problem — run `wsl --shutdown` from a
Windows terminal (not from inside WSL) to restart WSL's display
subsystem, then reopen your WSL terminal and try again.

## Setup (for building from source)

    python -m venv .venv
    .venv/Scripts/python.exe -m pip install -r requirements-dev.txt   # Windows
    .venv/bin/pip install -r requirements-dev.txt                     # Linux/macOS

## Run

    .venv/Scripts/python.exe main.py   # Windows
    .venv/bin/python main.py           # Linux/macOS

## Test

    .venv/Scripts/python.exe -m pytest   # Windows
    .venv/bin/python -m pytest           # Linux/macOS

## Build installers

- Windows: `powershell -File packaging\windows\build.ps1` (requires Inno
  Setup — see `packaging/windows/build.ps1` for the check/install step).
  Produces `packaging/windows/output/SpectraToolsSetup.exe`.
- Linux: two scripts on two separate build hosts (RPM and DEB packaging
  tools don't coexist on one distro; built and tested via WSL in this
  repo's history):
  - `bash packaging/linux/build.sh` on AlmaLinux 8 (glibc 2.28, for
    broad RHEL-family compatibility) — PyInstaller onedir build, then
    produces `packaging/linux/output/spectratools-<version>-1.x86_64.rpm`.
  - `bash packaging/linux/build_deb.sh` on Ubuntu 24.04, run *after*
    build.sh (it reuses build.sh's onedir output rather than rebuilding)
    — produces `packaging/linux/output/spectratools_<version>_amd64.deb`.

  See `packaging/linux/INSTALL.md` for end-user install instructions.

`build.ps1` and `build.sh` each generate `assets/icon.png`/`assets/icon.ico`
on first run if not already present (via `packaging/make_icon.py`), using
the same drawing code as the app's own window icon. `build_deb.sh` doesn't
generate it — it expects `build.sh` to have already run first (the normal
pipeline order, since it reuses `build.sh`'s onedir output anyway) and
just copies the existing `assets/icon.png` into the `.deb`.

## File format

See `docs/superpowers/specs/2026-07-08-histogram-viewer-design.md` for the
full design spec, including the ASCII file format and channel-count
bucketing rules.
