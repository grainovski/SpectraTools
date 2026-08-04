# SpectraTools

A cross-platform desktop app (Windows + Linux) for opening ASCII
histogram `.txt` files and plotting them — the foundation of the
"PeakFinderFitting" project.

## Setup

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

Both build scripts generate `assets/icon.png`/`assets/icon.ico` on first
run if not already present (via `packaging/make_icon.py`), using the same
drawing code as the app's own window icon.

## File format

See `docs/superpowers/specs/2026-07-08-histogram-viewer-design.md` for the
full design spec, including the ASCII file format and channel-count
bucketing rules.
