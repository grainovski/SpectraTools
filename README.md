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
- Linux: `bash packaging/linux/build.sh`, built and tested via WSL Ubuntu
  in this repo's history. Produces
  `packaging/linux/output/SpectraTools-x86_64.AppImage`. Downloads
  `appimagetool` on first run if it isn't already present under
  `packaging/linux/tools/` (requires internet access).

Both build scripts generate `assets/icon.png`/`assets/icon.ico` on first
run if not already present (via `packaging/make_icon.py`), using the same
drawing code as the app's own window icon.

## File format

See `docs/superpowers/specs/2026-07-08-histogram-viewer-design.md` for the
full design spec, including the ASCII file format and channel-count
bucketing rules.
