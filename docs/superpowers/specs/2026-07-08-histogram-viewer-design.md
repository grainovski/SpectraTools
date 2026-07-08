# Histogram Viewer — Design Spec

Date: 2026-07-08

## Purpose

A cross-platform (Windows + Linux) desktop GUI application that lets a user
open an ASCII histogram file and view it as a plot. This is the first piece
of the broader "PeakFinderFitting" project; later work (peak finding,
curve fitting) will build on top of this viewer.

## Input file format

Observed in `test.txt` / `test1.txt` (identical, 4097 lines each):

- Line 1: a non-numeric header/comment line, e.g. `#:MatrixFormat:4k.txt:1`
- Lines 2–4097: one integer per line — the counts for channel 0, 1, 2, … in
  order (4096 channels total in the sample files).

The parser must not assume a fixed header length or position: it should
tolerate files with no header, a header of arbitrary length, or blank lines,
by skipping any line that fails to parse as a number, wherever it occurs.
Channel index is assigned by position among the numeric lines that remain,
starting at 0.

The binary `eu.spe` (Maestro format) file in the working directory is out of
scope for this version — this app only reads the ASCII `.txt` format
described above.

### Channel count bucketing

MCA spectra conventionally have a channel count that's a power of two
(4096, 8192, 16384, …), and some exports truncate trailing all-zero
channels, so the number of numeric data lines in a file (`N`) may be less
than the spectrum's true channel count. The app infers the true channel
count by rounding `N` up to the smallest power-of-two bucket, floored at
`2^12 = 4096`:

- `N <= 4096` → 4096 channels
- `4096 < N <= 8192` → 8192 channels
- `8192 < N <= 16384` → 16384 channels
- … and so on.

Any channels beyond `N` (i.e. the truncated trailing channels) are
zero-padded. An exact fit (e.g. `N == 4096`) stays in its own bucket — it is
not bumped up to the next one.

## Tech stack

- Python 3.13, PySide6 (Qt) for the GUI, Matplotlib embedded via
  `FigureCanvasQTAgg` for the plot.
- Chosen over Tkinter (plainer widgets) and C++/Qt (native, but far slower to
  build and to extend later with numpy/scipy-based peak fitting).
- Packaged into real OS installers via PyInstaller + Inno Setup (Windows) and
  PyInstaller + AppImage (Linux) — see Packaging section.

## Architecture / files

- `main.py` — entry point; creates the `QApplication` and `MainWindow`.
- `histogram_io.py` — `load_histogram(path) -> np.ndarray`, plus a
  `ParseError` exception.
- `main_window.py` — `MainWindow`: menu bar, embedded plot canvas +
  toolbar, status bar.
- `settings.py` — thin wrapper around `QSettings` for last-used folder and
  recent-files list (persisted automatically: registry on Windows, config
  file on Linux).

## UI layout

- **File menu:** Open… (Ctrl+O), Recent Files (submenu, last 8 entries),
  Exit.
- **View menu:** "Log scale Y" checkable toggle.
- **Central widget:** Matplotlib canvas plotting counts vs. channel as a step
  line (`drawstyle='steps-mid'` — a bar chart would be unreadable at
  thousands of channels), x-axis "Channel", y-axis "Counts", window title
  set to the opened filename.
- **Toolbar:** the standard Matplotlib navigation toolbar (Home/reset, Pan,
  box-Zoom, Save-as-image) plus two custom buttons: "Zoom In X" / "Zoom Out
  X".
- **Status bar:** live "Channel: N  Counts: M" readout following the mouse
  over the plot.

## X-axis zoom with Y auto-rescale

Two ways to trigger it:

1. **Scroll wheel** over the plot zooms the X-axis in/out centered on the
   cursor's x-position.
2. **Toolbar buttons** ("Zoom In X" / "Zoom Out X") step the X-range
   in/out by a fixed factor (e.g. 1.5x) centered on the current view's
   midpoint.

After either action, the Y-axis automatically rescales to fit the min/max of
the counts currently visible within the new X-range (with a small margin).
This is layered on top of — not a replacement for — the standard Matplotlib
Pan/box-Zoom tools, which remain available for full 2D control and are
unaffected by the auto-rescale logic.

## Data flow

1. User picks a file via File → Open… (dialog filtered to `*.txt`, with an
   "All files" option) or clicks a Recent Files entry.
2. `histogram_io.load_histogram(path)` reads the file line by line, attempts
   `int(line.strip())` on each line, skips lines that fail to parse. The
   remaining values give `N` data points.
3. If zero numeric lines are found, raises `ParseError` with a clear
   message.
4. The channel count is rounded up to the appropriate power-of-two bucket
   (see "Channel count bucketing" above), and the values are placed into a
   numpy integer array of that length, zero-padded beyond `N`.
5. `MainWindow` plots the array, sets the window title to the filename.
6. On success, the last-used folder and recent-files list are updated via
   `QSettings`.

## Error handling

- File unreadable / permission error → `QMessageBox.critical` with the OS
  error message.
- No numeric data found → `QMessageBox.warning("No histogram data found in
  file")`.
- A Recent Files entry pointing at a file that's been deleted/moved → warn
  and remove it from the list on click.

## Testing

- **Unit tests** (pytest) for `histogram_io.load_histogram`:
  - Parses `test.txt` / `test1.txt` to 4096 values starting
    `[4, 0, 1, 0, 1, ...]`.
  - Handles a file with no header line.
  - Handles stray blank lines interspersed with data.
  - Raises `ParseError` on a file with no numeric data.
  - Channel-count bucketing: a file with fewer than 4096 data lines pads out
    to 4096 channels; a file with 4097–8192 data lines pads out to 8192
    channels; a file with exactly 4096 data lines (like test.txt) stays at
    4096 channels (no bump to 8192).
- **Manual GUI smoke test** (checklist, run once before calling the app
  done): open a file, confirm the plot renders, toggle log-scale, hover for
  the channel/count readout, scroll-zoom and toolbar-zoom X with Y
  auto-rescaling, confirm the recent-files list survives an app restart.
- No automated GUI test framework (e.g. `pytest-qt`) in this first version —
  would be overkill for a single-window app; can be added later if the UI
  grows.

## Packaging (installable distributions)

- **Windows:** PyInstaller (`--onefile --windowed`) bundles
  Python+PySide6+Matplotlib into a single `.exe` requiring no separate
  Python install on the target machine. An Inno Setup script (`.iss`) then
  wraps that into a proper installer: Start Menu shortcut, Add/Remove
  Programs entry, uninstaller. Built and smoke-tested directly on the
  Windows dev machine.
- **Linux:** PyInstaller (`--onedir`) build, then packaged into an
  **AppImage** via `appimagetool` — a single executable the user
  `chmod +x`'s and runs directly, no root or package manager required, works
  across distros. Built and smoke-tested inside the WSL Ubuntu 24.04
  environment available on the dev machine.
- Both build scripts live in the repo (`packaging/windows/`,
  `packaging/linux/`) so they can be rerun after future changes.

## Out of scope for this version

- Reading the binary `.spe` format.
- Peak finding / curve fitting (future project phase).
- Overlaying multiple histograms in one plot.
- Automated GUI testing.
