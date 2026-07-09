# Multi-Spectrum Support — Design Spec

Date: 2026-07-09

## Purpose

Extend the histogram viewer (previously single-spectrum only) to load
multiple ASCII spectrum files at once, list them, let the user toggle
each one's visibility independently, and display all currently-visible
spectra overlaid on one plot in distinct colors. Checking exactly one
spectrum is how you view it "separately"; checking several overlays
them for comparison — one mechanism serves both requests. This was
explicitly deferred as future work in the original design spec
(`2026-07-08-histogram-viewer-design.md`, "Out of scope" section).

## Data model

A small `LoadedSpectrum` class (in `main_window.py`) replaces the
single `self.data` array:

```python
class LoadedSpectrum:
    def __init__(self, path, data, color):
        self.path = path
        self.data = data
        self.color = color
        self.visible = True
```

`MainWindow.spectra` becomes `list[LoadedSpectrum]` (was `self.data`,
a single array or `None`). Colors are assigned from Matplotlib's
default color cycle (`axes.prop_cycle`), cycling by index — with more
than ~10 loaded spectra colors repeat, which is an accepted minor
limitation for this tool's scale, not worth solving now.

## Loading

- `File → Open…` uses `QFileDialog.getOpenFileNames` (plural) instead
  of `getOpenFileName`, allowing multi-select.
- Each selected path is loaded via a per-file helper that parses it,
  and on success appends a new `LoadedSpectrum` to `self.spectra`
  (never replacing existing entries), updates last-folder/recent-files,
  and refreshes the spectrum list panel.
- Opening a path that's already loaded (same path) is a no-op — it is
  not added twice.
- If a multi-selected batch has some files that fail to parse, the
  ones that succeed still load; a single `QMessageBox` at the end
  lists which file(s) failed and why, rather than aborting the batch.
- Clicking a Recent Files entry also appends (doesn't replace),
  consistent with the new accumulate-based model.

## Spectrum list panel

A new `QDockWidget` on the left, containing a `QListWidget`:

- One row per loaded spectrum: a checkbox (visible/hidden, defaults
  checked), a small solid-color icon swatch matching its assigned
  plot color, and the filename (not full path) as the label.
- Toggling a checkbox re-plots immediately, showing/hiding that
  spectrum.
- Right-click on a row shows a context menu with "Remove", which
  drops that spectrum from `self.spectra` entirely and re-plots.
  Removing the last remaining spectrum returns to today's empty/no-data
  plot state.

## Plot behavior

- Every spectrum with `visible == True` is drawn as its own step-line,
  in its assigned color, all on the same `Axes`.
- The X-axis range spans the full channel range of the widest
  *currently visible* spectrum (channel counts can differ between
  loaded files): `(0, max(len(s.data) for s in visible_spectra) - 1)`.
- "Show Full Spectrum" resets to that combined full range.
- Log scale, zoom (scroll + toolbar buttons), and Y auto-rescale all
  operate the same as before, but now consider the min/max across all
  visible spectra within the current X-range rather than a single
  array.
- No on-plot legend — the side panel's color swatches already serve
  that purpose, and adding a second one would be redundant clutter.
- If no spectra are currently visible (all unchecked, or the list is
  empty), the plot shows the same empty/no-data state as before this
  feature — no error, nothing drawn.

## Status bar on hover

Shows counts from every currently-visible spectrum at the hovered
channel, e.g.:

```
Channel: 120  |  file1.txt: 452  |  file2.txt: 108
```

If a channel is out of range for a shorter visible spectrum, that
entry shows `-` instead of a count, e.g. `file2.txt: -`.

## Error handling

- Multi-select batch with partial failures: succeed on the good files,
  one combined `QMessageBox` names the failed file(s) and why (reusing
  the existing `ParseError`/`OSError` distinction from `histogram_io`).
- Already-loaded path: no-op, not added twice.
- Removing all spectra: returns to the existing empty/no-data plot
  state (unchanged from before this feature).
- More than ~10 loaded spectra: colors start repeating from the cycle.
  Accepted, not solved here.

## Testing

Same approach as the rest of this app (no automated GUI test
framework, per the original design spec): an offscreen script that
loads two files, verifies both appear in the list with distinct
colors, toggles visibility and confirms the plot's visible lines match
the checked state, confirms the combined X-range spans the wider file,
and confirms the status-bar text format with multiple spectra visible
— plus a visual screenshot check of the panel + overlay. No manual
smoke-test handoff for this feature; verified directly against the
rebuilt installer instead.

## Out of scope

- Per-spectrum manual color selection (auto-assigned only).
- Any color-repeat handling beyond the plain cycle wraparound.
- A legend drawn on the plot itself.
