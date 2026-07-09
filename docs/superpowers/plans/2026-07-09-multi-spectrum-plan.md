# Multi-Spectrum Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the histogram viewer to load multiple spectrum files at once, list them in a panel with visibility checkboxes and color swatches, and overlay every currently-checked spectrum on one plot in distinct colors.

**Architecture:** A new `spectrum.py` module holds a small framework-agnostic `LoadedSpectrum` data class plus the color-cycle helper. `main_window.py`'s `self.data` (single array) becomes `self.spectra` (a `list[LoadedSpectrum]`); a new `QDockWidget`/`QListWidget` panel lets the user toggle visibility and remove entries; `_plot_data`, `_autoscale_y`, `_zoom_x`, `_show_full_spectrum`, and `_on_mouse_move` are updated to operate over the currently-visible subset of `self.spectra` instead of a single array.

**Tech Stack:** Same as the rest of the app — Python, PySide6, Matplotlib. No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-07-09-multi-spectrum-design.md`

---

## Notes for whoever executes this plan

- Work from `C:\Users\RIG\Documents\Claude\PeakFinderFitting\.worktrees\histogram-viewer` on branch `feature/histogram-viewer`. Use `.venv/Scripts/python.exe` for Python/pytest (already has all deps installed).
- As with the rest of this app, there's no automated GUI test framework — verification is via one-off offscreen scripts (prefix commands with `QT_QPA_PLATFORM=offscreen`) that construct `MainWindow`, exercise it programmatically, and assert on its state.
- `tests/fixtures/test.txt` and `tests/fixtures/test1.txt` are identical (both 4096-channel spectra) — for tasks that need two *different-length* fixtures, generate a second one on the fly with `tmp_path`-style scratch files, or reuse the existing `no_header.txt` fixture (6 data lines, padded to 4096 channels — same length as test.txt, so not useful for testing *different* lengths). Where a task needs two different channel counts, it creates its own scratch fixture inline (shown in that task).

---

### Task 1: `spectrum.py` — data model and color cycle

**Files:**
- Create: `spectrum.py`

- [ ] **Step 1: Implement `spectrum.py`**

```python
COLOR_CYCLE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


class LoadedSpectrum:
    def __init__(self, path, data, color):
        self.path = path
        self.data = data
        self.color = color
        self.visible = True


def next_color(index):
    return COLOR_CYCLE[index % len(COLOR_CYCLE)]
```

- [ ] **Step 2: Verify it imports and behaves correctly**

Run:
```
.venv/Scripts/python.exe -c "from spectrum import LoadedSpectrum, next_color, COLOR_CYCLE; s = LoadedSpectrum('a.txt', [1,2,3], next_color(0)); print(s.path, s.color, s.visible); print(next_color(0), next_color(10), next_color(0) == next_color(10))"
```
Expected output:
```
a.txt #1f77b4 True
#1f77b4 #1f77b4 True
```
(confirms the color cycle wraps around after 10 entries)

- [ ] **Step 3: Commit**

```bash
git add spectrum.py
git commit -m "Add LoadedSpectrum data model and color cycle for multi-spectrum support"
```

---

### Task 2: Loading logic — accumulate multiple spectra

**Files:**
- Modify: `main_window.py`

- [ ] **Step 1: Import `LoadedSpectrum`/`next_color` and replace `self.data` with `self.spectra`**

Change:
```python
from histogram_io import ParseError, load_histogram
from settings import Settings
```
to:
```python
from histogram_io import ParseError, load_histogram
from settings import Settings
from spectrum import LoadedSpectrum, next_color
```

In `__init__`, change:
```python
        self.data = None
```
to:
```python
        self.spectra = []
```

- [ ] **Step 2: Make Open multi-select, and load in a batch**

Change:
```python
    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Histogram", self.settings.last_folder(), "Text files (*.txt);;All files (*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path):
        try:
            data = load_histogram(path)
        except ParseError as exc:
            QMessageBox.warning(self, "No histogram data found in file", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Could not open file", str(exc))
            return

        self.data = data
        self.setWindowTitle(f"Histogram Viewer - {os.path.basename(path)}")
        self.settings.set_last_folder(os.path.dirname(path))
        self.settings.add_recent_file(path)
        self._update_recent_menu()
        self._plot_data()
```
to:
```python
    def _open_file_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Histogram", self.settings.last_folder(), "Text files (*.txt);;All files (*)"
        )
        if paths:
            self._load_files(paths)

    def _try_load_spectrum(self, path):
        try:
            data = load_histogram(path)
        except ParseError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        except OSError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        color = next_color(len(self.spectra))
        return LoadedSpectrum(path, data, color), None

    def _load_files(self, paths):
        failures = []
        loaded_any = False
        for path in paths:
            if any(s.path == path for s in self.spectra):
                continue
            spectrum, error = self._try_load_spectrum(path)
            if error:
                failures.append(error)
                continue
            self.spectra.append(spectrum)
            self.settings.set_last_folder(os.path.dirname(path))
            self.settings.add_recent_file(path)
            loaded_any = True

        if loaded_any:
            self._update_recent_menu()
            self._plot_data()

        if failures:
            QMessageBox.warning(self, "Some files could not be loaded", "\n".join(failures))
```

Note: this removes the window-title update (`setWindowTitle(f"Histogram Viewer - {...}")`) since there's no longer a single "current" file with multiple spectra loaded — leave the title as the plain "Histogram Viewer" set in `__init__`. Don't add a replacement for it; that's intentional, not an oversight.

- [ ] **Step 3: Update `_open_recent` to use the new batch loader**

Change:
```python
    def _open_recent(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "File not found", f"{path} no longer exists.")
            self.settings.remove_recent_file(path)
            self._update_recent_menu()
            return
        self._load_file(path)
```
to:
```python
    def _open_recent(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "File not found", f"{path} no longer exists.")
            self.settings.remove_recent_file(path)
            self._update_recent_menu()
            return
        self._load_files([path])
```

- [ ] **Step 4: Verify (this will fail until later tasks fix `_plot_data`/`_autoscale_y` — that's expected)**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "from PySide6.QtWidgets import QApplication; from main_window import MainWindow; app = QApplication([]); w = MainWindow(); w._load_files(['tests/fixtures/test.txt', 'tests/fixtures/test1.txt']); print(len(w.spectra)); print([s.path for s in w.spectra]); print([s.color for s in w.spectra]); w._load_files(['tests/fixtures/test.txt']); print(len(w.spectra))"
```
Expected: this will likely raise an `AttributeError` or similar inside `_plot_data()`/`_autoscale_y()` since they still reference `self.data` (removed) — that's expected at this point in the plan; Task 4 fixes it. If you want to confirm just the loading logic in isolation before that, temporarily comment out the `self._plot_data()` call inside `_load_files`, run the same command (expect `2`, then the two file paths, then two color hex strings, then `2` again confirming the duplicate `test.txt` load was skipped), and then uncomment it before moving on.

- [ ] **Step 5: Commit**

```bash
git add main_window.py
git commit -m "Replace single-spectrum loading with accumulating multi-spectrum loader"
```

---

### Task 3: Spectrum list panel — visibility checkboxes, color swatches, remove

**Files:**
- Modify: `main_window.py`

- [ ] **Step 1: Add imports**

Change:
```python
from PySide6.QtGui import QAction, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QVBoxLayout, QWidget
```
to:
```python
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)
```

- [ ] **Step 2: Add a color-swatch icon helper**

Append this function near the other icon helpers (after `_full_spectrum_icon`, before `class MainWindow`):

```python
def _color_swatch_icon(color):
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)
```

- [ ] **Step 3: Build the dock panel and wire it up in `__init__`**

Change:
```python
        self._build_zoom_buttons()
```
to:
```python
        self._build_zoom_buttons()
        self._build_spectrum_panel()
```

Append this method to `MainWindow` (e.g. right after `_build_zoom_buttons`... place it anywhere in the class; exact location doesn't matter, just don't nest it inside another method):

```python

    def _build_spectrum_panel(self):
        self.spectrum_list = QListWidget()
        self.spectrum_list.itemChanged.connect(self._on_spectrum_item_changed)
        self.spectrum_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.spectrum_list.customContextMenuRequested.connect(self._on_spectrum_context_menu)

        dock = QDockWidget("Loaded Spectra", self)
        dock.setWidget(self.spectrum_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _update_spectrum_list(self):
        self.spectrum_list.blockSignals(True)
        self.spectrum_list.clear()
        for spectrum in self.spectra:
            item = QListWidgetItem(os.path.basename(spectrum.path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if spectrum.visible else Qt.CheckState.Unchecked
            )
            item.setIcon(_color_swatch_icon(spectrum.color))
            item.setData(Qt.ItemDataRole.UserRole, spectrum.path)
            self.spectrum_list.addItem(item)
        self.spectrum_list.blockSignals(False)

    def _on_spectrum_item_changed(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        for spectrum in self.spectra:
            if spectrum.path == path:
                spectrum.visible = item.checkState() == Qt.CheckState.Checked
                break
        self._plot_data()

    def _on_spectrum_context_menu(self, position):
        item = self.spectrum_list.itemAt(position)
        if item is None:
            return
        menu = QMenu(self)
        remove_action = menu.addAction("Remove")
        chosen = menu.exec(self.spectrum_list.viewport().mapToGlobal(position))
        if chosen == remove_action:
            path = item.data(Qt.ItemDataRole.UserRole)
            self.spectra = [s for s in self.spectra if s.path != path]
            self._update_spectrum_list()
            self._plot_data()
```

- [ ] **Step 4: Call `_update_spectrum_list()` whenever spectra are loaded**

In `_load_files`, change:
```python
        if loaded_any:
            self._update_recent_menu()
            self._plot_data()
```
to:
```python
        if loaded_any:
            self._update_recent_menu()
            self._update_spectrum_list()
            self._plot_data()
```

- [ ] **Step 5: Verify (still expected to fail inside `_plot_data`/`_autoscale_y` until Task 4 — that's fine)**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
app = QApplication([])
w = MainWindow()
w.spectra = []
w._update_spectrum_list()
print('rows after empty update:', w.spectrum_list.count())
"
```
Expected: `rows after empty update: 0` (this only exercises the panel-building/list-refresh code, not the plotting code that Task 4 still needs to fix).

- [ ] **Step 6: Commit**

```bash
git add main_window.py
git commit -m "Add spectrum list panel with visibility checkboxes, color swatches, and remove"
```

---

### Task 4: Overlay plotting and multi-spectrum Y auto-rescale

**Files:**
- Modify: `main_window.py`

- [ ] **Step 1: Rewrite `_plot_data` to draw every visible spectrum**

Change:
```python
    def _plot_data(self):
        self.axes.clear()
        channels = np.arange(len(self.data))
        self.axes.plot(channels, self.data, drawstyle="steps-mid")
        self.axes.set_xlabel("Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.axes.set_yscale("log" if self.log_scale_action.isChecked() else "linear")
        # Channel numbers can't be negative; override Matplotlib's default
        # 5% autoscale margin, which would otherwise show them as such.
        self.axes.set_xlim(0, len(self.data) - 1)
        self.canvas.draw()
        # Our custom zoom bypasses the toolbar's usual box-zoom/pan path, so
        # without this the Home/Back/Forward buttons wouldn't know about
        # this view. Reset the navigation history and record this full view
        # as the new "home" baseline.
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()
```
to:
```python
    def _plot_data(self):
        self.axes.clear()
        visible = [s for s in self.spectra if s.visible]
        for spectrum in visible:
            channels = np.arange(len(spectrum.data))
            self.axes.plot(channels, spectrum.data, drawstyle="steps-mid", color=spectrum.color)
        self.axes.set_xlabel("Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.axes.set_yscale("log" if self.log_scale_action.isChecked() else "linear")
        if visible:
            # Channel numbers can't be negative; override Matplotlib's
            # default 5% autoscale margin, which would otherwise show them
            # as such. Span the widest currently-visible spectrum, since
            # loaded files can have different channel counts.
            max_channel = max(len(s.data) for s in visible) - 1
            self.axes.set_xlim(0, max_channel)
            self._autoscale_y((0, max_channel))
        self.canvas.draw()
        # Our custom zoom bypasses the toolbar's usual box-zoom/pan path, so
        # without this the Home/Back/Forward buttons wouldn't know about
        # this view. Reset the navigation history and record this full view
        # as the new "home" baseline.
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()
```

- [ ] **Step 2: Rewrite `_autoscale_y` to consider every visible spectrum**

Change:
```python
    def _autoscale_y(self, xlim):
        lo = max(0, int(np.floor(xlim[0])))
        hi = min(len(self.data), int(np.ceil(xlim[1])) + 1)
        if lo >= hi:
            return
        visible = self.data[lo:hi]
        y_min = float(np.min(visible))
        y_max = float(np.max(visible))
        if self.log_scale_action.isChecked():
            # A log-scaled axis silently ignores set_ylim() with a
            # non-positive bound, so a plain "y_min - margin" (common
            # here since spectra often have zero-count channels) would
            # leave the Y-axis un-rescaled. Floor to a small positive
            # value and use a multiplicative margin instead.
            y_min = max(y_min, 1.0)
            y_max = max(y_max, y_min * 1.1)
            self.axes.set_ylim(y_min / 1.1, y_max * 1.1)
        else:
            margin = (y_max - y_min) * 0.05 or 1.0
            self.axes.set_ylim(y_min - margin, y_max + margin)
```
to:
```python
    def _autoscale_y(self, xlim):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        lo_bound = max(0, int(np.floor(xlim[0])))
        hi_bound = int(np.ceil(xlim[1])) + 1
        slices = []
        for spectrum in visible:
            lo = min(lo_bound, len(spectrum.data))
            hi = min(hi_bound, len(spectrum.data))
            if lo < hi:
                slices.append(spectrum.data[lo:hi])
        if not slices:
            return
        combined = np.concatenate(slices)
        y_min = float(np.min(combined))
        y_max = float(np.max(combined))
        if self.log_scale_action.isChecked():
            # A log-scaled axis silently ignores set_ylim() with a
            # non-positive bound, so a plain "y_min - margin" (common
            # here since spectra often have zero-count channels) would
            # leave the Y-axis un-rescaled. Floor to a small positive
            # value and use a multiplicative margin instead.
            y_min = max(y_min, 1.0)
            y_max = max(y_max, y_min * 1.1)
            self.axes.set_ylim(y_min / 1.1, y_max * 1.1)
        else:
            margin = (y_max - y_min) * 0.05 or 1.0
            self.axes.set_ylim(y_min - margin, y_max + margin)
```

- [ ] **Step 3: Verify — single spectrum still works, and two spectra overlay correctly**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
app = QApplication([])
w = MainWindow()
w._load_files(['tests/fixtures/test.txt'])
print('single xlim:', w.axes.get_xlim())
print('num lines:', len(w.axes.get_lines()))
w._load_files(['tests/fixtures/test1.txt'])
print('two xlim:', w.axes.get_xlim())
print('num lines:', len(w.axes.get_lines()))
print('colors:', [line.get_color() for line in w.axes.get_lines()])
"
```
Expected:
```
single xlim: (0.0, 4095.0)
num lines: 1
two xlim: (0.0, 4095.0)
num lines: 2
colors: ['#1f77b4', '#ff7f0e']
```

- [ ] **Step 4: Verify combined X-range spans a wider second file**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "
from pathlib import Path
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
app = QApplication([])
w = MainWindow()
w._load_files(['tests/fixtures/test.txt'])
scratch = Path('scratch_wide.txt')
scratch.write_text('\n'.join(str(i % 10) for i in range(5000)) + '\n')
w._load_files([str(scratch)])
print('combined xlim:', w.axes.get_xlim())
scratch.unlink()
"
```
Expected: `combined xlim: (0.0, 8191.0)` (the second file has 5000 data lines, bucketed to 8192 channels per `histogram_io`'s power-of-two rule, wider than `test.txt`'s 4096).

- [ ] **Step 5: Commit**

```bash
git add main_window.py
git commit -m "Draw every visible spectrum overlaid, auto-rescale Y across all of them"
```

---

### Task 5: Zoom, scroll, and Show Full Spectrum across multiple spectra

**Files:**
- Modify: `main_window.py`

- [ ] **Step 1: Update `_on_scroll`'s guard (no longer references `self.data`)**

Change:
```python
    def _on_scroll(self, event):
        if self.data is None or event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)
```
to:
```python
    def _on_scroll(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)
```
(`_zoom_x` itself now checks whether there's anything visible to zoom into — see next step — so this guard only needs to check the mouse event is actually over the axes.)

- [ ] **Step 2: Rewrite `_zoom_x` to compute the max channel across visible spectra**

Change:
```python
    def _zoom_x(self, factor, center=None):
        if self.data is None:
            return
        xlim = self.axes.get_xlim()
        if center is None:
            center = (xlim[0] + xlim[1]) / 2
        half_width = (xlim[1] - xlim[0]) / 2 * factor
        max_channel = len(self.data) - 1
        # Clamp to valid channel numbers -- zooming/scrolling must never
        # show negative channels or channels past the end of the data.
        new_lo = max(0.0, center - half_width)
        new_hi = min(float(max_channel), center + half_width)
        if new_hi <= new_lo:
            new_hi = min(float(max_channel), new_lo + 1)
        new_xlim = (new_lo, new_hi)
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()
```
to:
```python
    def _zoom_x(self, factor, center=None):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        xlim = self.axes.get_xlim()
        if center is None:
            center = (xlim[0] + xlim[1]) / 2
        half_width = (xlim[1] - xlim[0]) / 2 * factor
        max_channel = max(len(s.data) for s in visible) - 1
        # Clamp to valid channel numbers -- zooming/scrolling must never
        # show negative channels or channels past the end of the data.
        new_lo = max(0.0, center - half_width)
        new_hi = min(float(max_channel), center + half_width)
        if new_hi <= new_lo:
            new_hi = min(float(max_channel), new_lo + 1)
        new_xlim = (new_lo, new_hi)
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()
```

- [ ] **Step 3: Rewrite `_show_full_spectrum` the same way**

Change:
```python
    def _show_full_spectrum(self):
        if self.data is None:
            return
        full_xlim = (0, len(self.data) - 1)
        self.axes.set_xlim(full_xlim)
        self._autoscale_y(full_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()
```
to:
```python
    def _show_full_spectrum(self):
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return
        full_xlim = (0, max(len(s.data) for s in visible) - 1)
        self.axes.set_xlim(full_xlim)
        self._autoscale_y(full_xlim)
        self.canvas.draw()
        self.nav_toolbar.push_current()
```

- [ ] **Step 4: Verify zoom and full-spectrum reset work with two spectra of different lengths visible**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "
from pathlib import Path
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
app = QApplication([])
w = MainWindow()
w._load_files(['tests/fixtures/test.txt'])
scratch = Path('scratch_wide2.txt')
scratch.write_text('\n'.join(str(i % 10) for i in range(5000)) + '\n')
w._load_files([str(scratch)])
scratch.unlink()
full_xlim = w.axes.get_xlim()
print('full xlim (should span the wider 8192-channel file):', full_xlim)
w._zoom_x(0.1, center=100)
print('zoomed xlim:', w.axes.get_xlim())
w._show_full_spectrum()
print('restored xlim:', w.axes.get_xlim())
assert w.axes.get_xlim() == full_xlim
print('OK')
"
```
Expected: `full xlim (should span the wider 8192-channel file): (0.0, 8191.0)`, a narrower zoomed xlim, then `restored xlim: (0.0, 8191.0)`, then `OK`.

- [ ] **Step 5: Commit**

```bash
git add main_window.py
git commit -m "Compute zoom/full-spectrum X-range across all visible spectra"
```

---

### Task 6: Per-spectrum status bar readout, log-scale guard update

**Files:**
- Modify: `main_window.py`

- [ ] **Step 1: Rewrite `_on_mouse_move` to show every visible spectrum's count**

Change:
```python
    def _on_mouse_move(self, event):
        if self.data is None or event.inaxes != self.axes or event.xdata is None:
            self.statusBar().clearMessage()
            return
        channel = int(round(event.xdata))
        if 0 <= channel < len(self.data):
            self.statusBar().showMessage(f"Channel: {channel}  Counts: {self.data[channel]}")
        else:
            self.statusBar().clearMessage()
```
to:
```python
    def _on_mouse_move(self, event):
        visible = [s for s in self.spectra if s.visible]
        if not visible or event.inaxes != self.axes or event.xdata is None:
            self.statusBar().clearMessage()
            return
        channel = int(round(event.xdata))
        parts = [f"Channel: {channel}"]
        for spectrum in visible:
            if 0 <= channel < len(spectrum.data):
                parts.append(f"{os.path.basename(spectrum.path)}: {spectrum.data[channel]}")
            else:
                parts.append(f"{os.path.basename(spectrum.path)}: -")
        self.statusBar().showMessage("  |  ".join(parts))
```

- [ ] **Step 2: Update `_on_log_scale_toggled`'s guard**

Change:
```python
    def _on_log_scale_toggled(self, checked):
        if self.data is not None:
            self._plot_data()
```
to:
```python
    def _on_log_scale_toggled(self, checked):
        if self.spectra:
            self._plot_data()
```

- [ ] **Step 3: Verify the status bar shows both spectra, with `-` for a channel past a shorter file's range**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "
from types import SimpleNamespace
from pathlib import Path
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
app = QApplication([])
w = MainWindow()
w._load_files(['tests/fixtures/test.txt'])
scratch = Path('scratch_short.txt')
scratch.write_text('\n'.join(str(i) for i in range(10)) + '\n')
w._load_files([str(scratch)])
scratch.unlink()
w._on_mouse_move(SimpleNamespace(inaxes=w.axes, xdata=10.4))
print(w.statusBar().currentMessage())
w._on_mouse_move(SimpleNamespace(inaxes=w.axes, xdata=2000))
print(w.statusBar().currentMessage())
"
```
Expected:
```
Channel: 10  |  test.txt: 2  |  scratch_short.txt: -
Channel: 2000  |  test.txt: <some count>  |  scratch_short.txt: -
```
(`scratch_short.txt` only has 10 real data values, so it's `-` at channel 10 and beyond, since channel 10 is the 11th value and the file only supplied indices 0-9 before padding to zero — wait, re-check: the fixture writes values `0..9` i.e. 10 values at channels 0-9, so channel 10 itself is already past the real data and within the zero-padded region up to 4096; `histogram_io` pads with zeros, it doesn't shrink the array, so channel 10 is *in range* (value `0`, from padding) not `-`. The `-` case only triggers for channels beyond `len(spectrum.data)` entirely, e.g. channel 2000 for a spectrum bucketed to a smaller channel count than 2000 — but this fixture pads to 4096 (the standard floor bucket), so 2000 is also in range. To actually see a real `-` in this test, the second fixture would need fewer than 2000 data lines *and* still land in the 4096 floor bucket, which channel 2000 is still within. In short: with today's `histogram_io` bucketing (floor of 4096 channels), you won't see `-` unless one visible spectrum has a smaller channel array than another (e.g. impossible today since the minimum bucket is always 4096) — so for this step, just verify the two-spectra message format at channel 10 renders both filenames and counts correctly; skip asserting on `-` here, since achieving a real length mismatch requires the >4096-line scratch fixture from Task 4/5's tests instead. Re-run with that wider fixture if you want to see `-` in action:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "
from types import SimpleNamespace
from pathlib import Path
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
app = QApplication([])
w = MainWindow()
w._load_files(['tests/fixtures/test.txt'])
scratch = Path('scratch_wide3.txt')
scratch.write_text('\n'.join(str(i % 10) for i in range(5000)) + '\n')
w._load_files([str(scratch)])
scratch.unlink()
w._on_mouse_move(SimpleNamespace(inaxes=w.axes, xdata=4200))
print(w.statusBar().currentMessage())
"
```
Expected: `Channel: 4200  |  test.txt: -  |  scratch_wide3.txt: <some count>` (channel 4200 is past `test.txt`'s 4096-channel range but within the wider file's 8192-channel range).)

- [ ] **Step 4: Commit**

```bash
git add main_window.py
git commit -m "Show every visible spectrum's count in the status bar readout"
```

---

### Task 7: Update `scripts/verify_render.py` for the renamed loading method

**Files:**
- Modify: `scripts/verify_render.py`

- [ ] **Step 1: Update the call site**

Change:
```python
def main():
    histogram_path, output_path = sys.argv[1], sys.argv[2]
    app = QApplication([])
    window = MainWindow()
    window._load_file(histogram_path)
    window.figure.savefig(output_path)
```
to:
```python
def main():
    histogram_path, output_path = sys.argv[1], sys.argv[2]
    app = QApplication([])
    window = MainWindow()
    window._load_files([histogram_path])
    window.figure.savefig(output_path)
```

- [ ] **Step 2: Verify it still renders correctly**

Run (write the PNG to a scratch location, not the repo):
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe scripts/verify_render.py tests/fixtures/test.txt "$SCRATCHPAD/render_multi_check.png"
```
Then read the resulting PNG and visually confirm it's the same step-line spectrum plot as before (single file, single color line).

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_render.py
git commit -m "Update verify_render.py for the renamed multi-spectrum loading method"
```

---

### Task 8: Full feature verification (offscreen + visual)

This task is performed directly by whoever is executing the plan (same as the original app's Task 12) — it produces things to actually look at.

**Files:** none (verification only)

- [ ] **Step 1: Offscreen end-to-end check**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
app = QApplication([])
w = MainWindow()
w._load_files(['tests/fixtures/test.txt', 'tests/fixtures/test1.txt'])
print('spectra loaded:', len(w.spectra))
print('list rows:', w.spectrum_list.count())
print('lines drawn:', len(w.axes.get_lines()))

# Uncheck the second spectrum -- only one line should remain visible.
item = w.spectrum_list.item(1)
from PySide6.QtCore import Qt
item.setCheckState(Qt.CheckState.Unchecked)
print('lines drawn after unchecking one:', len(w.axes.get_lines()))

# Recheck it, remove the first spectrum via the data model directly
# (context-menu removal is exercised visually in Step 2, not here).
item.setCheckState(Qt.CheckState.Checked)
w.spectra = [s for s in w.spectra if s.path != 'tests/fixtures/test.txt']
w._update_spectrum_list()
w._plot_data()
print('spectra after removing one:', len(w.spectra))
print('lines drawn after removing one:', len(w.axes.get_lines()))
"
```
Expected:
```
spectra loaded: 2
list rows: 2
lines drawn: 2
lines drawn after unchecking one: 1
spectra after removing one: 1
lines drawn after removing one: 1
```

- [ ] **Step 2: Visual check — render two overlaid spectra to a PNG**

Write a small throwaway script (don't commit it) at `$SCRATCHPAD/verify_overlay.py`:
```python
import sys
sys.path.insert(0, r"C:\Users\RIG\Documents\Claude\PeakFinderFitting\.worktrees\histogram-viewer")
from PySide6.QtWidgets import QApplication
from main_window import MainWindow

app = QApplication([])
w = MainWindow()
w._load_files([
    r"C:\Users\RIG\Documents\Claude\PeakFinderFitting\.worktrees\histogram-viewer\tests\fixtures\test.txt",
    r"C:\Users\RIG\Documents\Claude\PeakFinderFitting\.worktrees\histogram-viewer\tests\fixtures\test1.txt",
])
w.figure.savefig(sys.argv[1])
```
Run: `QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe "$SCRATCHPAD/verify_overlay.py" "$SCRATCHPAD/overlay_check.png"`, then read the PNG and visually confirm two overlapping step-line traces in different colors (they'll look identical in shape since `test.txt`/`test1.txt` are the same data, but should render as two distinctly-colored lines, e.g. by briefly editing the script to load a scratch file with different values for one of them if the overlap makes it hard to tell — use your judgement, the main thing to confirm is two colors are actually present, which `get_lines()` colors already confirmed numerically in Step 1).

- [ ] **Step 3: Launch the real app and screenshot the panel**

Run (adjust temp path as needed), following the same approach used for the original app's launch check:
```powershell
$proc = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "main.py" -PassThru
Start-Sleep -Seconds 3
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$outPath = "$env:TEMP\multi_spectrum_launch.png"
$bmp.Save($outPath)
Stop-Process -Id $proc.Id -Force
Write-Output $outPath
```
Read the resulting PNG and visually confirm the "Loaded Spectra" dock panel appears (even if empty, since no file was opened via this script) docked on the left side of the window.

- [ ] **Step 4: No commit for this task** (verification only, nothing new to commit — Task 7's commit already captured the render script update).

---

### Task 9: Rebuild and verify the Windows installer

**Files:** none (build artifacts only, already gitignored)

- [ ] **Step 1: Run the build**

Run: `powershell -File packaging\windows\build.ps1`
Expected: ends with `Installer built in packaging/windows/output/`. If the Inno Setup compile step fails with an access violation (a known intermittent flakiness in this machine's Inno Setup install, unrelated to the app), just retry: `& "C:\Users\RIG\AppData\Local\Programs\Inno Setup 6\ISCC.exe" "packaging\windows\installer.iss"`.

- [ ] **Step 2: Smoke-test the installer**

Run:
```powershell
$testDir = "$env:TEMP\HVVerifyMulti"
if (Test-Path $testDir) { Remove-Item -Recurse -Force $testDir -Confirm:$false }
Start-Process -FilePath "packaging\windows\output\HistogramViewerSetup.exe" -ArgumentList @("/CURRENTUSER","/VERYSILENT","/DIR=$testDir","/NOICONS") -Wait
Test-Path "$testDir\HistogramViewer.exe"
$proc = Start-Process -FilePath "$testDir\HistogramViewer.exe" -PassThru
Start-Sleep -Seconds 3
$alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($alive) { "Launched OK, no crash" } else { "CRASHED" }
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
```
Expected: `Test-Path` prints `True`, then `Launched OK, no crash`.

- [ ] **Step 3: No commit** (installer binaries are gitignored build artifacts, not tracked).

---

## Self-Review

**Spec coverage:**
- Data model (`LoadedSpectrum`, color cycle) → Task 1
- Multi-select loading, accumulate not replace, duplicate no-op, partial-batch-failure messaging → Task 2
- Spectrum list panel (checkbox visibility, color swatch, remove via context menu) → Task 3
- Overlay plotting in distinct colors, combined X-range across visible spectra → Task 4
- Y auto-rescale across all visible spectra (log-scale-safe, reusing the existing floor/margin logic) → Task 4
- Zoom / scroll / Show Full Spectrum using the combined visible range → Task 5
- Per-spectrum status bar readout with `-` for out-of-range channels → Task 6
- Recent Files / last folder unchanged → Task 2 (untouched aside from being called per-file inside the loop, same as before)
- No on-plot legend, no per-spectrum manual color picker → intentionally not implemented anywhere (matches spec's "Out of scope")
- Rebuilt installer for user testing → Task 9

**Placeholder scan:** No TODOs/TBDs. Task 6's Step 3 has an unusually long inline explanation of why the naive `-` test doesn't trigger with `histogram_io`'s current bucketing, and supplies a corrected command that does — this is deliberate precision, not a placeholder.

**Type consistency:** `LoadedSpectrum(path, data, color)` (Task 1) is constructed identically in `_try_load_spectrum` (Task 2). `spectrum.visible`, `spectrum.path`, `spectrum.data`, `spectrum.color` attribute names are used consistently across `_plot_data`, `_autoscale_y`, `_zoom_x`, `_show_full_spectrum`, `_on_mouse_move`, `_update_spectrum_list`, and `_on_spectrum_item_changed`.
