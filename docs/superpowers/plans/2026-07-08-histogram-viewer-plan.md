# Histogram Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cross-platform (Windows + Linux) PySide6/Matplotlib desktop app that opens an ASCII histogram `.txt` file and plots it, with X-axis zoom (scroll wheel + toolbar buttons) and automatic Y-axis rescaling, then package it into a Windows installer and a Linux AppImage.

**Architecture:** `histogram_io.py` parses the file into a numpy array (power-of-two channel bucketing, per spec). `main_window.py` (`MainWindow`, a `QMainWindow`) owns the embedded Matplotlib canvas, menu, toolbar, and status bar. `settings.py` wraps `QSettings` for last-folder/recent-files persistence. `main.py` is the entry point. Packaging scripts under `packaging/windows/` and `packaging/linux/` wrap PyInstaller builds into an Inno Setup installer and an AppImage respectively.

**Tech Stack:** Python 3.13, PySide6, Matplotlib (`QtAgg` backend), numpy, pytest (tests), PyInstaller + Inno Setup (Windows packaging), PyInstaller + AppImage (Linux packaging, built via WSL Ubuntu-24.04).

**Reference spec:** `docs/superpowers/specs/2026-07-08-histogram-viewer-design.md`

---

## Notes for whoever executes this plan

- All Python commands below assume a project-local virtualenv at `.venv` (created in Task 1) and are invoked as `.venv/Scripts/python.exe ...` (Windows). Run everything from the repo root: `C:\Users\RIG\Documents\Claude\PeakFinderFitting`.
- Verification commands that set `QT_QPA_PLATFORM=offscreen` let Qt run without a visible window — use this prefix for automated checks. Tasks that need a real, visible window (packaging smoke tests, the final screenshot check) omit it.
- `main_window.py` has no automated GUI test suite (decided in the spec — a single-window app doesn't warrant `pytest-qt`). Instead, each GUI task ends with a short one-off verification script (not a permanent test file) that constructs the window offscreen and asserts on its state. This catches wiring mistakes cheaply without building a GUI test framework.
- Task 13 (manual interactive smoke test) cannot be performed by an agent — there's no tool available here to simulate mouse scroll/click on a native desktop window. That task's steps are a checklist for a human to run.
- Installing Inno Setup (Task 14) and WSL packages (Task 15) modifies the host machine. If a `winget`/`apt` command prompts for elevation and hangs, stop and ask the user to run that one command interactively, then resume.

---

### Task 1: Project scaffolding

**Files:**
- Modify: `.gitignore`
- Create: `conftest.py`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`

- [ ] **Step 1: Extend `.gitignore` for venvs and packaging outputs**

Replace the whole file with:

```
__pycache__/
*.pyc
.venv/
.venv-linux/
venv/
build/
dist/
*.spec
!packaging/**/*.spec
*.egg-info/
.pytest_cache/
packaging/windows/output/
packaging/linux/output/
packaging/linux/tools/
*.AppDir/
```

- [ ] **Step 2: Create `conftest.py`** (repo root — guarantees `histogram_io`, `main_window`, `settings` are importable from `tests/` regardless of how pytest is invoked)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

- [ ] **Step 3: Create `requirements.txt`**

```
PySide6>=6.6
matplotlib>=3.8
numpy>=1.26
```

- [ ] **Step 4: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest>=8.0
pyinstaller>=6.3
```

- [ ] **Step 5: Create the virtualenv and install dependencies**

Run:
```
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```
Expected: install completes with no errors. Verify with:
```
.venv/Scripts/python.exe -m pip list
```
Expected: output includes `PySide6`, `matplotlib`, `numpy`, `pytest`, `pyinstaller`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore conftest.py requirements.txt requirements-dev.txt
git commit -m "Add project scaffolding: venv requirements, gitignore, conftest"
```

---

### Task 2: `histogram_io.py` — core parser with power-of-two channel bucketing

**Files:**
- Create: `histogram_io.py`
- Create: `tests/fixtures/test.txt`
- Create: `tests/fixtures/test1.txt`
- Create: `tests/test_histogram_io.py`

- [ ] **Step 1: Copy the sample fixture files**

Run:
```
mkdir -p tests/fixtures
cp test.txt tests/fixtures/test.txt
cp test1.txt tests/fixtures/test1.txt
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_histogram_io.py`:

```python
from pathlib import Path

from histogram_io import load_histogram

FIXTURES = Path(__file__).parent / "fixtures"


def test_parses_test_txt_to_4096_channels():
    data = load_histogram(str(FIXTURES / "test.txt"))
    assert len(data) == 4096
    assert list(data[:5]) == [4, 0, 1, 0, 1]


def test_parses_test1_txt_same_as_test_txt():
    data = load_histogram(str(FIXTURES / "test1.txt"))
    assert len(data) == 4096
    assert list(data[:5]) == [4, 0, 1, 0, 1]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_histogram_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'histogram_io'`.

- [ ] **Step 4: Implement `histogram_io.py`**

```python
import numpy as np


class ParseError(Exception):
    """Raised when a file contains no parseable histogram data."""


def _bucket_channel_count(n: int) -> int:
    size = 4096
    while size < n:
        size *= 2
    return size


def load_histogram(path: str) -> np.ndarray:
    values = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values.append(int(stripped))
            except ValueError:
                continue

    if not values:
        raise ParseError(f"No histogram data found in file: {path}")

    channel_count = _bucket_channel_count(len(values))
    data = np.zeros(channel_count, dtype=np.int64)
    data[: len(values)] = values
    return data
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_histogram_io.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add histogram_io.py tests/test_histogram_io.py tests/fixtures/test.txt tests/fixtures/test1.txt
git commit -m "Add histogram_io parser with power-of-two channel bucketing"
```

---

### Task 3: `histogram_io` regression tests — no-header file and blank lines

**Files:**
- Create: `tests/fixtures/no_header.txt`
- Create: `tests/fixtures/blank_lines.txt`
- Modify: `tests/test_histogram_io.py`

- [ ] **Step 1: Create `tests/fixtures/no_header.txt`**

```
4
0
1
0
1
0
```

- [ ] **Step 2: Create `tests/fixtures/blank_lines.txt`**

```
#:MatrixFormat:4k.txt:1
4

0
1
```

- [ ] **Step 3: Add regression tests**

Append to `tests/test_histogram_io.py`:

```python


def test_handles_file_without_header_line():
    data = load_histogram(str(FIXTURES / "no_header.txt"))
    assert len(data) == 4096
    assert list(data[:6]) == [4, 0, 1, 0, 1, 0]
    assert list(data[6:]) == [0] * (4096 - 6)


def test_handles_blank_lines_interspersed():
    data = load_histogram(str(FIXTURES / "blank_lines.txt"))
    assert len(data) == 4096
    assert list(data[:3]) == [4, 0, 1]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_histogram_io.py -v`
Expected: 4 passed. (No production code changes were needed — the generic "skip lines that don't parse as a number" logic from Task 2 already covers both cases. This step is a regression-test addition confirming that.)

- [ ] **Step 5: Commit**

```bash
git add tests/test_histogram_io.py tests/fixtures/no_header.txt tests/fixtures/blank_lines.txt
git commit -m "Add regression tests for headerless files and blank lines"
```

---

### Task 4: `histogram_io` regression test — `ParseError` on no numeric data

**Files:**
- Create: `tests/fixtures/no_numeric_data.txt`
- Modify: `tests/test_histogram_io.py`

- [ ] **Step 1: Create `tests/fixtures/no_numeric_data.txt`**

```
#:MatrixFormat:4k.txt:1
#comment only
```

- [ ] **Step 2: Add the test**

Append to `tests/test_histogram_io.py` (add `import pytest` at the top of the file first, alongside the existing `from pathlib import Path` line):

```python
import pytest
```

Then append:

```python


def test_raises_parse_error_on_no_numeric_data():
    with pytest.raises(ParseError):
        load_histogram(str(FIXTURES / "no_numeric_data.txt"))
```

Also update the existing import line from:
```python
from histogram_io import load_histogram
```
to:
```python
from histogram_io import ParseError, load_histogram
```

- [ ] **Step 3: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_histogram_io.py -v`
Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_histogram_io.py tests/fixtures/no_numeric_data.txt
git commit -m "Add regression test for ParseError on files with no numeric data"
```

---

### Task 5: `histogram_io` regression tests — bucket-size edge cases

**Files:**
- Modify: `tests/test_histogram_io.py`

- [ ] **Step 1: Add the tests**

Append to `tests/test_histogram_io.py`:

```python


def test_bucket_under_4096_pads_to_4096(tmp_path):
    file_path = tmp_path / "short.txt"
    values = list(range(10))
    file_path.write_text("\n".join(str(v) for v in values) + "\n")

    data = load_histogram(str(file_path))

    assert len(data) == 4096
    assert list(data[:10]) == values
    assert list(data[10:]) == [0] * (4096 - 10)


def test_bucket_between_4096_and_8192_pads_to_8192(tmp_path):
    file_path = tmp_path / "medium.txt"
    values = [i % 10 for i in range(5000)]
    file_path.write_text("\n".join(str(v) for v in values) + "\n")

    data = load_histogram(str(file_path))

    assert len(data) == 8192
    assert list(data[:5000]) == values
    assert list(data[5000:]) == [0] * (8192 - 5000)
```

- [ ] **Step 2: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_histogram_io.py -v`
Expected: 7 passed. (Again, no production code change — this confirms `_bucket_channel_count`'s doubling loop from Task 2 already satisfies both bucket cases, plus the "exact 4096 stays at 4096" rule already covered by Task 2's `test_parses_test_txt_to_4096_channels`.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_histogram_io.py
git commit -m "Add regression tests for channel bucket-size boundaries"
```

---

### Task 6: `settings.py` — last folder + recent files persistence

**Files:**
- Create: `settings.py`

- [ ] **Step 1: Implement `settings.py`**

```python
from PySide6.QtCore import QSettings

ORG_NAME = "PeakFinderFitting"
APP_NAME = "HistogramViewer"
MAX_RECENT_FILES = 8


class Settings:
    def __init__(self):
        self._settings = QSettings(ORG_NAME, APP_NAME)

    def last_folder(self) -> str:
        value = self._settings.value("last_folder", "")
        return value if isinstance(value, str) else ""

    def set_last_folder(self, folder: str) -> None:
        self._settings.setValue("last_folder", folder)
        self._settings.sync()

    def recent_files(self) -> list:
        value = self._settings.value("recent_files", [])
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    def add_recent_file(self, path: str) -> None:
        files = self.recent_files()
        if path in files:
            files.remove(path)
        files.insert(0, path)
        self._settings.setValue("recent_files", files[:MAX_RECENT_FILES])
        self._settings.sync()

    def remove_recent_file(self, path: str) -> None:
        files = self.recent_files()
        if path in files:
            files.remove(path)
            self._settings.setValue("recent_files", files)
            self._settings.sync()
```

- [ ] **Step 2: Verify it works**

Run:
```
.venv/Scripts/python.exe -c "from settings import Settings; s = Settings(); s.set_last_folder('C:/tmp'); print(s.last_folder()); s.add_recent_file('a.txt'); s.add_recent_file('b.txt'); print(s.recent_files())"
```
Expected output:
```
C:/tmp
['b.txt', 'a.txt']
```
(This writes to `HKEY_CURRENT_USER\Software\PeakFinderFitting\HistogramViewer` — the same registry location the real app will use. That's expected; it's exactly the persistence behavior being verified.)

- [ ] **Step 3: Commit**

```bash
git add settings.py
git commit -m "Add Settings wrapper for last-folder and recent-files persistence"
```

---

### Task 7: `main_window.py` skeleton — window, Open dialog, plot; `main.py` entry point

**Files:**
- Create: `main_window.py`
- Create: `main.py`

- [ ] **Step 1: Implement `main_window.py`**

```python
import os

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from histogram_io import ParseError, load_histogram


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histogram Viewer")
        self.resize(900, 600)

        self.data = None

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        self._build_menu()

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Histogram", "", "Text files (*.txt);;All files (*)"
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
        self._plot_data()

    def _plot_data(self):
        self.axes.clear()
        channels = np.arange(len(self.data))
        self.axes.plot(channels, self.data, drawstyle="steps-mid")
        self.axes.set_xlabel("Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.canvas.draw()
```

- [ ] **Step 2: Implement `main.py`**

```python
import sys

import matplotlib

matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify it constructs and loads a file without error**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "from PySide6.QtWidgets import QApplication; from main_window import MainWindow; app = QApplication([]); w = MainWindow(); w._load_file('tests/fixtures/test.txt'); assert w.data is not None and len(w.data) == 4096; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add main_window.py main.py
git commit -m "Add MainWindow skeleton with Open dialog and histogram plot"
```

---

### Task 8: Status bar channel/count readout

**Files:**
- Modify: `main_window.py`

- [ ] **Step 1: Connect the mouse-move event**

In `__init__`, change:
```python
        self._build_menu()
```
to:
```python
        self._build_menu()

        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
```

- [ ] **Step 2: Add the handler**

Append this method to `MainWindow` (e.g. after `_plot_data`):

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

- [ ] **Step 3: Verify**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "from types import SimpleNamespace; from PySide6.QtWidgets import QApplication; from main_window import MainWindow; app = QApplication([]); w = MainWindow(); w._load_file('tests/fixtures/test.txt'); w._on_mouse_move(SimpleNamespace(inaxes=w.axes, xdata=10.4)); print(w.statusBar().currentMessage())"
```
Expected: `Channel: 10  Counts: 2`

- [ ] **Step 4: Commit**

```bash
git add main_window.py
git commit -m "Add status bar channel/count readout on mouse move"
```

---

### Task 9: Log scale Y toggle

**Files:**
- Modify: `main_window.py`

- [ ] **Step 1: Add the View menu**

In `_build_menu`, change:
```python
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
```
to:
```python
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = self.menuBar().addMenu("&View")
        self.log_scale_action = QAction("Log scale Y", self)
        self.log_scale_action.setCheckable(True)
        self.log_scale_action.toggled.connect(self._on_log_scale_toggled)
        view_menu.addAction(self.log_scale_action)
```

- [ ] **Step 2: Use it in `_plot_data` and add the toggle handler**

In `_plot_data`, change:
```python
        self.axes.set_xlabel("Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.canvas.draw()
```
to:
```python
        self.axes.set_xlabel("Channel")
        self.axes.set_ylabel("Counts")
        self.axes.grid(True)
        self.axes.set_yscale("log" if self.log_scale_action.isChecked() else "linear")
        self.canvas.draw()
```

Append this method to `MainWindow`:

```python

    def _on_log_scale_toggled(self, checked):
        if self.data is not None:
            self._plot_data()
```

- [ ] **Step 3: Verify**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "from PySide6.QtWidgets import QApplication; from main_window import MainWindow; app = QApplication([]); w = MainWindow(); w._load_file('tests/fixtures/test.txt'); w.log_scale_action.setChecked(True); print(w.axes.get_yscale())"
```
Expected: `log`

- [ ] **Step 4: Commit**

```bash
git add main_window.py
git commit -m "Add Log scale Y toggle to View menu"
```

---

### Task 10: Recent Files menu + last-folder persistence

**Files:**
- Modify: `main_window.py`

- [ ] **Step 1: Import `Settings` and construct it**

At the top of `main_window.py`, change:
```python
from histogram_io import ParseError, load_histogram
```
to:
```python
from histogram_io import ParseError, load_histogram
from settings import Settings
```

In `__init__`, change:
```python
        self._build_menu()

        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
```
to:
```python
        self.settings = Settings()

        self._build_menu()
        self._update_recent_menu()

        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
```

- [ ] **Step 2: Add the Recent Files submenu**

In `_build_menu`, change:
```python
        file_menu.addAction(open_action)

        file_menu.addSeparator()
```
to:
```python
        file_menu.addAction(open_action)

        self.recent_menu = file_menu.addMenu("Recent Files")

        file_menu.addSeparator()
```

- [ ] **Step 3: Use the last folder in the Open dialog, and persist on successful load**

Change:
```python
    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Histogram", "", "Text files (*.txt);;All files (*)"
        )
        if path:
            self._load_file(path)
```
to:
```python
    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Histogram", self.settings.last_folder(), "Text files (*.txt);;All files (*)"
        )
        if path:
            self._load_file(path)
```

Change:
```python
        self.data = data
        self.setWindowTitle(f"Histogram Viewer - {os.path.basename(path)}")
        self._plot_data()
```
to:
```python
        self.data = data
        self.setWindowTitle(f"Histogram Viewer - {os.path.basename(path)}")
        self.settings.set_last_folder(os.path.dirname(path))
        self.settings.add_recent_file(path)
        self._update_recent_menu()
        self._plot_data()
```

- [ ] **Step 4: Add the menu-refresh and recent-file-open handlers**

Append these methods to `MainWindow`:

```python

    def _update_recent_menu(self):
        self.recent_menu.clear()
        for path in self.settings.recent_files():
            action = QAction(path, self)
            action.triggered.connect(lambda checked=False, p=path: self._open_recent(p))
            self.recent_menu.addAction(action)

    def _open_recent(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "File not found", f"{path} no longer exists.")
            self.settings.remove_recent_file(path)
            self._update_recent_menu()
            return
        self._load_file(path)
```

- [ ] **Step 5: Verify**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "from PySide6.QtWidgets import QApplication; from main_window import MainWindow; app = QApplication([]); w = MainWindow(); w._load_file('tests/fixtures/test.txt'); print([a.text() for a in w.recent_menu.actions()])"
```
Expected: a one-item list containing `tests/fixtures/test.txt` (or its resolved path).

- [ ] **Step 6: Commit**

```bash
git add main_window.py
git commit -m "Add Recent Files menu and last-folder persistence"
```

---

### Task 11: X-axis zoom (scroll wheel + toolbar buttons) with Y auto-rescale

**Files:**
- Modify: `main_window.py`

- [ ] **Step 1: Add the zoom factor constant**

At the top of `main_window.py`, change:
```python
from histogram_io import ParseError, load_histogram
from settings import Settings
```
to:
```python
from histogram_io import ParseError, load_histogram
from settings import Settings

ZOOM_FACTOR = 1.5
```

- [ ] **Step 2: Wire up the scroll event and the zoom toolbar buttons**

In `__init__`, change:
```python
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
```
to:
```python
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._build_zoom_buttons()
```

- [ ] **Step 3: Implement the zoom methods**

Append these methods to `MainWindow`:

```python

    def _build_zoom_buttons(self):
        self.nav_toolbar.addSeparator()

        zoom_in_action = QAction("Zoom In X", self)
        zoom_in_action.triggered.connect(lambda: self._zoom_x(1 / ZOOM_FACTOR))
        self.nav_toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out X", self)
        zoom_out_action.triggered.connect(lambda: self._zoom_x(ZOOM_FACTOR))
        self.nav_toolbar.addAction(zoom_out_action)

    def _on_scroll(self, event):
        if self.data is None or event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)

    def _zoom_x(self, factor, center=None):
        if self.data is None:
            return
        xlim = self.axes.get_xlim()
        if center is None:
            center = (xlim[0] + xlim[1]) / 2
        half_width = (xlim[1] - xlim[0]) / 2 * factor
        new_xlim = (center - half_width, center + half_width)
        self.axes.set_xlim(new_xlim)
        self._autoscale_y(new_xlim)
        self.canvas.draw()

    def _autoscale_y(self, xlim):
        lo = max(0, int(np.floor(xlim[0])))
        hi = min(len(self.data), int(np.ceil(xlim[1])) + 1)
        if lo >= hi:
            return
        visible = self.data[lo:hi]
        y_min = float(np.min(visible))
        y_max = float(np.max(visible))
        margin = (y_max - y_min) * 0.05 or 1.0
        self.axes.set_ylim(y_min - margin, y_max + margin)
```

- [ ] **Step 4: Verify**

Run:
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "from PySide6.QtWidgets import QApplication; from main_window import MainWindow; app = QApplication([]); w = MainWindow(); w._load_file('tests/fixtures/test.txt'); before = w.axes.get_xlim(); w._zoom_x(0.5, center=100); after = w.axes.get_xlim(); print(before, after, w.axes.get_ylim()); assert after[1] - after[0] < before[1] - before[0]; print('OK')"
```
Expected: prints the before/after xlim tuples and a ylim tuple, then `OK`.

- [ ] **Step 5: Commit**

```bash
git add main_window.py
git commit -m "Add X-axis zoom (scroll wheel + toolbar buttons) with Y auto-rescale"
```

---

### Task 12: Automated visual verification (render check + live-window screenshot)

This task is performed directly by whoever (or whichever agent) is executing the plan — it produces images to actually look at, since there's no automated GUI test suite to lean on.

**Files:**
- Create: `scripts/verify_render.py`

- [ ] **Step 1: Write the render-verification script**

```python
"""One-off dev helper: render a histogram file to a PNG for visual sanity-checking.

Usage: QT_QPA_PLATFORM=offscreen python scripts/verify_render.py <histogram.txt> <output.png>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    histogram_path, output_path = sys.argv[1], sys.argv[2]
    app = QApplication([])
    window = MainWindow()
    window._load_file(histogram_path)
    window.figure.savefig(output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Render `tests/fixtures/test.txt` and view it**

Run (write the PNG to the scratchpad directory, not the repo):
```
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe scripts/verify_render.py tests/fixtures/test.txt "$SCRATCHPAD/render.png"
```
Then read the resulting PNG and visually confirm: a step-line plot, "Channel" on the x-axis, "Counts" on the y-axis, shape consistent with the counts in `test.txt` (small values near channel 0, generally larger further along).

- [ ] **Step 3: Launch the real (visible) app and screenshot it**

Run in PowerShell (adjust `$env:TEMP` path as needed):
```powershell
$proc = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "main.py" -PassThru
Start-Sleep -Seconds 3
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$outPath = "$env:TEMP\histogram_viewer_launch.png"
$bmp.Save($outPath)
Stop-Process -Id $proc.Id -Force
Write-Output $outPath
```
Then read the resulting PNG and visually confirm the window opened with its menu bar (File, View), the Matplotlib toolbar (including the two Zoom In X / Zoom Out X buttons), and an empty plot area (no file loaded yet — this checks the app launches cleanly, not the plot itself).

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_render.py
git commit -m "Add one-off render-verification script for GUI sanity-checking"
```

---

### Task 13: Manual interactive smoke test (human-run checklist)

No agent tool here can simulate mouse scroll/click on a native desktop window, so this task is a checklist for a person to run through by hand. Whoever is executing this plan should present it to the user rather than trying to automate it.

- [ ] Run `.venv/Scripts/python.exe main.py`
- [ ] File → Open… → select `test.txt` from the repo root → confirm the plot renders
- [ ] Toggle View → Log scale Y → confirm the y-axis switches to log scale and back
- [ ] Hover the mouse over the plot → confirm the status bar shows `Channel: N  Counts: M` and updates as you move
- [ ] Scroll the mouse wheel over the plot → confirm the x-axis zooms in/out centered on the cursor, and the y-axis rescales to fit what's visible
- [ ] Click the "Zoom In X" / "Zoom Out X" toolbar buttons → confirm the same zoom + auto-rescale behavior, centered on the current view
- [ ] Open File → Recent Files → confirm `test.txt` is listed; click it → confirm it reloads
- [ ] Close the app and relaunch it → confirm File → Recent Files still shows `test.txt` (persistence survived restart)

If any step fails, note which one and go back to the corresponding task above.

---

### Task 14: Windows packaging — installer via PyInstaller + Inno Setup

**Files:**
- Create: `packaging/windows/build.ps1`
- Create: `packaging/windows/installer.iss`

- [ ] **Step 1: Check for Inno Setup, install if missing**

Run:
```powershell
Get-Command ISCC.exe -ErrorAction SilentlyContinue
```
If nothing is found, install it:
```powershell
winget install --id JRSoftware.InnoSetup -e
```
If this prompts for elevation and hangs (no interactive input is available here), stop and ask the user to run that one command themselves, then resume once `ISCC.exe` is on `PATH` (default install location is `C:\Program Files (x86)\Inno Setup 6\ISCC.exe`).

- [ ] **Step 2: Write `packaging/windows/installer.iss`**

```
[Setup]
AppName=Histogram Viewer
AppVersion=0.1.0
DefaultDirName={autopf}\HistogramViewer
DefaultGroupName=Histogram Viewer
OutputDir=output
OutputBaseFilename=HistogramViewerSetup
Compression=lzma
SolidCompression=yes

[Files]
Source: "..\..\dist\HistogramViewer.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Histogram Viewer"; Filename: "{app}\HistogramViewer.exe"
Name: "{autodesktop}\Histogram Viewer"; Filename: "{app}\HistogramViewer.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
```

- [ ] **Step 3: Write `packaging/windows/build.ps1`**

```powershell
$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/../.."
Set-Location $root

& "$root/.venv/Scripts/python.exe" -m PyInstaller --noconfirm --onefile --windowed --name HistogramViewer main.py

$iscc = (Get-Command ISCC.exe).Source
& $iscc "packaging/windows/installer.iss"

Write-Host "Installer built in packaging/windows/output/"
```

- [ ] **Step 4: Run the build**

Run: `powershell -File packaging\windows\build.ps1`
Expected: PyInstaller produces `dist/HistogramViewer.exe`, then Inno Setup produces `packaging/windows/output/HistogramViewerSetup.exe`. Verify:
```powershell
Test-Path packaging\windows\output\HistogramViewerSetup.exe
```
Expected: `True`

- [ ] **Step 5: Smoke-test the installer**

Run a silent install to a throwaway directory, confirm it launches, then clean up:
```powershell
$testDir = "$env:TEMP\HistogramViewerTest"
Start-Process -FilePath "packaging\windows\output\HistogramViewerSetup.exe" -ArgumentList "/VERYSILENT", "/DIR=$testDir", "/NOICONS" -Wait
Test-Path "$testDir\HistogramViewer.exe"
$proc = Start-Process -FilePath "$testDir\HistogramViewer.exe" -PassThru
Start-Sleep -Seconds 3
Stop-Process -Id $proc.Id -Force
Remove-Item -Recurse -Force $testDir
```
Expected: `Test-Path` prints `True` and the process starts without error (no crash dialog).

- [ ] **Step 6: Commit**

```bash
git add packaging/windows/build.ps1 packaging/windows/installer.iss
git commit -m "Add Windows installer packaging (PyInstaller + Inno Setup)"
```

---

### Task 15: Linux packaging — AppImage via PyInstaller (built in WSL Ubuntu-24.04)

**Files:**
- Create: `packaging/linux/build.sh`
- Create: `packaging/linux/histogramviewer.desktop`
- Create: `packaging/linux/make_icon.py`
- Create: `packaging/linux/icon.png` (generated by the script above, then committed as a static asset)

- [ ] **Step 1: Ensure pip and a build venv exist in WSL**

Run:
```bash
wsl -d Ubuntu-24.04 -- bash -lc 'python3 -m ensurepip --upgrade && python3 -m pip --version'
```
Expected: prints a pip version.

Run:
```bash
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting && python3 -m venv .venv-linux && .venv-linux/bin/pip install -r requirements-dev.txt'
```
Expected: install completes with no errors.

- [ ] **Step 2: Write `packaging/linux/make_icon.py`**

```python
"""One-off script: generate a simple icon for the AppImage using matplotlib
(avoids adding Pillow as a dependency just for icon generation)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
heights = [0.2, 0.5, 0.9, 0.6, 0.3, 0.7, 0.4]
ax.bar(range(len(heights)), heights, color="#2b7de9", width=0.8)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.axis("off")
fig.tight_layout(pad=0)
fig.savefig("packaging/linux/icon.png", dpi=100)
```

- [ ] **Step 3: Generate the icon**

Run:
```bash
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting && .venv-linux/bin/python packaging/linux/make_icon.py'
```
Expected: `packaging/linux/icon.png` is created.

- [ ] **Step 4: Write `packaging/linux/histogramviewer.desktop`**

```
[Desktop Entry]
Type=Application
Name=Histogram Viewer
Exec=HistogramViewer
Icon=histogramviewer
Categories=Science;
```

- [ ] **Step 5: Download `appimagetool`**

Run:
```bash
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting && mkdir -p packaging/linux/tools && curl -L -o packaging/linux/tools/appimagetool https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage && chmod +x packaging/linux/tools/appimagetool'
```
Expected: file downloaded and made executable.

- [ ] **Step 6: Write `packaging/linux/build.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

.venv-linux/bin/pyinstaller --noconfirm --onedir --windowed --name HistogramViewer main.py

APPDIR="packaging/linux/HistogramViewer.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp -r dist/HistogramViewer/* "$APPDIR/usr/bin/"

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/HistogramViewer" "$@"
EOF
chmod +x "$APPDIR/AppRun"

cp packaging/linux/histogramviewer.desktop "$APPDIR/histogramviewer.desktop"
cp packaging/linux/icon.png "$APPDIR/histogramviewer.png"

mkdir -p packaging/linux/output
packaging/linux/tools/appimagetool --appimage-extract-and-run "$APPDIR" packaging/linux/output/HistogramViewer-x86_64.AppImage

echo "AppImage built at packaging/linux/output/HistogramViewer-x86_64.AppImage"
```

- [ ] **Step 7: Run the build**

Run:
```bash
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting && bash packaging/linux/build.sh'
```
Expected: ends with `AppImage built at packaging/linux/output/HistogramViewer-x86_64.AppImage`. If `appimagetool` fails complaining about FUSE, retry the invocation inside `build.sh` — it already uses `--appimage-extract-and-run`, which bypasses the need for a working FUSE mount in WSL.

- [ ] **Step 8: Smoke-test the AppImage**

Run:
```bash
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting && QT_QPA_PLATFORM=offscreen timeout 5 packaging/linux/output/HistogramViewer-x86_64.AppImage --appimage-extract-and-run; echo "exit code: $?"'
```
Expected: the process starts and runs until `timeout` kills it after 5 seconds (exit code 124), rather than exiting immediately with an error — confirming the bundled app launches.

- [ ] **Step 9: Commit**

```bash
git add packaging/linux/build.sh packaging/linux/histogramviewer.desktop packaging/linux/make_icon.py packaging/linux/icon.png
git commit -m "Add Linux AppImage packaging (PyInstaller, built via WSL)"
```

---

### Task 16: README and final wrap-up

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Histogram Viewer

A cross-platform desktop app (Windows + Linux) for opening an ASCII
histogram `.txt` file and plotting it — the foundation of the
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
  Produces `packaging/windows/output/HistogramViewerSetup.exe`.
- Linux: `bash packaging/linux/build.sh`, built and tested via WSL Ubuntu
  in this repo's history. Produces
  `packaging/linux/output/HistogramViewer-x86_64.AppImage`.

## File format

See `docs/superpowers/specs/2026-07-08-histogram-viewer-design.md` for the
full design spec, including the ASCII file format and channel-count
bucketing rules.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Add README with setup, run, test, and packaging instructions"
```

---

## Self-Review

**Spec coverage:**
- Ascii format + header tolerance → Task 2, 3
- Power-of-two channel bucketing (incl. exact-fit boundary) → Task 2, 5
- Tech stack (PySide6 + Matplotlib QtAgg) → Task 7
- UI layout (File/View menus, canvas, toolbar, status bar) → Tasks 7, 8, 9, 10
- X-zoom (scroll + buttons) with Y auto-rescale → Task 11
- Error handling (unreadable file, no data, missing recent file) → Tasks 7, 10
- Settings (last folder, recent files) → Tasks 6, 10
- Unit tests per spec's Testing section → Tasks 2–5
- Manual GUI smoke test per spec → Task 13
- Windows/Linux installer packaging → Tasks 14, 15
- Out-of-scope items (`.spe`, peak fitting, overlay, automated GUI tests) → intentionally not implemented; noted only in the spec.

**Placeholder scan:** No TODOs/TBDs; every step has complete, runnable code or an exact command with expected output.

**Type consistency:** `load_histogram(path) -> np.ndarray` (Task 2) is used identically in `main_window.py` (Task 7) and `scripts/verify_render.py` (Task 12). `Settings` method names (`last_folder`, `set_last_folder`, `recent_files`, `add_recent_file`, `remove_recent_file` — Task 6) match their call sites in `main_window.py` (Task 10) exactly.
