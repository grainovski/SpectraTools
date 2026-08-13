import os

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matrix_cut import compute_projection
from mtx_io import load_mtx
from theme import style_axes

CUT_REGION_COLOR = "tab:red"
CUT_REGION_ALPHA = 0.25
BG_REGION_COLOR = "tab:green"
BG_REGION_ALPHA = 0.3


class MatrixCutState:
    """Tracks in-progress cut-region/background-region marks on a
    matrix's working projection -- exactly one cut region, any number
    of background regions (unlike fit_mode.py's FitModeState, which
    caps background regions at 2; TV's own philosophy is "the more
    background you mark, the better"). Two-click pairing, modeled on
    FitModeState (fit_mode.py:29-159)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cut_region = None
        self._pending_cut_click = None
        self.bg_regions = []
        self._pending_bg_click = None

    def add_cut_click(self, x):
        if self._pending_cut_click is None:
            self._pending_cut_click = x
            return False
        lo, hi = sorted((self._pending_cut_click, x))
        self.cut_region = (lo, hi)
        self._pending_cut_click = None
        return True

    def add_bg_click(self, x):
        if self._pending_bg_click is None:
            self._pending_bg_click = x
            return False
        lo, hi = sorted((self._pending_bg_click, x))
        self.bg_regions.append((lo, hi))
        self._pending_bg_click = None
        return True

    def clear_cut(self):
        self.cut_region = None
        self._pending_cut_click = None

    def clear_bg(self):
        self.bg_regions = []
        self._pending_bg_click = None


class MatrixCutController(QObject):
    """Held-key + click marker placement for the matrix panel's
    projection view -- hold C for the cut region, hold B for a
    background region. Modeled on FitModeController's eventFilter/
    _held_key mechanism (fit_mode.py:382-428), simplified to two mark
    types instead of three."""

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.state = MatrixCutState()
        self._held_key = None
        self._artists = []
        panel.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        panel.canvas.installEventFilter(self)
        panel.canvas.mpl_connect("figure_enter_event", lambda event: panel.canvas.setFocus())
        panel.canvas.mpl_connect("button_press_event", self.on_click)

    def eventFilter(self, obj, event):
        if obj is self.panel.canvas:
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                if event.key() == Qt.Key.Key_C:
                    self._held_key = "cut"
                elif event.key() == Qt.Key.Key_B:
                    self._held_key = "bg"
            elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
                self._held_key = None
        return False

    def on_click(self, event):
        if event.inaxes != self.panel.axes or event.xdata is None or event.button != 1:
            return
        if self._held_key == "cut":
            self.state.add_cut_click(event.xdata)
        elif self._held_key == "bg":
            self.state.add_bg_click(event.xdata)
        else:
            return
        self._redraw_markers()
        self.panel._update_activate_button()

    def clear(self):
        self.state.reset()
        self._redraw_markers()
        self.panel._update_activate_button()

    def _redraw_markers(self):
        for artist in self._artists:
            try:
                artist.remove()
            except NotImplementedError:
                pass
        self._artists = []

        axes = self.panel.axes
        if self.state.cut_region is not None:
            lo, hi = self.state.cut_region
            self._artists.append(axes.axvspan(lo, hi, color=CUT_REGION_COLOR, alpha=CUT_REGION_ALPHA))
        for lo, hi in self.state.bg_regions:
            self._artists.append(axes.axvspan(lo, hi, color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA))
        self.panel.canvas.draw()


class MatrixPanel(QMainWindow):
    """A separate top-level window for TV-style matrix gate/cut
    analysis. Loads a matrix and computes both its X and Y
    projections up front (this app has the whole matrix in memory,
    unlike TV which required opening a second, transposed matrix file
    for the other axis). The user picks which projection to work on,
    marks a cut region and background regions on it (Task 4), and
    activates the cut to hand a resulting spectrum off to the main
    window. Coexists with MainWindow -- activating one disables, but
    does not close, the other (wired in Task 6)."""

    def __init__(self, main_window, path):
        super().__init__()
        self.main_window = main_window
        self.path = path
        self.matrix = load_mtx(path)
        self.projections = {
            "x": compute_projection(self.matrix, "x"),
            "y": compute_projection(self.matrix, "y"),
        }
        self.working_axis = "x"
        # A list, not a single attribute -- MatrixHeatmapWindow is a
        # parentless, non-modal top-level window, so under Qt's ownership
        # rules the Python-side reference is what keeps it alive. A
        # single attribute would get overwritten (and the previous
        # window's only reference dropped, making it eligible for
        # garbage collection and liable to vanish) if "Show Heatmap..."
        # is clicked again while an earlier heatmap window is still open.
        self._heatmap_windows = []

        self.setWindowTitle(f"Matrix -- {os.path.basename(path)}")
        self.resize(800, 500)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)

        self.axis_selector = QComboBox()
        self.axis_selector.addItem("X projection", "x")
        self.axis_selector.addItem("Y projection", "y")
        self.axis_selector.currentIndexChanged.connect(self._on_axis_changed)

        self.heatmap_button = QPushButton("Show Heatmap...")
        self.heatmap_button.clicked.connect(self._open_heatmap)

        self.activate_cut_button = QPushButton("Activate Cut")
        self.activate_cut_button.setEnabled(False)
        self.activate_cut_button.clicked.connect(self._activate_cut)

        self.clear_marks_button = QPushButton("Clear Marks")
        self.clear_marks_button.clicked.connect(self._clear_marks)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Working on:"))
        top_bar.addWidget(self.axis_selector)
        top_bar.addStretch()
        top_bar.addWidget(self.clear_marks_button)
        top_bar.addWidget(self.activate_cut_button)
        top_bar.addWidget(self.heatmap_button)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(top_bar)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        self.cut_controller = MatrixCutController(self)

        self._plot_projection()

    def _on_axis_changed(self, index):
        self.working_axis = self.axis_selector.itemData(index)
        self.cut_controller.clear()
        self._plot_projection()

    def _plot_projection(self):
        self.axes.clear()
        style_axes(self.axes, self.main_window._theme)
        data = self.projections[self.working_axis]
        self.axes.plot(range(len(data)), data, drawstyle="steps-mid", linewidth=0.8)
        self.axes.set_xlabel(f"{self.working_axis.upper()} channel")
        self.axes.set_ylabel("Counts")
        self.canvas.draw()

    def _clear_marks(self):
        self.cut_controller.clear()

    def _update_activate_button(self):
        self.activate_cut_button.setEnabled(self.cut_controller.state.cut_region is not None)

    def _activate_cut(self):
        from matrix_cut import compute_cut

        state = self.cut_controller.state
        result = compute_cut(self.matrix, self.working_axis, state.cut_region, state.bg_regions)
        label = (
            f"{os.path.basename(self.path)} {self.working_axis} cut "
            f"[{state.cut_region[0]:.1f}, {state.cut_region[1]:.1f}]"
        )
        self.main_window._add_combined_spectrum(label, result)

    def _open_heatmap(self):
        from matrix_heatmap import MatrixHeatmapWindow

        window = MatrixHeatmapWindow(self.matrix, self.path, self.main_window._theme, self)
        self._heatmap_windows.append(window)
        window.show()

    def showEvent(self, event):
        super().showEvent(event)
        # Same rationale as MainWindow.showEvent: hold-C/hold-B marking
        # depends on the canvas holding keyboard focus, and the top bar's
        # combo box/buttons are focusable widgets that can claim initial
        # focus on some window managers before the user has ever hovered
        # the canvas (figure_enter_event alone wouldn't cover that case).
        self.canvas.setFocus()

    def _on_activated(self):
        self.main_window.setEnabled(False)
        # Generalizes MainWindow._on_activated's own loop to the
        # multi-panel case: with two or more matrix panels open,
        # activating this one must disable every OTHER panel too, not
        # just MainWindow -- otherwise two panels could end up enabled
        # simultaneously, breaking the "activating one disables the
        # other" guarantee for anything beyond exactly one panel.
        for panel in self.main_window._matrix_panels:
            if panel is not self:
                panel.setEnabled(False)
        self.setEnabled(True)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowActivate:
            self._on_activated()

    def closeEvent(self, event):
        # Same parentless-top-level-window concern as MainWindow.closeEvent:
        # a MatrixHeatmapWindow left open would keep the app from quitting
        # even after this panel (and MainWindow) are both closed.
        for window in list(self._heatmap_windows):
            window.close()
        super().closeEvent(event)
        if self in self.main_window._matrix_panels:
            self.main_window._matrix_panels.remove(self)
        # Only fall back to re-enabling MainWindow if nothing else in the
        # exclusivity group is currently enabled. With a single panel this
        # is always true (MainWindow was the one disabled), but with two
        # or more panels open, closing a PANEL THAT WASN'T THE ACTIVE ONE
        # must not steal focus-exclusivity away from whichever window
        # (MainWindow or another panel) is actually still active.
        others_enabled = self.main_window.isEnabled() or any(
            panel.isEnabled() for panel in self.main_window._matrix_panels
        )
        if not others_enabled:
            self.main_window.setEnabled(True)
