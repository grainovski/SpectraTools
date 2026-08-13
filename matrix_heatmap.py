import os

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from theme import style_axes

_MAX_DISPLAY_DIM = 1024


def _downsample_for_display(matrix, max_dim=_MAX_DISPLAY_DIM):
    """Block-sums `matrix` down to at most `max_dim` in each dimension
    for display purposes only -- the heatmap is explicitly a visual
    overview, not a precision tool (markers/gating always happen on
    the full-resolution projection, never here), so trading resolution
    for responsiveness is the right call on an 8192x8192 real matrix,
    which otherwise takes ~79s to render (SymLogNorm + imshow +
    colorbar over 67M cells). Sums (not averages) matching this app's
    existing rebin convention (spectrum_operations.py) -- counts add,
    they don't average, when you coarsen a histogram's binning."""
    factor = max(1, max(matrix.shape) // max_dim)
    if factor == 1:
        return matrix
    rows = (matrix.shape[0] // factor) * factor
    cols = (matrix.shape[1] // factor) * factor
    trimmed = matrix[:rows, :cols]
    return trimmed.reshape(rows // factor, factor, cols // factor, factor).sum(axis=(1, 3))


class MatrixHeatmapWindow(QMainWindow):
    """A separate, non-modal, view-only visualization of a loaded
    matrix -- opened on demand from the matrix panel. No markers, no
    interaction beyond the standard matplotlib zoom/pan/reset toolbar.
    TV itself never had any 2D matrix rendering at all (confirmed
    absent from its entire source tree during design); this is a
    genuine enhancement, not a port."""

    def __init__(self, matrix, path, theme, panel):
        super().__init__()
        self.panel = panel
        self.setWindowTitle(f"Heatmap -- {os.path.basename(path)}")
        self.resize(700, 700)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        # The stock (untrimmed) toolbar -- unlike this app's 1D
        # spectrum view, a 2D image needs rectangle-select Zoom, which
        # _TrimmedNavigationToolbar deliberately removes for the 1D
        # case (main_window.py's own _zoom_x buttons cover that
        # instead). Home/Pan/Zoom/Save all apply naturally to an image.
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        # Match the app's dark/light theme, same as every other plotting
        # surface (main_window.py, matrix_panel.py's _plot_projection).
        # This window never redraws after construction, so a single call
        # here (rather than re-styling on every draw, as _plot_projection
        # must since it's invoked repeatedly) is sufficient.
        style_axes(self.axes, theme)

        # Downsample before norming/rendering -- the heatmap is a visual
        # overview, not a precision tool, and norming+rendering the full
        # array at real (e.g. 8192x8192) matrix sizes is what made this
        # window take ~79s to open (see _downsample_for_display above).
        display_matrix = _downsample_for_display(matrix)

        # SymLogNorm handles the negative values this app's matrix
        # data can genuinely contain (random-coincidence subtraction)
        # without error -- plain LogNorm requires strictly positive
        # data and would raise on any matrix with a negative cell.
        norm = SymLogNorm(linthresh=1.0, vmin=display_matrix.min(), vmax=max(display_matrix.max(), 1))
        self.image = self.axes.imshow(display_matrix, norm=norm, origin="lower", aspect="auto")
        self.colorbar = self.figure.colorbar(self.image, ax=self.axes)
        # style_axes(self.axes, ...) above also sets the *figure's*
        # facecolor (shared by every Axes in it), but a colorbar draws its
        # ticks/labels on its own separate Axes (matplotlib creates it
        # internally, distinct from self.axes) that style_axes(self.axes,
        # ...) can't reach. Without this second call, the colorbar's tick
        # numbers would keep matplotlib's hardcoded black and become
        # unreadable against the now-dark figure background in dark
        # theme -- not just mismatched, but actually illegible.
        style_axes(self.colorbar.ax, theme)
        self.axes.set_xlabel("X channel")
        self.axes.set_ylabel("Y channel")
        self.canvas.draw()

    def closeEvent(self, event):
        # Parentless top-level window (see matrix_panel.py's own closeEvent
        # comment) -- panel._heatmap_windows holding a reference is what
        # keeps this window alive. Without self-removal here, closing a
        # heatmap window directly (its own titlebar, not via the panel)
        # would leave a dead reference retained for the panel's entire
        # remaining lifetime, growing unboundedly across repeat open/close.
        super().closeEvent(event)
        if self in self.panel._heatmap_windows:
            self.panel._heatmap_windows.remove(self)
