import numpy as np
from matplotlib.colors import to_rgba

from matrix_heatmap import MatrixHeatmapWindow, _downsample_for_display
from theme import DARK_BG, DARK_TEXT


def test_matrix_heatmap_window_displays_matrix(qapp):
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "light")

    assert window.windowTitle() == "Heatmap -- test.mtx"
    assert window.image is not None


def test_matrix_heatmap_window_has_navigation_toolbar_for_zoom(qapp):
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "light")

    # NavigationToolbar2QT's stock "Zoom" tool is what provides
    # rectangle-select zoom in/out -- confirm it's present (this app's
    # own _TrimmedNavigationToolbar deliberately drops it for 1D
    # spectra, but a 2D image genuinely needs it, so the heatmap uses
    # the stock toolbar, not the trimmed one).
    action_texts = [a.text() for a in window.nav_toolbar.actions()]
    assert "Zoom" in action_texts


def test_matrix_heatmap_handles_negative_values_without_crashing(qapp):
    matrix = np.array([[-5, 10], [20, -1]], dtype=np.int64)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "light")
    assert window.image is not None


def test_matrix_heatmap_window_applies_theme_to_plot_and_colorbar(qapp):
    # Regression guard: passing theme="dark" must actually change the
    # rendered colors, not just be accepted as an argument -- a test that
    # only checked "constructor doesn't crash" would pass even if the
    # style_axes() calls were silently removed from __init__, exactly the
    # class of "all tests green, feature dead in real use" bug this app's
    # own history has already hit twice (matrix_panel.py's focus-policy
    # and eventFilter bugs, caught in Task 4's review).
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "dark")

    # style_axes(self.axes, theme) sets the *figure's* facecolor, shared
    # by every Axes drawn in it (including the colorbar's own).
    assert window.figure.get_facecolor() == to_rgba(DARK_BG)

    # The colorbar draws its tick labels on its own separate Axes
    # (self.colorbar.ax), distinct from self.axes -- style_axes(self.axes,
    # ...) alone can't reach them. Confirms the second, explicit
    # style_axes(self.colorbar.ax, theme) call is what keeps the colorbar's
    # tick numbers legible (not left at matplotlib's hardcoded black,
    # which would be illegible against the now-dark figure background).
    tick_colors = {t.get_color() for t in window.colorbar.ax.get_yticklabels()}
    assert tick_colors == {DARK_TEXT}


def test_downsample_for_display_block_sums_correctly():
    matrix = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.int64,
    )

    result = _downsample_for_display(matrix, max_dim=2)

    # Hand-computed 2x2 block sums -- top-left block is rows 0-1/cols 0-1
    # (1+2+5+6=14), top-right is rows 0-1/cols 2-3 (3+4+7+8=22), bottom-left
    # is rows 2-3/cols 0-1 (9+10+13+14=46), bottom-right is rows 2-3/cols
    # 2-3 (11+12+15+16=54). Sums, not averages, matching this app's
    # existing rebin convention (spectrum_operations.py): counts add, they
    # don't average, when a histogram's binning is coarsened.
    expected = np.array([[14, 22], [46, 54]])
    np.testing.assert_array_equal(result, expected)


def test_downsample_for_display_passes_through_small_matrix_unchanged():
    # A matrix already at or under max_dim in both dimensions should take
    # the factor == 1 early-return path and come back unchanged -- the
    # default max_dim (1024) comfortably covers this 10x10 matrix.
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)

    result = _downsample_for_display(matrix)

    np.testing.assert_array_equal(result, matrix)
