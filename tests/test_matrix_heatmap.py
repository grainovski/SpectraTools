import os

import numpy as np

from matrix_heatmap import MatrixHeatmapWindow, _downsample_for_display

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_matrix_heatmap_window_displays_matrix(qapp):
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx")

    assert window.windowTitle() == "Heatmap -- test.mtx"
    assert window.image is not None


def test_matrix_heatmap_window_has_navigation_toolbar_for_zoom(qapp):
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx")

    # NavigationToolbar2QT's stock "Zoom" tool is what provides
    # rectangle-select zoom in/out -- confirm it's present (this app's
    # own _TrimmedNavigationToolbar deliberately drops it for 1D
    # spectra, but a 2D image genuinely needs it, so the heatmap uses
    # the stock toolbar, not the trimmed one).
    action_texts = [a.text() for a in window.nav_toolbar.actions()]
    assert "Zoom" in action_texts


def test_matrix_heatmap_handles_negative_values_without_crashing(qapp):
    matrix = np.array([[-5, 10], [20, -1]], dtype=np.int64)
    window = MatrixHeatmapWindow(matrix, "test.mtx")
    assert window.image is not None


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
