import os

import numpy as np

from matrix_heatmap import MatrixHeatmapWindow

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
