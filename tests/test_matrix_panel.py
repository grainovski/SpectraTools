import os

from main_window import MainWindow
from matrix_panel import MatrixPanel

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_matrix_panel_loads_matrix_and_computes_both_projections(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.matrix.shape == (8192, 8192)
    assert len(panel.projections["x"]) == 8192
    assert len(panel.projections["y"]) == 8192


def test_matrix_panel_defaults_to_x_projection(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.working_axis == "x"


def test_matrix_panel_switching_axis_updates_working_axis(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    panel.axis_selector.setCurrentIndex(1)

    assert panel.working_axis == "y"
