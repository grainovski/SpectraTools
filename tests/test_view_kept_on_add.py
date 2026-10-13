"""Opening a spectrum or adding a derived one (a gate, Add/Subtract, ...)
keeps the X range the user is looking at instead of jumping to full scale.

Only the FIRST spectrum opens at full scale: there is no view to keep yet.
Rebin still resets on purpose -- it renumbers the channels, so the old
range would point at the wrong place.
"""

import os

import numpy as np
import pytest

from main_window import MainWindow

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SPE = os.path.join(FIXTURES, "eu.spe")
SPK = os.path.join(FIXTURES, "demo.spk")
ROOT = os.path.join(FIXTURES, "compass", "co60.root")
ZOOM = (300.0, 450.0)


@pytest.fixture
def window(qapp, monkeypatch):
    # A failed load reports through a modal box; stub it so a regression
    # fails the test instead of hanging a headless run.
    monkeypatch.setattr("main_window.QMessageBox.warning",
                        lambda *a, **k: pytest.fail("unexpected warning: %r" % (a[2:],)))
    w = MainWindow()
    yield w
    w.close()


def _zoomed(window):
    window.axes.set_xlim(ZOOM)
    return window


def test_the_first_spectrum_still_opens_at_full_scale(window):
    window._load_files([SPE])
    n = len(window.spectra[0].data)
    assert window.axes.get_xlim() == pytest.approx((0, n - 1))


def test_opening_another_spectrum_keeps_the_zoom(window):
    window._load_files([SPE])
    _zoomed(window)
    window._load_files([SPK])
    assert len(window.spectra) == 2
    assert window.axes.get_xlim() == pytest.approx(ZOOM)


def test_opening_root_spectra_keeps_the_zoom(window):
    pytest.importorskip("uproot")
    window._load_files([SPE])
    _zoomed(window)
    window._load_root_spectra(ROOT, ["ENERGY/_F_EnergyCH0@DT5780SCP_58363;1"])
    assert len(window.spectra) == 2
    assert window.axes.get_xlim() == pytest.approx(ZOOM)


def test_a_gate_or_other_derived_spectrum_keeps_the_zoom(window):
    """_add_combined_spectrum is where every derived spectrum arrives: a
    matrix-panel gate, Add/Subtract, efficiency correction (Multiply rescales in place and always kept the view)."""
    window._load_files([SPE])
    _zoomed(window)
    window._add_combined_spectrum("gate_x_cut_166_172.json",
                                  np.arange(4096, dtype=float))
    assert window.axes.get_xlim() == pytest.approx(ZOOM)


def test_the_y_axis_refits_to_the_kept_range(window):
    """Y follows what is on screen, as it does when panning: a new
    spectrum far taller than anything in view must raise the Y limit."""
    window._load_files([SPE])
    _zoomed(window)
    low_top = window.axes.get_ylim()[1]
    tall = np.zeros(4096)
    tall[350:400] = 50 * max(low_top, 1.0)
    window._add_combined_spectrum("tall.spe", tall)
    assert window.axes.get_xlim() == pytest.approx(ZOOM)
    assert window.axes.get_ylim()[1] >= tall.max()


def test_with_nothing_visible_a_new_spectrum_opens_at_full_scale(window):
    """'Keep the view' means keep what the user is LOOKING at. With every
    spectrum hidden there is no such view."""
    window._load_files([SPE])
    _zoomed(window)
    for s in window.spectra:
        s.visible = False
    window._plot_data(preserve_view=True)
    window._load_files([SPK])
    n = max(len(s.data) for s in window.spectra if s.visible)
    assert window.axes.get_xlim() == pytest.approx((0, n - 1))


def test_full_view_still_restores_full_scale(window):
    window._load_files([SPE])
    _zoomed(window)
    window._load_files([SPK])
    window._show_full_spectrum()        # Show Full Spectrum, Ctrl+0
    n = max(len(s.data) for s in window.spectra)
    assert window.axes.get_xlim() == pytest.approx((0, n - 1))
