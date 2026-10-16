"""Channel axes: ROOT histograms shown against their own coordinates, the
way ROOT's browser and HDTV show them.

The case that motivated it is a CoMPASS file: 16383-bin 1D energy spectra
and a 1023 x 1023 energy-energy matrix over the SAME 0-16383 ADC range, so
each matrix bin holds ~16 ADC channels. The fixture below is a synthetic
file of exactly that shape, with two coincident lines at ADC 4700 and 5340.
Everything is checked against those ADC values: a cut of the 1k matrix has
to put its peak where the 16k spectrum has it, not at 1/16 of that.
"""

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtWidgets import QMessageBox

import channel_axis
from calibration import Calibration
from channel_axis import ChannelAxis, compose, decompose, result_in_channels
from main_window import MainWindow
from peak_fit import FitResult, IntegrationResult, PeakResult

uproot = pytest.importorskip("uproot")

LINE_A, LINE_B = 4700.0, 5340.0
SIGMA_ADC = 6.0
WIDTH = 16383.0 / 1023


@pytest.fixture(scope="module")
def compass_file(tmp_path_factory):
    rng = np.random.default_rng(20261016)
    path = tmp_path_factory.mktemp("compass") / "HcompassF_synthetic.root"

    # 16k 1D spectrum: two Gaussian lines on a flat floor.
    adc = np.concatenate([
        rng.normal(LINE_A, SIGMA_ADC, 40000),
        rng.normal(LINE_B, SIGMA_ADC, 36000),
        rng.uniform(0, 16383, 30000),
    ])
    counts16k, edges16k = np.histogram(adc, bins=16383, range=(0.0, 16383.0))

    # E-E matrix: each cascade puts one line on X and the other on Y,
    # either way round, plus uncorrelated background.
    n = 30000
    xa = rng.normal(LINE_A, SIGMA_ADC, n)
    yb = rng.normal(LINE_B, SIGMA_ADC, n)
    swap = rng.random(n) < 0.5
    x = np.where(swap, yb, xa)
    y = np.where(swap, xa, yb)
    bx = rng.uniform(0, 16383, 20000)
    by = rng.uniform(0, 16383, 20000)
    x = np.concatenate([x, bx])
    y = np.concatenate([y, by])
    matrix_xy, ex, ey = np.histogram2d(
        x, y, bins=1023, range=((0.0, 16383.0), (0.0, 16383.0)))

    with uproot.recreate(path) as f:
        f["ENERGY/_F_EnergyCH0"] = (counts16k.astype(float), edges16k)
        f["ENERGY_ENERGY/_F_EvsE"] = (matrix_xy, ex, ey)
    return str(path)


@pytest.fixture
def warnings(monkeypatch):
    collected = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args, **kwargs: collected.append(args[2]))
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: collected.append(args[2]))
    return collected


def _open_matrix(main_window, compass_file):
    main_window._open_root_matrix(compass_file, "ENERGY_ENERGY/_F_EvsE;1")
    return main_window._matrix_panels[-1]


def _cut_on_x(panel, lo_adc, hi_adc):
    """Gate the X projection on [lo_adc, hi_adc], as hold-C clicks would."""
    state = panel.cut_controller.state
    state.add_cut_click(panel.display_to_channel(lo_adc))
    state.add_cut_click(panel.display_to_channel(hi_adc))
    panel._activate_cut()


def _fit_single_peak(main_window, centre, half_width, bg_gap):
    """Mark bg / region / peak in DISPLAY coordinates and fit, exactly as
    clicks at those x positions would."""
    to_bin = main_window.display_to_channel
    state = main_window.fit_controller.state
    for x in (centre - bg_gap - 40, centre - bg_gap, centre + bg_gap, centre + bg_gap + 40):
        state.add_bg_click(to_bin(x))
    state.add_fit_click(to_bin(centre - half_width))
    state.add_fit_click(to_bin(centre + half_width))
    state.toggle_peak(to_bin(centre), proximity=0.0)
    main_window.fit_controller.run_fit()


# --- the axis itself ----------------------------------------------------


def test_compose_and_decompose_are_inverse_and_exact():
    axis = ChannelAxis(7.5, WIDTH)
    raw = Calibration(kind="quadratic", a=1.2, b=0.33, c=2e-7)
    in_bins = compose(raw, axis)
    for i in (0.0, 100.0, 1022.0):
        assert in_bins.apply(i) == pytest.approx(raw.apply(axis.to_channel(i)), rel=1e-12)
    back = decompose(in_bins, axis)
    assert (back.a, back.b, back.c) == pytest.approx((raw.a, raw.b, raw.c), rel=1e-9)


def test_identity_is_none_and_costs_nothing():
    assert channel_axis.from_bin_edges(np.arange(0, 4097)) is None
    cal = Calibration(kind="linear", a=1.0, b=0.5)
    assert compose(cal, None) is cal
    result = object()
    assert result_in_channels(result, None) is result


def test_result_in_channels_moves_positions_and_scales_widths_only():
    axis = ChannelAxis(7.5, 16.0)
    peak = PeakResult(position=293.0, position_err=0.1, fwhm=1.0, fwhm_err=0.05,
                      area=1000.0, area_err=30.0, amplitude=250.0, sigma=0.42)
    fit = FitResult(left_bg_region=(280, 285), right_bg_region=(300, 305),
                    fit_region=(288, 298), background_slope=2.0,
                    background_intercept=10.0, peaks=[peak],
                    background_anchors=((282.5, 4.0), (302.5, 5.0)))
    out = result_in_channels(fit, axis)
    p = out.peaks[0]
    assert p.position == pytest.approx(7.5 + 16 * 293)
    assert (p.position_err, p.fwhm, p.sigma) == pytest.approx((1.6, 16.0, 0.42 * 16))
    # Counts are counts: the same bins hold them.
    assert (p.area, p.amplitude) == (1000.0, 250.0)
    assert out.fit_region == pytest.approx((7.5 + 16 * 288, 7.5 + 16 * 298))
    # The background line still gives the same level at the same place.
    for i in (280.0, 300.0):
        ch = axis.to_channel(i)
        assert out.background_slope * ch + out.background_intercept == pytest.approx(
            fit.background_slope * i + fit.background_intercept)
    assert fit.peaks[0].position == 293.0, "the stored result must be untouched"

    integ = IntegrationResult(None, None, (10, 20), 0.0, 5.0, 1.0, 15.0, 0.2, 3.0,
                              0.1, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 5.0, 1.0,
                              15.0, 0.2, 3.0, 0.1, 0.0, 0.0)
    out = result_in_channels(integ, axis)
    assert out.gross_centroid == pytest.approx(7.5 + 16 * 15)
    assert out.gross_fwhm == pytest.approx(48.0)
    assert out.gross_area == 5.0


# --- the matrix panel ---------------------------------------------------


def test_the_root_matrix_projection_spans_the_full_adc_range(qapp, compass_file):
    main_window = MainWindow()
    panel = _open_matrix(main_window, compass_file)

    assert panel.matrix.shape == (1023, 1023)
    xdata = panel.axes.get_lines()[0].get_xdata()
    assert len(xdata) == 1023
    assert xdata[0] == pytest.approx(WIDTH / 2 - 0.5)
    assert xdata[-1] == pytest.approx(16383 - WIDTH / 2 - 0.5)
    lo, hi = panel.axes.get_xlim()
    assert hi > 16000, "the axis must read 0-16383 as in ROOT/HDTV, not 0-1022"


def test_the_projection_peak_sits_at_its_adc_value(qapp, compass_file):
    main_window = MainWindow()
    panel = _open_matrix(main_window, compass_file)
    proj = panel.spectra[0].data
    near_a = slice(int((LINE_A - 200) / WIDTH), int((LINE_A + 200) / WIDTH))
    peak_bin = near_a.start + int(np.argmax(proj[near_a]))
    assert panel.channel_to_display(peak_bin) == pytest.approx(LINE_A, abs=WIDTH)


def test_the_panel_readout_shows_the_adc_channel(qapp, compass_file):
    main_window = MainWindow()
    panel = _open_matrix(main_window, compass_file)
    panel.canvas.draw()
    x_px, y_px = panel.axes.transData.transform((LINE_A, 1.0))
    event = MouseEvent("motion_notify_event", panel.canvas, x_px, y_px)
    panel._on_mouse_move(event)
    message = panel.statusBar().currentMessage()
    channel = int(message.split("Channel: ")[1].split()[0])
    assert channel == pytest.approx(LINE_A, abs=WIDTH)


def test_the_heatmap_spans_the_adc_range(qapp, compass_file):
    main_window = MainWindow()
    panel = _open_matrix(main_window, compass_file)
    panel._open_heatmap()
    window = panel._heatmap_windows[-1]
    x0, x1, y0, y1 = window.image.get_extent()
    assert (x0, x1) == pytest.approx((-0.5, 16382.5))
    assert (y0, y1) == pytest.approx((-0.5, 16382.5))
    window.close()


def test_a_cut_carries_the_other_axis_and_lands_on_the_16k_peak(qapp, compass_file):
    """Gate on line B on X; the cut is a spectrum of Y and must show line
    A at ADC ~4700 -- the same place the 16k spectrum has it, so the two
    overlay peak on peak."""
    main_window = MainWindow()
    main_window._load_root_spectra(compass_file, ["ENERGY/_F_EnergyCH0;1"])
    panel = _open_matrix(main_window, compass_file)
    _cut_on_x(panel, LINE_B - 25, LINE_B + 25)

    cut = main_window.spectra[-1]
    assert cut.active and cut.axis is not None
    assert cut.axis.width == pytest.approx(WIDTH)
    # The cut is named for its gate in ADC channels, as it was marked.
    span = cut.path.rsplit("_x_cut_", 1)[1].split(".")[0].split("_")
    assert int(span[0]) == pytest.approx(LINE_B - 25, abs=WIDTH)
    assert int(span[1]) == pytest.approx(LINE_B + 25, abs=WIDTH)

    near = slice(int((LINE_A - 200) / WIDTH), int((LINE_A + 200) / WIDTH))
    cut_peak_bin = near.start + int(np.argmax(cut.data[near]))
    x_cut = main_window.spectrum_to_display(cut, cut_peak_bin)
    big = main_window.spectra[0]
    big_peak = int(np.argmax(big.data[4500:4900])) + 4500
    x_big = main_window.spectrum_to_display(big, big_peak)
    assert x_cut == pytest.approx(x_big, abs=WIDTH)

    # And the full view spans both: the 16k spectrum sets the extent.
    lo, hi = main_window._full_display_limits([big, cut])
    assert (lo, hi) == pytest.approx((0.0, 16382.0), abs=1.0)


def test_a_fit_on_a_cut_reports_adc_channels_everywhere(qapp, compass_file, tmp_path):
    main_window = MainWindow()
    panel = _open_matrix(main_window, compass_file)
    _cut_on_x(panel, LINE_B - 25, LINE_B + 25)
    cut = main_window.spectra[-1]
    cut.path = str(tmp_path / "cut_x_cut_5315_5365.root")

    _fit_single_peak(main_window, LINE_A, half_width=60, bg_gap=100)
    assert cut.fits, main_window.statusBar().currentMessage()
    peak = cut.fits[-1].peaks[0]
    # Stored in bins, as the fit engine works...
    assert peak.position == pytest.approx((LINE_A + 0.5) / WIDTH - 0.5, abs=1.0)

    # ...but every place a user reads it says ADC channels:
    shown = main_window.fit_controller.results_table.item(0, 0).text()
    assert float(shown.split("(")[0]) == pytest.approx(LINE_A, abs=3.0)
    fwhm_shown = main_window.fit_controller.results_table.item(0, 2).text()
    assert float(fwhm_shown.split("(")[0]) == pytest.approx(2.3548 * SIGMA_ADC, rel=0.6)

    choices = main_window.fitted_peak_choices()
    assert choices[0][1] == pytest.approx(LINE_A, abs=3.0)
    assert choices[0][3] == pytest.approx(2.3548 * SIGMA_ADC, rel=0.6)

    import json
    log = tmp_path / "cut_x_cut_5315_5365_fits.jsonl"
    record = json.loads(log.read_text().splitlines()[-1])
    assert record["peaks"][0]["position"] == pytest.approx(LINE_A, abs=3.0)

    # The annotation on the plot is the display coordinate, i.e. ADC.
    labels = [t.get_text() for t in main_window.axes.texts]
    assert any(abs(float(t) - LINE_A) < 3.0 for t in labels if t.replace(".", "").isdigit())


def test_one_calibration_serves_the_16k_spectrum_and_the_1k_cut(qapp, compass_file):
    """The point of doing it HDTV's way: a calibration is a function of the
    ADC channel, so made once it is right for both."""
    main_window = MainWindow()
    main_window._load_root_spectra(compass_file, ["ENERGY/_F_EnergyCH0;1"])
    panel = _open_matrix(main_window, compass_file)
    _cut_on_x(panel, LINE_B - 25, LINE_B + 25)
    cal = Calibration(kind="linear", a=0.0, b=1173.2 / LINE_A)
    main_window._apply_calibration_change(cal, True)

    cut = main_window.spectra[-1]
    line = [ln for ln in main_window.axes.get_lines()
            if len(ln.get_xdata()) == len(cut.data)][0]
    xdata = line.get_xdata()
    assert xdata[100] == pytest.approx(cal.apply(cut.axis.to_channel(100)))

    _fit_single_peak(main_window, 1173.2, half_width=15, bg_gap=25)
    shown = main_window.fit_controller.results_table.item(0, 0).text()
    assert float(shown.split("(")[0]) == pytest.approx(1173.2, abs=1.0)
    # The calibration itself is untouched by the cut's binning.
    assert main_window._calibration == cal


def test_go_to_on_a_cut_goes_to_the_adc_channel(qapp, compass_file):
    main_window = MainWindow()
    panel = _open_matrix(main_window, compass_file)
    _cut_on_x(panel, LINE_B - 25, LINE_B + 25)
    main_window.goto_display_value(LINE_A)
    lo, hi = sorted(main_window.axes.get_xlim())
    assert lo < LINE_A < hi
    assert hi - lo == pytest.approx(100 * WIDTH, rel=0.01)
    assert main_window._goto_marker_channel == pytest.approx(LINE_A)


def test_rebinning_a_cut_widens_its_axis_and_leaves_the_calibration(qapp, compass_file):
    main_window = MainWindow()
    panel = _open_matrix(main_window, compass_file)
    _cut_on_x(panel, LINE_B - 25, LINE_B + 25)
    cal = Calibration(kind="linear", a=0.0, b=0.25)
    main_window._apply_calibration_change(cal, True)
    cut = main_window.spectra[-1]
    first_before = cut.axis.to_channel(0)
    main_window._apply_rebin(cut, 2)
    assert cut.axis.width == pytest.approx(2 * WIDTH)
    # New bin 0 holds old bins 0 and 1: its channel is their midpoint.
    assert cut.axis.to_channel(0) == pytest.approx(first_before + WIDTH / 2)
    assert main_window._calibration == cal


def test_combining_differently_binned_spectra_is_refused(qapp, compass_file, warnings):
    main_window = MainWindow()
    main_window._load_root_spectra(compass_file, ["ENERGY/_F_EnergyCH0;1"])
    panel = _open_matrix(main_window, compass_file)
    _cut_on_x(panel, LINE_B - 25, LINE_B + 25)
    before = len(main_window.spectra)
    main_window._apply_add(main_window.spectra[0], main_window.spectra[1], 1.0)
    assert len(main_window.spectra) == before
    assert warnings and "binned differently" in warnings[-1]


def test_symmetrize_takes_positions_in_adc_channels(qapp, compass_file):
    from symmetrize_dialog import SymmetrizeDialog

    main_window = MainWindow()
    panel = _open_matrix(main_window, compass_file)
    dialog = SymmetrizeDialog(panel, panel.projections["x"], panel.projections["y"],
                              axes=(panel.channel_axes["x"], panel.channel_axes["y"]))
    assert dialog.x_boxes[0].maximum() > 16000
    for boxes in (dialog.x_boxes, dialog.y_boxes):
        boxes[0].setValue(LINE_A)
        boxes[1].setValue(LINE_B)
    a, b = dialog.mapping
    # Same detector scale on both axes: the bin-space map is the identity.
    assert b == pytest.approx(1.0, abs=0.01)
    assert abs(a) < 1.0
    text = dialog.summary.text()
    # Reported back in the channels typed, refined to the centroids.
    assert float(text.split("Line 1: Y ")[1].split()[0]) == pytest.approx(LINE_A, abs=3.0)
    dialog.close()


def test_marks_stay_on_the_same_channels_when_the_active_spectrum_changes(qapp, compass_file):
    main_window = MainWindow()
    main_window._load_root_spectra(compass_file, ["ENERGY/_F_EnergyCH0;1"])
    panel = _open_matrix(main_window, compass_file)
    _cut_on_x(panel, LINE_B - 25, LINE_B + 25)
    big, cut = main_window.spectra[0], main_window.spectra[-1]
    state = main_window.fit_controller.state
    state.add_fit_click(main_window.display_to_channel(LINE_A - 60))
    state.add_fit_click(main_window.display_to_channel(LINE_A + 60))
    state.toggle_peak(main_window.display_to_channel(LINE_A), proximity=0.0)

    main_window._on_active_toggled(big.path, True)

    lo, hi = state.fit_region
    assert main_window.channel_to_display(lo) == pytest.approx(LINE_A - 60, abs=0.01)
    assert main_window.channel_to_display(hi) == pytest.approx(LINE_A + 60, abs=0.01)
    assert main_window.channel_to_display(state.peak_positions[0]) == pytest.approx(LINE_A, abs=0.01)


def test_normalize_reads_every_spectrum_over_the_same_channels(qapp, compass_file):
    """A region marked on the 1k cut is read over the same ADC range on
    the 16k spectrum, so the two end up with equal counts there."""
    main_window = MainWindow()
    main_window._load_root_spectra(compass_file, ["ENERGY/_F_EnergyCH0;1"])
    panel = _open_matrix(main_window, compass_file)
    _cut_on_x(panel, LINE_B - 25, LINE_B + 25)
    big, cut = main_window.spectra[0], main_window.spectra[-1]
    state = main_window.fit_controller.state
    state.add_fit_click(main_window.display_to_channel(LINE_A - 40))
    state.add_fit_click(main_window.display_to_channel(LINE_A + 40))
    region = state.fit_region
    lo_c, hi_c = (cut.axis.to_channel(region[0]), cut.axis.to_channel(region[1]))
    main_window._normalize_spectra()

    from spectrum_operations import reference_value
    cut_sum = reference_value(cut.data, region=region)
    big_sum = reference_value(big.data, region=(lo_c, hi_c))
    assert big_sum == pytest.approx(cut_sum, rel=0.02)
