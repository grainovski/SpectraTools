"""The button that launches the efficiency calibration, and what disables it."""

import numpy as np
import pytest

from calibration import Calibration
from calibration_plot_dialog import CalibrationPlotDialog


class _Line:
    """Enough of sou_io.SourceLine for build_rows."""

    def __init__(self, energy, intensity=1000.0, intensity_err=10.0):
        self.energy = energy
        self.energy_err = 0.01
        self.intensity = intensity
        self.intensity_err = intensity_err


def _dialog(qapp, points, lines, excluded=()):
    return CalibrationPlotDialog(
        None, Calibration("linear", 0.0, 0.5), points, lines, 4095,
        "out.txt", excluded=excluded)


def _points(n=8):
    """(channel, channel_err, area, area_err, energy) per point."""
    return [(100.0 * (i + 1), 0.1, 5000.0 - 300.0 * i, 70.0,
             50.0 * (i + 1)) for i in range(n)]


def _lines(n=8):
    return [_Line(50.0 * (i + 1)) for i in range(n)]


def test_the_button_is_enabled_with_enough_good_points(qapp):
    d = _dialog(qapp, _points(), _lines())
    assert d.efficiency_button.isEnabled()
    d.close()


def test_too_few_points_disables_it_with_a_reason(qapp):
    """Radware has five free parameters, so ndf = n - 5 must exceed zero."""
    d = _dialog(qapp, _points(4), _lines(4))
    assert not d.efficiency_button.isEnabled()
    assert "6" in d.efficiency_button.toolTip()
    d.close()


def test_a_peak_with_no_matching_source_line_does_not_count(qapp):
    """build_rows skips it, so it cannot reach the fit. If dropping it takes
    the total below the minimum the button must go with it, rather than
    letting the fit fail later on a count the dialog said was fine."""
    d = _dialog(qapp, _points(8), _lines(8)[:6])
    assert d.efficiency_button.isEnabled()      # 6 left, exactly the minimum

    d2 = _dialog(qapp, _points(8), _lines(8)[:5])
    assert not d2.efficiency_button.isEnabled()
    d.close(); d2.close()


def test_a_peak_with_no_usable_uncertainty_does_not_count(qapp):
    """eps = N/I with no error on either side gives an efficiency point of
    infinite weight. build_rows already refuses those rows; the button has
    to agree with it or the two disagree about how many points there are."""
    points = _points(8)
    lines = _lines(8)
    for i in (0, 1, 2):
        ch, dch, area, _area_err, energy = points[i]
        points[i] = (ch, dch, area, 0.0, energy)             # no area error
        lines[i] = _Line(50.0 * (i + 1), intensity_err=0.0)  # nor intensity
    d = _dialog(qapp, points, lines)
    assert not d.efficiency_button.isEnabled()
    d.close()


def test_an_excluded_point_does_not_count(qapp):
    """A point the user rejected from the energy calibration must not
    silently steer the efficiency curve either."""
    d = _dialog(qapp, _points(8), _lines(8), excluded=(350.0, 400.0))
    assert d.efficiency_button.isEnabled()   # 6 left

    d2 = _dialog(qapp, _points(8), _lines(8), excluded=(300.0, 350.0, 400.0))
    assert not d2.efficiency_button.isEnabled()
    d.close(); d2.close()


def test_a_zero_area_disables_it(qapp):
    """eps = N/I must be positive to fit at all, and build_rows does not
    check the sign -- only that an uncertainty exists."""
    points = _points(8)
    points[2] = (300.0, 0.1, 0.0, 70.0, 150.0)
    d = _dialog(qapp, points, _lines())
    assert not d.efficiency_button.isEnabled()
    assert "area" in d.efficiency_button.toolTip().lower()
    d.close()


def test_the_enable_check_can_fail(qapp):
    """Control: the good case must really be enabled, or every assertion
    above passes for the wrong reason."""
    good = _dialog(qapp, _points(), _lines())
    bad = _dialog(qapp, _points(4), _lines(4))
    assert good.efficiency_button.isEnabled()
    assert not bad.efficiency_button.isEnabled()
    good.close(); bad.close()


# --- the results window --------------------------------------------------


@pytest.fixture
def opened(qapp):
    """A real window over the reference's Ba-133 data, with a short MC."""
    import os

    from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo
    from efficiency_dialog import EfficiencyDialog

    path = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff",
                        "demo1.txt")
    data = np.loadtxt(path, ndmin=2)
    N, dN, E, I, dI = (data[:, 2], data[:, 3], data[:, 4],
                       data[:, 5], data[:, 6])
    fit = fit_efficiency(E, N, dN, I, dI)
    result = EfficiencyResult(
        fit=fit, mc=run_monte_carlo(fit, N, dN, I, dI, iterations=200),
        model="kfr")
    dialog = EfficiencyDialog(None, result, np.full(len(E), 0.01))
    yield dialog
    dialog.close()


def test_it_draws_both_curves_and_the_points(opened):
    assert len(opened.axes.lines) >= 2
    assert len(opened.axes.collections) >= 1, "no error bars drawn"


def test_it_draws_a_residual_for_each_model(opened):
    assert len(opened.residual_axes.lines) >= 1


def test_examine_reports_both_models_at_an_energy(opened):
    text = opened.examine(250.0)
    assert "KFR" in text and "Radware" in text
    assert "250" in text


def test_examine_shows_the_mc_mean_and_the_best_fit_as_two_numbers(opened):
    """They are different quantities and CalEnEff shows both. If the window
    ever printed one of them twice, the disagreement the second exists to
    expose would be invisible."""
    mean, sigma, best = opened.result.predict(250.0, "kfr")
    text = opened.examine(250.0)
    assert "%.6g" % mean in text
    assert "%.6g" % best in text
    assert mean != pytest.approx(best, rel=1e-12)


def test_selecting_the_other_model_redraws_and_rescales(opened):
    before = opened.result.normalisation
    opened.select_model("rw")
    assert opened.result.model == "rw"
    assert opened.result.normalisation != pytest.approx(before)


def test_the_summary_reports_the_mc_counts(opened):
    """A band from few survivors means something different from one built on
    thousands, so the number is on screen rather than implied."""
    text = opened.summary_text()
    assert "accepted" in text.lower()
    assert str(opened.result.mc.kfr_accepted) in text


def test_it_survives_radware_not_converging(qapp):
    """rw_params is None whenever the Radware fit fails. The window must
    still open on the KFR curve rather than raising -- a failed second model
    is a normal outcome, not a broken calibration."""
    import os

    from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo
    from efficiency_dialog import EfficiencyDialog

    path = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff",
                        "demo1.txt")
    data = np.loadtxt(path, ndmin=2)
    N, dN, E, I, dI = (data[:, 2], data[:, 3], data[:, 4],
                       data[:, 5], data[:, 6])
    fit = fit_efficiency(E, N, dN, I, dI)
    fit.rw_params = None                      # as a failed Radware fit leaves it
    result = EfficiencyResult(
        fit=fit, mc=run_monte_carlo(fit, N, dN, I, dI, iterations=100),
        model="kfr")

    dialog = EfficiencyDialog(None, result, np.full(len(E), 0.01))
    assert "did not converge" in dialog.summary_text().lower()
    assert "did not converge" in dialog.examine(250.0).lower()
    dialog.close()


# --- running it off the UI thread ---------------------------------------


def test_the_mc_runs_off_the_ui_thread(qapp, monkeypatch):
    """10,000 iterations across two models is about a minute. Running it in
    the handler would freeze the window for that whole time, and a frozen
    window is indistinguishable from a crashed one."""
    import calibration_plot_dialog as mod

    seen = {}
    monkeypatch.setattr(mod.EfficiencyWorker, "start",
                        lambda self, *a, **k: seen.setdefault("started", True),
                        raising=False)
    d = _dialog(qapp, _points(), _lines())
    d._on_efficiency_clicked()
    assert seen.get("started"), "the Monte Carlo was not handed to a worker"
    d.close()


def test_the_worker_really_is_a_thread(qapp):
    """Control for the test above. A plain object with a start() method
    would satisfy it while still running everything on the UI thread."""
    from PySide6.QtCore import QThread

    from calibration_plot_dialog import EfficiencyWorker

    assert issubclass(EfficiencyWorker, QThread)


def test_cancelling_tells_the_worker_to_stop(qapp):
    """The progress dialog's Cancel has to reach the loop. run_monte_carlo
    stops when its progress callback returns False, so the worker's callback
    must start returning False once cancelled."""
    from calibration_plot_dialog import EfficiencyWorker

    worker = EfficiencyWorker(None, _rows_for(_points(), _lines()))
    assert worker._progress(0, 100) is True
    worker.cancel()
    assert worker._progress(1, 100) is False
    worker.deleteLater()


def test_a_blocked_calibration_never_starts_a_worker(qapp, monkeypatch):
    """The button is disabled in that case, but the handler is still
    reachable by keyboard and by code. It must refuse rather than run a fit
    the dialog already said was impossible."""
    import calibration_plot_dialog as mod

    seen = {}
    monkeypatch.setattr(mod.EfficiencyWorker, "start",
                        lambda self, *a, **k: seen.setdefault("started", True),
                        raising=False)
    monkeypatch.setattr(mod.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    d = _dialog(qapp, _points(4), _lines(4))       # too few points
    d._on_efficiency_clicked()
    assert not seen.get("started")
    d.close()


def test_the_worker_computes_a_usable_result(qapp):
    """Run the worker's own computation synchronously, on the same rows the
    dialog would hand it. Threading is tested above; this checks the thing
    the thread exists to do actually works."""
    from calibration_plot_dialog import EfficiencyWorker

    worker = EfficiencyWorker(None, _rows_for(_points(), _lines()))
    fit, mc = worker.compute(iterations=50)
    assert fit.kfr_params is not None
    assert mc.kfr_accepted > 0
    worker.deleteLater()


def _rows_for(points, lines):
    """The same rows _efficiency_rows would produce for these inputs."""
    from caleneff_export import build_rows

    rows, _skipped = build_rows(points, lines)
    return rows


def test_the_worker_does_not_shadow_qthreads_finished_signal():
    """QThread already defines `finished`. A custom signal of that name
    shadows it, and the connection then goes somewhere else with no error at
    all -- nothing raises, nothing warns, the handler simply never runs.

    This is pinned directly rather than through behaviour, because the other
    threading tests monkeypatch start() and so never exercise the real
    threaded path: mutating `succeeded` to `finished` leaves all of them
    passing. A structural assertion is the only cheap way to catch it.
    """
    from PySide6.QtCore import QThread

    from calibration_plot_dialog import EfficiencyWorker

    assert EfficiencyWorker.finished is QThread.finished, (
        "EfficiencyWorker defines its own 'finished', shadowing QThread's")
    for name in ("progressed", "succeeded", "failed"):
        assert hasattr(EfficiencyWorker, name), name
