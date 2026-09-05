"""X1: fits survive closing the app."""

import json
import math

import numpy as np
import pytest

import fit_persist
from fit_persist import FitFileError
from peak_fit import fit_peaks


def _spectrum(channels=300, centre=150.0, sigma=4.0, amplitude=900.0):
    x = np.arange(channels, dtype=float)
    clean = 30.0 + amplitude * np.exp(-((x - centre) ** 2) / (2.0 * sigma ** 2))
    return x, np.random.default_rng(3).poisson(clean).astype(float)


def _fit(**kwargs):
    x, y = _spectrum()
    return fit_peaks(x, y, (60.0, 100.0), (200.0, 240.0), (120.0, 180.0), [150.0],
                     **kwargs)


# --- round trip --------------------------------------------------------


def test_a_saved_fit_comes_back_with_the_same_numbers(tmp_path):
    original = _fit()
    path = tmp_path / "fits.json"
    fit_persist.save(path, [original], "spec.txt")

    restored, spectrum_path = fit_persist.load(path)
    assert spectrum_path == "spec.txt"
    assert len(restored) == 1

    got, want = restored[0], original
    assert got.fit_region == want.fit_region
    assert got.left_bg_region == want.left_bg_region
    assert got.background_slope == pytest.approx(want.background_slope)
    assert got.reduced_chi2 == pytest.approx(want.reduced_chi2)
    assert got.link_widths == want.link_widths
    assert len(got.peaks) == len(want.peaks)
    for a, b in zip(got.peaks, want.peaks):
        assert a.position == pytest.approx(b.position)
        assert a.area == pytest.approx(b.area)
        assert a.area_err == pytest.approx(b.area_err)
        assert a.full_area_err == pytest.approx(b.full_area_err)


def test_a_restored_fit_can_still_draw_its_background_band(tmp_path):
    """The background error model travels with the fit, so a restored one
    is not silently missing its uncertainty band."""
    original = _fit()
    path = tmp_path / "band.json"
    fit_persist.save(path, [original], "spec.txt")
    restored, _ = fit_persist.load(path)

    at = np.array([120.0, 150.0, 180.0])
    assert np.allclose(
        np.asarray(restored[0].background_level_error(at), dtype=float),
        np.asarray(original.background_level_error(at), dtype=float),
    )


def test_a_jointly_fitted_background_round_trips(tmp_path):
    original = _fit(fit_background=True)
    assert original.fit_background is True

    path = tmp_path / "joint.json"
    fit_persist.save(path, [original], "spec.txt")
    restored, _ = fit_persist.load(path)[0][0], None

    assert restored.fit_background is True
    assert restored.background_covariance
    assert float(restored.background_level_error(150.0)) == pytest.approx(
        float(original.background_level_error(150.0))
    )


def test_a_tailed_fit_round_trips_its_tail_parameters(tmp_path):
    original = _fit(enable_left_tail=True)
    path = tmp_path / "tail.json"
    fit_persist.save(path, [original], "s.txt")
    restored, _ = fit_persist.load(path)

    assert restored[0].tail_fraction == pytest.approx(original.tail_fraction)
    assert restored[0].tail_beta == pytest.approx(original.tail_beta)


def test_an_untailed_fit_keeps_none_rather_than_nan(tmp_path):
    """None means "this fit had no tail at all", which is a different
    statement from "the tail was undetermined" -- collapsing them would
    make an untailed fit look like a failed one."""
    original = _fit(enable_left_tail=False)
    assert original.tail_fraction is None

    path = tmp_path / "notail.json"
    fit_persist.save(path, [original], "s.txt")
    restored, _ = fit_persist.load(path)
    assert restored[0].tail_fraction is None
    assert restored[0].tail_beta is None


def test_an_undetermined_uncertainty_is_null_in_the_file_and_nan_on_the_way_back(tmp_path):
    """NaN is not JSON. Writing it raw produces a file that other tools
    reject, so it is stored as null -- the same convention the auto-log
    already uses."""
    original = _fit()
    original.peaks[0].area_err = float("nan")

    path = tmp_path / "nan.json"
    fit_persist.save(path, [original], "s.txt")

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    json.loads(raw)  # must parse as strict JSON

    restored, _ = fit_persist.load(path)
    assert math.isnan(restored[0].peaks[0].area_err)


def test_several_fits_keep_their_order(tmp_path):
    fits = [_fit(), _fit(enable_left_tail=True), _fit(fit_background=True)]
    path = tmp_path / "many.json"
    fit_persist.save(path, fits, "s.txt")

    restored, _ = fit_persist.load(path)
    assert len(restored) == 3
    assert restored[1].tail_fraction is not None
    assert restored[2].fit_background is True


# --- the schema version ------------------------------------------------


def test_the_file_records_its_schema_version(tmp_path):
    path = tmp_path / "v.json"
    fit_persist.save(path, [_fit()], "s.txt")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["format"] == "spectratools-fits"
    assert document["schema_version"] == fit_persist.SCHEMA_VERSION


def test_an_unknown_schema_version_is_refused_by_name(tmp_path):
    """A file from a FUTURE build must say so rather than being read with
    today's assumptions and producing quietly wrong numbers."""
    path = tmp_path / "future.json"
    fit_persist.save(path, [_fit()], "s.txt")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = 99
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(FitFileError, match="schema version 99"):
        fit_persist.load(path)


def test_a_reader_exists_for_every_version_ever_written():
    """The discipline this is modelled on: HDTV keeps a reader for each
    historical schema so old files still open. A version with no reader
    means somebody's saved work stopped opening."""
    assert set(fit_persist._READERS) == set(range(1, fit_persist.SCHEMA_VERSION + 1))


def test_a_file_that_is_not_ours_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"format": "something-else", "fits": []}), encoding="utf-8")
    with pytest.raises(FitFileError, match="Not a SpectraTools fit file"):
        fit_persist.load(path)


def test_malformed_json_is_refused_with_the_path(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(FitFileError, match="Not a valid fit file"):
        fit_persist.load(path)


def test_a_binary_file_is_refused_not_crashed_on(tmp_path):
    """Picking a binary file (an installer, a .mtx, a .root) in the Load
    Fits dialog fails DECODING before json ever parses -- and the dialog
    catches only FitFileError, so an uncaught UnicodeDecodeError here
    crashed the whole app. Bytes chosen to be invalid UTF-8 from the
    first read."""
    path = tmp_path / "binary.json"
    path.write_bytes(b"\xff\xfe\x00\x01" + bytes(range(256)))
    with pytest.raises(FitFileError, match="not UTF-8 text"):
        fit_persist.load(path)


# --- refit -------------------------------------------------------------


def test_refit_reruns_from_the_marks_and_lands_in_the_same_place(tmp_path):
    x, y = _spectrum()
    original = fit_peaks(x, y, (60.0, 100.0), (200.0, 240.0), (120.0, 180.0), [150.0])

    path = tmp_path / "r.json"
    fit_persist.save(path, [original], "s.txt")
    restored, _ = fit_persist.load(path)

    again = fit_persist.refit(restored[0], x, y)
    assert again.peaks[0].position == pytest.approx(original.peaks[0].position, abs=1e-6)
    assert again.peaks[0].area == pytest.approx(original.peaks[0].area, rel=1e-6)


def test_refit_preserves_the_configuration_the_fit_was_made_with(tmp_path):
    """A refit that quietly dropped the tail, the width linking or the
    joint background would not be the same fit -- it would be a different
    one wearing the same marks."""
    x, y = _spectrum()
    original = fit_peaks(x, y, (60.0, 100.0), (200.0, 240.0), (120.0, 180.0), [150.0],
                         enable_left_tail=True, link_widths=False, fit_background=True)
    again = fit_persist.refit(original, x, y)

    assert again.tail_fraction is not None
    assert again.link_widths is False
    assert again.fit_background is True


# --- through the window ------------------------------------------------


def _window_with_fit(tmp_path):
    from main_window import MainWindow

    path = tmp_path / "run.txt"
    x, y = _spectrum()
    path.write_text("\n".join(str(int(v)) for v in y), encoding="utf-8")

    main_window = MainWindow()
    main_window._load_files([str(path)])
    active = main_window.spectra[0]
    active.fits.append(
        fit_peaks(np.arange(len(active.data), dtype=float), active.data,
                  (60.0, 100.0), (200.0, 240.0), (120.0, 180.0), [150.0])
    )
    return main_window, active


def test_save_and_load_fits_actions_track_what_is_possible(qapp, tmp_path):
    from main_window import MainWindow

    main_window = MainWindow()
    main_window._update_operations_availability()
    assert main_window.save_fits_action.isEnabled() is False
    assert main_window.load_fits_action.isEnabled() is False

    main_window, active = _window_with_fit(tmp_path)
    main_window._update_operations_availability()
    assert main_window.save_fits_action.isEnabled() is True
    assert main_window.load_fits_action.isEnabled() is True

    # Loading only needs somewhere to put the fits, not existing ones.
    active.fits.clear()
    main_window._update_operations_availability()
    assert main_window.save_fits_action.isEnabled() is False
    assert main_window.load_fits_action.isEnabled() is True


def test_restoring_adds_the_saved_fits_to_the_spectrum(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    main_window, active = _window_with_fit(tmp_path)
    target = tmp_path / "saved.json"
    fit_persist.save(target, active.fits, active.path)

    active.fits.clear()
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(target), ""))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)

    main_window._load_fits_dialog()
    assert len(active.fits) == 1
    assert active.fits[0].peaks[0].position == pytest.approx(150.0, abs=1.0)


def test_choosing_refit_reruns_rather_than_restoring(qapp, tmp_path, monkeypatch):
    """The two paths must be genuinely different: restoring reproduces the
    stored numbers, refitting recomputes them with today's code. A refit
    stamps a fresh timestamp, which is what proves it ran."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    main_window, active = _window_with_fit(tmp_path)
    active.fits[0].timestamp = "1999-01-01T00:00:00"
    target = tmp_path / "old.json"
    fit_persist.save(target, active.fits, active.path)
    active.fits.clear()

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(target), ""))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.No)

    main_window._load_fits_dialog()
    assert len(active.fits) == 1
    assert active.fits[0].timestamp != "1999-01-01T00:00:00"


def test_cancelling_the_load_leaves_the_spectrum_alone(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    main_window, active = _window_with_fit(tmp_path)
    target = tmp_path / "c.json"
    fit_persist.save(target, active.fits, active.path)

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(target), ""))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Cancel)

    main_window._load_fits_dialog()
    assert len(active.fits) == 1, "cancelling must not append anything"
def test_a_stepped_fit_round_trips_its_step(tmp_path):
    original = _fit(enable_step=True, fit_background=True)
    assert original.step_fraction is not None
    path = tmp_path / "step.json"
    fit_persist.save(path, [original], "s.txt")
    restored, _ = fit_persist.load(path)

    assert restored[0].step_fraction == pytest.approx(original.step_fraction)
    assert restored[0].step_fraction_err == pytest.approx(original.step_fraction_err)


def test_a_fit_with_no_step_keeps_none_rather_than_nan(tmp_path):
    """Same distinction the tail draws: None says the fit had no step at
    all, which is not the same as a step the fit could not determine."""
    original = _fit()
    assert original.step_fraction is None
    path = tmp_path / "nostep.json"
    fit_persist.save(path, [original], "s.txt")
    restored, _ = fit_persist.load(path)
    assert restored[0].step_fraction is None
    assert restored[0].step_fraction_err is None


def test_a_file_written_before_the_step_existed_still_loads(tmp_path):
    """Every fits file in anyone's hands predates the step. The key is
    simply absent from those records and must read back as None, not
    raise and not become a zero step."""
    original = _fit()
    path = tmp_path / "old.json"
    fit_persist.save(path, [original], "s.txt")
    document = json.loads(path.read_text(encoding="utf-8"))
    for record in document["fits"]:
        record.pop("step_fraction", None)
        record.pop("step_fraction_err", None)
    path.write_text(json.dumps(document), encoding="utf-8")

    restored, _ = fit_persist.load(path)
    assert restored[0].step_fraction is None
    assert restored[0].peaks[0].area == pytest.approx(original.peaks[0].area)
