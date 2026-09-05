"""The Automatic Calibration dialog: source loading, the live peak
count, and what Run hands back.

Driven the way test_energy_assign_source.py drives its dialog -- call
the handlers directly -- rather than through the file picker, which is
the one piece that needs a human.
"""

import os

import numpy as np
import pytest
from PySide6.QtWidgets import QDialog

import auto_calibrate
import peak_search
from auto_calibrate_dialog import (
    MAX_SENSITIVITY,
    MIN_SENSITIVITY,
    AutoCalibrateDialog,
)
from sou_io import load_sou
from synthetic_calibration import fixture_sou, spectrum_from_source


@pytest.fixture(scope="module")
def counts():
    return spectrum_from_source(fixture_sou("eu152.sou"), 12.5, 0.40)


class _Settings:
    def __init__(self, folder=""):
        self.folder = folder
        self.remembered = []

    def last_folder(self):
        return self.folder

    def set_last_folder(self, folder):
        self.remembered.append(folder)


# --- the sensitivity control and its live count ------------------------


def test_starts_at_the_default_sensitivity_with_a_live_count(qapp, counts):
    dialog = AutoCalibrateDialog(None, counts)
    assert dialog.sensitivity.value() == peak_search.DEFAULT_SENSITIVITY
    expected = len(peak_search.search(counts))
    assert expected > 1
    assert dialog.found_count() == expected
    assert dialog.peaks_label.text() == f"{expected} peaks found"


def test_the_count_follows_the_sensitivity(qapp, counts):
    dialog = AutoCalibrateDialog(None, counts)
    before = dialog.found_count()
    dialog.sensitivity.setValue(20.0)
    after = len(peak_search.search(counts, 20.0))
    assert after < before
    assert dialog.found_count() == after
    assert dialog.peaks_label.text().startswith(f"{after} peak")


def test_the_sensitivity_range_brackets_the_default(qapp, counts):
    dialog = AutoCalibrateDialog(None, counts)
    assert dialog.sensitivity.minimum() == MIN_SENSITIVITY == 2.0
    assert dialog.sensitivity.maximum() == MAX_SENSITIVITY == 30.0
    assert MIN_SENSITIVITY < peak_search.DEFAULT_SENSITIVITY < MAX_SENSITIVITY


def test_a_single_peak_reads_in_the_singular(qapp):
    x = np.arange(2000, dtype=float)
    one = 50.0 + 5000.0 * np.exp(-((x - 900.0) ** 2) / (2 * 4.0 ** 2))
    dialog = AutoCalibrateDialog(None, np.random.default_rng(2).poisson(one))
    assert dialog.found_count() == 1
    assert dialog.peaks_label.text() == "1 peak found"


# --- loading a source --------------------------------------------------


def test_run_needs_a_source(qapp, counts):
    dialog = AutoCalibrateDialog(None, counts)
    assert not dialog.run_button.isEnabled()
    assert dialog.source_name() is None
    dialog._on_run()
    assert dialog.outcome is None
    assert "Load a source file first" in dialog.status.text()


def test_loading_a_source_names_it_and_enables_run(qapp, counts):
    dialog = AutoCalibrateDialog(None, counts)
    path = fixture_sou("eu152.sou")
    assert dialog.load_source(path)
    assert dialog.source_label.text() == f"eu152.sou: {len(load_sou(path))} lines"
    assert dialog.run_button.isEnabled()
    assert dialog.source_name() == "eu152.sou"
    assert dialog.source_path() == path


def test_a_bad_source_is_reported_and_the_previous_one_kept(qapp, counts, tmp_path):
    dialog = AutoCalibrateDialog(None, counts)
    assert dialog.load_source(fixture_sou("eu152.sou"))
    bad = tmp_path / "bad.sou"
    bad.write_text("60 .1 1\n", encoding="utf-8")
    assert dialog.load_source(str(bad)) is False
    assert "bad.sou" in dialog.status.text()
    assert dialog.source_name() == "eu152.sou"
    assert dialog.run_button.isEnabled()


def test_loading_remembers_the_folder_like_the_other_dialogs(qapp, counts):
    settings = _Settings()
    dialog = AutoCalibrateDialog(None, counts, settings=settings)
    dialog.load_source(fixture_sou("eu152.sou"))
    assert settings.remembered == [os.path.dirname(fixture_sou("eu152.sou"))]


def test_a_remembered_source_is_loaded_up_front(qapp, counts, tmp_path):
    dialog = AutoCalibrateDialog(None, counts, source_path=fixture_sou("ba133.sou"))
    assert dialog.source_name() == "ba133.sou"
    assert dialog.run_button.isEnabled()
    # A path that no longer exists is simply not offered.
    dialog = AutoCalibrateDialog(None, counts, source_path=str(tmp_path / "gone.sou"))
    assert dialog.source_name() is None
    assert dialog.source_label.text() == "No source loaded"


# --- running -----------------------------------------------------------


def test_run_fits_identifies_and_accepts(qapp, counts):
    dialog = AutoCalibrateDialog(None, counts)
    dialog.load_source(fixture_sou("eu152.sou"))
    dialog._on_run()
    outcome = dialog.outcome
    assert outcome is not None
    assert outcome.match.ok, outcome.match.reason
    assert len(outcome.results) > 0
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_run_with_the_wrong_source_still_hands_back_the_fits(qapp, counts):
    """A refusal is a result, not an error: the fits are real and the
    caller opens them for assignment by hand with the reason on show."""
    dialog = AutoCalibrateDialog(None, counts)
    dialog.load_source(fixture_sou("ba133.sou"))
    dialog._on_run()
    outcome = dialog.outcome
    assert outcome is not None
    assert not outcome.match.ok
    assert outcome.match.reason
    assert len(outcome.results) > 0
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_run_with_nothing_found_stays_open_and_says_so(qapp):
    flat = np.full(3000, 20.0)
    dialog = AutoCalibrateDialog(None, flat)
    dialog.load_source(fixture_sou("eu152.sou"))
    assert dialog.found_count() == 0
    dialog._on_run()
    assert dialog.outcome is None
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.status.text() == "No peaks were found at this sensitivity; lower it."


def test_run_passes_the_sensitivity_and_variance_through(qapp, counts, monkeypatch):
    seen = {}
    real = auto_calibrate.calibrate

    def spy(x, y, lines, **kwargs):
        seen.update(kwargs)
        seen["lines"] = lines
        return real(x, y, lines, **kwargs)

    monkeypatch.setattr(auto_calibrate, "calibrate", spy)
    variance = 2.0 * np.maximum(counts, 1.0)
    dialog = AutoCalibrateDialog(None, counts, variance=variance)
    dialog.load_source(fixture_sou("eu152.sou"))
    dialog.sensitivity.setValue(7.5)
    dialog._on_run()
    assert seen["sensitivity"] == 7.5
    assert seen["variance"] is variance
    assert seen["lines"] is dialog.source_lines
    assert dialog.outcome is not None
