"""The list-mode dialogs and files name the detectors chosen in step 1.

Step 1 lets the user pick any two channels as X and Y. Every label that
names a detector, and the JSON metadata, must follow that choice rather
than say CH1 / CH0 whatever was picked. Each check has a default-channel
control, so a label that ignores the session fails here.
"""

import json
from types import SimpleNamespace

import numpy as np

from gain_match import GainMatch
from gain_match_dialog import GainMatchDialog, relabel
from listmode_output_dialog import ListModeOutputDialog, OutputChoice, sort_and_write
from listmode_session import ListModeSession
from listmode_sort import Pairs, window_weight
from time_window_dialog import TimeWindowDialog

G = 10_000
MAX = 2_000_000
PROMPT = (-G, G)
RANDOMS = [(-MAX, -500_000), (500_000, MAX)]


def _session(tmp_path, x, y, nchan=64):
    run = tmp_path / "Run_B"
    (run / "UNFILTERED").mkdir(parents=True, exist_ok=True)
    s = ListModeSession(path=str(run / "UNFILTERED" / "SData_Run_B.root"),
                        max_dt_ps=MAX, nchan=nchan, x_channel=x, y_channel=y)
    s.listmode = SimpleNamespace(granularity_ps=G, stream="UNFILTERED")
    rng = np.random.default_rng(5)
    n = 500
    dt = rng.choice([-G, 0, G], size=n).astype(np.int64)
    dt[:100] = rng.integers(-MAX // G, MAX // G + 1, size=100) * G
    s.pairs = Pairs(dt, rng.integers(0, nchan, size=n), rng.integers(0, nchan, size=n))
    s.gain = GainMatch([0.0, 1.0], nchan=nchan)
    s.prompt_window = PROMPT
    s.random_windows = list(RANDOMS)
    s.weight, s.n_prompt_bins, s.n_random_bins = window_weight(PROMPT, RANDOMS, G, MAX)
    return s


def test_relabel_swaps_both_names_in_one_pass():
    assert relabel("CH1 onto CH0's scale", 1, 0) == "CH1 onto CH0's scale"
    assert relabel("CH1 onto CH0's scale", 3, 2) == "CH3 onto CH2's scale"
    # Swapped assignment: a two-pass replace would make both CH0.
    assert relabel("CH1 onto CH0", 0, 1) == "CH0 onto CH1"
    assert relabel("CH10 and CH01", 3, 2) == "CH10 and CH01"


def test_json_metadata_names_the_chosen_channels(tmp_path):
    for (x, y), expected in (((1, 0), {"x": "CH1", "y": "CH0"}),
                             ((3, 2), {"x": "CH3", "y": "CH2"})):
        s = _session(tmp_path, x, y)
        out = tmp_path / f"out_{x}{y}"
        out.mkdir()
        choice = OutputChoice(out, "m", write_mtx=False, write_sparse=True)
        outcome = sort_and_write(s, choice)
        assert outcome.error is None, outcome.error
        meta = json.loads((out / "m.json").read_text(encoding="utf-8"))["spectratools"]
        assert meta["axes"] == expected
        assert meta["gain"] == f"{expected['x']} mapped onto {expected['y']}"


def test_output_dialog_box_title_names_the_chosen_channels(qapp, tmp_path):
    d = ListModeOutputDialog(_session(tmp_path, 3, 2))
    from PySide6.QtWidgets import QGroupBox
    titles = [b.title() for b in d.findChildren(QGroupBox)]
    assert any("X = CH3, Y = CH2" in t for t in titles), titles
    assert not any("CH1" in t or "CH0" in t for t in titles), titles


def test_time_window_axis_names_the_chosen_channels(qapp, tmp_path):
    d = TimeWindowDialog(_session(tmp_path, 3, 2))
    assert d.axes.get_xlabel().startswith("Δt = t(CH3) − t(CH2)")
    d1 = TimeWindowDialog(_session(tmp_path, 1, 0))
    assert d1.axes.get_xlabel().startswith("Δt = t(CH1) − t(CH0)")


def test_gain_dialog_labels_name_the_chosen_channels(qapp, tmp_path):
    s = _session(tmp_path, 3, 2)
    s.gain = None
    d = GainMatchDialog(s)
    headers = [d.table.horizontalHeaderItem(i).text() for i in range(d.table.columnCount())]
    assert headers[1:3] == ["CH3 centroid", "CH2 centroid"], headers
    assert headers[4:6] == ["FWHM CH3", "FWHM CH2"], headers
    texts = [d.skip_check.text(), d.skip_label.text(), d.cutoff.suffix(),
             d.overlay_axes.get_title(), d.map_axes.get_xlabel()]
    joined = " | ".join(texts)
    assert "CH3" in d.skip_check.text() and "CH2" in d.cutoff.suffix(), joined
    assert "CH1" not in joined and "CH0" not in joined, joined
    # Default control: the original labels.
    d0 = GainMatchDialog(ListModeSession())
    h0 = [d0.table.horizontalHeaderItem(i).text() for i in range(d0.table.columnCount())]
    assert h0[1:3] == ["CH1 centroid", "CH0 centroid"]
