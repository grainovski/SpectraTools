"""Step 4 of the list-mode operation: the prompt/random time windows.

The dialog's readout must be exactly what listmode_sort will do with the
same windows, so every number is checked against window_mask /
window_weight on the same pairs, not against the dialog's own
arithmetic.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from listmode_session import NEXT, ListModeSession
from listmode_sort import Pairs, window_mask, window_weight
from time_window_dialog import TimeWindowDialog, default_windows, snapped_max_dt

G = 10_000          # 10 ns ticks, as on the DT5780 run
MAX = 2_000_000


def synthetic_pairs(seed=1, n_prompt=5000, n_random=3000, max_dt=MAX):
    """A prompt peak on the -10/0/+10 ns ticks over a flat random floor."""
    rng = np.random.default_rng(seed)
    prompt_dt = rng.choice([-G, 0, G], size=n_prompt, p=[0.05, 0.9, 0.05])
    k = max_dt // G
    random_dt = rng.integers(-k, k + 1, size=n_random) * G
    dt = np.concatenate([prompt_dt, random_dt]).astype(np.int64)
    ex = rng.integers(0, 64, size=len(dt))
    ey = rng.integers(0, 64, size=len(dt))
    return Pairs(dt, ex, ey)


def make_session(pairs=None, granularity=G, max_dt=MAX):
    s = ListModeSession(max_dt_ps=max_dt, nchan=64)
    s.listmode = SimpleNamespace(granularity_ps=granularity)
    s.pairs = synthetic_pairs() if pairs is None else pairs
    return s


def test_defaults_prompt_around_peak_and_side_bands(qapp):
    s = make_session()
    d = TimeWindowDialog(s)
    prompt, randoms = d.windows()
    assert prompt == (-G, G)
    assert randoms == [(-MAX, -500_000), (500_000, MAX)]
    assert d.readout["error"] is None
    assert d.next_button.isEnabled()
    assert d.readout["n_prompt_bins"] == 3
    assert d.readout["n_random_bins"] == 2 * 151


def test_default_prompt_follows_an_offset_peak():
    centres = np.arange(-200, 201) * G
    counts = np.zeros(len(centres))
    counts[200 + 7] = 100       # peak at +70 ns
    prompt, randoms = default_windows(centres, counts, G, MAX)
    assert prompt == (6 * G, 8 * G)
    assert randoms[1][0] == 500_000


def test_readout_matches_window_mask_and_window_weight(qapp):
    s = make_session()
    d = TimeWindowDialog(s)
    d.set_windows((-30_000, 20_000), [(-1_500_000, -600_000), (700_000, 1_999_000)])
    prompt, randoms = d.windows()
    dt = s.pairs.dt
    assert d.readout["prompt_counts"] == int(window_mask(dt, prompt, G).sum())
    for w, n in zip(randoms, d.readout["random_counts"]):
        assert n == int(window_mask(dt, w, G).sum())
    w, nbp, nbr = window_weight(prompt, randoms, G, MAX)
    assert d.readout["w"] == w
    assert (d.readout["n_prompt_bins"], d.readout["n_random_bins"]) == (nbp, nbr)
    n_r = int(window_mask(dt, randoms, G).sum())
    assert d.readout["expected_random"] == pytest.approx(w * n_r)
    assert d.readout["net"] == pytest.approx(d.readout["prompt_counts"] - w * n_r)


def test_numeric_edit_updates_readout(qapp):
    s = make_session()
    d = TimeWindowDialog(s)
    before = d.readout["prompt_counts"]
    d.prompt_row.hi.setValue(0.0)          # [-10, 0] ns: drops the +10 tick
    assert d.windows()[0] == (-G, 0)
    assert d.readout["n_prompt_bins"] == 2
    assert d.readout["prompt_counts"] == int(window_mask(s.pairs.dt, (-G, 0), G).sum())
    assert d.readout["prompt_counts"] < before
    assert "2 bins" in d.prompt_row.readout.text()


@pytest.mark.parametrize("prompt, randoms, needle", [
    ((-G, G), [(0, 100_000)], "overlaps the prompt"),
    ((-G, G), [(500_000, 900_000), (800_000, 1_000_000)], "overlap"),
    ((-G, G), [], "At least one random window"),
    ((2_000, 3_000), [(500_000, 900_000)], "contains no dt bin"),
])
def test_bad_windows_block_next(qapp, monkeypatch, prompt, randoms, needle):
    import listmode_session
    warned = []
    monkeypatch.setattr(listmode_session.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[2]))
    s = make_session()
    d = TimeWindowDialog(s)
    d.set_windows(prompt, randoms)
    assert needle in d.readout["error"]
    assert needle in d.error_label.text()
    assert not d.next_button.isEnabled()
    # Even if Next were reached, apply refuses and the session is untouched.
    assert needle in d.apply_to_session()
    assert s.weight is None and s.prompt_window is None
    d._on_next()
    assert warned and d.step_result != NEXT


def test_add_and_remove_random_window(qapp):
    s = make_session()
    d = TimeWindowDialog(s)
    added = d.add_random_window()
    assert added is not None
    prompt, randoms = d.windows()
    assert len(randoms) == 3 and d.readout["error"] is None
    # The free stretch sits between the prompt and a side band, one bin clear.
    assert added[0] > prompt[1] or added[1] < prompt[0]
    d.remove_random_window(0)
    assert len(d.windows()[1]) == 2
    assert d.readout["error"] is None
    d.remove_random_window(0)
    d.remove_random_window(0)
    assert d.windows()[1] == []
    assert "At least one random window" in d.readout["error"]


def test_drag_marks_prompt_or_adds_random_snapped_to_bins(qapp):
    s = make_session()
    d = TimeWindowDialog(s)
    d.set_windows((-G, G), [(500_000, MAX)])
    assert d.mark_span(-14.0, 13.0, mode="prompt") == (-G, G)
    assert d.windows()[0] == (-G, G)
    d.random_mode.setChecked(True)
    d.mark_span(-1234.0, -601.0)               # radio says: add a random
    assert d.windows()[1][-1] == (-1_230_000, -600_000)
    d._keys.held = "prompt"                    # held P overrides the radio
    d.mark_span(-26.0, 4.0)
    assert d.windows()[0] == (-30_000, 0)
    assert len(d.windows()[1]) == 2


def test_apply_writes_session_and_back_visit_restores(qapp):
    s = make_session()
    s.prompt_matrix = "stale"
    d = TimeWindowDialog(s)
    d.set_windows((-G, G), [(-MAX, -500_000), (500_000, MAX)])
    d.next_button.click()
    assert d.step_result == NEXT
    assert s.prompt_window == (-G, G)
    assert s.random_windows == [(-MAX, -500_000), (500_000, MAX)]
    assert s.weight == pytest.approx(3 / 302)
    assert (s.n_prompt_bins, s.n_random_bins) == (3, 302)
    assert s.prompt_matrix is None             # invalidate_from(5)

    s.prompt_matrix = "kept"
    d2 = TimeWindowDialog(s)                   # Back from step 5
    assert d2.windows() == ((-G, G), [(-MAX, -500_000), (500_000, MAX)])
    d2.next_button.click()
    assert s.prompt_matrix == "kept"           # unchanged windows keep results

    d3 = TimeWindowDialog(s)
    d3.prompt_row.lo.setValue(-20.0)
    d3.next_button.click()
    assert s.prompt_window == (-20_000, G) and s.prompt_matrix is None


def test_max_dt_snapped_to_granularity(qapp):
    assert snapped_max_dt(2_000_005, G) == MAX
    s = make_session(max_dt=2_004_999)
    d = TimeWindowDialog(s)
    assert d._max == MAX
    assert d.windows()[1][-1] == (500_000, MAX)


def test_log_toggle_and_dark_theme(qapp):
    from PySide6.QtWidgets import QWidget
    parent = QWidget()
    parent._theme = "dark"
    s = make_session()
    d = TimeWindowDialog(s, parent)
    assert d._theme == "dark"
    d.set_log_y(True)
    assert d.axes.get_yscale() == "log" and d.log_y.isChecked()
    d.log_y.setChecked(False)
    assert d.axes.get_yscale() == "linear"


def test_no_pairs_blocks_next(qapp):
    s = ListModeSession()
    s.listmode = SimpleNamespace(granularity_ps=G)
    d = TimeWindowDialog(s)
    assert not d.next_button.isEnabled()
    assert "no pairs" in d.apply_to_session()


# ----------------------------------------------------------------------
# Real data
# ----------------------------------------------------------------------

RUN = Path(__file__).resolve().parent.parent / "Na22_261126_ANI_STEF_180"
UNFILTERED = RUN / "UNFILTERED" / "SData_Na22_261126_ANI_STEF.root"


@pytest.mark.skipif(not UNFILTERED.exists(),
                    reason="Na22_261126_ANI_STEF_180/ (untracked run) absent")
def test_real_na22_prompt_and_random_windows(qapp):
    import listmode_io
    from listmode_sort import pair_events
    lm = listmode_io.open_listmode(UNFILTERED)
    assert lm.granularity_ps == G
    pairs = pair_events(listmode_io.iter_events(lm), max_dt_ps=MAX)
    s = ListModeSession(max_dt_ps=MAX, path=str(UNFILTERED))
    s.listmode = lm
    s.pairs = pairs
    d = TimeWindowDialog(s)
    d.set_windows((-G, G), [(-MAX, -500_000), (500_000, MAX)])
    r = d.readout
    assert r["error"] is None
    assert r["n_prompt_bins"] == 3 and r["n_random_bins"] == 302
    assert r["prompt_counts"] >= 43_387
    assert r["prompt_counts"] == int(window_mask(pairs.dt, (-G, G), G).sum())
    print(f"\nNa-22 real: w={r['w']:.6g} prompt={r['prompt_counts']} "
          f"random={r['random_total']} expected={r['expected_random']:.2f} "
          f"net={r['net']:.2f} +- {r['net_sigma']:.2f}")
