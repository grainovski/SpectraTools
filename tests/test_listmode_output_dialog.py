"""Step 5 of the list-mode operation: sort, subtract, write, open.

Every file is read back with the app's own readers and compared with an
independent recomputation from listmode_sort, so the test proves what is
on disk, not what the dialog meant to write.
"""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import listmode_output_dialog as lod
from gain_match import GainMatch
from json_matrix_io import (
    REQUIRED_META_KEYS, axis_for_compression, compress_matrix,
    load_json_matrix_with_meta,
)
from listmode_output_dialog import (
    ListModeOutputDialog, OutputChoice, default_base_name, default_folder,
    dense_factors, sort_and_write,
)
from listmode_session import NEXT, ListModeSession
from listmode_sort import (
    Pairs, sort_matrix, stochastic_round, subtract, to_dense, window_mask,
    window_weight,
)
from matrix_symmetrize import DEFAULT_SEED
from mtx_io import load_mtx

G = 10_000
MAX = 2_000_000
NCHAN = 64
PROMPT = (-G, G)
RANDOMS = [(-MAX, -500_000), (500_000, MAX)]


def synthetic_pairs(seed=3, n_prompt=4000, n_random=6000):
    """Prompt pairs on a few correlated cells plus a flat random floor;
    some X energies lie above the matrix so dropping is exercised."""
    rng = np.random.default_rng(seed)
    p_dt = rng.choice([-G, 0, G], size=n_prompt, p=[0.05, 0.9, 0.05])
    p_ex = rng.choice([10, 30, 50, 70], size=n_prompt)       # 70 > 63: off the edge
    p_ey = rng.choice([12, 33], size=n_prompt)
    k = MAX // G
    r_dt = rng.integers(-k, k + 1, size=n_random) * G
    r_ex = rng.integers(0, NCHAN + 4, size=n_random)
    r_ey = rng.integers(0, NCHAN, size=n_random)
    return Pairs(np.concatenate([p_dt, r_dt]).astype(np.int64),
                 np.concatenate([p_ex, r_ex]).astype(np.int64),
                 np.concatenate([p_ey, r_ey]).astype(np.int64))


def make_session(tmp_path, gain="fit", nchan=NCHAN, pairs=None):
    run = tmp_path / "Run_A"
    (run / "UNFILTERED").mkdir(parents=True)
    s = ListModeSession(path=str(run / "UNFILTERED" / "SData_Run_A.root"),
                        max_dt_ps=MAX, nchan=nchan)
    s.listmode = SimpleNamespace(granularity_ps=G)
    s.pairs = synthetic_pairs() if pairs is None else pairs
    if gain == "fit":
        s.gain = GainMatch([1.5, 0.93], nchan=nchan)
    else:
        s.gain_skipped = True
    s.prompt_window = PROMPT
    s.random_windows = list(RANDOMS)
    s.weight, s.n_prompt_bins, s.n_random_bins = window_weight(PROMPT, RANDOMS, G, MAX)
    return s


def expected_matrices(s, seed=DEFAULT_SEED):
    """P, R, S recomputed independently of the dialog."""
    def x_map(seed_):
        if s.gain_skipped:
            return None

        def f(e):
            ch, ok = s.gain.apply(e, seed=seed_, nchan=s.nchan)
            return np.where(ok, ch, -1)
        return f
    shape = (s.nchan, s.nchan)
    pm = window_mask(s.pairs.dt, s.prompt_window, G)
    rm = window_mask(s.pairs.dt, s.random_windows, G)
    P = sort_matrix(s.pairs.ex, s.pairs.ey, pm, shape, x_map(seed))
    R = sort_matrix(s.pairs.ex, s.pairs.ey, rm, shape, x_map(seed + 1))
    return P, R, subtract(P.matrix, R.matrix, s.weight), int(pm.sum()), int(rm.sum())


def pick_factor(d, f):
    d.factor_combo.setCurrentIndex(d.factor_combo.findData(f))


@pytest.fixture
def no_dialogs(monkeypatch):
    """Record warnings instead of showing them; never block."""
    seen = {"warning": [], "question": []}
    monkeypatch.setattr(lod.QMessageBox, "warning",
                        lambda *a, **k: seen["warning"].append(a[2]))
    return seen


def test_helpers_defaults_and_factors(tmp_path):
    s = make_session(tmp_path)
    assert default_folder(s) == (tmp_path / "Run_A").resolve()
    assert default_base_name(s) == "Run_A_gg_sub"
    s.run_info = SimpleNamespace(run_id="Na22_261126_ANI_STEF")
    assert default_base_name(s) == "Na22_261126_ANI_STEF_gg_sub"
    assert dense_factors(16384) == [4, 8, 16, 32, 64]
    assert dense_factors(64) == [1, 2, 4, 8, 16, 32, 64]
    assert dense_factors(5000) == [2, 4, 8]
    c = OutputChoice(tmp_path, "b", True, True, 16)
    assert [p.name for _k, p in c.files()] == ["b.mtx", "b.json", "b_x16.json"]


def test_sort_and_save_writes_every_ticked_format_exactly(qapp, tmp_path, no_dialogs):
    s = make_session(tmp_path)
    d = ListModeOutputDialog(s)
    assert d.next_button.text() == "Sort and Save"
    assert d.folder_edit.text() == str((tmp_path / "Run_A").resolve())
    d.folder_edit.setText(str(tmp_path))
    d.dense_check.setChecked(True)
    pick_factor(d, 4)
    assert "Run_A_gg_sub_x4.json" in d.planned_label.text()
    d.next_button.click()                      # Sort and Save
    assert not no_dialogs["warning"], no_dialogs["warning"]
    assert d.next_button.text() == "Finish"

    P, R, S, n_p, n_r = expected_matrices(s)
    mtx = tmp_path / "Run_A_gg_sub.mtx"
    sparse = tmp_path / "Run_A_gg_sub.json"
    dense = tmp_path / "Run_A_gg_sub_x4.json"
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == sorted(
        [mtx.name, sparse.name, dense.name])

    # .mtx: the stochastically rounded S, seed + 2.
    expected_int = to_dense(stochastic_round(S, seed=DEFAULT_SEED + 2))
    np.testing.assert_array_equal(load_mtx(str(mtx)), expected_int)
    # Must-fail control: the comparison can tell rounding streams apart.
    assert not np.array_equal(load_mtx(str(mtx)),
                              to_dense(stochastic_round(S, seed=DEFAULT_SEED)))
    # Sparse JSON: S exactly.
    m, meta, axes = load_json_matrix_with_meta(str(sparse))
    np.testing.assert_array_equal(m, to_dense(S))
    assert axes == (None, None)
    for key in REQUIRED_META_KEYS:
        assert key in meta
    assert meta["axes"] == {"x": "CH1", "y": "CH0"}
    assert meta["compression"] == 1
    assert meta["windows"]["prompt"] == list(PROMPT)
    assert meta["windows"]["randoms"] == [list(r) for r in RANDOMS]
    assert meta["w"] == s.weight and meta["seed"] == DEFAULT_SEED
    assert meta["gain_coefficients"] == [1.5, 0.93]
    assert meta["source_run"]["run_id"] == "Run_A"
    assert meta["totals"]["net"] == pytest.approx(
        meta["totals"]["prompt"] - s.weight * meta["totals"]["random"])
    # Dense JSON: S compressed by 4, with the matching channel axes.
    md, metad, axesd = load_json_matrix_with_meta(str(dense))
    np.testing.assert_allclose(md, compress_matrix((S.shape, S.index, S.values), 4),
                               rtol=0, atol=1e-9)
    assert metad["compression"] == 4
    assert axesd == (axis_for_compression(4), axis_for_compression(4))

    # Session and summary.
    assert s.result_matrix.index.tolist() == S.index.tolist()
    np.testing.assert_array_equal(s.result_matrix.values, S.values)
    assert s.prompt_matrix.total() == P.matrix.total()
    assert s.dropped == P.dropped + R.dropped > 0
    assert s.written == [str(mtx), str(sparse), str(dense)]
    o = d.outcome
    assert (o.n_prompt_pairs, o.n_random_pairs) == (n_p, n_r)
    assert float(o.result.total()) == pytest.approx(
        P.matrix.total() - s.weight * R.matrix.total())
    text = d.summary_label.text()
    assert f"{n_p:,} pairs" in text and "Written:" in text and mtx.name in text

    d.next_button.click()                      # Finish
    assert d.step_result == NEXT


def test_only_ticked_files_are_written(qapp, tmp_path, no_dialogs):
    s = make_session(tmp_path, gain="skip")
    d = ListModeOutputDialog(s)
    d.folder_edit.setText(str(tmp_path))
    d.mtx_check.setChecked(False)
    assert d.sort_and_save()
    files = [p.name for p in tmp_path.iterdir() if p.is_file()]
    assert files == ["Run_A_gg_sub.json"]
    _P, _R, S, _np, _nr = expected_matrices(s)
    m, meta, _axes = load_json_matrix_with_meta(str(tmp_path / files[0]))
    np.testing.assert_array_equal(m, to_dense(S))
    assert meta["gain_coefficients"] is None and meta["gain"] == "skipped"


def test_gain_skipped_is_identity_and_counts_off_edge(qapp, tmp_path, no_dialogs):
    s = make_session(tmp_path, gain="skip")
    o = sort_and_write(s, OutputChoice(tmp_path, "x", False, False, None))
    pm = window_mask(s.pairs.dt, PROMPT, G)
    off = int(np.count_nonzero(s.pairs.ex[pm] >= NCHAN))
    assert o.prompt.dropped == off > 0
    # Identity: the prompt X projection is the raw ex histogram.
    ex_in = s.pairs.ex[pm][s.pairs.ex[pm] < NCHAN]
    np.testing.assert_array_equal(o.prompt.matrix.x_projection(),
                                  np.bincount(ex_in, minlength=NCHAN))
    assert o.written == []


def test_nothing_ticked_or_bad_folder_refused(qapp, tmp_path, no_dialogs):
    s = make_session(tmp_path)
    d = ListModeOutputDialog(s)
    d.mtx_check.setChecked(False)
    d.sparse_check.setChecked(False)
    assert not d.sort_and_save()
    assert "Tick at least one" in no_dialogs["warning"][-1]
    d.sparse_check.setChecked(True)
    d.folder_edit.setText(str(tmp_path / "missing"))
    assert not d.sort_and_save()
    assert "does not exist" in no_dialogs["warning"][-1]
    d.folder_edit.setText(str(tmp_path))
    d.base_edit.setText("a/b")
    assert not d.sort_and_save()
    assert s.result_matrix is None and s.written == []


def test_overwrite_warning(qapp, tmp_path, no_dialogs, monkeypatch):
    s = make_session(tmp_path)
    target = tmp_path / "Run_A_gg_sub.json"
    target.write_text("old")
    asked = []
    answer = {"value": lod.QMessageBox.StandardButton.No}

    def question(parent, title, text, *a, **k):
        asked.append(text)
        return answer["value"]
    monkeypatch.setattr(lod.QMessageBox, "question", question)
    d = ListModeOutputDialog(s)
    d.folder_edit.setText(str(tmp_path))
    d.mtx_check.setChecked(False)
    assert not d.sort_and_save()
    assert asked and str(target) in asked[0]
    assert target.read_text() == "old" and s.written == []

    answer["value"] = lod.QMessageBox.StandardButton.Yes
    assert d.sort_and_save()
    assert target.read_text() != "old"
    assert s.written == [str(target)]


def test_failed_writer_leaves_nothing_claimed(qapp, tmp_path, no_dialogs, monkeypatch):
    s = make_session(tmp_path)

    def broken(path, *a, **k):
        Path(path).write_text("half")          # a truncated file ...
        raise OSError("disk full")              # ... then the failure
    monkeypatch.setattr(lod, "save_json_matrix", broken)
    d = ListModeOutputDialog(s)
    d.folder_edit.setText(str(tmp_path))
    assert not d.sort_and_save()
    assert "disk full" in no_dialogs["warning"][-1]
    names = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert names == ["Run_A_gg_sub.mtx"]       # neither .json nor .partial
    assert s.written == [str(tmp_path / "Run_A_gg_sub.mtx")]
    assert d.next_button.text() == "Sort and Save"


def test_sorting_error_reported_and_nothing_recorded(qapp, tmp_path, no_dialogs):
    s = make_session(tmp_path)
    s.gain = None                               # neither matched nor skipped
    d = ListModeOutputDialog(s)
    d.folder_edit.setText(str(tmp_path))
    assert not d.sort_and_save()
    assert "Gain matching has not been done" in no_dialogs["warning"][-1]
    assert s.result_matrix is None and s.written == []
    assert [p for p in tmp_path.iterdir() if p.is_file()] == []


def test_open_result_full_and_compressed(qapp, tmp_path, no_dialogs, monkeypatch):
    s = make_session(tmp_path)
    opened = []
    s.open_matrix = lambda name, matrix, axes: opened.append((name, matrix, axes))
    d = ListModeOutputDialog(s)
    d.folder_edit.setText(str(tmp_path))
    d.mtx_check.setChecked(False)
    assert not d.open_button.isEnabled()
    assert d.sort_and_save()
    assert d.open_button.isEnabled()
    _P, _R, S, _a, _b = expected_matrices(s)

    # Small: opens straight away at full resolution.
    assert d.open_result() == "full"
    name, matrix, axes = opened[-1]
    assert name == "Run_A_gg_sub" and axes == (None, None)
    assert matrix.dtype == np.float64 and matrix.shape == (NCHAN, NCHAN)
    np.testing.assert_array_equal(matrix, to_dense(S))

    # Big (threshold lowered): asks, and offers the dense factor ticked.
    monkeypatch.setattr(lod, "OPEN_WARN_BYTES", 0)
    offers = []
    d._ask_open = lambda nbytes, factor: offers.append((nbytes, factor)) or "compressed"
    d.dense_check.setChecked(True)
    pick_factor(d, 8)
    assert d.open_result() == "compressed"
    assert offers[-1] == (NCHAN * NCHAN * 8, 8)
    name, matrix, axes = opened[-1]
    assert matrix.shape == (8, 8) and matrix.dtype == np.float64
    np.testing.assert_allclose(matrix, compress_matrix((S.shape, S.index, S.values), 8),
                               atol=1e-9)
    assert axes == (axis_for_compression(8), axis_for_compression(8))

    d._ask_open = lambda nbytes, factor: None   # Cancel
    n = len(opened)
    assert d.open_result() is None and len(opened) == n


def test_option_change_after_save_rearms_sort_and_save(qapp, tmp_path, no_dialogs):
    s = make_session(tmp_path)
    d = ListModeOutputDialog(s)
    d.folder_edit.setText(str(tmp_path))
    d.mtx_check.setChecked(False)
    assert d.sort_and_save() and d.next_button.text() == "Finish"
    d.mtx_check.setChecked(True)
    assert d.next_button.text() == "Sort and Save"


# ----------------------------------------------------------------------
# Real data
# ----------------------------------------------------------------------

RUN = Path(__file__).resolve().parent.parent / "Na22_261126_ANI_STEF_180"
UNFILTERED = RUN / "UNFILTERED" / "SData_Na22_261126_ANI_STEF.root"


def _read_mtx_sparse(path):
    """(flat index, value) of a .mtx's non-zero cells without a dense
    16384^2 array (load_mtx would build 2 GiB)."""
    from lc_codec import decode_row
    blob = Path(path).read_bytes()
    _magic, _ver, _lev, lines, cols, tpos = struct.unpack("<6I", blob[:24])
    table = struct.unpack(f"<{2 * lines}I", blob[tpos:tpos + 8 * lines])
    idx, vals = [], []
    for r in range(lines):
        pos, n = table[2 * r], table[2 * r + 1]
        row = np.fromiter(decode_row(blob[pos:pos + n], cols, path, kind="mtx"),
                          dtype=np.int64, count=cols)
        nz = np.flatnonzero(row)
        idx.append(r * cols + nz)
        vals.append(row[nz])
    return (lines, cols), np.concatenate(idx), np.concatenate(vals)


@pytest.mark.skipif(not UNFILTERED.exists(),
                    reason="Na22_261126_ANI_STEF_180/ (untracked run) absent")
def test_real_na22_full_resolution_write(qapp, tmp_path, no_dialogs):
    import listmode_io
    from listmode_sort import pair_events
    lm = listmode_io.open_listmode(UNFILTERED)
    pairs = pair_events(listmode_io.iter_events(lm), max_dt_ps=MAX)
    s = ListModeSession(path=str(UNFILTERED), max_dt_ps=MAX, nchan=16384)
    s.listmode = lm
    s.run_info = listmode_io.read_run_info(UNFILTERED)
    s.pairs = pairs
    s.gain_skipped = True
    s.prompt_window = PROMPT
    s.random_windows = list(RANDOMS)
    s.weight, s.n_prompt_bins, s.n_random_bins = window_weight(PROMPT, RANDOMS, G, MAX)
    d = ListModeOutputDialog(s)
    assert d.folder_edit.text() == str(RUN.resolve())
    d.folder_edit.setText(str(tmp_path))
    assert d.sort_and_save(), no_dialogs["warning"]
    o = d.outcome
    S = o.result
    print(f"\nNa-22 real sort: w={o.w:.6g} prompt pairs={o.n_prompt_pairs} "
          f"random pairs={o.n_random_pairs} P={int(o.prompt.matrix.total())} "
          f"R={int(o.random.matrix.total())} net={float(S.total()):.2f} "
          f"dropped={o.dropped} cells={len(S.index)} "
          f"files={[(p.name, n) for p, n in o.written]}")
    assert o.n_prompt_pairs >= 43_387
    base = d.choice().base
    doc = json.loads((tmp_path / f"{base}.json").read_text())
    assert doc["shape"] == [16384, 16384]
    flat = np.asarray(doc["y"], dtype=np.int64) * 16384 + np.asarray(doc["x"])
    np.testing.assert_array_equal(flat, S.index)
    np.testing.assert_array_equal(np.asarray(doc["v"], dtype=np.float64), S.values)
    shape, idx, vals = _read_mtx_sparse(tmp_path / f"{base}.mtx")
    rounded = stochastic_round(S, seed=DEFAULT_SEED + 2)
    assert shape == (16384, 16384)
    np.testing.assert_array_equal(idx, rounded.index)
    np.testing.assert_array_equal(vals, rounded.values)
