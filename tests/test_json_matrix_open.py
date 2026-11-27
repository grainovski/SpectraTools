"""File > Open Matrix of the JSON files SpectraTools writes (7.0.0): the
reader with metadata, the channel axes it gives the panel, and the
up-front refusal when a full-resolution sparse file would not fit in
memory (v7 plan, Task 6, carried over from Task 4)."""

import os

import numpy as np
import pytest

import system_memory
from channel_axis import ChannelAxis
from json_matrix_io import (
    axis_for_compression, compress_matrix, load_json_matrix_with_meta,
    memory_needed_to_open, save_json_matrix, sparse_shape,
)
from main_window import MainWindow

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
COMPASS_JSON = os.path.join(FIXTURES, "compass", "co60_EvsE.json")

META = {"compression": 1, "source_run": {"run_id": "R"}, "windows": {},
        "w": 0.25, "seed": 1, "gain_coefficients": None}


def _signed(n=64, seed=2):
    rng = np.random.default_rng(seed)
    m = rng.poisson(1.0, (n, n)) - rng.poisson(1.0, (n, n)) / 4.0
    m[10, 40] += 300.0
    return m


@pytest.fixture
def popups(monkeypatch):
    seen = []
    monkeypatch.setattr("main_window.QMessageBox.warning",
                        lambda parent, title, text, *a, **k: seen.append((title, text)))
    return seen


def _open_via_menu(monkeypatch, window, path):
    monkeypatch.setattr("main_window.QFileDialog.getOpenFileName",
                        lambda *a, **k: (str(path), ""))
    before = len(window._matrix_panels)
    window._open_matrix_dialog()
    return window._matrix_panels[before:]


def test_compressed_spectratools_json_opens_with_its_channel_axes(
        qapp, monkeypatch, tmp_path, popups):
    full = _signed(64)
    dense = compress_matrix(full, 4)
    path = tmp_path / "R_gg_sub_x4.json"
    save_json_matrix(path, dense, dict(META, compression=4), layout="dense")

    window = MainWindow()
    (panel,) = _open_via_menu(monkeypatch, window, path)
    assert popups == []
    axis = ChannelAxis(1.5, 4.0)
    assert axis_for_compression(4) == axis
    assert panel.channel_axes == {"x": axis, "y": axis}
    assert panel.matrix.dtype == np.float64
    np.testing.assert_allclose(panel.matrix, dense, atol=1e-12)
    # A peak at full-resolution X channel 40 reads as channel 41.5 (the
    # mean of 40..43, the bin holding it), not as bin 10.
    x_bin = int(np.argmax(panel.projections["x"]))
    assert x_bin == 40 // 4
    assert panel.channel_axes["x"].to_channel(x_bin) == pytest.approx(41.5)
    panel.close()
    window.close()


def test_full_resolution_sparse_json_opens_exactly(qapp, monkeypatch, tmp_path, popups):
    full = _signed(64)
    path = tmp_path / "R_gg_sub.json"
    save_json_matrix(path, full, META, layout="sparse")
    window = MainWindow()
    (panel,) = _open_via_menu(monkeypatch, window, path)
    assert popups == []
    assert panel.channel_axes == {"x": None, "y": None}
    np.testing.assert_array_equal(panel.matrix, full)
    panel.close()
    window.close()


def test_compass_json_still_opens_as_before(qapp, monkeypatch, popups):
    window = MainWindow()
    (panel,) = _open_via_menu(monkeypatch, window, COMPASS_JSON)
    assert popups == []
    assert panel.channel_axes == {"x": None, "y": None}
    assert panel.matrix.dtype == np.int64
    panel.close()
    window.close()


def _full_res_sparse(tmp_path, n=16384):
    """A 16384 x 16384 sparse file of a handful of cells: kilobytes on
    disk, 2 GiB once densified."""
    path = tmp_path / "full_gg_sub.json"
    idx = np.array([0, 5, n * n // 2 + 7, n * n - 1])
    save_json_matrix(path, ((n, n), idx, np.array([1.5, -2.0, 3.0, 4.0])), META)
    return path


def test_memory_estimate_for_a_full_resolution_sparse_file(tmp_path):
    path = _full_res_sparse(tmp_path)
    assert sparse_shape(path) == (16384, 16384)
    needed = memory_needed_to_open(path)
    assert needed >= 8 * 16384 * 16384          # the dense matrix alone
    assert needed < 8 * 16384 * 16384 + 2**20   # plus a tiny file's parse
    # Not estimated for the dense layouts (at most 4096 x 4096).
    assert sparse_shape(COMPASS_JSON) is None
    assert memory_needed_to_open(COMPASS_JSON) is None


def test_a_full_resolution_file_is_refused_when_memory_is_short(
        qapp, monkeypatch, tmp_path, popups):
    path = _full_res_sparse(tmp_path)
    monkeypatch.setattr(system_memory, "available_bytes", lambda: 1 * 2**30)
    loaded = []
    monkeypatch.setattr("main_window.load_json_matrix_with_meta",
                        lambda p: loaded.append(p))
    window = MainWindow()
    panels = _open_via_menu(monkeypatch, window, path)
    assert panels == [] and loaded == [], "the 2 GiB array must not even be tried"
    (title, text), = popups
    assert title == "Could not open matrix"
    assert "16,384 x 16,384" in text
    assert "2.0 GiB" in text and "only 1.0 GiB is available" in text
    window.close()


def test_the_same_file_opens_when_memory_suffices(qapp, monkeypatch, tmp_path, popups):
    path = _full_res_sparse(tmp_path, n=512)    # same layout, small
    monkeypatch.setattr(system_memory, "available_bytes", lambda: 64 * 2**30)
    window = MainWindow()
    (panel,) = _open_via_menu(monkeypatch, window, path)
    assert popups == []
    assert panel.matrix[0, 0] == 1.5 and panel.matrix[0, 5] == -2.0
    panel.close()
    window.close()


def test_memory_unknown_does_not_refuse(monkeypatch):
    monkeypatch.setattr(system_memory, "available_bytes", lambda: None)
    assert system_memory.shortfall_message("x", 10**15) is None


# --- the loader builds the dense array once --------------------------------


def test_sparse_loader_types_and_duplicates(tmp_path):
    p = tmp_path / "a.json"
    # Whole values -> int64; duplicates summed before the type is decided.
    p.write_text('{"format": "sparse", "shape": [2, 3], "spectratools": '
                 '{"compression": 1}, "x": [0, 0, 2], "y": [1, 1, 0], '
                 '"v": [0.5, 0.5, -3]}')
    m, meta, axes = load_json_matrix_with_meta(p)
    assert m.dtype == np.int64
    np.testing.assert_array_equal(m, [[0, 0, -3], [1, 0, 0]])
    # No cells at all: an all-zero matrix, not an error.
    p.write_text('{"format": "sparse", "shape": [2, 2], "spectratools": '
                 '{"compression": 1}, "x": [], "y": [], "v": []}')
    m, _meta, _axes = load_json_matrix_with_meta(p)
    np.testing.assert_array_equal(m, np.zeros((2, 2)))


def test_sparse_loader_makes_no_full_size_temporaries(tmp_path):
    """Checking and casting used to run on the dense array: rint, a
    compare, abs and an int64 cast each made a full-size array, every page
    of it touched. Now the values are checked on the sparse cells and the
    dense array is made once, in its final dtype, so only the pages that
    hold a cell are ever touched. Measured in a fresh process (peak RSS is
    monotonic, so an earlier test's peak would hide this one), as the
    process's peak working set from the OS."""
    import subprocess
    import sys

    n = 4096                                     # 128 MiB dense
    path = tmp_path / "big.json"
    rng = np.random.default_rng(0)
    idx = np.unique(rng.integers(0, n * n, 2000))
    # Whole values: the int64 path, which also used to cast a full copy.
    save_json_matrix(path, ((n, n), idx, rng.integers(-5, 6, idx.size) * 1.0), META)
    code = (
        "import sys, system_memory as sm\n"
        "from json_matrix_io import load_json_matrix_with_meta\n"
        "base = sm.process_peak_rss_bytes()\n"
        "m, _, _ = load_json_matrix_with_meta(sys.argv[1])\n"
        "peak = sm.process_peak_rss_bytes()\n"
        "print(m.dtype, m.nbytes, -1 if base is None else peak - base)\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code, str(path)], cwd=root,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    dtype, nbytes, grew = out.stdout.split()
    if int(grew) < 0:
        pytest.skip("the OS does not report process memory here")
    assert dtype == "int64" and int(nbytes) == 8 * n * n
    # One full-size temporary would be 128 MiB; the cells touch < 8 MiB.
    assert int(grew) < int(nbytes) // 4, (int(grew) / 2**20, "MiB")


def test_system_memory_reports_this_machine():
    avail, total = system_memory.available_bytes(), system_memory.total_bytes()
    rss = system_memory.process_rss_bytes()
    assert avail is not None and total is not None and rss is not None
    assert 0 < avail <= total and 0 < rss < total
    before = system_memory.process_rss_bytes()
    block = np.ones(64 * 2**20 // 8)             # 64 MiB, every page touched
    grew = system_memory.process_rss_bytes() - before
    assert 48 * 2**20 < grew < 96 * 2**20, grew / 2**20
    del block
