"""Tests for the CoMPASS list-mode reader (listmode_io).

Synthetic SData files (tests/synthetic_sdata.py) cover the logic: every
field of the description, chunked reading, sortedness and granularity
detection, and each error. The real Na-22 run is untracked; its tests
skip by name when it is absent, and check the reader against numbers
CoMPASS itself wrote (entry counts, the run's EspectrumF .txt spectra,
the info.txt calibration).
"""

from pathlib import Path

import numpy as np
import pytest

uproot = pytest.importorskip("uproot")

import listmode_io
from histogram_io import ParseError
from listmode_io import (
    EventChunk,
    default_excluded_flags,
    describe_flags,
    iter_events,
    open_listmode,
    read_run_info,
)
from synthetic_sdata import simple_events, write_sdata

REPO = Path(__file__).resolve().parent.parent
RUN_DIR = REPO / "Na22_261126_ANI_STEF_180"
UNFILTERED = RUN_DIR / "UNFILTERED" / "SData_Na22_261126_ANI_STEF.root"
FILTERED = RUN_DIR / "FILTERED" / "SDataF_Na22_261126_ANI_STEF.root"
RAW = RUN_DIR / "RAW" / "SDataR_Na22_261126_ANI_STEF.root"
INFO = RUN_DIR / "Na22_261126_ANI_STEF_info.txt"


def _need(path):
    if not path.exists():
        pytest.skip(f"untracked real run file absent: {path.relative_to(REPO)}")


# ---------------------------------------------------------------- description


def test_description_of_synthetic_file(tmp_path):
    ch, ts, en, fl = simple_events(1000)
    path = write_sdata(tmp_path / "SData_x.root", ch, ts, en, fl, chunks=3)
    info = open_listmode(path)
    assert info.path == Path(path)
    assert info.tree_name == "Data"
    assert info.stream == "UNFILTERED"
    assert info.n_entries == 1000
    assert info.events_per_channel == {
        0: int(np.sum(ch == 0)), 1: int(np.sum(ch == 1))}
    values, counts = np.unique(fl, return_counts=True)
    assert info.flag_counts == dict(zip(values.tolist(), counts.tolist()))
    assert info.granularity_ps == 10000
    assert info.is_sorted is True


@pytest.mark.parametrize("tree,stream", [
    ("Data", "UNFILTERED"), ("Data_F", "FILTERED"), ("Data_R", "RAW")])
def test_stream_kind_from_tree_name(tmp_path, tree, stream):
    ch, ts, en, fl = simple_events(50)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl, tree=tree)
    info = open_listmode(path)
    assert (info.tree_name, info.stream) == (tree, stream)


def test_unsorted_file_reports_false(tmp_path):
    ch, ts, en, fl = simple_events(1000)
    ts = ts.copy()
    ts[[400, 401]] = ts[[401, 400]]   # one adjacent swap is enough
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl)
    assert open_listmode(path).is_sorted is False


def test_unsorted_across_a_chunk_boundary_is_detected(tmp_path):
    """The only decrease sits exactly between two read chunks, so a check
    that looked only inside each chunk would miss it."""
    ch, ts, en, fl = simple_events(1000)
    ts = ts.copy()
    ts[500:] -= ts[500] - ts[499] + np.uint64(10000)   # ts[500] < ts[499]
    assert np.all(np.diff(ts[:500].astype(np.int64)) > 0)
    assert np.all(np.diff(ts[500:].astype(np.int64)) > 0)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl)
    assert listmode_io.open_listmode(path, step_size=500).is_sorted is False
    assert listmode_io.open_listmode(path, step_size=100000).is_sorted is False


def test_equal_timestamps_still_count_as_sorted(tmp_path):
    """Two detectors firing in the same 10 ns tick share a timestamp; that
    is coincidence data, not disorder."""
    ch, ts, en, fl = simple_events(100)
    ts = ts.copy()
    ts[11] = ts[10]
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl)
    assert open_listmode(path).is_sorted is True


@pytest.mark.parametrize("tick", [1, 2000, 10000, 4000000])
def test_granularity_is_gcd_of_timestamps(tmp_path, tick):
    ch, ts, en, fl = simple_events(500, tick_ps=tick)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl)
    assert open_listmode(path).granularity_ps == tick


def test_granularity_ignores_zero_timestamps(tmp_path):
    ch, ts, en, fl = simple_events(200)
    ts = ts.copy()
    ts[0] = 0
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl)
    assert open_listmode(path).granularity_ps == 10000


def test_granularity_one_odd_timestamp_drops_it(tmp_path):
    """Must-fail control for the gcd: a single timestamp off the grid has to
    change the answer, or the detection proves nothing."""
    ch, ts, en, fl = simple_events(200)
    ts = ts.copy()
    ts[50] += np.uint64(5000)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl)
    assert open_listmode(path).granularity_ps == 5000


def test_granularity_none_when_all_timestamps_zero(tmp_path):
    ch, ts, en, fl = simple_events(20)
    path = write_sdata(tmp_path / "s.root", ch, np.zeros(20), en, fl)
    assert open_listmode(path).granularity_ps is None


def test_granularity_sample_is_limited(tmp_path):
    """Only the first `granularity_sample` events feed the gcd."""
    ch, ts, en, fl = simple_events(400)
    ts = ts.copy()
    ts[300] += np.uint64(5000)       # off-grid, but past the sample
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl)
    info = listmode_io.open_listmode(path, granularity_sample=200)
    assert info.granularity_ps == 10000


def test_empty_tree_describes_cleanly(tmp_path):
    path = write_sdata(tmp_path / "s.root", [], [], [], [])
    info = open_listmode(path)
    assert info.n_entries == 0
    assert info.events_per_channel == {}
    assert info.flag_counts == {}
    assert info.granularity_ps is None
    assert info.is_sorted is True
    assert list(iter_events(path)) == []


# ------------------------------------------------------------------ iteration


def test_iter_events_chunks_concatenate_to_whole(tmp_path):
    ch, ts, en, fl = simple_events(1000)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl, chunks=4)
    chunks = list(iter_events(path, step_size=300))
    assert len(chunks) == 4
    assert [len(c.channel) for c in chunks] == [300, 300, 300, 100]
    for c in chunks:
        assert isinstance(c, EventChunk)
        assert c._fields == ("channel", "timestamp", "energy", "flags")
        assert c.channel.dtype == np.uint16
        assert c.timestamp.dtype == np.int64
        assert c.energy.dtype == np.uint16
        assert c.flags.dtype == np.uint32
    whole = EventChunk(*(np.concatenate(parts) for parts in zip(*chunks)))
    np.testing.assert_array_equal(whole.channel, ch)
    np.testing.assert_array_equal(whole.timestamp, ts.astype(np.int64))
    np.testing.assert_array_equal(whole.energy, en)
    np.testing.assert_array_equal(whole.flags, fl)


def test_iter_events_accepts_listmode_file(tmp_path):
    ch, ts, en, fl = simple_events(100)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl, tree="Data_F")
    info = open_listmode(path)
    chunks = list(iter_events(info))
    np.testing.assert_array_equal(
        np.concatenate([c.energy for c in chunks]), en)


def test_iter_events_validates_a_bare_path(tmp_path):
    ch, ts, en, fl = simple_events(10)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl, drop=("Flags",))
    with pytest.raises(ParseError, match="Flags"):
        list(iter_events(path))


def test_timestamp_beyond_int64_is_refused(tmp_path):
    """Picoseconds as uint64 only overflow int64 after ~106 days of run
    time; such a value is corrupt rather than real, and must not wrap
    silently to a negative time."""
    ch, ts, en, fl = simple_events(10)
    ts = ts.copy()
    ts[5] = np.uint64(2**63 + 10)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl)
    with pytest.raises(ParseError, match="int64|range"):
        list(iter_events(path))


# --------------------------------------------------------------------- errors


def test_missing_file(tmp_path):
    missing = tmp_path / "nope.root"
    with pytest.raises(ParseError, match="nope.root"):
        open_listmode(missing)


def test_non_root_file(tmp_path):
    path = tmp_path / "notroot.root"
    path.write_text("Channel;Timestamp\n0;1\n")
    with pytest.raises(ParseError, match="notroot.root"):
        open_listmode(path)


def test_histograms_only_file(tmp_path):
    hist = (np.arange(10.0), np.linspace(0, 10, 11))
    path = write_sdata(tmp_path / "Hcompass.root", [], [], [], tree=None,
                       extra_objects={"Energy/h": hist})
    with pytest.raises(ParseError) as exc:
        open_listmode(path)
    assert "Hcompass.root" in str(exc.value)
    assert "histogram" in str(exc.value).lower()


def test_wrong_tree_name(tmp_path):
    ch, ts, en, fl = simple_events(10)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl, tree="Events")
    with pytest.raises(ParseError) as exc:
        open_listmode(path)
    msg = str(exc.value)
    assert "s.root" in msg and "Events" in msg and "Data_F" in msg


@pytest.mark.parametrize("branch", list(listmode_io.REQUIRED_BRANCHES))
def test_missing_branch(tmp_path, branch):
    ch, ts, en, fl = simple_events(10)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl, drop=(branch,))
    with pytest.raises(ParseError) as exc:
        open_listmode(path)
    assert branch in str(exc.value) and "s.root" in str(exc.value)


def test_wrong_branch_type(tmp_path):
    ch, ts, en, fl = simple_events(10)
    path = write_sdata(tmp_path / "s.root", ch, ts, en, fl,
                       dtypes={"Energy": np.float32})
    with pytest.raises(ParseError) as exc:
        open_listmode(path)
    assert "Energy" in str(exc.value) and "uint16" in str(exc.value)


# ---------------------------------------------------------------------- flags


def test_describe_flags_known_and_unknown_bits():
    assert describe_flags(0) == []
    assert describe_flags(0x8000) == [listmode_io.FLAG_BITS[0x8000]]
    both = describe_flags(0x8800)
    assert both == [listmode_io.FLAG_BITS[0x800], listmode_io.FLAG_BITS[0x8000]]
    assert describe_flags(0x2) == ["unknown (bit 0x00000002)"]


def test_flag_table_bits_are_single_bits():
    for bit in listmode_io.FLAG_BITS:
        assert bit > 0 and bit & (bit - 1) == 0


def test_default_excluded_flags():
    seen = [0x0, 0x80, 0x480, 0x800, 0xC80, 0x8000, 0x8800]
    assert default_excluded_flags(seen) == {0x80, 0x480, 0xC80, 0x8000, 0x8800}
    assert default_excluded_flags([0x0, 0x800]) == set()
    assert default_excluded_flags([]) == set()


# ------------------------------------------------------------------- run info


INFO_TEXT = (
    "Run ID = R1\r\n"
    "Start time = Thu Nov 26 11:41:27 2026\r\n"
    "Stop time = Thu Nov 26 11:51:27 2026\r\n"
    "\r\n"
    "CH0@DT5780SCP_58363\r\n"
    "\tEnergy calibration: C0 = -1.5; C1 = 0.25; C2 = 1e-06; unit = keV\r\n"
    "\tReal time = 1:02:03.500\r\n"
    "\tLive time = 0:59:59.250\r\n"
    "\tDead time = 0.17 %\r\n"
    "\tInput counts = 1000; average rate (cps) = 3.90\xb10.81\r\n"
    "\tPile up counts = 7; average rate (cps) = 0.343\xb10.024\r\n"
    "\tOutput counts = 900; average rate (cps) = 2.1\xb10.60\r\n"
    "\r\n"
    "CH3@DT5780SCP_58363\r\n"
    "\tEnergy calibration: C0 = 0; C1 = 1; C2 = 0; unit = ch\r\n"
    "\tReal time = 0:00:10.000\r\n"
    "\tInput counts = 5; average rate (cps) = 0.5\r\n"
)


def _write_run(tmp_path, text=INFO_TEXT, run_id="R1"):
    run = tmp_path / "run"
    (run / "UNFILTERED").mkdir(parents=True)
    root = run / "UNFILTERED" / f"SData_{run_id}.root"
    root.write_bytes(b"")
    (run / f"{run_id}_info.txt").write_bytes(text.encode("latin-1"))
    return root


def test_read_run_info_synthetic(tmp_path):
    info = read_run_info(_write_run(tmp_path))
    assert info.run_id == "R1"
    assert info.start == "Thu Nov 26 11:41:27 2026"
    assert info.stop == "Thu Nov 26 11:51:27 2026"
    assert sorted(info.channels) == [0, 3]
    ch0 = info.channels[0]
    assert ch0.label == "CH0@DT5780SCP_58363"
    assert ch0.board == "DT5780SCP_58363"
    assert (ch0.c0, ch0.c1, ch0.c2, ch0.unit) == (-1.5, 0.25, 1e-06, "keV")
    assert ch0.real_time_s == pytest.approx(3723.5)
    assert ch0.live_time_s == pytest.approx(3599.25)
    assert ch0.dead_time_percent == pytest.approx(0.17)
    assert ch0.input_counts == 1000
    assert ch0.output_counts == 900
    assert ch0.counts["Pile up"] == 7
    ch3 = info.channels[3]
    assert ch3.live_time_s is None and ch3.output_counts is None
    assert ch3.unit == "ch"


def test_read_run_info_accepts_folder_or_info_path(tmp_path):
    root = _write_run(tmp_path)
    assert read_run_info(root.parent).run_id == "R1"           # stream folder
    assert read_run_info(root.parent.parent).run_id == "R1"    # run folder
    info_file = root.parent.parent / "R1_info.txt"
    assert read_run_info(info_file).run_id == "R1"


def test_read_run_info_prefers_matching_run_id(tmp_path):
    root = _write_run(tmp_path)
    other = INFO_TEXT.replace("Run ID = R1", "Run ID = AAA")
    (root.parent.parent / "AAA_info.txt").write_bytes(other.encode("latin-1"))
    assert read_run_info(root).run_id == "R1"


def test_read_run_info_missing_returns_none(tmp_path):
    lone = tmp_path / "a" / "b" / "SData_X.root"
    lone.parent.mkdir(parents=True)
    lone.write_bytes(b"")
    assert read_run_info(lone) is None
    assert read_run_info(tmp_path / "does" / "not" / "exist.root") is None


def test_read_run_info_garbage_returns_none(tmp_path):
    root = _write_run(tmp_path, text="\x00\xff not an info file \x01")
    assert read_run_info(root) is None


# ------------------------------------------------------------ real run (opt.)


@pytest.mark.parametrize("path,tree,stream,n,per_channel", [
    (UNFILTERED, "Data", "UNFILTERED", 655_336, {0: 328_431, 1: 326_905}),
    (FILTERED, "Data_F", "FILTERED", 264_245, {0: 131_707, 1: 132_538}),
    (RAW, "Data_R", "RAW", 656_167, None),
])
def test_real_run_description(path, tree, stream, n, per_channel):
    _need(path)
    info = open_listmode(path)
    assert (info.tree_name, info.stream, info.n_entries) == (tree, stream, n)
    if per_channel is not None:
        assert info.events_per_channel == per_channel
    assert sum(info.events_per_channel.values()) == n
    assert info.granularity_ps == 10000
    assert info.is_sorted is True


def test_real_filtered_flags():
    _need(FILTERED)
    assert open_listmode(FILTERED).flag_counts == {0x0: 263_993, 0x800: 252}


def test_real_raw_flag_bits_match_info_counts():
    """CAEN's flag meanings, checked against CoMPASS's own bookkeeping: the
    info.txt 'Pile up counts' equal the events with bit 0x8000, and its
    'Saturation counts' the events with bit 0x80, per channel."""
    _need(RAW)
    _need(INFO)
    run = read_run_info(RAW)
    pileup = {0: 0, 1: 0}
    saturation = {0: 0, 1: 0}
    for chunk in iter_events(RAW, step_size=200_000):
        for c in (0, 1):
            f = chunk.flags[chunk.channel == c]
            pileup[c] += int(np.count_nonzero(f & 0x8000))
            saturation[c] += int(np.count_nonzero(f & 0x80))
    for c in (0, 1):
        assert pileup[c] == run.channels[c].counts["Pile up"]
        assert saturation[c] == run.channels[c].counts["Saturation"]


def _espectrum_txt(c):
    return (RUN_DIR / "FILTERED" /
            f"CH{c}@DT5780SCP_58363_EspectrumF_Na22_261126_ANI_STEF_20261126_115129.txt")


def test_real_filtered_bincount_equals_compass_spectrum():
    _need(FILTERED)
    for c in (0, 1):
        _need(_espectrum_txt(c))
    hists = {0: np.zeros(16383, dtype=np.int64), 1: np.zeros(16383, dtype=np.int64)}
    n_chunks = 0
    for chunk in iter_events(FILTERED, step_size=100_000):
        n_chunks += 1
        for c in (0, 1):
            e = chunk.energy[chunk.channel == c]
            hists[c] += np.bincount(e, minlength=16383)[:16383]
    assert n_chunks >= 3          # the comparison really went through chunks
    for c in (0, 1):
        expected = np.loadtxt(_espectrum_txt(c))
        assert expected.shape == (16383,)
        assert expected.sum() > 100_000
        np.testing.assert_array_equal(hists[c], expected)
        # Must-fail control: one count moved must break the equality.
        tampered = hists[c].copy()
        peak = int(np.argmax(tampered))
        tampered[peak] -= 1
        tampered[peak + 1] += 1
        assert not np.array_equal(tampered, expected)


def test_real_run_info():
    _need(INFO)
    _need(UNFILTERED)
    info = read_run_info(UNFILTERED)
    assert info.run_id == "Na22_261126_ANI_STEF"
    assert info.start == "Thu Nov 26 11:41:27 2026"
    assert info.stop == "Thu Nov 26 11:51:27 2026"
    assert (info.channels[0].c0, info.channels[0].c1) == (-19.567, 0.509)
    assert (info.channels[1].c0, info.channels[1].c1) == (-35.906, 0.515)
    assert info.channels[0].unit == "keV"
    assert info.channels[0].real_time_s == pytest.approx(600.120)
    assert info.channels[0].live_time_s == pytest.approx(599.092)
    assert info.channels[0].input_counts == 236_386
    assert info.channels[0].output_counts == 131_707
    assert info.channels[1].output_counts == 132_538
