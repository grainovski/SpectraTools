"""Read-only reader for CoMPASS list-mode ("SData") ROOT files.

CAEN's CoMPASS acquisition software can save every event, not just the
histograms built from them, as a ROOT TTree: one entry per event with the
channel, a picosecond timestamp, the ADC energy and a word of flags. That
list is what gamma-gamma sorting needs (v7 plan, Task 1); the histogram
files beside it (Hcompass*.root) are handled by root_io instead.

CoMPASS writes one tree per "stream", and the tree NAME is the only thing
that says which stream a file holds:

    Data_R  RAW         everything the board sent
    Data    UNFILTERED  RAW minus pile-up and saturated events
    Data_F  FILTERED    UNFILTERED after CoMPASS's energy/PSD/time cuts and
                        its software coincidence

(The RAW -> UNFILTERED difference is measured, not assumed: in the Na-22
reference run RAW has exactly the 0x8000 + 0x80 flagged events more, see
FLAG_BITS.) The file is never loaded whole: the description is built in one
chunked pass, and events are handed out as chunks, because a long run is
easily tens of millions of events.

Like root_io, built on uproot (pure Python, freezable) and imported lazily.
"""

import math
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

from histogram_io import ParseError

# Tree name -> stream kind. Recognised by name, then validated by branches.
STREAMS = {"Data_R": "RAW", "Data": "UNFILTERED", "Data_F": "FILTERED"}

# The six branches CoMPASS 2.x writes, with their element types (stored
# big-endian; uproot reports the native type it converts to). All six are
# required even though sorting reads only four: a file that lacks one is
# not the format this reader was checked against, and saying so up front
# beats a half-working sort.
REQUIRED_BRANCHES = {
    "Channel": np.dtype(np.uint16),
    "Timestamp": np.dtype(np.uint64),
    "Board": np.dtype(np.uint16),
    "Energy": np.dtype(np.uint16),
    "CalibEnergy": np.dtype(np.float64),
    "Flags": np.dtype(np.uint32),
}

# What iter_events reads. CalibEnergy (= C0 + C1*Energy, recomputable) and
# Board (one board here) are not needed to sort, so they are not read.
_EVENT_BRANCHES = ["Channel", "Timestamp", "Energy", "Flags"]

DEFAULT_STEP_SIZE = 1_000_000          # entries per chunk (~19 MB of arrays)
DEFAULT_GRANULARITY_SAMPLE = 200_000   # events feeding the timestamp gcd

_INT64_MAX = np.iinfo(np.int64).max

# --------------------------------------------------------------------- flags
#
# Meaning of the CoMPASS list-mode `Flags` bits.
#
# Source: CAEN, "User Manual UM5960 - CoMPASS: Multiparametric DAQ Software
# for Physics Applications", list-file flag table. The manual itself is
# behind CAEN's documentation area (https://www.caen.it/support-services/
# documentation-area/); the text was read from the copy at
# https://docplayer.net/89228693-User-manual-um5960-compass-multiparametric-
# daq-software-for-physics-applications.html (revision not visible there).
# Only the bits listed below could be read from that source; every other bit
# is reported as "unknown" rather than guessed.
#
# Independent cross-checks, 2026-11-27:
#  * The Berkeley Nuclear Data software's CoMPASS event class
#    (https://nucleardata.berkeley.edu/nsd_software/doxy/html/d7/dcd/
#    classcompass_ev_strct.html) tests bit 15 for pile-up, bit 10 for ADC
#    (input) saturation and bit 7 for trapezoid saturation in DPP-PHA
#    (QDC saturation in DPP-PSD) -- the same three bits.
#  * CoMPASS's own <run>_info.txt for the Na-22 reference run: per channel,
#    'Pile up counts' equals the number of RAW events with 0x8000 set
#    (206 / 243), and 'Saturation counts' the number with 0x80 set
#    (174 / 208). Every 0x400 event in that run also has 0x80.
FLAG_BITS = {
    0x00000080: "saturating inside the gate (clipping)",
    0x00000400: "input saturating",
    0x00000800: "N triggers counted (N set by register 0x1n84 bits [17:16])",
    0x00004000: "event with fine time stamp",
    0x00008000: "piled-up event",
}

# Bits the source names as pile-up or saturation. Nothing else is excluded
# by default: 0x800 is a trigger-counter marker on an otherwise good event
# (UNFILTERED keeps it), and 0x4000 says only that a fine time is present.
PILEUP_SATURATION_BITS = 0x00000080 | 0x00000400 | 0x00008000


def describe_flags(value):
    """List of descriptions, one per bit set in `value`, lowest bit first.
    Bits the source does not document come back as 'unknown (bit 0x...)'
    so they are visible, not silently dropped."""
    value = int(value)
    out = []
    bit = 1
    while bit <= value:
        if value & bit:
            out.append(FLAG_BITS.get(bit, f"unknown (bit 0x{bit:08X})"))
        bit <<= 1
    return out


def default_excluded_flags(flag_values):
    """The subset of `flag_values` to exclude by default: every value with a
    pile-up or saturation bit set (0x8000, 0x400, 0x80), as UM5960 names
    them. CoMPASS's UNFILTERED stream drops exactly these, so applying the
    default to a RAW file reproduces UNFILTERED's event selection."""
    return {int(v) for v in flag_values if int(v) & PILEUP_SATURATION_BITS}


# ------------------------------------------------------------------- reading


class EventChunk(NamedTuple):
    """One block of consecutive events, in file order.

    Timestamps are int64 picoseconds (the file stores uint64): signed so
    that time differences can be taken directly without wrap-around.
    Consumers rely on exactly these four field names.
    """
    channel: np.ndarray    # uint16
    timestamp: np.ndarray  # int64, ps
    energy: np.ndarray     # uint16, ADC channel
    flags: np.ndarray      # uint32


class ListModeFile:
    """Description of one SData file, built by open_listmode().

    Holds only numbers, not an open handle: the file is reopened by
    iter_events, so a description can be kept in a dialog for as long as
    needed without pinning the file.

    Attributes:
        path               Path of the file
        tree_name          'Data', 'Data_F' or 'Data_R'
        stream             'UNFILTERED', 'FILTERED' or 'RAW'
        n_entries          number of events
        events_per_channel {channel: count}, channels present only
        flag_counts        {flags value: count}, values present only
        granularity_ps     gcd of the non-zero timestamps in the first
                           granularity_sample events; None if all were 0
        is_sorted          True if timestamps never decrease over the file
    """

    def __init__(self, path, tree_name, n_entries, events_per_channel,
                 flag_counts, granularity_ps, is_sorted):
        self.path = Path(path)
        self.tree_name = tree_name
        self.stream = STREAMS[tree_name]
        self.n_entries = n_entries
        self.events_per_channel = events_per_channel
        self.flag_counts = flag_counts
        self.granularity_ps = granularity_ps
        self.is_sorted = is_sorted

    def __repr__(self):
        return (f"ListModeFile({str(self.path)!r}, tree={self.tree_name!r}, "
                f"entries={self.n_entries}, granularity_ps="
                f"{self.granularity_ps}, sorted={self.is_sorted})")


def _require_uproot(path):
    try:
        import uproot
    except ImportError as exc:  # pragma: no cover - exercised by the message
        raise ParseError(
            f"Reading list-mode ROOT files needs the 'uproot' package, "
            f"which is not installed: {path}") from exc
    return uproot


def _open_tree(path):
    """(uproot file handle, tree name) for a validated SData file.

    The caller closes the handle. Every failure is a ParseError naming the
    file and what was wrong, so the dialog can show it as is.
    """
    uproot = _require_uproot(path)
    path = Path(path)
    if not path.is_file():
        raise ParseError(f"File not found: {path}")
    try:
        handle = uproot.open(path)
    except Exception as exc:
        raise ParseError(f"Not a readable ROOT file ({exc}): {path}") from exc
    try:
        classes = {k.split(";")[0]: c
                   for k, c in handle.classnames(recursive=True).items()}
        trees = [k for k, c in classes.items() if c == "TTree"]
        found = [k for k in trees if k in STREAMS]
        if not found:
            if trees:
                raise ParseError(
                    f"No CoMPASS list-mode tree in {path}: it has TTree(s) "
                    f"{', '.join(sorted(trees))}, expected one named "
                    f"{', '.join(STREAMS)}")
            hists = [k for k, c in classes.items() if c.startswith("TH")]
            if hists:
                raise ParseError(
                    f"{path} holds only histograms ({len(hists)}), no "
                    f"list-mode TTree: open it as a spectrum or matrix "
                    f"instead")
            raise ParseError(
                f"No CoMPASS list-mode tree ({', '.join(STREAMS)}) in {path}")
        if len(found) > 1:
            raise ParseError(
                f"{path} holds several list-mode trees "
                f"({', '.join(sorted(found))}); expected exactly one")
        name = found[0]
        tree = handle[name]
        branches = {b.name: b for b in tree.branches}
        missing = [b for b in REQUIRED_BRANCHES if b not in branches]
        if missing:
            raise ParseError(
                f"List-mode tree '{name}' in {path} is missing branch(es) "
                f"{', '.join(missing)}")
        for bname, want in REQUIRED_BRANCHES.items():
            got = getattr(branches[bname].interpretation, "to_dtype", None)
            if got is None or np.dtype(got) != want:
                raise ParseError(
                    f"Branch '{bname}' of tree '{name}' in {path} has type "
                    f"{got if got is not None else branches[bname].interpretation}"
                    f", expected {want}")
    except BaseException:
        handle.close()
        raise
    return handle, name


def _to_chunk(arrays, path):
    ts = arrays["Timestamp"]
    if ts.size and ts.max() > _INT64_MAX:
        raise ParseError(
            f"Timestamp {int(ts.max())} ps is out of int64 range (corrupt "
            f"event?): {path}")
    return EventChunk(
        channel=arrays["Channel"].astype(np.uint16, copy=False),
        timestamp=ts.astype(np.int64),
        energy=arrays["Energy"].astype(np.uint16, copy=False),
        flags=arrays["Flags"].astype(np.uint32, copy=False),
    )


def _iterate(handle, tree_name, path, step_size):
    tree = handle[tree_name]
    if tree.num_entries == 0:
        return
    for arrays in tree.iterate(_EVENT_BRANCHES, step_size=step_size,
                               library="np"):
        yield _to_chunk(arrays, path)


def iter_events(path_or_file, step_size=DEFAULT_STEP_SIZE):
    """Yield EventChunks of `step_size` entries (the last may be shorter),
    in file order, from a path or a ListModeFile.

    The file is validated first, so a bad file fails here with the same
    ParseError open_listmode would give. Order is file order, which is
    time order only when the description says is_sorted.
    """
    path = (path_or_file.path if isinstance(path_or_file, ListModeFile)
            else Path(path_or_file))
    handle, tree_name = _open_tree(path)
    with handle:
        yield from _iterate(handle, tree_name, path, step_size)


def num_entries(path):
    """The number of entries (events, all channels) in an SData file's
    list-mode tree, read from the tree header without touching the
    events. Raises ParseError for anything open_listmode would refuse."""
    handle, tree_name = _open_tree(path)
    with handle:
        return int(handle[tree_name].num_entries)


def is_listmode_file(path):
    """Whether `path` is a readable CoMPASS list-mode (SData) file: a ROOT
    file holding exactly one valid Data / Data_F / Data_R tree. Never
    raises; for the main window to decide whether a .root file is a
    list-mode run to sort rather than histograms to open."""
    try:
        handle, _tree_name = _open_tree(path)
    except Exception:  # noqa: BLE001 -- any failure means "no"
        return False
    handle.close()
    return True


def open_listmode(path, step_size=DEFAULT_STEP_SIZE,
                  granularity_sample=DEFAULT_GRANULARITY_SAMPLE):
    """Validate an SData file and describe it (see ListModeFile).

    One chunked pass over Channel/Timestamp/Energy/Flags gives exact counts
    and a sortedness check that includes the step between chunks. The
    granularity is the gcd of the first `granularity_sample` non-zero
    timestamps: CoMPASS writes coarse 10 ns ticks unless the fine timestamp
    is on, and the Δt spectrum must be binned at that tick or every other
    bin comes out empty. A sample is enough -- one off-grid value already
    lowers the gcd -- and keeps the cost bounded.
    """
    path = Path(path)
    handle, tree_name = _open_tree(path)
    channel_counts = np.zeros(0, dtype=np.int64)
    flag_counts = {}
    is_sorted = True
    last_ts = None
    gcd = 0
    sampled = 0
    with handle:
        n_entries = int(handle[tree_name].num_entries)
        for chunk in _iterate(handle, tree_name, path, step_size):
            counts = np.bincount(chunk.channel)
            if counts.size > channel_counts.size:
                counts[:channel_counts.size] += channel_counts
                channel_counts = counts
            else:
                channel_counts[:counts.size] += counts
            values, n = np.unique(chunk.flags, return_counts=True)
            for v, k in zip(values.tolist(), n.tolist()):
                flag_counts[v] = flag_counts.get(v, 0) + k
            ts = chunk.timestamp
            if is_sorted and ts.size:
                if last_ts is not None and ts[0] < last_ts:
                    is_sorted = False
                elif np.any(ts[1:] < ts[:-1]):
                    is_sorted = False
                last_ts = ts[-1]
            if sampled < granularity_sample:
                take = ts[:granularity_sample - sampled]
                sampled += take.size
                nonzero = take[take != 0]
                if nonzero.size:
                    gcd = math.gcd(gcd, int(np.gcd.reduce(nonzero)))
    events_per_channel = {c: int(k) for c, k in enumerate(channel_counts)
                          if k}
    return ListModeFile(
        path, tree_name, n_entries, events_per_channel,
        dict(sorted(flag_counts.items())), gcd or None, is_sorted)


# ------------------------------------------------------------------ run info


class ChannelInfo(NamedTuple):
    """One channel's block of <run>_info.txt. Fields CoMPASS did not write
    are None. `counts` holds every '<name> counts = N' line by name
    ('Input', 'Pile up', 'Saturation', 'Output', ...)."""
    label: str                 # e.g. 'CH0@DT5780SCP_58363'
    board: str                 # e.g. 'DT5780SCP_58363'
    c0: float | None
    c1: float | None
    c2: float | None
    unit: str | None
    real_time_s: float | None
    live_time_s: float | None
    dead_time_percent: float | None
    input_counts: int | None
    output_counts: int | None
    counts: dict


class RunInfo(NamedTuple):
    run_id: str | None
    start: str | None          # as CoMPASS wrote it, not parsed
    stop: str | None
    channels: dict             # {channel number: ChannelInfo}
    path: Path                 # the info.txt that was read


_CH_HEADER = re.compile(r"^CH(\d+)@(\S+)\s*$")
_CALIB = re.compile(
    r"C0\s*=\s*([^;]+);\s*C1\s*=\s*([^;]+);\s*C2\s*=\s*([^;]+);"
    r"\s*unit\s*=\s*(\S+)")
_COUNTS = re.compile(r"^(.+?) counts\s*=\s*(\d+)")
_HMS = re.compile(r"^(\d+):(\d+):(\d+(?:\.\d*)?)$")


def _seconds(text):
    m = _HMS.match(text.strip())
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _number(text):
    try:
        return float(text.strip().rstrip("%").strip())
    except ValueError:
        return None


def _parse_info(text, path):
    head = {}
    blocks = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _CH_HEADER.match(raw.rstrip())
        if m and not raw[:1].isspace():
            current = {"label": line, "board": m.group(2), "counts": {}}
            blocks.setdefault(int(m.group(1)), current)
            continue
        if current is None:
            key, sep, value = line.partition("=")
            if sep:
                head[key.strip()] = value.strip()
            continue
        if line.startswith("Energy calibration"):
            cm = _CALIB.search(line)
            if cm:
                current["c0"], current["c1"], current["c2"] = (
                    _number(cm.group(i)) for i in (1, 2, 3))
                current["unit"] = cm.group(4)
            continue
        cm = _COUNTS.match(line)
        if cm:
            current["counts"][cm.group(1).strip()] = int(cm.group(2))
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if key == "Real time":
            current["real_time_s"] = _seconds(value)
        elif key == "Live time":
            current["live_time_s"] = _seconds(value)
        elif key == "Dead time":
            current["dead_time_percent"] = _number(value)
    if "Run ID" not in head and not blocks:
        return None
    channels = {}
    for ch, b in blocks.items():
        channels[ch] = ChannelInfo(
            label=b["label"], board=b["board"],
            c0=b.get("c0"), c1=b.get("c1"), c2=b.get("c2"),
            unit=b.get("unit"),
            real_time_s=b.get("real_time_s"), live_time_s=b.get("live_time_s"),
            dead_time_percent=b.get("dead_time_percent"),
            input_counts=b["counts"].get("Input"),
            output_counts=b["counts"].get("Output"),
            counts=b["counts"])
    return RunInfo(head.get("Run ID"), head.get("Start time"),
                   head.get("Stop time"), channels, Path(path))


def _run_id_from_sdata(name):
    m = re.match(r"^SData[RF]?_(.+)\.root$", name)
    return m.group(1) if m else None


def _find_info(path):
    """The info.txt that belongs to `path`, or None.

    CoMPASS writes <RunID>_info.txt in the run folder, one level above the
    RAW/UNFILTERED/FILTERED stream folders that hold the SData files. A
    path to the info file itself, a stream folder or the run folder is
    accepted too. With several candidates, the one named after the run ID
    in the SData file name wins.
    """
    if path.is_file() and path.name.endswith("_info.txt"):
        return path
    if path.is_dir():
        folders = [path, path.parent]
        wanted = None
    else:
        folders = [path.parent.parent, path.parent]
        wanted = _run_id_from_sdata(path.name)
    for folder in folders:
        try:
            candidates = sorted(folder.glob("*_info.txt"))
        except OSError:
            continue
        if wanted:
            for c in candidates:
                if c.name == f"{wanted}_info.txt":
                    return c
        if candidates:
            return candidates[0]
    return None


def read_run_info(path):
    """Parse the run's <RunID>_info.txt, or return None.

    Optional by design: CoMPASS writes it only at the end of a run, and a
    copied SData file often travels without it, so a missing, unreadable or
    unrecognisable file gives None, never an exception. The file is decoded
    as latin-1 because CoMPASS writes the '±' in its rate lines as a single
    non-UTF-8 byte (0xB1).
    """
    try:
        info_path = _find_info(Path(path))
        if info_path is None:
            return None
        text = info_path.read_bytes().decode("latin-1")
        return _parse_info(text, info_path)
    except Exception:
        return None
