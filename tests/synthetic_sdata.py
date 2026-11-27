"""Writes small CoMPASS-style SData ROOT files for the list-mode tests.

Generated with uproot's writer rather than committed as binaries, like the
ROOT histogram fixtures in test_root_io: the data stays readable as code,
and each test states exactly the events it is about. The real run these
files imitate is untracked (see the v7 plan, "Fixtures").

The branch layout copies what CoMPASS writes: six branches, in this order,
with these element types. CoMPASS stores them big-endian; uproot converts
on write and read, so the byte order is not something a test can or needs
to control.
"""

import numpy as np

# Branch name -> numpy dtype, exactly as CoMPASS 2.x writes them.
SDATA_BRANCHES = {
    "Channel": np.uint16,
    "Timestamp": np.uint64,
    "Board": np.uint16,
    "Energy": np.uint16,
    "CalibEnergy": np.float64,
    "Flags": np.uint32,
}


def write_sdata(path, channel, timestamp, energy, flags=None, *,
                tree="Data", drop=(), dtypes=None, chunks=1,
                extra_objects=None):
    """Write one SData file and return its path.

    `drop` removes branches (to test the missing-branch error), `dtypes`
    overrides element types (to test the wrong-type error), and `chunks`
    splits the events over several `extend` calls, so the file has several
    baskets the way a long CoMPASS run does. `extra_objects` adds other
    top-level objects (e.g. a histogram) beside the tree.
    """
    import uproot

    channel = np.asarray(channel)
    n = len(channel)
    if flags is None:
        flags = np.zeros(n)
    columns = {
        "Channel": channel,
        "Timestamp": np.asarray(timestamp),
        "Board": np.zeros(n),
        "Energy": np.asarray(energy),
        "CalibEnergy": 1.5 + 0.5 * np.asarray(energy, dtype=np.float64),
        "Flags": np.asarray(flags),
    }
    types = dict(SDATA_BRANCHES)
    if dtypes:
        types.update(dtypes)
    columns = {name: columns[name].astype(types[name])
               for name in SDATA_BRANCHES if name not in drop}
    with uproot.recreate(path) as handle:
        if tree is not None:
            branch_types = {name: col.dtype for name, col in columns.items()}
            handle.mktree(tree, branch_types, title="CoMPASS events TTree")
            bounds = np.linspace(0, n, chunks + 1).astype(int)
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                if hi > lo:
                    handle[tree].extend(
                        {name: col[lo:hi] for name, col in columns.items()})
        for name, obj in (extra_objects or {}).items():
            handle[name] = obj
    return path


def simple_events(n=1000, seed=1, tick_ps=10000):
    """Two channels of time-sorted events on a `tick_ps` grid.

    Returns (channel, timestamp, energy, flags) as numpy arrays. Timestamps
    start well above 0 and strictly increase, so the gcd is exactly the
    tick and sortedness is True.
    """
    rng = np.random.default_rng(seed)
    channel = rng.integers(0, 2, n).astype(np.uint16)
    steps = rng.integers(1, 50, n).astype(np.uint64)
    timestamp = (np.cumsum(steps) + 7) * np.uint64(tick_ps)
    energy = rng.integers(0, 16384, n).astype(np.uint16)
    flags = np.zeros(n, dtype=np.uint32)
    flags[::97] = 0x800
    flags[5::211] = 0x8000
    return channel, timestamp, energy, flags
