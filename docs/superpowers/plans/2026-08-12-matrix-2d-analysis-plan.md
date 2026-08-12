# 2D Matrix Analysis (mtx/gate/cut) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TV-style 2D matrix analysis (`.mtx` files, X/Y projections, gate/background marking, background-subtracted cut spectra) plus an on-demand 2D heatmap viewer, per the approved design spec.

**Architecture:** `mtx_io.py` reads the `lc`-format binary matrix (level 0 only); `matrix_cut.py` holds the pure projection/cut math; `matrix_panel.py` is a new top-level window for the marking/cut workflow, reusing `fit_mode.py`'s marker-placement conventions; `matrix_heatmap.py` is a separate on-demand visualization window; `main_window.py` gains an "Open Matrix..." action and the focus-exclusivity behavior between the two windows. A completed cut becomes an ordinary `LoadedSpectrum` via the existing `_add_combined_spectrum` mechanism.

**Tech Stack:** Python, numpy, PySide6, matplotlib. No new dependency.

**Full design rationale:** `docs/superpowers/specs/2026-08-12-matrix-2d-analysis-design.md`.

---

### Task 1: `mtx_io.py` — the `lc`-format matrix reader

**Files:**
- Create: `mtx_io.py`
- Create: `tests/test_mtx_io.py`
- Fixtures already in place: `tests/fixtures/gpff.mtx`, `tests/fixtures/gg.mtx` (real 8192x8192 matrices, copied from the repo root this session)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mtx_io.py`:

```python
import os
import struct

import numpy as np
import pytest

from histogram_io import ParseError
from mtx_io import MAGIC_LC, load_mtx

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# Oracle values obtained by compiling libmfile-1.0.7 from source (under
# WSL, bypassing its CRLF-broken autotools configure by invoking gcc
# directly on the .c files) and calling its real mgetint() against the
# actual fixture files -- not hand-computed. checksum = sum(v*(i+1) for
# i, v in enumerate(row)), a cheap way to catch a decode error anywhere
# in a full 8192-value row, not just its first few values.
_ORACLE_ROWS = {
    ("gpff.mtx", 0): {"sum": 301, "min": -5, "max": 6, "checksum": 278695},
    ("gpff.mtx", 1): {"sum": 57, "min": -2, "max": 3, "checksum": 17951},
    ("gpff.mtx", 2): {"sum": 181, "min": -3, "max": 106, "checksum": 65609},
    ("gpff.mtx", 100): {"sum": 74, "min": -2, "max": 10, "checksum": 34558},
    ("gpff.mtx", 500): {"sum": 1045, "min": -4, "max": 8, "checksum": 948725},
    ("gpff.mtx", 1000): {"sum": 9140, "min": -9, "max": 52, "checksum": 6801414},
    ("gpff.mtx", 4096): {"sum": 3446, "min": -16, "max": 43, "checksum": 2809152},
    ("gpff.mtx", 8000): {"sum": 153, "min": -2, "max": 3, "checksum": 101229},
    ("gpff.mtx", 8191): {"sum": 108, "min": -2, "max": 2, "checksum": 93110},
    ("gg.mtx", 0): {"sum": 331252, "min": -6, "max": 206242, "checksum": 83310174},
    ("gg.mtx", 1): {"sum": 30165, "min": -3, "max": 9669, "checksum": 14434711},
    ("gg.mtx", 2): {"sum": 3964, "min": -3, "max": 955, "checksum": 2078348},
    ("gg.mtx", 100): {"sum": 1059634, "min": -1, "max": 2788, "checksum": 813141814},
    ("gg.mtx", 500): {"sum": 402288, "min": -2, "max": 1121, "checksum": 315223325},
    ("gg.mtx", 1000): {"sum": 189809, "min": -2, "max": 403, "checksum": 144945290},
    ("gg.mtx", 4096): {"sum": 14828, "min": -3, "max": 145, "checksum": 10284507},
    ("gg.mtx", 8000): {"sum": 0, "min": 0, "max": 0, "checksum": 0},
    ("gg.mtx", 8191): {"sum": 0, "min": 0, "max": 0, "checksum": 0},
}


def _row_checksum(row):
    return int(sum(int(v) * (i + 1) for i, v in enumerate(row)))


@pytest.mark.parametrize("filename", ["gpff.mtx", "gg.mtx"])
def test_load_mtx_real_file_shape_and_dtype(filename):
    data = load_mtx(os.path.join(FIXTURES, filename))
    assert data.shape == (8192, 8192)
    assert data.dtype == np.int64


@pytest.mark.parametrize("filename,row", list(_ORACLE_ROWS.keys()))
def test_load_mtx_real_file_row_matches_oracle(filename, row):
    data = load_mtx(os.path.join(FIXTURES, filename))
    expected = _ORACLE_ROWS[(filename, row)]
    actual_row = data[row, :]
    assert int(actual_row.sum()) == expected["sum"]
    assert int(actual_row.min()) == expected["min"]
    assert int(actual_row.max()) == expected["max"]
    assert _row_checksum(actual_row) == expected["checksum"]


def test_load_mtx_real_file_spot_check_values():
    gpff = load_mtx(os.path.join(FIXTURES, "gpff.mtx"))
    assert list(gpff[0, :20]) == [
        6, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 1,
    ]
    assert list(gpff[1, :20]) == [
        3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]

    gg = load_mtx(os.path.join(FIXTURES, "gg.mtx"))
    assert list(gg[0, :20]) == [
        206242, 9669, 955, 234, 137, 82, 78, 64, 49, 39, 36, 49, 52, 57, 61, 53, 65, 76, 116, 132,
    ]
    assert list(gg[1, :20]) == [
        9669, 690, 107, 21, 14, 8, 3, 6, 5, 5, 5, 4, 4, 11, 18, 15, 14, 11, 9, 7,
    ]


def test_load_mtx_real_file_negative_values_preserved():
    # gpff.mtx level 0 is real random-coincidence-subtracted data,
    # confirmed by the user -- negative values must survive unclamped.
    gpff = load_mtx(os.path.join(FIXTURES, "gpff.mtx"))
    assert (gpff < 0).any()


def test_load_mtx_symmetric_file_is_actually_symmetric():
    gg = load_mtx(os.path.join(FIXTURES, "gg.mtx"))
    assert gg[10, 2000] == gg[2000, 10] == 1


def _build_lc_header(levels, lines, columns, poslentablepos, version=2):
    return struct.pack(
        "<11I", MAGIC_LC, version, levels, lines, columns, poslentablepos, 0, 0, 0, 0, 0
    )


def _build_minimal_lc_file(tmp_path, rows_bytes, columns, name="test.mtx"):
    """rows_bytes: list of compressed-row byte strings, one per row,
    level 0 (levels=1). An empty bytes object for a row means "never
    written" -- table entry (0, 0), matching the real format's
    convention for a row libmfile's readline() short-circuits to zero
    without decoding."""
    lines = len(rows_bytes)
    header_size = 44
    poslentablepos = header_size
    header = _build_lc_header(1, lines, columns, poslentablepos)

    table = b""
    data = b""
    pos = header_size + lines * 8
    for row in rows_bytes:
        table += struct.pack("<II", pos if row else 0, len(row))
        data += row
        pos += len(row)

    path = tmp_path / name
    path.write_bytes(header + table + data)
    return str(path)


def test_load_mtx_single_fallback_values(tmp_path):
    # Hand-encoded against lc2_uncompress's traced algorithm: last=0;
    # tag 0x8A (n=10, plain, <=59) -> zigzag_decode(10)=5, last=5;
    # tag 0x8D (n=13) -> zigzag_decode(13)=-7, last=-2;
    # tag 0x84 (n=4) -> zigzag_decode(4)=2, last=0.
    row0 = bytes([0x8A, 0x8D, 0x84])
    path = _build_minimal_lc_file(tmp_path, [row0], columns=3)
    data = load_mtx(path)
    assert data.shape == (1, 3)
    assert list(data[0]) == [5, -2, 0]


def test_load_mtx_zero_length_row_is_all_zero(tmp_path):
    row0 = bytes([0x8A, 0x8D, 0x84])
    path = _build_minimal_lc_file(tmp_path, [row0, b""], columns=3)
    data = load_mtx(path)
    assert list(data[1]) == [0, 0, 0]


def test_load_mtx_bad_magic_raises(tmp_path):
    path = tmp_path / "bad.mtx"
    path.write_bytes(struct.pack("<11I", 0xDEADBEEF, 2, 1, 1, 1, 44, 0, 0, 0, 0, 0))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_truncated_header_raises(tmp_path):
    path = tmp_path / "short.mtx"
    path.write_bytes(b"\x00" * 10)
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_unsupported_version_raises(tmp_path):
    path = tmp_path / "v1.mtx"
    path.write_bytes(struct.pack("<11I", MAGIC_LC, 1, 1, 1, 1, 44, 0, 0, 0, 0, 0))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_zero_levels_raises(tmp_path):
    path = tmp_path / "zerolev.mtx"
    path.write_bytes(struct.pack("<11I", MAGIC_LC, 2, 0, 1, 1, 44, 0, 0, 0, 0, 0))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_truncated_row_table_raises(tmp_path):
    # Header claims 5 lines (needs a 40-byte table) but the file ends
    # right after the header.
    path = tmp_path / "notable.mtx"
    path.write_bytes(_build_lc_header(1, 5, 3, 44))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_truncated_row_data_raises(tmp_path):
    # Table claims row 0 is 3 bytes long at offset 52, but only 1 byte
    # actually follows the table in the file.
    header = _build_lc_header(1, 1, 3, 44)
    table = struct.pack("<II", 52, 3)
    path = tmp_path / "trunc.mtx"
    path.write_bytes(header + table + bytes([0x8A]))
    with pytest.raises(ParseError):
        load_mtx(str(path))


def test_load_mtx_same_run_and_packed_tags(tmp_path):
    # Exercises the 2-pack and same-run branches together (the
    # real-file tests above already exercise every branch on real
    # data; this is a small from-first-principles cross-check). Both
    # 2-pack values are computed against the SAME original `last`, not
    # chained -- this is the single easiest part of the format to get
    # wrong, per the design research, so it's spelled out in full:
    #   0x69 (2-pack): a=1->decode(1)=-1 -> value1 = last(0)+(-1) = -1
    #                  (last is NOT updated by this first value);
    #                  b=5->decode(5)=-3 -> last = 0+(-3) = -3,
    #                  value2 = -3 (last IS updated by the second value)
    #   0xC6 (same-run, n=6): diff=6&1=0, same=(6>>1)+3=6 ->
    #                  value = last(-3)+diff(0) = -3, then 6 more
    #                  copies of -3 (last stays -3 throughout,
    #                  unchanged by a same-run tag) -- 7 values total
    # 2 (2-pack) + 7 (same-run) = 9 values.
    row0 = bytes([0x69, 0xC6])
    path = _build_minimal_lc_file(tmp_path, [row0], columns=9)
    data = load_mtx(path)
    assert list(data[0]) == [-1, -3, -3, -3, -3, -3, -3, -3, -3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mtx_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mtx_io'`.

- [ ] **Step 3: Create `mtx_io.py`**

```python
import struct

import numpy as np

from histogram_io import ParseError

MAGIC_LC = 0x80FFFF10
_HEADER_FORMAT = "<11I"
_HEADER_SIZE = 44


def _zigzag_decode(n):
    return -((n >> 1) + 1) if (n & 1) else (n >> 1)


def _decode_row(data, num_values, path):
    """Decodes one lc-format v2 compressed row into `num_values`
    integers. A faithful port of lc2_uncompress
    (libmfile-1.0.7/src/lc_c2.c:134-200), including its two least
    obvious behaviors: a 3-pack/2-pack tag's deltas are all computed
    against the SAME pre-tag `last` (not chained value-to-value), and
    a same-run tag's repeated values equal the pre-run `last`
    unchanged -- `last` is not updated by a same-run tag at all, only
    by the other three tag kinds. Verified byte-for-byte against real
    compressed rows from both fixture files during planning."""
    values = []
    last = 0
    pos = 0
    nleft = num_values
    try:
        while nleft > 0:
            t = data[pos]
            pos += 1

            if t & 0x80:
                n = t & 0x3F
                if n > 59:
                    bytes_extra = n - 59
                    n = 59
                    for i in range(bytes_extra):
                        b = data[pos]
                        pos += 1
                        n += (b + 1) << (i * 8)

                if t & 0x40:
                    diff = n & 1
                    same = (n >> 1) + 3
                    values.append(last + diff)
                    nleft -= same
                    if nleft <= 0:
                        raise ParseError(f"lc matrix file: same-run tag overruns row: {path}")
                    values.extend([last] * same)
                else:
                    last = last + _zigzag_decode(n)
                    values.append(last)
                nleft -= 1

            elif t & 0x40:
                nleft -= 2
                if nleft < 0:
                    raise ParseError(f"lc matrix file: 2-value pack overruns row: {path}")
                a = t & 0x7
                b = (t >> 3) & 0x7
                values.append(last + _zigzag_decode(a))
                last = last + _zigzag_decode(b)
                values.append(last)

            else:
                nleft -= 3
                if nleft < 0:
                    raise ParseError(f"lc matrix file: 3-value pack overruns row: {path}")
                a = t & 0x3
                b = (t >> 2) & 0x3
                c = (t >> 4) & 0x3
                values.append(last + _zigzag_decode(a))
                values.append(last + _zigzag_decode(b))
                last = last + _zigzag_decode(c)
                values.append(last)
    except IndexError as exc:
        raise ParseError(f"lc matrix file: row data ends mid-tag: {path}") from exc

    return values


def load_mtx(path):
    """Reads level 0 of an lc-format (line-compressed) TV matrix file,
    returning a (lines, columns) int64 array. Read-only -- no writer,
    matching TV's own architecture. Negative values (e.g. from
    random-coincidence subtraction) are preserved as-is throughout.
    Always reads level 0 and never exposes level selection -- matching
    TV's own matrix-open command, which has no level argument at all."""
    try:
        with open(path, "rb") as f:
            header_bytes = f.read(_HEADER_SIZE)
            if len(header_bytes) < _HEADER_SIZE:
                raise ParseError(f"File too small to be an lc-format matrix: {path}")
            (
                magic, version, levels, lines, columns, poslentablepos,
                _freepos, _freelistpos, _used, _free, _status,
            ) = struct.unpack(_HEADER_FORMAT, header_bytes)

            if magic != MAGIC_LC:
                raise ParseError(f"Not an lc-format matrix file (bad magic): {path}")
            if version != 2:
                raise ParseError(
                    f"Unsupported lc format version {version} (only version 2 is supported): {path}"
                )
            if levels < 1:
                raise ParseError(f"lc matrix file declares zero levels: {path}")

            # Row-table entries are {u_int pos, u_int len}, one per
            # (level, line), ordered level-major then row -- level 0's
            # rows come first, so only the first `lines` entries matter.
            f.seek(poslentablepos)
            table_bytes = f.read(lines * 8)
            if len(table_bytes) < lines * 8:
                raise ParseError(f"lc matrix file's row table is truncated: {path}")
            row_table = struct.unpack(f"<{lines * 2}I", table_bytes)

            data = np.zeros((lines, columns), dtype=np.int64)
            for row in range(lines):
                row_pos = row_table[row * 2]
                row_len = row_table[row * 2 + 1]
                if row_len == 0:
                    continue  # never written -- treated as all-zero
                f.seek(row_pos)
                row_bytes = f.read(row_len)
                if len(row_bytes) < row_len:
                    raise ParseError(f"lc matrix file row {row} data is truncated: {path}")
                values = _decode_row(row_bytes, columns, path)
                try:
                    data[row, :] = values
                except OverflowError as exc:
                    raise ParseError(
                        f"lc matrix file row {row} contains an out-of-range value: {path}"
                    ) from exc
    except OSError as exc:
        raise ParseError(f"Could not read {path}: {exc}") from exc

    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mtx_io.py -v`
Expected: PASS, all tests (19 real-file row tests + shape/dtype + spot-check + negative-values + symmetry + 8 synthetic tests).

Run: `pytest -q` (full suite)
Expected: PASS (aside from the pre-existing, unrelated `test_theme.py::test_zoom_icons_and_builtin_save_icon_match_color_in_both_themes` failure — if you see exactly that one and nothing else, it's not yours to fix).

- [ ] **Step 5: Measure real-file load time**

Run a quick timing check (not a formal test, just a sanity measurement — this app has no existing convention for a slow-operation progress indicator, and 8192x8192 decode is a genuinely large amount of Python-level work):

```bash
python -c "import time; from mtx_io import load_mtx; t=time.time(); load_mtx('tests/fixtures/gg.mtx'); print(time.time()-t)"
```

If this takes more than a few seconds, note it in your report for Task 6 (main_window integration) — opening a matrix from the UI may need a "loading..." status message or busy cursor rather than appearing to hang. Don't optimize the decoder itself unless the plan is later revised to ask for it.

- [ ] **Step 6: Commit**

```bash
git add mtx_io.py tests/test_mtx_io.py tests/fixtures/gpff.mtx tests/fixtures/gg.mtx
git commit -m "feat: add lc-format matrix reader (mtx_io.py)"
```

---

### Task 2: `matrix_cut.py` — projection and cut computation

**Files:**
- Create: `matrix_cut.py`
- Create: `tests/test_matrix_cut.py`

**Depends on Task 1** only for the array shape/dtype convention (matrix rows=Y, columns=X) — otherwise independent, pure functions, no file I/O.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matrix_cut.py`:

```python
import numpy as np
import pytest

from matrix_cut import compute_cut, compute_projection


def _small_matrix():
    # 4 rows (Y: 0..3) x 5 columns (X: 0..4), simple deterministic
    # values so projections/cuts are easy to hand-verify.
    return np.array(
        [
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            [-1, -2, -3, -4, -5],
            [100, 100, 100, 100, 100],
        ],
        dtype=np.int64,
    )


def test_compute_projection_x_sums_over_rows():
    matrix = _small_matrix()
    proj = compute_projection(matrix, "x")
    assert list(proj) == [110, 120, 130, 140, 150]  # column sums


def test_compute_projection_y_sums_over_columns():
    matrix = _small_matrix()
    proj = compute_projection(matrix, "y")
    assert list(proj) == [15, 150, -15, 500]  # row sums


def test_compute_projection_invalid_axis_raises():
    matrix = _small_matrix()
    with pytest.raises(ValueError):
        compute_projection(matrix, "z")


def test_compute_cut_gating_on_x_produces_y_indexed_result():
    # Gate on columns 1-2 (X range), no background -- net = raw sum of
    # those two columns, indexed by row (Y).
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(1, 2), bg_regions=[])
    assert list(result) == [5, 50, -5, 200]  # row-wise sum of columns 1,2


def test_compute_cut_gating_on_y_produces_x_indexed_result():
    # Gate on rows 0-1 (Y range), no background -- net = raw sum of
    # those two rows, indexed by column (X).
    matrix = _small_matrix()
    result = compute_cut(matrix, "y", cut_region=(0, 1), bg_regions=[])
    assert list(result) == [11, 22, 33, 44, 55]  # column-wise sum of rows 0,1


def test_compute_cut_zero_background_regions_means_no_subtraction():
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[])
    assert list(result) == [1, 10, -1, 100]  # just column 0, unsubtracted


def test_compute_cut_applies_gate_width_weighted_subtraction():
    # cut_region (0,1) on axis 'x' -> width 2, sums columns 0+1 per row.
    # bg_regions [(3,4)] on axis 'x' -> width 2, sums columns 3+4 per row.
    # Equal widths (2/2=1.0) -> net = pos - bg exactly.
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0, 1), bg_regions=[(3, 4)])
    # pos (cols 0+1 per row): [3, 30, -3, 200]
    # bg  (cols 3+4 per row): [9, 90, -9, 200]
    # net = pos - (2/2)*bg = pos - bg
    assert list(result) == [-6, -60, 6, 0]


def test_compute_cut_unequal_width_background_scales_correctly():
    # cut_region (0,0) width=1; bg_regions [(2,4)] width=3.
    # pos (col 0): [1, 10, -1, 100]
    # bg (cols 2+3+4 summed per row): [12, 120, -12, 300]
    # net = pos - (1/3)*bg
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(2, 4)])
    expected = [1 - 12 / 3, 10 - 120 / 3, -1 - (-12) / 3, 100 - 300 / 3]
    assert result == pytest.approx(expected)


def test_compute_cut_multiple_background_regions_are_pooled():
    # Two separate bg regions, total width 2 (one col each) -- summed
    # together (not weighted per-region) before the shared ratio is
    # applied, matching TV's own gate-width weighting.
    matrix = _small_matrix()
    result_combined = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 3), (4, 4)])
    result_single = compute_cut(matrix, "x", cut_region=(0, 0), bg_regions=[(3, 4)])
    assert result_combined == pytest.approx(result_single)


def test_compute_cut_preserves_negative_results_unclamped():
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0, 1), bg_regions=[(3, 4)])
    assert any(v < 0 for v in result)  # confirmed by the prior test's exact values


def test_compute_cut_rounds_half_away_from_zero_like_tv():
    # Region bounds land on a .5 boundary -- TV's NINT rounds
    # half-away-from-zero (0.5 -> 1, -0.5 -> -1), not Python's default
    # round-half-to-even. Using (0.5, 1.5) on a 5-wide axis should
    # select columns 1 and 2 (round(0.5)->1, round(1.5)->2 per NINT).
    matrix = _small_matrix()
    result = compute_cut(matrix, "x", cut_region=(0.5, 1.5), bg_regions=[])
    assert list(result) == [5, 50, -5, 200]  # same as columns 1,2 exactly
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_cut.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'matrix_cut'`.

- [ ] **Step 3: Create `matrix_cut.py`**

```python
import numpy as np


def _nint(x):
    """Round-half-away-from-zero, matching TV's own NINT macro
    (lib/tv/vsTypes.h) -- Python's built-in round() uses
    round-half-to-even instead, which would disagree with TV exactly
    on a .5 boundary."""
    return int(x + 0.5) if x >= 0 else int(x - 0.5)


def _region_bounds(matrix, axis, region):
    """Returns (lo, hi) integer indices (inclusive), clamped to the
    matrix's extent along `axis`."""
    lo, hi = region
    size = matrix.shape[0] if axis == "y" else matrix.shape[1]
    lo_idx = max(0, _nint(lo))
    hi_idx = min(size - 1, _nint(hi))
    return lo_idx, hi_idx


def _region_width(matrix, axis, region):
    lo_idx, hi_idx = _region_bounds(matrix, axis, region)
    return hi_idx - lo_idx + 1


def _region_sum(matrix, axis, region):
    """Raw (unweighted) sum over `region`'s channel range on `axis`,
    returned as a 1D array indexed by the COMPLEMENTARY axis."""
    lo_idx, hi_idx = _region_bounds(matrix, axis, region)
    if axis == "y":
        return matrix[lo_idx : hi_idx + 1, :].sum(axis=0)
    else:
        return matrix[:, lo_idx : hi_idx + 1].sum(axis=1)


def compute_projection(matrix, axis):
    """axis='x' -> matrix.sum(axis=0), a spectrum indexed by X-channel
    (column). axis='y' -> matrix.sum(axis=1), a spectrum indexed by
    Y-channel (row)."""
    if axis == "x":
        return matrix.sum(axis=0)
    elif axis == "y":
        return matrix.sum(axis=1)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def compute_cut(matrix, axis, cut_region, bg_regions):
    """`axis` is which projection was marked -- the axis being gated
    ON, in that projection's own channel units. Implements TV's
    gate-width-weighted background subtraction:
        net[ch] = pos[ch] - (pos_width/bg_width) * bg[ch]
    where pos/bg are raw sums over the marked row-range (axis='y') or
    column-range (axis='x') of `matrix`. Returns a 1D array along the
    COMPLEMENTARY axis (axis='x' input -> Y-indexed output, and vice
    versa). Zero bg_regions => net = pos (no subtraction, not an
    error). Never clamps negative results."""
    pos = _region_sum(matrix, axis, cut_region)

    if not bg_regions:
        return pos

    pos_width = _region_width(matrix, axis, cut_region)
    bg = np.zeros_like(pos)
    bg_width = 0
    for region in bg_regions:
        bg = bg + _region_sum(matrix, axis, region)
        bg_width += _region_width(matrix, axis, region)

    return pos - (pos_width / bg_width) * bg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_cut.py -v`
Expected: PASS, all 11 tests.

Run: `pytest -q` (full suite)
Expected: PASS (aside from the known pre-existing `test_theme.py` failure).

- [ ] **Step 5: Commit**

```bash
git add matrix_cut.py tests/test_matrix_cut.py
git commit -m "feat: add matrix projection and gate-width-weighted cut computation"
```

---

### Task 3: `matrix_panel.py` — window skeleton and projection display

**Files:**
- Create: `matrix_panel.py`
- Create: `tests/test_matrix_panel.py`

**Depends on Tasks 1-2.**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matrix_panel.py`:

```python
import os

from main_window import MainWindow
from matrix_panel import MatrixPanel

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_matrix_panel_loads_matrix_and_computes_both_projections(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.matrix.shape == (8192, 8192)
    assert len(panel.projections["x"]) == 8192
    assert len(panel.projections["y"]) == 8192


def test_matrix_panel_defaults_to_x_projection(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.working_axis == "x"


def test_matrix_panel_switching_axis_updates_working_axis(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    panel.axis_selector.setCurrentIndex(1)

    assert panel.working_axis == "y"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'matrix_panel'`.

- [ ] **Step 3: Create `matrix_panel.py`**

```python
import os

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matrix_cut import compute_projection
from mtx_io import load_mtx
from theme import style_axes


class MatrixPanel(QMainWindow):
    """A separate top-level window for TV-style matrix gate/cut
    analysis. Loads a matrix and computes both its X and Y
    projections up front (this app has the whole matrix in memory,
    unlike TV which required opening a second, transposed matrix file
    for the other axis). The user picks which projection to work on,
    marks a cut region and background regions on it (Task 4), and
    activates the cut to hand a resulting spectrum off to the main
    window. Coexists with MainWindow -- activating one disables, but
    does not close, the other (wired in Task 6)."""

    def __init__(self, main_window, path):
        super().__init__()
        self.main_window = main_window
        self.path = path
        self.matrix = load_mtx(path)
        self.projections = {
            "x": compute_projection(self.matrix, "x"),
            "y": compute_projection(self.matrix, "y"),
        }
        self.working_axis = "x"

        self.setWindowTitle(f"Matrix -- {os.path.basename(path)}")
        self.resize(800, 500)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)

        self.axis_selector = QComboBox()
        self.axis_selector.addItem("X projection", "x")
        self.axis_selector.addItem("Y projection", "y")
        self.axis_selector.currentIndexChanged.connect(self._on_axis_changed)

        self.heatmap_button = QPushButton("Show Heatmap...")

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Working on:"))
        top_bar.addWidget(self.axis_selector)
        top_bar.addStretch()
        top_bar.addWidget(self.heatmap_button)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(top_bar)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        self._plot_projection()

    def _on_axis_changed(self, index):
        self.working_axis = self.axis_selector.itemData(index)
        self._plot_projection()

    def _plot_projection(self):
        self.axes.clear()
        style_axes(self.axes, self.main_window._theme)
        data = self.projections[self.working_axis]
        self.axes.plot(range(len(data)), data, linewidth=0.8)
        self.axes.set_xlabel(f"{self.working_axis.upper()} channel")
        self.axes.set_ylabel("Counts")
        self.canvas.draw()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, all 3 tests.

Run: `pytest -q` (full suite)
Expected: PASS (aside from the known pre-existing `test_theme.py` failure).

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "feat: add matrix panel window with projection display"
```

---

### Task 4: `matrix_panel.py` — marker placement, cut activation, and handoff

**Files:**
- Modify: `matrix_panel.py`
- Modify: `tests/test_matrix_panel.py`

**Depends on Task 3.**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_matrix_panel.py`:

```python
from matplotlib.backend_bases import MouseEvent


def _click(panel, xdata, ydata=10.0):
    ax = panel.axes
    px, py = ax.transData.transform((xdata, ydata))
    event = MouseEvent("button_press_event", panel.canvas, px, py, button=1)
    panel.canvas.callbacks.process("button_press_event", event)


def _held_key_click(panel, key, xdata, ydata=10.0):
    panel.cut_controller._held_key = key
    _click(panel, xdata, ydata)
    panel.cut_controller._held_key = None


def test_matrix_panel_marking_cut_region_with_two_clicks(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "cut", 100.0)
    _held_key_click(panel, "cut", 200.0)

    assert panel.cut_controller.state.cut_region == (100.0, 200.0)


def test_matrix_panel_marking_background_region_with_two_clicks(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "bg", 300.0)
    _held_key_click(panel, "bg", 350.0)

    assert panel.cut_controller.state.bg_regions == [(300.0, 350.0)]


def test_matrix_panel_multiple_background_regions_allowed(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "bg", 300.0)
    _held_key_click(panel, "bg", 350.0)
    _held_key_click(panel, "bg", 500.0)
    _held_key_click(panel, "bg", 550.0)
    _held_key_click(panel, "bg", 700.0)
    _held_key_click(panel, "bg", 750.0)

    assert len(panel.cut_controller.state.bg_regions) == 3


def test_matrix_panel_activate_cut_disabled_without_cut_region(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    assert panel.activate_cut_button.isEnabled() is False


def test_matrix_panel_activate_cut_enabled_once_cut_region_marked(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "cut", 100.0)
    _held_key_click(panel, "cut", 200.0)

    assert panel.activate_cut_button.isEnabled() is True


def test_matrix_panel_activate_cut_adds_spectrum_to_main_window(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "cut", 100.0)
    _held_key_click(panel, "cut", 300.0)
    before = len(main_window.spectra)

    panel._activate_cut()

    assert len(main_window.spectra) == before + 1
    added = main_window.spectra[-1]
    assert added.active is True
    assert len(added.data) == 8192  # Y-indexed result (gated on X projection)


def test_matrix_panel_activate_cut_result_matches_direct_computation(qapp):
    from matrix_cut import compute_cut

    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    _held_key_click(panel, "cut", 100.0)
    _held_key_click(panel, "cut", 300.0)
    _held_key_click(panel, "bg", 400.0)
    _held_key_click(panel, "bg", 450.0)
    panel._activate_cut()

    expected = compute_cut(panel.matrix, "x", (100.0, 300.0), [(400.0, 450.0)])
    added = main_window.spectra[-1]
    assert added.data == pytest.approx(expected)
```

Add `import pytest` to the top of `tests/test_matrix_panel.py`'s imports (needed for `pytest.approx`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: FAIL — `AttributeError: 'MatrixPanel' object has no attribute 'cut_controller'` (and similar).

- [ ] **Step 3: Update `matrix_panel.py`**

Add the import, alongside the existing ones:
```python
from PySide6.QtCore import QEvent, Qt
```

Add near the top of the file, after the imports:
```python
CUT_REGION_COLOR = "tab:red"
CUT_REGION_ALPHA = 0.25
BG_REGION_COLOR = "tab:green"
BG_REGION_ALPHA = 0.3


class MatrixCutState:
    """Tracks in-progress cut-region/background-region marks on a
    matrix's working projection -- exactly one cut region, any number
    of background regions (unlike fit_mode.py's FitModeState, which
    caps background regions at 2; TV's own philosophy is "the more
    background you mark, the better"). Two-click pairing, modeled on
    FitModeState (fit_mode.py:29-159)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cut_region = None
        self._pending_cut_click = None
        self.bg_regions = []
        self._pending_bg_click = None

    def add_cut_click(self, x):
        if self._pending_cut_click is None:
            self._pending_cut_click = x
            return False
        lo, hi = sorted((self._pending_cut_click, x))
        self.cut_region = (lo, hi)
        self._pending_cut_click = None
        return True

    def add_bg_click(self, x):
        if self._pending_bg_click is None:
            self._pending_bg_click = x
            return False
        lo, hi = sorted((self._pending_bg_click, x))
        self.bg_regions.append((lo, hi))
        self._pending_bg_click = None
        return True

    def clear_cut(self):
        self.cut_region = None
        self._pending_cut_click = None

    def clear_bg(self):
        self.bg_regions = []
        self._pending_bg_click = None


class MatrixCutController:
    """Held-key + click marker placement for the matrix panel's
    projection view -- hold C for the cut region, hold B for a
    background region. Modeled on FitModeController's eventFilter/
    _held_key mechanism (fit_mode.py:382-428), simplified to two mark
    types instead of three."""

    def __init__(self, panel):
        self.panel = panel
        self.state = MatrixCutState()
        self._held_key = None
        self._artists = []
        panel.canvas.installEventFilter(self)
        panel.canvas.mpl_connect("button_press_event", self.on_click)

    def eventFilter(self, obj, event):
        if obj is self.panel.canvas:
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                if event.key() == Qt.Key.Key_C:
                    self._held_key = "cut"
                elif event.key() == Qt.Key.Key_B:
                    self._held_key = "bg"
            elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
                self._held_key = None
        return False

    def on_click(self, event):
        if event.inaxes != self.panel.axes or event.xdata is None or event.button != 1:
            return
        if self._held_key == "cut":
            self.state.add_cut_click(event.xdata)
        elif self._held_key == "bg":
            self.state.add_bg_click(event.xdata)
        else:
            return
        self._redraw_markers()
        self.panel._update_activate_button()

    def clear(self):
        self.state.reset()
        self._redraw_markers()
        self.panel._update_activate_button()

    def _redraw_markers(self):
        for artist in self._artists:
            try:
                artist.remove()
            except NotImplementedError:
                pass
        self._artists = []

        axes = self.panel.axes
        if self.state.cut_region is not None:
            lo, hi = self.state.cut_region
            self._artists.append(axes.axvspan(lo, hi, color=CUT_REGION_COLOR, alpha=CUT_REGION_ALPHA))
        for lo, hi in self.state.bg_regions:
            self._artists.append(axes.axvspan(lo, hi, color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA))
        self.panel.canvas.draw()
```

Replace the `MatrixPanel.__init__` method's body (everything from `self.heatmap_button = QPushButton(...)` through `self._plot_projection()`) with:
```python
        self.heatmap_button = QPushButton("Show Heatmap...")

        self.activate_cut_button = QPushButton("Activate Cut")
        self.activate_cut_button.setEnabled(False)
        self.activate_cut_button.clicked.connect(self._activate_cut)

        self.clear_marks_button = QPushButton("Clear Marks")
        self.clear_marks_button.clicked.connect(self._clear_marks)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Working on:"))
        top_bar.addWidget(self.axis_selector)
        top_bar.addStretch()
        top_bar.addWidget(self.clear_marks_button)
        top_bar.addWidget(self.activate_cut_button)
        top_bar.addWidget(self.heatmap_button)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(top_bar)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        self.cut_controller = MatrixCutController(self)

        self._plot_projection()
```

Update `_on_axis_changed` to clear stale marks when switching projections (marker positions are in the OLD projection's channel space and would be meaningless on the new one):
```python
    def _on_axis_changed(self, index):
        self.working_axis = self.axis_selector.itemData(index)
        self.cut_controller.clear()
        self._plot_projection()
```

Add these new methods to `MatrixPanel` (after `_plot_projection`):
```python
    def _clear_marks(self):
        self.cut_controller.clear()

    def _update_activate_button(self):
        self.activate_cut_button.setEnabled(self.cut_controller.state.cut_region is not None)

    def _activate_cut(self):
        from matrix_cut import compute_cut

        state = self.cut_controller.state
        result = compute_cut(self.matrix, self.working_axis, state.cut_region, state.bg_regions)
        label = (
            f"{os.path.basename(self.path)} cut "
            f"[{state.cut_region[0]:.1f}, {state.cut_region[1]:.1f}]"
        )
        self.main_window._add_combined_spectrum(label, result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, all 10 tests.

Run: `pytest -q` (full suite)
Expected: PASS (aside from the known pre-existing `test_theme.py` failure).

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "feat: add cut/background marking and cut activation to matrix panel"
```

---

### Task 5: `matrix_heatmap.py` — the heatmap window

**Files:**
- Create: `matrix_heatmap.py`
- Create: `tests/test_matrix_heatmap.py`

**Depends on Task 1** for the matrix array shape/dtype only.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matrix_heatmap.py`:

```python
import os

import numpy as np

from matrix_heatmap import MatrixHeatmapWindow

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_matrix_heatmap_window_displays_matrix(qapp):
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx")

    assert window.windowTitle() == "Heatmap -- test.mtx"
    assert window.image is not None


def test_matrix_heatmap_window_has_navigation_toolbar_for_zoom(qapp):
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx")

    # NavigationToolbar2QT's stock "Zoom" tool is what provides
    # rectangle-select zoom in/out -- confirm it's present (this app's
    # own _TrimmedNavigationToolbar deliberately drops it for 1D
    # spectra, but a 2D image genuinely needs it, so the heatmap uses
    # the stock toolbar, not the trimmed one).
    action_texts = [a.text() for a in window.nav_toolbar.actions()]
    assert "Zoom" in action_texts


def test_matrix_heatmap_handles_negative_values_without_crashing(qapp):
    matrix = np.array([[-5, 10], [20, -1]], dtype=np.int64)
    window = MatrixHeatmapWindow(matrix, "test.mtx")
    assert window.image is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_heatmap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'matrix_heatmap'`.

- [ ] **Step 3: Create `matrix_heatmap.py`**

```python
import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget


class MatrixHeatmapWindow(QMainWindow):
    """A separate, non-modal, view-only visualization of a loaded
    matrix -- opened on demand from the matrix panel. No markers, no
    interaction beyond the standard matplotlib zoom/pan/reset toolbar.
    TV itself never had any 2D matrix rendering at all (confirmed
    absent from its entire source tree during design); this is a
    genuine enhancement, not a port."""

    def __init__(self, matrix, path):
        super().__init__()
        self.setWindowTitle(f"Heatmap -- {os.path.basename(path)}")
        self.resize(700, 700)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        # The stock (untrimmed) toolbar -- unlike this app's 1D
        # spectrum view, a 2D image needs rectangle-select Zoom, which
        # _TrimmedNavigationToolbar deliberately removes for the 1D
        # case (main_window.py's own _zoom_x buttons cover that
        # instead). Home/Pan/Zoom/Save all apply naturally to an image.
        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self, coordinates=False)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

        # SymLogNorm handles the negative values this app's matrix
        # data can genuinely contain (random-coincidence subtraction)
        # without error -- plain LogNorm requires strictly positive
        # data and would raise on any matrix with a negative cell.
        norm = SymLogNorm(linthresh=1.0, vmin=matrix.min(), vmax=max(matrix.max(), 1))
        self.image = self.axes.imshow(matrix, norm=norm, origin="lower", aspect="auto")
        self.figure.colorbar(self.image, ax=self.axes)
        self.axes.set_xlabel("X channel")
        self.axes.set_ylabel("Y channel")
        self.canvas.draw()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_heatmap.py -v`
Expected: PASS, all 3 tests.

Run: `pytest -q` (full suite)
Expected: PASS (aside from the known pre-existing `test_theme.py` failure).

- [ ] **Step 5: Wire the heatmap button in `matrix_panel.py`**

Add this method to `MatrixPanel` (it was referenced as a TODO-free forward reference in Task 3's skeleton but never connected):
```python
    def _open_heatmap(self):
        from matrix_heatmap import MatrixHeatmapWindow

        self._heatmap_window = MatrixHeatmapWindow(self.matrix, self.path)
        self._heatmap_window.show()
```

In `__init__`, connect the button (add right after `self.heatmap_button = QPushButton("Show Heatmap...")`):
```python
        self.heatmap_button.clicked.connect(self._open_heatmap)
```

Add a test to `tests/test_matrix_panel.py`:
```python
def test_matrix_panel_heatmap_button_opens_heatmap_window(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    panel._open_heatmap()

    assert panel._heatmap_window is not None
    assert panel._heatmap_window.isVisible()
```

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, all 11 tests.

- [ ] **Step 6: Commit**

```bash
git add matrix_heatmap.py tests/test_matrix_heatmap.py matrix_panel.py tests/test_matrix_panel.py
git commit -m "feat: add on-demand matrix heatmap window"
```

---

### Task 6: `main_window.py` integration

**Files:**
- Modify: `main_window.py`
- Create: `tests/test_matrix_integration.py`

**Depends on Tasks 3-5.**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matrix_integration.py`:

```python
import os

from main_window import MainWindow

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_open_matrix_action_exists(qapp):
    main_window = MainWindow()
    assert hasattr(main_window, "open_matrix_action")
    assert main_window.open_matrix_action.shortcut().toString() == "Ctrl+Shift+O"


def test_activating_main_window_disables_matrix_panel(qapp):
    main_window = MainWindow()
    panel = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))

    main_window._on_activated()

    assert panel.isEnabled() is False


def test_activating_matrix_panel_disables_main_window(qapp):
    main_window = MainWindow()
    panel = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))

    panel._on_activated()

    assert main_window.isEnabled() is False


def test_closing_matrix_panel_reenables_main_window(qapp):
    main_window = MainWindow()
    panel = main_window._open_matrix_panel(os.path.join(FIXTURES, "gg.mtx"))
    panel._on_activated()
    assert main_window.isEnabled() is False

    panel.close()

    assert main_window.isEnabled() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_matrix_integration.py -v`
Expected: FAIL — `AttributeError: 'MainWindow' object has no attribute 'open_matrix_action'`.

- [ ] **Step 3: Update `main_window.py`**

Add the import, alongside the existing loader imports:
```python
from n42_io import load_n42
```
becomes (insert a new line right after it):
```python
from n42_io import load_n42
from matrix_panel import MatrixPanel
```

In `_build_menu`, add the new action right after the existing `Open...` action block (before `Save Spectrum...`):
```python
        self.open_matrix_action = QAction("Open Matrix...", self)
        self.open_matrix_action.setShortcut("Ctrl+Shift+O")
        self.open_matrix_action.triggered.connect(self._open_matrix_dialog)
        self.file_menu.addAction(self.open_matrix_action)
```

In `MainWindow.__init__`, add right after `self.spectra = []`:
```python
        self._matrix_panels = []  # currently-open MatrixPanel windows -- see _on_activated
```

**Why an explicit list instead of `self.findChildren(MatrixPanel)`:** `MatrixPanel` is constructed as its own independent top-level window (`super().__init__()` with no `parent=` argument -- Task 3), specifically so it behaves as a real, separate window rather than being embedded in `MainWindow`'s window group. That means it is never a Qt child of `MainWindow`, so `findChildren` would silently return an empty list. An explicit list is the correct, working mechanism here.

Add these new methods to `MainWindow` (near `_open_file_dialog`):
```python
    def _open_matrix_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Matrix", self.settings.last_folder(), "Matrix files (*.mtx);;All files (*)"
        )
        if path:
            try:
                self._open_matrix_panel(path)
            except ParseError as exc:
                QMessageBox.warning(self, "Could not open matrix", str(exc))

    def _open_matrix_panel(self, path):
        panel = MatrixPanel(self, path)
        self._matrix_panels.append(panel)
        panel.show()
        return panel

    def _on_activated(self):
        for panel in self._matrix_panels:
            panel.setEnabled(False)
        self.setEnabled(True)
```

Add the corresponding methods to `MatrixPanel` in `matrix_panel.py` (the other half of the mutual-exclusivity pair, plus registry cleanup on close):
```python
    def _on_activated(self):
        self.main_window.setEnabled(False)
        self.setEnabled(True)
```

**Detecting "this window became active" the correct way:** Qt's `focusInEvent` is about keyboard focus *within* the currently-active window (it can land on a child widget like the canvas, and the top-level window itself may never receive it) -- not what's needed here. The right mechanism is `QEvent.Type.WindowActivate`, which Qt delivers to a top-level window itself via `changeEvent()` whenever the OS-level active window changes (clicking into a different window, alt-tab, etc.) -- exactly the "activating one window" semantic this behavior needs.

`main_window.py` does not currently import `QEvent`. Change:
```python
from PySide6.QtCore import Qt, QRectF
```
to:
```python
from PySide6.QtCore import QEvent, Qt, QRectF
```

In `main_window.py`'s `MainWindow` class, add:
```python
    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowActivate:
            self._on_activated()
```
And correspondingly in `matrix_panel.py`'s `MatrixPanel` class (which already imports `QEvent` from Task 4):
```python
    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowActivate:
            self._on_activated()

    def closeEvent(self, event):
        super().closeEvent(event)
        if self in self.main_window._matrix_panels:
            self.main_window._matrix_panels.remove(self)
        self.main_window.setEnabled(True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_integration.py -v`
Expected: PASS, all 4 tests.

Run: `pytest -q` (full suite)
Expected: PASS (aside from the known pre-existing `test_theme.py` failure).

- [ ] **Step 5: Commit**

```bash
git add main_window.py matrix_panel.py tests/test_matrix_integration.py
git commit -m "feat: wire matrix panel into main window (Open Matrix..., focus exclusivity)"
```

---

### Task 7: HowTo documentation

**Files:**
- Modify: `help_content.py`
- Modify: `tests/test_help_content.py`

**Depends on Tasks 3-6** for the feature to accurately document.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_help_content.py`:

```python
def test_howto_html_documents_matrix_analysis():
    html = build_howto_html()
    assert "<h3>11. Matrix analysis</h3>" in html
    assert "Open Matrix" in html
    assert "Ctrl+Shift+O" in html.replace("&#43;", "+")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_help_content.py -k matrix -v`
Expected: FAIL — section not present.

- [ ] **Step 3: Update `build_howto_html()` in `help_content.py`**

Add a new section right before the closing `"""` of the HowTo body (after the existing "10. View options" section):

```html
<h3>11. Matrix analysis</h3>
<p><b>File &gt; Open Matrix...</b> (<kbd>Ctrl+Shift+O</kbd>) opens a
2D coincidence matrix (<b>.mtx</b>) in its own window. Only the raw
histogram is read -- there's no way to save a matrix back out. The
matrix panel computes both its X and Y projections up front; pick
which one to work on from the dropdown. Hold <kbd>C</kbd> and click
twice to mark the cut (signal) region, and hold <kbd>B</kbd> and click
twice for each background region -- any number of background regions
are allowed, and more background generally means better statistics.
<b>Activate Cut</b> computes a background-subtracted spectrum
(weighted automatically by region width, same convention TV uses) and
adds it to the main window like any other loaded spectrum -- you can
fit, calibrate, or export it exactly the same way. Negative counts can
appear in the result and are expected, not an error. <b>Show
Heatmap...</b> opens a separate, view-only 2D intensity map of the
matrix with its own zoom/pan controls -- purely for visual reference;
marking and cutting always happens on the projection, not the
heatmap.</p>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_help_content.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add help_content.py tests/test_help_content.py
git commit -m "docs: document matrix analysis in HowTo"
```

---

### Task 8: Windows verification

**Files:** None (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: PASS, full suite.

- [ ] **Step 2: Build the Windows package**

Run `packaging/windows/build.ps1` (do not redirect stderr via `2>&1`).

- [ ] **Step 3: Smoke-test the built exe**

Launch `dist/SpectraTools.exe`. Find the real worker process (the launched PID is the onefile bootloader, not the app) via `Get-CimInstance Win32_Process -Filter "ParentProcessId = <launched PID>"`, then confirm its window via direct Win32 API calls (`EnumWindows` filtered to its PID, `GetWindowRect`/`IsWindowVisible`). Kill both PIDs explicitly afterward.

- [ ] **Step 4: Manually exercise the real feature** (not just launch-and-quit, since this is a large new UI surface with no prior manual verification)

Using the running app: File > Open Matrix..., pick `tests/fixtures/gg.mtx` (or a copy of the real root-level sample files if still present), confirm the matrix panel opens with a projection plotted, switch between X/Y projections, mark a cut region and a background region with C/B holds, confirm Activate Cut is enabled only once a cut region exists, activate it, confirm a new spectrum appears in the main window's Spectra list, open the heatmap and confirm it renders with working zoom, confirm clicking between the main window and matrix panel correctly enables/disables each other without either closing.

- [ ] **Step 5: Report**

Report pass/fail of Steps 1-4 plainly, including any issues found during manual exercise.

---

### Task 9: Linux verification

**Files:** None (verification only).

- [ ] **Step 1: Run the full test suite under WSL**

Using a real script file (not an inline `bash -c` one-liner), run `pytest -q` inside Ubuntu-24.04.

- [ ] **Step 2: Build both packages**

Run `packaging/linux/build.sh` (RPM, built on AlmaLinux-8) and `packaging/linux/build_deb.sh` (DEB, Ubuntu-24.04).

- [ ] **Step 3: Smoke-test**

Install using an **absolute path** to each built package. Smoke-test the RPM on **AlmaLinux-10** (never AlmaLinux-8, which is build-host-only) and the DEB on Ubuntu-24.04. Before trusting either result, confirm the installed binary's checksum matches the freshly-built `dist-onedir/.../SpectraTools` binary. Launch under `timeout -k 1 <seconds> spectratools`, treat exit 124 as the only real "it's alive" signal.

- [ ] **Step 4: Report**

Report pass/fail of Steps 1-3 plainly. Full manual feature exercise (per Task 8 Step 4) is not required on both platforms if Windows manual verification already passed, but flag anything that looks Linux-specific (font rendering, window manager quirks with the two-window focus-exclusivity behavior) if you notice it.

---

## Self-Review Notes

- **Spec coverage:** every architecture piece from the design spec has a task — `mtx_io.py` (Task 1), `matrix_cut.py` (Task 2), `matrix_panel.py` skeleton+markers (Tasks 3-4), `matrix_heatmap.py` (Task 5), `main_window.py` integration (Task 6), HowTo (Task 7). Every Resolved Decision in the spec is reflected: one cut region + unlimited background regions (Task 4's `MatrixCutState`), automatic gate-width-only weighting (Task 2's `compute_cut`), negative-value preservation (tested explicitly in Tasks 1, 2, 4), level-0-only reading with no user-facing selection (Task 1's `load_mtx` signature takes no level parameter at all), read-only matrix support (no writer anywhere in this plan), focus-exclusivity between the two windows (Task 6), heatmap as a separate on-demand non-modal window with zoom (Task 5).
- **Values verified, not hand-transcribed:** every oracle value in Task 1's tests was generated by actually compiling and running libmfile against the real fixture files (not computed by hand), and the `lc_c2` decode algorithm itself was independently hand-traced against real compressed bytes from both files across all four tag branches before being transcribed into `mtx_io.py` — see the design spec and this session's research for the full verification trail.
- **Placeholder scan:** no TBD/TODO; every code block is complete, runnable code, including the trickiest part (the bit-exact `lc_c2` decompression).
- **Type/name consistency:** `load_mtx` (Task 1) is imported and called identically in Task 3. `compute_projection`/`compute_cut` (Task 2) are imported and called identically in Tasks 3-4, with signatures matching between definition and every call site. `MatrixPanel(main_window, path)` (Task 3) is constructed identically in Tasks 6's tests and integration code. `_add_combined_spectrum(label, data)` (Task 4) matches the existing, already-verified signature in `main_window.py`. `MatrixHeatmapWindow(matrix, path)` (Task 5) is constructed identically from `matrix_panel.py`'s `_open_heatmap` and from its own tests.
- **Scope discipline:** no `save_mtx`, no `CUT_WFIT`, no multiple cut regions, no level selection, no `.gate`/`.cutdir` persistence — matching the design spec's explicit Out of Scope list. The heatmap has no interaction beyond the standard matplotlib toolbar, matching "visualization only."
