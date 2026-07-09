# .spk Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `.spk` reader (tv/Mfile native spectrum format) to SpectraTools, supporting both real-world sub-formats confirmed against sample files: MAT_LC (compressed) and oldmat/raw arrays.

**Architecture:** A new standalone parsing module `spk_io.py` (mirroring the existing `histogram_io.py`/`spe_io.py` pattern: pure functions, no Qt dependency, raising the shared `ParseError` on any malformed input) exposing `load_spk(path) -> np.ndarray`. Wired into `main_window.py`'s existing per-extension dispatch, exactly like `.spe` was.

**Tech Stack:** Python 3.13, `struct` (binary parsing), `numpy` (array construction), `pytest` (tests).

**Spec:** `docs/superpowers/specs/2026-07-09-spk-reader-design.md` — read this first for the full byte-layout rationale; this plan doesn't re-derive it, only implements it.

---

### Task 1: Test fixtures

**Files:**
- Create: `tests/fixtures/demo.spk` (copy of repo-root `demo.spk`)
- Create: `tests/fixtures/pg_25um_f.spk` (copy of repo-root `pg_25um_f.spk`)

- [ ] **Step 1: Copy the two real sample files into the test fixtures directory**

```bash
cp ../../demo.spk tests/fixtures/demo.spk
cp ../../pg_25um_f.spk tests/fixtures/pg_25um_f.spk
```

(Paths above are relative to the worktree root `C:\Users\RIG\Documents\Claude\PeakFinderFitting\.worktrees\histogram-viewer`; the source files live at the main repo root `C:\Users\RIG\Documents\Claude\PeakFinderFitting\demo.spk` and `...\pg_25um_f.spk`. Use the absolute paths for `cp` if not already `cd`'d into the worktree.)

- [ ] **Step 2: Verify the copies are correct**

```bash
ls -la tests/fixtures/demo.spk tests/fixtures/pg_25um_f.spk
```

Expected: `demo.spk` is 7393 bytes, `pg_25um_f.spk` is 65600 bytes.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/demo.spk tests/fixtures/pg_25um_f.spk
git commit -m "Add real .spk sample files as test fixtures"
```

---

### Task 2: `spk_io.py` — MAT_LC (compressed) path

**Files:**
- Create: `spk_io.py`
- Test: `tests/test_spk_io.py`

This task implements the compressed variant only. `load_spk` will correctly parse MAT_LC files and reject anything else with `ParseError` (the oldmat path is added in Task 3 — until then, every non-LC input is expected to fail, which is exactly what the tests below check).

- [ ] **Step 1: Write the failing tests for the LC path**

Create `tests/test_spk_io.py`:

```python
import struct
from pathlib import Path

import pytest

from histogram_io import ParseError
from spk_io import load_spk

FIXTURES = Path(__file__).parent / "fixtures"


def _lc_header(version, levels, lines, columns, poslentablepos):
    return struct.pack(
        "<11I",
        0x80FFFF10, version, levels, lines, columns, poslentablepos,
        0, 0, 0, 0, 0,
    )


def test_parses_real_demo_spk():
    # demo.spk is a real tv/Mfile MAT_LC file (magic 0x80FFFF10, version 2,
    # 4096 channels) -- verified byte-for-byte during design against the
    # libmfile-1.0.7 source, and the decompressed result was independently
    # sanity-checked to have zero negative values and a clear multi-peak
    # gamma-spectrum shape (see design spec).
    data = load_spk(str(FIXTURES / "demo.spk"))
    assert len(data) == 4096
    assert list(data[:20]) == [
        0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0,
    ]
    assert int(data[724]) == 41580
    assert all(v >= 0 for v in data)
    assert int(data.sum()) == 13069569


def test_lc2_3value_tag(tmp_path):
    # tag byte 0x24 = 0b00_10_00_01... bits: a=0(->+0), b=1(->-1), c=2(->+1)
    # (3-value pack: tag bits 0-1 of the byte's top 2 bits are both 0)
    header = _lc_header(version=2, levels=1, lines=1, columns=3, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc2_3value.spk"
    file_path.write_bytes(header + poslen + bytes([0x24]))

    data = load_spk(str(file_path))

    assert list(data) == [0, -1, 1]


def test_lc2_2value_tag(tmp_path):
    # tag byte 0x53 = 0x40 + a(3) + (b(2)<<3): a=3(->-2), b=2(->+1)
    header = _lc_header(version=2, levels=1, lines=1, columns=2, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc2_2value.spk"
    file_path.write_bytes(header + poslen + bytes([0x53]))

    data = load_spk(str(file_path))

    assert list(data) == [-2, 1]


def test_lc2_1value_tag(tmp_path):
    # tag byte 0x85 = 0x80 + 5: n=5 -> zigzag decode -3
    header = _lc_header(version=2, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc2_1value.spk"
    file_path.write_bytes(header + poslen + bytes([0x85]))

    data = load_spk(str(file_path))

    assert list(data) == [-3]


def test_lc2_same_run_tag(tmp_path):
    # tag byte 0xC7 = 0xC0 + 7: n=7 -> diff=1, same=(7>>1)+3=6.
    # Produces [last+diff] followed by 6 copies of the *unchanged* last (0).
    header = _lc_header(version=2, levels=1, lines=1, columns=7, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc2_same_run.spk"
    file_path.write_bytes(header + poslen + bytes([0xC7]))

    data = load_spk(str(file_path))

    assert list(data) == [1, 0, 0, 0, 0, 0, 0]


def test_lc2_extended_single_value_tag(tmp_path):
    # tag byte 0xBD: low 6 bits = 61 (0x3D) -> 2 extension bytes needed.
    # extension bytes [0x00, 0x00] -> n = 59 + (0+1) + (0+1)*256 = 316
    # -> zigzag decode = +158
    header = _lc_header(version=2, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 3)
    file_path = tmp_path / "lc2_extended.spk"
    file_path.write_bytes(header + poslen + bytes([0xBD, 0x00, 0x00]))

    data = load_spk(str(file_path))

    assert list(data) == [158]


def test_lc1_3value_tag(tmp_path):
    # tag byte 0x39: a=1,b=2,c=3 (each 2 bits) -> cumulative deltas -1,+1,-2
    header = _lc_header(version=1, levels=1, lines=1, columns=3, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc1_3value.spk"
    file_path.write_bytes(header + poslen + bytes([0x39]))

    data = load_spk(str(file_path))

    assert list(data) == [-1, 0, -2]


def test_lc1_2value_tag(tmp_path):
    # tag byte 0x6B = 0x40 + a(3) + (b(5)<<3) -> cumulative deltas -2,-3
    header = _lc_header(version=1, levels=1, lines=1, columns=2, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc1_2value.spk"
    file_path.write_bytes(header + poslen + bytes([0x6B]))

    data = load_spk(str(file_path))

    assert list(data) == [-2, -5]


def test_lc1_1value_tag(tmp_path):
    # tag byte 0x8A = 0x80 + 10: n=10 -> zigzag decode +5
    header = _lc_header(version=1, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc1_1value.spk"
    file_path.write_bytes(header + poslen + bytes([0x8A]))

    data = load_spk(str(file_path))

    assert list(data) == [5]


def test_lc1_extended_tag(tmp_path):
    # tag byte 0xD4: initial 6 bits = 20, one continuation byte 0x03
    # (bit7 clear = last continuation byte) -> i = 20 + (3<<6) = 212
    # -> zigzag decode +106
    header = _lc_header(version=1, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 2)
    file_path = tmp_path / "lc1_extended.spk"
    file_path.write_bytes(header + poslen + bytes([0xD4, 0x03]))

    data = load_spk(str(file_path))

    assert list(data) == [106]


def test_lc_empty_line_is_all_zero(tmp_path):
    header = _lc_header(version=2, levels=1, lines=1, columns=5, poslentablepos=44)
    poslen = struct.pack("<2I", 0, 0)  # len == 0 means "empty line" per source
    file_path = tmp_path / "lc_empty.spk"
    file_path.write_bytes(header + poslen)

    data = load_spk(str(file_path))

    assert list(data) == [0, 0, 0, 0, 0]


def test_lc_rejects_unsupported_version(tmp_path):
    header = _lc_header(version=3, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc_badversion.spk"
    file_path.write_bytes(header + poslen + bytes([0x80]))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_lc_rejects_2d_matrix(tmp_path):
    # levels/lines checked immediately after the header is parsed, before
    # the position table is even read -- so a bare 44-byte header is enough
    # to exercise this rejection.
    header = _lc_header(version=2, levels=1, lines=2, columns=1, poslentablepos=44)
    file_path = tmp_path / "lc_2d.spk"
    file_path.write_bytes(header)

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_lc_rejects_truncated_compressed_data(tmp_path):
    header = _lc_header(version=2, levels=1, lines=1, columns=1, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 5)  # claims 5 bytes but none follow
    file_path = tmp_path / "lc_truncated.spk"
    file_path.write_bytes(header + poslen)

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_lc_rejects_stream_that_runs_out_of_bytes(tmp_path):
    # A 2-value tag only ever produces 2 values, but 3 channels are
    # declared -- decoding must run out of compressed bytes trying to
    # satisfy the third and raise, rather than reading past the buffer.
    header = _lc_header(version=2, levels=1, lines=1, columns=3, poslentablepos=44)
    poslen = struct.pack("<2I", 44 + 8, 1)
    file_path = tmp_path / "lc_short_stream.spk"
    file_path.write_bytes(header + poslen + bytes([0x53]))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_raises_parse_error_on_unrecognized_file(tmp_path):
    file_path = tmp_path / "not_a_spk_file.spk"
    file_path.write_bytes(b"this is not a valid tv/Mfile spk file at all" + b"\x00" * 40)

    with pytest.raises(ParseError):
        load_spk(str(file_path))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_spk_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spk_io'`

- [ ] **Step 3: Implement `spk_io.py` (LC path only)**

Create `spk_io.py`:

```python
import struct

import numpy as np

from histogram_io import ParseError

LC_MAGIC = 0x80FFFF10
LC_HEADER_SIZE = 44
LC_POSLEN_SIZE = 8


def load_spk(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()

    if len(data) >= 4 and struct.unpack_from("<I", data, 0)[0] == LC_MAGIC:
        return _load_lc(data, path)

    raise ParseError(
        f"Not a recognized tv/Mfile .spk file (no LC magic found): {path}"
    )


def _zigzag_decode(i: int) -> int:
    if i & 1:
        return -((i >> 1) + 1)
    return i >> 1


def _lc1_uncompress(data: bytes, num: int, path: str) -> list:
    out = []
    last = 0
    nleft = num
    pos = 0
    n = len(data)

    def next_byte():
        nonlocal pos
        if pos >= n:
            raise ParseError(f"Truncated .spk LC1 compressed stream: {path}")
        b = data[pos]
        pos += 1
        return b

    while nleft > 0:
        t = next_byte()
        tag = t >> 6
        if tag == 0:
            nleft -= 3
            if nleft < 0:
                raise ParseError(f"Corrupt .spk LC1 stream (tag overruns channel count): {path}")
            for shift in (0, 2, 4):
                i = (t >> shift) & 0x3
                last += _zigzag_decode(i)
                out.append(last)
        elif tag == 1:
            nleft -= 2
            if nleft < 0:
                raise ParseError(f"Corrupt .spk LC1 stream (tag overruns channel count): {path}")
            for shift in (0, 3):
                i = (t >> shift) & 0x7
                last += _zigzag_decode(i)
                out.append(last)
        elif tag == 2:
            nleft -= 1
            i = t & 0x3F
            last += _zigzag_decode(i)
            out.append(last)
        else:
            nleft -= 1
            s = 6
            i = t & 0x3F
            while True:
                t = next_byte()
                i += (t & 0x7F) << s
                s += 7
                if not (t & 0x80):
                    break
            last += _zigzag_decode(i)
            out.append(last)

    return out


def _lc2_uncompress(data: bytes, num: int, path: str) -> list:
    out = []
    last = 0
    nleft = num
    pos = 0
    n = len(data)

    def next_byte():
        nonlocal pos
        if pos >= n:
            raise ParseError(f"Truncated .spk LC2 compressed stream: {path}")
        b = data[pos]
        pos += 1
        return b

    while nleft > 0:
        t = next_byte()
        if t & 0x80:
            v = t & 0x3F
            if v > 59:
                extra_bytes = v - 59
                v = 59
                for i in range(extra_bytes):
                    b = next_byte()
                    v += (b + 1) << (8 * i)
            if t & 0x40:
                diff = v & 1
                same = (v >> 1) + 3
                out.append(last + diff)
                nleft -= same
                if nleft <= 0:
                    raise ParseError(f"Corrupt .spk LC2 stream (run overruns channel count): {path}")
                out.extend([last] * same)
            else:
                last += _zigzag_decode(v)
                out.append(last)
            nleft -= 1
        elif t & 0x40:
            nleft -= 2
            if nleft < 0:
                raise ParseError(f"Corrupt .spk LC2 stream (tag overruns channel count): {path}")
            a = t & 0x7
            out.append(last + _zigzag_decode(a))
            b = (t >> 3) & 0x7
            last += _zigzag_decode(b)
            out.append(last)
        else:
            nleft -= 3
            if nleft < 0:
                raise ParseError(f"Corrupt .spk LC2 stream (tag overruns channel count): {path}")
            a = t & 0x3
            out.append(last + _zigzag_decode(a))
            b = (t >> 2) & 0x3
            out.append(last + _zigzag_decode(b))
            c = (t >> 4) & 0x3
            last += _zigzag_decode(c)
            out.append(last)

    return out


def _load_lc(data: bytes, path: str) -> np.ndarray:
    if len(data) < LC_HEADER_SIZE:
        raise ParseError(f"File too short for a complete .spk LC header: {path}")

    fields = struct.unpack_from("<11I", data, 0)
    (_magic, version, levels, lines, columns, poslentablepos,
     _freepos, _freelistpos, _used, _free, _status) = fields

    if version not in (1, 2):
        raise ParseError(f"Unsupported .spk LC version {version}: {path}")
    if levels != 1 or lines != 1:
        raise ParseError(
            f"2-D .spk matrices (levels={levels}, lines={lines}) are not supported: {path}"
        )
    if columns <= 0:
        raise ParseError(f"Invalid channel count in .spk file: {path}")

    if len(data) < poslentablepos + LC_POSLEN_SIZE:
        raise ParseError(f"File too short for .spk position table: {path}")

    pos, length = struct.unpack_from("<2I", data, poslentablepos)

    if length == 0:
        return np.zeros(columns, dtype=np.int64)

    if len(data) < pos + length:
        raise ParseError(f"File too short for .spk compressed data: {path}")

    compressed = data[pos:pos + length]
    uncompress = _lc1_uncompress if version == 1 else _lc2_uncompress
    values = uncompress(compressed, columns, path)
    return np.array(values, dtype=np.int64)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_spk_io.py -v`
Expected: all tests PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
git add spk_io.py tests/test_spk_io.py
git commit -m "Add .spk MAT_LC (line-compressed) reader with LC1/LC2 decompression"
```

---

### Task 3: `spk_io.py` — oldmat (raw array) path

**Files:**
- Modify: `spk_io.py`
- Test: `tests/test_spk_io.py`

- [ ] **Step 1: Write the failing tests for the oldmat path**

Append to `tests/test_spk_io.py`:

```python
def test_parses_real_pg_25um_f_spk():
    # pg_25um_f.spk is a real tv/Mfile oldmat/MAT_LF4 file: 16384 raw
    # little-endian float32 channel values, no header, identified by a
    # 64-byte "\nMatFmt: 16k.lf4:2\n" trailer at EOF -- verified during
    # design against the libmfile-1.0.7 source and the file's own bytes.
    data = load_spk(str(FIXTURES / "pg_25um_f.spk"))
    assert len(data) == 16384

    with open(FIXTURES / "pg_25um_f.spk", "rb") as f:
        raw = f.read()
    expected_first5 = struct.unpack_from("<5f", raw, 0)
    assert list(data[:5]) == [round(v) for v in expected_first5]
    expected_last = struct.unpack_from("<f", raw, 4 * 16383)[0]
    assert int(data[16383]) == round(expected_last)


def _oldmat_bytes(fmt_string, payload):
    trailer = b"\nMatFmt: " + fmt_string.encode("ascii") + b"\n"
    trailer = trailer.ljust(64, b"\x00")
    return payload + trailer


def test_oldmat_le4(tmp_path):
    values = [10, -5, 2000]
    payload = struct.pack("<3i", *values)
    file_path = tmp_path / "oldmat_le4.spk"
    file_path.write_bytes(_oldmat_bytes("3.le4", payload))

    data = load_spk(str(file_path))

    assert list(data) == values


def test_oldmat_lf4(tmp_path):
    values = [1.4, 2.6, -3.2]
    payload = struct.pack("<3f", *values)
    file_path = tmp_path / "oldmat_lf4.spk"
    file_path.write_bytes(_oldmat_bytes("3.lf4:2", payload))

    data = load_spk(str(file_path))

    assert list(data) == [round(v) for v in values]


def test_oldmat_he2(tmp_path):
    values = [1, 60000, 3]
    payload = struct.pack(">3H", *values)
    file_path = tmp_path / "oldmat_he2.spk"
    file_path.write_bytes(_oldmat_bytes("3.he2", payload))

    data = load_spk(str(file_path))

    assert list(data) == values


def test_oldmat_rejects_unsupported_fmtname(tmp_path):
    payload = struct.pack("<3i", 1, 2, 3)
    file_path = tmp_path / "oldmat_bad_fmt.spk"
    file_path.write_bytes(_oldmat_bytes("3.xyz", payload))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_oldmat_rejects_2d_matrix(tmp_path):
    # "2.3.le4" parses as lines=2, columns=3 (levels defaults to 1) --
    # lines != 1 must still be rejected as a 2-D matrix.
    payload = struct.pack("<6i", 1, 2, 3, 4, 5, 6)
    file_path = tmp_path / "oldmat_2d.spk"
    file_path.write_bytes(_oldmat_bytes("2.3.le4", payload))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_oldmat_rejects_size_mismatch(tmp_path):
    payload = struct.pack("<3i", 1, 2, 3)
    file_path = tmp_path / "oldmat_mismatch.spk"
    # declares 5 channels but the payload only has 3
    file_path.write_bytes(_oldmat_bytes("5.le4", payload))

    with pytest.raises(ParseError):
        load_spk(str(file_path))


def test_oldmat_rejects_malformed_trailer(tmp_path):
    payload = struct.pack("<3i", 1, 2, 3)
    # magic prefix present but no terminating newline after the fmt string
    trailer = (b"\nMatFmt: 3.le4" + b"\x00" * 50)[:64]
    file_path = tmp_path / "oldmat_malformed.spk"
    file_path.write_bytes(payload + trailer)

    with pytest.raises(ParseError):
        load_spk(str(file_path))
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_spk_io.py -v -k oldmat`
Expected: FAIL (`pg_25um_f.spk` test raises `ParseError` since oldmat isn't wired up yet; others fail similarly)

- [ ] **Step 3: Implement the oldmat path and wire it into the dispatcher**

Add to the top of `spk_io.py`, right after the existing `LC_POSLEN_SIZE = 8` line:

```python
OLDMAT_TRAILER_SIZE = 64
OLDMAT_MAGIC = b"\nMatFmt: "

OLDMAT_DTYPES = {
    "le4": "<i4",
    "he4": ">i4",
    "le2": "<u2",
    "he2": ">u2",
    "le2s": "<i2",
    "he2s": ">i2",
    "lf4": "<f4",
    "hf4": ">f4",
}
```

Replace `load_spk` with:

```python
def load_spk(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()

    if len(data) >= 4 and struct.unpack_from("<I", data, 0)[0] == LC_MAGIC:
        return _load_lc(data, path)

    if len(data) >= OLDMAT_TRAILER_SIZE:
        trailer = data[-OLDMAT_TRAILER_SIZE:]
        if trailer.startswith(OLDMAT_MAGIC):
            return _load_oldmat(data, trailer, path)

    raise ParseError(
        f"Not a recognized tv/Mfile .spk file (no LC magic, no MatFmt trailer): {path}"
    )
```

Append at the end of `spk_io.py`:

```python
def _parse_oldmat_format(fmt: str, path: str):
    text = fmt.strip()
    i = 0
    n = len(text)
    nums = []

    while i < n and text[i].isdigit():
        start = i
        while i < n and text[i].isdigit():
            i += 1
        value = int(text[start:i])
        if i < n and text[i] == "k":
            i += 1
            value *= 1024
        if value == 0:
            raise ParseError(f"Invalid .spk MatFmt dimension in '{fmt}': {path}")
        nums.append(value)
        if len(nums) > 3:
            raise ParseError(f"Too many dimensions in .spk MatFmt string '{fmt}': {path}")
        if i < n and text[i] == ".":
            i += 1
        else:
            break

    name_start = i
    while i < n and text[i] != ":":
        i += 1
    fmtname = text[name_start:i]

    if fmtname not in OLDMAT_DTYPES:
        raise ParseError(f"Unsupported .spk Mfile format '{fmtname}': {path}")

    if i < n and text[i] == ":":
        i += 1
        while i < n and text[i].isdigit():
            i += 1

    if i != n:
        raise ParseError(f"Unexpected trailing text in .spk MatFmt string '{fmt}': {path}")

    if not nums:
        raise ParseError(f"Missing channel count in .spk MatFmt string '{fmt}': {path}")
    if len(nums) == 1:
        levels, lines, columns = 1, 1, nums[0]
    elif len(nums) == 2:
        levels, lines, columns = 1, nums[0], nums[1]
    else:
        levels, lines, columns = nums[0], nums[1], nums[2]

    return levels, lines, columns, OLDMAT_DTYPES[fmtname]


def _load_oldmat(data: bytes, trailer: bytes, path: str) -> np.ndarray:
    text = trailer[len(OLDMAT_MAGIC):]
    newline = text.find(b"\n")
    if newline == -1:
        raise ParseError(f"Malformed .spk MatFmt trailer (no terminator): {path}")

    try:
        fmt = text[:newline].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ParseError(f"Malformed .spk MatFmt trailer: {path}") from exc

    levels, lines, columns, dtype_str = _parse_oldmat_format(fmt, path)

    if levels != 1 or lines != 1:
        raise ParseError(
            f"2-D .spk matrices (levels={levels}, lines={lines}) are not supported: {path}"
        )

    dtype = np.dtype(dtype_str)
    expected_size = columns * dtype.itemsize + OLDMAT_TRAILER_SIZE
    if len(data) != expected_size:
        raise ParseError(
            f"File size {len(data)} does not match declared .spk dimensions "
            f"({columns} channels of {dtype_str}, expected {expected_size}): {path}"
        )

    channels = np.frombuffer(data, dtype=dtype, count=columns, offset=0)
    if np.issubdtype(dtype, np.floating):
        return np.round(channels).astype(np.int64)
    return channels.astype(np.int64)
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `pytest tests/test_spk_io.py -v`
Expected: all tests PASS (24 tests total: 16 from Task 2 + 8 new)

- [ ] **Step 5: Commit**

```bash
git add spk_io.py tests/test_spk_io.py
git commit -m "Add .spk oldmat (raw-array) reader for le4/he4/le2/le2s/he2/he2s/lf4/hf4"
```

---

### Task 4: Integrate into the loading pipeline

**Files:**
- Modify: `main_window.py:30-33` (imports)
- Modify: `main_window.py:177-197` (`_open_file_dialog`, `_try_load_spectrum`)

- [ ] **Step 1: Add the import**

In `main_window.py`, change:

```python
from histogram_io import ParseError, load_histogram
from settings import Settings
from spe_io import load_spe
from spectrum import LoadedSpectrum, next_color
```

to:

```python
from histogram_io import ParseError, load_histogram
from settings import Settings
from spe_io import load_spe
from spk_io import load_spk
from spectrum import LoadedSpectrum, next_color
```

- [ ] **Step 2: Update the file dialog filter and the loader dispatch**

In `main_window.py`, change:

```python
    def _open_file_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Histogram",
            self.settings.last_folder(),
            "Spectrum files (*.txt *.spe);;Text files (*.txt);;SPE files (*.spe);;All files (*)",
        )
        if paths:
            self._load_files(paths)

    def _try_load_spectrum(self, path):
        loader = load_spe if path.lower().endswith(".spe") else load_histogram
        try:
            data = loader(path)
        except ParseError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        except OSError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        color = next_color(self._next_color_index)
        self._next_color_index += 1
        return LoadedSpectrum(path, data, color), None
```

to:

```python
    def _open_file_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Histogram",
            self.settings.last_folder(),
            "Spectrum files (*.txt *.spe *.spk);;Text files (*.txt);;"
            "SPE files (*.spe);;SPK files (*.spk);;All files (*)",
        )
        if paths:
            self._load_files(paths)

    def _try_load_spectrum(self, path):
        lower = path.lower()
        if lower.endswith(".spe"):
            loader = load_spe
        elif lower.endswith(".spk"):
            loader = load_spk
        else:
            loader = load_histogram
        try:
            data = loader(path)
        except ParseError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        except OSError as exc:
            return None, f"{os.path.basename(path)}: {exc}"
        color = next_color(self._next_color_index)
        self._next_color_index += 1
        return LoadedSpectrum(path, data, color), None
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest -v`
Expected: all tests PASS (previous `histogram_io`/`spe_io` tests + the 24 new `spk_io` tests, no regressions)

- [ ] **Step 4: Verify the real pipeline end-to-end (offscreen)**

Run (from the worktree root, Git Bash):

```bash
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
app = QApplication([])
w = MainWindow()
w._load_files(['tests/fixtures/demo.spk', 'tests/fixtures/pg_25um_f.spk'])
print('spectra loaded:', len(w.spectra))
for s in w.spectra:
    print(s.path, len(s.data), s.color)
"
```

Expected output: `spectra loaded: 2`, followed by two lines showing `demo.spk` with 4096 channels and `pg_25um_f.spk` with 16384 channels, each with a distinct color (e.g. `#1f77b4` and `#ff7f0e`).

- [ ] **Step 5: Render `demo.spk` to a PNG for a visual sanity check**

```bash
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe scripts/verify_render.py tests/fixtures/demo.spk /tmp/demo_spk_render.png
```

Then view `/tmp/demo_spk_render.png` (use the Read tool if running as an agent) and confirm it shows a plausible step-line gamma spectrum with several sharp peaks (matching the peak positions found during design: channels ~724-726, ~909-911, ~930-932, ~1260-1262), not noise or a flat line.

- [ ] **Step 6: Commit**

```bash
git add main_window.py
git commit -m "Wire .spk loading into the file dialog and spectrum-loading dispatch"
```

---

### Task 5: Rebuild and verify the Windows installer

**Files:** none (build artifacts only)

- [ ] **Step 1: Rebuild**

```powershell
powershell -File packaging\windows\build.ps1
```

Expected: build completes successfully, producing `packaging/windows/output/SpectraToolsSetup.exe`. (If Inno Setup's `islzma.dll` access-violation recurs, simply retry the same command — this has been an intermittent, non-deterministic packaging flake throughout the project, not a code issue.)

- [ ] **Step 2: Silent-install, launch, and confirm no crash**

```bash
"packaging/windows/output/SpectraToolsSetup.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /NOICONS
```

Then launch the installed `SpectraTools.exe` (typically under `Program Files (x86)\SpectraTools`), confirm it starts without crashing (check the process stays running for a few seconds), then terminate it.

- [ ] **Step 3: Clean up the test install**

Run the installed uninstaller silently (`unins000.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART`), then confirm the install directory is gone and no stray desktop shortcut was left behind.

- [ ] **Step 4: Report completion**

Report to the user that the `.spk` reader sub-project is complete (MAT_LC + oldmat/raw variants, both verified against real sample files), and ask whether to proceed to the third and final approved sub-project: peak fitting.
