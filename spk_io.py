import struct

import numpy as np

from histogram_io import ParseError
from int64_cast import checked_round_to_int64
from lc_codec import (
    DIM_MAX as MAT_COLMAX, decode_row, encode_row, put_tag_n, zigzag_decode as _zigzag_decode,
    zigzag_encode,
)

LC_MAGIC = 0x80FFFF10
LC_HEADER_SIZE = 44
LC_POSLEN_SIZE = 8

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


# The LC2 encoder moved to lc_codec in 7.0.0 so save_mtx can share it
# (one compressor for both writers). These names stay because tests and
# older call sites import them from here.
_zigzag_encode = zigzag_encode
_put_tag_n = put_tag_n


def _lc2_compress(values) -> bytes:
    """Ported from libmfile's lc2_compress (lc_c2.c:63-126); the
    implementation is lc_codec.encode_row. No int32 wrapping here, as
    before: save_spk's values are plain Python ints and an out-of-range
    delta raises ValueError rather than being stored wrapped."""
    return encode_row(values)


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
    """Kept as a distinct module-level name (rather than importing
    decode_row directly under this name) because tests call it
    directly and internal callers below reference it by this name.
    The actual bit-level tag decoding lives in lc_codec.decode_row,
    shared with mtx_io.py's matrix-row decoding -- same LC2 codec,
    just applied to a whole spectrum here instead of one matrix row.
    Passes its own `kind` so a corrupt .spk file's ParseError doesn't
    claim to be about a matrix file (decode_row's default wording);
    the exception type and the conditions that trigger it are
    unchanged."""
    return decode_row(data, num, path, kind=".spk file")


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
    if columns <= 0 or columns > MAT_COLMAX:
        raise ParseError(
            f"Invalid channel count {columns} in .spk file (must be 1-{MAT_COLMAX}): {path}"
        )

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
    try:
        return np.array(values, dtype=np.int64)
    except OverflowError as exc:
        raise ParseError(f"Decoded channel value out of range in .spk file: {path}") from exc


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
        # The non-floating branch below can't hit this: integer dtypes
        # narrower than int64 can't produce out-of-range values on a
        # same-or-widening cast.
        rounded = np.round(channels)
        return checked_round_to_int64(
            rounded,
            lambda: ParseError(f"File contains an out-of-range channel value in .spk file: {path}"),
        )
    return channels.astype(np.int64)


def save_spk(path: str, data) -> None:
    """Writes `data` as a single-spectrum, LC2-compressed .spk file --
    the modern MAT_LC format, version 2, the only writable .spk variant
    (see design spec). Layout ported from libmfile's own new-file
    behavior (lc_minfo.c's init_lci/lc_flush, lc_getput.c's writeline):
    a 44-byte header, one 8-byte position/length table entry right
    after it, then the LC2-compressed payload."""
    values = [int(v) for v in data]
    columns = len(values)
    if not (1 <= columns <= MAT_COLMAX):
        raise ValueError(
            f"Channel count {columns} is out of range for a .spk file (must be 1-{MAT_COLMAX})"
        )
    compressed = _lc2_compress(values)

    poslentablepos = LC_HEADER_SIZE
    data_pos = poslentablepos + LC_POSLEN_SIZE
    freepos = data_pos + len(compressed)

    header = struct.pack(
        "<11I", LC_MAGIC, 2, 1, 1, columns, poslentablepos, freepos, 0, 0, 0, 0,
    )
    poslen = struct.pack("<2I", data_pos, len(compressed))

    with open(path, "wb") as f:
        f.write(header)
        f.write(poslen)
        f.write(compressed)
