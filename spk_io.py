import struct

import numpy as np

from histogram_io import ParseError

LC_MAGIC = 0x80FFFF10
LC_HEADER_SIZE = 44
LC_POSLEN_SIZE = 8
MAT_COLMAX = 1 << 16  # matches libmfile-1.0.7's own MAT_COLMAX buffer-size limit

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


def _zigzag_decode(i: int) -> int:
    if i & 1:
        return -((i >> 1) + 1)
    return i >> 1


def _zigzag_encode(value: int) -> int:
    """Inverse of _zigzag_decode above. value can be any integer
    (positive, negative, or zero)."""
    return 2 * value if value >= 0 else -2 * value - 1


def _put_tag_n(tag_base: int, value: int) -> bytes:
    """Ported from libmfile's put_tag_n macro (lc_c2.c:39-53): encodes a
    non-negative `value` as a single tag byte (tag_base + value) when
    value <= 59, or a tag byte (tag_base + 60 + extra_byte_count) plus
    1-4 little-endian-ish extension bytes for larger values. Shared by
    single-value tags (tag_base=0x80) and same-run tags (tag_base=0xC0)."""
    if value <= 59:
        return bytes([tag_base + value])
    t = value - 60
    extension = [t & 0xFF]
    extra = 0
    t >>= 8
    while t:
        t -= 1
        extension.append(t & 0xFF)
        extra += 1
        t >>= 8
    return bytes([tag_base + 60 + extra]) + bytes(extension)


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
        return np.round(channels).astype(np.int64)
    return channels.astype(np.int64)
