"""Shared LC2 bit-level codec, extracted from spk_io.py and mtx_io.py --
both files independently implemented this same tag decoding for the
"lc" compressed-integer format (libmfile's lc2_uncompress, lc_c2.c),
one for single-spectrum .spk files, the other for matrix rows. This
module is the single canonical implementation; both callers delegate
to it.

The encoder (encode_row, a port of lc2_compress) lives here too since
7.0.0: save_spk wrote single spectra with it from spk_io, and save_mtx
needs the identical compressor for every matrix row. One copy, so a fix
to one writer cannot leave the other behind."""

from bisect import bisect_right

import numpy as np

from histogram_io import ParseError

DIM_MAX = 1 << 16  # matches libmfile-1.0.7's own MAT_COLMAX buffer-size limit

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
_UINT32_MASK = 0xFFFFFFFF


def zigzag_decode(n):
    return -((n >> 1) + 1) if (n & 1) else (n >> 1)


# Precomputed zigzag_decode(0..63): covers every single-value tag's
# `n` below the extension threshold (60, the common case) and, as a
# subset, every 2-bit/3-bit pack-tag field. A list-index beats calling
# zigzag_decode() or re-evaluating its ternary inline millions of
# times per row -- see decode_row's docstring for why this matters.
_ZIGZAG_TABLE = [zigzag_decode(i) for i in range(64)]


def decode_row(data, num_values, path, *, kind):
    """Decodes one lc-format v2 compressed row into `num_values`
    integers. A faithful port of lc2_uncompress
    (libmfile-1.0.7/src/lc_c2.c:134-200), including its two least
    obvious behaviors: a 3-pack/2-pack tag's deltas are all computed
    against the SAME pre-tag `last` (not chained value-to-value), and
    a same-run tag's repeated values equal the pre-run `last`
    unchanged -- `last` is not updated by a same-run tag at all, only
    by the other three tag kinds. Verified byte-for-byte against real
    compressed rows from both fixture files during planning. `kind`
    is the leading noun phrase in error messages (e.g. "lc matrix
    file" for mtx_io.py, ".spk file" for spk_io.py) -- required and
    keyword-only so a future caller can't silently inherit another
    format's wording by omission, the exact bug this parameter was
    added to fix.

    Performance: _ZIGZAG_TABLE above (no per-value zigzag_decode()
    call for the common case) is what makes this fast -- worth ~18% on
    real 8192x8192 matrix fixtures, and essentially the whole win.

    Two things that sound faster and are NOT, both measured rather
    than reasoned about, so neither gets re-attempted for free:

    * A pre-sized output list (`values = [0] * num_values` with an
      index, instead of append()/extend()) shipped here from c9ffee6
      until 2026-08-24 on the assumption that it avoided growth
      overhead. A clean A/B on Python 3.13 -- same code, same rows,
      only that one difference -- put it 5-6% SLOWER than append()
      across three trials (ratios 1.058/1.062/1.050). CPython
      over-allocates on append and the amortised cost is below what
      the extra index bookkeeping costs in the loop. Removed.
    * A full numpy vectorization, verified byte-for-byte correct,
      measured no faster at all: the tag stream is variable-length,
      so reconstructing where each value lands needs a sequential
      scan that alone costs about what this whole function does.
      See commit c9ffee6's message.

    tests/test_mtx_io.py guards the table's contribution by timing
    this function against a frozen copy of the pre-c9ffee6 decoder in
    the same process and asserting a ratio, so the guard holds on any
    machine."""
    values = []
    last = 0
    pos = 0
    nleft = num_values
    zz = _ZIGZAG_TABLE
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
                        raise ParseError(f"{kind}: same-run tag overruns row: {path}")
                    values.extend([last] * same)
                else:
                    last += zz[n] if n < 64 else zigzag_decode(n)
                    values.append(last)
                nleft -= 1

            elif t & 0x40:
                nleft -= 2
                if nleft < 0:
                    raise ParseError(f"{kind}: 2-value pack overruns row: {path}")
                a = t & 0x7
                b = (t >> 3) & 0x7
                values.append(last + zz[a])
                last += zz[b]
                values.append(last)

            else:
                nleft -= 3
                if nleft < 0:
                    raise ParseError(f"{kind}: 3-value pack overruns row: {path}")
                a = t & 0x3
                b = (t >> 2) & 0x3
                c = (t >> 4) & 0x3
                values.append(last + zz[a])
                values.append(last + zz[b])
                last += zz[c]
                values.append(last)
    except IndexError as exc:
        raise ParseError(f"{kind}: row data ends mid-tag: {path}") from exc

    return values


def wrap_int32(value):
    """`value` reduced to a signed 32-bit int, the way C's int arithmetic
    wraps on the machines libmfile runs on. lc2_compress computes every
    delta as `*src - last` in plain int (lc_c2.c:68, :85) and
    lc2_uncompress adds it back into an int (lc_c2.c:162), so a delta
    between, say, INT32_MAX and INT32_MIN is stored WRAPPED -- as +1 --
    and only comes back right because the decoder wraps too."""
    return ((value - INT32_MIN) & _UINT32_MASK) + INT32_MIN


def zigzag_encode(value):
    """Inverse of zigzag_decode. `value` can be any integer (positive,
    negative, or zero). Same mapping as lc_c2.c:31's encode() macro:
    0, -1, 1, -2, ... -> 0, 1, 2, 3, ..."""
    return 2 * value if value >= 0 else -2 * value - 1


def put_tag_n(tag_base, value):
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
        if extra == 3:
            raise ValueError(
                f"value {value} is too large to encode in LC2's tag format "
                "(supports at most 4 extension bytes)"
            )
        t -= 1
        extension.append(t & 0xFF)
        extra += 1
        t >>= 8
    return bytes([tag_base + 60 + extra]) + bytes(extension)


def _fits_in_bits(value, bits):
    return (value >> bits) == 0


# How far encode_row scans a run of equal values one element at a time
# before it looks the run's end up instead. Short runs (the common case
# in a dense spectrum) never pay for the lookup; a long run -- the
# thousands of empty cells in a row of a sparse gamma-gamma matrix --
# costs one bisect instead of a Python step per cell. 16 is not tuned:
# any small constant gives the same bytes, only the speed differs.
_RUN_SCAN_LIMIT = 16


def _change_points(values):
    """Sorted indices b (1 <= b < len) where values[b] != values[b - 1]:
    the starts of every run of equal values after the first. Computed
    once per row, and only when a long run actually turns up."""
    try:
        arr = np.asarray(values, dtype=np.int64)
    except OverflowError:
        # Values beyond int64 (save_spk is handed plain Python ints and
        # must reach its own "too large" ValueError, not this one).
        return [b for b in range(1, len(values)) if values[b] != values[b - 1]]
    return (np.flatnonzero(arr[1:] != arr[:-1]) + 1).tolist()


def encode_row(values, *, wrap32=False):
    """Ported from libmfile's lc2_compress (lc_c2.c:63-126). `values` is
    a sequence of ints (one spectrum's channels, or one matrix row).
    Returns the LC2-compressed bytes, choosing the most compact of: a
    run of the previous value (>=4 elements), a 3-value pack (2 bits
    each, anchor-relative to the same running `last`), a 2-value pack
    (3 bits each), or a single value (with extension bytes for large
    deltas).

    `wrap32=True` reproduces the C exactly for int32 data: each delta is
    wrapped to a signed 32-bit int before it is classified and encoded,
    as lc_c2.c:68/:85 do implicitly. It only changes the output when two
    neighbouring values are more than 2**31 apart (e.g. INT32_MIN next
    to a positive count), and then it is what makes the bytes identical
    to libmfile's and decodable by TV. save_mtx uses it; the caller must
    have checked every value is within int32. The default (no wrapping)
    is save_spk's long-standing behaviour, kept unchanged: there an
    out-of-range delta raises ValueError from put_tag_n instead.

    Long runs of one value are measured with a bisect over the row's
    change points instead of a per-element loop (see _RUN_SCAN_LIMIT);
    the bytes are identical either way, which the demo.spk re-encode
    test and the gg.mtx re-encode test both pin. A numpy row is accepted
    as well as a list, and is the faster input: the change points then
    come straight from the array instead of converting the list back."""
    array = values if isinstance(values, np.ndarray) else None
    if array is not None:
        values = array.tolist()  # Python ints: fast scalar access below
    out = bytearray()
    last = 0
    n = len(values)
    idx = 0
    bounds = None

    def delta(v):
        d = v - last
        if wrap32 and not (INT32_MIN <= d <= INT32_MAX):
            d = wrap_int32(d)
        return d

    while idx < n:
        remaining = n - idx
        d = delta(values[idx])
        i = 1
        if 0 <= d < 2:  # lc_c2.c:73 -- `unsigned d < 2`
            while i < remaining and i < _RUN_SCAN_LIMIT and values[idx + i] == last:
                i += 1
            if i == _RUN_SCAN_LIMIT and i < remaining and values[idx + i] == last:
                if bounds is None:
                    bounds = _change_points(values if array is None else array)
                j = bisect_right(bounds, idx + i)
                i = (bounds[j] if j < len(bounds) else n) - idx
        same = i - 1

        if same >= 3:
            out += put_tag_n(0xC0, ((same - 3) << 1) + d)
            idx += i
            continue

        s0 = values[idx]
        a = zigzag_encode(d)

        if _fits_in_bits(a, 3) and remaining >= 2:
            s1 = values[idx + 1]
            b = zigzag_encode(delta(s1))

            if _fits_in_bits(a | b, 2) and remaining >= 3:
                s2 = values[idx + 2]
                c = zigzag_encode(delta(s2))
                if _fits_in_bits(c, 2):
                    out.append(a + (b << 2) + (c << 4))
                    idx += 3
                    last = s2
                    continue

            if _fits_in_bits(b, 3):
                out.append(0x40 + a + (b << 3))
                idx += 2
                last = s1
                continue

        out += put_tag_n(0x80, a)
        idx += 1
        last = s0

    return bytes(out)
