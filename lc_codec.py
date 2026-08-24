"""Shared LC2 bit-level codec, extracted from spk_io.py and mtx_io.py --
both files independently implemented this same tag decoding for the
"lc" compressed-integer format (libmfile's lc2_uncompress, lc_c2.c),
one for single-spectrum .spk files, the other for matrix rows. This
module is the single canonical implementation; both callers delegate
to it."""

from histogram_io import ParseError

DIM_MAX = 1 << 16  # matches libmfile-1.0.7's own MAT_COLMAX buffer-size limit


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
