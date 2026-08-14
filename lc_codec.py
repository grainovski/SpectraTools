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
    added to fix."""
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
                        raise ParseError(f"{kind}: same-run tag overruns row: {path}")
                    values.extend([last] * same)
                else:
                    last = last + zigzag_decode(n)
                    values.append(last)
                nleft -= 1

            elif t & 0x40:
                nleft -= 2
                if nleft < 0:
                    raise ParseError(f"{kind}: 2-value pack overruns row: {path}")
                a = t & 0x7
                b = (t >> 3) & 0x7
                values.append(last + zigzag_decode(a))
                last = last + zigzag_decode(b)
                values.append(last)

            else:
                nleft -= 3
                if nleft < 0:
                    raise ParseError(f"{kind}: 3-value pack overruns row: {path}")
                a = t & 0x3
                b = (t >> 2) & 0x3
                c = (t >> 4) & 0x3
                values.append(last + zigzag_decode(a))
                values.append(last + zigzag_decode(b))
                last = last + zigzag_decode(c)
                values.append(last)
    except IndexError as exc:
        raise ParseError(f"{kind}: row data ends mid-tag: {path}") from exc

    return values
