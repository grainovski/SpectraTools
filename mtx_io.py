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
