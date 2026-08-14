import struct

import numpy as np

from histogram_io import ParseError
from lc_codec import DIM_MAX, decode_row

MAGIC_LC = 0x80FFFF10
_HEADER_FORMAT = "<11I"
_HEADER_SIZE = 44


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
            if not (1 <= lines <= DIM_MAX) or not (1 <= columns <= DIM_MAX):
                raise ParseError(
                    f"Invalid matrix dimensions {lines}x{columns} "
                    f"(must be 1-{DIM_MAX} each): {path}"
                )

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
                values = decode_row(row_bytes, columns, path)
                try:
                    data[row, :] = values
                except OverflowError as exc:
                    raise ParseError(
                        f"lc matrix file row {row} contains an out-of-range value: {path}"
                    ) from exc
    except OSError as exc:
        raise ParseError(f"Could not read {path}: {exc}") from exc

    return data
