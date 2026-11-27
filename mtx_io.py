import struct

import numpy as np

from histogram_io import ParseError
from lc_codec import DIM_MAX, INT32_MAX, INT32_MIN, decode_row, encode_row

MAGIC_LC = 0x80FFFF10
_HEADER_FORMAT = "<11I"
_HEADER_SIZE = 44
_POSLEN_SIZE = 8  # one {u_int pos, u_int len} row-table entry, lc_minfo.h


def load_mtx(path, progress=None):
    """Reads level 0 of an lc-format (line-compressed) TV matrix file,
    returning a (lines, columns) int64 array. (The writer is save_mtx,
    below, added in 7.0.0.) Negative values (e.g. from
    random-coincidence subtraction) are preserved as-is throughout.
    Always reads level 0 and never exposes level selection -- matching
    TV's own matrix-open command, which has no level argument at all.

    `progress`, if given, is called as progress(rows_done, rows_total)
    roughly every 256 rows and once at the end. It exists so a caller
    running this off the GUI thread can show real progress rather than a
    spinner: the decode is several seconds of CPU that cannot be
    vectorised (see lc_codec.decode_row). Called at a coarse interval --
    32 times for a real 8192-line matrix -- so it costs nothing
    measurable, and it must not raise: this function makes no attempt to
    unwind a partially built matrix."""
    try:
        with open(path, "rb") as f:
            # Whole file in ONE read, then sliced below, rather than a
            # seek+read per row. The row loop used to issue one pair per
            # line -- 8194 syscalls for a real 8192-line matrix -- which
            # is invisible on a local disk but brutal across a network
            # or virtualised filesystem, where per-operation latency
            # dominates and bulk throughput does not predict it at all:
            # measured on WSL with the file on the Windows drive (9p),
            # those reads cost 1.63 s versus 0.02 s for the identical
            # file on the same machine's native filesystem, taking the
            # whole load from 5.2 s to 10.5 s.
            #
            # The buffer is dwarfed by what this function already
            # allocates -- a 31 MB file decodes into a 536 MB int64
            # array -- so holding the bytes costs nothing in relative
            # terms.
            blob = f.read()
            if len(blob) < _HEADER_SIZE:
                raise ParseError(f"File too small to be an lc-format matrix: {path}")
            header_bytes = blob[:_HEADER_SIZE]
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
            table_bytes = blob[poslentablepos:poslentablepos + lines * 8]
            if len(table_bytes) < lines * 8:
                raise ParseError(f"lc matrix file's row table is truncated: {path}")
            row_table = struct.unpack(f"<{lines * 2}I", table_bytes)

            data = np.zeros((lines, columns), dtype=np.int64)
            for row in range(lines):
                if progress is not None and (row & 0xFF) == 0:
                    progress(row, lines)
                row_pos = row_table[row * 2]
                row_len = row_table[row * 2 + 1]
                if row_len == 0:
                    continue  # never written -- treated as all-zero
                # Explicit bounds check before slicing. f.read(n) used to
                # short-read a truncated file and the length test below
                # caught it; a slice past the end returns silently short
                # (or empty, for a start offset past EOF) instead, so
                # this has to be checked rather than inferred.
                row_bytes = blob[row_pos:row_pos + row_len]
                if len(row_bytes) < row_len:
                    raise ParseError(f"lc matrix file row {row} data is truncated: {path}")
                values = decode_row(row_bytes, columns, path, kind="lc matrix file")
                try:
                    # np.fromiter, not a plain `data[row, :] = values`.
                    # Assigning a Python list into an int64 row makes
                    # numpy walk it as a generic sequence of objects;
                    # fromiter with an explicit dtype and count takes a
                    # tighter path. Measured 183 -> 137 us per 8192-wide
                    # row, i.e. ~1.50 s -> ~1.12 s across a real matrix.
                    #
                    # Deliberately NOT decoding straight into a typed
                    # buffer (array('q')) instead of a list: writing into
                    # one during the decode loop is slower per element
                    # than a list store, and that loop is 68% of the
                    # runtime against this step's 30% -- it would lose
                    # more than it saves. Converting the finished list to
                    # array('q') and using frombuffer measures the same
                    # 137 us as fromiter, so it buys nothing over the
                    # simpler call.
                    #
                    # The out-of-range guard below is unaffected:
                    # fromiter raises OverflowError on a value too large
                    # for int64, exactly as slice assignment did.
                    decoded = np.fromiter(values, dtype=np.int64, count=columns)
                except OverflowError as exc:
                    raise ParseError(
                        f"lc matrix file row {row} contains an out-of-range value: {path}"
                    ) from exc
                # The format's cells are C ints and libmfile's decoder sums
                # deltas in int arithmetic (lc_c2.c:162), so a delta that
                # lc2_compress stored WRAPPED -- INT32_MAX next to
                # INT32_MIN is written as +1 -- only decodes right modulo
                # 2**32. decode_row works in unbounded Python ints, so wrap
                # here, exactly as TV would read the cell. Two reductions
                # per row, about 1% of the decode; a row that never leaves
                # int32 (every real one seen so far) is left untouched.
                if decoded.max() > INT32_MAX or decoded.min() < INT32_MIN:
                    decoded = ((decoded - INT32_MIN) & 0xFFFFFFFF) + INT32_MIN
                data[row, :] = decoded
    except OSError as exc:
        raise ParseError(f"Could not read {path}: {exc}") from exc

    if progress is not None:
        progress(lines, lines)
    return data


def coalesce_sparse(shape, flat_indices, values):
    """Validates a sparse matrix given as (shape, flat_indices, values)
    and returns (ny, nx, indices, values): indices sorted ascending and
    unique, duplicates summed (the usual COO convention, so a sorter may
    hand over one entry per event), and explicit zeros dropped.

    `shape` is (ny, nx) and a flat index is y * nx + x -- the app's
    [y][x] order, rows first, i.e. numpy's own ravel order. Values keep
    their dtype (integers for save_mtx; floats are fine for the JSON
    writer). Shared by save_mtx and json_matrix_io.save_json_matrix so
    both read the sparse form identically. Raises ValueError."""
    try:
        ny, nx = (int(n) for n in shape)
    except (TypeError, ValueError):
        raise ValueError(f"sparse matrix shape must be (lines, columns), got {shape!r}")
    if ny < 1 or nx < 1:
        raise ValueError(f"sparse matrix shape must be at least 1 x 1, got {ny} x {nx}")
    indices = np.asarray(flat_indices)
    vals = np.asarray(values)
    if indices.ndim != 1 or vals.ndim != 1 or indices.size != vals.size:
        raise ValueError(
            "sparse matrix indices and values must be two 1-D arrays of the "
            f"same length (got shapes {indices.shape} and {vals.shape})"
        )
    if indices.size and indices.dtype.kind not in "iu":
        raise ValueError("sparse matrix indices must be integers")
    indices = indices.astype(np.int64, copy=False)
    if indices.size and (indices.min() < 0 or indices.max() >= ny * nx):
        raise ValueError(
            f"sparse matrix index out of range for a {ny} x {nx} matrix "
            f"(valid flat indices are 0..{ny * nx - 1})"
        )
    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    vals = vals[order]
    if indices.size > 1 and np.any(indices[1:] == indices[:-1]):
        starts = np.flatnonzero(np.r_[True, indices[1:] != indices[:-1]])
        if vals.dtype.kind in "iu":
            # int64 sums: int32-range counts cannot overflow at any
            # realistic multiplicity, and save_mtx range-checks the result.
            vals = np.add.reduceat(vals.astype(np.int64), starts)
        else:
            vals = np.add.reduceat(vals, starts)
        indices = indices[starts]
    keep = vals != 0
    return ny, nx, indices[keep], vals[keep]


def _check_dims(lines, columns):
    if not (1 <= lines <= DIM_MAX) or not (1 <= columns <= DIM_MAX):
        raise ValueError(
            f"A {lines} x {columns} matrix cannot be written as .mtx: each "
            f"dimension must be 1-{DIM_MAX} (libmfile's MAT_COLMAX)."
        )


def _check_int32(lo, hi):
    if lo < INT32_MIN or hi > INT32_MAX:
        raise ValueError(
            f"Counts from {lo} to {hi} do not fit a .mtx file, whose cells "
            f"are 32-bit integers ({INT32_MIN} to {INT32_MAX})."
        )


def _require_integer_dtype(arr):
    if arr.dtype.kind not in "iu":
        raise ValueError(
            f"Matrix values must be integers to be written as .mtx (got "
            f"{arr.dtype}); the format stores whole counts only. Round "
            "first -- the sorting code uses seeded stochastic rounding so "
            "each cell keeps its expected value."
        )


def save_mtx(path, matrix, progress=None):
    """Writes `matrix` as an lc-format (line-compressed) matrix, version 2,
    one level -- the file libmfile itself writes for a 1-level matrix when
    every line is put once, in order, with a full-width mputint. Verified
    byte for byte against the vendored libmfile-1.0.7 built from source
    (tests/fixtures/libmfile_written_7x9.mtx and the test beside it).

    `matrix` is either a dense 2-D integer array indexed [y][x] (rows =
    lines = Y, the app's convention), or the sparse triple
    (shape, flat_indices, values) described in coalesce_sparse. The sparse
    form is written row by row from one reusable row buffer, so a
    16384 x 16384 matrix never exists densely here.

    Cells are C ints in the format, so values must be integers within
    int32; anything else is refused with ValueError rather than silently
    rounded or wrapped (rounding is the caller's decision -- see the 7.0.0
    plan's seeded stochastic rounding).

    Layout, following libmfile:
      * header, 11 little-endian u_ints (lc_minfo.h lc_header), filled
        exactly as lc_flush does (lc_minfo.c:194-206): magic 0x80FFFF10,
        version 2, levels 1, lines, columns, poslentablepos, freepos,
        freelistpos, then used = free = status = 0 (libmfile's own
        comment: "not yet implemented"; always written 0).
      * the row table straight after the header: a new file gets
        poslentablepos = sizeof(lc_header) = 44, its first free byte at
        44 + levels * lines * 8, and freelistpos = 0 (init_lci,
        lc_minfo.c:107-109). Nothing in libmfile ever sets a free list,
        so freelistpos stays 0.
      * each row's compressed bytes appended at freepos in line order
        (writeline, lc_getput.c:111-128: a new line's pos/len are 0, so
        it lands at fp and fp advances by its length); the final freepos
        is the file size.
    An all-zero row is written like any other, as its compressed form
    (one same-run tag, 1-3 bytes), never as a zero-length "unwritten"
    entry: writeline compresses whatever buffer it is given and lc2
    always yields at least one byte, so libmfile only leaves len = 0 for
    a line nobody put. Both read back as zeros (readline returns early
    on len 0, lc_getput.c:89), but only this matches libmfile's bytes.

    `progress`, if given, is called as progress(rows_done, rows_total)
    every 256 rows and once at the end, the same contract as load_mtx's:
    compressing is pure Python per tag (about 30 s for 10**7 scattered
    cells at 16384 x 16384), so a GUI caller wants real progress.
    """
    sparse = isinstance(matrix, tuple)
    if sparse:
        if len(matrix) != 3:
            raise ValueError(
                "a sparse matrix must be given as (shape, flat_indices, values)"
            )
        shape, flat_indices, values = matrix
        vals_in = np.asarray(values)
        if vals_in.size:
            _require_integer_dtype(vals_in)
        lines, columns, indices, vals = coalesce_sparse(shape, flat_indices, vals_in)
        _check_dims(lines, columns)
        if vals.size:
            _check_int32(int(vals.min()), int(vals.max()))
        rows_of = indices // columns
        cols_of = indices - rows_of * columns
        # Row r's entries are the slice row_start[r]:row_start[r + 1]
        # (indices are sorted, so each row's entries are contiguous).
        row_start = np.searchsorted(rows_of, np.arange(lines + 1), side="left")
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            raise ValueError(
                f"a matrix must be 2-D to be written as .mtx (got {dense.ndim}-D)"
            )
        lines, columns = dense.shape
        _check_dims(lines, columns)
        _require_integer_dtype(dense)
        _check_int32(int(dense.min()), int(dense.max()))

    table = np.zeros((lines, 2), dtype="<u4")
    pos = _HEADER_SIZE + lines * _POSLEN_SIZE
    zero_row = None
    row_buf = np.zeros(columns, dtype=np.int64) if sparse else None

    with open(path, "wb") as f:
        # Placeholder header and table, rewritten once the row positions
        # are known -- the same order libmfile uses (rows as they are put,
        # header and table at lc_flush).
        f.write(b"\0" * pos)
        for row in range(lines):
            if progress is not None and (row & 0xFF) == 0:
                progress(row, lines)
            if sparse:
                a, b = row_start[row], row_start[row + 1]
                if a == b:
                    if zero_row is None:
                        zero_row = encode_row([0] * columns, wrap32=True)
                    encoded = zero_row
                else:
                    row_buf[cols_of[a:b]] = vals[a:b]
                    encoded = encode_row(row_buf, wrap32=True)
                    row_buf[cols_of[a:b]] = 0
            else:
                encoded = encode_row(dense[row], wrap32=True)
            if pos + len(encoded) > 0xFFFFFFFF:
                raise ValueError(
                    "the compressed matrix exceeds 4 GiB, beyond the 32-bit "
                    "row positions of the .mtx format"
                )
            f.write(encoded)
            table[row] = (pos, len(encoded))
            pos += len(encoded)
        header = struct.pack(
            _HEADER_FORMAT, MAGIC_LC, 2, 1, lines, columns,
            _HEADER_SIZE, pos, 0, 0, 0, 0,
        )
        f.seek(0)
        f.write(header)
        f.write(table.tobytes())
    if progress is not None:
        progress(lines, lines)
