"""Gamma-gamma matrices as JSON: CAEN CoMPASS's export, and the files
SpectraTools itself writes since 7.0.0.

**CoMPASS files.** The file is one JSON object. Only its ``data`` member
is read: a rectangular 2-D array of counts.

**Orientation.** CoMPASS writes ``data[x][y]`` -- each row of the array is
one X bin. This app indexes matrices ``[y][x]`` (rows are Y; see
root_io._extract, which faces the same choice for uproot), so the array is
transposed on the way in. That was established by comparing a JSON export
cell for cell with the ROOT file CoMPASS wrote for the same run, whose axes
are labelled: read as [x][y] all 1,048,576 cells agree, read the other way
228,998 counts move. A swap raises no error on a square matrix -- every cut
would simply land on the other detector -- so the comparison is a test
(tests/test_json_matrix_io.py), not just this note.

**What is deliberately ignored.** The file also carries ``calibration``,
``UoM`` and per-axis ``limits``. They do not describe the bins: with them
the Co-60 pair lands near 388 and 445 keV instead of 1173 and 1332, the
``limits`` imply 5.5 ADC channels per bin where the ROOT file of the same
run has 16, and one calibration cannot serve two detectors with different
gains anyway. So, exactly as for ROOT matrices, the panel works in channels
and projections are calibrated in the app.

The JSON's array is one bin longer per axis than the ROOT histogram's
in-range bins; that last row and column correspond to ROOT's overflow bin.
They are kept rather than trimmed, so no count in the file is dropped.

**SpectraTools files (7.0.0).** A random-subtracted matrix holds negative
and fractional counts, which a CoMPASS file never does, and at the
digitizer's full 16384 x 16384 a dense array would be gigabytes of text.
So save_json_matrix writes one of two layouts (the plan's option (c)):

* **sparse**, any size, read by SpectraTools only::

      {"format": "sparse", "shape": [ny, nx], "spectratools": {...},
       "x": [...], "y": [...], "v": [...]}

  one (x, y, v) triple per non-zero cell, ``v`` the exact value;
* **dense CoMPASS layout** -- ``data[x][y]`` plus CoMPASS's ``title``,
  ``xaxis`` and ``yaxis`` -- only up to 4096 bins per axis, i.e. for a
  matrix compressed by a factor of 4 or more from 16384.

Both carry a ``spectratools`` metadata block. Its presence is what tells
the reader the file is ours: only then are negative and fractional values
accepted. A CoMPASS file (no block) keeps the strict whole-non-negative
check unchanged. The block's ``compression`` factor gives the panel the
right ChannelAxis (axis_for_compression), so a peak in a compressed matrix
reads at the same ADC channel as in the 16k singles
([[feedback_hdtv_axis_convention_for_2d]]).
"""

import json
import math

import numpy as np

from channel_axis import ChannelAxis
from histogram_io import ParseError
from int64_cast import checked_round_to_int64
from mtx_io import coalesce_sparse

META_KEY = "spectratools"

# What every SpectraTools matrix file records about how it was made, so a
# result can be traced back and reproduced. `axes` is filled in if the
# caller leaves it out; the rest must be supplied.
REQUIRED_META_KEYS = ("compression", "source_run", "windows", "w", "seed",
                      "gain_coefficients")
DEFAULT_AXES = {"x": "CH1", "y": "CH0"}

# The largest dense CoMPASS-layout matrix this writes: 4096 x 4096 is the
# 16384-channel digitizer compressed by 4, already ~100 MB of JSON text.
DENSE_MAX = 4096

# Values per json.dumps call while streaming the big arrays, so a file of
# 10**7 cells is written without first holding 10**7 Python objects.
_CHUNK = 65536


def axis_for_compression(factor):
    """The ChannelAxis of a matrix compressed by `factor`, or None for 1
    (one bin per ADC channel, the identity axis).

    Compression sums factor x factor cells, so bin b holds the integer ADC
    channels f*b .. f*b + f - 1 and its channel is their mean,
    f*b + (f - 1)/2 -- channel_axis.from_bin_edges's integer-mean rule for
    the edges [f*b, f*b + f), which is what makes a peak land on the same
    channel as in the full-resolution singles."""
    f = _check_factor(factor)
    if f == 1:
        return None
    return ChannelAxis(offset=(f - 1) / 2.0, width=float(f))


def _check_factor(factor):
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError(f"compression factor must be a positive integer, got {factor!r}")
    return int(factor)


def compress_matrix(matrix, factor):
    """Sums `factor` x `factor` blocks of a [y][x] matrix (dense 2-D array,
    or the (shape, flat_indices, values) sparse triple of
    mtx_io.coalesce_sparse) into a dense array of shape (ny/f, nx/f).

    The factor must divide both dimensions: a ragged last bin would hold
    fewer channels and sit off the axis axis_for_compression describes.
    Integer input stays int64, anything else becomes float64. The sparse
    form never builds the full-resolution matrix."""
    f = _check_factor(factor)
    if isinstance(matrix, tuple):
        if len(matrix) != 3:
            raise ValueError("a sparse matrix must be given as (shape, flat_indices, values)")
        ny, nx, idx, vals = coalesce_sparse(*matrix)
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2:
            raise ValueError(f"a matrix must be 2-D (got {dense.ndim}-D)")
        ny, nx = dense.shape
    if ny % f or nx % f:
        raise ValueError(
            f"compression factor {f} does not divide the {ny} x {nx} matrix"
        )
    cy, cx = ny // f, nx // f
    if not isinstance(matrix, tuple):
        out = dense.reshape(cy, f, cx, f)
        out = out.astype(np.int64 if dense.dtype.kind in "iub" else np.float64)
        return out.sum(axis=(1, 3))
    integer = vals.dtype.kind in "iu"
    target = (idx // nx // f) * cx + (idx % nx) // f
    out = np.zeros(cy * cx, dtype=np.int64 if integer else np.float64)
    np.add.at(out, target, vals.astype(out.dtype, copy=False))
    return out.reshape(cy, cx)


# --- writing --------------------------------------------------------------


def _check_meta(meta):
    if not isinstance(meta, dict):
        raise ValueError("matrix metadata must be a dict")
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise ValueError(
            "matrix metadata is missing " + ", ".join(missing)
            + f" (required: {', '.join(REQUIRED_META_KEYS)})"
        )
    out = dict(meta)
    out["compression"] = _check_factor(out["compression"])
    out.setdefault("axes", dict(DEFAULT_AXES))
    try:
        json.dumps(out, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"matrix metadata cannot be stored as JSON: {exc}") from exc
    return out


def _value_kind(values):
    """'int' if every value can be written as a JSON integer without loss,
    else 'float'. Refuses booleans, non-numbers and non-finite values."""
    if values.dtype.kind in "iu":
        return "int"
    if values.dtype.kind != "f":
        raise ValueError(
            f"matrix values must be numbers (got {values.dtype}); "
            "booleans and other types are not counts"
        )
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError("matrix values must be finite (no NaN or infinity)")
    if values.size == 0:
        return "int"
    whole = np.all(values == np.rint(values)) and np.max(np.abs(values)) < 2.0**53
    return "int" if whole else "float"


def _json_numbers(values, kind):
    """Comma-separated JSON numbers for a 1-D array, exact: integers as
    integers, floats via Python's shortest round-trip repr (json's own)."""
    if kind == "int":
        values = values.astype(np.int64, copy=False)
    else:
        values = values.astype(np.float64, copy=False)
    return json.dumps(values.tolist(), separators=(",", ":"), allow_nan=False)[1:-1]


def _write_array(fh, values, kind):
    fh.write("[")
    for start in range(0, values.size, _CHUNK):
        if start:
            fh.write(",")
        fh.write(_json_numbers(values[start:start + _CHUNK], kind))
    fh.write("]")


def save_json_matrix(path, matrix, meta, layout="sparse"):
    """Writes a [y][x] matrix (dense 2-D array, integer or float, or the
    sparse triple (shape, flat_indices, values) of mtx_io.coalesce_sparse)
    as JSON, with `meta` stored as the ``spectratools`` block.

    layout="sparse" (default): the sparse layout, any size, streamed
    from the non-zero cells only -- a 16384 x 16384 sparse input is never
    made dense. x is the column
    (CH1), y the row (CH0), v the exact value.

    layout="dense": the CoMPASS layout, data[x][y] (the app's [y][x]
    transposed back, the exact inverse of load_json_matrix), refused above
    DENSE_MAX bins per axis with a message pointing at the sparse layout.

    `meta` must hold REQUIRED_META_KEYS; `compression` is the factor the
    matrix was compressed by (1 = full resolution), from which the reader
    rebuilds the channel axes. Raises ValueError for bad input."""
    meta = _check_meta(meta)
    if layout not in ("sparse", "dense"):
        raise ValueError(f'layout must be "sparse" or "dense", got {layout!r}')

    if isinstance(matrix, tuple):
        if len(matrix) != 3:
            raise ValueError("a sparse matrix must be given as (shape, flat_indices, values)")
        ny, nx, idx, vals = coalesce_sparse(*matrix)
        dense = None
    else:
        dense = np.asarray(matrix)
        if dense.ndim != 2 or dense.size == 0:
            raise ValueError(f"a matrix must be a non-empty 2-D array (got shape {dense.shape})")
        ny, nx = dense.shape

    if layout == "dense":
        if ny > DENSE_MAX or nx > DENSE_MAX:
            raise ValueError(
                f"A {ny} x {nx} matrix is too large for the dense CoMPASS JSON "
                f"layout (at most {DENSE_MAX} bins per axis). Compress it by a "
                f"factor of {math.ceil(max(ny, nx) / DENSE_MAX)} or more, or "
                'save it with layout="sparse", which has no size limit.'
            )
        if dense is None:
            dense = np.zeros(ny * nx, dtype=vals.dtype)
            dense[idx] = vals
            dense = dense.reshape(ny, nx)
        kind = _value_kind(dense)
        f = meta["compression"]
        document_head = {
            "title": meta.get("title", "SpectraTools gamma-gamma matrix"),
            "xaxis": {"label": "ADC channel " + str(meta["axes"].get("x", "")).strip(),
                      "limits": [0, nx * f], "nBins": nx},
            "yaxis": {"label": "ADC channel " + str(meta["axes"].get("y", "")).strip(),
                      "limits": [0, ny * f], "nBins": ny},
            META_KEY: meta,
        }
        transposed = np.ascontiguousarray(dense.T)      # CoMPASS rows are X
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(document_head)[:-1])
            fh.write(', "data": [')
            for x in range(nx):
                if x:
                    fh.write(",")
                fh.write("[" + _json_numbers(transposed[x], kind) + "]")
            fh.write("]}")
        return

    if dense is not None:
        idx = np.flatnonzero(dense)
        vals = dense.ravel()[idx]
    kind = _value_kind(vals)
    head = {"format": "sparse", "shape": [ny, nx], META_KEY: meta}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(head)[:-1])
        fh.write(', "x": ')
        _write_array(fh, idx % nx, "int")
        fh.write(', "y": ')
        _write_array(fh, idx // nx, "int")
        fh.write(', "v": ')
        _write_array(fh, vals, kind)
        fh.write("}")


# --- reading --------------------------------------------------------------


def load_json_matrix(path):
    """Read a JSON matrix as an array indexed [y][x].

    A CoMPASS file (no ``spectratools`` block) gives int64 and raises
    ParseError, naming the file, for anything that is not a rectangular
    2-D array of finite, non-negative whole-number counts. A SpectraTools
    file may hold negative and fractional values; see
    load_json_matrix_with_meta for its dtype and for the channel axes.
    """
    matrix, _meta, _axes = load_json_matrix_with_meta(path)
    return matrix


def load_json_matrix_with_meta(path):
    """Read a JSON matrix of either layout. Returns (matrix, meta, axes):

    * matrix -- indexed [y][x], C-contiguous. int64 when every value is a
      whole number; float64 when a SpectraTools file holds fractional
      values (a random-subtracted matrix);
    * meta -- the ``spectratools`` block as a dict, or None for a CoMPASS
      file;
    * axes -- (x_axis, y_axis) as MatrixPanel takes them: the ChannelAxis
      for the file's compression factor, or (None, None) at full
      resolution and for every CoMPASS file (whose own limits are ignored,
      see the module docstring).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            document = json.load(fh)
    except OSError as exc:
        raise ParseError(f"Could not read {path}: {exc}") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        # json.JSONDecodeError is a ValueError.
        raise ParseError(f"File is not valid JSON ({exc}): {path}") from exc

    meta = document.get(META_KEY) if isinstance(document, dict) else None
    if meta is not None and not isinstance(meta, dict):
        raise ParseError(f'JSON matrix\'s "{META_KEY}" block is not an object: {path}')
    axes = (None, None)
    if meta is not None:
        try:
            axis = axis_for_compression(meta.get("compression", 1))
        except ValueError as exc:
            raise ParseError(f"JSON matrix metadata: {exc}: {path}") from exc
        axes = (axis, axis)

    if isinstance(document, dict) and document.get("format") == "sparse":
        values = _sparse_from_document(document, path)
    else:
        values = _dense_from_document(document, path)

    if meta is None:
        return _strict_counts(values, path), None, axes
    return _signed_values(values, path), meta, axes


def _sparse_from_document(document, path):
    """The sparse layout as a dense float64 [y][x] array (values checked
    later, like the dense layout's)."""
    try:
        ny, nx = (int(n) for n in document["shape"])
        xs = np.asarray(document["x"])
        ys = np.asarray(document["y"])
        vs = np.asarray(document["v"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ParseError(
            f'Sparse JSON matrix needs "shape" [ny, nx] and "x", "y", "v" '
            f"arrays: {path}"
        ) from exc
    if ny < 1 or nx < 1:
        raise ParseError(f"Sparse JSON matrix has an empty shape {ny} x {nx}: {path}")
    if not (xs.ndim == ys.ndim == vs.ndim == 1 and xs.size == ys.size == vs.size):
        raise ParseError(
            f'Sparse JSON matrix\'s "x", "y" and "v" must be flat arrays of '
            f"the same length: {path}"
        )
    if xs.size and (xs.dtype.kind not in "iu" or ys.dtype.kind not in "iu"):
        raise ParseError(f"Sparse JSON matrix holds non-integer coordinates: {path}")
    if xs.size and (xs.min() < 0 or xs.max() >= nx or ys.min() < 0 or ys.max() >= ny):
        raise ParseError(
            f"Sparse JSON matrix holds a coordinate outside its {ny} x {nx} shape: {path}"
        )
    if vs.size and vs.dtype.kind not in "iuf" or any(v is True or v is False for v in document["v"]):
        raise ParseError(f"JSON matrix holds entries that are not plain numbers: {path}")
    out = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(out, (ys.astype(np.int64), xs.astype(np.int64)), vs.astype(np.float64))
    # Already [y][x]: the sparse layout names its axes, no transpose.
    return out


def _dense_from_document(document, path):
    """The CoMPASS layout's data[x][y] as a float64 array, still [x][y].
    Shape and type checks only; the value checks follow per file kind."""
    if not isinstance(document, dict) or "data" not in document:
        raise ParseError(
            f'JSON file has no "data" array, so it is not a CoMPASS matrix: {path}'
        )

    try:
        values = np.asarray(document["data"])
    except ValueError as exc:
        # numpy refuses a ragged nested list outright.
        raise ParseError(
            f"JSON matrix has rows of different lengths, so it is not a "
            f"rectangular matrix: {path}"
        ) from exc

    if values.dtype == object:
        # Also how numpy reports a ragged list on versions that build an
        # object array instead of raising -- tell the two apart.
        if any(isinstance(row, list) for row in document["data"]):
            lengths = {len(r) for r in document["data"] if isinstance(r, list)}
            if len(lengths) > 1:
                raise ParseError(
                    f"JSON matrix has rows of different lengths, so it is not "
                    f"a rectangular matrix: {path}"
                )
        raise ParseError(
            f"JSON matrix holds entries that are not plain numbers: {path}"
        )
    if values.ndim != 2:
        if values.ndim == 1 and values.size:
            raise ParseError(
                f"JSON data is 1-D -- a spectrum, not a matrix. Open it as a "
                f"spectrum instead: {path}"
            )
        if values.size == 0:
            raise ParseError(f"JSON matrix is empty: {path}")
        raise ParseError(
            f"JSON data is {values.ndim}-D; a matrix must be 2-D: {path}"
        )
    if values.size == 0:
        raise ParseError(f"JSON matrix is empty: {path}")
    # 'b' (bool) is excluded on purpose: true/false are not counts, and
    # numpy would otherwise happily read them as 1 and 0. Strings and nulls
    # never reach here as numbers -- they make a 'U' or object array.
    if values.dtype.kind not in "iuf":
        raise ParseError(
            f"JSON matrix holds entries that are not plain numbers: {path}"
        )
    # The dtype check above misses a MIX of booleans and numbers: numpy
    # quietly promotes true/false to 1/0 alongside the integers. Only the
    # parsed values can tell, so scan them -- about 45 ms for a 1024x1024
    # matrix, and the load runs off the GUI thread.
    if any(v is True or v is False for row in document["data"] for v in row):
        raise ParseError(
            f"JSON matrix holds entries that are not plain numbers: {path}"
        )

    values = values.astype(np.float64, copy=False)
    # Transposed: CoMPASS's rows are X, the app's rows are Y. Contiguous
    # because the cut code slices rows, and a bare .T is a column-major view.
    return np.ascontiguousarray(values.T)


def _strict_counts(values, path):
    """CoMPASS files: finite, non-negative whole numbers only (unchanged
    from before 7.0.0)."""
    if not np.all(np.isfinite(values)):
        raise ParseError(f"JSON matrix holds counts that are not finite: {path}")
    if np.any(values < 0):
        raise ParseError(f"JSON matrix holds negative counts: {path}")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-9):
        raise ParseError(
            f"JSON matrix holds fractional (weighted or scaled) contents, "
            f"which this app cannot use as counts: {path}"
        )
    return checked_round_to_int64(
        rounded,
        lambda: ParseError(
            f"JSON matrix holds a count outside the representable integer "
            f"range: {path}"
        ),
    )


def _signed_values(values, path):
    """SpectraTools files: any finite value. int64 when all are whole
    numbers that fit, otherwise the exact float64 values."""
    if not np.all(np.isfinite(values)):
        raise ParseError(f"JSON matrix holds counts that are not finite: {path}")
    if np.all(values == np.rint(values)) and np.max(np.abs(values)) < 2.0**53:
        return values.astype(np.int64)
    return values
