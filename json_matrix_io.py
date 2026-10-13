"""Gamma-gamma matrices exported by CAEN CoMPASS as JSON.

The file is one JSON object. Only its ``data`` member is read: a
rectangular 2-D array of counts.

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
"""

import json

import numpy as np

from histogram_io import ParseError
from int64_cast import checked_round_to_int64


def load_json_matrix(path):
    """Read a CoMPASS JSON matrix as an int64 array indexed [y][x].

    Raises ParseError, naming the file, for anything that is not a
    rectangular 2-D array of finite, non-negative whole-number counts.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            document = json.load(fh)
    except OSError as exc:
        raise ParseError(f"Could not read {path}: {exc}") from exc
    except (ValueError, UnicodeDecodeError) as exc:
        # json.JSONDecodeError is a ValueError.
        raise ParseError(f"File is not valid JSON ({exc}): {path}") from exc

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
    counts = checked_round_to_int64(
        rounded,
        lambda: ParseError(
            f"JSON matrix holds a count outside the representable integer "
            f"range: {path}"
        ),
    )
    # Transposed: CoMPASS's rows are X, the app's rows are Y. Contiguous
    # because the cut code slices rows, and a bare .T is a column-major view.
    return np.ascontiguousarray(counts.T)
