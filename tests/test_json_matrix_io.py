"""Reading a gamma-gamma matrix exported by CAEN CoMPASS as JSON.

The orientation is the part that matters and the part that fails silently.
CoMPASS writes ``data[x][y]`` -- each row of the JSON array is one X bin --
while this app indexes matrices ``[y][x]``. Getting it backwards raises no
error on a square matrix; every cut just lands on the other detector.

The convention was not taken from documentation. It was established by
comparing a JSON export cell for cell against the ROOT file CoMPASS wrote
for the SAME run, whose axes are labelled: all 1,048,576 cells agreed read
as [x][y], and the transposed reading differed by 228,998 counts. The
fixtures below are those two real files, so the comparison is re-run by
every suite run rather than remembered.
"""

import json
import os

import numpy as np
import pytest

from histogram_io import ParseError
from json_matrix_io import load_json_matrix
from matrix_cut import compute_projection

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "compass")
JSON_PATH = os.path.join(FIXTURES, "co60_EvsE.json")
ROOT_PATH = os.path.join(FIXTURES, "co60.root")
ROOT_MATRIX = "ENERGY_ENERGY/_F_EvsECH0@DT5780SCP_58363_CH1@DT5780SCP_58363"


def _write(tmp_path, payload, name="m.json"):
    path = tmp_path / name
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return str(path)


# --- orientation ----------------------------------------------------------


def test_rows_of_the_json_are_x_so_the_matrix_is_transposed(tmp_path):
    """A deliberately NON-square matrix with one count at a known cell.

    Square data cannot catch a swapped axis: the shape is the same either
    way. Here 3 X bins by 5 Y bins, so a swap changes the shape AND moves
    the marker.
    """
    nx, ny = 3, 5
    data = [[0] * ny for _ in range(nx)]       # CoMPASS layout: data[x][y]
    data[2][4] = 7                             # x = 2, y = 4
    m = load_json_matrix(_write(tmp_path, {"data": data}))

    assert m.shape == (ny, nx), "the app's matrices are [y][x]"
    assert m[4, 2] == 7
    assert m.sum() == 7
    # And the projections agree with the app's own definition of X and Y.
    assert len(compute_projection(m, "x")) == nx
    assert compute_projection(m, "x")[2] == 7
    assert compute_projection(m, "y")[4] == 7


def test_the_json_matches_the_root_file_coMPASS_wrote_for_the_same_run():
    """The real anchor for the orientation: the app's ROOT reader, whose
    orientation was established independently, must give the same matrix
    as this reader from the same CoMPASS run."""
    pytest.importorskip("uproot")
    from root_io import load_matrix

    from_json = load_json_matrix(JSON_PATH)
    from_root, _x_cal, _y_cal = load_matrix(ROOT_PATH, ROOT_MATRIX)

    # ROOT holds 1023 in-range bins per axis; CoMPASS's JSON has 1024, the
    # last row and column corresponding to ROOT's overflow bin. Compare the
    # in-range block, and require the extra row and column to be what the
    # overflow holds here: nothing.
    n = from_root.shape[0]
    assert from_root.shape == (1023, 1023)
    assert from_json.shape == (1024, 1024)
    assert np.array_equal(from_json[:n, :n], from_root)
    assert from_json[n:, :].sum() == 0 and from_json[:, n:].sum() == 0

    # Control: the comparison CAN fail. This matrix is not symmetric, so
    # the transposed reading must disagree.
    assert not np.array_equal(from_json[:n, :n].T, from_root)


def test_the_co60_lines_sit_where_each_axis_puts_them():
    """Independent of ROOT: the two detectors have slightly different
    gains, so the Co-60 pair lands in different bins on each axis."""
    m = load_json_matrix(JSON_PATH)
    px, py = compute_projection(m, "x"), compute_projection(m, "y")

    assert int(np.argmax(px[130:160])) + 130 == 149     # 1173 keV, X
    assert int(np.argmax(px[160:190])) + 160 == 169     # 1332 keV, X
    assert int(np.argmax(py[130:160])) + 130 == 152     # 1173 keV, Y
    assert int(np.argmax(py[160:190])) + 160 == 172     # 1332 keV, Y


# --- what comes back ------------------------------------------------------


def test_the_real_file_reads_as_integer_counts():
    m = load_json_matrix(JSON_PATH)
    assert m.dtype == np.int64
    assert m.flags["C_CONTIGUOUS"], "the cut code slices rows"
    assert int(m.sum()) == 259307
    assert int(m.max()) == 488


def test_the_embedded_calibration_and_limits_are_ignored(tmp_path):
    """Only the counts are read. The file's calibration does not fit its
    own bins (see the module docstring of json_matrix_io), so nothing of it
    may leak into the result."""
    data = [[1, 2], [3, 4]]
    with_meta = {"data": data, "UoM": "keV", "calibration": [-35.9, 0.515, 0],
                 "xaxis": {"limits": [0, 5637.16], "nBins": 2},
                 "yaxis": {"limits": [0, 5797.31], "nBins": 2}}
    a = load_json_matrix(_write(tmp_path, with_meta, "a.json"))
    b = load_json_matrix(_write(tmp_path, {"data": data}, "b.json"))
    assert np.array_equal(a, b)
    assert isinstance(a, np.ndarray)


def test_whole_number_floats_are_accepted(tmp_path):
    m = load_json_matrix(_write(tmp_path, {"data": [[1.0, 2.0], [3.0, 0.0]]}))
    assert m.dtype == np.int64
    assert m.tolist() == [[1, 3], [2, 0]]


# --- what is refused, and how it is reported -----------------------------


@pytest.mark.parametrize("payload, needle", [
    ("this is not json", "not valid JSON"),
    ([[1, 2], [3, 4]], "no \"data\""),
    ({"counts": [[1]]}, "no \"data\""),
    ({"data": [1, 2, 3]}, "1-D"),
    ({"data": [[[1]]]}, "3-D"),
    ({"data": []}, "empty"),
    ({"data": [[]]}, "empty"),
    ({"data": [[1, 2], [3]]}, "rows of different lengths"),
    ({"data": [[1, "2"], [3, 4]]}, "not plain numbers"),
    ({"data": [[1, None], [3, 4]]}, "not plain numbers"),
    ({"data": [[True, False], [1, 0]]}, "not plain numbers"),
    ({"data": [[1, -2], [3, 4]]}, "negative"),
    ({"data": [[1, 2.5], [3, 4]]}, "fractional"),
    ("{\"data\": [[1, NaN], [3, 4]]}", "not finite"),
    ("{\"data\": [[1, Infinity], [3, 4]]}", "not finite"),
    ({"data": [[1, 1e300], [3, 4]]}, "range"),
])
def test_a_malformed_file_raises_a_parse_error_naming_it(tmp_path, payload, needle):
    path = _write(tmp_path, payload)
    with pytest.raises(ParseError) as err:
        load_json_matrix(path)
    message = str(err.value)
    assert needle in message, message
    assert path in message, "the message must name the file"


def test_an_unreadable_file_raises_a_parse_error(tmp_path):
    missing = str(tmp_path / "nope.json")
    with pytest.raises(ParseError) as err:
        load_json_matrix(missing)
    assert "Could not read" in str(err.value)
