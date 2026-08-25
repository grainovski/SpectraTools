"""Tests for the ROOT reader.

Fixtures are generated with uproot's writer rather than committed as
binaries, so the suite carries no opaque test data and the fixtures stay
readable as code. The one real acquisition file this was developed
against (rootfile.root) is deliberately untracked, like the other sample
spectra in this repo.
"""

import numpy as np
import pytest

import root_io
from root_io import RootError

uproot = pytest.importorskip("uproot")


def _write(path, objects):
    with uproot.recreate(path) as f:
        for name, obj in objects.items():
            f[name] = obj


def _th1(counts, low=0.0, high=None):
    """A 1D histogram with the given contents over [low, high]."""
    counts = np.asarray(counts, dtype=float)
    if high is None:
        high = float(len(counts))
    return counts, np.linspace(low, high, len(counts) + 1)


def _th2(values_yx, x_range=None, y_range=None):
    """`values_yx` is given in THIS APP's convention -- [y][x], rows are Y
    -- and transposed here, because uproot's writer (like ROOT) wants
    [x][y]. Keeping the tests in the app's own convention is what makes
    the transpose in root_io visible if it is ever removed."""
    values_yx = np.asarray(values_yx, dtype=float)
    ny, nx = values_yx.shape
    x_range = x_range or (0.0, float(nx))
    y_range = y_range or (0.0, float(ny))
    return (values_yx.T,
            np.linspace(x_range[0], x_range[1], nx + 1),
            np.linspace(y_range[0], y_range[1], ny + 1))


@pytest.fixture
def simple_file(tmp_path):
    path = tmp_path / "simple.root"
    counts = np.zeros(64)
    counts[30:35] = [4, 20, 50, 18, 3]
    matrix = np.arange(6 * 8, dtype=float).reshape(6, 8)
    _write(path, {
        "spec": _th1(counts),
        "dir/nested": _th1(counts),
        "kev": _th1(counts, low=100.0, high=100.0 + 64 * 0.5),
        "mat": _th2(matrix),
    })
    return path


# --- listing -----------------------------------------------------------


def test_list_objects_finds_histograms_across_directories(simple_file):
    found = root_io.list_objects(simple_file)
    kinds = {key.split(";")[0]: (cls, kind) for key, cls, kind in found}

    assert kinds["spec"][1] == "spectrum"
    assert kinds["mat"][1] == "matrix"
    # A ROOT file is a directory TREE; nested objects must be reachable or
    # the picker would show only the top level.
    assert "dir/nested" in kinds


def test_list_objects_omits_what_the_reader_cannot_use(tmp_path):
    """Unusable objects are left out rather than listed and then refused on
    selection -- the file this was developed against carries a
    vendor-specific CalibrationCoefficient class that uproot cannot
    interpret at all."""
    path = tmp_path / "mixed.root"
    with uproot.recreate(path) as f:
        f["good"] = _th1(np.arange(10, dtype=float))
        f["atree"] = {"branch": np.arange(10)}

    listed = {key.split(";")[0] for key, _, _ in root_io.list_objects(path)}
    assert "good" in listed
    assert "atree" not in listed


def test_listing_a_file_that_is_not_root_fails_with_the_path(tmp_path):
    path = tmp_path / "nope.root"
    path.write_bytes(b"this is not a ROOT file")
    with pytest.raises(RootError, match="Could not read ROOT file"):
        root_io.list_objects(path)


# --- 1D ----------------------------------------------------------------


def test_load_spectrum_returns_counts_and_no_calibration_for_channels(simple_file):
    data, calibration = root_io.load_spectrum(simple_file, "spec")
    assert data.dtype == np.int64
    assert data.shape == (64,)
    assert data.sum() == 95
    assert list(data[30:35]) == [4, 20, 50, 18, 3]
    # A plain channel axis is not a calibration; reporting one would put an
    # identity transform in front of the user for no reason.
    assert calibration is None


def test_load_spectrum_derives_a_calibration_from_a_real_axis(simple_file):
    _, calibration = root_io.load_spectrum(simple_file, "kev")
    assert calibration is not None
    assert calibration.kind == "linear"
    # Bin i spans [100 + 0.5i, 100 + 0.5(i+1)], so its CENTRE is at
    # 100.25 + 0.5i -- the half-bin offset is the part that is easy to
    # drop, and dropping it shifts every energy by half a channel.
    assert calibration.b == pytest.approx(0.5)
    assert calibration.a == pytest.approx(100.25)


def test_under_and_overflow_bins_are_excluded(tmp_path):
    """ROOT keeps under/overflow as extra bins either side. Including them
    would shift every channel by one and add two channels of unrelated
    counts."""
    path = tmp_path / "flow.root"
    _write(path, {"h": _th1(np.arange(1, 11, dtype=float))})
    data, _ = root_io.load_spectrum(path, "h")
    assert data.shape == (10,)
    assert list(data) == list(range(1, 11))


def test_loading_a_matrix_as_a_spectrum_is_refused(simple_file):
    with pytest.raises(RootError, match="not a 1D histogram"):
        root_io.load_spectrum(simple_file, "mat")


def test_loading_a_missing_object_names_it(simple_file):
    with pytest.raises(RootError, match="no object 'absent'"):
        root_io.load_spectrum(simple_file, "absent")


# --- 2D ----------------------------------------------------------------


def test_load_matrix_returns_the_array_and_both_axis_calibrations(tmp_path):
    path = tmp_path / "m.root"
    values = np.arange(6 * 8, dtype=float).reshape(6, 8)  # [y][x]: 6 rows, 8 cols
    _write(path, {"m": _th2(values, x_range=(0.0, 16.0), y_range=(0.0, 6.0))})

    matrix, x_cal, y_cal = root_io.load_matrix(path, "m")
    assert matrix.dtype == np.int64
    assert matrix.shape == (6, 8)
    assert matrix.sum() == int(values.sum())
    # x has 8 bins over 16 units -> width 2; y has 6 bins over 6 -> width 1
    # starting at 0, i.e. centres at 0.5, which is a plain channel axis.
    assert x_cal is not None and x_cal.b == pytest.approx(2.0)
    assert y_cal is None


def test_a_non_square_matrix_is_not_transposed(tmp_path):
    """The axis convention, pinned on a NON-SQUARE matrix with a marker in
    a known cell.

    uproot returns values indexed [x][y]; this app indexes them [y][x].
    Getting that backwards puts every cut on the wrong axis, and on the
    square matrices real acquisition files tend to contain it changes no
    shape and raises nothing -- so only a deliberately non-square fixture
    can catch it.
    """
    path = tmp_path / "asym.root"
    values = np.zeros((5, 3))          # 5 rows (Y), 3 columns (X)
    values[4, 2] = 99.0                # last Y, last X
    values[0, 1] = 7.0
    _write(path, {"m": _th2(values)})

    matrix, _, _ = root_io.load_matrix(path, "m")
    assert matrix.shape == (5, 3), "rows must be Y and columns X"
    assert matrix[4, 2] == 99
    assert matrix[0, 1] == 7
    assert np.array_equal(matrix, values.astype(np.int64))


def test_loading_a_spectrum_as_a_matrix_is_refused(simple_file):
    with pytest.raises(RootError, match="not a 2D histogram"):
        root_io.load_matrix(simple_file, "spec")


def test_a_root_matrix_feeds_the_existing_cut_code_unchanged(tmp_path):
    """The reason no VMatrix-style backend interface was needed: once
    loaded, a ROOT matrix is an ordinary channel-indexed numpy array, so
    compute_cut and compute_projection take it as-is."""
    from matrix_cut import compute_cut, compute_projection

    path = tmp_path / "cut.root"
    values = np.zeros((40, 50), dtype=float)
    values[10:15, :] = 7.0
    values[25:30, :] = 3.0
    _write(path, {"m": _th2(values)})

    matrix, _, _ = root_io.load_matrix(path, "m")
    assert compute_projection(matrix, "x").sum() == matrix.sum()
    # Gate rows 10-14 against background rows 25-29: equal widths, so the
    # net is 5*(7-3) = 20 per column.
    net = compute_cut(matrix, "y", (10, 14), [(25, 29)])
    assert net == pytest.approx(np.full(50, 20.0))


# --- refusals that protect the numbers ---------------------------------


def test_fractional_contents_are_refused_rather_than_rounded(tmp_path):
    """A scaled or weighted histogram holds genuinely fractional contents.
    Rounding them silently would corrupt the data; the message names the
    object instead."""
    path = tmp_path / "weighted.root"
    _write(path, {"w": _th1(np.array([1.5, 2.25, 3.75, 0.5]))})
    with pytest.raises(RootError, match="fractional"):
        root_io.load_spectrum(path, "w")


def test_out_of_range_contents_are_refused_rather_than_wrapped(tmp_path):
    """A FINITE value beyond int64's range is its own rint, passes the
    fractional-contents check, and used to sail into a bare
    .astype(np.int64), silently wrapping to a garbage sentinel count.
    The checked cast raises with the object's name instead -- same guard
    as spe_io/spk_io."""
    path = tmp_path / "huge.root"
    counts = np.zeros(8)
    counts[3] = 1e19
    _write(path, {"h": _th1(counts)})
    with pytest.raises(RootError, match="outside the representable integer range"):
        root_io.load_spectrum(path, "h")


def test_non_finite_contents_are_refused_not_imported(tmp_path):
    """NaN and +-inf are caught upstream by _open_object's isfinite
    check, before the cast is ever reached -- pinned here so a NaN or
    inf bin can never import as data whichever guard happens to fire."""
    for name, bad in (("nan", np.nan), ("plusinf", np.inf), ("minusinf", -np.inf)):
        path = tmp_path / f"{name}.root"
        counts = np.zeros(8)
        counts[3] = bad
        _write(path, {"h": _th1(counts)})
        with pytest.raises(RootError, match="non-finite"):
            root_io.load_spectrum(path, "h")


def test_non_uniform_binning_is_refused(tmp_path):
    """This app's calibration is a polynomial in channel number and cannot
    express arbitrary bin edges, so approximating them would be a lie."""
    path = tmp_path / "variable.root"
    with uproot.recreate(path) as f:
        f["v"] = (np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 3.0, 10.0]))
    with pytest.raises(RootError, match="non-uniform"):
        root_io.load_spectrum(path, "v")


def test_an_empty_histogram_is_refused(tmp_path):
    path = tmp_path / "empty.root"
    with uproot.recreate(path) as f:
        f["e"] = (np.array([], dtype=float), np.array([0.0]))
    with pytest.raises(RootError, match="empty"):
        root_io.load_spectrum(path, "e")


# --- metadata ----------------------------------------------------------


def test_livetime_is_none_when_the_file_does_not_record_it(simple_file):
    """Best-effort metadata: a file without a LiveTime object is entirely
    normal and must not raise."""
    assert root_io.livetime_seconds(simple_file) is None
