"""Read-only reader for ROOT files (https://root.cern/).

Extracts 1D histograms as spectra and 2D histograms as matrices, plus the
linear calibration their axes imply. Nothing is written, matching every
other reader here (n42_io, spe_io, spk_io, mtx_io).

Built on `uproot`, which is pure Python over numpy, rather than PyROOT.
HDTV -- the ancestor this support is modelled on -- uses PyROOT, but that
needs a full ROOT installation and cannot be frozen into the onefile .exe
and native Linux packages this app ships as. uproot also reads TTrees,
which HDTV never supported at all.

PACKAGING NOTE, found by building rather than by reading docs: uproot
pulls in `awkward`, whose `awkward_cpp` component loads a DLL through
ctypes. PyInstaller does not detect that automatically, and a onefile
build without it fails at import with "Failed to load dynlib/dll
awkward-cpu-kernels.dll". `--collect-all awkward_cpp` fixes it; the cost
is about 35 MB.
"""

import numpy as np

from calibration import Calibration
from int64_cast import checked_round_to_int64


class RootError(Exception):
    """Raised when a ROOT file cannot be read, or does not hold what was
    asked for."""


# uproot names classes as ROOT does. Anything deriving from TH1/TH2 shows
# up with a concrete element type (TH1D, TH1F, TH2I, ...), so membership is
# tested on the prefix rather than an exhaustive list.
_ONE_D = ("TH1",)
_TWO_D = ("TH2",)


def _require_uproot():
    """Imported lazily so the rest of the app -- and its tests -- run
    without uproot present. Only opening a ROOT file needs it."""
    try:
        import uproot
    except ImportError as exc:  # pragma: no cover - exercised by the message
        raise RootError(
            "Reading ROOT files needs the 'uproot' package, which is not installed"
        ) from exc
    return uproot


def _kind_of(classname):
    """'spectrum', 'matrix', or None for anything this reader ignores."""
    if classname.startswith(_TWO_D):
        return "matrix"
    if classname.startswith(_ONE_D):
        return "spectrum"
    return None


def list_objects(path):
    """Every readable histogram in the file, as a list of
    (object_path, classname, kind) sorted by path.

    A ROOT file holds a directory TREE, so a file can carry dozens of
    histograms across nested directories -- this is what lets the open
    dialog show the user what is in there instead of guessing. Objects
    this reader cannot use (TTree, TGraph, vendor-specific classes) are
    left out rather than listed and then rejected on selection.
    """
    uproot = _require_uproot()
    try:
        with uproot.open(path) as handle:
            found = []
            for key, classname in handle.classnames(recursive=True).items():
                kind = _kind_of(classname)
                if kind is not None:
                    found.append((key, classname, kind))
    except RootError:
        raise
    except Exception as exc:
        raise RootError(f"Could not read ROOT file: {path} ({exc})") from exc
    return sorted(found)


def _axis_calibration(axis):
    """The linear calibration an axis implies, or None for a plain channel
    axis.

    A ROOT axis carries real coordinates, so a histogram binned in keV
    already describes its own calibration: bin i spans [edges[i],
    edges[i+1]], so its centre is at edges[0] + width*(i + 1/2). That maps
    exactly onto this app's linear calibration, E = a + b*channel, with
    b = width and a = edges[0] + width/2.

    Returns None when the axis is already plain channels (offset 0.5,
    width 1), which is what an uncalibrated ROOT spectrum looks like --
    reporting that as a calibration would clutter the UI with an identity
    transform. Non-uniform binning is refused rather than approximated:
    this app's calibration model is a polynomial in channel number and
    cannot represent arbitrary bin edges.
    """
    edges = np.asarray(axis.edges(), dtype=float)
    if edges.size < 2:
        return None
    widths = np.diff(edges)
    width = float(widths[0])
    if not np.allclose(widths, width, rtol=1e-9, atol=0.0):
        raise RootError(
            "ROOT histogram has non-uniform bin widths, which cannot be expressed "
            "as a channel calibration"
        )
    if width <= 0 or not np.isfinite(width):
        raise RootError(f"ROOT histogram has an unusable bin width ({width})")
    offset = float(edges[0]) + width / 2.0
    if np.isclose(offset, 0.5) and np.isclose(width, 1.0):
        return None
    return Calibration(kind="linear", a=offset, b=width)


def _open_object(path, object_path, want):
    uproot = _require_uproot()
    try:
        handle = uproot.open(path)
    except Exception as exc:
        raise RootError(f"Could not read ROOT file: {path} ({exc})") from exc
    with handle:
        try:
            obj = handle[object_path]
        except Exception as exc:
            raise RootError(f"ROOT file has no object {object_path!r}: {path}") from exc
        classname = getattr(obj, "classname", type(obj).__name__)
        kind = _kind_of(classname)
        if kind != want:
            raise RootError(
                f"{object_path!r} is a {classname}, not a "
                f"{'2D histogram' if want == 'matrix' else '1D histogram'}: {path}"
            )
        return _extract(obj, want, path, object_path)


def _extract(obj, want, path, object_path):
    # flow=False so under/overflow bins stay out of the data. ROOT keeps
    # them as extra bins either side; including them would silently shift
    # every channel by one and add two channels of unrelated counts.
    values = np.asarray(obj.values(flow=False))
    if values.size == 0:
        raise RootError(f"ROOT histogram {object_path!r} is empty: {path}")
    if not np.all(np.isfinite(values)):
        raise RootError(f"ROOT histogram {object_path!r} contains non-finite counts: {path}")

    if want == "matrix":
        if values.ndim != 2:
            raise RootError(
                f"{object_path!r} reports 2 dimensions but holds {values.ndim}: {path}"
            )
        # TRANSPOSED, deliberately. uproot returns values indexed [x][y]
        # (axis 0 is x), while this app indexes matrices [y][x] -- rows are
        # Y, columns are X, which is what makes compute_projection(m, "x")
        # a sum over axis 0 and what _region_bounds assumes when it reads
        # the Y extent from shape[0].
        #
        # Loading without this would transpose every ROOT matrix silently,
        # and the mistake is close to invisible: the file this was
        # developed against holds 256x256 matrices, where a swap changes no
        # shape and raises no error -- it just puts every cut on the wrong
        # axis. Established by writing a deliberately non-square histogram
        # with a marker in a known bin and reading it back, not by reading
        # documentation.
        #
        # ascontiguousarray because the cut code slices rows, and a bare
        # .T leaves a column-major view where every row slice would stride.
        matrix = np.ascontiguousarray(_as_counts(values, object_path, path).T)
        x_cal = _axis_calibration(obj.axis(0))
        y_cal = _axis_calibration(obj.axis(1))
        return matrix, x_cal, y_cal

    if values.ndim != 1:
        raise RootError(
            f"{object_path!r} reports 1 dimension but holds {values.ndim}: {path}"
        )
    return _as_counts(values, object_path, path), _axis_calibration(obj.axis(0))


def _as_counts(values, object_path, path):
    """Histogram contents as int64 counts.

    A TH1D/TH2D stores doubles even when every entry is a whole number,
    which is the normal case for a counting histogram. A histogram that has
    been scaled or weighted holds genuinely fractional contents; rounding
    those silently would corrupt them, so it is refused with a message
    naming the object rather than quietly producing wrong numbers.
    """
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-9):
        raise RootError(
            f"ROOT histogram {object_path!r} holds fractional (weighted or scaled) "
            f"contents, which this app cannot use as counts: {path}"
        )
    # Guarded, not a bare .astype(): NaN/inf are already refused upstream
    # (_open_object's isfinite check), but a FINITE value beyond int64's
    # range is its own rint, passes allclose, and a bare cast silently
    # wraps it to a garbage sentinel count. Same guard as every other
    # float-source reader (spe_io, spk_io).
    return checked_round_to_int64(
        rounded,
        lambda: RootError(
            f"ROOT histogram {object_path!r} holds a count outside the "
            f"representable integer range: {path}"
        ),
    )


def load_spectrum(path, object_path):
    """Reads one 1D histogram, returning (data, calibration) exactly as
    load_n42 does -- calibration is None when the axis is plain channels."""
    return _open_object(path, object_path, "spectrum")


def load_matrix(path, object_path):
    """Reads one 2D histogram, returning (matrix, x_calibration,
    y_calibration).

    Note the deliberate asymmetry with list_objects(): a matrix must be
    named explicitly. HDTV refuses shell-glob expansion for matrices for
    the same reason, and says why in its own docstring -- they use enough
    memory that loading several by accident can take the program down
    (rootInterface.py:247). Spectra are small enough that the convenience
    is worth it; matrices are not.
    """
    return _open_object(path, object_path, "matrix")
