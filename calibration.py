"""Channel-to-energy calibration: linear or quadratic, ported from TV's
own position calibration (tv-1.9.13/lib/tv/vsCal.c). Pure Python, no Qt
dependency -- calibration_dialog.py is the thin Qt layer on top of this."""

import math
from dataclasses import dataclass

import numpy as np

# TV's own Newton's-method precision/iteration-cap constants
# (vsCal.c:15-16), reused verbatim rather than re-derived, for exact
# parity with an already source-verified reference.
_NEWTON_PRECISION = 0.01
_NEWTON_MAXITER = 10000


class CalibrationError(Exception):
    """Raised when a Calibration's coefficients are invalid."""


class CalibrationFileError(Exception):
    """Raised when an ASCII calibration coefficient file can't be parsed."""


@dataclass
class Calibration:
    kind: str  # "linear" or "quadratic"
    a: float
    b: float
    c: float = 0.0  # unused for "linear"

    def __post_init__(self):
        # TV's own inversion (CalC, vsCal.c:1047-1082) divides by b for
        # its initial guess with no guard, silently producing NaN if
        # b == 0. This is a new UI entry point, not a raw port of an
        # existing TV-driven flow, so it's validated here instead of
        # replicating that silent failure mode.
        #
        # Finiteness is checked FIRST, and here in __post_init__ rather
        # than at each parse site, because every way a Calibration can
        # come into existence funnels through this constructor: the
        # dialog's own text fields (float("nan")/float("inf") both parse
        # happily), an ASCII coefficients file, and an N42 file's
        # CoefficientValues. A non-finite coefficient isn't caught by the
        # b == 0.0 test below either -- nan == 0.0 is False -- and would
        # otherwise sail through to make apply() return NaN for every
        # channel, i.e. an active calibration that silently displays a
        # blank/NaN energy everywhere.
        for name in ("a", "b", "c"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise CalibrationError(
                    f"Calibration coefficient '{name}' must be a finite number, got {value!r}"
                )
        if self.b == 0.0:
            raise CalibrationError("Calibration coefficient 'b' must not be zero")

    def apply(self, channel):
        """channel -> energy (keV). E = a + b*channel + c*channel**2,
        Horner's method. Matches TV's CalP (vsCal.c:981-988). Works on
        a scalar or a numpy array via broadcasting."""
        return self.a + channel * (self.b + channel * self.c)

    def derivative(self, channel):
        """dE/dchannel at `channel` -- b + 2*c*channel. Matches the
        gradient computation inside TV's CalC (vsCal.c:1066-1068), used
        there for Newton's-method inversion and here for both that and
        keV-uncertainty propagation in the results display."""
        return self.b + 2.0 * self.c * channel

    def invert(self, energy):
        """energy (keV) -> channel, via Newton-Raphson. Ported from TV's
        CalC (vsCal.c:1047-1082, the non-start-channel branch, since this
        app has no "start channel" concept). Initial guess is the linear
        approximation x0 = (energy - a) / b regardless of `kind`,
        matching TV exactly; refines against the polynomial's own
        derivative until the residual is within _NEWTON_PRECISION or
        _NEWTON_MAXITER iterations are exhausted. Scalar input only (this
        app only ever calls it with a single click/mouse-move
        coordinate). Raises CalibrationError if an iterate lands exactly
        on the calibration curve's vertex (zero derivative), where
        Newton's method cannot proceed -- same reasoning as the b=0
        guard in __post_init__: fail clearly rather than propagate a
        divide-by-zero crash or silent NaN."""
        x = (energy - self.a) / self.b
        de = self.apply(x) - energy
        iterations = 0
        while abs(de) > _NEWTON_PRECISION and iterations < _NEWTON_MAXITER:
            gradient = self.derivative(x)
            if gradient == 0.0:
                raise CalibrationError(
                    f"Cannot invert energy={energy!r}: Newton's-method "
                    "iteration reached the calibration curve's vertex "
                    "(zero derivative), where inversion is undefined"
                )
            x -= de / gradient
            de = self.apply(x) - energy
            iterations += 1
        return x

    def rescaled(self, factor):
        """A new Calibration equivalent to this one after rebinning by
        `factor`.

        Rebinning sums old channels [k*n, k*n + n - 1] into new channel k,
        so new channel k sits at the CENTRE of that group -- old channel
        k*n + (n-1)/2, not k*n. Substituting

            ch_old = ch_new * n + d,    d = (n - 1) / 2

        into E = a + b*ch_old + c*ch_old**2 gives

            a' = a + b*d + c*d**2
            b' = n * (b + 2*c*d)
            c' = c * n**2

        This previously mapped new channel k to old channel k*n exactly,
        i.e. to the FIRST old channel of the group rather than its centre,
        leaving every rebinned spectrum's energy axis low by b*(n-1)/2 --
        half a channel at factor 2, three and a half at factor 8. The
        centre is the right anchor because a rebinned bin's counts come
        from the whole group, and it is the convention used everywhere
        else here: root_io derives a calibration from a ROOT axis as
        edges[0] + width/2, the centre of bin 0.
        """
        offset = (factor - 1) / 2.0
        return Calibration(
            kind=self.kind,
            a=self.a + self.b * offset + self.c * offset ** 2,
            b=factor * (self.b + 2.0 * self.c * offset),
            c=self.c * factor ** 2,
        )


def from_points(channels, energies, quadratic=False):
    """Least-squares Calibration through (channel, energy) pairs.

    This is what turns calibration from a clerical job into a physics one:
    the channels come from FITTED peak centroids rather than from reading
    a cursor position off the screen, so the calibration inherits the
    fit's precision instead of the user's aim.

    A quadratic needs at least three points and a line at least two --
    fewer would pass exactly through them and describe nothing. Exactly
    the minimum is allowed but is an interpolation, not a fit: it cannot
    disagree with the data, so it says nothing about how good it is.
    """
    channels = np.asarray(channels, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if channels.size != energies.size:
        raise CalibrationError("Each channel needs exactly one energy")

    degree = 2 if quadratic else 1
    needed = degree + 1
    if channels.size < needed:
        raise CalibrationError(
            f"A {'quadratic' if quadratic else 'linear'} calibration needs at least "
            f"{needed} assigned peaks; {channels.size} given"
        )
    if len(set(channels.tolist())) < needed:
        # Two peaks at the same channel constrain one point, not two, and
        # numpy would return a fit with a silently meaningless slope.
        raise CalibrationError(
            "Assigned peaks must be at different channels to determine a calibration"
        )

    coefficients = np.polyfit(channels, energies, degree)
    if not np.all(np.isfinite(coefficients)):
        raise CalibrationError("Could not determine a calibration from those points")
    # polyfit returns highest power first; this app stores lowest first.
    if quadratic:
        c, b, a = coefficients
        return Calibration(kind="quadratic", a=float(a), b=float(b), c=float(c))
    b, a = coefficients
    return Calibration(kind="linear", a=float(a), b=float(b))


def residuals(calibration, channels, energies):
    """Assigned energy minus what the calibration predicts, per point.

    Worth showing rather than just the coefficients: a single mistyped
    energy, or a peak assigned to the wrong line, moves the whole fit and
    is obvious in the residuals while being invisible in a and b."""
    channels = np.asarray(channels, dtype=float)
    energies = np.asarray(energies, dtype=float)
    return energies - np.array([calibration.apply(ch) for ch in channels])


def read_coefficients_file(path, quadratic):
    """Reads a and b (and c if `quadratic`) from `path`, one coefficient
    per line, blank lines ignored. Raises CalibrationFileError on a
    wrong line count, non-numeric content, or an unreadable file."""
    expected = 3 if quadratic else 2
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise CalibrationFileError(f"Could not read {path}: {reason}") from exc
    except UnicodeDecodeError as exc:
        raise CalibrationFileError(f"Could not read {path}: not a text file ({exc})") from exc
    if len(lines) != expected:
        kind_name = "quadratic" if quadratic else "linear"
        raise CalibrationFileError(
            f"Expected {expected} coefficient(s) (one per line) for "
            f"{kind_name} calibration, found {len(lines)} in {path}"
        )
    try:
        values = [float(line) for line in lines]
    except ValueError as exc:
        raise CalibrationFileError(f"Non-numeric value in {path}: {exc}") from exc
    # float() accepts "nan"/"inf"/"-inf" as valid numbers, so the parse
    # above can succeed on a corrupted file and still yield a coefficient
    # set that makes every displayed energy NaN. Calibration.__post_init__
    # rejects these too, but only once the user presses OK -- catching it
    # here names the offending file in the message instead.
    for value in values:
        if not math.isfinite(value):
            raise CalibrationFileError(
                f"Non-finite coefficient ({value!r}) in {path}: "
                "calibration coefficients must be finite numbers"
            )
    return values
