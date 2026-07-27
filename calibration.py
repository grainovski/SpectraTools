"""Channel-to-energy calibration: linear or quadratic, ported from TV's
own position calibration (tv-1.9.13/lib/tv/vsCal.c). Pure Python, no Qt
dependency -- calibration_dialog.py is the thin Qt layer on top of this."""

from dataclasses import dataclass

# Newton's-method convergence control. TV's vsCal.c used 0.01 keV; the
# Python UI's interactive use (live mouse-move coordinate inversion) needs
# tighter round-trip accuracy (~1e-6 channel), requiring ~1e-8 keV precision.
# Quadratic convergence means only a few extra iterations vs TV's original.
_NEWTON_PRECISION = 1e-8
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
        coordinate)."""
        x = (energy - self.a) / self.b
        de = self.apply(x) - energy
        iterations = 0
        while abs(de) > _NEWTON_PRECISION and iterations < _NEWTON_MAXITER:
            x -= de / self.derivative(x)
            de = self.apply(x) - energy
            iterations += 1
        return x
