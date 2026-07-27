"""Channel-to-energy calibration: linear or quadratic, ported from TV's
own position calibration (tv-1.9.13/lib/tv/vsCal.c). Pure Python, no Qt
dependency -- calibration_dialog.py is the thin Qt layer on top of this."""

from dataclasses import dataclass


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
