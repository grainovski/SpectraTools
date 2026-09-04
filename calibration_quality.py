"""How well an energy calibration fits the points it was built from.

Kept out of calibration.py deliberately: a Calibration is a coordinate
transform and is used in places that have no points to judge it by, so
its identity should not carry a goodness-of-fit number. This module is
the judge, and it is pure -- no Qt, no spectra.
"""

import math

#: Free parameters per calibration kind, the p in (n - p).
_PARAMETERS = {"linear": 2, "quadratic": 3}


def _usable_errors(channel_errors, count):
    """The errors as floats, or None if any one of them is unusable.

    All-or-nothing on purpose, matching from_points: it falls back to an
    unweighted fit if ANY error is missing, zero, negative or non-finite,
    so a chi-squared computed from the usable subset would be judging a
    fit that was never performed.
    """
    if channel_errors is None or len(channel_errors) != count:
        return None
    values = []
    for error in channel_errors:
        if error is None:
            return None
        try:
            value = float(error)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        values.append(value)
    return values


def reduced_chi_squared(calibration, channels, energies, channel_errors):
    """(value, None) or (None, reason).

    chi2 = sum( ((E - f(ch)) / (sigma_ch * dE/dch))**2 ) / (n - p)

    The channel uncertainty is converted to an energy uncertainty
    through the calibration's own local slope, which is what makes the
    residual and its uncertainty comparable.
    """
    count = len(channels)
    parameters = _PARAMETERS.get(calibration.kind, 2)

    if count <= parameters:
        return None, "undefined (no degrees of freedom)"

    errors = _usable_errors(channel_errors, count)
    if errors is None:
        return None, "undefined (unweighted fit)"

    total = 0.0
    for channel, energy, sigma in zip(channels, energies, errors):
        slope = calibration.derivative(channel)
        energy_sigma = abs(sigma * slope)
        if not math.isfinite(energy_sigma) or energy_sigma <= 0.0:
            # dE/dch is zero at a quadratic's vertex, where a channel
            # uncertainty maps to no energy uncertainty at all.
            return None, "undefined (a point sits at the calibration's turning point)"
        total += ((energy - calibration.apply(channel)) / energy_sigma) ** 2

    value = total / (count - parameters)
    if not math.isfinite(value):
        return None, "undefined"
    return value, None
