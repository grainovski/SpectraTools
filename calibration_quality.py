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


def _literature_errors(energy_errors, count):
    """The stated uncertainties on the line energies themselves, as
    floats, with anything missing or unusable taken as zero.

    The opposite of _usable_errors, and deliberately so. A channel
    uncertainty of zero would carry infinite weight, so one bad value
    invalidates the whole weighting and the calculation refuses. An
    energy uncertainty of zero is a perfectly good statement -- it says
    the literature value contributes nothing beyond what the centroid
    already contributes -- and is the honest reading of an energy typed
    by hand with no uncertainty attached.
    """
    if energy_errors is None:
        return [0.0] * count
    if len(energy_errors) != count:
        return [0.0] * count
    values = []
    for error in energy_errors:
        try:
            value = float(error)
        except (TypeError, ValueError):
            value = 0.0
        values.append(value if math.isfinite(value) and value > 0.0 else 0.0)
    return values


def reduced_chi_squared(calibration, channels, energies, channel_errors,
                        energy_errors=None):
    """(value, None) or (None, reason).

    chi2 = sum( (E - f(ch))**2 / ((sigma_ch * dE/dch)**2 + sigma_E**2) ) / (n - p)

    The residual has two sources of uncertainty and the test is only
    fair if it counts both. `sigma_ch` is the fitted centroid's own,
    converted to energy through the calibration's local slope so that
    residual and uncertainty are in the same units. `sigma_E` is what
    the literature knows about the line -- the dE column of a `.sou`
    file.

    Leaving the second one out made the test far harsher than the data
    warrants. On a real Eu-152 calibration it halved the reported value,
    995 to 482, and the lines it matters for are exactly the ones quoted
    to a keV rather than to a thousandth of one: eu152's 1084.0(10) keV
    carries an uncertainty eight hundred times the centroid's.

    What remains after both are counted is not a defect in the
    arithmetic. A calibration fitted through centroids measured to a
    hundredth of a channel is being asked to describe a detector whose
    channel-to-energy relation is not exactly a straight line, and the
    reduced chi-squared reports that honestly -- see the Knowledge
    Database, "Reduced chi-squared, and when there is none".
    """
    count = len(channels)
    parameters = _PARAMETERS.get(calibration.kind, 2)

    if count <= parameters:
        return None, "undefined (no degrees of freedom)"

    errors = _usable_errors(channel_errors, count)
    if errors is None:
        return None, "undefined (unweighted fit)"
    literature = _literature_errors(energy_errors, count)

    total = 0.0
    for channel, energy, sigma, sigma_e in zip(channels, energies, errors, literature):
        slope = calibration.derivative(channel)
        variance = (sigma * slope) ** 2 + sigma_e ** 2
        if not math.isfinite(variance) or variance <= 0.0:
            # dE/dch is zero at a quadratic's vertex, where a channel
            # uncertainty maps to no energy uncertainty at all -- and the
            # line's own uncertainty, if it has one, is all that is left.
            return None, "undefined (a point sits at the calibration's turning point)"
        total += (energy - calibration.apply(channel)) ** 2 / variance

    value = total / (count - parameters)
    if not math.isfinite(value):
        return None, "undefined"
    return value, None
