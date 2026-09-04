"""Compact value-with-uncertainty notation, e.g. 352.7217(14).

The digits in parentheses are the uncertainty expressed in units of the
last decimal place shown, which is how nuclear data tables quote a
measurement. It is the notation the Fit Results panel and the
calibration plot window both use, and it lives in its own module so the
two cannot drift apart -- a value rendered one way in the table and
another in the plot of the same fit would be worse than either.
"""

import math

#: Significant digits kept in the uncertainty. Two is the usual choice in
#: nuclear data compilations: one digit throws away real precision when
#: the leading digit is 1, and three implies more than a fit can support.
_UNCERTAINTY_SIG_DIGITS = 2

#: Decimals used when there is no usable uncertainty to set the scale.
_FALLBACK_DECIMALS = 4

#: Decimals the Fit Results table shows for a position or a width. Two is
#: what an energy in keV is read to in practice; the fitted precision
#: beyond it set the column width without telling anyone anything.
CAPPED_DECIMALS = 2


def _scale(sigma):
    """(decimals, rounded sigma) for `sigma` kept to
    _UNCERTAINTY_SIG_DIGITS significant digits.

    Shared by compact and compact_capped so the two cannot disagree
    about where an uncertainty's digits begin.
    """
    # Decimal place of the uncertainty's leading digit, then widened to
    # keep _UNCERTAINTY_SIG_DIGITS of it. exponent 0 means units, -2
    # means hundredths.
    exponent = math.floor(math.log10(sigma)) - (_UNCERTAINTY_SIG_DIGITS - 1)
    decimals = max(0, -exponent)

    # Round the uncertainty FIRST, then read its digits off at that same
    # decimal place. Doing it the other way round lets 0.0099 print as
    # '(99)' beside a value rounded as if sigma were 0.01.
    quantum = 10.0 ** exponent
    sigma_rounded = round(sigma / quantum) * quantum
    if sigma_rounded > 0:
        # Rounding can carry (9.99 -> 10.0), which moves the decimal place.
        new_exponent = math.floor(math.log10(sigma_rounded)) - (_UNCERTAINTY_SIG_DIGITS - 1)
        if new_exponent != exponent:
            exponent = new_exponent
            decimals = max(0, -exponent)
            quantum = 10.0 ** exponent
            sigma_rounded = round(sigma / quantum) * quantum
    return decimals, sigma_rounded


def _usable(error):
    """An uncertainty that actually says something.

    peak_fit reports 0.0 for a parameter held fixed and NaN for one the
    data does not constrain. Neither is a measurement, and rendering
    either as '(0)' or '(nan)' would claim something false.
    """
    if error is None:
        return False
    try:
        value = float(error)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def compact(value, error):
    """'352.7217(14)', or the value alone when `error` is unusable.

    Returns an em dash for a non-finite value: there is no number to
    show, and printing 'nan' in a results table reads as data rather
    than as an absent measurement.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"

    if not _usable(error):
        # Trailing zeros trimmed so a value with no uncertainty does not
        # imply precision it never claimed.
        text = f"{number:.{_FALLBACK_DECIMALS}f}".rstrip("0").rstrip(".")
        return text or "0"

    decimals, sigma_rounded = _scale(float(error))
    # In units of the LAST PRINTED decimal place, which is 10**-decimals.
    # That equals the rounding quantum only while the exponent is <= 0;
    # once sigma reaches 100 the max(0, ...) clamp in _scale pins decimals
    # at zero while the quantum keeps growing, and dividing by the quantum
    # there reported 170 as "(17)".
    digits = int(round(sigma_rounded * (10.0 ** decimals)))
    return f"{number:.{decimals}f}({digits})"


def compact_capped(value, error, max_decimals=CAPPED_DECIMALS):
    """`compact`, but never showing more than `max_decimals` decimals.

    The uncertainty is dropped entirely when it is smaller than the
    finest place those decimals can express. At two decimals a sigma of
    0.0014 cannot be written at all: '(0)' would claim a perfect
    measurement, and widening the value to 352.7217(14) to fit it is
    exactly what this cap exists to prevent -- a position column whose
    width is set by digits nobody reads pushes the numbers that matter
    sideways.

    Where the uncertainty IS expressible the digits are re-read at the
    capped place, so 661.657(30) becomes 661.66(3) rather than keeping
    parenthesised digits that refer to a decimal no longer shown.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"

    if not _usable(error):
        return f"{number:.{max_decimals}f}"

    sigma = float(error)
    if sigma < 10.0 ** (-max_decimals):
        return f"{number:.{max_decimals}f}"

    decimals, sigma_rounded = _scale(sigma)
    if decimals > max_decimals:
        # Re-round at the capped place rather than truncating the digits
        # computed for a finer one.
        decimals = max_decimals
        sigma_rounded = round(sigma, max_decimals)
    digits = int(round(sigma_rounded * (10.0 ** decimals)))
    if digits == 0:
        return f"{number:.{decimals}f}"
    return f"{number:.{decimals}f}({digits})"
