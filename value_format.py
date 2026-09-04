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

    sigma = float(error)
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

    digits = int(round(sigma_rounded / quantum))
    return f"{number:.{decimals}f}({digits})"
