"""Pure-logic spectrum-processing operations (multiply/rebin/normalize)
-- no Qt dependency, mirroring calibration.py's split from its own Qt
dialog layer. main_window.py is the thin UI layer on top of this."""

import numpy as np

from int64_cast import checked_round_to_int64


def _checked_int64(rounded):
    """A validated-positive-finite UI factor can still overflow here:
    e.g. a factor like 1e20 is finite and > 0, so it passes the dialog's
    own validation, but still overflows for any spectrum with
    non-trivial counts. See int64_cast.checked_round_to_int64 for why
    this needs numpy's own cast machinery rather than a hand-written
    magnitude check."""
    return checked_round_to_int64(
        rounded, lambda: ValueError("Result is out of range -- try a smaller factor.")
    )


def multiply(data, factor):
    """Every channel's count scaled by `factor` and rounded to the
    nearest integer (numpy's round-half-to-even). `factor` is assumed
    already validated (> 0) by the caller."""
    return _checked_int64(np.round(data * factor))


def rebin(data, factor):
    """Sums every `factor` adjacent channels into one. Zero-pads the
    end of `data` up to the next multiple of `factor` first if it
    doesn't divide evenly, so the final output bin sums fewer real
    channels than the others (the padding contributes zero) rather
    than dropping real counts. `factor` is assumed already validated
    (integer >= 2) by the caller."""
    remainder = len(data) % factor
    if remainder:
        pad = np.zeros(factor - remainder, dtype=data.dtype)
        data = np.concatenate([data, pad])
    return data.reshape(-1, factor).sum(axis=1)


def reference_value(data, channel=None, region=None):
    """The value to normalize a spectrum against: the count at a single
    channel, or the summed counts across a (lo, hi) region (inclusive,
    clamped to data's own bounds -- spectra being normalized against
    each other may have different lengths). Exactly one of
    channel/region should be given. Returns 0 for an out-of-range
    channel or an empty clamped region, rather than raising."""
    if region is not None:
        lo, hi = region
        lo = max(0, int(round(lo)))
        hi = min(len(data) - 1, int(round(hi)))
        if lo > hi:
            return 0
        return int(data[lo:hi + 1].sum())
    index = int(round(channel))
    if index < 0 or index >= len(data):
        return 0
    return int(data[index])


def normalize_factors(values):
    """Given each visible spectrum's reference value, returns a
    parallel list of scale factors: 1.0 for the maximum, max/value for
    every other positive entry, or None where the value is 0 (can't be
    scaled up to a nonzero max by multiplication -- the caller should
    skip that spectrum). If every value is 0, every factor is 1.0 (all
    already "at the max" trivially)."""
    peak = max(values)
    factors = []
    for value in values:
        if value == peak:
            factors.append(1.0)
        elif value == 0:
            factors.append(None)
        else:
            factors.append(peak / value)
    return factors


def add(data_a, data_b, factor):
    """result[i] = A[i] + factor * B[i], rounded to nearest integer
    (matches multiply()'s rounding convention). A and B are assumed
    already validated as equal length by the caller. Unlike multiply()/
    rebin(), negative results are NOT clamped -- matches TV's own
    SpcAdd, which never clamps (tv-1.9.13/lib/tv/vsSpectra.c)."""
    return _checked_int64(np.round(data_a + factor * data_b))


def subtract(data_a, data_b, factor):
    """result[i] = A[i] - factor * B[i], rounded to nearest integer.
    Same assumptions and TV-parity notes as add() above."""
    return _checked_int64(np.round(data_a - factor * data_b))


def scaled_variance(variance, factor, data):
    """Variance of multiply()'s result: var(f*X) = f**2 * var(X).

    Multiply and Normalize used to replace a spectrum's counts and leave
    its propagated variance untouched, which told every later fit the data
    was far better known than it is -- measured on a real matrix cut scaled
    by 3, the reported peak area uncertainty came out 668.93 against a
    correct 2006.80, understated by exactly the factor and with no warning
    of any kind.

    `data` is the spectrum's counts BEFORE the multiply. It matters for a
    spectrum with no propagated variance (None, meaning "came from a file,
    assume Poisson"): that assumption is a statement about THESE counts,
    and it stops being true the moment they are scaled. After multiply the
    counts are f*N, and reading them as Poisson claims var = f*N where the
    truth is f**2 * var(X) = f**2 * N -- wrong by exactly the factor, in
    every fit weight, error bar and chi-square that follows, with no
    warning of any kind. So the Poisson variance is materialized here,
    from the pre-scale counts, and scaled like any other variance. (This
    used to pass None through on the grounds that "the counts themselves
    still describe it"; for any factor other than 1 they do not.)

    A factor of exactly 1 leaves the counts untouched, so the Poisson
    assumption remains exactly true and None stays None -- a no-op
    multiply must not change what the spectrum claims about itself (a
    materialized variance also flips behaviour that keys on "carries a
    variance", e.g. integrate_region's negative-count refusal).
    """
    if variance is None:
        if factor == 1:
            return None
        variance = poisson_variance(data)
    return np.asarray(variance, dtype=float) * float(factor) ** 2


def rebinned_variance(variance, factor):
    """Variance of rebin()'s result.

    Rebinning SUMS adjacent channels, and variances of summed independent
    channels add, so this is the identical grouping and the identical sum --
    rebin() itself does the work, on floats rather than counts.

    Getting this wrong was not merely inaccurate: leaving the variance at
    its pre-rebin length made the spectrum permanently unfittable, because
    fit_peaks checks the variance against the spectrum's length and raises.
    """
    if variance is None:
        return None
    return rebin(np.asarray(variance, dtype=float), factor)


def poisson_variance(data):
    """Variance to assume for a spectrum that does not carry its own:
    the counts themselves, floored at zero.

    Correct for raw counts read from a file, where each channel is a
    Poisson sample. The floor matters because a spectrum may already
    hold subtracted, negative counts -- there is no Poisson variance to
    read off those, and zero is the honest answer rather than a negative
    variance.
    """
    return np.maximum(np.asarray(data, dtype=float), 0.0)


def combined_variance(data_a, data_b, factor, variance_a=None, variance_b=None):
    """Per-channel variance of add()/subtract()'s result.

        var(A +/- f*B) = var(A) + f**2 * var(B)

    The factor is squared either way, so addition and subtraction share
    this. Note the variances ADD even when the counts cancel: two
    spectra that subtract to zero produce a result whose uncertainty is
    larger than either input's, not zero. Fitting such a result with
    sqrt(counts) weights would claim near-perfect knowledge of a channel
    about which nothing is known.

    A None variance means the corresponding spectrum came from a file and
    is assumed Poisson. Mirrors matrix_cut's `pos + factor**2 * bg`.
    """
    va = poisson_variance(data_a) if variance_a is None else np.asarray(variance_a, dtype=float)
    vb = poisson_variance(data_b) if variance_b is None else np.asarray(variance_b, dtype=float)
    return va + float(factor) ** 2 * vb
