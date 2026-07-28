"""Pure-logic spectrum-processing operations (multiply/rebin/normalize)
-- no Qt dependency, mirroring calibration.py's split from its own Qt
dialog layer. main_window.py is the thin UI layer on top of this."""

import numpy as np


def multiply(data, factor):
    """Every channel's count scaled by `factor` and rounded to the
    nearest integer (numpy's round-half-to-even). `factor` is assumed
    already validated (> 0) by the caller."""
    return np.round(data * factor).astype(np.int64)


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
