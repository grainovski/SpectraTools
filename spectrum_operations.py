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
