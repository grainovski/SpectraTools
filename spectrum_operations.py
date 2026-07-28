"""Pure-logic spectrum-processing operations (multiply/rebin/normalize)
-- no Qt dependency, mirroring calibration.py's split from its own Qt
dialog layer. main_window.py is the thin UI layer on top of this."""

import numpy as np


def multiply(data, factor):
    """Every channel's count scaled by `factor` and rounded to the
    nearest integer (numpy's round-half-to-even). `factor` is assumed
    already validated (> 0) by the caller."""
    return np.round(data * factor).astype(np.int64)
