"""Divide a spectrum by an efficiency curve, bin by bin.

Pure: counts in, counts out. Nothing here knows about Qt, and nothing here
builds a LoadedSpectrum -- main_window does that with the result.

The curve is evaluated outside the range it was fitted over, without
restriction. That is the user's explicit choice, made with the divergence
risk stated: both models can run away when extrapolated.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CorrectedSpectrum:
    counts: np.ndarray
    variance: np.ndarray
    zeroed_nonpositive: int
    zeroed_nonfinite: int

    @property
    def zeroed(self):
        return self.zeroed_nonpositive + self.zeroed_nonfinite


def apply_efficiency(counts, calibration, curve, variance=None, model=None):
    """counts / eps(E(channel)), with non-usable efficiencies zeroed.

    A bin is corrected only where eps is **finite and > 0**, and is zeroed
    otherwise. Stating it that way rather than as `eps <= 0` is deliberate:
    under IEEE comparison `NaN <= 0` is False, so a literal test of the
    user's rule would let a NaN efficiency through and leave a NaN count --
    which then spreads through autoscaling, plotting, fitting and
    integration. `eps = +inf` already divides to about zero, so zeroing it
    changes nothing and keeps one rule instead of three.

    Returns a CorrectedSpectrum. `counts` is never modified in place.
    """
    counts = np.asarray(counts, dtype=float)
    channels = np.arange(len(counts), dtype=float)
    energies = np.asarray(calibration.apply(channels), dtype=float)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eff = np.asarray(curve.curve(energies, model), dtype=float)

    usable = np.isfinite(eff) & (eff > 0.0)
    nonfinite = int(np.count_nonzero(~np.isfinite(eff)))
    nonpositive = int(np.count_nonzero(np.isfinite(eff) & (eff <= 0.0)))

    # np.divide with where= leaves the untouched entries at the `out` value,
    # which is why out is pre-filled with zeros rather than left empty.
    corrected = np.zeros_like(counts)
    np.divide(counts, eff, out=corrected, where=usable)

    if variance is None:
        new_variance = None
    else:
        new_variance = np.zeros_like(corrected)
        np.divide(np.asarray(variance, dtype=float), eff * eff,
                  out=new_variance, where=usable)

    return CorrectedSpectrum(
        counts=corrected, variance=new_variance,
        zeroed_nonpositive=nonpositive, zeroed_nonfinite=nonfinite,
    )
