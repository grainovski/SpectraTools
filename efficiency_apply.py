"""Divide a spectrum by an efficiency curve, bin by bin.

Pure: counts in, counts out. Nothing here knows about Qt, and nothing here
builds a LoadedSpectrum -- main_window does that with the result.

The curve is evaluated outside the range it was fitted over. That is the
user's explicit choice, made with the divergence risk stated: both models
can run away when extrapolated. The one exception is the bottom
ZEROED_LOW_CHANNELS bins, which are zeroed outright -- extrapolating that
far below the lowest calibration line produces efficiencies near zero and
so corrected counts large enough to swamp the whole spectrum.
"""

from dataclasses import dataclass

import numpy as np

#: Channels zeroed at the bottom of every corrected spectrum, whatever the
#: efficiency there works out to.
#:
#: The correction divides by the efficiency, and below the energies the
#: curve was fitted over the efficiency falls away towards zero, so the
#: division explodes. On a 0.5 keV/channel calibration against a curve
#: fitted from 121.8 keV up, channel 1 corrects a flat 1000 counts to
#: 5.7e+96 -- a value that sets the plot's autoscale and hides the spectrum
#: entirely. These bins hold no usable signal to lose: they sit far below
#: the lowest calibration line and, on a real detector, below the
#: threshold.
ZEROED_LOW_CHANNELS = 10


@dataclass
class CorrectedSpectrum:
    counts: np.ndarray
    variance: np.ndarray
    zeroed_nonpositive: int
    zeroed_nonfinite: int
    zeroed_low: int = 0

    @property
    def zeroed(self):
        return self.zeroed_nonpositive + self.zeroed_nonfinite + self.zeroed_low


def apply_efficiency(counts, calibration, curve, variance=None, model=None):
    """counts / eps(E(channel)), with non-usable efficiencies zeroed.

    A bin is corrected only where eps is **finite and > 0**, and is zeroed
    otherwise. Stating it that way rather than as `eps <= 0` is deliberate:
    under IEEE comparison `NaN <= 0` is False, so a literal test of the
    user's rule would let a NaN efficiency through and leave a NaN count --
    which then spreads through autoscaling, plotting, fitting and
    integration. `eps = +inf` already divides to about zero, so zeroing it
    changes nothing and keeps one rule instead of three.

    The first ZEROED_LOW_CHANNELS bins are zeroed as well, whatever their
    efficiency evaluates to -- see that constant for why.

    Returns a CorrectedSpectrum. `counts` is never modified in place.
    """
    counts = np.asarray(counts, dtype=float)
    channels = np.arange(len(counts), dtype=float)
    energies = np.asarray(calibration.apply(channels), dtype=float)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eff = np.asarray(curve.curve(energies, model), dtype=float)

    low = min(ZEROED_LOW_CHANNELS, len(counts))
    usable = np.isfinite(eff) & (eff > 0.0)
    usable[:low] = False

    # The three counts are DISJOINT, so they add up to `zeroed` and the
    # message built from them is arithmetic the user can check. Channel 0
    # is routinely both non-finite and inside the low band -- counting it
    # twice would report more zeroed bins than the spectrum has.
    nonfinite = int(np.count_nonzero(~np.isfinite(eff[low:])))
    nonpositive = int(np.count_nonzero(
        np.isfinite(eff[low:]) & (eff[low:] <= 0.0)))

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
        zeroed_low=low,
    )
