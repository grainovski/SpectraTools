"""Divide a spectrum by an efficiency curve, bin by bin.

Pure: counts in, counts out. Nothing here knows about Qt, and nothing here
builds a LoadedSpectrum -- main_window does that with the result.

The curve is evaluated outside the range it was fitted over. That is the
user's explicit choice, made with the divergence risk stated: both models
can run away when extrapolated. The one exception is the low-energy end:
everything below ZEROED_BELOW_KEV is zeroed outright, because extrapolating
that far below the lowest calibration line produces efficiencies near zero
and so corrected counts large enough to swamp the whole spectrum.
"""

from dataclasses import dataclass

import numpy as np

#: Energy below which every bin is zeroed, whatever the efficiency there
#: works out to.
#:
#: The correction divides by the efficiency, and below the energies the
#: curve was fitted over the efficiency falls away towards zero, so the
#: division explodes. On a 0.5 keV/channel calibration against a curve
#: fitted from 121.8 keV up, channel 1 corrects a flat 1000 counts to
#: 5.7e+96 -- a value that sets the plot's autoscale and hides the spectrum
#: entirely.
#:
#: Stated in keV rather than as a channel count on purpose. How far the
#: blow-up reaches is a property of the ENERGIES the curve is asked about,
#: so a fixed number of channels covers it only at one gain: ten channels
#: was enough at 2 keV/channel and left 14 ruined bins at 0.5. A threshold
#: in keV holds at any gain, and self-limits -- a spectrum that starts
#: above it loses nothing.
#:
#: These bins hold no usable signal to lose: they sit below the lowest
#: calibration line of any ordinary source set and, on a real detector,
#: at or below the noise threshold.
ZEROED_BELOW_KEV = 50.0


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

    Every bin below ZEROED_BELOW_KEV is zeroed as well, whatever its
    efficiency evaluates to -- see that constant for why.

    Returns a CorrectedSpectrum. `counts` is never modified in place.
    """
    counts = np.asarray(counts, dtype=float)
    channels = np.arange(len(counts), dtype=float)
    energies = np.asarray(calibration.apply(channels), dtype=float)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eff = np.asarray(curve.curve(energies, model), dtype=float)

    # NaN energies (a calibration can produce them) are not "below" the
    # threshold under IEEE comparison, so they fall through to the finite
    # test below and are caught there.
    below = energies < ZEROED_BELOW_KEV
    usable = np.isfinite(eff) & (eff > 0.0)
    usable &= ~below

    # The three counts are DISJOINT, so they add up to `zeroed` and the
    # message built from them is arithmetic the user can check. A bin below
    # the threshold usually ALSO has a non-finite or non-positive
    # efficiency -- counting it twice would report more zeroed bins than
    # the spectrum has.
    low = int(np.count_nonzero(below))
    nonfinite = int(np.count_nonzero(~below & ~np.isfinite(eff)))
    nonpositive = int(np.count_nonzero(
        ~below & np.isfinite(eff) & (eff <= 0.0)))

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
