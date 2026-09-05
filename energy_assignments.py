"""What the user told the calibration dialog, kept so a refit does not
throw it away.

Session-lived and per spectrum: this rides on the in-memory
LoadedSpectrum, is not written to disk, and does not travel with Save
Fits. The user clears it explicitly or closes the application.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

#: Tolerance for a peak whose FWHM the fit could not determine. Narrow
#: on purpose -- with no width to reason about, only a centroid that
#: barely moved should be treated as the same peak.
_FALLBACK_TOLERANCE = 1.0

#: A stored line is left blank rather than guessed when the second-nearest
#: peak is closer than this fraction of a tolerance behind the nearest one.
#:
#: The two outcomes are not equally bad. A blank costs the user one
#: retyped energy. A WRONG energy produces a calibration that passes
#: through every assigned point, so its coefficients and residuals both
#: look reasonable and nothing on screen says otherwise. Refusing to
#: guess is therefore the cheap option, and this constant buys it.
#:
#: 0.30 measured over 209,341 generated peaks: zero misassignments at
#: every separation a fit can actually resolve (0.8 x FWHM and above),
#: keeping 95% of restores there and 100% from 1.5 x FWHM. What it gives
#: up is concentrated below half a FWHM, where the two peaks are not
#: separable and the match is a coin flip either way.
_AMBIGUITY_MARGIN = 0.30

#: Stands in for "these two cannot be matched at all" in the cost matrix.
#: linear_sum_assignment needs a finite cost everywhere, so forbidden
#: pairs get a number large enough that it never beats a real distance,
#: and the result is filtered afterwards.
_IMPOSSIBLE = 1e9


@dataclass(frozen=True)
class EnergyAssignments:
    """`pairs` is ((channel, energy), ...) as assigned at the time.

    `excluded` holds the energies the user unticked, so a point judged
    an outlier stays out of the calibration fit across a refit or a
    reopened dialog. Keyed by ENERGY rather than by row or channel
    because that is the part the user chose and the part a refit leaves
    unchanged -- a channel moves slightly every time the peaks are
    fitted again. Two rows carrying the same energy would be a mistake
    in the assignment itself, so it is unique in practice.

    Defaulted, so every existing two-argument construction still works.
    """
    source_path: str
    pairs: tuple
    excluded: tuple = ()


def _tolerance(fwhm):
    """How far a centroid may move and still be the same peak.

    Its own FWHM: a peak that shifted by more than its width between
    fits is a different peak, and restoring the old energy onto it would
    produce a calibration that looks entirely reasonable and is wrong.
    """
    try:
        width = float(fwhm)
    except (TypeError, ValueError):
        return _FALLBACK_TOLERANCE
    if not math.isfinite(width) or width <= 0.0:
        return _FALLBACK_TOLERANCE
    return width


def restore(assignments, peaks):
    """{row index: energy} for `peaks`, a list of (channel, fwhm).

    Each stored assignment goes to at most one row, so two peaks can
    never claim the same energy.

    The pairing is the one that minimises TOTAL displacement across all
    peaks at once, not a greedy nearest-first pass. Greedy is wrong in a
    doublet: the first peak to be considered takes the stored line it is
    nearest to, and the second is then left with whatever remains, even
    when swapping the two would have moved both less. Measured over
    209,341 generated peaks, that cost 6,324 misassignments where optimal
    matching costs 3,013 -- while restoring 262 MORE lines, since greedy
    can also strand a line it should have matched.

    A match whose runner-up is nearly as good is refused rather than
    guessed; see _AMBIGUITY_MARGIN for why a blank is the cheap outcome.
    """
    if assignments is None or not assignments.pairs or not peaks:
        return {}

    pairs = assignments.pairs
    cost = np.full((len(peaks), len(pairs)), _IMPOSSIBLE)
    for row, (channel, fwhm) in enumerate(peaks):
        tolerance = _tolerance(fwhm)
        for index, (stored_channel, _energy) in enumerate(pairs):
            distance = abs(channel - stored_channel)
            if distance <= tolerance:
                cost[row, index] = distance

    out = {}
    for row, index in zip(*linear_sum_assignment(cost)):
        if cost[row, index] >= _IMPOSSIBLE:
            continue  # matched only because the solver needed a full assignment
        # Ambiguity is a property of the STORED line, not of whichever
        # peak the solver handed it to: if two peaks sit almost equally
        # close to it, no rule can say which one it belongs to.
        column = np.sort(cost[:, index])
        if len(column) > 1 and column[1] < _IMPOSSIBLE:
            if (column[1] - column[0]) < _AMBIGUITY_MARGIN * _tolerance(peaks[row][1]):
                continue
        out[row] = pairs[index][1]
    return out
