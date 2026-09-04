"""What the user told the calibration dialog, kept so a refit does not
throw it away.

Session-lived and per spectrum: this rides on the in-memory
LoadedSpectrum, is not written to disk, and does not travel with Save
Fits. The user clears it explicitly or closes the application.
"""

import math
from dataclasses import dataclass

#: Tolerance for a peak whose FWHM the fit could not determine. Narrow
#: on purpose -- with no width to reason about, only a centroid that
#: barely moved should be treated as the same peak.
_FALLBACK_TOLERANCE = 1.0


@dataclass(frozen=True)
class EnergyAssignments:
    """`pairs` is ((channel, energy), ...) as assigned at the time."""
    source_path: str
    pairs: tuple


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

    Each stored assignment goes to at most one row -- the nearest row
    within tolerance -- so two peaks can never claim the same energy.
    """
    if assignments is None or not assignments.pairs:
        return {}

    # Candidates as (distance, row, energy), then taken in order of
    # increasing distance so the nearest row wins each assignment.
    candidates = []
    for row, (channel, fwhm) in enumerate(peaks):
        tolerance = _tolerance(fwhm)
        for index, (stored_channel, energy) in enumerate(assignments.pairs):
            distance = abs(channel - stored_channel)
            if distance <= tolerance:
                candidates.append((distance, row, index, energy))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    taken_rows, taken_pairs, out = set(), set(), {}
    for _distance, row, index, energy in candidates:
        if row in taken_rows or index in taken_pairs:
            continue
        taken_rows.add(row)
        taken_pairs.add(index)
        out[row] = energy
    return out
