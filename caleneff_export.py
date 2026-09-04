"""Writes fitted peaks in CalEnEff's efficiency-calibration format.

CalEnEff (C:\\Users\\RIG\\Documents\\Claude\\efficieny) reads seven
whitespace-separated columns with no header, documented in its own
ra226_gui.py:18 as carrying ABSOLUTE uncertainties:

    ch   delta_ch   N   delta_N   E[keV]   I[%]   delta_I[%]

The first four come from the fit, the last three from the .sou source
line the peak was assigned to.

N is the NET area. CalEnEff computes eff = N / I_pct, so a gross area
would fold the background into the efficiency curve.
"""

import math
from dataclasses import dataclass

#: The strongest line in a source is normalised to this. .sou intensities
#: have no common convention -- the nine sample files peak at values from
#: 1000 to 100000 -- so a scale has to be chosen. Normalising to the
#: strongest line matches CalEnEff's own 226Ra sample, where 609.312 keV
#: carries I = 100. Because eff = N / I_pct, a constant factor rescales
#: the whole efficiency curve without changing its shape, and the
#: relative uncertainties CalEnEff propagates are invariant under it.
_STRONGEST_LINE_PERCENT = 100.0

#: Energies are matched to source lines by value, not by index, because
#: the assignment stores the energy alone. Both sides came from the same
#: file, so this only has to absorb float round-tripping through text.
_ENERGY_MATCH_TOLERANCE = 1e-6


class ExportError(Exception):
    """Raised when the file cannot be produced or would be unusable."""


@dataclass(frozen=True)
class ExportRow:
    channel: float
    channel_err: float
    area: float
    area_err: float
    energy: float
    intensity_pct: float
    intensity_pct_err: float


def build_rows(assignments, source_lines):
    """(rows, skipped) for `assignments`, a list of
    (channel, channel_err, area, area_err, energy) tuples.

    An assignment whose energy matches no source line is SKIPPED and
    counted rather than exported with a zero intensity: CalEnEff refuses
    rows where dN and dI are both zero, because the efficiency
    uncertainty would come out zero, and a made-up intensity would
    silently distort the curve.
    """
    if source_lines:
        maximum = max(line.intensity for line in source_lines)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ExportError(
                "The source has no positive intensity to normalise by, so "
                "I[%] cannot be computed."
            )
        scale = _STRONGEST_LINE_PERCENT / maximum
    else:
        scale = None

    rows, skipped = [], 0
    for channel, channel_err, area, area_err, energy in assignments:
        line = _matching_line(energy, source_lines)
        if line is None or scale is None:
            skipped += 1
            continue
        rows.append(ExportRow(
            channel=float(channel),
            channel_err=float(channel_err),
            area=float(area),
            area_err=float(area_err),
            energy=float(energy),
            intensity_pct=line.intensity * scale,
            intensity_pct_err=line.intensity_err * scale,
        ))
    return rows, skipped


def _matching_line(energy, source_lines):
    for line in source_lines:
        if abs(line.energy - energy) <= _ENERGY_MATCH_TOLERANCE:
            return line
    return None


def write_caleneff(path, rows):
    """Write `rows` as CalEnEff's seven columns.

    Refuses an empty file: CalEnEff would fail to load it, and failing
    here names the reason while the user is still looking at the dialog.
    """
    if not rows:
        raise ExportError(
            "There are no rows to export: no assigned peak had an "
            "intensity from a source file."
        )
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    f"{row.channel:<14.6f} {row.channel_err:<12.6f} "
                    f"{row.area:<12.1f} {row.area_err:<10.2f} "
                    f"{row.energy:<10.3f} {row.intensity_pct:<8.3f} "
                    f"{row.intensity_pct_err:.3f}\n"
                )
    except OSError as exc:
        raise ExportError(f"Cannot write {path}: {exc}") from None
