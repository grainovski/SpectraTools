"""Writes fitted peaks in CalEnEff's efficiency-calibration format.

CalEnEff reads seven whitespace-separated columns with no header,
documented in its own ra226_gui.py:18 as carrying ABSOLUTE
uncertainties:

    ch   delta_ch   N   delta_N   E[keV]   I[%]   delta_I[%]

The first four come from the fit, the last three from the .sou source
line the peak was assigned to.

N is the NET area. CalEnEff computes eff = N / I_pct, so a gross area
would fold the background into the efficiency curve.

delta_N is the fit's area uncertainty widened by efficiency_area_error
wherever the peak fit is poor, since 6.1.1 -- the same number the
efficiency window fits with, so CalEnEff and this program see one data
set.
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


def efficiency_area_error(area_err, reduced_chi2):
    """The uncertainty a peak's area carries onto the efficiency curve:
    the fit's own `area_err`, times sqrt(chi2/ndf) of the fit when that
    exceeds 1 -- inflate-only, exactly as the Birge ratio treats the
    efficiency fit itself (efficiency.EfficiencyMC._band).

    The fit's error is counting statistics alone -- its covariance is not
    scaled, as TV's is not (vsCurFit.c CurFinish takes the errors straight
    from the inverted curvature matrix) -- which is right when the shape
    describes the peak and optimistic when it does not. On real spectra
    it mostly does not: median chi2/ndf 2.5-2.9 over the Ra-226 lines and
    8-10 over the Eu-152 ones, up to ~470 on the strongest peaks, where a
    million counts show every departure of a Gaussian from the real
    shape. And a peak with an unmodelled neighbour in its window fits
    badly for that very reason. Left as they were, those areas claimed a
    precision their own fits deny, and the efficiency fit's Birge ratio
    came out 4.6-6.5 on those spectra.

    Only the efficiency points and the CalEnEff export see this. The
    peak's reported area_err -- Fit Results, the fit logs, the reports
    -- stays the fit's own.

    A missing or non-finite chi2/ndf (a fit with no degrees of freedom)
    leaves the error as it was: there is no evidence either way.
    """
    area_err = float(area_err)
    if reduced_chi2 is None:
        return area_err
    reduced_chi2 = float(reduced_chi2)
    if not math.isfinite(reduced_chi2) or reduced_chi2 <= 1.0:
        return area_err
    return area_err * math.sqrt(reduced_chi2)


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

    That same rule is enforced here rather than merely cited. A matched
    line whose area error AND intensity error are both zero produces
    exactly the row CalEnEff rejects, so it is skipped and counted too.
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
        area_err = float(area_err)
        intensity_pct_err = line.intensity_err * scale
        if area_err == 0.0 and intensity_pct_err == 0.0:
            # eff = N / I_pct would carry no uncertainty at all, which is
            # the row CalEnEff refuses to load.
            skipped += 1
            continue
        rows.append(ExportRow(
            channel=float(channel),
            channel_err=float(channel_err),
            area=float(area),
            area_err=area_err,
            energy=float(energy),
            intensity_pct=line.intensity * scale,
            intensity_pct_err=intensity_pct_err,
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
    # A non-finite value formats as the literal text "nan" or "inf",
    # which np.loadtxt reads back as a float and CalEnEff then computes
    # with. Refusing names the peak while the user can still act on it.
    for row in rows:
        for field, value in (
            ("channel", row.channel), ("channel error", row.channel_err),
            ("area", row.area), ("area error", row.area_err),
            ("energy", row.energy), ("intensity", row.intensity_pct),
            ("intensity error", row.intensity_pct_err),
        ):
            if not math.isfinite(value):
                raise ExportError(
                    f"The peak at channel {row.channel:.2f} has a "
                    f"non-finite {field}, which cannot be exported."
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
