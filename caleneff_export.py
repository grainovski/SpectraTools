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


#: How near a fitted peak's centroid, in that fit's own FWHM, a line of the
#: source must fall for the peak to be taken as containing it -- when no
#: other peak is nearer and no peak has been given that line's energy. One
#: width: a single-peak fit of two lines that close broadens to take in
#: both (386.77 and 388.89 keV fitted as one peak came out 3.45 keV wide,
#: where that spectrum's lines are 2.0), while a line further out keeps
#: most of its counts outside the fitted shape and is usually found as a
#: peak of its own. The automatic refit's blends sit at a third of a width
#: on real spectra, well inside this.
SHARED_PEAK_FWHM = 1.0


def blend_partners(peaks, energies, source_lines, calibration):
    """{peak index: [SourceLine]}, the unassigned lines inside each assigned
    peak -- see share_blended_areas for `peaks` and `energies`.

    A line is inside a peak when the calibration puts it nearer to that
    peak than to any other, within SHARED_PEAK_FWHM of the peak's fitted
    width, and no peak has been given its energy. A peak fitted for the
    line itself sits at the line's place, nearer than any other -- the
    second peak the automatic refit pins there sits exactly on it -- so a
    line with a peak of its own is not counted as part of another's.
    """
    if calibration is None or not source_lines or not peaks:
        return {}
    taken = set()
    for energy in energies:
        if energy is not None:
            line = _matching_line(energy, source_lines)
            if line is not None:
                taken.add(line.energy)
    partners = {}
    for line in source_lines:
        if line.energy in taken:
            continue
        try:
            channel = float(calibration.invert(line.energy))
        except Exception:  # noqa: BLE001 -- a line the calibration cannot place
            continue
        if not math.isfinite(channel):
            continue
        nearest = min(range(len(peaks)), key=lambda i: abs(peaks[i][0] - channel))
        centre, _err, fwhm, _area, _area_err = peaks[nearest]
        if energies[nearest] is None or fwhm is None or not (fwhm > 0.0):
            continue
        if abs(channel - centre) <= SHARED_PEAK_FWHM * fwhm:
            partners.setdefault(nearest, []).append(line)
    return partners


def committed_peaks(fits, energy_of):
    """(peaks, energies) for share_blended_areas, from fit results: every
    peak of every fit, in order, with the area error an efficiency point
    carries (efficiency_area_error). `energy_of` maps id(peak) to the
    energy that peak was given; any other peak gets None. Results without
    peaks -- integrations -- contribute nothing.
    """
    peaks, energies = [], []
    for result in fits:
        for peak in getattr(result, "peaks", None) or []:
            peaks.append((peak.position, peak.position_err or 0.0, peak.fwhm, peak.area,
                          efficiency_area_error(peak.area_err,
                                                getattr(result, "reduced_chi2", None))))
            energies.append(energy_of.get(id(peak)))
    return peaks, energies


def share_blended_areas(peaks, energies, source_lines, calibration):
    """Efficiency points, each carrying only its own line's share of its
    peak.

    `peaks` is (channel, channel_err, fwhm, area, area_err) for EVERY peak
    on the spectrum, assigned or not -- a peak nobody named still says
    where a line's counts went. `energies` is parallel to it: the energy
    given to each peak, or None. Returns (channel, channel_err, area,
    area_err, energy) for each assigned peak, in `peaks` order: what the
    calibration plot fits and build_rows writes.

    A line of the source that nobody assigned but that lies inside an
    assigned peak (blend_partners) is part of it, unresolved, and its
    counts are in the fitted area. That area is shared in proportion to
    intensity -- what a merged doublet in a .sou does -- and the partners'
    intensity uncertainty joins the area's: N_line = N * I / (I + S), so
    dN_line / N_line gains dS / (I + S) in quadrature. On real Ra-226
    spectra the Bi-214 line at 273.79 keV sits inside 274.80 keV like
    this, a third as strong, and would otherwise add 38% of its counts.

    One rule for both routes to an efficiency: the automatic calibration
    and Calibrate from Fitted Peaks each pass their peaks through here, so
    reopening the dialog after an automatic run shows the same points the
    run did. Without a calibration or a source there is nothing to place a
    partner with, and the areas pass through unchanged.
    """
    partners = blend_partners(peaks, energies, source_lines, calibration)
    points = []
    for index, ((channel, channel_err, _fwhm, area, area_err), energy) in enumerate(
            zip(peaks, energies)):
        if energy is None:
            continue
        sharing = partners.get(index, ())
        line = _matching_line(energy, source_lines) if sharing else None
        if line is not None and line.intensity > 0.0:
            total = line.intensity + sum(other.intensity for other in sharing)
            share = line.intensity / total
            extra = math.sqrt(sum(other.intensity_err ** 2 for other in sharing)) / total
            area_err = math.hypot(area_err * share, area * share * extra)
            area = area * share
        points.append((channel, channel_err, area, area_err, energy))
    return points


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
