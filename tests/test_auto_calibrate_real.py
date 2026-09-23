"""The bug report, as a test: a real Eu-152 spectrum whose strongest
line, 121.78 keV, was found by the search and then lost -- grouped into
a four-peak fit with a broad Compton feature eight channels away, whose
spurious 23-channel width gave the group a 68-channel reach. Nothing in
channels 100-260 survived that fit and the strongest peak in the
spectrum went unassigned.

Also the efficiency test and the refit pass, which have no synthetic
equivalent worth trusting: the synthetic spectra are sparse and their
areas are exactly proportional to the intensities, so they exercise
neither the crowded-background search nor the scatter test.
"""

import os

import numpy as np
import pytest

import auto_calibrate
import calibration as C
from peak_fit import channel_indices
from sou_io import load_sou
from spe_io import load_spe
from synthetic_calibration import source_file

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


@pytest.fixture(scope="module")
def eu152():
    counts = np.asarray(load_spe(os.path.join(FIXTURES, "eu152_real.spe")), dtype=float)
    lines = load_sou(source_file("eu152.sou"))
    outcome = auto_calibrate.calibrate(channel_indices(len(counts)), counts, lines)
    return counts, lines, outcome


def test_the_strongest_line_is_identified(eu152):
    _counts, _lines, outcome = eu152
    assert outcome.match.ok, outcome.match.reason
    assigned = {round(e, 3): c for c, e in outcome.pairs}
    assert 121.782 in assigned, "121.78 keV -- the strongest peak -- is unassigned"
    assert assigned[121.782] == pytest.approx(194.8, abs=1.0)


def test_the_calibration_is_the_known_one(eu152):
    """This detector runs at ~0.625 keV/channel. A wrong pairing gives a
    plausible-looking different number, which is the whole danger."""
    _counts, _lines, outcome = eu152
    assert outcome.match.gain == pytest.approx(0.625, rel=0.002)
    assert abs(outcome.match.offset) < 1.0
    assert outcome.match.rms < 0.3


def test_the_broad_compton_feature_is_set_aside_not_fitted(eu152):
    _counts, _lines, outcome = eu152
    assert outcome.fits.broad >= 5
    assert not any(abs(p.position - 187) < 2 for p in outcome.peaks)


def test_the_lines_a_broad_feature_used_to_hide_are_all_found(eu152):
    """The second bug report: 344.28, 367.79, 411.12, 416.02 and 443.97
    keV were all missing, 344.28 being the cleanest strong line in the
    spectrum. Every one of them sat behind a Compton structure at
    channel 643 whose measured width -- 56 channels -- blanked out 169
    channels of spectrum to the background search, or behind a fit
    window floored back out onto a stronger neighbour."""
    _counts, _lines, outcome = eu152
    assigned = {round(e, 4): c for c, e in outcome.pairs}
    for energy, channel in ((344.2785, 551.0), (367.7891, 588.7),
                            (411.1165, 658.0), (416.0200, 665.8),
                            (443.9653, 710.6)):
        assert energy in assigned, f"{energy} keV is unassigned again"
        assert assigned[energy] == pytest.approx(channel, abs=1.0)


def test_no_peak_is_fitted_twice(eu152):
    """Two fits landing on one peak let the matcher give it two
    energies."""
    _counts, _lines, outcome = eu152
    positions = sorted(p.position for p in outcome.peaks)
    assert all(b - a > 1.0 for a, b in zip(positions, positions[1:]))


def test_coverage_on_a_crowded_real_spectrum(eu152):
    """The first flat-background search left 35 of 68 photopeaks
    unfitted and matched 15 lines. These floors pin the tuned version."""
    _counts, _lines, outcome = eu152
    assert outcome.fits.skipped <= 2
    assert len(outcome.pairs) >= 32


def test_the_areas_trace_one_efficiency_curve(eu152):
    _counts, _lines, outcome = eu152
    assert outcome.match.efficiency_scatter is not None
    assert outcome.match.efficiency_scatter < 0.15
    assert outcome.match.suspect == []


def test_a_wrong_pairing_fails_the_efficiency_test(eu152):
    """CONTROL: shift every assignment to the neighbouring source line.
    The areas are then divided by the wrong intensities and must scatter
    far more than the true pairing does -- otherwise the test could not
    tell a right calibration from a wrong one."""
    _counts, lines, outcome = eu152
    energies = sorted(line.energy for line in lines)
    intensity = {line.energy: line.intensity for line in lines}
    peaks = outcome.peaks

    def scatter(pairs):
        e = [energy for _k, energy in pairs]
        a = [peaks[k].area for k, _e in pairs]
        i = [intensity[energy] for _k, energy in pairs]
        return auto_calibrate.efficiency_scatter(e, a, i)[0]

    shifted = []
    for k, energy in outcome.match.pairs:
        j = min(energies.index(energy) + 1, len(energies) - 1)
        shifted.append((k, energies[j]))
    good, bad = scatter(outcome.match.pairs), scatter(shifted)
    assert bad > 3 * good
    assert bad > auto_calibrate.EFFICIENCY_MAX_SCATTER


def test_efficiency_scatter_needs_enough_points():
    rms, residuals = auto_calibrate.efficiency_scatter([100.0, 200.0], [1.0, 2.0], [1.0, 1.0])
    assert rms is None
    assert np.all(np.isnan(residuals))


def test_refit_covers_the_visible_source_lines(eu152):
    counts, lines, outcome = eu152
    used = [(c, e) for c, e in outcome.pairs if e not in outcome.suspect_energies]
    cal = C.from_points([c for c, _e in used], [e for _c, e in used], False)
    refit = auto_calibrate.refit_source_lines(
        channel_indices(len(counts)), counts, lines, cal)

    assert len(refit.results) >= 32
    # The line is always peaks[0]. Anything after it is a listed line with
    # no found peak of its own, fitted beside it and pinned exactly where
    # the calibration puts that line -- one per `alongside`, and never
    # anything else.
    extra = [peak for result in refit.results for peak in result.peaks[1:]]
    assert len(extra) == refit.alongside
    pinned = [cal.invert(line.energy) for line in lines]
    for peak in extra:
        assert min(abs(peak.position - c) for c in pinned) < 1e-9
        assert peak.position_err == 0.0
    energies = [e for _c, e in refit.pairs]
    assert any(abs(e - 121.7817) < 1e-3 for e in energies)
    # every fitted line is a line of the source, each at most once
    source = {line.energy for line in lines}
    assert all(e in source for e in energies)
    assert len(set(energies)) == len(energies)
    # each fitted centroid sits where the calibration says its line is
    for (channel, energy), result in zip(refit.pairs, refit.results):
        assert channel == result.peaks[0].position
        assert abs(cal.invert(energy) - channel) < 4.0
    # every line of the source is accounted for exactly once
    accounted = (refit.outside + refit.invisible + refit.alongside + refit.blended
                 + refit.skipped + refit.failed + refit.runaway + len(refit.results))
    assert accounted == len(lines)
    # This spectrum has an unresolved pair (674.64 / 678.62 keV, six
    # channels apart) whose single-peak fit slides onto the stronger
    # line; the guard must catch it rather than export it.
    assert refit.runaway >= 1, "a fit that slid onto a neighbour was exported"
    assert "ran onto a neighbour" in refit.summary("eu152.sou")


def test_refit_puts_lines_the_calibration_places_off_the_end_outside(eu152):
    counts, lines, _outcome = eu152
    off_the_end = C.Calibration(kind="linear", a=0.0, b=0.01)   # 4096 ch -> 41 keV
    refit = auto_calibrate.refit_source_lines(channel_indices(len(counts)), counts,
                                              lines, off_the_end)
    assert refit.results == []
    assert refit.outside == len(lines)
    assert "outside the spectrum" in refit.summary("eu152.sou")


def test_refit_with_no_peaks_reports_every_line_invisible():
    flat = np.full(500, 30.0)
    lines = load_sou(source_file("co56.sou"))
    cal = C.Calibration(kind="linear", a=0.0, b=10.0)
    refit = auto_calibrate.refit_source_lines(channel_indices(500), flat, lines, cal)
    assert refit.results == []
    assert refit.invisible == len(lines)
    text = refit.summary("co56.sou")
    assert "0 lines of co56.sou refitted" in text
    assert "with no visible peak" in text


# --- points the user unticked stay out of the refit ---------------------


def _refit(counts, lines, excluded=()):
    cal = C.Calibration(kind="linear", a=0.0, b=0.6249)
    return auto_calibrate.refit_source_lines(
        channel_indices(len(counts)), counts, lines, cal, excluded=excluded)


def _accounted(refit, lines):
    return (refit.outside + refit.invisible + refit.alongside + refit.blended
            + refit.skipped + refit.failed + refit.runaway + refit.excluded
            + len(refit.results)) == len(lines)


def test_an_unticked_line_is_not_refitted_and_not_exported(eu152):
    """Unticking in an automatic run means the area does not sit on the
    efficiency curve, and the refit exists to measure areas for exactly
    that curve. Exporting it anyway would feed CalEnEff the one number
    the run had already judged wrong."""
    counts, lines, _outcome = eu152
    kept = _refit(counts, lines)
    dropped = _refit(counts, lines, excluded=(344.2785,))

    assert any(abs(e - 344.2785) < 1e-6 for _c, e in kept.pairs)
    assert not any(abs(e - 344.2785) < 1e-6 for _c, e in dropped.pairs)
    assert len(dropped.results) == len(kept.results) - 1
    assert dropped.excluded == 1
    assert _accounted(dropped, lines)
    assert "1 left unticked in the calibration" in dropped.summary("eu152.sou")


def test_a_rounded_energy_still_names_its_line(eu152):
    """The cell may hold what the user typed rather than the line's own
    value to seven figures."""
    counts, lines, _outcome = eu152
    refit = _refit(counts, lines, excluded=(344.28, 1408.01))
    assert refit.excluded == 2
    for energy in (344.2785, 1408.013):
        assert not any(abs(e - energy) < 1e-6 for _c, e in refit.pairs)


def test_an_energy_belonging_to_no_line_excludes_nothing(eu152):
    """CONTROL: the matching is by nearest line within a tolerance, so a
    value from nowhere must leave the refit untouched rather than take
    whichever line happens to be closest."""
    counts, lines, _outcome = eu152
    kept = _refit(counts, lines)
    stray = _refit(counts, lines, excluded=(999.0,))
    assert stray.excluded == 0
    assert len(stray.results) == len(kept.results)


def test_an_unticked_line_still_claims_its_peak(eu152):
    """963.37 and 964.06 keV are one peak at this resolution and the
    stronger of them owns it. Unticking the stronger must not hand the
    peak -- and its whole area -- to the weaker."""
    counts, lines, _outcome = eu152
    refit = _refit(counts, lines, excluded=(964.057,))
    assert not any(abs(e - 964.057) < 1e-6 for _c, e in refit.pairs)
    assert not any(abs(e - 963.367) < 1e-6 for _c, e in refit.pairs)
    assert _accounted(refit, lines)


def test_every_line_unticked_leaves_a_spectrum_with_no_peaks_accounted_for():
    flat = np.full(500, 30.0)
    lines = load_sou(source_file("co56.sou"))
    cal = C.Calibration(kind="linear", a=0.0, b=10.0)
    refit = auto_calibrate.refit_source_lines(
        channel_indices(500), flat, lines, cal,
        excluded=[line.energy for line in lines])
    assert refit.results == []
    assert refit.excluded == len(lines)
    assert refit.invisible == 0
    assert _accounted(refit, lines)


# --- the refit judges a fit against the same trend the first pass does ---


def test_the_refit_guard_uses_the_fitted_width_trend(eu152):
    """Both passes claim to apply "the same judgement". The refit used to
    measure against the trend through the SEARCH's widths, which are read
    off a smoothed copy and run about 1.8 times wider -- 5.03 channels
    against 2.83 at channel 551 of this spectrum -- so the export, the
    more consequential of the two paths, had the weaker gate."""
    counts, lines, _outcome = eu152
    cal = C.Calibration(kind="linear", a=0.0, b=0.6249)
    refit = auto_calibrate.refit_source_lines(
        channel_indices(len(counts)), counts, lines, cal)

    positions = [r.peaks[0].position for r in refit.results]
    widths = [r.peaks[0].fwhm for r in refit.results]
    trend = auto_calibrate.expected_widths(positions, widths)
    for result, expected in zip(refit.results, trend):
        peak = result.peaks[0]
        assert peak.fwhm <= auto_calibrate.WIDTH_OUTLIER_RATIO * expected, (
            f"a {peak.fwhm:.2f}-channel fit at {peak.position:.1f} was exported "
            f"against a {expected:.2f}-channel trend")


def test_the_refit_still_accounts_for_every_line(eu152):
    """The guard moved to a second pass; the arithmetic must survive it."""
    counts, lines, _outcome = eu152
    cal = C.Calibration(kind="linear", a=0.0, b=0.6249)
    refit = auto_calibrate.refit_source_lines(
        channel_indices(len(counts)), counts, lines, cal)
    total = (refit.outside + refit.invisible + refit.alongside + refit.blended
             + refit.skipped + refit.failed + refit.runaway + refit.excluded
             + len(refit.results))
    assert total == len(lines)
    assert refit.runaway >= 1, "the 674.64/678.62 stray is still supposed to be caught"


def test_the_points_share_exactly_the_lines_the_refit_counted_as_blended(eu152):
    """The status bar says "N blended into a stronger neighbour, the area
    shared by intensity", counted by the refit from the found peaks; the
    sharing itself is done afterwards, by caleneff_export's rule from the
    fitted peaks. On this real spectrum the two must name the same line
    -- 963.37 keV inside 964.06 keV -- or the message promises a split
    the points do not make."""
    from caleneff_export import blend_partners, committed_peaks

    counts, lines, _outcome = eu152
    refit = _refit(counts, lines)
    peaks, energies = committed_peaks(
        refit.results,
        {id(r.peaks[0]): e for r, (_c, e) in zip(refit.results, refit.pairs)})
    cal = C.Calibration(kind="linear", a=0.0, b=0.6249)
    partners = blend_partners(peaks, energies, lines, cal)
    shared = sorted(line.energy for group in partners.values() for line in group)
    assert refit.blended == len(shared) == 1
    assert abs(shared[0] - 963.37) < 0.01
    named = [energies[i] for i in partners]
    assert len(named) == 1 and abs(named[0] - 964.057) < 1e-3
