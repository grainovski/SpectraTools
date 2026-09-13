"""The matcher that decides which fitted peak is which source line with
no calibration to go on, and the fitting pass that feeds it.

Every spectrum here is built from a real `.sou` file by
synthetic_calibration, so the calibration is known and each assignment
can be checked against it. The wrong-source cases are the control: a
matcher that "succeeds" on those is not matching, it is guessing.
"""

import math

import numpy as np
import pytest

import auto_calibrate
import peak_fit
import peak_search
from auto_calibrate import MIN_INLIERS, MatchResult, expected_widths, match
from peak_fit import FitError
from sou_io import load_sou
from synthetic_calibration import (
    fixture_sou,
    spectrum_from_source,
    true_fwhm,
    wrong_assignments,
)

#: (source, offset, gain) for spectra that MUST calibrate. Gains and
#: offsets are varied so that no single arithmetic coincidence can
#: satisfy them all.
CALIBRATES = [
    ("eu152.sou", 0.0, 0.25),
    ("eu152.sou", 12.5, 0.40),
    ("ba133.sou", 3.0, 0.12),
    ("ra226.sou", -5.0, 0.35),
    ("co56.sou", 0.0, 0.50),
    ("ta182.sou", 1.0, 0.20),
]

#: (spectrum source, offered source, offset, gain): the nuclide in the
#: spectrum is not the one in the file, and the matcher must say so.
WRONG_SOURCE = [
    ("eu152.sou", "ba133.sou", 0.0, 0.25),
    ("co56.sou", "am241.sou", 0.0, 0.50),
    ("ba133.sou", "co56.sou", 3.0, 0.12),
]


def _lines(name):
    lines = load_sou(fixture_sou(name))
    return [line.energy for line in lines], [line.intensity for line in lines]


def _search_match(spectrum_sou, offered_sou, offset, gain):
    counts = spectrum_from_source(fixture_sou(spectrum_sou), offset, gain)
    found = peak_search.search(counts)
    energies, intensities = _lines(offered_sou)
    result = match([p.channel for p in found], [p.fwhm for p in found],
                   [p.prominence for p in found], energies, intensities)
    return found, result


def _pipeline(spectrum_sou, offset, gain):
    """search -> fit -> match, exactly as the dialog runs it."""
    counts = spectrum_from_source(fixture_sou(spectrum_sou), offset, gain)
    x = np.arange(len(counts), dtype=float)
    found = peak_search.search(counts)
    fitted = auto_calibrate.fit_found_peaks(x, counts, found)
    peaks = [p for _i, p in fitted.peaks]
    energies, intensities = _lines(spectrum_sou)
    result = match([p.position for p in peaks], [p.fwhm for p in peaks],
                   [p.area for p in peaks], energies, intensities)
    return fitted, peaks, result


# --- matching from the search alone ------------------------------------


@pytest.mark.parametrize("sou, offset, gain", CALIBRATES)
def test_every_source_is_calibrated_with_no_wrong_assignment(sou, offset, gain):
    found, result = _search_match(sou, sou, offset, gain)
    assert result.ok, result.reason
    assert result.gain == pytest.approx(gain, rel=1e-3)
    assert result.offset == pytest.approx(offset, abs=1.0)
    assert len(result.pairs) >= MIN_INLIERS
    positions = [p.channel for p in found]
    assert wrong_assignments(result.pairs, positions, offset, gain) == []


@pytest.mark.parametrize("spectrum_sou, offered_sou, offset, gain", WRONG_SOURCE)
def test_the_wrong_source_is_refused(spectrum_sou, offered_sou, offset, gain):
    """The control. A refusal carries a reason for the status line and no
    pairs; `ok` must be False even though pairings were tried."""
    _found, result = _search_match(spectrum_sou, offered_sou, offset, gain)
    assert not result.ok
    assert result.reason
    assert result.pairs == []


def test_too_few_peaks_is_refused_before_anything_is_tried():
    result = match([100.0, 200.0, 300.0], [5.0] * 3, [1.0] * 3, [100.0, 200.0, 300.0])
    assert not result.ok
    assert "at least" in result.reason
    assert result.considered == 0


def test_a_source_with_one_line_is_refused():
    result = match([100.0, 200.0, 300.0, 400.0], [5.0] * 4, [1.0] * 4, [661.657])
    assert not result.ok
    assert "fewer than two lines" in result.reason


def test_result_repr_names_the_outcome():
    assert "failed" in repr(MatchResult(reason="no"))
    assert "3 pairs" in repr(MatchResult(pairs=[(0, 1.0), (1, 2.0), (2, 3.0)],
                                        gain=0.5, offset=0.0, rms=0.1))


# --- the FWHM tolerance ------------------------------------------------
#
# Diagnosed on the Ra-226 spectrum below: three assignments were wrong by
# 10-18 keV, every one of them a fit component that had run away to a
# width of 120-150 channels against a local trend of 10-18. The inlier
# tolerance was that peak's OWN fitted width times the gain, so a bad fit
# bought itself a tolerance of 20 keV and was matched to a line far away.


def _ra226_reference_peaks(gain=0.35):
    """The twelve strongest Ra-226 lines as perfectly fitted peaks."""
    lines = load_sou(fixture_sou("ra226.sou"))
    strongest = sorted(lines, key=lambda line: -line.intensity)[:12]
    channels = [line.energy / gain for line in strongest]
    return (channels, [true_fwhm(c) for c in channels],
            [line.intensity for line in strongest],
            [line.energy for line in lines], [line.intensity for line in lines])


def test_a_runaway_width_is_not_assigned_to_a_line_far_away():
    """A fit component 150 channels wide with a positive area is not a
    peak, whatever its centroid says. Under the old rule its tolerance
    was 0.35 * 150 * 0.5 = 26 keV, enough to reach 1729.6 keV from a
    centroid 18 keV away."""
    channels, fwhms, weights, energies, intensities = _ra226_reference_peaks()
    junk = len(channels)
    channels.append(1747.8 / 0.35)
    fwhms.append(150.0)
    weights.append(float(np.median(weights)))

    result = match(channels, fwhms, weights, energies, intensities)
    assert result.ok, result.reason
    assert all(k != junk for k, _e in result.pairs), "the runaway width was assigned"
    assert result.unusable == 1
    for k, energy in result.pairs:
        assert energy == pytest.approx(0.35 * channels[k], abs=1e-6)


def test_an_inflated_width_does_not_buy_a_looser_tolerance():
    """Below the refusal ratio the peak is still assignable, but on the
    resolution its neighbours establish, not on its own width. At 1.5x
    the local width the old tolerance was 4.6 keV; the trend's is 3.1,
    and a centroid 3.5 keV from the nearest line is left alone. The line
    probed, 1729.6 keV, is not among the twelve reference peaks, so
    nothing but the tolerance decides."""
    channels, fwhms, weights, energies, intensities = _ra226_reference_peaks()
    suspect = len(channels)
    channels.append((1729.595 + 3.5) / 0.35)
    fwhms.append(1.5 * true_fwhm(channels[-1]))
    weights.append(float(np.median(weights)))

    result = match(channels, fwhms, weights, energies, intensities)
    assert result.ok, result.reason
    assert all(k != suspect for k, _e in result.pairs)
    assert result.unusable == 0


def test_a_negative_area_is_not_a_peak():
    """Sitting exactly on a free line, with a plausible width: only the
    sign of the area says this component is modelling background, and
    that alone must keep it out of the calibration."""
    channels, fwhms, weights, energies, intensities = _ra226_reference_peaks()
    junk = len(channels)
    channels.append(1729.595 / 0.35)
    fwhms.append(true_fwhm(channels[-1]))
    weights.append(-250000.0)
    result = match(channels, fwhms, weights, energies, intensities)
    assert result.ok, result.reason
    assert all(k != junk for k, _e in result.pairs)
    assert result.unusable == 1


def test_too_few_usable_widths_is_refused_with_a_reason():
    channels, fwhms, weights, energies, intensities = _ra226_reference_peaks()
    fwhms = [math.nan] * (len(fwhms) - 3) + fwhms[-3:]
    result = match(channels, fwhms, weights, energies, intensities)
    assert not result.ok
    assert "plausible width" in result.reason
    assert result.unusable == len(channels) - 3


def test_expected_widths_follow_the_trend_through_a_runaway_outlier():
    channels = np.linspace(200.0, 7000.0, 9)
    fwhms = true_fwhm(channels)
    fwhms[3] = 130.0
    expected = expected_widths(channels, fwhms)
    assert expected == pytest.approx(true_fwhm(channels), rel=0.05)


def test_expected_widths_fall_back_to_the_median_with_too_few_peaks():
    expected = expected_widths([100.0, 5000.0], [4.0, 20.0])
    assert expected == pytest.approx([12.0, 12.0])


def test_expected_widths_ignore_unusable_widths_and_never_reach_zero():
    channels = np.array([100.0, 2000.0, 4000.0, 6000.0, 8000.0])
    fwhms = np.array([math.nan, 0.0, -1.0, 18.0, 22.0])
    expected = expected_widths(channels, fwhms)
    assert np.all(np.isfinite(expected))
    assert np.all(expected > 0.0)


@pytest.mark.parametrize("sou, offset, gain", [
    ("ra226.sou", -5.0, 0.35),
    ("eu152.sou", 12.5, 0.40),
    ("co56.sou", 0.0, 0.50),
])
def test_the_full_pipeline_makes_no_wrong_assignment(sou, offset, gain):
    """Fitted centroids, fitted widths, fitted areas -- the inputs the
    dialog really passes. Before the tolerance was made robust this case
    produced three wrong assignments on Ra-226, two on Eu-152 and one on
    Co-56, each a runaway fit component matched 10-18 keV away."""
    _fitted, peaks, result = _pipeline(sou, offset, gain)
    assert result.ok, result.reason
    positions = [p.position for p in peaks]
    assert wrong_assignments(result.pairs, positions, offset, gain) == []
    assert result.gain == pytest.approx(gain, rel=1e-3)
    # Setting junk aside must not cost the calibration its real peaks.
    assert len(result.pairs) >= 0.8 * len(peaks)


# --- the fitting pass --------------------------------------------------


def _eight_line_spectrum():
    rng = np.random.default_rng(3)
    x = np.arange(4096, dtype=float)
    clean = 400.0 * np.exp(-x / 1200.0) + 40.0
    centres = [300.0, 700.0, 1100.0, 1500.0, 2000.0, 2600.0, 3200.0, 3800.0]
    for centre in centres:
        sigma = 1.5 + 0.0012 * centre
        clean = clean + 20000.0 * np.exp(-((x - centre) ** 2) / (2 * sigma * sigma))
    return x, rng.poisson(clean).astype(float), centres


def test_fit_found_peaks_fits_every_photopeak_on_its_own():
    """One fit per peak, each with its own markers -- what a person does
    by hand. Grouping neighbours was dropped after it chained the
    strongest line of a real Eu-152 spectrum to a broad Compton feature
    and lost it (see test_auto_calibrate_real.py)."""
    x, y, centres = _eight_line_spectrum()
    found = peak_search.search(y)
    outcome = auto_calibrate.fit_found_peaks(x, y, found)
    photopeaks, _broad = peak_search.reject_broad(found)
    assert outcome.attempted == len(photopeaks)
    assert outcome.failed == 0
    assert outcome.skipped == 0
    assert len(outcome.results) == outcome.attempted
    assert all(len(result.peaks) == 1 for result in outcome.results)
    fitted = sorted(p.position for _i, p in outcome.peaks)
    assert fitted == pytest.approx(centres, abs=0.5)
    # (result index, peak) pairs must index the results they came from.
    for index, peak in outcome.peaks:
        assert peak in outcome.results[index].peaks


def test_fit_found_peaks_skips_a_peak_with_no_background_room_at_the_edge():
    x, y, _centres = _eight_line_spectrum()
    found = peak_search.search(y)
    found.append(peak_search.FoundPeak(channel=15.0, fwhm=4.0, prominence=500.0,
                                       significance=20.0, height=600.0))
    outcome = auto_calibrate.fit_found_peaks(x, y, found)
    assert outcome.skipped == 1
    photopeaks, _broad = peak_search.reject_broad(found)
    assert outcome.attempted == len(photopeaks) - 1


def test_one_failing_fit_does_not_abort_the_rest(monkeypatch):
    x, y, _centres = _eight_line_spectrum()
    found = peak_search.search(y)
    real = peak_fit.fit_peaks
    calls = []

    def flaky(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise FitError("did not converge")
        return real(*args, **kwargs)

    monkeypatch.setattr(peak_fit, "fit_peaks", flaky)
    outcome = auto_calibrate.fit_found_peaks(x, y, found)
    assert outcome.failed == 1
    assert outcome.attempted == len(calls)
    assert len(outcome.results) == (outcome.attempted - outcome.failed
                                    - outcome.runaway - outcome.duplicate)


def test_fit_found_peaks_passes_the_spectrum_variance_through(monkeypatch):
    """A matrix cut or an Add/Subtract result is not Poisson in its own
    counts; the fitter is told so through `variance`, exactly as the
    interactive Fit button tells it."""
    x, y, _centres = _eight_line_spectrum()
    found = peak_search.search(y)[:1]
    seen = []
    real = peak_fit.fit_peaks

    def spy(*args, **kwargs):
        seen.append(kwargs.get("variance"))
        return real(*args, **kwargs)

    monkeypatch.setattr(peak_fit, "fit_peaks", spy)
    variance = 2.0 * np.maximum(y, 1.0)
    auto_calibrate.fit_found_peaks(x, y, found, variance=variance)
    assert len(seen) == 1
    assert seen[0] is variance


# --- what the settling pass throws out ----------------------------------


def _fake_fit(position, fwhm, area):
    peak = peak_fit.PeakResult(position=position, position_err=0.01, fwhm=fwhm,
                               fwhm_err=0.01, area=area, area_err=1.0,
                               amplitude=area / max(fwhm, 1e-9), sigma=fwhm / 2.3548)
    return peak_fit.FitResult(left_bg_region=(position - 30, position - 20),
                              right_bg_region=(position + 20, position + 30),
                              fit_region=(position - 10, position + 10),
                              background_slope=0.0, background_intercept=1.0,
                              peaks=[peak])


def _seed(channel):
    return peak_search.FoundPeak(channel=channel, fwhm=4.0, prominence=100.0,
                                 significance=20.0, height=100.0)


def _trend(n, start=100.0, step=100.0, fwhm=4.0):
    """n well-behaved fits, evenly spaced and all the same width, so that
    the width trend is flat and anything added to them is judged against
    a known expectation."""
    return [(_seed(start + step * i), _fake_fit(start + step * i, fwhm, 1000.0))
            for i in range(n)]


def test_settle_keeps_fits_that_landed_where_they_were_seeded():
    kept, runaway, duplicate = auto_calibrate._settle(_trend(8))
    assert len(kept) == 8
    assert (runaway, duplicate) == (0, 0)


def test_settle_drops_a_fit_that_ran_onto_its_neighbour():
    """The 416.02 keV case: seeded on one peak, converged on the strong
    line beside it. Committing it would put a second, spurious peak on
    top of one already fitted and let the matcher name it."""
    fitted = _trend(8)
    fitted.append((_seed(850.0), _fake_fit(841.0, 4.0, 5000.0)))
    kept, runaway, duplicate = auto_calibrate._settle(fitted)
    assert runaway == 1
    assert duplicate == 0
    assert all(abs(r.peaks[0].position - 841.0) > 1.0 for r in kept)


def test_settle_drops_a_fit_far_wider_than_the_trend():
    fitted = _trend(8)
    fitted.append((_seed(850.0), _fake_fit(850.0, 40.0, 5000.0)))
    kept, runaway, _duplicate = auto_calibrate._settle(fitted)
    assert runaway == 1
    assert all(r.peaks[0].fwhm < 40.0 for r in kept)


def test_settle_drops_a_fit_with_no_positive_area():
    fitted = _trend(8)
    fitted.append((_seed(850.0), _fake_fit(850.0, 4.0, -20.0)))
    kept, runaway, _duplicate = auto_calibrate._settle(fitted)
    assert runaway == 1
    assert all(r.peaks[0].area > 0 for r in kept)


def test_settle_keeps_the_stronger_of_two_fits_of_one_peak():
    """Two adjacent found peaks can both converge on the same line. Kept
    both, they gave one physical peak two different energies -- 1084.00
    and 1085.84 keV on a real Eu-152 spectrum."""
    fitted = _trend(8)
    fitted.append((_seed(801.0), _fake_fit(800.4, 4.0, 40.0)))
    kept, runaway, duplicate = auto_calibrate._settle(fitted)
    assert (runaway, duplicate) == (0, 1)
    assert len(kept) == 8
    areas = {r.peaks[0].area for r in kept if abs(r.peaks[0].position - 800.0) < 2}
    assert areas == {1000.0}, "the weaker of the two was the one kept"


def test_settle_keeps_two_peaks_that_are_genuinely_apart():
    """CONTROL for the test above: the same extra fit, moved far enough
    from its neighbour to be a peak of its own, is kept."""
    fitted = _trend(8)
    fitted.append((_seed(803.0), _fake_fit(803.0, 4.0, 40.0)))
    kept, runaway, duplicate = auto_calibrate._settle(fitted)
    assert (runaway, duplicate) == (0, 0)
    assert len(kept) == 9


def test_settle_returns_the_kept_fits_in_channel_order():
    fitted = _trend(6)
    fitted.append((_seed(250.0), _fake_fit(250.0, 4.0, 90000.0)))
    kept, _runaway, _duplicate = auto_calibrate._settle(fitted)
    positions = [r.peaks[0].position for r in kept]
    assert positions == sorted(positions)


def test_settle_on_nothing():
    assert auto_calibrate._settle([]) == ([], 0, 0)


# --- the whole thing, as the dialog runs it -----------------------------


def test_calibrate_runs_search_fit_and_match_together():
    counts = spectrum_from_source(fixture_sou("eu152.sou"), 12.5, 0.40)
    x = np.arange(len(counts), dtype=float)
    outcome = auto_calibrate.calibrate(x, counts, load_sou(fixture_sou("eu152.sou")))
    assert outcome.match.ok, outcome.match.reason
    assert len(outcome.results) == (outcome.fits.attempted - outcome.fits.failed
                                    - outcome.fits.runaway - outcome.fits.duplicate)
    positions = [p.position for p in outcome.peaks]
    assert wrong_assignments(outcome.match.pairs, positions, 12.5, 0.40) == []
    # The pairs the dialog is handed are (channel, energy) on the FITTED
    # centroids, so the assign dialog's restore lands on them exactly.
    assert len(outcome.pairs) == len(outcome.match.pairs)
    for (channel, energy), (k, e) in zip(outcome.pairs, outcome.match.pairs):
        assert channel == outcome.peaks[k].position
        assert energy == e


def test_calibrate_with_the_wrong_source_keeps_the_fits_and_gives_the_reason():
    counts = spectrum_from_source(fixture_sou("eu152.sou"), 0.0, 0.25)
    x = np.arange(len(counts), dtype=float)
    outcome = auto_calibrate.calibrate(x, counts, load_sou(fixture_sou("ba133.sou")))
    assert not outcome.match.ok
    assert outcome.match.reason
    assert outcome.results, "the fits are kept even when nothing is identified"
    assert outcome.pairs == []


def test_calibrate_with_no_peaks_found_says_so():
    x = np.arange(500, dtype=float)
    flat = np.full(500, 20.0)
    outcome = auto_calibrate.calibrate(x, flat, load_sou(fixture_sou("ba133.sou")),
                                       sensitivity=30.0)
    assert outcome.found == []
    assert outcome.results == []
    assert not outcome.match.ok
    assert "no peaks" in outcome.match.reason.lower()


def test_summary_reports_every_count_the_status_line_needs():
    counts = spectrum_from_source(fixture_sou("eu152.sou"), 12.5, 0.40)
    x = np.arange(len(counts), dtype=float)
    outcome = auto_calibrate.calibrate(x, counts, load_sou(fixture_sou("eu152.sou")))
    text = outcome.summary(existing_fits=2, source_name="eu152.sou")
    assert f"{len(outcome.found)} peaks found" in text
    assert f"{outcome.fits.attempted} attempted" in text
    assert f"{outcome.fits.failed} failed" in text
    assert f"{outcome.fits.broad} broad features set aside" in text
    assert f"{outcome.fits.skipped} skipped for want of a clear background" in text
    assert f"{outcome.fits.runaway} that strayed" in text
    assert f"{outcome.fits.duplicate} that repeated another" in text
    assert f"{len(outcome.match.pairs)} identified in eu152.sou" in text
    assert "2 fits already on the spectrum were kept" in text
    # A peak set aside for its width is reported as such, not folded into
    # "unidentified" -- when there is one; with individual fits there may
    # be none. Likewise a point the efficiency test doubts.
    if outcome.match.unusable:
        assert f"{outcome.match.unusable} set aside for an implausible" in text
    if outcome.match.suspect:
        assert f"{len(outcome.match.suspect)} of them left out of the fit as suspect" in text

    refused = auto_calibrate.calibrate(x, counts, load_sou(fixture_sou("ba133.sou")))
    text = refused.summary(existing_fits=0, source_name="ba133.sou")
    assert "none identified" in text
    assert refused.match.reason in text
    assert "no fits were on the spectrum before" in text


# --- saying which way the peaks were lost --------------------------------


def test_no_peaks_survived_says_which_way_they_went(monkeypatch):
    """"Could not be fitted" reads as a solver failure and usually is
    not: on a crowded spectrum the fits succeed and are then dropped for
    landing somewhere other than the peak they were seeded on."""
    x, y, _centres = _eight_line_spectrum()

    def all_stray(fitted):
        return [], len(fitted), 0

    monkeypatch.setattr(auto_calibrate, "_settle", all_stray)
    outcome = auto_calibrate.calibrate(x, y, load_sou(fixture_sou("eu152.sou")))

    assert not outcome.match.ok
    assert "whose fit strayed" in outcome.match.reason
    assert outcome.fits.runaway > 0


def test_the_reason_stays_short_when_there_is_nothing_to_add(monkeypatch):
    """CONTROL: with no counts to report, the sentence is the plain one
    and gains no trailing colon."""
    x, y, _centres = _eight_line_spectrum()
    monkeypatch.setattr(auto_calibrate, "_settle", lambda fitted: ([], 0, 0))
    monkeypatch.setattr(peak_search, "reject_broad", lambda found, **k: (list(found), []))
    outcome = auto_calibrate.calibrate(x, y, load_sou(fixture_sou("eu152.sou")))
    assert outcome.match.reason.endswith("peaks found could be fitted")


# --- the matcher's candidate lookup at the ends of the line list ---------


def test_a_peak_predicted_below_every_line_still_matches_the_first(monkeypatch):
    """The three candidates around the insertion point include one that
    is out of range at each end of the array. Vectorising that lookup
    must keep an out-of-range neighbour out of the running rather than
    wrapping to the other end."""
    result = match([10.0, 20.0, 30.0, 40.0, 50.0], [2.0] * 5, [100.0] * 5,
                   [100.0, 200.0, 300.0, 400.0, 500.0])
    assert result.ok
    assigned = dict(result.pairs)
    assert assigned[0] == pytest.approx(100.0)
    assert assigned[4] == pytest.approx(500.0)


def test_the_reported_scatter_excludes_the_points_it_calls_suspect():
    """A suspect point is removed from the calibration fit and from the
    export, so the scatter that is reported alongside it must describe
    the curve that survives -- not the one the bad point distorted.

    Measured on the real Eu-152 spectrum: one peak in a crowded triplet
    was flagged suspect, correctly, and yet still counted in the number.
    It alone carried the scatter from 0.045 to 0.078, which is what made
    a better-fitting peak shape look worse than it was.
    """
    channels, fwhms, weights, energies, intensities = _ra226_reference_peaks()
    # One peak whose AREA is far off the efficiency curve while its
    # position is exactly right, so it is assigned and then doubted.
    strayed = len(channels)
    channels.append(1729.595 / 0.35)
    fwhms.append(true_fwhm(channels[-1]))
    # 4x, not 40x: enough to be doubted, not enough to trip the hard
    # EFFICIENCY_MAX_SCATTER gate, which refuses the whole assignment
    # before suspects are ever computed.
    weights.append(float(np.median(weights)) * 4.0)
    energies = list(energies) + [1729.595]
    intensities = list(intensities) + [float(np.median(intensities))]

    result = match(channels, fwhms, weights, energies, intensities)
    assert result.ok, result.reason
    assert any(k == strayed for k, _e in result.pairs), "the probe was not assigned"
    assert result.suspect, "an area 4x off the curve should be suspect"

    kept = [(k, e) for k, e in result.pairs if (k, e) not in result.suspect]
    rms_kept, _r = auto_calibrate.efficiency_scatter(
        [e for _k, e in kept],
        [weights[k] for k, _e in kept],
        [intensities[energies.index(e)] for _k, e in kept],
    )
    assert result.efficiency_scatter == pytest.approx(rms_kept, rel=1e-9), (
        f"reported {result.efficiency_scatter:.4f} but the surviving points "
        f"scatter by {rms_kept:.4f}"
    )
