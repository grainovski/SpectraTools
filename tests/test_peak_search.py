"""Finding peaks in raw counts: what is found, what is not, and how the
found peaks are grouped and given regions to fit in.

The spectrum is synthetic with the answer known, so "found" and
"spurious" are facts rather than impressions.
"""

import numpy as np
import pytest

import os

from peak_search import (
    DEFAULT_SENSITIVITY, FoundPeak, background_window, fit_window, flatness,
    regions, reject_broad, search,
)
from spe_io import load_spe

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

#: (centre, amplitude) of the eight lines, Eu-152-like in their spread of
#: strengths: the weakest is forty times below the strongest.
TRUTH = [(600, 40000), (1200, 12000), (1700, 30000), (2400, 3000),
         (3900, 9000), (4800, 6000), (5500, 1500), (6950, 14000)]


def _sigma(channel):
    return 2.0 + 0.0009 * channel


def _spectrum(seed=1, channels=8192):
    rng = np.random.default_rng(seed)
    x = np.arange(channels, dtype=float)
    clean = 3000.0 * np.exp(-x / 2500.0) + 60.0
    for centre, amplitude in TRUTH:
        s = _sigma(centre)
        clean = clean + amplitude * np.exp(-((x - centre) ** 2) / (2 * s * s))
    return rng.poisson(clean).astype(float)


def _real_and_spurious(found, within=5.0):
    real = [c for c, _a in TRUTH if any(abs(p.channel - c) < within for p in found)]
    spurious = [p for p in found
                if not any(abs(p.channel - c) < within for c, _a in TRUTH)]
    return real, spurious


# --- searching ---------------------------------------------------------


def test_the_default_sensitivity_finds_every_line_and_nothing_else():
    found = search(_spectrum())
    real, spurious = _real_and_spurious(found)
    assert len(real) == len(TRUTH)
    assert spurious == []
    assert len(found) == len(TRUTH)


def test_the_default_is_what_search_uses_when_not_told():
    counts = _spectrum()
    assert [p.channel for p in search(counts)] == \
        [p.channel for p in search(counts, DEFAULT_SENSITIVITY)]


def test_lower_sensitivity_finds_more_and_higher_finds_fewer():
    counts = _spectrum()
    counts_found = [len(search(counts, s)) for s in (2.0, 5.0, 15.0, 60.0)]
    assert counts_found == sorted(counts_found, reverse=True)
    assert counts_found[0] > counts_found[-1]


def test_the_threshold_is_applied_to_the_reported_significance():
    """The weakest line is the 1500-count one at channel 5500. A
    threshold a hair above its reported significance must drop exactly
    it and nothing else -- which is only true if the number reported is
    the number the threshold is compared against."""
    counts = _spectrum()
    found = search(counts)
    weakest = min(found, key=lambda p: p.significance)
    assert weakest.channel == pytest.approx(5500, abs=5)
    above = search(counts, sensitivity=weakest.significance + 1e-6)
    assert all(abs(p.channel - weakest.channel) > 5 for p in above)
    assert len(above) == len(found) - 1


def test_positions_and_widths_are_close_to_the_truth():
    """Read off the smoothed copy, so not exact -- but grouping and
    region placement only need them within a quarter, and the fit that
    follows measures them properly."""
    for peak in search(_spectrum()):
        centre = min((c for c, _a in TRUTH), key=lambda c: abs(c - peak.channel))
        assert peak.channel == pytest.approx(centre, abs=2.0)
        assert peak.fwhm == pytest.approx(2.3548 * _sigma(centre), rel=0.25)
        assert peak.height > 0
        assert peak.prominence > 0
        assert peak.significance >= DEFAULT_SENSITIVITY


def test_significance_is_in_sigmas_not_counts():
    """The same spectrum at a hundredth of the counts must still yield
    its strong lines: the threshold is relative to the noise, so a weak
    measurement is searched on the same terms as a strong one."""
    rng = np.random.default_rng(7)
    x = np.arange(8192, dtype=float)
    clean = 30.0 * np.exp(-x / 2500.0) + 0.6
    for centre, amplitude in TRUTH:
        s = _sigma(centre)
        clean = clean + amplitude / 100.0 * np.exp(-((x - centre) ** 2) / (2 * s * s))
    found = search(rng.poisson(clean).astype(float))
    real, spurious = _real_and_spurious(found)
    assert 600 in real and 1700 in real and 6950 in real
    assert spurious == []


def test_noise_alone_yields_nothing_at_the_default_and_plenty_below_it():
    """The control for every "nothing spurious" assertion above: the
    same flat noise must produce peaks when the threshold is lowered,
    or a search that found nothing anywhere would pass them all."""
    flat = np.random.default_rng(11).poisson(200.0, size=8192).astype(float)
    assert search(flat) == []
    assert len(search(flat, sensitivity=1.0)) > 20


def test_empty_short_and_zero_spectra_find_nothing():
    assert search(np.zeros(1000)) == []
    assert search([1.0, 2.0, 3.0]) == []
    assert search([]) == []


def test_found_peaks_come_back_in_channel_order():
    channels = [p.channel for p in search(_spectrum())]
    assert channels == sorted(channels)


def test_repr_reads_as_a_peak():
    text = repr(FoundPeak(channel=600.0, fwhm=5.1, prominence=100.0,
                          significance=12.5, height=140.0))
    assert "600.00" in text and "5.10" in text and "12.5" in text


# --- broad features ----------------------------------------------------


def _peak(channel, fwhm, prominence=100.0):
    return FoundPeak(channel=channel, fwhm=fwhm, prominence=prominence,
                     significance=20.0, height=200.0)


def test_a_feature_far_wider_than_the_trend_is_not_a_photopeak():
    """Channel 187 of the real Eu-152 spectrum: a Compton feature 4.6
    times the width of every real peak around it, which used to be
    grouped with the 121.78 keV line eight channels away and ruin its
    fit. It is set aside, and marked so the windows still avoid it."""
    found = [_peak(c, 5.0) for c in (120.0, 140.0, 195.0, 390.0, 550.0)]
    found.append(_peak(187.0, 22.7))
    peaks, broad = reject_broad(found)
    assert [p.channel for p in broad] == [187.0]
    assert all(p.broad for p in broad)
    assert not any(p.broad for p in peaks)
    assert len(peaks) == 5


def test_real_peaks_within_the_trend_are_all_kept():
    found = [_peak(c, 4.0 + 0.001 * c) for c in range(100, 4000, 300)]
    peaks, broad = reject_broad(found)
    assert broad == []
    assert len(peaks) == len(found)


def test_too_few_candidates_to_judge_keeps_everything():
    peaks, broad = reject_broad([_peak(100.0, 5.0), _peak(200.0, 50.0)])
    assert len(peaks) == 2
    assert broad == []


def test_the_real_eu152_spectrum_keeps_channel_195_and_rejects_187():
    """The bug report itself, at the search level."""
    counts = np.asarray(load_spe(os.path.join(FIXTURES, "eu152_real.spe")), dtype=float)
    peaks, broad = reject_broad(search(counts))
    assert any(abs(p.channel - 195) < 1 for p in peaks)
    assert any(abs(p.channel - 187) < 1 for p in broad)
    assert 5 <= len(broad) <= 12


# --- fit windows -------------------------------------------------------


def test_fit_window_is_three_widths_each_side_when_alone():
    peak = _peak(500.0, 4.0)
    assert fit_window(peak, [peak]) == (488.0, 512.0)


def test_fit_window_is_clipped_at_the_midpoint_to_a_neighbour():
    """A neighbour inside the window would be absorbed into this peak's
    area; the window stops halfway to it, on both sides symmetrically."""
    peak, other = _peak(500.0, 4.0), _peak(510.0, 4.0)
    assert fit_window(peak, [peak, other]) == (488.0, 505.0)
    assert fit_window(other, [peak, other]) == (505.0, 522.0)


def test_fit_window_never_narrows_below_the_floor():
    """A neighbour almost on top of the peak would clip the window to
    nothing; the floor keeps enough of the peak to fit at all."""
    peak, other = _peak(500.0, 4.0), _peak(502.0, 4.0)
    lo, hi = fit_window(peak, [peak, other])
    assert hi == pytest.approx(500.0 + 1.2 * 4.0)
    assert lo == 488.0


# --- background windows ------------------------------------------------


def _flat_spectrum(n=2000, level=400.0, seed=0):
    return np.random.default_rng(seed).poisson(level, n).astype(float)


def test_flatness_is_about_one_for_poisson_noise_and_large_for_a_peak():
    counts = _flat_spectrum()
    quiet = flatness(counts, 800, 830)
    counts[1010:1020] += 3000.0
    bumpy = flatness(counts, 1000, 1030)
    assert 0.3 < quiet < 2.5
    assert bumpy > 10 * quiet


def test_background_windows_avoid_every_found_peak_and_sit_beyond_the_fit_window():
    counts = _flat_spectrum()
    peak = _peak(1000.0, 4.0)
    neighbour = _peak(1022.0, 4.0)      # exactly where the nearest right window would go
    found = [peak, neighbour]
    left = background_window(counts, peak, -1, found)
    right = background_window(counts, peak, +1, found)
    assert left is not None and right is not None
    assert left[1] <= 1000.0 - 3.25 * 4.0
    assert right[0] >= 1000.0 + 3.25 * 4.0
    # clear of the neighbour by one and a half of ITS width
    assert right[0] >= 1022.0 + 1.5 * 4.0


def test_the_nearest_flat_window_is_taken():
    counts = _flat_spectrum()
    peak = _peak(1000.0, 4.0)
    right = background_window(counts, peak, +1, [peak])
    assert right[0] == pytest.approx(1000.0 + 3.25 * 4.0)
    assert right[1] - right[0] == pytest.approx(1.5 * 4.0)


def test_a_bump_in_the_nearest_window_pushes_the_search_outward():
    """Structure the search did not flag as a peak: the window is not
    flat, so the search moves on past it."""
    counts = _flat_spectrum()
    counts[1014:1020] += 2000.0          # inside the first candidate window
    peak = _peak(1000.0, 4.0)
    right = background_window(counts, peak, +1, [peak])
    assert right[0] > 1020.0


def test_no_room_before_the_edge_gives_no_window():
    counts = _flat_spectrum(n=60)
    assert background_window(counts, _peak(12.0, 4.0), -1, []) is None


def test_a_crowded_neighbourhood_falls_back_to_avoiding_only_what_matters():
    """Weak peaks every seven channels leave nothing clear of every found
    peak within reach. The fallback ignores neighbours below a twentieth
    of this peak's prominence -- but never a broad feature."""
    counts = _flat_spectrum()
    peak = _peak(1000.0, 4.0, prominence=10000.0)
    weak = [_peak(float(c), 4.0, prominence=50.0)
            for c in range(880, 1130, 7) if abs(c - 1000) > 8]
    assert background_window(counts, peak, +1, [peak] + weak) is not None

    broad = _peak(1030.0, 20.0, prominence=50.0)
    broad.broad = True
    right = background_window(counts, peak, +1, [peak] + weak + [broad])
    assert right is not None
    assert right[0] >= 1030.0 + 1.5 * 20.0


def test_regions_refuse_a_peak_with_no_background_on_one_side():
    counts = _flat_spectrum(n=200)
    assert regions(_peak(10.0, 4.0), counts, []) is None
    counts = _flat_spectrum()
    assert regions(_peak(1000.0, 4.0), counts, []) is not None


def test_regions_fit_one_position_only():
    counts = _flat_spectrum()
    peak = _peak(1000.0, 4.0)
    _l, _r, fit_region, positions = regions(peak, counts, [peak, _peak(1100.0, 4.0)])
    assert positions == [1000.0]
    assert fit_region == (988.0, 1012.0)


def test_every_photopeak_of_the_synthetic_spectrum_gets_regions():
    counts = _spectrum()
    found = search(counts)
    peaks, _broad = reject_broad(found)
    assert peaks
    assert all(regions(p, counts, found) is not None for p in peaks)
