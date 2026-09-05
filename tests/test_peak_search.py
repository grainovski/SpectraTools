"""Finding peaks in raw counts: what is found, what is not, and how the
found peaks are grouped and given regions to fit in.

The spectrum is synthetic with the answer known, so "found" and
"spurious" are facts rather than impressions.
"""

import numpy as np
import pytest

from peak_search import DEFAULT_SENSITIVITY, FoundPeak, group, regions, search

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


# --- grouping ----------------------------------------------------------


def _peak(channel, fwhm):
    return FoundPeak(channel=channel, fwhm=fwhm, prominence=100.0,
                     significance=20.0, height=200.0)


def test_peaks_within_three_widths_are_grouped_and_others_are_not():
    groups = group([_peak(100.0, 5.0), _peak(112.0, 5.0), _peak(200.0, 5.0)])
    assert [[p.channel for p in g] for g in groups] == [[100.0, 112.0], [200.0]]


def test_grouping_chains_through_a_multiplet():
    groups = group([_peak(100.0, 5.0), _peak(112.0, 5.0), _peak(124.0, 5.0)])
    assert [[p.channel for p in g] for g in groups] == [[100.0, 112.0, 124.0]]


def test_the_wider_peak_decides_the_reach():
    """Ten channels apart: inside three widths of a 5-wide peak, outside
    three widths of a 2-wide one. The wider of the pair is what sets
    whether their tails overlap."""
    assert len(group([_peak(100.0, 2.0), _peak(110.0, 5.0)])) == 1
    assert len(group([_peak(100.0, 2.0), _peak(110.0, 2.0)])) == 2


def test_grouping_sorts_by_channel_and_handles_nothing():
    groups = group([_peak(300.0, 4.0), _peak(100.0, 4.0)])
    assert [[p.channel for p in g] for g in groups] == [[100.0], [300.0]]
    assert group([]) == []


# --- regions -----------------------------------------------------------


def test_regions_are_laid_out_in_widths_around_a_single_peak():
    bg_left, bg_right, fit_region, positions = regions([_peak(500.0, 4.0)], 1000)
    assert bg_left == (500.0 - 26.0, 500.0 - 16.0)
    assert bg_right == (500.0 + 16.0, 500.0 + 26.0)
    assert fit_region == (500.0 - 12.0, 500.0 + 12.0)
    assert positions == [500.0]


def test_regions_span_a_whole_group_using_its_widest_member():
    bg_left, bg_right, fit_region, positions = regions(
        [_peak(500.0, 4.0), _peak(530.0, 6.0)], 1000
    )
    assert fit_region == (500.0 - 18.0, 530.0 + 18.0)
    assert bg_left == (500.0 - 39.0, 500.0 - 24.0)
    assert bg_right == (530.0 + 24.0, 530.0 + 39.0)
    assert positions == [500.0, 530.0]


def test_a_group_too_near_either_edge_is_refused():
    """A background region clipped to the spectrum edge would fit its
    slope to almost nothing; refusing is the honest answer."""
    assert regions([_peak(20.0, 4.0)], 1000) is None
    assert regions([_peak(26.0, 4.0)], 1000) is not None
    assert regions([_peak(980.0, 4.0)], 1000) is None
    assert regions([_peak(973.0, 4.0)], 1000) is not None


def test_regions_of_the_found_peaks_all_have_room_on_the_synthetic_spectrum():
    counts = _spectrum()
    found = search(counts)
    assert all(regions(g, len(counts)) is not None for g in group(found))
