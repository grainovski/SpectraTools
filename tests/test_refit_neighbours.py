"""A line's efficiency point counts that line, not its neighbours.

Two kinds of neighbour used to hand their counts to the line being
measured:

- a weaker listed line inside the line's fit window with no found peak
  of its own -- the automatic refit (auto_calibrate.refit_source_lines)
  now fits it beside the line as a second peak, pinned where the
  calibration puts it, and does not export it;
- a weaker listed line close enough to sit inside the SAME peak -- its
  share of that peak's area is now taken off the line's point by
  caleneff_export.share_blended_areas, the one rule both the automatic
  calibration and Calibrate from Fitted Peaks go through.

On real spectra the first put 18% on top of Eu-152's 563.99 keV line
(566.44 keV beside it) and the second 38% on top of Ra-226's 274.80 keV
(Bi-214 273.79 keV under it).

Every scenario has a CONTROL: the same spectrum with the neighbour left
out of the source list, which reproduces the old number and must be
visibly wrong -- otherwise nothing here could tell the two apart.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import auto_calibrate
from caleneff_export import (
    SHARED_PEAK_FWHM,
    blend_partners,
    committed_peaks,
    efficiency_area_error,
    share_blended_areas,
)
from calibration import Calibration
from peak_fit import channel_indices
from sou_io import SourceLine

SIGMA = 2.0
FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0)) * SIGMA
CHANNELS = 3000
#: energy == channel, so every line sits exactly where it is drawn
IDENTITY = Calibration(kind="linear", a=0.0, b=1.0)
#: Well-separated peaks elsewhere, so the width trend has something to
#: stand on, as it would in a real spectrum.
ISOLATED = [(400.0, 80000.0), (700.0, 60000.0), (1500.0, 50000.0), (2400.0, 40000.0)]


def _line(energy, intensity, intensity_err=None):
    return SourceLine(energy, 0.01, intensity,
                      0.01 * intensity if intensity_err is None else intensity_err)


def _spectrum(peaks, seed):
    x = np.arange(CHANNELS, dtype=float)
    mean = np.full(CHANNELS, 40.0)
    for centre, area in ISOLATED + peaks:
        mean += (area / (SIGMA * math.sqrt(2.0 * math.pi))
                 * np.exp(-0.5 * ((x - centre) / SIGMA) ** 2))
    return np.random.default_rng(seed).poisson(mean).astype(float)


def _refit(counts, lines, excluded=()):
    """(refit, every line of the source), as the automatic run has them."""
    source = [_line(centre, area / 200.0) for centre, area in ISOLATED] + list(lines)
    return auto_calibrate.refit_source_lines(
        channel_indices(CHANNELS), counts, source, IDENTITY, excluded=excluded), source


def _points(refit, source):
    """What main_window hands the calibration plot after an automatic run:
    every committed peak, the refit's lines named, through the one rule."""
    peaks, energies = committed_peaks(
        refit.results,
        {id(r.peaks[0]): e for r, (_c, e) in zip(refit.results, refit.pairs)})
    return share_blended_areas(peaks, energies, source, IDENTITY)


def _point(points, energy):
    found = [p for p in points if abs(p[4] - energy) < 1e-9]
    assert len(found) == 1, f"{energy} keV exported {len(found)} times"
    return found[0]


# --- a listed line inside the window, with no found peak of its own ---------

BESIDE = 1000.0 + 1.1 * FWHM


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_line_inside_the_window_is_fitted_beside_it_not_counted_as_it(seed):
    counts = _spectrum([(1000.0, 200000.0), (BESIDE, 53000.0)], seed)
    refit, source = _refit(counts, [_line(1000.0, 1000.0), _line(BESIDE, 265.0)])

    # The scenario itself: the neighbour has no found peak, so it is
    # neither fitted on its own nor blended -- it is fitted beside.
    assert refit.alongside == 1 and refit.blended == 0
    result = [r for r, (_c, e) in zip(refit.results, refit.pairs) if e == 1000.0][0]
    assert len(result.peaks) == 2
    assert result.peaks[1].position == BESIDE      # pinned by the calibration
    assert not any(abs(e - BESIDE) < 1e-9 for _c, e in refit.pairs), "exported"

    points = _points(refit, source)
    assert _point(points, 1000.0)[2] == pytest.approx(200000.0, rel=0.015)
    # Fitted as its own peak, the neighbour is not ALSO counted as a share.
    assert blend_partners(*committed_peaks(
        refit.results, {id(r.peaks[0]): e for r, (_c, e) in zip(refit.results, refit.pairs)}),
        source, IDENTITY) == {}

    # CONTROL: the old number, with the neighbour's counts in it.
    old, old_source = _refit(counts, [_line(1000.0, 1000.0)])
    assert old.alongside == 0
    assert _point(_points(old, old_source), 1000.0)[2] > 1.10 * 200000.0


# --- lines close enough to sit inside the same peak --------------------------

UNDER = 2000.0 + 0.2 * FWHM


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_lines_sharing_one_peak_split_its_area_by_intensity(seed):
    counts = _spectrum([(2000.0, 100000.0), (UNDER, 30000.0)], seed)
    refit, source = _refit(counts, [_line(2000.0, 1000.0), _line(UNDER, 300.0, 30.0)])

    assert refit.blended == 1 and refit.alongside == 0
    result = [r for r, (_c, e) in zip(refit.results, refit.pairs) if e == 2000.0][0]
    whole = result.peaks[0]
    channel, _dc, area, area_err, _e = _point(_points(refit, source), 2000.0)
    share = 1000.0 / 1300.0
    assert area == pytest.approx(whole.area * share)
    assert area == pytest.approx(100000.0, rel=0.015)
    # N_line = N * I / (I + S): the partner's 30-count intensity error
    # joins the area's as 30 / 1300.
    own = efficiency_area_error(whole.area_err, result.reduced_chi2)
    assert area_err == pytest.approx(math.hypot(own * share, area * 30.0 / 1300.0))

    old, old_source = _refit(counts, [_line(2000.0, 1000.0)])
    assert old.blended == 0
    assert _point(_points(old, old_source), 2000.0)[2] > 1.25 * 100000.0


def test_an_unticked_partner_still_takes_its_share():
    """Unticking says the partner's own area is not to be exported. Its
    counts are still inside the peak, so the line keeps only its share."""
    counts = _spectrum([(2000.0, 100000.0), (UNDER, 30000.0)], 1)
    refit, source = _refit(counts, [_line(2000.0, 1000.0), _line(UNDER, 300.0, 30.0)],
                           excluded=(UNDER,))
    assert _point(_points(refit, source), 2000.0)[2] == pytest.approx(100000.0, rel=0.015)
    assert not any(abs(e - UNDER) < 1e-9 for _c, e in refit.pairs)


def test_the_committed_fit_keeps_its_whole_area_and_its_own_error():
    """Fit Results reports the peak; the efficiency point reports the
    line. Computing the point must not rewrite the fit."""
    counts = _spectrum([(2000.0, 100000.0), (UNDER, 30000.0)], 1)
    refit, source = _refit(counts, [_line(2000.0, 1000.0), _line(UNDER, 300.0, 30.0)])
    result = [r for r, (_c, e) in zip(refit.results, refit.pairs) if e == 2000.0][0]
    before = (result.peaks[0].area, result.peaks[0].area_err)
    _points(refit, source)
    assert (result.peaks[0].area, result.peaks[0].area_err) == before
    assert result.peaks[0].area == pytest.approx(130000.0, rel=0.015)


def test_the_summary_says_what_happened_to_the_neighbours():
    counts = _spectrum([(1000.0, 200000.0), (BESIDE, 53000.0),
                        (2000.0, 100000.0), (UNDER, 30000.0)], 1)
    refit, _source = _refit(counts, [_line(1000.0, 1000.0), _line(BESIDE, 265.0),
                                     _line(2000.0, 1000.0), _line(UNDER, 300.0, 30.0)])
    text = refit.summary("test.sou")
    assert "1 fitted beside a stronger neighbour" in text
    assert "1 blended into a stronger neighbour, the area shared by intensity" in text


# --- the rule itself, on hand-made peaks -------------------------------------
#
# (channel, channel_err, fwhm, area, area_err) per peak, energy == channel.

LINE = _line(500.0, 1000.0)
NEAR = _line(503.0, 250.0, 25.0)       # 0.6 of the 5-channel width away


def _shared(peaks, energies, lines=(LINE, NEAR), calibration=IDENTITY):
    return share_blended_areas(peaks, energies, list(lines), calibration)


def test_a_line_within_one_width_of_a_named_peak_takes_its_share():
    (_c, _dc, area, area_err, energy), = _shared(
        [(500.0, 0.1, 5.0, 1000.0, 30.0)], [500.0])
    share = 1000.0 / 1250.0
    assert (area, energy) == (pytest.approx(1000.0 * share), 500.0)
    assert area_err == pytest.approx(math.hypot(30.0 * share, area * 25.0 / 1250.0))


def test_a_line_beyond_one_width_is_not_inside_the_peak():
    far = _line(500.0 + 1.01 * SHARED_PEAK_FWHM * 5.0, 250.0)
    (_c, _dc, area, _err, _e), = _shared([(500.0, 0.1, 5.0, 1000.0, 30.0)], [500.0],
                                         lines=(LINE, far))
    assert area == 1000.0


def test_a_line_with_a_peak_of_its_own_is_not_shared():
    """The nearest peak owns a line's counts -- here an unnamed peak fitted
    right where it is, as the refit's pinned second peak would be."""
    points = _shared([(500.0, 0.1, 5.0, 1000.0, 30.0), (503.0, 0.0, 5.0, 250.0, 16.0)],
                     [500.0, None])
    assert [p[2] for p in points] == [1000.0]


def test_a_line_named_on_another_peak_is_not_shared():
    points = _shared([(500.0, 0.1, 5.0, 1000.0, 30.0), (540.0, 0.1, 5.0, 250.0, 16.0)],
                     [500.0, 503.0])
    assert [p[2] for p in points] == [1000.0, 250.0]


@pytest.mark.parametrize("peaks, lines, calibration", [
    ([(500.0, 0.1, 5.0, 1000.0, 30.0)], (LINE, NEAR), None),     # no calibration
    ([(500.0, 0.1, 5.0, 1000.0, 30.0)], (), IDENTITY),           # no source
    ([(500.0, 0.1, None, 1000.0, 30.0)], (LINE, NEAR), IDENTITY),  # no fitted width
])
def test_without_what_it_needs_the_rule_passes_the_area_through(peaks, lines, calibration):
    (_c, _dc, area, area_err, _e), = _shared(peaks, [500.0], lines, calibration)
    assert (area, area_err) == (1000.0, 30.0)


def test_unnamed_peaks_are_not_exported_and_order_is_kept():
    peaks = [(300.0, 0.1, 5.0, 10.0, 1.0), (500.0, 0.1, 5.0, 1000.0, 30.0),
             (700.0, 0.1, 5.0, 20.0, 2.0)]
    points = _shared(peaks, [300.0, None, 700.0], lines=())
    assert [p[4] for p in points] == [300.0, 700.0]


# --- the error each point carries ---------------------------------------------


def _fits(reduced_chi2):
    peak = SimpleNamespace(position=500.0, position_err=None, fwhm=5.0,
                           area=1000.0, area_err=10.0)
    return [SimpleNamespace(peaks=[peak], reduced_chi2=reduced_chi2)], peak


def test_a_poor_fit_widens_the_points_error_by_root_chi2():
    fits, peak = _fits(9.0)
    peaks, energies = committed_peaks(fits, {id(peak): 661.657})
    assert peaks == [(500.0, 0.0, 5.0, 1000.0, pytest.approx(30.0))]
    assert energies == [661.657]


@pytest.mark.parametrize("chi2", [0.3, 1.0, None])
def test_a_good_fit_leaves_the_error_alone(chi2):
    fits, peak = _fits(chi2)
    peaks, _energies = committed_peaks(fits, {id(peak): 661.657})
    assert peaks[0][4] == pytest.approx(10.0)


def test_an_integration_result_contributes_no_peak():
    fits, peak = _fits(1.0)
    peaks, energies = committed_peaks([SimpleNamespace(net_area=5.0)] + fits, {})
    assert len(peaks) == 1 and energies == [None]


def test_the_point_uses_the_same_rule_as_the_export():
    """One rule, not two: committed_peaks and the hand path's choices both
    go through caleneff_export.efficiency_area_error."""
    for chi2 in (0.5, 2.0, 37.0):
        fits, _peak = _fits(chi2)
        assert committed_peaks(fits, {})[0][0][4] == pytest.approx(
            efficiency_area_error(10.0, chi2))
