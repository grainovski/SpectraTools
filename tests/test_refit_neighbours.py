"""A line's efficiency point counts that line, not its neighbours.

The refit for CalEnEff (auto_calibrate.refit_source_lines) is where the
efficiency points of an automatic calibration come from. Two kinds of
neighbour used to hand their counts to the line being fitted there:

- a weaker listed line inside the line's fit window with no found peak
  of its own -- now fitted beside it as a second peak, pinned where the
  calibration puts it, and not exported;
- a weaker listed line close enough to land on the SAME found peak --
  now given its share of that peak's area, in proportion to intensity.

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
from caleneff_export import efficiency_area_error
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
    isolated = [_line(centre, area / 200.0) for centre, area in ISOLATED]
    return auto_calibrate.refit_source_lines(
        channel_indices(CHANNELS), counts, isolated + lines, IDENTITY, excluded=excluded)


def _point(refit, energy):
    found = [p for p in refit.efficiency_points() if abs(p[4] - energy) < 1e-9]
    assert len(found) == 1, f"{energy} keV exported {len(found)} times"
    return found[0]


# --- a listed line inside the window, with no found peak of its own ---------

BESIDE = 1000.0 + 1.1 * FWHM


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_line_inside_the_window_is_fitted_beside_it_not_counted_as_it(seed):
    counts = _spectrum([(1000.0, 200000.0), (BESIDE, 53000.0)], seed)
    refit = _refit(counts, [_line(1000.0, 1000.0), _line(BESIDE, 265.0)])

    # The scenario itself: the neighbour has no found peak, so it is
    # neither fitted on its own nor blended -- it is fitted beside.
    assert refit.alongside == 1 and refit.blended == 0
    result = [r for r, (_c, e) in zip(refit.results, refit.pairs) if e == 1000.0][0]
    assert len(result.peaks) == 2
    assert result.peaks[1].position == BESIDE      # pinned by the calibration
    assert not any(abs(e - BESIDE) < 1e-9 for _c, e in refit.pairs), "exported"

    assert _point(refit, 1000.0)[2] == pytest.approx(200000.0, rel=0.015)

    # CONTROL: the old number, with the neighbour's counts in it.
    old = _refit(counts, [_line(1000.0, 1000.0)])
    assert old.alongside == 0
    assert _point(old, 1000.0)[2] > 1.10 * 200000.0


# --- lines close enough to land on the same found peak -----------------------

UNDER = 2000.0 + 0.2 * FWHM


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_lines_sharing_one_peak_split_its_area_by_intensity(seed):
    counts = _spectrum([(2000.0, 100000.0), (UNDER, 30000.0)], seed)
    refit = _refit(counts, [_line(2000.0, 1000.0), _line(UNDER, 300.0, 30.0)])

    assert refit.blended == 1 and refit.alongside == 0
    share, extra = [s for s, (_c, e) in zip(refit.shares, refit.pairs) if e == 2000.0][0]
    assert share == pytest.approx(1000.0 / 1300.0)
    # N_line = N * I / (I + S): the partner's 30-count intensity error
    # joins the area's as 30 / 1300.
    assert extra == pytest.approx(30.0 / 1300.0)
    assert _point(refit, 2000.0)[2] == pytest.approx(100000.0, rel=0.015)

    old = _refit(counts, [_line(2000.0, 1000.0)])
    assert old.blended == 0
    assert _point(old, 2000.0)[2] > 1.25 * 100000.0


def test_an_unticked_partner_still_takes_its_share():
    """Unticking says the partner's own area is not to be exported. Its
    counts are still inside the peak, so the line keeps only its share."""
    counts = _spectrum([(2000.0, 100000.0), (UNDER, 30000.0)], 1)
    refit = _refit(counts, [_line(2000.0, 1000.0), _line(UNDER, 300.0, 30.0)],
                   excluded=(UNDER,))
    assert _point(refit, 2000.0)[2] == pytest.approx(100000.0, rel=0.015)
    assert not any(abs(e - UNDER) < 1e-9 for _c, e in refit.pairs)


def test_the_committed_fit_keeps_its_whole_area_and_its_own_error():
    """Fit Results reports the peak; the efficiency point reports the
    line. Computing the point must not rewrite the fit."""
    counts = _spectrum([(2000.0, 100000.0), (UNDER, 30000.0)], 1)
    refit = _refit(counts, [_line(2000.0, 1000.0), _line(UNDER, 300.0, 30.0)])
    result = [r for r, (_c, e) in zip(refit.results, refit.pairs) if e == 2000.0][0]
    before = (result.peaks[0].area, result.peaks[0].area_err)
    point = _point(refit, 2000.0)
    assert (result.peaks[0].area, result.peaks[0].area_err) == before
    assert result.peaks[0].area == pytest.approx(130000.0, rel=0.015)
    assert point[2] == pytest.approx(before[0] * 1000.0 / 1300.0)


def test_the_summary_says_what_happened_to_the_neighbours():
    counts = _spectrum([(1000.0, 200000.0), (BESIDE, 53000.0),
                        (2000.0, 100000.0), (UNDER, 30000.0)], 1)
    refit = _refit(counts, [_line(1000.0, 1000.0), _line(BESIDE, 265.0),
                            _line(2000.0, 1000.0), _line(UNDER, 300.0, 30.0)])
    text = refit.summary("test.sou")
    assert "1 fitted beside a stronger neighbour" in text
    assert "1 blended into a stronger neighbour, the area shared by intensity" in text


# --- the point's uncertainty -------------------------------------------------


def _outcome(reduced_chi2, share=(1.0, 0.0)):
    peak = SimpleNamespace(position=500.0, position_err=0.02, area=1000.0, area_err=10.0)
    outcome = auto_calibrate.RefitOutcome()
    outcome.results.append(SimpleNamespace(peaks=[peak], reduced_chi2=reduced_chi2))
    outcome.pairs.append((500.0, 661.657))
    outcome.shares.append(share)
    return outcome


def test_a_poor_fit_widens_the_points_error_by_root_chi2():
    assert _outcome(9.0).efficiency_points() == [
        (500.0, 0.02, 1000.0, pytest.approx(30.0), 661.657)]


def test_a_good_fit_leaves_the_error_alone():
    for chi2 in (0.3, 1.0, None):
        assert _outcome(chi2).efficiency_points()[0][3] == pytest.approx(10.0)


def test_the_widening_and_the_split_combine_in_quadrature():
    (_c, _dc, area, area_err, _e), = _outcome(9.0, (0.5, 0.02)).efficiency_points()
    assert area == pytest.approx(500.0)
    assert area_err == pytest.approx(500.0 * math.hypot(0.03, 0.02))


def test_the_point_uses_the_same_rule_as_the_export():
    """One rule, not two: the refit's points and the hand path's choices
    both go through caleneff_export.efficiency_area_error."""
    for chi2 in (0.5, 2.0, 37.0):
        expected = efficiency_area_error(10.0, chi2)
        assert _outcome(chi2).efficiency_points()[0][3] == pytest.approx(expected)
