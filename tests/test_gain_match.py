"""Gain matching CH1 onto CH0 (gain_match), against answers known in advance.

Synthetic spectra carry a known quadratic gain, so the recovered
coefficients can be checked against the truth within their own errors.
The pairing tests check the failure the matcher exists to prevent: a
pairing that is only one of several possibilities must be REPORTED, never
picked, since a gain fitted to a wrong pairing looks perfectly good. The
dither test has a must-fail control -- the same map without the dither
must show the picket fence -- so a broken chi-square cannot pass as a
flat spectrum. The real Na-22 run checks that the route the sorter will
use, apply() on events, puts CH1's lines on CH0's.
"""

import os

import numpy as np
import pytest
from scipy.special import ndtr

import gain_match as G
from sou_io import load_sou
from synthetic_calibration import source_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA22_RUN = os.path.join(REPO, "Na22_261126_ANI_STEF_180")
NA22_FILTERED = os.path.join(NA22_RUN, "FILTERED")
NA22_STAMP = "Na22_261126_ANI_STEF_20261126_115129"


# ----------------------------------------------------------- synthetic


def _spectrum(centroids, fwhms, areas, rng, nchan=G.DEFAULT_NCHAN,
              bg_amp=2000.0, bg_scale=1500.0, bg_flat=20.0):
    """Poisson counts of Gaussian peaks on a falling exponential plus a
    flat floor. Centroids are in centre coordinates (channel i covers
    [i - 1/2, i + 1/2)), the convention every fit here reports in."""
    edges = np.arange(nchan + 1, dtype=float) - 0.5
    mean = bg_amp * np.exp(-np.arange(nchan) / bg_scale) + bg_flat
    for c, f, a in zip(centroids, fwhms, areas):
        sigma = f / 2.3548200450309493
        mean = mean + a * np.diff(ndtr((edges - c) / sigma))
    return rng.poisson(mean).astype(float)


def _inverse(coefficients, targets):
    """ch1 with P(ch1) = target, by Newton from the linear guess."""
    a = np.asarray(coefficients, dtype=float)
    t = np.asarray(targets, dtype=float)
    x = (t - a[0]) / a[1]
    for _ in range(50):
        p = np.polynomial.polynomial.polyval(x, a)
        dp = np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(a))
        x = x - (p - t) / dp
    return x


def _pair_of_spectra(truth, c0, fwhm0, areas, seed, ratio=None, **bg):
    """(CH1 spectrum, CH0 spectrum, true CH1 centroids): CH0 as given,
    CH1 the same lines seen through the inverse of `truth`, with widths
    shrunk or stretched by the local slope. `ratio` optionally scales
    CH1's areas (a smooth relative efficiency)."""
    rng = np.random.default_rng(seed)
    c0 = np.asarray(c0, dtype=float)
    c1 = _inverse(truth, c0)
    slope = np.polynomial.polynomial.polyval(c1, np.polynomial.polynomial.polyder(truth))
    a1 = np.asarray(areas, dtype=float) * (1.0 if ratio is None else ratio(c0))
    s0 = _spectrum(c0, fwhm0, areas, rng, **bg)
    s1 = _spectrum(c1, np.asarray(fwhm0) / slope, a1, rng, **bg)
    return s1, s0, c1


def _assert_recovers(gm, truth, sigmas=5.0):
    scale = np.sqrt(max(1.0, gm.reduced_chi2 or 1.0))
    for k, (got, want, err) in enumerate(zip(gm.coefficients, truth, gm.errors)):
        assert abs(got - want) <= sigmas * err * scale, (
            f"a{k} = {got:.6g} +- {err:.2g} (x{scale:.2f}), truth {want:.6g}")


SCINT_TRUTH = [-20.0, 0.95, 3.0e-6]
SCINT_C0 = [700.0, 1600.0, 2700.0, 4300.0, 6400.0, 9100.0, 12600.0]
SCINT_AREAS = [40000.0, 30000.0, 60000.0, 25000.0, 30000.0, 20000.0, 15000.0]


@pytest.fixture(scope="module")
def scintillator():
    """~50 channels FWHM at 2700 channels, growing as sqrt(channel): at
    the top the peaks are ~108 channels wide, beyond what peak_search
    reports at full resolution, so this also exercises compression."""
    fwhm0 = 50.0 * np.sqrt(np.asarray(SCINT_C0) / 2700.0)
    s1, s0, c1 = _pair_of_spectra(SCINT_TRUTH, SCINT_C0, fwhm0, SCINT_AREAS, seed=11,
                                  ratio=lambda c: 0.9 + 0.1 * c / 16384.0)
    return s1, s0, c1, fwhm0


def test_scintillator_quadratic_gain_is_recovered(scintillator):
    s1, s0, c1, fwhm0 = scintillator
    p1, p0 = G.find_peaks(s1), G.find_peaks(s0)
    paired = G.pair_peaks(p1, p0)
    assert paired.ok, paired.reason
    assert paired.method == "match"
    assert len(paired.pairs) == len(SCINT_C0)
    assert all(p.included for p in paired.pairs)
    # every pair is a true pair, not merely a self-consistent one
    for pair in paired.pairs:
        k = int(np.argmin(np.abs(np.asarray(SCINT_C0) - pair.ch0)))
        assert abs(pair.ch1 - c1[k]) < 0.1 * fwhm0[k]
    gm = G.fit_gain(paired)
    assert gm.order == 2
    _assert_recovers(gm, SCINT_TRUTH)
    # residuals a small fraction of the resolution
    assert max(abs(r) for r in gm.residuals) < 0.05 * min(fwhm0)


def test_compression_is_chosen_for_wide_peaks(scintillator):
    s1, s0, _c1, _f = scintillator
    peaks = G.find_peaks(s0)
    assert {p.compression for p in peaks} == {2}
    # at full resolution the widest lines are lost; that is why
    full = G.find_peaks(s0, compression=1)
    assert len(full) < len(peaks)


#: A small bow, as two HPGe channels of one digitizer have (0.6 channels
#: off the chord at mid-range). Pairing runs on a straight line with a
#: tolerance of half a FWHM -- 1.5 channels here -- so a bow like the
#: scintillator's (476 channels) would leave these narrow peaks unpaired;
#: see test_strong_bow_with_narrow_peaks_is_refused_not_guessed.
HPGE_TRUTH = [12.0, 1.37, -2.0e-8]
HPGE_C0 = [900.0, 2100.0, 3500.0, 5200.0, 7000.0, 9400.0, 12100.0, 15200.0]
HPGE_AREAS = [8000.0, 20000.0, 6000.0, 12000.0, 9000.0, 15000.0, 5000.0, 7000.0]


def test_hpge_like_narrow_peaks_are_recovered():
    fwhm0 = 3.0 + 0.0002 * np.asarray(HPGE_C0)
    s1, s0, c1 = _pair_of_spectra(HPGE_TRUTH, HPGE_C0, fwhm0, HPGE_AREAS, seed=23,
                                  bg_amp=500.0, bg_flat=5.0)
    p1, p0 = G.find_peaks(s1), G.find_peaks(s0)
    assert {p.compression for p in p1} == {1}
    paired = G.pair_peaks(p1, p0)
    assert paired.ok, paired.reason
    assert len(paired.pairs) == len(HPGE_C0)
    gm = G.fit_gain(paired)
    assert gm.order == 2
    _assert_recovers(gm, HPGE_TRUTH)
    assert np.allclose(gm.map(c1), HPGE_C0, atol=0.1)


def test_strong_bow_with_narrow_peaks_is_refused_not_guessed():
    """The limitation, pinned: narrow peaks under a strongly curved
    relative gain cannot be paired by a straight line, and the result is
    a refusal, never a partial guess."""
    fwhm0 = 3.0 + 0.0002 * np.asarray(HPGE_C0)
    s1, s0, _c1 = _pair_of_spectra([12.0, 1.37, -2.0e-6], HPGE_C0, fwhm0, HPGE_AREAS,
                                   seed=23, bg_amp=500.0, bg_flat=5.0)
    paired = G.pair_peaks(G.find_peaks(s1), G.find_peaks(s0))
    assert not paired.ok
    assert paired.reason


def test_gain_match_convenience_runs_the_whole_step(scintillator):
    s1, s0, _c1, _f = scintillator
    gm = G.gain_match(s1, s0, source="synthetic")
    _assert_recovers(gm, SCINT_TRUTH)
    assert gm.source == "synthetic"


def test_source_lines_pair_by_energy():
    """Both channels identified against Eu-152 separately; pairs carry
    the energy, and the gain comes out the same."""
    lines = load_sou(source_file("eu152.sou"))
    strong = sorted((l for l in lines if l.intensity >= 300), key=lambda l: -l.intensity)
    kept = []
    for line in strong:  # one line per resolved peak
        if all(abs(line.energy - k.energy) > 8.0 for k in kept):
            kept.append(line)
    energies = np.array([l.energy for l in kept])
    c0 = 15.0 + energies / 0.1          # 0.1 keV per CH0 channel
    keep = c0 < 16000
    c0, energies = c0[keep], energies[keep]
    areas = np.array([l.intensity for l in kept])[keep] * 5.0
    fwhm0 = 10.0 + 0.004 * c0
    truth = [-8.0, 1.05, -3.0e-7]
    s1, s0, c1 = _pair_of_spectra(truth, c0, fwhm0, areas, seed=5, bg_amp=800.0)
    paired = G.pair_peaks(G.find_peaks(s1), G.find_peaks(s0), source_lines=lines)
    assert paired.ok, paired.reason
    assert paired.method == "source"
    assert len(paired.pairs) >= 6
    for pair in paired.pairs:
        k = int(np.argmin(np.abs(energies - pair.energy)))
        assert abs(pair.energy - energies[k]) < 0.01
        assert abs(pair.ch0 - c0[k]) < max(0.5, 5 * pair.ch0_err)
        assert abs(pair.ch1 - c1[k]) < max(0.5, 5 * pair.ch1_err)
    gm = G.fit_gain(paired)
    _assert_recovers(gm, truth)


# ----------------------------------------------- pairing: report, don't guess


def _peak(c, area=10000.0, fwhm=8.0, err=0.05):
    return G.GainPeak(centroid=c, centroid_err=err, fwhm=fwhm, area=area, area_err=100.0)


def test_two_against_three_peaks_is_ambiguous_and_refused():
    """CH1 shows 3000 and 3400; CH0 shows 3000, 3400 and 3800, all equally
    strong. Gain 1 at offset 0 and gain 1 at offset 400 both explain both
    CH1 peaks: the gains agree, the pairings do not. Refused."""
    ch1 = [_peak(3000.0), _peak(3400.0)]
    ch0 = [_peak(3000.0), _peak(3400.0), _peak(3800.0)]
    result = G.pair_peaks(ch1, ch0)
    assert not result.ok
    assert "different pairings" in result.reason
    with pytest.raises(G.GainMatchError):
        G.fit_gain(result)


def test_two_against_two_peaks_is_unique_but_unconfirmed():
    """With two peaks each and a positive gain there is exactly one
    pairing. It is accepted -- it is the only one -- and says that
    nothing confirms it; the fit is then order 1."""
    ch1 = [_peak(2000.0, area=30000.0), _peak(5000.0)]
    ch0 = [_peak(2050.0, area=31000.0), _peak(5120.0)]
    result = G.pair_peaks(ch1, ch0)
    assert result.ok, result.reason
    assert result.method == "few"
    assert result.confirmations == 0
    assert "nothing confirms" in result.note
    gm = G.fit_gain(result)
    assert gm.order == 1
    assert gm.map([2000.0, 5000.0]) == pytest.approx([2050.0, 5120.0], abs=1e-6)
    assert gm.reduced_chi2 is None  # no degrees of freedom


def test_fewer_than_two_pairs_is_refused():
    result = G.pair_peaks([_peak(2000.0)], [_peak(2000.0), _peak(4000.0)])
    assert not result.ok
    assert "at least two" in result.reason
    with pytest.raises(G.GainMatchError, match="at least two"):
        G.fit_gain([G.GainPair(100.0, 0.1, 100.0, 0.1)])


def test_order_two_with_three_pairs_is_refused():
    pairs = [G.GainPair(c, 0.1, 1.1 * c, 0.1) for c in (1000.0, 3000.0, 6000.0)]
    with pytest.raises(G.GainMatchError, match="Order 2 needs at least 4"):
        G.fit_gain(pairs, order=2)
    assert G.fit_gain(pairs).order == 1


def test_excluded_pairs_stay_out_of_the_fit_but_get_a_residual():
    pairs = [G.GainPair(c, 0.1, 2.0 * c + 5.0, 0.1) for c in (1000.0, 3000.0, 6000.0)]
    pairs.append(G.GainPair(8000.0, 0.1, 9000.0, 0.1, included=False))
    gm = G.fit_gain(pairs)
    assert gm.coefficients == pytest.approx([5.0, 2.0], abs=1e-6)
    assert gm.residuals[3] == pytest.approx(9000.0 - 16005.0, abs=1e-3)


def test_a_map_turning_over_in_range_is_refused():
    # a quadratic through these points peaks inside 0..16384
    pairs = [G.GainPair(c, 0.1, c - 1.0e-4 * c * c, 0.1)
             for c in (1000.0, 2000.0, 3000.0, 4000.0)]
    with pytest.raises(G.GainMatchError, match="not increasing"):
        G.fit_gain(pairs)


# ---------------------------------------------------------------- dither


def _flatness_chi2(channels, lo, hi):
    counts = np.bincount(channels, minlength=hi + 1)[lo:hi]
    mean = counts.mean()
    return float(np.sum((counts - mean) ** 2 / mean)), counts.size - 1


def test_dither_leaves_a_flat_spectrum_flat_and_no_dither_does_not():
    gm = G.GainMatch([7.3, 1.37])
    rng = np.random.default_rng(3)
    energies = rng.integers(500, 10500, size=3_000_000)
    lo, hi = int(gm.map(600)), int(gm.map(10400))

    channels, inside = gm.apply(energies)
    assert inside.all()
    chi2, ndf = _flatness_chi2(channels[inside], lo, hi)
    tolerance = 5.0 * np.sqrt(2.0 / ndf)
    assert abs(chi2 / ndf - 1.0) < tolerance, chi2 / ndf

    # must-fail control: the same map, every event at its channel centre
    plain = np.floor(gm.map(energies.astype(float)) + 0.5).astype(np.int64)
    chi2_plain, ndf_plain = _flatness_chi2(plain, lo, hi)
    assert chi2_plain / ndf_plain > 20.0, chi2_plain / ndf_plain


def test_dither_is_unbiased_in_centre_coordinates():
    """Averaged over channels, an event lands at P(E) itself. The plan's
    floor(P(E + u)) applied to a centre-fitted P would land at
    P(E) + (P' - 1)/2, 0.185 channels high for this gain."""
    gm = G.GainMatch([7.3, 1.37])
    energies = np.random.default_rng(4).integers(500, 10500, size=1_000_000)
    channels, _inside = gm.apply(energies)
    bias = float(np.mean(channels - gm.map(energies)))
    assert abs(bias) < 0.01, bias


def test_identity_map_returns_every_channel_unchanged():
    gm = G.GainMatch([0.0, 1.0])
    energies = np.arange(16384)
    channels, inside = gm.apply(energies)
    assert np.array_equal(channels, energies)
    assert inside.all()


def test_dither_is_reproducible_and_counts_the_edge():
    gm = G.GainMatch([-100.0, 1.2])
    energies = np.arange(16384)
    a, inside_a = gm.apply(energies, seed=99)
    b, inside_b = gm.apply(energies, seed=99)
    c, _inside_c = gm.apply(energies, seed=100)
    assert np.array_equal(a, b) and np.array_equal(inside_a, inside_b)
    assert not np.array_equal(a, c)
    assert not inside_a[:50].any()             # mapped below channel 0
    assert not inside_a[-2000:].any()          # mapped above 16383
    assert np.all((a[inside_a] >= 0) & (a[inside_a] < 16384))


# -------------------------------------------------------------- sidecar


def test_save_load_round_trip(tmp_path):
    pairs = [G.GainPair(c, 0.1, 0.95 * c + 3e-6 * c * c - 20.0, 0.12,
                        energy=None if c < 2000 else c / 4.0,
                        included=c != 6000.0, fwhm1=30.0, area1=1e4)
             for c in (800.0, 2500.0, 4100.0, 6000.0, 9000.0)]
    gm = G.fit_gain(pairs, source="Eu-152 calibration run")
    path = tmp_path / "gain.json"
    gm.save(path)
    back = G.GainMatch.load(path)
    assert back.coefficients == gm.coefficients
    assert np.array_equal(back.covariance, gm.covariance)
    assert back.order == gm.order and back.source == gm.source
    assert back.date == gm.date and back.nchan == gm.nchan
    assert back.rms == gm.rms and back.reduced_chi2 == gm.reduced_chi2
    assert [p.to_dict() for p in back.pairs] == [p.to_dict() for p in gm.pairs]
    assert np.isnan(back.pairs[0].fwhm0)   # unknown stays unknown, not 0
    e = np.arange(0, 16384, 7)
    assert np.array_equal(back.apply(e)[0], gm.apply(e)[0])
    raw = path.read_bytes()
    assert b"NaN" not in raw   # strict JSON


def test_load_refuses_a_file_that_is_not_a_gain_match(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"format": "sparse", "coefficients": [0, 1]}', encoding="utf-8")
    with pytest.raises(G.GainMatchError, match="not a SpectraTools gain-match"):
        G.GainMatch.load(path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(G.GainMatchError, match="Cannot read"):
        G.GainMatch.load(path)


# ------------------------------------------------------------ real Na-22


@pytest.fixture(scope="module")
def na22_singles():
    if not os.path.isdir(NA22_RUN):
        pytest.skip("untracked run Na22_261126_ANI_STEF_180/ is absent")
    spectra = {}
    for ch in ("CH0", "CH1"):
        path = os.path.join(NA22_FILTERED,
                            f"{ch}@DT5780SCP_58363_EspectrumF_{NA22_STAMP}.txt")
        spectra[ch] = np.loadtxt(path)
    assert spectra["CH0"].size == 16383
    return spectra


def _line_near(peaks, channel):
    return min(peaks, key=lambda p: abs(p.centroid - channel))


def _check_lines_land_on_ch0(spectra, gm):
    """Map CH1's EVENTS with apply() -- the sorter's route -- re-fit the
    result, and compare 511 and 1275 keV with CH0's own fits."""
    ch1 = spectra["CH1"].astype(np.int64)
    events = np.repeat(np.arange(ch1.size), ch1)
    channels, inside = gm.apply(events)
    mapped = np.bincount(channels[inside], minlength=G.DEFAULT_NCHAN).astype(float)
    p0 = G.find_peaks(spectra["CH0"])
    pm = G.find_peaks(mapped)
    offsets = {}
    for guess in (2173.0, 5391.0):  # 511 and 1274.5 keV in CH0, measured
        ref, got = _line_near(p0, guess), _line_near(pm, guess)
        assert abs(ref.centroid - guess) < 50
        offsets[guess] = (got.centroid - ref.centroid, ref.fwhm)
        assert abs(got.centroid - ref.centroid) < ref.fwhm, offsets
    return offsets


def test_na22_no_source(na22_singles):
    p1, p0 = G.find_peaks(na22_singles["CH1"]), G.find_peaks(na22_singles["CH0"])
    paired = G.pair_peaks(p1, p0)
    assert paired.ok, paired.reason
    gm = G.fit_gain(paired)
    offsets = _check_lines_land_on_ch0(na22_singles, gm)
    print(f"\nNa-22, no source: {gm}, errors {gm.errors}, "
          f"pairs {[(round(p.ch1, 2), round(p.ch0, 2)) for p in paired.pairs]}, "
          f"line offsets after mapping {offsets}")


def test_na22_with_a_two_line_source(na22_singles, tmp_path):
    """No Na-22 .sou ships in sources/, so one is written here (ENSDF:
    1274.537 keV, 99.94%; the 511 keV annihilation line at 180%).
    auto_calibrate.match refuses two lines outright (MIN_INLIERS = 4);
    the two-pair rule accepts them because the two strongest peaks in
    each channel are the only credible anchors."""
    path = tmp_path / "na22.sou"
    path.write_text("511.0 0.0 18000.0 100.0\n1274.537 0.007 9994.0 13.0\n",
                    encoding="utf-8")
    lines = load_sou(str(path))
    p1, p0 = G.find_peaks(na22_singles["CH1"]), G.find_peaks(na22_singles["CH0"])
    paired = G.pair_peaks(p1, p0, source_lines=lines)
    assert paired.ok, paired.reason
    assert [p.energy for p in paired.pairs] == [511.0, 1274.537]
    assert paired.confirmations == 0
    gm = G.fit_gain(paired)
    assert gm.order == 1
    offsets = _check_lines_land_on_ch0(na22_singles, gm)
    print(f"\nNa-22, two-line source: {gm}, line offsets {offsets}")
