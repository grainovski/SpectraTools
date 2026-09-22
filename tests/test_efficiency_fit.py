"""Our efficiency fit against CalEnEff's, on CalEnEff's own data.

This is an EXTERNAL oracle, which is rare and is the reason to trust the
rest of the feature. The expected numbers below were produced by running
the reference engine (ra226_gui.CalibrationEngine) on these same three
files; they are not our own output fed back in.

What is compared, and why in this order:

  curve values  The thing that actually matters. Fit parameters can trade
                against each other, so two genuinely equivalent fits can
                carry different parameters; the curve they describe is what
                the spectrum correction divides by.
  chi-squared   The quantity being minimised, robust to that degeneracy.
  equivalence   Whether the reference's parameters and ours describe the
                same fit, measured as the chi-squared one set scores on the
                other's data. Replaces a direct parameter comparison, which
                asked a question this data cannot answer -- see below.

WHERE WE DELIBERATELY NO LONGER MATCH THE REFERENCE

KRF still has to reproduce CalEnEff outright, and does. Radware is held to
"never worse", because efficiency.RW_SCALE_TARGETS searches curves the
reference cannot reach.

The reason is that the Radware family is NOT closed under a constant factor
on eps, so the fitted curve depended on the arbitrary normalisation of the
intensity column -- a constant our own .sou files do not agree on (most peak
at 10000, co56.sou at 100000, na24.sou at 1000). Measured before the fix, a
100x change in it moved the reported curve by up to 12% at the bottom of the
range while leaving KRF identical to 2e-8 on these three files. The reference
has the same property and does nothing about it, so removing it necessarily
means parting company with the reference on any file whose scale differs from
the one it happened to be handed.

KRF turned out to have the same disease by a different route -- its solution
is scale-invariant but its SEARCH was not, and a sweep found data where the
same points scored 21.707 at one intensity scale and 136.412 at another. It
is fitted at a canonical scale too now, but that conversion is EXACT, so it
still reproduces the reference here. See
test_the_krf_fit_does_not_depend_on_the_intensity_scale.

On 226Ra and demo2 the ladder lands on the reference's own answer and the
full strict comparison still applies. On demo1 it finds a genuinely better
fit -- chi-squared 1.6373 against the reference's 2.9282 -- which is listed
in RW_BEATS_REFERENCE and pinned as a floor, so a ladder that stopped
working would fail rather than quietly revert.

WHY PARAMETERS ARE NOT COMPARED DIRECTLY

This check used to assert the parameters themselves matched to rel=1e-4,
and demo2's Radware a1/a2 failed it by 61% and 15% while the curve and
the chi-squared agreed to 1e-7. The parameters were not wrong; the
assertion was.

On demo2 the low-energy branch f1 = a1 + a2*x never activates: f1 > f2
at all 19 points, and with the blend exponent g = 15 it reaches the curve
only where the branches come closest, at the single lowest line. Two
parameters, one constraint. Measured: corr(a1, a2) = +1.000000000, the
Hessian's softest eigenvalue -1.5e-11 against a largest of 2.2e+05, and
the flat direction (+0.1934, -0.9811, 0, 0, 0) -- exactly the line
a1 + a2*ln(E_min/100) = const, on which the two solutions agree to
5.7e-6. An optimizer stops wherever rounding leaves it along a direction
that flat, so rel=1e-4 was recording one machine's floating-point noise.

WHAT THIS LEAVES UNTESTED, DELIBERATELY

The two solutions are NOT identical everywhere. Between the two lowest
demo2 lines the curves differ by up to 1.29e-2 (worst at 124.19 keV,
between lines at 121.78 and 244.69 keV) -- around one error bar of the
data there, and 4.2% of the measured range exceeds 1e-3. That is a real
ambiguity in the calibration, not a defect in either implementation: the
data pin f1 at the lowest line but not its slope, so the efficiency just
above that line is genuinely under-determined. Do not "fix" this by
tightening a dense-grid tolerance -- any threshold that passes today
encodes where this machine's optimizer happened to stop.
"""

import os

import numpy as np
import pytest

from efficiency import fit_efficiency

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")

#: Produced by the reference implementation, 2026-09-20.
GOLDEN = {
    "226Ra_En_Area.txt": {
        "n": 23,
        "krf": [6.274371440450e-01, 7.844365398647e+05,
                -7.258867102388e-04, -1.467068952279e+02],
        "krf_chi2": 117.5085777268, "krf_ndf": 19,
        "rw": [-3.765575465541e+01, 7.438801254775e+01, 6.386965185696e+00,
               -7.193596739758e-01, -2.424310667484e-02],
        "rw_chi2": 97.8603554131, "rw_ndf": 18,
    },
    "demo1.txt": {
        "n": 9,
        "krf": [1.138577308963e+03, 2.962038674033e+07,
                -5.801324494336e-03, -1.015250508519e+02],
        "krf_chi2": 3.9694346123, "krf_ndf": 5,
        "rw": [12.192811084709, -0.257477640308, 8.924249211442,
               -2.565507279293, -0.599464105736],
        "rw_chi2": 2.9281520555, "rw_ndf": 4,
    },
    "demo2.txt": {
        "n": 19,
        "krf": [8.896741862449e+00, 7.089236616431e+06,
                -1.081881901237e-03, -1.160635814659e+02],
        "krf_chi2": 47.6128442300, "krf_ndf": 15,
        "rw": [3.012598062741e+00, 3.897772490500e+01, 8.473633125424e+00,
               -8.688228294051e-01, -3.357195834004e-02],
        "rw_chi2": 42.1858733891, "rw_ndf": 14,
    },
}


def _load(name):
    """The reference's 7-column format: ch dch N dN E I dI."""
    data = np.loadtxt(os.path.join(FIXTURES, name), ndmin=2)
    return data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]


#: How far apart two parameter sets may score on the same data and still
#: count as the same fit. The natural unit is the one-sigma contour, where
#: chi-squared rises by 1.0, so this is 10,000x tighter than "within the
#: fit's own uncertainty". Measured across all three files and both models
#: on 2026-09-21: worst 1.118e-07 (demo2 Radware, the case that used to
#: fail), everything else below 2e-11. That leaves ~900x headroom, while a
#: genuinely different fit misses by of order 1 or more.
_EQUIVALENT_CHI2 = 1e-4


#: Fixtures where our Radware fit is expected to BEAT the reference's, and
#: by how much at least. See the module docstring: the scale ladder searches
#: a family the reference does not, so on demo1 it reaches a fit the
#: reference never sees. Pinned as a floor rather than a value so the ladder
#: staying switched on is what the test checks -- losing it drops demo1 back
#: to the reference's 2.9282 and fails here.
RW_BEATS_REFERENCE = {"demo1.txt": 0.30}


def _rw_curve(got, E, params=None):
    """The Radware curve on the DATA's scale.

    `rw_params` describe eff*rw_scale, so the divisor is part of reading
    them -- see efficiency.RW_SCALE_TARGETS. `params` overrides which
    parameter set is evaluated, for scoring the reference's against ours.
    """
    from efficiency import f_radware_5p
    return f_radware_5p(E, *(got.rw_params if params is None
                             else params)) / got.rw_scale


def _chi2(model, got, params, scale=1.0):
    """What `params` scores on the data `got` was fitted to.

    Evaluated through our own pipeline for both sides, so the comparison
    is of the parameters alone and not of two chi-squared conventions."""
    return float(np.sum(((got.eff - model(got.E, *params) / scale)
                         / got.deff) ** 2))


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_our_fit_reproduces_caleneff(name):
    from efficiency import f_krf, f_radware_5p

    want = GOLDEN[name]
    N, dN, E, I, dI = _load(name)
    assert len(E) == want["n"]

    got = fit_efficiency(E, N, dN, I, dI)

    # KRF still has to reproduce the reference outright. It is fitted at a
    # canonical scale now too, but that conversion is exact (the model is
    # linear in a and b), so unlike the Radware ladder it changes which
    # minimum is found and never what the answer means.
    assert f_krf(E, *got.krf_params) == pytest.approx(
        f_krf(E, *want["krf"]), rel=1e-6), (
        "%s: our KRF curve differs from CalEnEff's at the data energies"
        % name)
    assert got.krf_chi2 == pytest.approx(want["krf_chi2"], rel=1e-6)
    assert got.krf_ndf == want["krf_ndf"]
    assert got.rw_ndf == want["rw_ndf"]
    ours_krf = _chi2(f_krf, got, got.krf_params)
    theirs_krf = _chi2(f_krf, got, want["krf"])
    assert abs(ours_krf - theirs_krf) < _EQUIVALENT_CHI2, (
        "%s: our KRF parameters and CalEnEff's are not the same fit -- "
        "they score %.9f and %.9f on the same data, a gap of %.3e against "
        "a one-sigma contour of 1.0"
        % (name, ours_krf, theirs_krf, abs(ours_krf - theirs_krf)))

    # Radware: never worse than the reference. It is allowed to be BETTER,
    # because the scale ladder searches curves the reference cannot reach,
    # and on demo1 it finds one. Anywhere it merely ties, the old strict
    # comparison still applies in full.
    floor = RW_BEATS_REFERENCE.get(name)
    assert got.rw_chi2 <= want["rw_chi2"] * (1.0 + 1e-6), (
        "%s: our Radware fit is WORSE than CalEnEff's -- %.9f against "
        "%.9f. The scale ladder may only ever improve on the reference."
        % (name, got.rw_chi2, want["rw_chi2"]))

    if floor is None:
        assert got.rw_chi2 == pytest.approx(want["rw_chi2"], rel=1e-6), (
            "%s: our Radware chi-squared moved away from CalEnEff's without "
            "being listed in RW_BEATS_REFERENCE" % name)
        assert _rw_curve(got, E) == pytest.approx(
            f_radware_5p(E, *want["rw"]), rel=1e-6), (
            "%s: our Radware curve differs from CalEnEff's at the data "
            "energies" % name)
        ours = _chi2(f_radware_5p, got, got.rw_params, got.rw_scale)
        theirs = _chi2(f_radware_5p, got, want["rw"])
        assert abs(ours - theirs) < _EQUIVALENT_CHI2, (
            "%s: our Radware parameters and CalEnEff's are not the same "
            "fit -- they score %.9f and %.9f on the same data, a gap of "
            "%.3e against a one-sigma contour of 1.0"
            % (name, ours, theirs, abs(ours - theirs)))
    else:
        assert got.rw_chi2 <= want["rw_chi2"] - floor, (
            "%s: the scale ladder is expected to beat CalEnEff here by at "
            "least %.2f in chi-squared, but scored %.9f against %.9f. A "
            "ladder that stopped working would land back on the reference's "
            "value." % (name, floor, got.rw_chi2, want["rw_chi2"]))


def test_the_oracle_can_fail():
    """Control. Without this the comparison above would pass just as happily
    against a curve that is wrong everywhere, if approx were mis-set or the
    golden values were accidentally derived from our own output."""
    from efficiency import f_krf

    want = GOLDEN["226Ra_En_Area.txt"]
    _, _, E, _, _ = _load("226Ra_En_Area.txt")
    wrong = list(want["krf"])
    wrong[0] *= 1.10                      # a 10% error in one parameter
    assert f_krf(E, *wrong) != pytest.approx(f_krf(E, *want["krf"]), rel=1e-6)


def test_the_equivalence_check_is_blind_only_to_the_free_direction():
    """Control for check 3: it must ignore the degeneracy and nothing else.

    A check blind to everything would pass this file however wrong the fit
    was -- the mirror image of the old parameter comparison, which failed
    on a difference that was not there. So move demo2's Radware parameters
    ALONG the direction its data leave free, and ACROSS it, and require
    opposite verdicts.

    Magnitudes measured 2026-09-21. Along the free direction the cost
    saturates at 1.4e-7: shifting a2 by 100, about 2.5x its own value, is
    still free, which is what an exactly flat direction means. Across it a
    relative 1e-5 in a4 already costs 6.6e-4. So this check is STRICTER
    than the rel=1e-4 comparison it replaced on the parameters the data
    actually determine, and blind only where they determine nothing."""
    from efficiency import f_radware_5p

    N, dN, E, I, dI = _load("demo2.txt")
    got = fit_efficiency(E, N, dN, I, dI)
    base = np.asarray(GOLDEN["demo2.txt"]["rw"])
    reference = _chi2(f_radware_5p, got, base)

    # Along: a1 + a2*ln(E_min/100) held constant while a2 moves by 100.
    x0 = float(np.log(E.min() * 0.01))
    free = base.copy()
    free[0] -= x0 * 100.0
    free[1] += 100.0
    assert abs(_chi2(f_radware_5p, got, free) - reference) < _EQUIVALENT_CHI2, (
        "check 3 is not blind to the direction demo2's data leave free -- "
        "it would fail on a harmless reparameterisation again"
    )

    # Across: the same parameter moved OFF that line, and the determined
    # branch nudged. Both must cost far more than the tolerance allows.
    for index, factor, label in ((1, 1.001, "a2"), (2, 1.0001, "a4")):
        broken = base.copy()
        broken[index] *= factor
        cost = abs(_chi2(f_radware_5p, got, broken) - reference)
        assert cost > _EQUIVALENT_CHI2, (
            "check 3 cannot see a relative %.0e error in %s: it costs only "
            "%.3e, so a genuinely different fit would pass"
            % (factor - 1.0, label, cost)
        )


def test_birge_is_one_when_there_is_no_redundancy():
    """ndf <= 0 carries no scatter information, so the reference returns 1.0
    rather than dividing by zero. Reachable here: six points against
    Radware's five free parameters leaves ndf = 1, and five would leave 0."""
    from efficiency import birge

    assert birge(10.0, 0) == 1.0
    assert birge(10.0, -1) == 1.0
    assert birge(8.0, 2) == pytest.approx(2.0)


def test_efficiency_points_and_their_errors():
    """eps = N/I with the two relative errors added in quadrature."""
    from efficiency import efficiency_points

    N = np.array([1000.0]); dN = np.array([30.0])
    I = np.array([50.0]); dI = np.array([2.0])
    eff, deff = efficiency_points(N, dN, I, dI)
    assert eff[0] == pytest.approx(20.0)
    assert deff[0] == pytest.approx(
        20.0 * np.sqrt((30.0 / 1000.0) ** 2 + (2.0 / 50.0) ** 2))


# --- normalisation -------------------------------------------------------


def _demo_result(name="demo1.txt"):
    from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo

    N, dN, E, I, dI = _load(name)
    fit = fit_efficiency(E, N, dN, I, dI)
    mc = run_monte_carlo(fit, N, dN, I, dI, iterations=200)
    return EfficiencyResult(fit=fit, mc=mc, model="krf")


def _norm_grid(r):
    """The SAME grid _compute_normalisation searches for the peak.

    It deliberately runs 10% past the data on each side, as the reference
    does. Testing the peak over the narrower data range instead would be
    wrong: for KRF the maximum falls OUTSIDE the measured points on two of
    the three reference datasets (109.6 keV against data from 122 keV on
    demo2; 167.6 keV against data from 186 keV on Ra-226), so such a test
    would pass on demo1 by luck and fail on the others.
    """
    lo = max(float(r.fit.E.min()) * 0.9, 1.0)
    return np.linspace(lo, float(r.fit.E.max()) * 1.1, 2000)


def test_the_selected_curve_peaks_at_one():
    r = _demo_result()
    assert np.nanmax(r.curve(_norm_grid(r))) == pytest.approx(1.0, rel=1e-6)


def test_the_peak_may_fall_outside_the_measured_points():
    """Not a defect, and worth pinning so nobody "fixes" it. A detector's
    efficiency crests below the lowest measured line on real data, so within
    the data range the curve can sit just under 1 while still peaking at
    exactly 1 on the normalisation grid."""
    r = _demo_result("226Ra_En_Area.txt")
    inside = np.linspace(r.fit.E.min(), r.fit.E.max(), 2000)
    assert np.nanmax(r.curve(inside)) <= 1.0 + 1e-9
    assert np.nanmax(r.curve(_norm_grid(r))) == pytest.approx(1.0, rel=1e-6)


def test_switching_model_rescales_both_curves():
    """The normalisation is the selected model's peak, so selecting the other
    model changes the numbers written to file. That is intended and is why
    the header records which model was active."""
    r = _demo_result()
    before = r.normalisation
    r.model = "rw"
    assert r.normalisation != pytest.approx(before)
    assert np.nanmax(r.curve(_norm_grid(r))) == pytest.approx(1.0, rel=1e-6)


def test_the_unselected_curve_may_exceed_one():
    """Both curves share one scale so the plot shows their real difference.
    Clipping the other to 1 would hide exactly the disagreement the second
    model exists to reveal."""
    r = _demo_result()
    grid = np.linspace(r.fit.E.min(), r.fit.E.max(), 500)
    other = r.curve(grid, model="rw")
    assert np.all(np.isfinite(other))


def test_the_normalised_curve_ignores_the_intensity_scale():
    """Our .sou intensities are on a 0-10000 scale and the reference's are
    percentages. Since eps = N/I is relative and is normalised again, a
    constant factor on every intensity must leave the normalised curve
    completely unchanged. Without this, the port would appear to work on the
    reference's data and quietly produce a differently-scaled curve on ours.

    KRF ONLY, and that is the gap this test used to have. The invariant
    above is the right one, but it was only ever exercised against the model
    that satisfies it by construction, so Radware breaking it went unnoticed
    until the 2026-09-21 audit -- by up to 12% of the reported curve. Radware
    is covered by
    test_the_radware_fit_does_not_depend_on_the_intensity_scale, which
    compares chi-squared rather than the curve: the best-fit curve still
    wanders along the flat direction described in this module's docstring,
    so a rel=1e-6 curve comparison would be pinning that wander and not this
    invariant. Do not "extend" this test to rw expecting it to pass."""
    from efficiency import EfficiencyResult, fit_efficiency

    N, dN, E, I, dI = _load("demo1.txt")
    grid = np.linspace(E.min(), E.max(), 200)

    a = EfficiencyResult(fit=fit_efficiency(E, N, dN, I, dI), mc=None,
                         model="krf")
    b = EfficiencyResult(fit=fit_efficiency(E, N, dN, I * 137.0, dI * 137.0),
                         mc=None, model="krf")
    assert a.curve(grid) == pytest.approx(b.curve(grid), rel=1e-6)


def test_the_curve_is_the_mc_mean_not_the_best_fit():
    """The reported efficiency is the Monte Carlo mean (user decision). The
    best fit is a different curve and the difference is small but real, so
    returning it here would silently change every saved number and every
    corrected spectrum."""
    r = _demo_result()
    grid = np.linspace(r.fit.E.min(), r.fit.E.max(), 40)
    expected = r._mc_mean_raw(grid, "krf") * r.normalisation
    assert r.curve(grid) == pytest.approx(expected, rel=1e-12, nan_ok=True)


def test_the_best_fit_would_fail_that():
    """Control, and the reason the test above is not vacuous: the two curves
    really are different. If this ever passes, the MC has collapsed onto the
    best fit and neither test is testing anything."""
    from efficiency import f_krf

    r = _demo_result()
    grid = np.linspace(r.fit.E.min(), r.fit.E.max(), 40)
    best = f_krf(grid, *r.fit.krf_params) * r.normalisation
    assert r.curve(grid) != pytest.approx(best, rel=1e-12)


def test_the_applied_curve_is_the_one_that_peaks_at_one():
    """The normalisation divides by the MC mean's peak, not the best fit's,
    so the curve actually applied is the one bounded by 1. Checked by
    confirming the best fit's own peak does NOT land on 1 -- if it did, the
    two normalisations would be indistinguishable here."""
    r = _demo_result()
    grid = _norm_grid(r)
    assert np.nanmax(r.curve(grid)) == pytest.approx(1.0, rel=1e-6)
    best = r.best_fit_raw(grid, "krf") * r.normalisation
    assert np.nanmax(best) != pytest.approx(1.0, rel=1e-9)


def test_predict_reports_the_mc_mean_and_the_best_fit_separately():
    """The examine panel shows both numbers side by side, as CalEnEff does.
    They must be two different quantities: reading the best fit from curve()
    would report the MC mean twice, and the difference between them is the
    whole reason both are shown."""
    r = _demo_result()
    energy = float(np.median(r.fit.E))
    mean, sigma, best = r.predict(energy)

    assert np.isfinite(mean) and np.isfinite(sigma) and np.isfinite(best)
    assert mean == pytest.approx(float(r.curve([energy])[0]), rel=1e-12)
    assert best == pytest.approx(
        float(r.best_fit_raw([energy], "krf")[0]) * r.normalisation, rel=1e-12)
    assert mean != pytest.approx(best, rel=1e-12), (
        "predict returned the same number for the MC mean and the best fit")
    assert sigma > 0.0


def test_predict_still_gives_the_best_fit_without_a_monte_carlo():
    """The best fit does not depend on the MC, so mc=None must not blank it."""
    from efficiency import EfficiencyResult, fit_efficiency

    N, dN, E, I, dI = _load("demo1.txt")
    r = EfficiencyResult(fit=fit_efficiency(E, N, dN, I, dI), mc=None,
                         model="krf")
    mean, sigma, best = r.predict(float(np.median(E)))
    assert np.isnan(mean) and np.isnan(sigma)
    assert np.isfinite(best) and best > 0.0


def test_an_unnormalised_curve_would_fail_that():
    """Control: raw eps does depend on the intensity scale, by exactly the
    factor applied."""
    from efficiency import f_krf, fit_efficiency

    N, dN, E, I, dI = _load("demo1.txt")
    raw_a = f_krf(E, *fit_efficiency(E, N, dN, I, dI).krf_params)
    raw_b = f_krf(E, *fit_efficiency(E, N, dN, I * 137.0,
                                     dI * 137.0).krf_params)
    assert raw_a != pytest.approx(raw_b, rel=1e-6)


#: Largest relative spread in rw_chi2 across the intensity scales below that
#: still counts as scale-free. Measured 2026-09-21 over eight scales spanning
#: 1e-3 to 1e6: worst 5.2e-09 (226Ra), demo1 4.0e-12, demo2 4.8e-10. Before
#: the scale ladder the same sweep spread 2.1e-02, 2.4e-02 and 5.7e-02, so
#: this sits ~200x above the noise and four orders below the defect.
_SCALE_FREE = 1e-6

#: Spans the full range our own .sou files disagree over -- sou_io records
#: most peaking at 10000, co56.sou at 100000 and na24.sou at 1000 -- with
#: room either side.
_INTENSITY_SCALES = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e4, 1e6)


def _chi2_spread(name, targets=None):
    """Relative spread of the Radware chi-squared over _INTENSITY_SCALES.

    The intensity column is a RELATIVE scale, so multiplying every line of
    one source by a constant describes the same physics and must give the
    same fit. `targets` overrides efficiency.RW_SCALE_TARGETS, which is how
    the control below reproduces the old behaviour.
    """
    import efficiency

    N, dN, E, I, dI = _load(name)
    saved = efficiency.RW_SCALE_TARGETS
    got = []
    try:
        for s in _INTENSITY_SCALES:
            if targets == "own":
                eff, _ = efficiency.efficiency_points(N, dN, I / s, dI / s)
                efficiency.RW_SCALE_TARGETS = (
                    float(np.exp(np.mean(np.log(np.maximum(eff, 1e-300))))),)
            got.append(fit_efficiency(E, N, dN, I / s, dI / s).rw_chi2)
    finally:
        efficiency.RW_SCALE_TARGETS = saved
    return (max(got) - min(got)) / float(np.mean(got))


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_the_radware_fit_does_not_depend_on_the_intensity_scale(name):
    """The bug this guards: the Radware family is not closed under a
    constant factor on eps, so fitting the intensities as given made the
    answer depend on an arbitrary constant in the source file -- one our own
    files do not even agree on. Measured at up to 12% of the reported curve.
    """
    spread = _chi2_spread(name)
    assert spread < _SCALE_FREE, (
        "%s: the Radware fit still depends on the intensity scale -- chi2 "
        "spreads by %.3e across factors of %g to %g, against a limit of %.0e"
        % (name, spread, min(_INTENSITY_SCALES), max(_INTENSITY_SCALES),
           _SCALE_FREE))


def test_that_scale_check_can_fail():
    """Control. Without the ladder the fit is done at whatever scale the
    caller's intensities happen to be on, which is exactly the defect. If
    this ever passes, the check above is measuring nothing."""
    spread = _chi2_spread("demo2.txt", targets="own")
    assert spread > 1e-3, (
        "the scale-free check cannot detect the bug it guards: fitting at "
        "the data's own scale spread chi2 by only %.3e" % spread)


#: Eight points that expose KRF's conditioning problem. Found by sweeping
#: 150 synthetic data sets on 2026-09-22; the three CalEnEff fixtures are all
#: well conditioned and show none of it, which is why it stayed invisible.
#: Kept as literals rather than regenerated so the case cannot drift away
#: from under the test.
_ILL_CONDITIONED = {
    "E":  (712.684, 861.667, 1055.5, 1099.85, 1216.52, 1240.93, 1429.48,
           1458.97),
    "N":  (1393.6, 1703.89, 1356.83, 113.403, 822.243, 471.786, 581.889,
           877.972),
    "dN": (7.97158, 60.1638, 33.4524, 5.39977, 27.3365, 8.06416, 21.2795,
           26.614),
    "I":  (3295.49, 4270.63, 4826.71, 437.052, 3034.21, 1989.43, 2434.29,
           3601.65),
    "dI": (141.566, 116.188, 219.386, 15.5743, 71.1513, 19.5244, 67.368,
           147.967),
}


def _ill_conditioned():
    d = _ILL_CONDITIONED
    return (np.array(d["E"]), np.array(d["N"]), np.array(d["dN"]),
            np.array(d["I"]), np.array(d["dI"]))


def test_the_krf_fit_does_not_depend_on_the_intensity_scale():
    """KRF's solution is scale-invariant; its SEARCH was not.

    The seeds' a and b track the data's magnitude and trf controls its steps
    in parameter space, so how well the fit converged depended on the
    arbitrary normalisation of the intensity column -- which sou_io records
    our own files disagreeing on by 100x. On this data the old path scored
    21.707 at two scales and 136.412 at the third.

    fit_efficiency now fits at a canonical scale and converts back, which is
    exact for KRF because the model is linear in a and b.
    """
    E, N, dN, I, dI = _ill_conditioned()
    got = [fit_efficiency(E, N, dN, I * s, dI * s).krf_chi2
           for s in (1.0, 10.0, 100.0)]
    spread = (max(got) - min(got)) / float(np.mean(got))
    assert spread < 1e-6, (
        "the KRF fit still depends on the intensity scale: chi-squared came "
        "out %s across factors of 1, 10 and 100" % [round(v, 6) for v in got])


def test_that_krf_scale_check_can_fail():
    """Control. Fitting this data WITHOUT the canonical scale must still
    drift, or the test above is pinning nothing -- most data is well
    conditioned and passes it for free."""
    import efficiency

    E, N, dN, I, dI = _ill_conditioned()
    got = []
    for s in (1.0, 10.0, 100.0):
        eff, deff = efficiency.efficiency_points(N, dN, I * s, dI * s)
        p = efficiency.multistart(
            efficiency.f_krf, E, eff, deff, efficiency._krf_seeds(E, eff),
            bounds=efficiency.KRF_BOUNDS, method="trf")
        assert p is not None
        got.append(float(np.sum(((eff - efficiency.f_krf(E, *p)) / deff) ** 2)))
    spread = (max(got) - min(got)) / float(np.mean(got))
    assert spread > 0.5, (
        "the un-normalised fit no longer drifts on this data (chi-squared "
        "%s), so test_the_krf_fit_does_not_depend_on_the_intensity_scale is "
        "measuring nothing" % [round(v, 6) for v in got])


def test_the_canonical_scale_conversion_is_exact_for_krf():
    """The conversion back must be exact, not approximate. KRF is linear in
    a and b, so fitting eps/g and multiplying them by g has to reproduce the
    same curve -- this is what lets fit_efficiency avoid carrying a divisor
    the way Radware does with rw_scale."""
    from efficiency import f_krf

    E, N, dN, I, dI = _ill_conditioned()
    fit = fit_efficiency(E, N, dN, I, dI)
    a, b, c, d = fit.krf_params
    g = 1234.5
    scaled = f_krf(E, a * g, b * g, c, d)
    assert scaled == pytest.approx(g * f_krf(E, a, b, c, d), rel=1e-12), (
        "scaling a and b no longer scales the KRF curve exactly; the "
        "canonical-scale conversion in fit_efficiency relies on it")


def test_canonical_scale_ignores_non_positive_points():
    """Both fits run at canonical_scale(eff) and convert back, so a single
    bad point must not be able to destroy that scale.

    A peak area can come out negative on a weak line after background
    subtraction. Clamping such a point to 1e-300 instead of dropping it
    pulled the geometric mean to 3.1e-50 on six points, handing the fit
    values around 1e+50 -- finite and positive, so nothing downstream would
    have caught it.
    """
    from efficiency import canonical_scale

    healthy = np.array([9.0, 7.0, 5.0, 3.8, 3.0, 2.4])
    clean = canonical_scale(healthy)
    assert clean == pytest.approx(float(np.exp(np.mean(np.log(healthy)))))

    for spoiled in ([-50.0] + list(healthy[1:]),
                    [0.0] + list(healthy[1:]),
                    [np.nan] + list(healthy[1:]),
                    [-50.0, -20.0, -5.0] + list(healthy[3:])):
        got = canonical_scale(np.array(spoiled, dtype=float))
        assert 0.1 < got < 100.0, (
            "one bad point moved the canonical scale to %.3e; it is meant "
            "to be the geometric mean of the POSITIVE points" % got)

    assert canonical_scale(np.array([-9.0, -7.0, -5.0])) == 1.0
    assert canonical_scale(np.array([])) == 1.0


def test_the_canonical_scale_is_equivariant():
    """Scaling every point must scale the answer by the same factor -- that
    property is the whole reason both fits are independent of the intensity
    column's normalisation. Without it the ladder rungs would move with the
    caller's units."""
    from efficiency import canonical_scale

    eff = np.array([9.0, 7.0, 5.0, 3.8, 3.0, 2.4])
    base = canonical_scale(eff)
    for factor in (1e-9, 1e-3, 7.0, 1e3, 1e9):
        assert canonical_scale(eff * factor) == pytest.approx(
            base * factor, rel=1e-12), (
            "canonical_scale is not equivariant at factor %g" % factor)
