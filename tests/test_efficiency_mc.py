"""The Monte Carlo that turns two best-fit curves into uncertainty bands."""

import os

import numpy as np
import pytest

from efficiency import fit_efficiency, run_monte_carlo, N_MC_EFFICIENCY

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")


def _load(name):
    data = np.loadtxt(os.path.join(FIXTURES, name), ndmin=2)
    return data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]


@pytest.fixture(scope="module")
def mc():
    """One run, shared: 10,000 iterations twice over is a minute of suite
    time and every test below reads the same object."""
    N, dN, E, I, dI = _load("demo1.txt")
    fit = fit_efficiency(E, N, dN, I, dI)
    return fit, run_monte_carlo(fit, N, dN, I, dI, iterations=400)


def test_it_stores_one_parameter_set_per_accepted_sample(mc):
    _fit, out = mc
    assert out.kfr_samples.ndim == 2 and out.kfr_samples.shape[1] == 4
    assert out.rw_samples.ndim == 2 and out.rw_samples.shape[1] == 5
    assert out.kfr_accepted == len(out.kfr_samples)
    assert out.kfr_accepted > 0


def test_rejected_samples_are_counted_not_hidden(mc):
    """A band from 200 survivors means something different from one built on
    9,900. Reporting the number is what lets the user tell them apart."""
    _fit, out = mc
    assert out.rejected >= 0
    assert out.kfr_accepted + out.rejected <= 400 + 1


def test_the_same_seed_reproduces(mc):
    fit, first = mc
    N, dN, E, I, dI = _load("demo1.txt")
    again = run_monte_carlo(fit, N, dN, I, dI, iterations=400)
    assert again.kfr_samples == pytest.approx(first.kfr_samples)


def test_a_different_seed_does_not(mc):
    """Control for the reproducibility test: if the RNG were ignored
    entirely, that test would pass for the wrong reason."""
    fit, first = mc
    N, dN, E, I, dI = _load("demo1.txt")
    other = run_monte_carlo(fit, N, dN, I, dI, iterations=400, seed=7)
    assert other.kfr_samples != pytest.approx(first.kfr_samples)


def test_the_band_brackets_its_centre(mc):
    """The 1-sigma band is built around a centre curve, so it must contain
    that curve at every energy it is drawn at."""
    from efficiency import f_kfr

    fit, out = mc
    grid = np.linspace(fit.E.min(), fit.E.max(), 50)
    lo, hi = out.kfr_band(grid, fit)
    centre = f_kfr(grid, *fit.kfr_params)
    assert np.all(lo <= centre + 1e-12)
    assert np.all(hi >= centre - 1e-12)


def test_the_band_can_be_centred_somewhere_else():
    """Task 5 reports the MC mean rather than the best fit, and the band has
    to follow it. Passing an explicit centre must move the band.

    demo2, NOT the demo1 fixture the rest of this file shares, and that is
    load-bearing. The band is `centre +- factor * (centre - percentile)`, so
    when factor is exactly 1 the centre cancels and the band IS the raw
    percentile interval, unmoved by any centre passed in. demo1's Birge
    ratio is below 1 and is clamped to exactly 1 (see
    test_the_birge_scaling_never_narrows_the_band), so this property has no
    content there. demo2 scores B = 1.78, where the scaling is live and the
    centre really does anchor it.
    """
    from efficiency import f_kfr

    N, dN, E, I, dI = _load("demo2.txt")
    fit = fit_efficiency(E, N, dN, I, dI)
    assert fit.kfr_birge > 1.0, (
        "demo2 no longer inflates (B = %.4f), so the centre cancels out of "
        "the band and this test is vacuous" % fit.kfr_birge)
    out = run_monte_carlo(fit, N, dN, I, dI, iterations=400)

    grid = np.linspace(fit.E.min(), fit.E.max(), 20)
    default = out.kfr_band(grid, fit)
    shifted = out.kfr_band(grid, fit, f_kfr(grid, *fit.kfr_params) * 1.5)
    assert shifted[0] != pytest.approx(default[0])


def test_too_few_survivors_reports_nan_not_a_fake_sigma(mc):
    """Fewer than 10 finite samples cannot support a sigma. Saying so is
    better than returning a number computed from three points."""
    from efficiency import finite_mean_std

    assert np.isnan(finite_mean_std(np.array([1.0, 2.0, 3.0]), min_n=10)[1])
    mean, std = finite_mean_std(np.arange(50.0), min_n=10)
    assert np.isfinite(mean) and np.isfinite(std)


def test_cancelling_stops_the_run():
    """progress() returning False must end the loop, or a Cancel button does
    nothing until the full minute is up."""
    N, dN, E, I, dI = _load("demo1.txt")
    fit = fit_efficiency(E, N, dN, I, dI)
    out = run_monte_carlo(fit, N, dN, I, dI, iterations=5000,
                          progress=lambda done, total: done < 100)
    assert out.kfr_accepted < 500, "cancel was ignored"


def test_the_default_iteration_count_matches_the_reference():
    assert N_MC_EFFICIENCY == 10000


def test_the_mc_weights_never_change_between_samples():
    """Every refit must use the ORIGINAL deff, not one recomputed from the
    resampled N and I.

    Recomputing looks like a correction and is not one: the fit weights
    would then vary with the noise draw, which changes what the spread of
    fitted parameters actually measures. The reference passes sigma=deff
    unchanged inside its loop.

    Nothing else in this file would notice if that stopped being true. The
    samples would still be finite, still reproducible under a fixed seed,
    and still bracketed by their own band -- so this invariant needs its own
    test or it is guarded by nothing.
    """
    import efficiency

    N, dN, E, I, dI = _load("demo1.txt")
    fit = fit_efficiency(E, N, dN, I, dI)

    efficiency._need_scipy()
    real = efficiency.curve_fit
    seen = []

    def recording(*args, **kwargs):
        seen.append(np.asarray(kwargs["sigma"], dtype=float).copy())
        return real(*args, **kwargs)

    efficiency.curve_fit = recording
    try:
        run_monte_carlo(fit, N, dN, I, dI, iterations=25)
    finally:
        efficiency.curve_fit = real

    assert len(seen) > 25, "expected roughly two refits per iteration"

    # Two weight vectors are legitimate, and both are FIXED for the whole
    # run: deff for the KFR arm, and deff*rw_scale for the Radware arm,
    # which fits a scaled copy of eps (see efficiency.RW_SCALE_TARGETS).
    # Scaling data and sigma by the same constant leaves chi-squared
    # unchanged, so that arm weights the points identically.
    #
    # The invariant is unchanged and so is this test's grip on it: weights
    # recomputed from the resampled N and I would move with the draw, and
    # so would NOT be a constant multiple of the original deff.
    deff = np.asarray(fit.deff, dtype=float)
    allowed = (1.0, float(fit.rw_scale))
    for sigma in seen:
        ratio = np.asarray(sigma, dtype=float) / deff
        assert ratio == pytest.approx(ratio[0]), (
            "a refit's weights are not a constant multiple of the original "
            "deff -- they were recomputed from the resampled data")
        assert min(abs(ratio[0] / a - 1.0) for a in allowed) < 1e-9, (
            "a refit used weights scaled by %.6g, which is neither 1 nor "
            "rw_scale (%.6g)" % (ratio[0], fit.rw_scale))


def test_recomputed_weights_really_would_differ():
    """Control for the test above. If resampling happened not to move deff,
    that assertion would hold no matter how the weights were computed and
    would be pinning nothing."""
    from efficiency import efficiency_points

    N, dN, E, I, dI = _load("demo1.txt")
    _eff, deff = efficiency_points(N, dN, I, dI)
    rng = np.random.default_rng(1)
    _eff_s, deff_s = efficiency_points(rng.normal(N, dN), dN,
                                       rng.normal(I, dI), dI)
    assert deff_s != pytest.approx(deff)


def test_an_all_nan_energy_is_quiet_but_still_nan():
    """Channel 0 maps to zero or negative energy under most calibrations, so
    every Monte Carlo sample diverges there and nanpercentile warns. That is
    ordinary, not exceptional -- it is precisely what efficiency_apply's
    zeroing rule handles -- so the warning is suppressed while the NaN it
    describes is still returned and still propagates.

    Pinned because silencing a warning is easy to overdo. Note what each
    assertion actually guards, since they are not equally strong: removing
    the suppression fails the first one, while the NaN in the second is
    guaranteed by the band's CENTRE being NaN at that energy, not by the
    percentile call. So this pins the behaviour a user sees -- quiet, and
    still NaN -- rather than the internals of band_percentiles.
    """
    import warnings

    from efficiency import EfficiencyResult

    N, dN, E, I, dI = _load("demo1.txt")
    fit = fit_efficiency(E, N, dN, I, dI)
    r = EfficiencyResult(fit=fit,
                         mc=run_monte_carlo(fit, N, dN, I, dI, iterations=100),
                         model="kfr")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lo, hi = r.band(np.array([0.0, 100.0, 200.0]))

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)], \
        [str(w.message) for w in caught]
    assert np.isnan(lo[0]) and np.isnan(hi[0]), "the bad energy stopped being NaN"
    assert np.isfinite(lo[1]) and np.isfinite(hi[1]), "a good energy was lost"


def test_the_band_is_withheld_where_too_few_samples_survive():
    """The band now applies MIN_MC_SAMPLES, as the MC mean and predict()
    always have.

    Without it band_percentiles takes its percentiles over however few
    samples happened to stay finite, so three survivors out of ten thousand
    draw a band indistinguishable from any other. The mean at such an
    energy is already NaN, which left the band as the one place a number
    with nothing behind it could still be drawn and exported.

    `func` is supplied rather than using a real model so the survivor count
    is set exactly, including the boundary: MIN_MC_SAMPLES itself is enough,
    one fewer is not.
    """
    from efficiency import MIN_MC_SAMPLES, EfficiencyMC

    n = 40
    samples = np.ones((n, 1))
    grid = np.array([10.0, 20.0, 30.0])

    def func(g, a):
        out = np.broadcast_to(g, (len(a), g.shape[1])).astype(float).copy()
        row = np.arange(len(a))
        out[:, 1] = np.where(row < MIN_MC_SAMPLES, 1.0, np.nan)
        out[:, 2] = np.where(row < MIN_MC_SAMPLES - 1, 1.0, np.nan)
        return out

    survivors = np.isfinite(func(grid[None, :], samples)).sum(axis=0)
    # Without this the test could pass having exercised nothing.
    assert survivors.tolist() == [n, MIN_MC_SAMPLES, MIN_MC_SAMPLES - 1]

    mc = EfficiencyMC(kfr_samples=None, rw_samples=samples,
                      kfr_accepted=0, rw_accepted=n, rejected=0)
    lo, hi = mc._band(grid, np.ones(3), samples, func, 1.0)

    assert np.isfinite(lo[0]) and np.isfinite(hi[0]), "plenty of samples"
    assert np.isfinite(lo[1]) and np.isfinite(hi[1]), "exactly MIN is enough"
    assert np.isnan(lo[2]) and np.isnan(hi[2]), (
        "a band was drawn from fewer than MIN_MC_SAMPLES survivors")


def _band_vs_percentiles(name, model):
    """(reported band width) / (raw Monte Carlo percentile width).

    The raw width is the same call with the Birge factor forced to 1, so
    the two differ only by the scaling under test.
    """
    import efficiency

    N, dN, E, I, dI = _load(name)
    fit = fit_efficiency(E, N, dN, I, dI)
    mc = run_monte_carlo(fit, N, dN, I, dI, iterations=800,
                         seed=efficiency.EFFICIENCY_SEED)
    grid = np.linspace(E.min(), E.max(), 80)

    if model == "kfr":
        func, samples, scale = efficiency.f_kfr, mc.kfr_samples, 1.0
        centre = func(grid, *fit.kfr_params)
    else:
        func, samples, scale = (efficiency.f_radware_5p, mc.rw_samples,
                                fit.rw_scale)
        centre = func(grid, *fit.rw_params) / scale

    reported = (mc.kfr_band(grid, fit) if model == "kfr"
                else mc.rw_band(grid, fit))
    raw = mc._band(grid, centre, samples, func, 1.0, scale=scale)

    width = lambda b: float(np.nanmax(np.asarray(b[1]) - np.asarray(b[0])))
    return width(reported) / width(raw), getattr(fit, model + "_birge")


@pytest.mark.parametrize("model", ("kfr", "rw"))
@pytest.mark.parametrize("name", ("226Ra_En_Area.txt", "demo1.txt",
                                  "demo2.txt"))
def test_the_birge_scaling_never_narrows_the_band(name, model):
    """A Birge ratio below 1 must leave the band alone, not shrink it.

    B < 1 means the fit tracks the points better than their stated errors
    require. The honest reading is that those errors were overstated, not
    that the curve is known more sharply than the resampling found, so
    scaling down on it would report a precision nothing measured. This is
    the PDG convention and a deliberate break from the reference, which
    scales unconditionally.

    demo1 is the case that makes it real: B = 0.891 (KFR) and 0.640
    (Radware), the second of which got worse when the scale ladder improved
    that fit.
    """
    ratio, b = _band_vs_percentiles(name, model)
    assert ratio >= 1.0 - 1e-9, (
        "%s %s: the reported band is %.4f of the Monte Carlo percentiles, "
        "i.e. narrower than the spread it came from (B = %.4f)"
        % (name, model, ratio, b))
    expected = max(1.0, b)
    assert ratio == pytest.approx(expected, rel=1e-6), (
        "%s %s: band scaled by %.6f, expected max(1, B) = %.6f"
        % (name, model, ratio, expected))


def test_a_birge_below_one_really_does_occur():
    """Control. Every assertion above holds trivially if no fixture ever
    produces B < 1 -- the clamp would then never be exercised and the test
    would be pinning nothing."""
    N, dN, E, I, dI = _load("demo1.txt")
    fit = fit_efficiency(E, N, dN, I, dI)
    assert fit.kfr_birge < 1.0 and fit.rw_birge < 1.0, (
        "demo1 no longer produces a Birge ratio below 1 (KFR %.4f, Radware "
        "%.4f), so the clamp is untested. Find a fixture that does."
        % (fit.kfr_birge, fit.rw_birge))


def test_the_reported_birge_is_still_the_true_ratio():
    """The clamp belongs to the band, not to the number. B < 1 says
    something real about the input errors, so the dialog and the export must
    keep showing it rather than a floored 1.0."""
    N, dN, E, I, dI = _load("demo1.txt")
    fit = fit_efficiency(E, N, dN, I, dI)
    assert fit.kfr_birge == pytest.approx(
        np.sqrt(fit.kfr_chi2 / fit.kfr_ndf))
    assert fit.rw_birge == pytest.approx(np.sqrt(fit.rw_chi2 / fit.rw_ndf))
    assert fit.rw_birge < 1.0, "expected demo1 Radware to sit below 1"
