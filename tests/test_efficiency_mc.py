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


def test_the_band_can_be_centred_somewhere_else(mc):
    """Task 5 reports the MC mean rather than the best fit, and the band has
    to follow it. Passing an explicit centre must move the band."""
    from efficiency import f_kfr

    fit, out = mc
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
    for sigma in seen:
        assert sigma == pytest.approx(fit.deff), (
            "a refit used weights other than the original deff")


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
