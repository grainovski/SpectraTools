"""Relative detector efficiency: the curve shapes.

A port of CalEnEff's two models (C:\\Users\\RIG\\Documents\\Claude\\efficieny,
ra226_gui.py). The formulae are not ours and are not improved on here --
matching the reference exactly is what lets tests/test_efficiency_fit.py
check this port against it.

scipy is imported lazily throughout this module. v5.2.5 took scipy off the
application's startup path and tests/test_startup_imports.py enforces it; a
module-level scipy import here would put it straight back.
"""

from dataclasses import dataclass

import numpy as np

#: Radford's own defaults in effit.c (freepars[2] = 0, freepars[6] = 0),
#: held fixed to leave five free parameters. Fixing them is what makes the
#: Levenberg-Marquardt fit stable on typical HPGe data.
RADWARE_C = 0.0
RADWARE_G = 15.0


def f_kfr(E, a, b, c, d):
    """KFR 4-parameter efficiency: eps(E) = (aE + b/E) * exp(cE + d/E)."""
    E = np.asarray(E, dtype=float)
    return (a * E + b / E) * np.exp(c * E + d / E)


def f_radware(E, a1, a2, a3, a4, a5, a6, g):
    """Radware 7-parameter efficiency (Radford, EFFIT v4.0, effit.c).

        ln eps(E) = f * (1 + r**g)**(-1/g)
        f1 = a1 + a2*x + a3*x**2      x = ln(E/100)
        f2 = a4 + a5*y + a6*y**2      y = ln(E/1000)
        f  = min(f1, f2)              F = max(f1, f2)      r = f/F

    f1 and f2 are LOG-efficiencies and are normally negative. They must not
    be clipped positive: the reference records that doing so made the loss
    landscape pathological and stalled the Monte Carlo.
    """
    E = np.asarray(E, dtype=float)
    x = np.log(E * 0.01)
    y = np.log(E * 0.001)
    f1 = a1 + (a2 + a3 * x) * x
    f2 = a4 + (a5 + a6 * y) * y

    f = np.minimum(f1, f2)
    F = np.maximum(f1, f2)
    # F == 0 is astronomically unlikely; the additive shift avoids a divide
    # without allocating a second array through np.where.
    r = np.maximum(f / (F + (F == 0) * 1e-300), 1e-300)

    # (1 + r**g)**(-1/g) via log-sum-exp, which is what keeps this finite
    # when the curve is extrapolated far outside the fitted range.
    g_log_r = np.clip(g * np.log(r), -700.0, 700.0)
    log_y3 = -np.logaddexp(0.0, g_log_r) / g
    log_eff = np.clip(f * np.exp(log_y3), -700.0, 700.0)
    return np.exp(log_eff)


def f_radware_5p(E, a1, a2, a4, a5, a6):
    """The 7-parameter model with Radford's C and G defaults held fixed."""
    return f_radware(E, a1, a2, RADWARE_C, a4, a5, a6, RADWARE_G)


curve_fit = None


def _need_scipy():
    """Bind scipy on first use. See this module's docstring for why it is
    not imported at the top."""
    global curve_fit
    if curve_fit is None:
        from scipy.optimize import curve_fit as _cf
        curve_fit = _cf


def kfr_seed(E, eff):
    """Data-driven KFR start: a and b from the geometric means of E and eps
    so the model reproduces the right scale, c and d small so the
    exponential starts near 1."""
    E = np.asarray(E, dtype=float)
    eff = np.asarray(eff, dtype=float)
    Eg = float(np.exp(np.mean(np.log(np.maximum(E, 1e-30)))))
    eg = float(np.exp(np.mean(np.log(np.maximum(eff, 1e-30)))))
    return [eg / (2.0 * Eg), eg * Eg / 2.0, -1e-4, 1.0]


def radware_seed_5p(E, eff):
    """Radford's parset() seed, with the two fixed entries stripped.

    Anchors the low-energy polynomial at 100 keV and the high-energy one at
    1000 keV using whichever data points sit nearest those references.
    """
    E = np.asarray(E, dtype=float)
    eff = np.asarray(eff, dtype=float)
    ix1 = int(np.argmin(np.abs(E - 100.0)))
    ix2 = int(np.argmin(np.abs(E - 1000.0)))
    a2, a5 = 1.5, -0.9
    a1 = float(np.log(max(eff[ix1], 1e-30)) + a2 * (np.log(100.0) - np.log(E[ix1])))
    a4 = float(np.log(max(eff[ix2], 1e-30)) + a5 * (np.log(1000.0) - np.log(E[ix2])))
    return [a1, a2, a4, a5, 0.0]


def radware_seed_polyfit_5p(E, eff):
    """Secondary seed: independent quadratic polyfits of ln eps against
    ln(E/100) and ln(E/1000). Less reliable than parset() but useful when
    eps is far from a clean power law."""
    E = np.asarray(E, dtype=float)
    lne = np.log(np.maximum(np.asarray(eff, dtype=float), 1e-30))
    cx = np.polyfit(np.log(E / 100.0), lne, 2)
    cy = np.polyfit(np.log(E / 1000.0), lne, 2)
    return [float(cx[2]), float(cx[1]), float(cy[2]), float(cy[1]), float(cy[0])]


def multistart(func, E, eff, deff, seeds, bounds=None, method="trf",
               maxfev=20000):
    """Fit from several starting points; keep the lowest weighted chi-squared.

    Multi-start, not tighter tolerances, is what reduces divergence between
    the two models. Candidates that fail to converge are skipped; None comes
    back only when every one of them failed.
    """
    _need_scipy()
    best_p, best_chi2 = None, np.inf
    for p0 in seeds:
        try:
            kwargs = dict(p0=p0, sigma=deff, absolute_sigma=True,
                          maxfev=maxfev, method=method)
            if bounds is not None:
                kwargs["bounds"] = bounds
            p, _ = curve_fit(func, E, eff, **kwargs)
            if not np.all(np.isfinite(p)):
                continue
            chi2 = float(np.sum(((eff - func(E, *p)) / deff) ** 2))
            if chi2 < best_chi2:
                best_chi2, best_p = chi2, p
        except Exception:
            pass
    return best_p


@dataclass
class EfficiencyFit:
    """The deterministic part of an efficiency calibration: the two best-fit
    curves and how well each describes the data. The Monte Carlo (Task 4)
    adds the uncertainty bands on top of this."""
    E: np.ndarray
    eff: np.ndarray
    deff: np.ndarray
    kfr_params: tuple
    kfr_chi2: float
    kfr_ndf: int
    kfr_rms: float
    kfr_birge: float
    rw_params: tuple = None
    rw_chi2: float = float("nan")
    rw_ndf: int = 0
    rw_rms: float = float("nan")
    rw_birge: float = 1.0


def efficiency_points(N, dN, I, dI):
    """eps = N/I and its 1-sigma error.

    The absolute scale of I does not matter and must not be "converted":
    eps is relative and is normalised again later, so a factor common to
    every line of one source divides straight back out. Our .sou intensities
    are on sou_io's 0-10000 scale, the reference's are percentages, and both
    give the same normalised curve.
    """
    N = np.asarray(N, dtype=float)
    dN = np.asarray(dN, dtype=float)
    I = np.asarray(I, dtype=float)
    dI = np.asarray(dI, dtype=float)
    eff = N / I
    deff = eff * np.sqrt((dN / N) ** 2 + (dI / I) ** 2)
    return eff, deff


def birge(chi2, ndf):
    """Birge ratio sqrt(chi2/ndf), or 1.0 when ndf <= 0.

    An exactly-determined fit carries no scatter information, so the stated
    sigma are used unscaled instead of dividing by zero.
    """
    return float(np.sqrt(chi2 / ndf)) if ndf > 0 else 1.0


#: KFR bounds from the reference: a, b >= 0 and c <= 0 keep the curve
#: physical; d is free.
KFR_BOUNDS = ([0.0, 0.0, -np.inf, -np.inf], [np.inf, np.inf, 0.0, np.inf])


def _kfr_seeds(E, eff):
    """The reference's five starting points: the data-driven seed, three
    perturbations of it, and the original fixed fallback."""
    s = kfr_seed(E, eff)
    return [
        s,
        [s[0] * 2.0, s[1] * 2.0, -1e-4, 1.0],
        [s[0] * 0.5, s[1] * 0.5, -5e-4, 0.5],
        [s[0], s[1], -1e-3, 2.0],
        [1.0, 1e3, -1e-3, 0.0],
    ]


def fit_efficiency(E, N, dN, I, dI):
    """Best-fit KFR and Radware curves for one set of peaks.

    Raises RuntimeError only if KFR fails from every starting point. A
    Radware failure is recorded as rw_params=None and reported to the user
    rather than raised: KFR alone is still a usable calibration.
    """
    _need_scipy()
    E = np.asarray(E, dtype=float)
    eff, deff = efficiency_points(N, dN, I, dI)

    kfr = multistart(f_kfr, E, eff, deff, _kfr_seeds(E, eff),
                     bounds=KFR_BOUNDS, method="trf")
    if kfr is None:
        raise RuntimeError(
            "The KFR efficiency fit did not converge from any starting "
            "point. Check that every peak has a positive area and every "
            "source line a positive intensity.")
    res_k = eff - f_kfr(E, *kfr)
    kfr_chi2 = float(np.sum((res_k / deff) ** 2))
    kfr_ndf = len(E) - 4

    # Radware: LM without bounds, parset seed first then the polyfit one.
    # Parameters beyond +-500 mean the polynomials have run away rather than
    # converged, which the reference also rejects.
    rw = None
    best = np.inf
    for seed in (radware_seed_5p(E, eff), radware_seed_polyfit_5p(E, eff)):
        try:
            pp, _ = curve_fit(f_radware_5p, E, eff, p0=seed, sigma=deff,
                              absolute_sigma=True, method="lm", maxfev=20000)
            if not (np.all(np.isfinite(pp)) and np.all(np.abs(pp) < 500)):
                continue
            chi2 = float(np.sum(((eff - f_radware_5p(E, *pp)) / deff) ** 2))
            if chi2 < best:
                best, rw = chi2, pp
        except Exception:
            pass

    out = EfficiencyFit(
        E=E, eff=eff, deff=deff,
        kfr_params=tuple(kfr), kfr_chi2=kfr_chi2, kfr_ndf=kfr_ndf,
        kfr_rms=float(np.sqrt(np.mean(res_k ** 2))),
        kfr_birge=birge(kfr_chi2, kfr_ndf),
    )
    if rw is not None:
        res_r = eff - f_radware_5p(E, *rw)
        out.rw_params = tuple(rw)
        out.rw_chi2 = float(np.sum((res_r / deff) ** 2))
        out.rw_ndf = len(E) - 5
        out.rw_rms = float(np.sqrt(np.mean(res_r ** 2)))
        out.rw_birge = birge(out.rw_chi2, out.rw_ndf)
    return out


#: Reference values (ra226_gui.py). N_BAND caps how many stored samples are
#: evaluated on the plotting grid -- evaluating all 10,000 is what makes a
#: redraw slow, and a percentile needs far fewer.
N_MC_EFFICIENCY = 10000
N_BAND = 4000
EFFICIENCY_SEED = 42
MC_MAXFEV = 1500
MC_TOL = 1e-5

#: Fewest finite MC samples that can support a reported value. Below this a
#: mean is an average over whichever handful happened not to diverge, which
#: is worse than reporting nothing.
MIN_MC_SAMPLES = 10


def finite_mean_std(values, min_n=1):
    """Mean and standard deviation over the finite entries, or (nan, nan)
    when fewer than min_n of them survive."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < min_n:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


def band_percentiles(band, lo=15.87, hi=84.13):
    """The 1-sigma envelope of an (n_samples, n_grid) family of curves."""
    with np.errstate(invalid="ignore"):
        return (np.nanpercentile(band, lo, axis=0),
                np.nanpercentile(band, hi, axis=0))


@dataclass
class EfficiencyMC:
    kfr_samples: np.ndarray
    rw_samples: np.ndarray
    kfr_accepted: int
    rw_accepted: int
    rejected: int

    def _band(self, grid, centre, samples, func, birge_factor):
        """Percentile envelope, Birge-scaled about `centre`.

        `centre` is normally the MC mean, because that is the curve being
        reported. The reference centres on its best fit instead; the two
        differ by far less than the band's own width, but centring on the
        curve actually drawn is what keeps the band symmetric about it.
        """
        if samples is None or len(samples) == 0:
            return centre, centre
        p = samples[:N_BAND]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            family = func(np.asarray(grid, dtype=float)[None, :],
                          *[p[:, i:i + 1] for i in range(p.shape[1])])
            lo, hi = band_percentiles(family)
        return (centre - birge_factor * (centre - lo),
                centre + birge_factor * (hi - centre))

    def kfr_band(self, grid, fit, centre=None):
        if centre is None:
            centre = f_kfr(np.asarray(grid, dtype=float), *fit.kfr_params)
        return self._band(grid, centre, self.kfr_samples, f_kfr, fit.kfr_birge)

    def rw_band(self, grid, fit, centre=None):
        if fit.rw_params is None:
            nan = np.full(len(np.asarray(grid)), float("nan"))
            return nan, nan
        if centre is None:
            centre = f_radware_5p(np.asarray(grid, dtype=float),
                                  *fit.rw_params)
        return self._band(grid, centre, self.rw_samples, f_radware_5p,
                          fit.rw_birge)


def run_monte_carlo(fit, N, dN, I, dI, iterations=N_MC_EFFICIENCY,
                    seed=EFFICIENCY_SEED, progress=None):
    """Resample N and I, refit both models, keep the parameter sets.

    Speed matters here: 10,000 iterations times two models has to finish in
    about a minute, which is why this uses LM rather than TRF, relaxes the
    tolerances to 1e-5 (sigma-level accuracy is all a band needs) and caps
    maxfev. All three are the reference's own choices.

    `progress(done, total)` is called every 100 iterations if given, and may
    return False to cancel.
    """
    _need_scipy()
    E = fit.E
    # The ORIGINAL deff, deliberately not recomputed from each resampled
    # N_s/I_s. The reference passes sigma=deff unchanged inside the loop.
    # Recomputing per sample looks like a correction and is not one: the fit
    # weights would then vary with the noise draw, which changes what the
    # spread of fitted parameters measures.
    deff = fit.deff
    rng = np.random.default_rng(seed)
    kfr_store, rw_store, rejected = [], [], 0

    for k in range(iterations):
        if progress is not None and k % 100 == 0:
            if progress(k, iterations) is False:
                break

        N_s = rng.normal(N, dN)
        I_s = rng.normal(I, dI)
        if (N_s <= 0).any() or (I_s <= 0).any():
            rejected += 1
            continue
        eff_s = N_s / I_s
        if not np.isfinite(eff_s).all() or (eff_s <= 0).any():
            rejected += 1
            continue

        try:
            pp, _ = curve_fit(f_kfr, E, eff_s, p0=fit.kfr_params, sigma=deff,
                              absolute_sigma=True, maxfev=MC_MAXFEV,
                              method="lm", ftol=MC_TOL, xtol=MC_TOL,
                              gtol=MC_TOL)
            if np.all(np.isfinite(pp)):
                kfr_store.append(pp)
        except Exception:
            pass

        if fit.rw_params is not None:
            # A fresh parset() seed per sample, not a rolling warm start:
            # effit.c calls parset() then fitter() for each new data set, and
            # chaining the previous result would bias the chain.
            try:
                pp_r, _ = curve_fit(f_radware_5p, E, eff_s,
                                    p0=radware_seed_5p(E, eff_s), sigma=deff,
                                    absolute_sigma=True, maxfev=MC_MAXFEV,
                                    method="lm", ftol=MC_TOL, xtol=MC_TOL,
                                    gtol=MC_TOL)
                if np.all(np.isfinite(pp_r)) and np.all(np.abs(pp_r) < 500):
                    rw_store.append(pp_r)
            except Exception:
                pass

    return EfficiencyMC(
        kfr_samples=(np.asarray(kfr_store, dtype=float) if kfr_store
                     else np.empty((0, 4))),
        rw_samples=(np.asarray(rw_store, dtype=float) if rw_store
                    else np.empty((0, 5))),
        kfr_accepted=len(kfr_store), rw_accepted=len(rw_store),
        rejected=rejected,
    )
