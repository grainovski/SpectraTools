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

import warnings

import numpy as np

#: Radford's own defaults in effit.c (freepars[2] = 0, freepars[6] = 0),
#: held fixed to leave five free parameters. Fixing them is what makes the
#: Levenberg-Marquardt fit stable on typical HPGe data.
RADWARE_C = 0.0
RADWARE_G = 15.0

#: Geometric-mean targets the Radware fit is tried at, plus the data's own
#: scale. The best-scoring one wins.
#:
#: The Radware family is NOT closed under a constant factor on eps. Scaling
#: eps shifts ln eps by a constant, and while a1 and a4 can absorb that
#: shift, doing so changes r = f/F and therefore the blend
#: (1 + r**g)**(-1/g) -- so eps*k is described by a genuinely different set
#: of curve shapes. KFR has no such problem: it is linear in a and b, so a
#: factor passes straight through.
#:
#: That made the fitted curve depend on the arbitrary normalisation of the
#: intensity column, which sou_io records is not even consistent between
#: our own files -- most peak at 10000, co56.sou at 100000, na24.sou at
#: 1000. Measured on the reference data, a 100x change in that constant
#: moved the reported Radware curve by up to 12% at the bottom of the
#: range, while leaving KFR identical to 2e-8.
#:
#: The fix uses the identity that fitting eps*k and reporting f/k is
#: exactly fitting the curve f/k to the original data:
#:
#:     chi2(k) = sum(((eps*k - f)/(deps*k))**2) = sum(((eps - f/k)/deps)**2)
#:
#: so chi-squared is directly comparable across k, and searching over k
#: fits a strictly larger family that IS scale-closed. Scaling the input by
#: c then hits the same targets at k/c and reports the same shape, which is
#: what removes the dependence rather than hiding it.
#:
#: Every entry is a target for the geometric mean, never a multiplier of
#: the data as given. That is what makes the ladder scale-equivariant:
#: scaling the input scales the geometric mean with it, so eff*k comes out
#: identical and the same rung wins. An "and also try the data's own scale"
#: rung would undo exactly that -- it was measured reintroducing the full
#: 12% drift on demo2 -- so there is deliberately no such entry.
#:
#: The ladder is wide and fine because the landscape is not smooth in k: a
#: target of 0.1 scores chi-squared 553 on demo1 where 1.0 scores 1.64.
#: Breadth is what replaces the missing own-scale rung, and a bad target
#: simply loses, so it costs only fit time -- about 30 short
#: Levenberg-Marquardt runs on a fit that happens once.
#:
#: The half-decade rungs earn nothing on the three reference files: dropping
#: to decades alone gives identical chi-squared on all three and the same
#: scale-independence, for 0.41 s a fit instead of 0.64 s. They are kept
#: anyway, as insurance on data we have not seen -- the 553-against-1.64
#: step above is what a coarser ladder risks straddling, and 0.23 s sits
#: against the ~33 s the Monte Carlo spends on the same calibration. Trim
#: to decades if that ever stops being true; the measurement is the reason,
#: not the fear.
RW_SCALE_TARGETS = (1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 3e2,
                    1e3, 3e3, 1e4, 3e4, 1e5)

#: How much better a rung must score to displace the one already chosen.
#:
#: Without this the ladder does not actually settle anything. On demo2 nine
#: rungs tie at chi-squared 42.185873246 within 1.5e-13 of each other, and
#: which one comes out lowest is decided by floating-point noise in the
#: geometric mean -- noise that moves when the caller's intensities are on a
#: different scale. Those tied fits are not the same curve: they agree
#: where the data constrain them and differ by about 8% at the bottom of the
#: range, which is the known Radware flat direction, so letting noise pick
#: between them put that 8% straight back into the answer.
#:
#: Requiring a real margin makes the FIRST rung to reach the best score keep
#: it, and RW_SCALE_TARGETS is a fixed ascending tuple, so the choice is
#: reproducible. Measured headroom: rung-to-rung noise across input scales
#: stays under 1e-8 relative, while a genuinely better rung wins by 0.3 or
#: more (demo1 improves by 44%), so 1e-6 sits about 100x above the noise and
#: 5 orders below any real difference.
RW_SCALE_TIE = 1e-6


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
            # A seed that leads nowhere is expected, not exceptional:
            # curve_fit raises RuntimeError when it will not converge and
            # ValueError on input it cannot use, and trying several starting
            # points is the whole reason this function exists. Every seed
            # failing is not silent -- best_p stays None and fit_efficiency
            # turns that into a RuntimeError the user reads.
            #
            # Deliberately broad, with a known cost: a programming error
            # inside one of the seed functions would be swallowed here and
            # reported as "did not converge", which points at the user's data
            # instead of at this file. Narrowing to (RuntimeError, ValueError)
            # would surface that, at the price of crashing a whole
            # calibration on one awkward data set. Graceful degradation wins
            # while the seeds stay covered by tests; revisit if a real bug
            # ever hides here.
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
    #: The scale `rw_params` were fitted at: they describe `eff * rw_scale`,
    #: so the Radware curve on the data's own scale is
    #: `f_radware_5p(E, *rw_params) / rw_scale`. Always 1.0 for KFR, which
    #: needs no such factor -- see RW_SCALE_TARGETS for why Radware does.
    rw_scale: float = 1.0


def efficiency_points(N, dN, I, dI):
    """eps = N/I and its 1-sigma error.

    The absolute scale of I does not matter and must not be "converted":
    eps is relative and is normalised again later, so a factor common to
    every line of one source divides straight back out. Our .sou intensities
    are on sou_io's 0-10000 scale, the reference's are percentages, and both
    give the same normalised curve.

    That holds for KFR by construction -- it is linear in a and b, so a
    common factor passes straight through. It does NOT hold for Radware by
    construction, and until 2026-09-21 this docstring was simply wrong about
    it: the fitted curve moved by up to 12% when the intensity column was
    rescaled. It holds now only because fit_efficiency searches over the
    scale as well; see RW_SCALE_TARGETS.
    """
    N = np.asarray(N, dtype=float)
    dN = np.asarray(dN, dtype=float)
    I = np.asarray(I, dtype=float)
    dI = np.asarray(dI, dtype=float)
    eff = N / I
    deff = eff * np.sqrt((dN / N) ** 2 + (dI / I) ** 2)
    return eff, deff


def canonical_scale(eff):
    """Geometric mean of the POSITIVE efficiencies, or 1.0 if there are none.

    Both fits are done at a canonical scale and converted back, and this is
    the scale. The geometric mean is the natural choice because it is
    equivariant -- scaling every point scales it by the same factor, which
    is exactly what makes the fits independent of the intensity column's
    arbitrary normalisation.

    Non-positive points are EXCLUDED rather than clamped. A peak area can
    come out negative on a weak line after background subtraction, and
    clamping it to 1e-300 instead of dropping it drags the geometric mean
    with it: measured, one negative area among six points pulled the scale
    to 3.1e-50, so the fit was handed values around 1e+50. That is finite
    and positive, so no guard downstream would have caught it. The fit on
    such data is poor either way -- the point is that it should be poor for
    the honest reason, not because the scale was destroyed.
    """
    eff = np.asarray(eff, dtype=float)
    positive = eff[np.isfinite(eff) & (eff > 0.0)]
    if positive.size == 0:
        return 1.0
    scale = float(np.exp(np.mean(np.log(positive))))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


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

    # Fitted at a canonical scale and converted back exactly.
    #
    # KFR's SOLUTION is scale-invariant -- eps = (aE + b/E)*exp(cE + d/E) is
    # linear in a and b, so scaling eps just scales them -- but its SEARCH is
    # not. The seeds' a and b track the data's magnitude, _kfr_seeds' last
    # fallback is the fixed [1.0, 1e3, -1e-3, 0.0], and trf controls its
    # steps in parameter space, so how well the fit converges depends on how
    # big a and b happen to be. The intensity column sets that, and sou_io
    # records our own files disagreeing on it by 100x (most peak at 10000,
    # co56.sou at 100000, na24.sou at 1000).
    #
    # Measured over 150 synthetic data sets across those three scales: 8%
    # moved the reported curve by more than 0.1%, 6% by more than 1%, and
    # one by 54%. One case scored chi-squared 37.1 at one scale and 5438.2
    # at another -- the same data, a different unit for the intensities. The
    # three CalEnEff fixtures show none of it, which is why it stayed
    # invisible: they are all well conditioned.
    #
    # Unlike Radware (see RW_SCALE_TARGETS) no divisor has to be carried
    # afterwards. Fitting eps/g and multiplying a and b back by g reproduces
    # the identical curve, so this changes which minimum is FOUND and
    # nothing about what the parameters mean.
    kfr_geo = canonical_scale(eff)
    eff_n, deff_n = eff / kfr_geo, deff / kfr_geo
    kfr = multistart(f_kfr, E, eff_n, deff_n, _kfr_seeds(E, eff_n),
                     bounds=KFR_BOUNDS, method="trf")
    if kfr is None:
        raise RuntimeError(
            "The KFR efficiency fit did not converge from any starting "
            "point. Check that every peak has a positive area and every "
            "source line a positive intensity.")
    kfr = np.array([kfr[0] * kfr_geo, kfr[1] * kfr_geo, kfr[2], kfr[3]])
    res_k = eff - f_kfr(E, *kfr)
    kfr_chi2 = float(np.sum((res_k / deff) ** 2))
    kfr_ndf = len(E) - 4

    # Radware: LM without bounds, parset seed first then the polyfit one.
    # Parameters beyond +-500 mean the polynomials have run away rather than
    # converged, which the reference also rejects.
    # Tried at each scale in RW_SCALE_TARGETS, and at NO other -- every rung
    # is a target for the geometric mean, so the set is the same whatever
    # scale the caller's intensities happen to be on. Falling back to the
    # data's own scale when the geometric mean is unusable keeps a
    # degenerate input (all-zero, non-finite) fitting as it used to.
    rw = None
    rw_scale = 1.0
    best = np.inf
    best_norm = np.inf
    geo = canonical_scale(eff)
    scales = [t / geo for t in RW_SCALE_TARGETS]
    for k in scales:
        if not np.isfinite(k) or k <= 0.0:
            continue
        effk, deffk = eff * k, deff * k
        for seed in (radware_seed_5p(E, effk),
                     radware_seed_polyfit_5p(E, effk)):
            try:
                # The covariance is discarded, so scipy warning that it
                # could not be estimated says nothing about the parameters
                # we keep. Left unsuppressed it printed to stderr once per
                # rung, which the ladder made routine rather than rare.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pp, _ = curve_fit(f_radware_5p, E, effk, p0=seed,
                                      sigma=deffk, absolute_sigma=True,
                                      method="lm", maxfev=20000)
                if not (np.all(np.isfinite(pp))
                        and np.all(np.abs(pp) < 500)):
                    continue
                chi2 = float(np.sum(
                    ((effk - f_radware_5p(E, *pp)) / deffk) ** 2))
                # Clearly better on chi-squared wins. Among fits the data
                # cannot tell apart, the one with the smallest parameters
                # wins instead: on demo2 a dozen of them tie to 1e-13 while
                # their curves differ by 8% below the lowest point, so
                # something has to choose, and left to chi-squared alone the
                # choice falls to floating-point noise. The tamest
                # polynomials are the least extrapolated guess and, unlike
                # the noise, are the same answer every run.
                norm = float(np.max(np.abs(pp)))
                if chi2 < best * (1.0 - RW_SCALE_TIE):
                    best, best_norm, rw, rw_scale = chi2, norm, pp, k
                elif (chi2 < best * (1.0 + RW_SCALE_TIE)
                      and norm < best_norm):
                    best, best_norm, rw, rw_scale = min(chi2, best), norm, pp, k
            except Exception:
                # Same bargain as multistart above: a rung that will not
                # converge simply loses, which is the point of having
                # fifteen of them. All of them failing leaves rw as None,
                # which fit_efficiency records and reports rather than
                # raising -- KFR alone is still a usable calibration.
                pass

    out = EfficiencyFit(
        E=E, eff=eff, deff=deff,
        kfr_params=tuple(kfr), kfr_chi2=kfr_chi2, kfr_ndf=kfr_ndf,
        kfr_rms=float(np.sqrt(np.mean(res_k ** 2))),
        kfr_birge=birge(kfr_chi2, kfr_ndf),
    )
    if rw is not None:
        # On the data's own scale, so chi2, RMS and the residuals plotted
        # under the curve all stay in the units the caller passed in.
        res_r = eff - f_radware_5p(E, *rw) / rw_scale
        out.rw_params = tuple(rw)
        out.rw_scale = rw_scale
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
    """The 1-sigma envelope of an (n_samples, n_grid) family of curves.

    An energy where every sample diverged leaves an all-NaN column, and
    np.nanpercentile emits a RuntimeWarning rather than raising on those.
    That case is ordinary rather than exceptional -- channel 0 maps to zero
    or negative energy under most calibrations, which is exactly what the
    zeroing rule in efficiency_apply exists to handle -- so the warning is
    suppressed here rather than printed on every save. The NaN itself is
    still returned and still propagates.
    """
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanpercentile(band, lo, axis=0),
                np.nanpercentile(band, hi, axis=0))


@dataclass
class EfficiencyMC:
    kfr_samples: np.ndarray
    rw_samples: np.ndarray
    kfr_accepted: int
    rw_accepted: int
    rejected: int

    def _band(self, grid, centre, samples, func, birge_factor, scale=1.0):
        """Percentile envelope, Birge-INFLATED about `centre`.

        The scaling only ever widens: see the clamp below for why a Birge
        ratio under 1 must not narrow the band.

        `centre` is normally the MC mean, because that is the curve being
        reported. The reference centres on its best fit instead; the two
        differ by far less than the band's own width, but centring on the
        curve actually drawn is what keeps the band symmetric about it.

        An energy where fewer than MIN_MC_SAMPLES of the family are finite
        gets NaN, the same rule the MC mean and predict() already apply.
        Without it `band_percentiles` would quietly take its percentiles
        over however few samples survived -- three out of ten thousand
        draws a band that looks exactly like any other. The mean at such an
        energy is already NaN, so the band was the one place a number with
        nothing behind it could still be drawn and exported.
        """
        if samples is None or len(samples) == 0:
            return centre, centre
        p = samples[:N_BAND]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            family = func(np.asarray(grid, dtype=float)[None, :],
                          *[p[:, i:i + 1] for i in range(p.shape[1])]) / scale
            lo, hi = band_percentiles(family)
            enough = np.isfinite(family).sum(axis=0) >= MIN_MC_SAMPLES
        lo = np.where(enough, lo, np.nan)
        hi = np.where(enough, hi, np.nan)
        # Inflate-only, the PDG convention. B > 1 means the points scatter
        # by more than their stated errors, and widening the band by it is
        # the standard admission that those errors are too small. B < 1 says
        # the opposite -- the fit tracks the data BETTER than their errors
        # require -- and the honest reading of that is that the errors were
        # overstated, not that the curve is known more sharply than the
        # resampling found. Scaling down on it reports a precision nothing
        # measured.
        #
        # Reachable on real data, not a theoretical worry: demo1 gives
        # B = 0.891 (KFR) and 0.640 (Radware), so the band was drawn at 89%
        # and 64% of the Monte Carlo percentiles. The Radware figure got
        # worse when the scale ladder improved that fit -- a better fit
        # shrank the band further, which is exactly backwards.
        #
        # The reference scales unconditionally (ra226_gui.py:1553-1554 and
        # 1570-1571), so this is a deliberate parity break. fit.kfr_birge
        # and fit.rw_birge keep the TRUE ratio: B < 1 is a real diagnostic
        # about the input errors and stays visible in the dialog and the
        # export. It just no longer narrows the band.
        factor = max(1.0, float(birge_factor))
        return (centre - factor * (centre - lo),
                centre + factor * (hi - centre))

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
                                  *fit.rw_params) / fit.rw_scale
        return self._band(grid, centre, self.rw_samples, f_radware_5p,
                          fit.rw_birge, scale=fit.rw_scale)


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
            # Correct by design here, unlike the seed loops: a resampled
            # data set that will not fit is a draw the band cannot use, and
            # dropping it is the intended behaviour. It is not hidden --
            # the sample simply never reaches kfr_store, so kfr_accepted
            # falls and the dialog and the export both show the count. A
            # band built on 200 survivors reads differently from one built
            # on 9,900, which is exactly why that number is on screen.
            pass

        if fit.rw_params is not None:
            # A fresh parset() seed per sample, not a rolling warm start:
            # effit.c calls parset() then fitter() for each new data set, and
            # chaining the previous result would bias the chain.
            # At the scale the deterministic fit settled on, so every stored
            # sample describes eff*rw_scale like fit.rw_params does and the
            # band can be evaluated with the one divisor.
            eff_k, deff_k = eff_s * fit.rw_scale, deff * fit.rw_scale
            try:
                # Covariance discarded here too, and at 10,000 draws an
                # un-suppressed warning is 10,000 lines of stderr.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pp_r, _ = curve_fit(f_radware_5p, E, eff_k,
                                        p0=radware_seed_5p(E, eff_k),
                                        sigma=deff_k, absolute_sigma=True,
                                        maxfev=MC_MAXFEV, method="lm",
                                        ftol=MC_TOL, xtol=MC_TOL,
                                        gtol=MC_TOL)
                if np.all(np.isfinite(pp_r)) and np.all(np.abs(pp_r) < 500):
                    rw_store.append(pp_r)
            except Exception:
                # As above: a dropped draw is counted through rw_accepted,
                # not swallowed. Radware rejects more often than KFR because
                # of the |p| < 500 runaway guard, which is why the two counts
                # are reported separately rather than as one total.
                pass

    return EfficiencyMC(
        kfr_samples=(np.asarray(kfr_store, dtype=float) if kfr_store
                     else np.empty((0, 4))),
        rw_samples=(np.asarray(rw_store, dtype=float) if rw_store
                    else np.empty((0, 5))),
        kfr_accepted=len(kfr_store), rw_accepted=len(rw_store),
        rejected=rejected,
    )


class EfficiencyResult:
    """A finished efficiency calibration: both curves, their bands, and
    which one the user has selected.

    The selected model sets the normalisation -- its peak becomes exactly
    1.0 and BOTH curves are scaled by that one factor, so the other stays
    directly comparable and may legitimately exceed 1 where it runs higher.
    """

    def __init__(self, fit, mc, model="kfr", calibration=None, source=None):
        self.fit = fit
        self.mc = mc
        self.calibration = calibration
        self.source = source
        self._model = None
        #: Normalisation per model. It depends only on `fit` and `mc`,
        #: neither of which changes after construction, so it is computed
        #: once per model rather than on every switch.
        self._normalisation_cache = {}
        self.model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if value not in ("kfr", "rw"):
            raise ValueError("model must be 'kfr' or 'rw', not %r" % (value,))
        if value == "rw" and self.fit.rw_params is None:
            raise ValueError("the Radware fit did not converge for this data")
        self._model = value
        cached = self._normalisation_cache.get(value)
        if cached is None:
            cached = self._compute_normalisation(value)
            self._normalisation_cache[value] = cached
        self.normalisation = cached

    def _samples(self, model):
        if self.mc is None:
            return None
        return self.mc.kfr_samples if model == "kfr" else self.mc.rw_samples

    def best_fit_raw(self, grid, model):
        """The best-fit curve, un-normalised.

        Kept because the fit-quality statistics describe it and because
        examining an energy reports it beside the MC mean, as CalEnEff does.
        It is NOT what gets applied or saved -- see curve().
        """
        grid = np.asarray(grid, dtype=float)
        if model == "kfr":
            return f_kfr(grid, *self.fit.kfr_params)
        return f_radware_5p(grid, *self.fit.rw_params) / self.fit.rw_scale

    def _mc_mean_raw(self, grid, model):
        """Mean of the Monte Carlo family at each energy, un-normalised.

        This is THE efficiency: applied to a spectrum, written to both files
        and drawn.

        Evaluated in chunks over energy on purpose. A per-bin file covers
        16,384 channels and the MC holds 10,000 parameter sets, so the whole
        family at once is 1.6e8 doubles -- about 1.3 GB in a single
        allocation. Chunking holds it to a few MB at a time and costs
        nothing else.

        A bin where fewer than MIN_MC_SAMPLES of the family are finite gets
        NaN, not a mean over the survivors. Far outside the fitted range most
        samples diverge, and averaging the few that happened not to would be
        a number with nothing behind it. efficiency_apply then zeroes those
        bins, which is the right outcome.
        """
        grid = np.atleast_1d(np.asarray(grid, dtype=float))
        samples = self._samples(model)
        if samples is None or len(samples) == 0:
            return np.full(grid.shape, float("nan"))
        func = f_kfr if model == "kfr" else f_radware_5p
        scale = 1.0 if model == "kfr" else self.fit.rw_scale
        out = np.empty(grid.shape, dtype=float)
        step = max(1, 2000000 // max(len(samples), 1))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for start in range(0, len(grid), step):
                chunk = grid[start:start + step]
                family = func(chunk[None, :],
                              *[samples[:, i:i + 1]
                                for i in range(samples.shape[1])]) / scale
                finite = np.isfinite(family)
                n_ok = finite.sum(axis=0)
                total = np.where(finite, family, 0.0).sum(axis=0)
                out[start:start + step] = np.where(
                    n_ok >= MIN_MC_SAMPLES, total / np.maximum(n_ok, 1),
                    np.nan)
        return out

    def _compute_normalisation(self, model=None):
        """1 / peak of `model`'s MC MEAN over the fitted range.

        Evaluating the Monte Carlo mean on a 2,000-point grid across every
        stored sample costs about 1.2 s with a full 10,000-sample run, which
        is why the caller caches the answer: paying it again on every toggle
        between the two models made switching feel slow for no reason.

        It must be the MC mean and not the best fit, because the MC mean is
        the curve that gets applied: normalising by the other one would leave
        the applied curve slightly off 1 at its peak, and "always less than
        1" is the whole point of normalising.

        The grid runs 10% past the data on each side, as the reference does.
        The curve can crest between two measured points, or below the lowest
        one -- for KFR the peak falls outside the measured points on two of
        the three reference datasets -- so normalising over the data range
        alone would leave the curve above 1 just outside it.

        Falls back to the best fit only when there is no Monte Carlo at all,
        which happens in tests that construct a result with mc=None.
        """
        model = model or self._model
        lo = max(float(self.fit.E.min()) * 0.9, 1.0)
        hi = float(self.fit.E.max()) * 1.1
        grid = np.linspace(lo, hi, 2000)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = (self._mc_mean_raw(grid, model)
                      if self.mc is not None
                      else self.best_fit_raw(grid, model))
        finite = values[np.isfinite(values) & (values > 0)]
        peak = float(np.max(finite)) if len(finite) else 1.0
        return 1.0 / max(peak, 1e-30)

    def curve(self, grid, model=None):
        """The normalised efficiency at `grid` -- the **Monte Carlo mean**.

        This is what is applied to a spectrum, written to both files and
        drawn (user decision, 2026-09-20). It is deliberately NOT the
        best-fit curve, which is what the reference plots: the two differ by
        a small but real amount -- on the reference's Ra-226 data at 1155
        keV, 534.762 MC mean against 534.562 best fit -- so which one is
        returned here changes every saved number and every corrected
        spectrum.

        The best fit stays available through best_fit_raw and is reported
        beside the MC mean when examining an energy, as CalEnEff does. The
        fit-quality statistics (chi2, ndf, Birge, RMS) remain best-fit
        quantities: they describe the fit, not this curve, and the CalEnEff
        oracle in tests/test_efficiency_fit.py compares them.
        """
        which = model or self._model
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if self.mc is None:
                return self.best_fit_raw(grid, which) * self.normalisation
            return self._mc_mean_raw(grid, which) * self.normalisation

    def band(self, grid, model=None, centre=None):
        """The normalised 1-sigma band, or (nan, nan) without a Monte Carlo.

        `centre` is the raw (un-normalised) MC mean when the caller has
        already computed it. Recomputing it here doubles the cost of writing
        a per-bin file, which is 16,384 evaluations over 10,000 parameter
        sets.
        """
        which = model or self._model
        if self.mc is None:
            nan = np.full(len(np.asarray(grid)), float("nan"))
            return nan, nan
        if centre is None:
            centre = self._mc_mean_raw(grid, which)
        lo, hi = (self.mc.kfr_band(grid, self.fit, centre) if which == "kfr"
                  else self.mc.rw_band(grid, self.fit, centre))
        return lo * self.normalisation, hi * self.normalisation

    def predict(self, energy, model=None):
        """(mc_mean, mc_sigma, best_fit) at one energy, all normalised.

        The MC mean comes first because it is the reported efficiency; the
        best fit is returned beside it so the examine panel can show both,
        as CalEnEff's does.

        `best_fit` must come from best_fit_raw and NOT from curve(): curve()
        returns the MC mean now, so reading it here would report the same
        number twice and the best fit would never be shown at all.

        Fewer than MIN_MC_SAMPLES surviving finite samples gives nan for the
        mean and sigma rather than a number computed from a handful of
        points. The best fit is still returned in that case -- it does not
        depend on the Monte Carlo.
        """
        which = model or self._model
        grid = np.asarray([float(energy)])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            best = float(self.best_fit_raw(grid, which)[0]) * self.normalisation
        samples = self._samples(which)
        if samples is None or len(samples) == 0:
            return float("nan"), float("nan"), best
        func = f_kfr if which == "kfr" else f_radware_5p
        scale = 1.0 if which == "kfr" else self.fit.rw_scale
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = func(float(energy),
                          *[samples[:, i]
                            for i in range(samples.shape[1])]) / scale
        mean, std = finite_mean_std(values, min_n=MIN_MC_SAMPLES)
        return mean * self.normalisation, std * self.normalisation, best
