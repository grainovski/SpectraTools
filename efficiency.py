"""Relative detector efficiency: the curve shapes.

A port of CalEnEff's two models (C:\\Users\\RIG\\Documents\\Claude\\efficieny,
ra226_gui.py). The formulae are not ours and are not improved on here --
matching the reference exactly is what lets tests/test_efficiency_fit.py
check this port against it.

scipy is imported lazily throughout this module. v5.2.5 took scipy off the
application's startup path and tests/test_startup_imports.py enforces it; a
module-level scipy import here would put it straight back.
"""

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
