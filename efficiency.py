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
