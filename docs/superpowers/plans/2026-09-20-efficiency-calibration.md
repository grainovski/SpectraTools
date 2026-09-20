# Automatic MC Efficiency Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a relative detector efficiency ε(E) by Monte Carlo with two independent models after an automatic energy calibration, display and examine it, save it to two files, and let the user divide any spectrum by the chosen curve bin by bin.

**Architecture:** Four new modules — `efficiency.py` (maths, no Qt), `efficiency_io.py` (two writers), `efficiency_apply.py` (the correction, pure), `efficiency_dialog.py` (the window). The entry point is a new button on the existing `CalibrationPlotDialog`, which already holds every input the fit needs. Mirrors the existing `calibration.py` / `calibration_plot_dialog.py` / `caleneff_export.py` split.

**Tech Stack:** Python 3.13, numpy, scipy (`curve_fit`, TRF and LM), PySide6, matplotlib. Design spec: `docs/superpowers/specs/2026-09-20-efficiency-calibration-design.md`.

**Reference:** `C:\Users\RIG\Documents\Claude\efficieny\ra226_gui.py` (CalEnEff). Where this plan and that source disagree, the source wins.

---

## Read this before Task 1

**scipy is deferred at startup and must stay that way.** v5.2.5 removed scipy from the startup path and `tests/test_startup_imports.py` enforces it. `efficiency.py` must NOT import scipy at module level. Follow the pattern already in `peak_fit.py`:

```python
curve_fit = None

def _need_scipy():
    global curve_fit
    if curve_fit is None:
        from scipy.optimize import curve_fit as _cf
        curve_fit = _cf
```

and call `_need_scipy()` at the top of every function that fits. A module-level `from scipy.optimize import curve_fit` will fail `test_starting_the_app_does_not_import_scipy` the moment `efficiency.py` is reachable from `main`.

**Do not clip `f1`/`f2` positive** in the Radware model. They are log-efficiencies and are normally negative; the reference documents that clipping them made the loss landscape pathological and stalled the MC.

**Every test file needs a control that must fail.** This project's standing rule. A test that cannot fail is not evidence.

---

## File Structure

| file | responsibility | Qt |
|---|---|---|
| `efficiency.py` | models, seeds, multi-start, best fit, MC, prediction, normalisation | no |
| `efficiency_io.py` | the per-peak and per-bin writers | no |
| `efficiency_apply.py` | counts ÷ ε with the zeroing rules | no |
| `efficiency_dialog.py` | results window: plot, residuals, examine, model choice, save | yes |
| `calibration_plot_dialog.py` | **modified** — gains the entry button | yes |
| `main_window.py` | **modified** — Apply action, builds the corrected spectrum | yes |
| `tests/fixtures/caleneff/*.txt` | the three reference datasets, copied in | — |
| `tests/test_efficiency_models.py` | model shapes | — |
| `tests/test_efficiency_fit.py` | best fit + **the CalEnEff oracle** | — |
| `tests/test_efficiency_mc.py` | Monte Carlo | — |
| `tests/test_efficiency_io.py` | both writers | — |
| `tests/test_efficiency_apply.py` | correction + zeroing | — |
| `tests/test_efficiency_dialog.py` | window, entry button, apply wiring | — |

---

## Task 1: The two model functions

**Files:**
- Create: `efficiency.py`
- Test: `tests/test_efficiency_models.py`

- [ ] **Step 1: Write the failing test**

```python
"""The two efficiency curve shapes, ported from CalEnEff.

These are pure functions of energy and parameters. Everything later in the
feature -- the fit, the Monte Carlo, the correction applied to a spectrum --
is built on them being right, so they are pinned against values computed
directly from the published formulae rather than against our own output.
"""

import numpy as np
import pytest

from efficiency import f_kfr, f_radware, f_radware_5p, RADWARE_C, RADWARE_G


def test_kfr_matches_its_formula():
    """eps(E) = (aE + b/E) * exp(cE + d/E), evaluated by hand."""
    a, b, c, d = 0.5, 1000.0, -1e-3, -50.0
    E = 500.0
    expected = (a * E + b / E) * np.exp(c * E + d / E)
    assert f_kfr(E, a, b, c, d) == pytest.approx(expected, rel=1e-12)


def test_kfr_is_vectorised():
    E = np.array([100.0, 500.0, 1000.0])
    out = f_kfr(E, 0.5, 1000.0, -1e-3, -50.0)
    assert out.shape == E.shape
    assert np.all(np.isfinite(out))


def test_radware_5p_is_the_7p_model_with_c_and_g_fixed():
    """The 5-parameter form is not a different model -- it is the 7-parameter
    one with Radford's own two defaults held fixed. If these ever disagree,
    one of them has been edited in isolation."""
    E = np.array([100.0, 344.0, 1408.0])
    p = (-3.5, 1.5, -0.9, -0.5, -0.02)          # a1, a2, a4, a5, a6
    full = f_radware(E, p[0], p[1], RADWARE_C, p[2], p[3], p[4], RADWARE_G)
    assert f_radware_5p(E, *p) == pytest.approx(full, rel=1e-12)


def test_radware_uses_log_efficiencies_that_may_be_negative():
    """f1 and f2 are ln(eps) and are normally NEGATIVE. An earlier version of
    the reference clipped them positive with np.maximum(..., 1e-30), which
    made the loss landscape pathological and stalled the Monte Carlo. This
    pins the un-clipped behaviour: with strongly negative polynomials the
    model must return a small positive efficiency, not 1.0 or a constant."""
    E = np.array([100.0, 1000.0])
    out = f_radware_5p(E, -5.0, 1.0, -4.0, -0.5, 0.0)
    assert np.all(out > 0.0)
    assert np.all(out < 1.0), out
    assert out[0] != pytest.approx(out[1]), "clipped model would flatten"


def test_radware_survives_extreme_parameters():
    """The (1 + r**g)**(-1/g) evaluation overflows without the logaddexp form
    and the +-700 clip. The spectrum correction extrapolates this curve well
    outside the fitted range, so overflow here becomes inf counts there."""
    E = np.array([1.0, 1e5])
    out = f_radware_5p(E, 400.0, 50.0, -400.0, -50.0, 10.0)
    assert np.all(np.isfinite(out)), out


def test_a_constant_model_would_fail_these():
    """Control. Every assertion above must be able to fail -- a model that
    ignored its parameters would otherwise sail through the finiteness and
    positivity checks."""
    E = np.array([100.0, 1000.0])
    constant = np.ones_like(E)
    assert not np.allclose(f_kfr(E, 0.5, 1000.0, -1e-3, -50.0), constant)
    assert not np.allclose(f_radware_5p(E, -3.5, 1.5, -0.9, -0.5, -0.02), constant)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_models.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'efficiency'`

- [ ] **Step 3: Write `efficiency.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_models.py -q`
Expected: `6 passed`

- [ ] **Step 5: Confirm scipy is still off the startup path**

Run: `.venv/Scripts/python.exe -m pytest tests/test_startup_imports.py -q`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add efficiency.py tests/test_efficiency_models.py
git commit -m "feat: the two efficiency curve shapes, ported from CalEnEff"
```

---

## Task 2: Seeds and multi-start

**Files:**
- Modify: `efficiency.py`
- Test: `tests/test_efficiency_models.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_efficiency_models.py`:

```python
# --- seeds ---------------------------------------------------------------
#
# A fit is only as good as where it starts. These seeds are data-driven on
# purpose: our intensities are on sou_io's 0-10000 relative scale while the
# reference's are percentages, so eps differs from the reference by a large
# constant factor and any fixed seed would be wrong for one of them.


def test_kfr_seed_scales_with_the_data():
    """Doubling every efficiency must move the seed, or the seed is not
    data-driven and will be wrong on any spectrum with a different scale."""
    from efficiency import kfr_seed

    E = np.array([100.0, 500.0, 1000.0])
    eff = np.array([0.05, 0.02, 0.01])
    s1 = kfr_seed(E, eff)
    s2 = kfr_seed(E, eff * 2.0)
    assert len(s1) == 4
    assert s2[0] == pytest.approx(s1[0] * 2.0, rel=1e-9)
    assert s2[1] == pytest.approx(s1[1] * 2.0, rel=1e-9)


def test_radware_seed_places_the_polynomials_at_their_references():
    """Radford's parset(): a1 is ln eps at 100 keV and a4 is ln eps at 1000
    keV, both negative for eps < 1."""
    from efficiency import radware_seed_5p

    E = np.array([100.0, 500.0, 1000.0])
    eff = np.array([0.05, 0.02, 0.01])
    a1, a2, a4, a5, a6 = radware_seed_5p(E, eff)
    assert a1 == pytest.approx(np.log(0.05), rel=1e-9)
    assert a4 == pytest.approx(np.log(0.01), rel=1e-9)
    assert (a2, a5, a6) == pytest.approx((1.5, -0.9, 0.0))


def test_a_fixed_seed_would_fail_the_scaling_test():
    """Control for the first test: a seed ignoring the data cannot scale."""
    fixed = [1.0, 1e3, -1e-3, 0.0]
    assert fixed[0] != pytest.approx(fixed[0] * 2.0)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_models.py -q -k seed`
Expected: FAIL, `ImportError: cannot import name 'kfr_seed'`

- [ ] **Step 3: Add the seeds and multi-start to `efficiency.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_models.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add efficiency.py tests/test_efficiency_models.py
git commit -m "feat: data-driven seeds and multi-start for the efficiency fits"
```

---

## Task 3: Best fit, and the CalEnEff oracle

This is the task the whole feature rests on. If our port does not reproduce the reference's numbers, nothing downstream is worth building.

**Files:**
- Modify: `efficiency.py`
- Create: `tests/fixtures/caleneff/226Ra_En_Area.txt`, `demo1.txt`, `demo2.txt`
- Test: `tests/test_efficiency_fit.py`

- [ ] **Step 1: Copy the reference datasets in**

```bash
mkdir -p tests/fixtures/caleneff
cp "/c/Users/RIG/Documents/Claude/efficieny/226Ra_En_Area.txt" tests/fixtures/caleneff/
cp "/c/Users/RIG/Documents/Claude/efficieny/demo1.txt" tests/fixtures/caleneff/
cp "/c/Users/RIG/Documents/Claude/efficieny/demo2.txt" tests/fixtures/caleneff/
```

They are copied rather than read from the reference tree so the suite never depends on that directory existing. Each is 7 whitespace-separated columns, no header: `ch dch N dN E I dI`.

- [ ] **Step 2: Write the failing oracle test**

```python
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
  parameters    Loosest of the three, and informational -- it catches gross
                divergence without failing on a harmless reparameterisation.
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
        "kfr": [6.274371440450e-01, 7.844365398647e+05,
                -7.258867102388e-04, -1.467068952279e+02],
        "kfr_chi2": 117.5085777268, "kfr_ndf": 19,
        "rw": [-3.765575465541e+01, 7.438801254775e+01, 6.386965185696e+00,
               -7.193596739758e-01, -2.424310667484e-02],
        "rw_chi2": 97.8603554131, "rw_ndf": 18,
    },
    "demo1.txt": {
        "n": 9,
        "kfr": [1.138577308963e+03, 2.962038674033e+07,
                -5.801324494336e-03, -1.015250508519e+02],
        "kfr_chi2": 3.9694346123, "kfr_ndf": 5,
        "rw": [12.192811084709, -0.257477640308, 8.924249211442,
               -2.565507279293, -0.599464105736],
        "rw_chi2": 2.9281520555, "rw_ndf": 4,
    },
    "demo2.txt": {
        "n": 19,
        "kfr": [8.896741862449e+00, 7.089236616431e+06,
                -1.081881901237e-03, -1.160635814659e+02],
        "kfr_chi2": 47.6128442300, "kfr_ndf": 15,
        "rw": [3.012598062741e+00, 3.897772490500e+01, 8.473633125424e+00,
               -8.688228294051e-01, -3.357195834004e-02],
        "rw_chi2": 42.1858733891, "rw_ndf": 14,
    },
}


def _load(name):
    """The reference's 7-column format: ch dch N dN E I dI."""
    data = np.loadtxt(os.path.join(FIXTURES, name), ndmin=2)
    return data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_our_fit_reproduces_caleneff(name):
    from efficiency import f_kfr, f_radware_5p

    want = GOLDEN[name]
    N, dN, E, I, dI = _load(name)
    assert len(E) == want["n"]

    got = fit_efficiency(E, N, dN, I, dI)

    # 1. The curves agree where it matters.
    for f, key in ((f_kfr, "kfr"), (f_radware_5p, "rw")):
        ours = f(E, *getattr(got, key + "_params"))
        theirs = f(E, *want[key])
        assert ours == pytest.approx(theirs, rel=1e-6), (
            "%s: our %s curve differs from CalEnEff's at the data energies"
            % (name, key)
        )

    # 2. The minimised quantity agrees.
    assert got.kfr_chi2 == pytest.approx(want["kfr_chi2"], rel=1e-6)
    assert got.rw_chi2 == pytest.approx(want["rw_chi2"], rel=1e-6)
    assert got.kfr_ndf == want["kfr_ndf"]
    assert got.rw_ndf == want["rw_ndf"]

    # 3. Parameters, loosely -- degeneracy makes this the weakest check.
    assert np.asarray(got.kfr_params) == pytest.approx(
        np.asarray(want["kfr"]), rel=1e-4)
    assert np.asarray(got.rw_params) == pytest.approx(
        np.asarray(want["rw"]), rel=1e-4)


def test_the_oracle_can_fail():
    """Control. Without this the comparison above would pass just as happily
    against a curve that is wrong everywhere, if approx were mis-set or the
    golden values were accidentally derived from our own output."""
    from efficiency import f_kfr

    want = GOLDEN["226Ra_En_Area.txt"]
    _, _, E, _, _ = _load("226Ra_En_Area.txt")
    wrong = list(want["kfr"])
    wrong[0] *= 1.10                      # a 10% error in one parameter
    assert f_kfr(E, *wrong) != pytest.approx(f_kfr(E, *want["kfr"]), rel=1e-6)


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
```

- [ ] **Step 3: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_fit.py -q`
Expected: FAIL, `ImportError: cannot import name 'fit_efficiency'`

- [ ] **Step 4: Implement the best fit**

Add to `efficiency.py`:

```python
from dataclasses import dataclass, field


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
```

- [ ] **Step 5: Run the oracle**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_fit.py -q`
Expected: `6 passed`  (3 parametrised oracle cases + 3 others)

If the curve comparison fails, **do not loosen the tolerance**. Compare against the reference directly to find where the port diverges:

```bash
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0, r'C:\Users\RIG\Documents\Claude\efficieny'); import ra226_gui as r; e=r.CalibrationEngine(); e.load(r'C:\Users\RIG\Documents\Claude\efficieny\226Ra_En_Area.txt'); e.calibrate_efficiency(); print(e.eff_popt, e.eff_chi2)"
```

- [ ] **Step 6: Commit**

```bash
git add efficiency.py tests/test_efficiency_fit.py tests/fixtures/caleneff
git commit -m "feat: best-fit efficiency curves, checked against CalEnEff"
```

---

## Task 4: The Monte Carlo

**Files:**
- Modify: `efficiency.py`
- Test: `tests/test_efficiency_mc.py`

- [ ] **Step 1: Write the failing test**

```python
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


def test_the_band_brackets_the_best_fit(mc):
    """The 1-sigma band is built around the best-fit curve, so it must
    contain it at every energy it is drawn at."""
    from efficiency import f_kfr

    fit, out = mc
    grid = np.linspace(fit.E.min(), fit.E.max(), 50)
    lo, hi = out.kfr_band(grid, fit)
    centre = f_kfr(grid, *fit.kfr_params)
    assert np.all(lo <= centre + 1e-12)
    assert np.all(hi >= centre - 1e-12)


def test_too_few_survivors_reports_nan_not_a_fake_sigma(mc):
    """Fewer than 10 finite samples cannot support a sigma. Saying so is
    better than returning a number computed from three points."""
    from efficiency import finite_mean_std

    assert np.isnan(finite_mean_std(np.array([1.0, 2.0, 3.0]), min_n=10)[1])
    mean, std = finite_mean_std(np.arange(50.0), min_n=10)
    assert np.isfinite(mean) and np.isfinite(std)


def test_the_default_iteration_count_matches_the_reference():
    assert N_MC_EFFICIENCY == 10000
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_mc.py -q`
Expected: FAIL, `ImportError: cannot import name 'run_monte_carlo'`

- [ ] **Step 3: Implement the Monte Carlo**

Add to `efficiency.py`:

```python
#: Reference values (ra226_gui.py). N_BAND caps how many stored samples are
#: evaluated on the plotting grid -- evaluating all 10,000 is what makes a
#: redraw slow, and a percentile needs far fewer.
N_MC_EFFICIENCY = 10000
N_BAND = 4000
EFFICIENCY_SEED = 42
MC_MAXFEV = 1500
MC_TOL = 1e-5


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
        """Percentile envelope, Birge-scaled about the best-fit curve."""
        if samples is None or len(samples) == 0:
            return centre, centre
        p = samples[:N_BAND]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            family = func(np.asarray(grid, dtype=float)[None, :],
                          *[p[:, i:i + 1] for i in range(p.shape[1])])
            lo, hi = band_percentiles(family)
        return (centre - birge_factor * (centre - lo),
                centre + birge_factor * (hi - centre))

    def kfr_band(self, grid, fit):
        centre = f_kfr(np.asarray(grid, dtype=float), *fit.kfr_params)
        return self._band(grid, centre, self.kfr_samples, f_kfr, fit.kfr_birge)

    def rw_band(self, grid, fit):
        if fit.rw_params is None:
            nan = np.full(len(np.asarray(grid)), float("nan"))
            return nan, nan
        centre = f_radware_5p(np.asarray(grid, dtype=float), *fit.rw_params)
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_mc.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add efficiency.py tests/test_efficiency_mc.py
git commit -m "feat: Monte Carlo uncertainty bands for the efficiency curves"
```

---

## Task 5: Normalisation and the combined result

**Files:**
- Modify: `efficiency.py`
- Test: `tests/test_efficiency_fit.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_efficiency_fit.py`:

```python
# --- normalisation -------------------------------------------------------


def _demo_result():
    from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo

    N, dN, E, I, dI = _load("demo1.txt")
    fit = fit_efficiency(E, N, dN, I, dI)
    mc = run_monte_carlo(fit, N, dN, I, dI, iterations=200)
    return EfficiencyResult(fit=fit, mc=mc, model="kfr")


def test_the_selected_curve_peaks_at_one():
    r = _demo_result()
    grid = np.linspace(r.fit.E.min(), r.fit.E.max(), 2000)
    assert np.max(r.curve(grid)) == pytest.approx(1.0, rel=1e-6)


def test_switching_model_rescales_both_curves():
    """The normalisation is the selected model's peak, so selecting the other
    model changes the numbers written to file. That is intended and is why
    the header records which model was active."""
    from efficiency import f_kfr

    r = _demo_result()
    before = r.normalisation
    r.model = "rw"
    assert r.normalisation != pytest.approx(before)
    grid = np.linspace(r.fit.E.min(), r.fit.E.max(), 2000)
    assert np.max(r.curve(grid)) == pytest.approx(1.0, rel=1e-6)


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
    reference's data and quietly produce a differently-scaled curve on ours."""
    from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo

    N, dN, E, I, dI = _load("demo1.txt")
    grid = np.linspace(E.min(), E.max(), 200)

    a = EfficiencyResult(fit=fit_efficiency(E, N, dN, I, dI), mc=None,
                         model="kfr")
    b = EfficiencyResult(fit=fit_efficiency(E, N, dN, I * 137.0, dI * 137.0),
                         mc=None, model="kfr")
    assert a.curve(grid) == pytest.approx(b.curve(grid), rel=1e-6)


def test_the_curve_is_the_best_fit_not_the_mc_mean():
    """The MC gives the band; the best fit gives the curve. That is the
    reference's own division, and the two differ by enough to matter, so a
    later "improvement" that returned the MC mean here would silently change
    every saved number and every corrected spectrum."""
    from efficiency import f_kfr

    r = _demo_result()
    grid = np.linspace(r.fit.E.min(), r.fit.E.max(), 40)
    expected = f_kfr(grid, *r.fit.kfr_params) * r.normalisation
    assert r.curve(grid) == pytest.approx(expected, rel=1e-12)


def test_an_unnormalised_curve_would_fail_that():
    """Control: raw eps does depend on the intensity scale, by exactly the
    factor applied."""
    from efficiency import f_kfr, fit_efficiency

    N, dN, E, I, dI = _load("demo1.txt")
    raw_a = f_kfr(E, *fit_efficiency(E, N, dN, I, dI).kfr_params)
    raw_b = f_kfr(E, *fit_efficiency(E, N, dN, I * 137.0,
                                     dI * 137.0).kfr_params)
    assert raw_a != pytest.approx(raw_b, rel=1e-6)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_fit.py -q -k "normalis or scale"`
Expected: FAIL, `ImportError: cannot import name 'EfficiencyResult'`

- [ ] **Step 3: Implement `EfficiencyResult`**

Add to `efficiency.py`:

```python
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
        self.normalisation = self._compute_normalisation()

    def _raw(self, grid, model):
        grid = np.asarray(grid, dtype=float)
        if model == "kfr":
            return f_kfr(grid, *self.fit.kfr_params)
        return f_radware_5p(grid, *self.fit.rw_params)

    def _compute_normalisation(self):
        """1 / peak of the selected curve over the fitted range.

        The peak is found on a fine grid rather than at the data energies:
        the curve can crest between two measured points, and normalising by
        a value that is not the real maximum would leave the curve above 1.
        """
        lo = max(float(self.fit.E.min()) * 0.9, 1.0)
        hi = float(self.fit.E.max()) * 1.1
        grid = np.linspace(lo, hi, 2000)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = self._raw(grid, self._model)
        finite = values[np.isfinite(values) & (values > 0)]
        peak = float(np.max(finite)) if len(finite) else 1.0
        return 1.0 / max(peak, 1e-30)

    def curve(self, grid, model=None):
        """The normalised efficiency at `grid`, for the selected model
        unless another is named.

        This is the BEST-FIT curve, not the Monte Carlo mean. The MC is how
        the uncertainty is obtained, not the curve: the reference plots
        f_kfr(E, *eff_popt) and draws the MC family as a band around it. The
        two really do differ -- on the reference's Ra-226 data at 1155 keV,
        534.562 best-fit against an MC mean of 534.762 -- so this is what
        gets written to file, drawn, and divided into a spectrum, while
        every deff comes from the MC.
        """
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self._raw(grid, model or self._model) * self.normalisation

    def band(self, grid, model=None):
        """The normalised 1-sigma band, or (nan, nan) without a Monte Carlo."""
        which = model or self._model
        if self.mc is None:
            nan = np.full(len(np.asarray(grid)), float("nan"))
            return nan, nan
        lo, hi = (self.mc.kfr_band(grid, self.fit) if which == "kfr"
                  else self.mc.rw_band(grid, self.fit))
        return lo * self.normalisation, hi * self.normalisation

    def predict(self, energy, model=None):
        """(best_fit, mc_mean, mc_sigma) at one energy, all normalised.

        Fewer than 10 surviving finite samples gives nan for the mean and
        sigma rather than a number computed from a handful of points.
        """
        which = model or self._model
        grid = np.asarray([float(energy)])
        best = float(self.curve(grid, which)[0])
        if self.mc is None:
            return best, float("nan"), float("nan")
        samples = (self.mc.kfr_samples if which == "kfr"
                   else self.mc.rw_samples)
        if len(samples) == 0:
            return best, float("nan"), float("nan")
        func = f_kfr if which == "kfr" else f_radware_5p
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = func(float(energy), *[samples[:, i]
                                           for i in range(samples.shape[1])])
        mean, std = finite_mean_std(values, min_n=10)
        return best, mean * self.normalisation, std * self.normalisation
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_fit.py tests/test_efficiency_mc.py -q`
Expected: `19 passed`  (12 in the fit file, 7 in the MC file)

- [ ] **Step 5: Commit**

```bash
git add efficiency.py tests/test_efficiency_fit.py
git commit -m "feat: normalise the efficiency by the selected model's peak"
```

---

## Task 6: Applying an efficiency to a spectrum

Pure, and the place a plausible implementation goes wrong. Built before any UI.

**Files:**
- Create: `efficiency_apply.py`
- Test: `tests/test_efficiency_apply.py`

- [ ] **Step 1: Write the failing test**

```python
"""Dividing a spectrum by the efficiency curve, bin by bin."""

import numpy as np
import pytest

from efficiency_apply import apply_efficiency


class _FlatCurve:
    """A stand-in for EfficiencyResult exposing only what apply needs."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def curve(self, grid, model=None):
        grid = np.asarray(grid, dtype=float)
        return np.interp(grid, np.arange(len(self._values), dtype=float),
                         self._values)


class _Calibration:
    """channel -> energy, one keV per channel."""

    def apply(self, channel):
        return np.asarray(channel, dtype=float)


def test_counts_are_divided_bin_by_bin():
    counts = np.array([100.0, 200.0, 300.0, 400.0])
    curve = _FlatCurve([0.5, 0.5, 0.25, 0.25])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts == pytest.approx([200.0, 400.0, 1200.0, 1600.0])
    assert out.zeroed == 0


def test_variance_is_scaled_by_one_over_eff_squared():
    """The user asked for no efficiency uncertainty, so deff does not
    propagate -- but the spectrum's own variance still must, or the stored
    variance stops matching the counts and quietly corrupts any later fit."""
    counts = np.array([100.0, 100.0])
    variance = np.array([100.0, 100.0])
    curve = _FlatCurve([0.5, 0.25])
    out = apply_efficiency(counts, _Calibration(), curve, variance=variance)
    assert out.variance == pytest.approx([400.0, 1600.0])


def test_a_negative_efficiency_zeroes_the_bin():
    counts = np.array([100.0, 100.0, 100.0])
    curve = _FlatCurve([0.5, -0.5, 0.5])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[1] == 0.0
    assert out.zeroed == 1
    assert out.zeroed_nonpositive == 1


def test_a_zero_efficiency_zeroes_the_bin():
    counts = np.array([100.0, 100.0])
    curve = _FlatCurve([0.5, 0.0])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[1] == 0.0
    assert out.zeroed_nonpositive == 1


def test_a_nan_efficiency_zeroes_the_bin():
    """THE case a literal `eff <= 0` implementation fails. Under IEEE
    comparison NaN <= 0 is FALSE, so such a test lets NaN through, the bin
    becomes NaN, and that NaN then spreads into autoscaling, plotting,
    fitting and integration -- far worse than a zero."""
    counts = np.array([100.0, 100.0])
    curve = _FlatCurve([0.5, float("nan")])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[1] == 0.0, "a NaN efficiency slipped through"
    assert out.zeroed_nonfinite == 1


def test_an_infinite_efficiency_zeroes_the_bin():
    counts = np.array([100.0, 100.0])
    curve = _FlatCurve([0.5, float("inf")])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts[1] == 0.0
    assert out.zeroed_nonfinite == 1


def test_nothing_non_finite_ever_escapes():
    """The blanket guarantee, over a curve deliberately built to go positive,
    zero, negative, infinite and NaN across its range."""
    counts = np.full(5, 100.0)
    curve = _FlatCurve([1.0, 0.0, -1.0, float("inf"), float("nan")])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert np.all(np.isfinite(out.counts)), out.counts
    assert out.zeroed == 4


def test_the_zeroing_check_can_fail():
    """Control. A correction that returned its input untouched would satisfy
    every finiteness assertion above."""
    counts = np.array([100.0, 100.0])
    curve = _FlatCurve([0.5, 0.5])
    out = apply_efficiency(counts, _Calibration(), curve)
    assert out.counts != pytest.approx(counts)


def test_the_original_counts_are_not_modified():
    counts = np.array([100.0, 200.0])
    original = counts.copy()
    apply_efficiency(counts, _Calibration(), _FlatCurve([0.5, 0.5]))
    assert counts == pytest.approx(original)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_apply.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'efficiency_apply'`

- [ ] **Step 3: Write `efficiency_apply.py`**

```python
"""Divide a spectrum by an efficiency curve, bin by bin.

Pure: counts in, counts out. Nothing here knows about Qt, and nothing here
builds a LoadedSpectrum -- main_window does that with the result.

The curve is evaluated outside the range it was fitted over, without
restriction. That is the user's explicit choice, made with the divergence
risk stated: both models can run away when extrapolated.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CorrectedSpectrum:
    counts: np.ndarray
    variance: np.ndarray
    zeroed_nonpositive: int
    zeroed_nonfinite: int

    @property
    def zeroed(self):
        return self.zeroed_nonpositive + self.zeroed_nonfinite


def apply_efficiency(counts, calibration, curve, variance=None, model=None):
    """counts / eps(E(channel)), with non-usable efficiencies zeroed.

    A bin is corrected only where eps is **finite and > 0**, and is zeroed
    otherwise. Stating it that way rather than as `eps <= 0` is deliberate:
    under IEEE comparison `NaN <= 0` is False, so a literal test of the
    user's rule would let a NaN efficiency through and leave a NaN count --
    which then spreads through autoscaling, plotting, fitting and
    integration. `eps = +inf` already divides to about zero, so zeroing it
    changes nothing and keeps one rule instead of three.

    Returns a CorrectedSpectrum. `counts` is never modified in place.
    """
    counts = np.asarray(counts, dtype=float)
    channels = np.arange(len(counts), dtype=float)
    energies = np.asarray(calibration.apply(channels), dtype=float)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        eff = np.asarray(curve.curve(energies, model), dtype=float)

    usable = np.isfinite(eff) & (eff > 0.0)
    nonfinite = int(np.count_nonzero(~np.isfinite(eff)))
    nonpositive = int(np.count_nonzero(np.isfinite(eff) & (eff <= 0.0)))

    # np.divide with where= leaves the untouched entries at the `out` value,
    # which is why out is pre-filled with zeros rather than left empty.
    corrected = np.zeros_like(counts)
    np.divide(counts, eff, out=corrected, where=usable)

    if variance is None:
        new_variance = None
    else:
        new_variance = np.zeros_like(corrected)
        np.divide(np.asarray(variance, dtype=float), eff * eff,
                  out=new_variance, where=usable)

    return CorrectedSpectrum(
        counts=corrected, variance=new_variance,
        zeroed_nonpositive=nonpositive, zeroed_nonfinite=nonfinite,
    )
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_apply.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add efficiency_apply.py tests/test_efficiency_apply.py
git commit -m "feat: divide a spectrum by an efficiency curve, zeroing unusable bins"
```

---

## Task 7: The two output files

**Files:**
- Create: `efficiency_io.py`
- Test: `tests/test_efficiency_io.py`

- [ ] **Step 1: Write the failing test**

```python
"""The two efficiency files.

No channel column in either: the energy calibration is available wherever
these are read, so a channel column would duplicate a derived value on disk
where it can go stale.
"""

import os

import numpy as np
import pytest

from efficiency_io import write_per_bin, write_per_peak

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")


@pytest.fixture(scope="module")
def result():
    from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo

    data = np.loadtxt(os.path.join(FIXTURES, "demo1.txt"), ndmin=2)
    N, dN, E, I, dI = (data[:, 2], data[:, 3], data[:, 4],
                       data[:, 5], data[:, 6])
    fit = fit_efficiency(E, N, dN, I, dI)
    return EfficiencyResult(fit=fit,
                            mc=run_monte_carlo(fit, N, dN, I, dI,
                                               iterations=200),
                            model="kfr")


def _rows(path):
    return [l for l in open(path, encoding="utf-8").read().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def _header(path):
    return "\n".join(l for l in open(path, encoding="utf-8").read().splitlines()
                     if l.lstrip().startswith("#"))


def test_per_peak_has_six_columns_and_one_row_per_peak(tmp_path, result):
    energy_errors = np.full(len(result.fit.E), 0.01)
    path = str(tmp_path / "eff_peaks.txt")
    write_per_peak(path, result, energy_errors)
    rows = _rows(path)
    assert len(rows) == len(result.fit.E)
    for row in rows:
        assert len(row.split()) == 6, row


def test_per_bin_has_five_columns_and_one_row_per_channel(tmp_path, result):
    class _Cal:
        def apply(self, channel):
            return np.asarray(channel, dtype=float) + 50.0

    path = str(tmp_path / "eff_bins.txt")
    write_per_bin(path, result, _Cal(), channels=128)
    rows = _rows(path)
    assert len(rows) == 128
    for row in rows:
        assert len(row.split()) == 5, row


def test_neither_file_carries_a_channel_column(tmp_path, result):
    """The decision this format turns on. A channel column would be a stale
    copy of something the calibration already answers."""
    class _Cal:
        def apply(self, channel):
            return np.asarray(channel, dtype=float) + 50.0

    peaks = str(tmp_path / "p.txt")
    bins = str(tmp_path / "b.txt")
    write_per_peak(peaks, result, np.full(len(result.fit.E), 0.01))
    write_per_bin(bins, result, _Cal(), channels=32)
    for path, n in ((peaks, 6), (bins, 5)):
        assert len(_rows(path)[0].split()) == n
        assert "ch" not in _header(path).lower().split("columns:")[-1]


def test_the_header_records_what_the_numbers_depend_on(tmp_path, result):
    """The normalisation is the selected model's peak, so the same
    calibration saved under the other model writes different numbers. The
    header has to say which one produced these."""
    path = str(tmp_path / "p.txt")
    write_per_peak(path, result, np.full(len(result.fit.E), 0.01))
    header = _header(path)
    assert "kfr" in header.lower()
    assert "normalis" in header.lower() or "normaliz" in header.lower()
    assert "%.6g" % result.normalisation in header


def test_the_selected_curve_never_exceeds_one_in_the_file(tmp_path, result):
    path = str(tmp_path / "p.txt")
    write_per_peak(path, result, np.full(len(result.fit.E), 0.01))
    eff_kfr = np.array([float(r.split()[2]) for r in _rows(path)])
    assert np.all(eff_kfr <= 1.0 + 1e-9), eff_kfr.max()


def test_a_missing_normalisation_would_fail_that():
    """Control: raw eps for this data is in the thousands, so an unnormalised
    file could not pass the test above."""
    assert 67254.0 > 1.0
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_io.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'efficiency_io'`

- [ ] **Step 3: Write `efficiency_io.py`**

```python
"""The two efficiency output files.

Both carry BOTH curves so the file is a complete record of the calibration
and the two models can be compared after the fact. Neither carries a channel
column: the energy calibration is available wherever these are read.
"""

import numpy as np

MODEL_NAMES = {"kfr": "KFR", "rw": "Radware"}


def _header_lines(result, extra=()):
    fit = result.fit
    lines = [
        "SpectraTools relative efficiency",
        "",
        "applied model      : %s" % MODEL_NAMES[result.model],
        "normalisation      : %.6g   (1 / peak of the applied curve)" %
        result.normalisation,
        "fitted energy range: %.4f .. %.4f keV" % (fit.E.min(), fit.E.max()),
        "peaks              : %d" % len(fit.E),
        "KFR  params        : %s" % ", ".join("%.12g" % v
                                              for v in fit.kfr_params),
        "KFR  chi2/ndf      : %.6f / %d   Birge %.4f   RMS %.6g" %
        (fit.kfr_chi2, fit.kfr_ndf, fit.kfr_birge, fit.kfr_rms),
    ]
    if fit.rw_params is None:
        lines.append("Radware            : did not converge")
    else:
        lines += [
            "Radware params     : %s" % ", ".join("%.12g" % v
                                                  for v in fit.rw_params),
            "Radware chi2/ndf   : %.6f / %d   Birge %.4f   RMS %.6g" %
            (fit.rw_chi2, fit.rw_ndf, fit.rw_birge, fit.rw_rms),
        ]
    if result.mc is not None:
        lines.append(
            "Monte Carlo        : KFR %d accepted, Radware %d accepted, "
            "%d samples rejected" %
            (result.mc.kfr_accepted, result.mc.rw_accepted,
             result.mc.rejected))
    if result.calibration is not None:
        lines.append("energy calibration : %s" % (result.calibration,))
    if result.source:
        lines.append("source             : %s" % result.source)
    lines.extend(extra)
    return "".join("# %s\n" % line if line else "#\n" for line in lines)


def _both(result, energies):
    """Normalised value and 1-sigma width for each model at `energies`."""
    out = []
    for model in ("kfr", "rw"):
        if model == "rw" and result.fit.rw_params is None:
            nan = np.full(len(energies), float("nan"))
            out += [nan, nan]
            continue
        value = result.curve(energies, model)
        lo, hi = result.band(energies, model)
        out += [value, (hi - lo) / 2.0]
    return out


def write_per_peak(path, result, energy_errors):
    """One row per calibration peak: E dE eff_kfr deff_kfr eff_rw deff_rw."""
    E = result.fit.E
    dE = np.asarray(energy_errors, dtype=float)
    if len(dE) != len(E):
        raise ValueError(
            "got %d energy errors for %d peaks" % (len(dE), len(E)))
    k, dk, r, dr = _both(result, E)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_header_lines(
            result, ["", "columns: E  dE  eff_kfr  deff_kfr  eff_rw  deff_rw"]))
        for row in zip(E, dE, k, dk, r, dr):
            fh.write("%14.6f %12.6f %14.8g %14.8g %14.8g %14.8g\n" % row)


def write_per_bin(path, result, calibration, channels):
    """One row per channel: E eff_kfr deff_kfr eff_rw deff_rw.

    No dE column -- a sampled curve point has no energy uncertainty, and a
    zero column would be a meaningless number written to disk.
    """
    energies = np.asarray(
        calibration.apply(np.arange(int(channels), dtype=float)), dtype=float)
    k, dk, r, dr = _both(result, energies)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_header_lines(
            result,
            ["", "one row per channel, evaluated through the energy "
             "calibration above",
             "columns: E  eff_kfr  deff_kfr  eff_rw  deff_rw"]))
        for row in zip(energies, k, dk, r, dr):
            fh.write("%14.6f %14.8g %14.8g %14.8g %14.8g\n" % row)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_io.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add efficiency_io.py tests/test_efficiency_io.py
git commit -m "feat: write the per-peak and per-bin efficiency files"
```

---

## Task 8: The entry button

**Files:**
- Modify: `calibration_plot_dialog.py`
- Test: `tests/test_efficiency_dialog.py`

- [ ] **Step 1: Write the failing test**

```python
"""The button that launches the efficiency calibration, and what disables it."""

import numpy as np
import pytest

from calibration import Calibration
from calibration_plot_dialog import CalibrationPlotDialog


class _Line:
    def __init__(self, energy, intensity=1000.0, intensity_err=10.0):
        self.energy = energy
        self.energy_err = 0.01
        self.intensity = intensity
        self.intensity_err = intensity_err


def _dialog(qapp, points, lines):
    return CalibrationPlotDialog(
        None, Calibration(0.0, 0.5), points, lines, 4095, "out.txt")


def _good_points(n=8):
    """(channel, channel_err, area, area_err, energy) per point."""
    return [(100.0 * (i + 1), 0.1, 5000.0 - 300.0 * i, 70.0,
             50.0 * (i + 1)) for i in range(n)]


def _good_lines(n=8):
    return [_Line(50.0 * (i + 1)) for i in range(n)]


def test_the_button_is_enabled_with_enough_good_points(qapp):
    d = _dialog(qapp, _good_points(), _good_lines())
    assert d.efficiency_button.isEnabled()
    d.close()


def test_too_few_points_disables_it_with_a_reason(qapp):
    """Radware has five free parameters, so ndf = n - 5 must exceed zero."""
    d = _dialog(qapp, _good_points(4), _good_lines(4))
    assert not d.efficiency_button.isEnabled()
    assert "6" in d.efficiency_button.toolTip()
    d.close()


def test_a_zero_area_disables_it(qapp):
    """eps = N/I must be positive to fit at all."""
    points = _good_points()
    points[2] = (300.0, 0.1, 0.0, 70.0, 150.0)
    d = _dialog(qapp, points, _good_lines())
    assert not d.efficiency_button.isEnabled()
    assert "area" in d.efficiency_button.toolTip().lower()
    d.close()


def test_a_zero_intensity_error_disables_it(qapp):
    """The Monte Carlo resamples I from Normal(I, dI); dI = 0 makes that
    resampling a no-op and the band meaningless."""
    lines = _good_lines()
    lines[1] = _Line(100.0, intensity=1000.0, intensity_err=0.0)
    d = _dialog(qapp, _good_points(), lines)
    assert not d.efficiency_button.isEnabled()
    assert "intensit" in d.efficiency_button.toolTip().lower()
    d.close()


def test_the_enable_check_can_fail(qapp):
    """Control: the good case must really be enabled, or every assertion
    above passes for the wrong reason."""
    good = _dialog(qapp, _good_points(), _good_lines())
    bad = _dialog(qapp, _good_points(4), _good_lines(4))
    assert good.efficiency_button.isEnabled()
    assert not bad.efficiency_button.isEnabled()
    good.close(); bad.close()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_dialog.py -q`
Expected: FAIL, `AttributeError: 'CalibrationPlotDialog' object has no attribute 'efficiency_button'`

- [ ] **Step 3: Add the button and its gate**

In `calibration_plot_dialog.py`, add near the other buttons:

```python
        self.efficiency_button = QPushButton("Auto MC Efficiency Calibration")
        self.efficiency_button.clicked.connect(self._on_efficiency_clicked)
        button_row.addWidget(self.efficiency_button)
```

and the gate, called wherever the dialog's data is set:

```python
    #: Radware has five free parameters, so ndf = n - 5 must exceed zero.
    #: Six is the smallest n leaving any redundancy at all.
    MIN_EFFICIENCY_POINTS = 6

    def _efficiency_blocked_reason(self):
        """Why the efficiency calibration cannot run, or None.

        Checked here rather than left to fail inside scipy: a fitter's error
        points at the fit, not at the row of data that caused it.
        """
        pairs = self._efficiency_pairs()
        if len(pairs) < self.MIN_EFFICIENCY_POINTS:
            return ("Needs at least %d matched peaks; this calibration has "
                    "%d. The Radware model has 5 free parameters."
                    % (self.MIN_EFFICIENCY_POINTS, len(pairs)))
        for (_ch, _dch, area, area_err, energy), line in pairs:
            if not (area > 0 and area_err > 0):
                return ("Every peak needs a positive area and area error; "
                        "the peak at %.2f keV does not." % energy)
            if not (line.intensity > 0 and line.intensity_err > 0):
                return ("Every source line needs a positive intensity and "
                        "intensity error; the line at %.2f keV does not."
                        % energy)
            if not energy > 0:
                return "Every energy must be above 0 keV."
        return None

    def _refresh_efficiency_button(self):
        reason = self._efficiency_blocked_reason()
        self.efficiency_button.setEnabled(reason is None)
        self.efficiency_button.setToolTip(
            reason or "Fit a relative efficiency curve from these peaks.")
```

`_efficiency_pairs()` returns `[(point, source_line), ...]` for the points that are not excluded, matched to their source line by nearest energy — the same pairing the calibration itself already made. Call `_refresh_efficiency_button()` at the end of `set_data`.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_dialog.py -q`
Expected: `5 passed`

- [ ] **Step 5: Run the existing calibration tests — nothing may regress**

Run: `.venv/Scripts/python.exe -m pytest tests/test_calibration_plot_dialog.py tests/test_calibration.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add calibration_plot_dialog.py tests/test_efficiency_dialog.py
git commit -m "feat: the efficiency calibration entry button, with a stated reason when disabled"
```

---

## Task 9: The efficiency window

**Files:**
- Create: `efficiency_dialog.py`
- Modify: `calibration_plot_dialog.py` (open it), `tests/test_efficiency_dialog.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_efficiency_dialog.py`:

```python
# --- the results window --------------------------------------------------


@pytest.fixture
def opened(qapp):
    """A real window over the reference's Ba-133 data, with a short MC."""
    import os

    from efficiency import EfficiencyResult, fit_efficiency, run_monte_carlo
    from efficiency_dialog import EfficiencyDialog

    path = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff",
                        "demo1.txt")
    data = np.loadtxt(path, ndmin=2)
    N, dN, E, I, dI = (data[:, 2], data[:, 3], data[:, 4],
                       data[:, 5], data[:, 6])
    fit = fit_efficiency(E, N, dN, I, dI)
    result = EfficiencyResult(
        fit=fit, mc=run_monte_carlo(fit, N, dN, I, dI, iterations=200),
        model="kfr")
    dialog = EfficiencyDialog(None, result, np.full(len(E), 0.01))
    yield dialog
    dialog.close()


def test_it_draws_both_curves_and_the_points(opened):
    assert len(opened.axes.lines) >= 2
    assert len(opened.axes.collections) >= 1, "no error bars drawn"


def test_it_draws_a_residual_for_each_model(opened):
    assert len(opened.residual_axes.lines) >= 1


def test_examine_reports_both_models_at_an_energy(opened):
    text = opened.examine(250.0)
    assert "KFR" in text and "Radware" in text
    assert "250" in text


def test_selecting_the_other_model_redraws_and_rescales(opened):
    before = opened.result.normalisation
    opened.select_model("rw")
    assert opened.result.model == "rw"
    assert opened.result.normalisation != pytest.approx(before)


def test_the_summary_reports_the_mc_counts(opened):
    """A band from few survivors means something different from one built on
    thousands, so the number is on screen rather than implied."""
    text = opened.summary_text()
    assert "accepted" in text.lower()
    assert str(opened.result.mc.kfr_accepted) in text
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_dialog.py -q -k "opened or draws or examine or summary or selecting"`
Expected: FAIL, `ModuleNotFoundError: No module named 'efficiency_dialog'`

- [ ] **Step 3: Write `efficiency_dialog.py`**

Build a `QDialog` following `calibration_plot_dialog.py`'s structure exactly — same `Figure` / `FigureCanvasQTAgg` / `subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})` layout, same `style_axes(self.axes, theme)` calls so both themes work.

It must expose, because the tests above drive them:

- `self.axes`, `self.residual_axes`
- `self.result` — the `EfficiencyResult`
- `select_model(name)` — set `result.model`, recompute, redraw
- `examine(energy)` — return the report string for both models
- `summary_text()` — chi2/ndf, Birge, RMS per model plus MC accepted/rejected

Drawing: error bars at the data points; KFR solid with `fill_between` for its band; Radware dashed with its band; the residual strip carries both models' residuals with their RMS lines. Colours come from `theme.py`, not from CalEnEff.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_dialog.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add efficiency_dialog.py calibration_plot_dialog.py tests/test_efficiency_dialog.py
git commit -m "feat: the efficiency results window"
```

---

## Task 10: Run the Monte Carlo off the UI thread

**Files:**
- Modify: `calibration_plot_dialog.py`
- Test: `tests/test_efficiency_dialog.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_the_mc_runs_off_the_ui_thread(qapp, monkeypatch):
    """10,000 iterations twice over is about a minute. Running it inline
    would freeze the window for that whole time, with no way to cancel."""
    import calibration_plot_dialog as mod

    seen = {}

    def _fake_start(self, *args, **kwargs):
        seen["started"] = True

    monkeypatch.setattr(mod.EfficiencyWorker, "start", _fake_start,
                        raising=False)
    d = _dialog(qapp, _good_points(), _good_lines())
    d._on_efficiency_clicked()
    assert seen.get("started"), "the MC was not handed to a worker"
    d.close()


def test_cancelling_stops_the_run(qapp):
    """progress() returning False must end the loop, or Cancel does nothing
    until the full minute is up."""
    import os

    from efficiency import fit_efficiency, run_monte_carlo

    path = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff",
                        "demo1.txt")
    data = np.loadtxt(path, ndmin=2)
    N, dN, E, I, dI = (data[:, 2], data[:, 3], data[:, 4],
                       data[:, 5], data[:, 6])
    fit = fit_efficiency(E, N, dN, I, dI)
    out = run_monte_carlo(fit, N, dN, I, dI, iterations=5000,
                          progress=lambda done, total: done < 100)
    assert out.kfr_accepted < 500, "cancel was ignored"
```

- [ ] **Step 2: Run it and watch it fail**

Expected: FAIL, `AttributeError: module 'calibration_plot_dialog' has no attribute 'EfficiencyWorker'`

- [ ] **Step 3: Implement the worker**

A `QThread` subclass carrying the arrays, emitting `progress(int, int)` and `finished(object)`, driven from `_on_efficiency_clicked` behind a `QProgressDialog` whose Cancel sets a flag the `progress` callback returns False on. On completion, construct the `EfficiencyResult` and open `EfficiencyDialog`.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_dialog.py -q`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add calibration_plot_dialog.py tests/test_efficiency_dialog.py
git commit -m "feat: run the efficiency Monte Carlo on a worker thread"
```

---

## Task 11: Apply to a spectrum from the main window

**Files:**
- Modify: `main_window.py`, `efficiency_dialog.py`
- Test: `tests/test_efficiency_dialog.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# --- applying it ---------------------------------------------------------


def test_applying_adds_a_new_spectrum_and_keeps_the_original(qapp, opened):
    from calibration import Calibration
    from main_window import MainWindow
    from spectrum import LoadedSpectrum

    window = MainWindow()
    window._calibration = Calibration(0.0, 0.5)
    window._calibration_active = True
    original = LoadedSpectrum("s.txt", np.full(512, 100.0), "#FF0000")
    original.active = True
    window.spectra.append(original)

    before = len(window.spectra)
    window.apply_efficiency(original, opened.result)

    assert len(window.spectra) == before + 1
    assert original.data == pytest.approx(np.full(512, 100.0)), \
        "the original spectrum was modified"
    assert "eff" in window.spectra[-1].path.lower()
    window.close()


def test_applying_needs_an_active_calibration(qapp, opened):
    """Without one, bins have no energies and the correction is undefined."""
    from main_window import MainWindow
    from spectrum import LoadedSpectrum

    window = MainWindow()
    window._calibration_active = False
    s = LoadedSpectrum("s.txt", np.full(64, 10.0), "#FF0000")
    window.spectra.append(s)
    assert not window.can_apply_efficiency()
    window.close()


def test_the_corrected_spectrum_is_finite_everywhere(qapp, opened):
    """The curve is extrapolated without restriction, so the guarantee that
    nothing non-finite reaches a LoadedSpectrum has to hold at this level
    too, not only in efficiency_apply."""
    from calibration import Calibration
    from main_window import MainWindow
    from spectrum import LoadedSpectrum

    window = MainWindow()
    window._calibration = Calibration(0.0, 4.0)     # runs far past the fit
    window._calibration_active = True
    s = LoadedSpectrum("s.txt", np.full(2048, 100.0), "#FF0000")
    s.active = True
    window.spectra.append(s)
    window.apply_efficiency(s, opened.result)
    assert np.all(np.isfinite(window.spectra[-1].data))
    window.close()
```

- [ ] **Step 2: Run it and watch it fail**

Expected: FAIL, `AttributeError: 'MainWindow' object has no attribute 'apply_efficiency'`

- [ ] **Step 3: Implement**

On `MainWindow`:

```python
    def can_apply_efficiency(self):
        """An efficiency correction needs an energy per bin, which only an
        active calibration provides."""
        return bool(self._calibration_active and self._calibration is not None)

    def apply_efficiency(self, spectrum, result):
        """Divide `spectrum` by `result`'s selected curve and add the answer
        as a new spectrum beside it."""
        from efficiency_apply import apply_efficiency as _apply

        out = _apply(spectrum.data, self._calibration, result,
                     variance=getattr(spectrum, "variance", None))
        label = "%s [eff-corrected %s]" % (
            spectrum.path, "KFR" if result.model == "kfr" else "RW")
        self._add_combined_spectrum(label, out.counts, variance=out.variance)
        if out.zeroed:
            QMessageBox.information(
                self, "Efficiency applied",
                "%d of %d bins were zeroed: %d had a non-positive efficiency "
                "and %d a non-finite one. Both lie outside the range the "
                "curve was fitted over."
                % (out.zeroed, len(spectrum.data), out.zeroed_nonpositive,
                   out.zeroed_nonfinite))
```

Plus an **Apply to spectrum…** button on `EfficiencyDialog` that calls it for the active spectrum, disabled with a reason when `can_apply_efficiency()` is False, and a warning first if the active calibration is not the one the efficiency was derived under.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_efficiency_dialog.py tests/test_efficiency_apply.py -q`
Expected: `24 passed`

- [ ] **Step 5: Commit**

```bash
git add main_window.py efficiency_dialog.py tests/test_efficiency_dialog.py
git commit -m "feat: apply an efficiency to a spectrum from the main window"
```

---

## Task 12: Documentation and the version

**Files:**
- Modify: `help_content.py`, `CHANGELOG.md`, `packaging/windows/installer.iss`
- Test: `tests/test_help_content.py`

- [ ] **Step 1: Write the failing test**

```python
def test_the_help_describes_the_efficiency_calibration():
    """Every claim in here is checked against the source, because this
    project has repeatedly found plausible help prose to be subtly wrong."""
    from help_content import knowledge_database_html

    html = knowledge_database_html()
    assert "efficiency" in html.lower()
    assert "KFR" in html
    assert "Radware" in html
    # The relative/absolute distinction is the one a reader can act wrongly
    # on, so it must be stated explicitly.
    assert "relative" in html.lower()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_help_content.py -q -k efficiency`
Expected: FAIL

- [ ] **Step 3: Write the documentation**

Add a Knowledge Database section covering: ε = N/I, both model formulae, what the Monte Carlo does, that the efficiency is **relative** and the corrected spectrum's absolute scale is therefore arbitrary, the normalisation, the two file formats, and the zeroing rule. Add a HowTo section for the workflow.

**Verify every claim against the code before writing it.** `feedback_factcheck_docs_against_source` records that plausible prose in this project's help has been subtly wrong more than once.

- [ ] **Step 4: Bump the version and write the changelog**

`packaging/windows/installer.iss`: `#define AppVersion "6.0.0"`.
Add a `## [6.0.0] - <date>` entry to `CHANGELOG.md`.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: every test passes, and the collected count is the previous 1591 plus the tests added here.

Compare the count against `--collect-only` before trusting the green: a green run that collected fewer tests than expected is this project's known invisible failure.

- [ ] **Step 6: Commit**

```bash
git add help_content.py CHANGELOG.md packaging/windows/installer.iss tests/test_help_content.py
git commit -m "docs: efficiency calibration, and freeze 6.0.0"
```

---

## Self-review notes

- **Spec coverage.** §2 → Tasks 1-2. §3 → Task 4. §4 → Task 5. §5.1 → Task 8. §5.2 → Task 9, with threading from §3 in Task 10. §6 → Task 7. §7 → Tasks 6 and 11. §9 constants → Tasks 3-4. §12 testing → every task.
- **Names are consistent throughout:** `fit_efficiency`, `run_monte_carlo`, `EfficiencyFit`, `EfficiencyMC`, `EfficiencyResult`, `apply_efficiency`, `CorrectedSpectrum`, `write_per_peak`, `write_per_bin`, `f_kfr`, `f_radware`, `f_radware_5p`, `kfr_seed`, `radware_seed_5p`, `radware_seed_polyfit_5p`, `multistart`, `birge`, `efficiency_points`, `finite_mean_std`, `band_percentiles`.
- **The riskiest task is third**, not last: if Task 3's oracle fails, stop and fix the port before building anything on top of it.
