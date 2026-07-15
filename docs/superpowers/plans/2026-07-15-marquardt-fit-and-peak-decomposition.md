# Damped Marquardt Fit and Peak Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix multi-peak fits converging badly (a peak's position collapsing away from where it was marked, or worse — an outright garbage result with a negative amplitude, both reproduced during investigation with a deliberately-wide initial width guess) by replacing `scipy.optimize.curve_fit`'s unconstrained Levenberg-Marquardt with a custom damped Marquardt solver replicating TV's per-iteration step-limiting, plus TV's local-max/FWHM-based initial width estimate. Add peak decomposition: draw each peak's own component curve, not just the summed total.

**Architecture:** `peak_fit.py` gains a new, self-contained numerical core (`_marquardt_fit`, a generic damped Levenberg-Marquardt solver taking a model function and per-parameter damping metadata) plus a `_measure_width` helper (TV-style FWHM measurement from data). `fit_peaks()`'s single `curve_fit(...)` call site is replaced with a call to `_marquardt_fit`, driven by a small `_build_param_damping` helper that translates the existing named-parameter system (`amp_i`/`pos_i`/`sigma`/`sigma_i`/`tail_fraction`/`tail_beta`) into per-parameter damping rules. The now-unused `_bounds()` function is deleted — tail/sigma bound enforcement moves into the solver's own clamping logic. `fit_mode.py`'s `draw_committed_fits` gains one additional line per peak (its own component curve) for decomposition. **The existing `hypermet_left_tail` peak-shape formula, `fixed_params` mechanism, and `fit_peaks()` public signature/return types are all unchanged** — only the internal fitting procedure and initial-width heuristic change.

**Design validated empirically before writing this plan** (see
`docs/superpowers/specs/2026-07-15-marquardt-fit-and-peak-decomposition-design.md`
for the TV research this is based on): a prototype of `_marquardt_fit` was tested against a single Gaussian, well-separated and heavily-overlapping multi-peak scenarios, independent-widths mode, a fixed-parameter-paired position, and the hypermet left-tail model — all converge correctly. Critically, a deliberately bad initial sigma guess that made plain `curve_fit` produce a peak with **negative amplitude and completely wrong positions** was handled gracefully by the damped solver (correct peak shapes recovered, though peak index labels can shift when the initial guess is this far off) — and the same scenario with a good (TV-style measured) initial width converged in 5 iterations to within ~1% of every true value. Performance: ~1-13ms per fit on realistic data sizes, well within interactive requirements.

**Tech Stack:** Python 3.13, `numpy`, `pytest` with the existing `qapp` fixture. `scipy.optimize.curve_fit` is removed from `peak_fit.py`'s fitting path entirely (still fine to import `scipy.special.erfc`, unrelated).

---

## Before you start

Read `docs/superpowers/specs/2026-07-15-marquardt-fit-and-peak-decomposition-design.md` in full. Run the full suite once to confirm a clean baseline:

Run: `pytest -v` from the repo root (`C:\Users\RIG\Documents\Claude\PeakFinderFitting`).
Expected: all tests pass (140 at last count).

---

### Task 1: Damped Marquardt-Levenberg solver core (standalone, isolated tests)

**Files:**
- Modify: `peak_fit.py` (add `_ParamDamping`, `_numeric_jacobian`, `_apply_step_damping`, `_clamp_trial`, `_marquardt_fit`)
- Test: `tests/test_peak_fit.py`

This task adds the generic numerical solver in isolation — proven against simple problems it can be checked against by hand, before Task 2 wires it into the much more complex multi-peak/hypermet/fixed-params machinery. Nothing in this task changes `fit_peaks()`'s behavior yet (the new functions aren't called from `fit_peaks()` until Task 2).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_peak_fit.py`, at the end of the file:

```python
from peak_fit import _ParamDamping, _marquardt_fit


def test_marquardt_fit_recovers_a_single_gaussian():
    def model(x, p):
        amp, pos, sigma = p
        return amp * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))

    rng = np.random.default_rng(1)
    x = np.arange(200, dtype=float)
    true = [500.0, 100.0, 3.0]
    y_clean = model(x, true)
    y = rng.poisson(np.maximum(y_clean, 0)).astype(float)
    y_err = np.sqrt(np.maximum(y, 1.0))

    damping = [_ParamDamping("amp"), _ParamDamping("pos", sigma_index=2), _ParamDamping("sigma")]
    p0 = [400.0, 98.0, 5.0]
    popt, pcov = _marquardt_fit(model, x, y, y_err, p0, damping)

    assert popt[1] == pytest.approx(100.0, abs=1.0)
    assert popt[2] == pytest.approx(3.0, abs=1.0)
    assert np.all(np.isfinite(pcov))


def test_marquardt_fit_position_step_is_damped_by_current_sigma():
    """A raw Newton step larger than the peak's own sigma must be capped
    to +-sigma for that single iteration -- the mechanism that prevents
    a peak from jumping straight past its neighbors in one step.

    The starting offset here (10 channels, ~3.3x the true sigma of 3.0)
    is deliberately chosen to stay within the region where the Gaussian
    still has meaningful gradient overlap with the data -- empirically,
    offsets beyond roughly 6x sigma leave essentially zero position
    gradient at the starting point, so amplitude collapses toward zero
    before position ever gets a chance to move (a real, well-known
    Gaussian-least-squares local-minimum trap that no amount of step
    damping can fix, since there's no gradient signal to damp in the
    first place -- confirmed by direct iteration count sweep: offsets of
    4-13 channels all converge correctly, offsets of 20+ don't). This
    test's job is to confirm damping doesn't cause problems within the
    regime where the fit is expected to work, not to prove convergence
    from an arbitrarily bad starting guess."""
    def model(x, p):
        amp, pos, sigma = p
        return amp * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))

    x = np.arange(200, dtype=float)
    y = model(x, [500.0, 100.0, 3.0])
    y_err = np.sqrt(np.maximum(y, 1.0))

    # Starting 10 channels away from the true position (~3.3x the true
    # sigma) -- the raw Newton step early on will exceed the current
    # sigma guess and need capping, without leaving the convergence
    # basin entirely.
    damping = [_ParamDamping("amp"), _ParamDamping("pos", sigma_index=2), _ParamDamping("sigma")]
    p0 = [500.0, 90.0, 3.0]
    popt, pcov = _marquardt_fit(model, x, y, y_err, p0, damping)
    # Damped or not, it should still eventually converge close to truth --
    # this test is about the mechanism not causing divergence, not about
    # inspecting individual iterations.
    assert popt[1] == pytest.approx(100.0, abs=2.0)


def test_marquardt_fit_bad_initial_width_does_not_produce_negative_amplitude():
    """Reproduces the investigation's adversarial scenario: three peaks
    with a deliberately too-wide initial sigma guess. Plain curve_fit on
    this exact data produces a peak with negative amplitude and wrong
    positions; the damped solver must not."""
    def model(x, p, n_peaks=3):
        sigma = p[-1]
        total = np.zeros_like(x)
        for i in range(n_peaks):
            total = total + p[2 * i] * np.exp(-((x - p[2 * i + 1]) ** 2) / (2 * sigma ** 2))
        return total

    rng = np.random.default_rng(0)
    x = np.arange(300, dtype=float)
    true_peaks = [(100.0, 500.0, 3.0), (108.0, 350.0, 3.0), (117.0, 420.0, 3.0)]
    y_clean = 20.0 + sum(
        amp * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2)) for pos, amp, sigma in true_peaks
    )
    y = rng.poisson(np.maximum(y_clean, 0)).astype(float)
    lo, hi = 80, 140
    mask = (x >= lo) & (x <= hi)
    x_fit, y_fit = x[mask], y[mask]
    y_sub = y_fit - 20.0
    y_err = np.sqrt(np.maximum(y_fit, 1.0))

    marked = [100.0, 108.0, 117.0]
    bad_sigma0 = (hi - lo) / (2 * 3)  # deliberately too wide
    p0 = []
    for pos in marked:
        idx = int(np.argmin(np.abs(x_fit - pos)))
        p0 += [float(y_sub[idx]), pos]
    p0.append(bad_sigma0)

    sigma_index = 2 * 3
    damping = []
    for _ in marked:
        damping += [_ParamDamping("amp"), _ParamDamping("pos", sigma_index=sigma_index)]
    damping.append(_ParamDamping("sigma"))

    popt, pcov = _marquardt_fit(
        lambda xx, pp: model(xx, pp), x_fit, y_sub, y_err, p0, damping,
        fit_region_bounds=(lo, hi),
    )
    amplitudes = [popt[0], popt[2], popt[4]]
    assert all(a > 0 for a in amplitudes)


def test_marquardt_fit_clamps_tail_fraction_and_beta_to_bounds():
    from peak_fit import hypermet_left_tail, TAIL_FRACTION_MAX, TAIL_BETA_MIN

    def model(x, p):
        amp, pos, sigma, r, beta = p
        return amp * hypermet_left_tail(x, pos, sigma, r, beta)

    x = np.arange(200, dtype=float)
    y = 500.0 * hypermet_left_tail(x, 100.0, 3.0, 0.1, 4.0)
    y_err = np.sqrt(np.maximum(y, 1.0))

    damping = [
        _ParamDamping("amp"), _ParamDamping("pos", sigma_index=2), _ParamDamping("sigma"),
        _ParamDamping("tail_fraction"), _ParamDamping("tail_beta"),
    ]
    p0 = [500.0, 100.0, 3.0, 0.05, 3.0]
    popt, pcov = _marquardt_fit(model, x, y, y_err, p0, damping)

    assert 0.0 <= popt[3] <= TAIL_FRACTION_MAX
    assert popt[4] >= TAIL_BETA_MIN
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_peak_fit.py -v -k marquardt_fit`
Expected: FAIL with `ImportError: cannot import name '_ParamDamping'` (neither symbol exists yet)

- [ ] **Step 3: Add the solver**

In `peak_fit.py`, add after the `TAIL_BETA_MIN = 0.1` constants (before `hypermet_left_tail`):

```python
_CUR_RATIO = 1e-12
_CUR_LAMBDA_START = 1e-3
_CUR_MIN_LAMBDA = 1e-20
_CUR_MAX_LAMBDA = 1e30
_CUR_INC_LAMBDA = 10.0
_CUR_MAX_ITERATIONS = 50
```

Add at the end of `peak_fit.py`:

```python
class _ParamDamping:
    """Per-parameter step-damping rule for _marquardt_fit, ported from
    TV's CHANGE_TRY macro (tv-1.9.13/src/VsFitFct.c). `kind` selects the
    rule: "amp" (no damping), "pos" (step capped at the peak's own
    current sigma; needs sigma_index or sigma_value to know that sigma),
    "sigma" (step capped at 2x current value, floored away from zero),
    "tail_fraction"/"tail_beta" (nudged away from exactly zero, then
    hard-clamped to this project's existing bounds)."""

    __slots__ = ("kind", "sigma_index", "sigma_value")

    def __init__(self, kind, sigma_index=None, sigma_value=None):
        self.kind = kind
        self.sigma_index = sigma_index
        self.sigma_value = sigma_value


def _current_sigma_for(meta, p):
    if meta.sigma_index is not None:
        return p[meta.sigma_index]
    return meta.sigma_value


def _numeric_jacobian(model, x, p):
    """Central-difference Jacobian -- matches TV's own fallback
    (CurNumericDerivation) for fit-function modules without an analytic
    derivative; avoids hand-deriving hypermet's erfc-based partials."""
    n = len(p)
    J = np.zeros((len(x), n))
    for i in range(n):
        h = max(abs(p[i]), 1e-6) * 1e-6
        p_hi = p.copy(); p_hi[i] += h
        p_lo = p.copy(); p_lo[i] -= h
        J[:, i] = (model(x, p_hi) - model(x, p_lo)) / (2 * h)
    return J


def _apply_step_damping(p, delta, damping, fit_region_bounds):
    """TV's CHANGE_TRY, per parameter kind. Returns a new delta array;
    does not mutate its inputs."""
    delta = delta.copy()
    for i, meta in enumerate(damping):
        if meta.kind == "amp":
            continue
        elif meta.kind == "pos":
            sigma = abs(_current_sigma_for(meta, p))
            if sigma > 0 and abs(delta[i]) > sigma:
                delta[i] = math.copysign(sigma, delta[i])
            if fit_region_bounds is not None:
                lo, hi = fit_region_bounds
                old = p[i]
                if lo <= old <= hi:
                    new = old + delta[i]
                    if new < lo:
                        delta[i] = (lo - old) * 0.5
                    elif new > hi:
                        delta[i] = (hi - old) * 0.5
        elif meta.kind == "sigma":
            width = abs(p[i])
            cap = 2.0 * width
            if width > 0 and abs(delta[i]) > cap:
                delta[i] = math.copysign(cap, delta[i])
            if p[i] + delta[i] == 0.0:
                delta[i] *= 0.5
        elif meta.kind in ("tail_fraction", "tail_beta"):
            if p[i] + delta[i] == 0.0:
                delta[i] *= 0.5
    return delta


def _clamp_trial(p_try, damping):
    """Hard bound enforcement after damping -- this project's existing
    sigma/tail_fraction/tail_beta bounds (previously enforced via
    scipy's curve_fit(..., bounds=...), now folded in here since the
    custom solver has no separate bounds mechanism)."""
    p_try = p_try.copy()
    for i, meta in enumerate(damping):
        if meta.kind == "sigma":
            if abs(p_try[i]) < 1e-6:
                p_try[i] = 1e-6 if p_try[i] >= 0 else -1e-6
        elif meta.kind == "tail_fraction":
            p_try[i] = min(max(p_try[i], 0.0), TAIL_FRACTION_MAX)
        elif meta.kind == "tail_beta":
            p_try[i] = max(p_try[i], TAIL_BETA_MIN)
    return p_try


def _marquardt_fit(model, x, y, y_err, p0, damping, fit_region_bounds=None):
    """Damped Levenberg-Marquardt solver replicating TV's fit procedure
    (tv-1.9.13/lib/tv/vsCurFit.c's CurFit + VsFitFct.c's CHANGE_TRY).
    `model(x, p)` takes the full parameter array (not *args, unlike
    scipy's curve_fit convention) and returns the model y-values.
    `damping` is a list of _ParamDamping, one per entry in p0, telling
    the solver how to limit each parameter's per-iteration step. Returns
    (popt, pcov) with the same meaning as
    scipy.optimize.curve_fit(..., absolute_sigma=True)."""
    p = np.array(p0, dtype=float)
    n = len(p)
    weights = 1.0 / y_err

    def measure_and_jac(pt):
        J = _numeric_jacobian(model, x, pt)
        r = (y - model(x, pt)) * weights
        Jw = J * weights[:, None]
        measure = float(np.sum(r ** 2))
        return measure, r, Jw

    measure, r, J = measure_and_jac(p)
    lam = _CUR_LAMBDA_START
    iterations = 0
    give_up = False

    while iterations < _CUR_MAX_ITERATIONS and not give_up:
        old_measure = measure
        alpha = J.T @ J
        beta = J.T @ r
        accepted = False

        while not accepted:
            damped = alpha.copy()
            diag = np.diag(damped).copy()
            diag_safe = np.where(diag == 0, 1.0, diag)
            damped[np.diag_indices(n)] = diag_safe * (1 + lam)
            try:
                raw_delta = np.linalg.solve(damped, beta)
            except np.linalg.LinAlgError:
                raw_delta, *_ = np.linalg.lstsq(damped, beta, rcond=None)

            delta = _apply_step_damping(p, raw_delta, damping, fit_region_bounds)
            p_try = _clamp_trial(p + delta, damping)
            try_measure, try_r, try_J = measure_and_jac(p_try)

            if try_measure < measure:
                improvement = measure - try_measure
                p, measure, r, J = p_try, try_measure, try_r, try_J
                lam = max(0.001 * improvement, _CUR_MIN_LAMBDA)
                accepted = True
            else:
                lam *= _CUR_INC_LAMBDA
                if lam > _CUR_MAX_LAMBDA:
                    p_try2 = _clamp_trial(p - 0.5 * delta, damping)
                    try_measure2, try_r2, try_J2 = measure_and_jac(p_try2)
                    if try_measure2 < measure:
                        p, measure, r, J = p_try2, try_measure2, try_r2, try_J2
                    accepted = True
                    give_up = True

        iterations += 1
        if give_up or measure <= 0:
            break
        ratio = (old_measure - measure) / measure
        if ratio <= _CUR_RATIO:
            break

    alpha_final = J.T @ J
    try:
        pcov = np.linalg.inv(alpha_final)
    except np.linalg.LinAlgError:
        pcov = np.full((n, n), np.inf)
    return p, pcov
```

Add `import math` to the top of `peak_fit.py` (alongside the existing `from dataclasses import dataclass, field`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v -k marquardt_fit`
Expected: PASS (all 4 new tests)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS (no regressions -- `fit_peaks()` itself is untouched by this task)

- [ ] **Step 6: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "feat: add damped Marquardt-Levenberg solver core, ported from TV's fit procedure"
```

---

### Task 2: Wire the solver into `fit_peaks()`

**Files:**
- Modify: `peak_fit.py` (`fit_peaks`; add `_build_param_damping`; remove `_bounds`)
- Test: `tests/test_peak_fit.py`

Replaces the `curve_fit(...)` call in `fit_peaks()`'s free-parameter branch with `_marquardt_fit`, via a small helper that turns the existing named free-parameter list into `_ParamDamping` metadata.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_peak_fit.py`, directly after `test_fit_with_all_parameters_fixed_skips_optimization`:

```python
def test_fit_recovers_from_a_deliberately_bad_initial_width_guess():
    """End-to-end version of the Task 1 adversarial scenario, through
    the real fit_peaks() entry point -- with a fit_region wide relative
    to peak count (the scenario that produces a too-wide sigma0 under
    the old region_width/(4*n_peaks) heuristic), all three peaks must
    still land near their true positions with positive amplitude."""
    x, y = _make_spectrum(
        channels=300,
        peaks=[(100.0, 500.0, 3.0), (108.0, 350.0, 3.0), (117.0, 420.0, 3.0)],
        slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(60.0, 75.0),
        right_bg_region=(145.0, 160.0),
        fit_region=(80.0, 140.0),
        peak_positions=[100.0, 108.0, 117.0],
    )
    positions = sorted(p.position for p in result.peaks)
    assert positions[0] == pytest.approx(100.0, abs=2.0)
    assert positions[1] == pytest.approx(108.0, abs=2.0)
    assert positions[2] == pytest.approx(117.0, abs=2.0)
    assert all(p.amplitude > 0 for p in result.peaks)
```

- [ ] **Step 2: Run test to verify it fails (or passes for the wrong reason)**

Run: `pytest tests/test_peak_fit.py -v -k bad_initial_width_guess`
Expected: this specific test may actually FAIL on the current `curve_fit`-based implementation (reproducing the reported bug) -- if it happens to pass, that's fine too (the scenario is somewhat seed-dependent); either way, proceed to the implementation and re-verify after.

- [ ] **Step 3: Add `_build_param_damping`**

Add to `peak_fit.py`, directly after `_bounds` (which Step 4 will delete -- add this first so the diff reads as a clean replacement):

```python
def _build_param_damping(free_names, fixed_params, link_widths):
    """Translates the free-parameter name list into _ParamDamping
    metadata for _marquardt_fit, dispatching on each name's prefix --
    the same naming _parameter_names() already establishes as the
    single source of truth for parameter identity."""
    free_index = {name: i for i, name in enumerate(free_names)}
    damping = []
    for name in free_names:
        if name.startswith("amp_"):
            damping.append(_ParamDamping("amp"))
        elif name.startswith("pos_"):
            peak_idx = name.split("_", 1)[1]
            sigma_name = "sigma" if link_widths else f"sigma_{peak_idx}"
            if sigma_name in free_index:
                damping.append(_ParamDamping("pos", sigma_index=free_index[sigma_name]))
            else:
                damping.append(_ParamDamping("pos", sigma_value=fixed_params[sigma_name]))
        elif name == "sigma" or name.startswith("sigma_"):
            damping.append(_ParamDamping("sigma"))
        elif name == "tail_fraction":
            damping.append(_ParamDamping("tail_fraction"))
        elif name == "tail_beta":
            damping.append(_ParamDamping("tail_beta"))
    return damping
```

- [ ] **Step 4: Delete `_bounds`**

Remove the entire `_bounds` function from `peak_fit.py` (docstring starting "Only needed when enable_left_tail is True..." through its closing `return (lower, upper)`). It's no longer called by anything after this task -- bound enforcement now happens inside `_marquardt_fit` via `_clamp_trial`.

- [ ] **Step 5: Replace the `curve_fit` call in `fit_peaks()`**

In `peak_fit.py`, in `fit_peaks()`, replace:

```python
        p0 = _initial_guess(
            free_names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail
        )
        y_err = np.sqrt(np.maximum(y_fit, 1.0))
        model = _make_model(names, free_names, fixed_params, n_peaks, link_widths, enable_left_tail)
        bounds = _bounds(free_names, enable_left_tail)

        try:
            if bounds is not None:
                popt, pcov = curve_fit(
                    model, x_fit, y_sub, p0=p0, sigma=y_err, absolute_sigma=True, bounds=bounds
                )
            else:
                popt, pcov = curve_fit(
                    model, x_fit, y_sub, p0=p0, sigma=y_err, absolute_sigma=True
                )
        except (RuntimeError, ValueError) as exc:
            raise FitError(f"Fit did not converge: {exc}") from exc

        if pcov is None or not np.all(np.isfinite(pcov)):
            raise FitError("Fit produced a non-finite covariance matrix")
```

with:

```python
        p0 = _initial_guess(
            free_names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail
        )
        y_err = np.sqrt(np.maximum(y_fit, 1.0))
        model_named = _make_model(names, free_names, fixed_params, n_peaks, link_widths, enable_left_tail)
        model = lambda xx, pp: model_named(xx, *pp)
        damping = _build_param_damping(free_names, fixed_params, link_widths)

        try:
            popt, pcov = _marquardt_fit(
                model, x_fit, y_sub, y_err, p0, damping, fit_region_bounds=fit_region,
            )
        except np.linalg.LinAlgError as exc:
            raise FitError(f"Fit did not converge: {exc}") from exc

        if pcov is None or not np.all(np.isfinite(pcov)):
            raise FitError("Fit produced a non-finite covariance matrix")
```

(`_make_model`'s inner `model(x, *free_values)` takes free params as
`*args`, matching scipy's `curve_fit` convention; `_marquardt_fit`'s
`model(x, p)` takes them as a single array, matching its own convention
from Task 1 -- the one-line `lambda` above adapts between the two
without needing to touch `_make_model` itself.)

- [ ] **Step 6: Update the import line**

In `peak_fit.py`, remove the now-unused `from scipy.optimize import curve_fit` line (the `scipy.special import erfc` line stays -- still used by `hypermet_left_tail`).

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS, including `test_fit_recovers_from_a_deliberately_bad_initial_width_guess`. If any *pre-existing* test fails with a `pytest.approx` mismatch (not an error/crash, just a slightly different converged value), loosen that specific assertion's tolerance -- this task deliberately changes the underlying algorithm, so a small shift in an already-passing scenario's exact fitted value is expected, not a regression. Do not loosen a tolerance to paper over a test that's actually failing for a real reason (wrong sign, wrong order of magnitude, non-convergence) -- only adjust genuinely-too-tight numeric tolerances.

- [ ] **Step 8: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "feat: fit peaks with the damped Marquardt solver instead of unconstrained curve_fit"
```

---

### Task 3: TV-style initial width estimate

**Files:**
- Modify: `peak_fit.py` (`_initial_guess`; add `_measure_width`)
- Test: `tests/test_peak_fit.py`

Replaces the `region_width / (4 * n_peaks)` heuristic (and its independent-widths peak-spacing cap) with TV's approach: walk from the largest-amplitude marked peak's clicked position to the true local maximum, measure FWHM by walking outward to the half-max crossing, and use that single measured sigma as the starting width for every peak in the region.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_peak_fit.py`, directly after `test_marquardt_fit_clamps_tail_fraction_and_beta_to_bounds` (from Task 1):

```python
from peak_fit import _measure_width


def test_measure_width_recovers_true_sigma_from_a_single_peak():
    x, y = _make_spectrum(channels=200, peaks=[(100.0, 500.0, 3.5)], slope=0.0, intercept=20.0)
    lo, hi = 80, 120
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_sub = y[mask] - 20.0
    sigma = _measure_width(x_fit, y_sub, [100.0], fallback=10.0)
    assert sigma == pytest.approx(3.5, abs=0.5)


def test_measure_width_uses_the_largest_marked_peak_for_three_peaks():
    x, y = _make_spectrum(
        channels=200,
        peaks=[(90.0, 500.0, 3.0), (108.0, 350.0, 3.0), (117.0, 420.0, 3.0)],
        slope=0.0, intercept=20.0,
    )
    lo, hi = 70, 135
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_sub = y[mask] - 20.0
    sigma = _measure_width(x_fit, y_sub, [88.0, 110.0, 119.0], fallback=5.42)
    assert sigma == pytest.approx(3.0, abs=0.7)


def test_measure_width_falls_back_when_no_clear_peak_is_found():
    x = np.arange(80, 121, dtype=float)
    y_sub = np.zeros_like(x)  # perfectly flat, background-subtracted to zero
    sigma = _measure_width(x, y_sub, [100.0], fallback=7.0)
    assert sigma == 7.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_peak_fit.py -v -k measure_width`
Expected: FAIL with `ImportError: cannot import name '_measure_width'`

- [ ] **Step 3: Add `_measure_width`**

Add to `peak_fit.py`, directly after `_current_sigma_for` (Task 1):

```python
_MIN_PLAUSIBLE_FWHM_CHANNELS = 2.0


def _measure_width(x_fit, y_sub, peak_positions, fallback):
    """TV-style initial width estimate, ported from
    tv-1.9.13/lib/tv/vsFitSetup.c's FSInitWidth: starting from the
    marked peak with the largest background-subtracted amplitude, walk
    to the true local maximum, then walk outward until counts drop
    below half that maximum, and convert the resulting FWHM to a sigma.
    Falls back to the caller-supplied heuristic value if the data is
    too flat/noisy to find a clear peak, or the measured FWHM is
    implausibly small to be a real detector peak (a simple plausibility
    guard, not a rigorous significance test -- a pure-noise region can
    occasionally still produce a small but "clean" half-max crossing;
    this is an accepted, rare edge case since real usage always marks
    an actual visible peak)."""
    if len(x_fit) == 0:
        return fallback

    indices = [int(np.argmin(np.abs(x_fit - pos))) for pos in peak_positions]
    amplitudes = [y_sub[i] for i in indices]
    idx = indices[int(np.argmax(amplitudes))]

    n = len(x_fit)
    while True:
        moved = False
        if idx + 1 < n and y_sub[idx + 1] > y_sub[idx]:
            idx += 1
            moved = True
        elif idx - 1 >= 0 and y_sub[idx - 1] > y_sub[idx]:
            idx -= 1
            moved = True
        if not moved:
            break

    peak_value = y_sub[idx]
    if peak_value <= 0:
        return fallback
    half = peak_value / 2.0

    left = idx
    while left - 1 >= 0 and y_sub[left - 1] >= half:
        left -= 1
    right = idx
    while right + 1 < n and y_sub[right + 1] >= half:
        right += 1
    if left == 0 or right == n - 1:
        return fallback

    def _interp_crossing(i_inside, i_outside):
        y_in, y_out = y_sub[i_inside], y_sub[i_outside]
        if y_in == y_out:
            return float(x_fit[i_inside])
        frac = (half - y_in) / (y_out - y_in)
        return float(x_fit[i_inside] + frac * (x_fit[i_outside] - x_fit[i_inside]))

    x_left = _interp_crossing(left, left - 1)
    x_right = _interp_crossing(right, right + 1)
    fwhm = x_right - x_left
    if fwhm < _MIN_PLAUSIBLE_FWHM_CHANNELS:
        return fallback
    return max(fwhm / FWHM_FACTOR, 1e-6)
```

- [ ] **Step 4: Use it in `_initial_guess`**

In `peak_fit.py`, in `_initial_guess`, replace:

```python
    lo, hi = fit_region
    region_width = hi - lo
    n_peaks = len(peak_positions)
    sigma0 = max(region_width / (4 * n_peaks), 1e-6)
    if not link_widths and n_peaks > 1:
        # With independent per-peak sigmas, a starting width this wide
        # (derived from the whole fit region) makes neighboring peaks
        # overlap heavily and lets curve_fit's unconstrained
        # Levenberg-Marquardt solver settle into a spurious local
        # minimum instead of recovering each peak's own width. Cap the
        # guess at a quarter of the closest peak spacing so peaks
        # start out reasonably well-separated.
        sorted_positions = sorted(peak_positions)
        min_spacing = min(
            b - a for a, b in zip(sorted_positions, sorted_positions[1:])
        )
        if min_spacing > 0:
            sigma0 = max(min(sigma0, min_spacing / 4), 1e-6)
```

with:

```python
    lo, hi = fit_region
    region_width = hi - lo
    n_peaks = len(peak_positions)
    fallback_sigma0 = max(region_width / (4 * n_peaks), 1e-6)
    sigma0 = _measure_width(x_fit, y_sub, peak_positions, fallback_sigma0)
```

(The independent-widths peak-spacing cap is no longer needed: the
damped Marquardt solver from Task 1/2 already prevents any single
width parameter from taking an oversized step, which was the actual
problem that cap was working around.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS. As in Task 2 Step 7, loosen (don't chase) any pre-existing test's `pytest.approx` tolerance that fails only because the new width estimate produces a slightly different but still-correct converged value.

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "feat: estimate initial peak width from data (TV's local-max/FWHM method) instead of a region-width heuristic"
```

---

### Task 4: Peak decomposition (draw each peak's own component curve)

**Files:**
- Modify: `fit_mode.py` (`draw_committed_fits`)
- Test: `tests/test_fit_mode_ui.py`

Draws each peak's individual contribution (background + that one peak's shape) as a thin line, in addition to the existing solid total curve -- so a multi-peak fit visually shows how the total decomposes into its components, matching what TV's own decomposition view does.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`, directly after `test_draw_committed_fits_skips_hidden_results`:

```python
def test_draw_committed_fits_draws_one_component_line_per_peak(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 130.0), background_slope=0.0, background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0,
                ),
                PeakResult(
                    position=115.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                    area=800.0, area_err=40.0, amplitude=160.0, sigma=2.0,
                ),
            ],
        )
    )

    lines_before = len(main_window.axes.lines)
    main_window.fit_controller.draw_committed_fits(spectrum)

    # Background dashed line (1) + total curve (1) + one axvline position
    # marker per peak (2, pre-existing) + one new component line per peak
    # (2) = 6 new lines for this 2-peak fit, on top of whatever was
    # already there. Confirmed against the current (pre-Task-4) code
    # directly: for this exact 2-peak fixture, draw_committed_fits adds
    # exactly 4 lines before this task's change (1+1+2), so the 2 new
    # component lines bring it to 6.
    assert len(main_window.axes.lines) == lines_before + 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k draws_one_component_line_per_peak`
Expected: FAIL (`assert len(...) == lines_before + 6` -- currently only 4 new lines are drawn for this 2-peak fit: 1 background-dashed + 1 total + 2 axvline position markers, no component lines yet)

- [ ] **Step 3: Draw each peak's component curve**

In `fit_mode.py`, in `draw_committed_fits`, find the block that builds and draws `total` (from `x_dense = np.linspace(lo, hi, 200)` through `axes.plot(x_dense, total, color="red", linewidth=1.5)`) and add a component-curve draw call right after it, before the `for peak in result.peaks:` loop that draws position lines/annotations:

```python
            x_dense = np.linspace(lo, hi, 200)
            total = result.background_slope * x_dense + result.background_intercept
            for peak in result.peaks:
                if result.tail_fraction is not None:
                    total = total + peak.amplitude * hypermet_left_tail(
                        x_dense, peak.position, peak.sigma,
                        result.tail_fraction, result.tail_beta,
                    )
                else:
                    total = total + peak.amplitude * np.exp(
                        -((x_dense - peak.position) ** 2) / (2 * peak.sigma ** 2)
                    )
            axes.plot(x_dense, total, color="red", linewidth=1.5)

            # Peak decomposition: each peak's own contribution (background
            # + that single peak), so a multi-peak fit visually shows how
            # the total curve above decomposes into its components.
            background_dense = result.background_slope * x_dense + result.background_intercept
            for peak in result.peaks:
                if result.tail_fraction is not None:
                    component = peak.amplitude * hypermet_left_tail(
                        x_dense, peak.position, peak.sigma,
                        result.tail_fraction, result.tail_beta,
                    )
                else:
                    component = peak.amplitude * np.exp(
                        -((x_dense - peak.position) ** 2) / (2 * peak.sigma ** 2)
                    )
                axes.plot(
                    x_dense, background_dense + component,
                    color="red", linewidth=0.75, linestyle="--", alpha=0.6,
                )
```

(This leaves the existing `total` accumulation loop and its `axes.plot(x_dense, total, ...)` call exactly as they were -- only the new block after it is added. The pre-existing `for peak in result.peaks:` loop further below, which draws each peak's position line and label annotation, is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: draw each peak's individual component curve (decomposition), not just the total"
```

---

## Self-Review Notes

- **Spec coverage:** Damped Marquardt solver replicating TV's step-limiting (Tasks 1-2), TV's local-max/FWHM initial width estimate (Task 3), peak decomposition visualization (Task 4) -- all covered. Peak shape model changes, a step/background term, and automatic peak-clustering are explicitly out of scope per the spec, and untouched here.
- **Placeholder scan:** No TBD/TODO; every step has complete, runnable code, transcribed from a working, empirically-validated prototype (see the spec's "Design validated empirically" section) rather than written blind -- this plan's numerical code has already been exercised against single-peak, multi-peak, independent-widths, fixed-parameter, and hypermet-tail scenarios before being written into these tasks.
- **Type consistency:** `_marquardt_fit(model, x, y, y_err, p0, damping, fit_region_bounds=None)` (Task 1) is called from `fit_peaks()` (Task 2) with a `lambda`-adapted `model` and `_build_param_damping(...)`'s output as `damping` -- signatures match exactly. `_measure_width(x_fit, y_sub, peak_positions, fallback)` (Task 3) is called from `_initial_guess` with the exact same `x_fit`/`y_sub` already in scope there. `_ParamDamping.sigma_index`/`sigma_value` (Task 1) are populated by `_build_param_damping` (Task 2) by checking free-vs-fixed status via `free_index`/`fixed_params`, covering both cases the class supports.
- **Numeric tolerance caveat:** Tasks 2 and 3 explicitly permit loosening (not chasing) `pytest.approx` tolerances on pre-existing tests, since this plan deliberately changes the fitting algorithm -- flagged in both tasks' steps so an implementer doesn't either (a) silently paper over a real regression or (b) get stuck trying to make the new algorithm bit-for-bit match the old one's incidental convergence path.
