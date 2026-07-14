# Fit Repeatability and Parameter Fixing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make marked fits repeatable under changed settings (independent widths, left tail) without re-marking, let the user fix any individual fit parameter to a constant via a new Fit Parameters panel, and report an uncertainty for every fit parameter.

**Architecture:** `peak_fit.py`'s rigid 4-combination parameter layout becomes a named-parameter system (`amp_0`, `pos_0`, `sigma`/`sigma_0`, `tail_fraction`, `tail_beta`, ...); `fit_peaks()` gains a `fixed_params` dict that removes named entries from `curve_fit`'s free vector entirely rather than bounding them to a point. `fit_mode.py`'s `FitModeController` stops clearing marks after a successful fit (every fit now always appends a new `FitResult`, per the approved design), gains a new "Fit Parameters" dock panel for fixing/freeing individual parameters between re-fits, and gains double-click-to-reload on the existing "Fit Results" panel.

**Tech Stack:** Python 3.13, `scipy.optimize.curve_fit`, `numpy`, PySide6/matplotlib (existing), `pytest` with the existing `qapp` fixture.

---

## Before you start

Read `docs/superpowers/specs/2026-07-14-fit-repeatability-and-parameter-fixing-design.md` in full. Run the full suite once to confirm a clean baseline:

Run: `pytest -v` from the repo root (`C:\Users\RIG\Documents\Claude\PeakFinderFitting`).
Expected: all tests pass (103 at last count).

---

### Task 1: Named-parameter refactor in `peak_fit.py` (no behavior change)

**Files:**
- Modify: `peak_fit.py` (replace `_unpack_params`, `_make_model`, `_initial_guess`, `_bounds`, and the body of `fit_peaks`)
- Test: `tests/test_peak_fit.py` (no new tests -- existing tests must pass unchanged, proving this is a pure refactor)

Today, `_unpack_params(params, n_peaks, link_widths, enable_left_tail)` walks a flat `curve_fit` parameter array *positionally*. To support fixing an arbitrary subset of parameters later (Task 2), every parameter slot first needs a stable *name* (`amp_0`, `pos_0`, `sigma_0` or shared `sigma`, `tail_fraction`, `tail_beta`) so a fixed-parameter dict can refer to it. This task introduces that naming and re-derives the exact same behavior through it -- no test changes needed, since nothing observable should change yet.

- [ ] **Step 1: Replace the four helper functions**

In `peak_fit.py`, delete `_unpack_params`, `_make_model`, `_initial_guess`, and `_bounds` entirely, and replace them with:

```python
def _parameter_names(n_peaks, link_widths, enable_left_tail):
    """Canonical, ordered list of every fittable parameter's name for
    a given (n_peaks, link_widths, enable_left_tail) configuration.
    This is the single source of truth for parameter identity, shared
    by the initial guess, bounds, model function, and (from Task 2)
    fixed-parameter handling in fit_peaks(). Public (no leading
    underscore) because fit_mode.py's Fit Parameters panel needs this
    same ordering to know which rows to show."""
    names = []
    for i in range(n_peaks):
        names.append(f"amp_{i}")
        names.append(f"pos_{i}")
        if not link_widths:
            names.append(f"sigma_{i}")
    if link_widths:
        names.append("sigma")
    if enable_left_tail:
        names.append("tail_fraction")
        names.append("tail_beta")
    return names


def _unpack_named(values_by_name, n_peaks, link_widths, enable_left_tail):
    """Extracts (amplitudes, positions, sigmas, tail_fraction, tail_beta)
    from a {name: value} mapping covering every name _parameter_names()
    would produce for this configuration. `sigmas` is always length
    n_peaks (the shared value repeated if linked); tail_fraction/
    tail_beta are None if enable_left_tail is False."""
    amplitudes = [values_by_name[f"amp_{i}"] for i in range(n_peaks)]
    positions = [values_by_name[f"pos_{i}"] for i in range(n_peaks)]
    if link_widths:
        sigmas = [values_by_name["sigma"]] * n_peaks
    else:
        sigmas = [values_by_name[f"sigma_{i}"] for i in range(n_peaks)]
    tail_fraction = values_by_name.get("tail_fraction")
    tail_beta = values_by_name.get("tail_beta")
    return amplitudes, positions, sigmas, tail_fraction, tail_beta


def _make_model(names, n_peaks, link_widths, enable_left_tail):
    def model(x, *params):
        values_by_name = dict(zip(names, params))
        amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_named(
            values_by_name, n_peaks, link_widths, enable_left_tail
        )
        result = np.zeros_like(x, dtype=float)
        for amplitude, position, sigma in zip(amplitudes, positions, sigmas):
            if enable_left_tail:
                result = result + amplitude * hypermet_left_tail(
                    x, position, sigma, tail_fraction, tail_beta
                )
            else:
                result = result + amplitude * np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
        return result
    return model


def _initial_guess(names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail):
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

    guess_by_name = {}
    for i, pos in enumerate(peak_positions):
        idx = int(np.argmin(np.abs(x_fit - pos)))
        guess_by_name[f"amp_{i}"] = float(y_sub[idx])
        guess_by_name[f"pos_{i}"] = float(pos)
        if not link_widths:
            guess_by_name[f"sigma_{i}"] = sigma0
    if link_widths:
        guess_by_name["sigma"] = sigma0
    if enable_left_tail:
        guess_by_name["tail_fraction"] = 0.05
        guess_by_name["tail_beta"] = max(sigma0, TAIL_BETA_MIN)
    return [guess_by_name[name] for name in names]


def _bounds(names, enable_left_tail):
    """Only needed when enable_left_tail is True (to keep the tail
    fraction genuinely small and the decay constant away from zero);
    returns None otherwise so the no-tail fits keep using curve_fit's
    default unconstrained method, unchanged from v1's behavior."""
    if not enable_left_tail:
        return None
    bounds_by_name = {}
    for name in names:
        if name == "tail_fraction":
            bounds_by_name[name] = (0.0, TAIL_FRACTION_MAX)
        elif name == "tail_beta":
            bounds_by_name[name] = (TAIL_BETA_MIN, np.inf)
        elif name == "sigma" or name.startswith("sigma_"):
            bounds_by_name[name] = (1e-6, np.inf)
        else:
            bounds_by_name[name] = (-np.inf, np.inf)
    lower = [bounds_by_name[name][0] for name in names]
    upper = [bounds_by_name[name][1] for name in names]
    return (lower, upper)
```

- [ ] **Step 2: Update `fit_peaks()` to use the named helpers**

Replace the body of `fit_peaks()` (keep the same signature) with:

```python
def fit_peaks(
    x, y, left_bg_region, right_bg_region, fit_region, peak_positions,
    link_widths=True, enable_left_tail=False,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not peak_positions:
        raise FitError("At least one peak must be marked")

    slope, intercept = _compute_background(x, y, left_bg_region, right_bg_region)

    lo, hi = fit_region
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size == 0:
        raise FitError(f"Fit region {fit_region} contains no data")

    n_peaks = len(peak_positions)
    names = _parameter_names(n_peaks, link_widths, enable_left_tail)
    if x_fit.size < len(names):
        raise FitError(
            f"Fit region has {x_fit.size} data points, need at least "
            f"{len(names)} for {n_peaks} peak(s)"
        )

    y_sub = y_fit - (slope * x_fit + intercept)
    p0 = _initial_guess(names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail)
    y_err = np.sqrt(np.maximum(y_fit, 1.0))
    model = _make_model(names, n_peaks, link_widths, enable_left_tail)
    bounds = _bounds(names, enable_left_tail)

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

    perr = np.sqrt(np.diag(pcov))
    values_by_name = dict(zip(names, popt))
    err_by_name = dict(zip(names, perr))
    amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_named(
        values_by_name, n_peaks, link_widths, enable_left_tail
    )
    amplitude_errs, position_errs, sigma_errs, tail_fraction_err, tail_beta_err = _unpack_named(
        err_by_name, n_peaks, link_widths, enable_left_tail
    )

    peaks = []
    for i in range(n_peaks):
        amplitude = amplitudes[i]
        position = positions[i]
        sigma = abs(sigmas[i])
        amplitude_err = amplitude_errs[i]
        position_err = position_errs[i]
        sigma_err = sigma_errs[i]
        fwhm = FWHM_FACTOR * sigma
        fwhm_err = FWHM_FACTOR * sigma_err
        # This is the analytic integral of the plain Gaussian core only.
        # When enable_left_tail is True, the fitted shape also carries a
        # tail term (see hypermet_left_tail) whose own contribution to the
        # true integral is not included here -- a known approximation.
        area = amplitude * sigma * np.sqrt(2 * np.pi)
        rel_err_sq = 0.0
        if amplitude != 0:
            rel_err_sq += (amplitude_err / amplitude) ** 2
        if sigma != 0:
            rel_err_sq += (sigma_err / sigma) ** 2
        area_err = abs(area) * np.sqrt(rel_err_sq)
        peaks.append(
            PeakResult(
                position=float(position), position_err=float(position_err),
                fwhm=float(fwhm), fwhm_err=float(fwhm_err),
                area=float(area), area_err=float(area_err),
                amplitude=float(amplitude), sigma=float(sigma),
            )
        )

    return FitResult(
        left_bg_region=tuple(left_bg_region), right_bg_region=tuple(right_bg_region),
        fit_region=tuple(fit_region), background_slope=float(slope),
        background_intercept=float(intercept), peaks=peaks,
        link_widths=link_widths,
        tail_fraction=float(tail_fraction) if tail_fraction is not None else None,
        tail_fraction_err=float(tail_fraction_err) if tail_fraction_err is not None else None,
        tail_beta=float(tail_beta) if tail_beta is not None else None,
        tail_beta_err=float(tail_beta_err) if tail_beta_err is not None else None,
    )
```

- [ ] **Step 3: Run the full peak_fit test file to confirm no regression**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS (all existing tests, unchanged -- this proves the refactor preserved behavior exactly)

- [ ] **Step 4: Commit**

```bash
git add peak_fit.py
git commit -m "refactor: name every fit_peaks parameter for upcoming fixed-parameter support"
```

---

### Task 2: `fixed_params` support and complete per-parameter uncertainties

**Files:**
- Modify: `peak_fit.py` (`PeakResult`/`FitResult` dataclasses, `_make_model`, `_initial_guess`, `_bounds`, `fit_peaks`)
- Test: `tests/test_peak_fit.py`

This is the core of the feature: any subset of the named parameters from Task 1 can be held fixed at a constant, excluded entirely from `curve_fit`'s free-parameter vector (not bounded to a point -- genuinely removed). `PeakResult` also gains `amplitude_err`/`sigma_err` (currently missing), and a fixed parameter's uncertainty is reported as exactly `0.0`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_peak_fit.py`, directly after `test_fit_result_accepts_explicit_tail_fields`:

```python
def test_peak_result_amplitude_and_sigma_err_default_to_zero():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    assert peak.amplitude_err == 0.0
    assert peak.sigma_err == 0.0


def test_fit_result_fixed_params_defaults_to_empty_dict():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    result = FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[peak],
    )
    assert result.fixed_params == {}
```

Add to `tests/test_peak_fit.py`, directly after `test_fit_single_peak_with_poisson_noise`:

```python
def test_fit_single_peak_reports_amplitude_and_sigma_uncertainties():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
    )
    peak = result.peaks[0]
    assert peak.amplitude_err > 0
    assert peak.sigma_err > 0
```

Add to `tests/test_peak_fit.py`, directly after `test_fit_independent_widths_with_left_tail_both_enabled`:

```python
def test_fit_with_fixed_sigma_holds_it_constant():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
        fixed_params={"sigma": 3.0},
    )
    peak = result.peaks[0]
    assert peak.sigma == 3.0
    assert peak.sigma_err == 0.0
    assert result.fixed_params == {"sigma": 3.0}


def test_fit_with_fixed_position_holds_it_constant():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
        fixed_params={"pos_0": 100.0},
    )
    peak = result.peaks[0]
    assert peak.position == 100.0
    assert peak.position_err == 0.0


def test_fit_with_fixed_amplitude_reduces_area_uncertainty():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    free_result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
    )
    fixed_result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
        fixed_params={"amp_0": free_result.peaks[0].amplitude},
    )
    assert fixed_result.peaks[0].amplitude_err == 0.0
    # Only sigma's uncertainty contributes now, so area_err must shrink.
    assert fixed_result.peaks[0].area_err < free_result.peaks[0].area_err


def test_fit_with_fixed_tail_fraction_and_beta():
    x = np.arange(200, dtype=float)
    y = 20.0 + 500.0 * hypermet_left_tail(x, position=100.0, sigma=3.0, r=0.1, beta=4.0)
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(80.0, 120.0),
        peak_positions=[100.0],
        enable_left_tail=True,
        fixed_params={"tail_fraction": 0.1, "tail_beta": 4.0},
    )
    assert result.tail_fraction == 0.1
    assert result.tail_fraction_err == 0.0
    assert result.tail_beta == 4.0
    assert result.tail_beta_err == 0.0


def test_fit_rejects_unknown_fixed_parameter_name():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(
            x, y,
            left_bg_region=(70.0, 85.0),
            right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0),
            peak_positions=[100.0],
            fixed_params={"not_a_real_param": 1.0},
        )


def test_fit_with_all_parameters_fixed_skips_optimization():
    x, y = _make_spectrum(
        channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
        fixed_params={"amp_0": 500.0, "pos_0": 100.0, "sigma": 3.0},
    )
    peak = result.peaks[0]
    assert peak.amplitude == 500.0
    assert peak.position == 100.0
    assert peak.sigma == 3.0
    assert peak.amplitude_err == 0.0
    assert peak.position_err == 0.0
    assert peak.sigma_err == 0.0
    assert peak.area_err == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_peak_fit.py -v -k "amplitude_and_sigma_err or fixed_params_defaults or amplitude_and_sigma_uncertainties or fixed_sigma or fixed_position or fixed_amplitude or fixed_tail or unknown_fixed_parameter or all_parameters_fixed"`
Expected: FAIL (`TypeError: PeakResult.__init__() got an unexpected keyword argument` is not raised since these are new attributes read, not passed -- expect `AttributeError: 'PeakResult' object has no attribute 'amplitude_err'` and `TypeError: fit_peaks() got an unexpected keyword argument 'fixed_params'`)

- [ ] **Step 3: Extend the dataclasses**

In `peak_fit.py`, change the import line from:

```python
from dataclasses import dataclass
```

to:

```python
from dataclasses import dataclass, field
```

Replace the `PeakResult` dataclass:

```python
@dataclass
class PeakResult:
    position: float
    position_err: float
    fwhm: float
    fwhm_err: float
    area: float
    area_err: float
    amplitude: float
    sigma: float
```

with:

```python
@dataclass
class PeakResult:
    position: float
    position_err: float
    fwhm: float
    fwhm_err: float
    area: float
    area_err: float
    amplitude: float
    sigma: float
    amplitude_err: float = 0.0
    sigma_err: float = 0.0
```

Replace the `FitResult` dataclass's last line (`tail_beta_err: float = None`) so the full dataclass reads:

```python
@dataclass
class FitResult:
    left_bg_region: tuple
    right_bg_region: tuple
    fit_region: tuple
    background_slope: float
    background_intercept: float
    peaks: list
    link_widths: bool = True
    tail_fraction: float = None
    tail_fraction_err: float = None
    tail_beta: float = None
    tail_beta_err: float = None
    fixed_params: dict = field(default_factory=dict)
```

- [ ] **Step 4: Generalize `_make_model`/`_initial_guess`/`_bounds` for a free-parameter subset**

Replace `_make_model` with:

```python
def _make_model(names, free_names, fixed_params, n_peaks, link_widths, enable_left_tail):
    def model(x, *free_values):
        values_by_name = dict(fixed_params)
        values_by_name.update(zip(free_names, free_values))
        amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_named(
            values_by_name, n_peaks, link_widths, enable_left_tail
        )
        result = np.zeros_like(x, dtype=float)
        for amplitude, position, sigma in zip(amplitudes, positions, sigmas):
            if enable_left_tail:
                result = result + amplitude * hypermet_left_tail(
                    x, position, sigma, tail_fraction, tail_beta
                )
            else:
                result = result + amplitude * np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
        return result
    return model
```

Change `_initial_guess`'s signature and final line from `names` to `free_names`:

```python
def _initial_guess(free_names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail):
```

(body unchanged except the last line, which becomes:)

```python
    return [guess_by_name[name] for name in free_names]
```

Change `_bounds`'s signature and loop from `names` to `free_names`:

```python
def _bounds(free_names, enable_left_tail):
    """Only needed when enable_left_tail is True (to keep the tail
    fraction genuinely small and the decay constant away from zero);
    returns None otherwise so the no-tail fits keep using curve_fit's
    default unconstrained method, unchanged from v1's behavior."""
    if not enable_left_tail:
        return None
    bounds_by_name = {}
    for name in free_names:
        if name == "tail_fraction":
            bounds_by_name[name] = (0.0, TAIL_FRACTION_MAX)
        elif name == "tail_beta":
            bounds_by_name[name] = (TAIL_BETA_MIN, np.inf)
        elif name == "sigma" or name.startswith("sigma_"):
            bounds_by_name[name] = (1e-6, np.inf)
        else:
            bounds_by_name[name] = (-np.inf, np.inf)
    lower = [bounds_by_name[name][0] for name in free_names]
    upper = [bounds_by_name[name][1] for name in free_names]
    return (lower, upper)
```

- [ ] **Step 5: Rewrite `fit_peaks()` to split free/fixed and handle the all-fixed case**

Replace the `fit_peaks()` signature and body with:

```python
def fit_peaks(
    x, y, left_bg_region, right_bg_region, fit_region, peak_positions,
    link_widths=True, enable_left_tail=False, fixed_params=None,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fixed_params = dict(fixed_params) if fixed_params else {}

    if not peak_positions:
        raise FitError("At least one peak must be marked")

    slope, intercept = _compute_background(x, y, left_bg_region, right_bg_region)

    lo, hi = fit_region
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size == 0:
        raise FitError(f"Fit region {fit_region} contains no data")

    n_peaks = len(peak_positions)
    names = _parameter_names(n_peaks, link_widths, enable_left_tail)
    unknown_fixed = set(fixed_params) - set(names)
    if unknown_fixed:
        raise FitError(f"Unknown fixed parameter name(s): {sorted(unknown_fixed)}")
    free_names = [name for name in names if name not in fixed_params]

    if free_names and x_fit.size < len(free_names):
        raise FitError(
            f"Fit region has {x_fit.size} data points, need at least "
            f"{len(free_names)} free parameter(s) for {n_peaks} peak(s)"
        )

    y_sub = y_fit - (slope * x_fit + intercept)

    if not free_names:
        # Every parameter is fixed -- nothing to optimize. Evaluate
        # directly at the fixed values instead of calling curve_fit
        # with an empty parameter vector. No fit was performed, so
        # every value's uncertainty is exactly 0.0.
        values_by_name = dict(fixed_params)
        err_by_name = {name: 0.0 for name in names}
    else:
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

        perr = np.sqrt(np.diag(pcov))
        values_by_name = dict(fixed_params)
        values_by_name.update(zip(free_names, popt))
        err_by_name = {name: 0.0 for name in fixed_params}
        err_by_name.update(zip(free_names, perr))

    amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_named(
        values_by_name, n_peaks, link_widths, enable_left_tail
    )
    amplitude_errs, position_errs, sigma_errs, tail_fraction_err, tail_beta_err = _unpack_named(
        err_by_name, n_peaks, link_widths, enable_left_tail
    )

    peaks = []
    for i in range(n_peaks):
        amplitude = amplitudes[i]
        position = positions[i]
        sigma = abs(sigmas[i])
        amplitude_err = amplitude_errs[i]
        position_err = position_errs[i]
        sigma_err = sigma_errs[i]
        fwhm = FWHM_FACTOR * sigma
        fwhm_err = FWHM_FACTOR * sigma_err
        # This is the analytic integral of the plain Gaussian core only.
        # When enable_left_tail is True, the fitted shape also carries a
        # tail term (see hypermet_left_tail) whose own contribution to the
        # true integral is not included here -- a known approximation.
        area = amplitude * sigma * np.sqrt(2 * np.pi)
        rel_err_sq = 0.0
        if amplitude != 0:
            rel_err_sq += (amplitude_err / amplitude) ** 2
        if sigma != 0:
            rel_err_sq += (sigma_err / sigma) ** 2
        area_err = abs(area) * np.sqrt(rel_err_sq)
        peaks.append(
            PeakResult(
                position=float(position), position_err=float(position_err),
                fwhm=float(fwhm), fwhm_err=float(fwhm_err),
                area=float(area), area_err=float(area_err),
                amplitude=float(amplitude), amplitude_err=float(amplitude_err),
                sigma=float(sigma), sigma_err=float(sigma_err),
            )
        )

    return FitResult(
        left_bg_region=tuple(left_bg_region), right_bg_region=tuple(right_bg_region),
        fit_region=tuple(fit_region), background_slope=float(slope),
        background_intercept=float(intercept), peaks=peaks,
        link_widths=link_widths,
        tail_fraction=float(tail_fraction) if tail_fraction is not None else None,
        tail_fraction_err=float(tail_fraction_err) if tail_fraction_err is not None else None,
        tail_beta=float(tail_beta) if tail_beta is not None else None,
        tail_beta_err=float(tail_beta_err) if tail_beta_err is not None else None,
        fixed_params=dict(fixed_params),
    )
```

- [ ] **Step 6: Add the cross-module helpers used by `fit_mode.py` later**

Add to `peak_fit.py`, after `fit_peaks()`:

```python
def parameter_names(n_peaks, link_widths, enable_left_tail):
    """Public alias of _parameter_names -- fit_mode.py's Fit Parameters
    panel needs this exact ordering to know which rows to display."""
    return _parameter_names(n_peaks, link_widths, enable_left_tail)


def fit_result_values_by_name(result):
    """Reconstructs the {name: value} mapping (matching
    parameter_names()'s canonical naming) from an already-committed
    FitResult -- used by the UI layer to populate the Fit Parameters
    panel and to pre-fill fixed_params when reloading an older fit."""
    n_peaks = len(result.peaks)
    link_widths = result.link_widths
    enable_left_tail = result.tail_fraction is not None
    values = {}
    for i, peak in enumerate(result.peaks):
        values[f"amp_{i}"] = peak.amplitude
        values[f"pos_{i}"] = peak.position
        if not link_widths:
            values[f"sigma_{i}"] = peak.sigma
    if link_widths and result.peaks:
        values["sigma"] = result.peaks[0].sigma
    if enable_left_tail:
        values["tail_fraction"] = result.tail_fraction
        values["tail_beta"] = result.tail_beta
    return values
```

- [ ] **Step 7: Run the full peak_fit test file**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS (all tests, including every pre-existing one -- `fixed_params` defaults to `None`/`{}` so none of Task 1's behavior changes for existing callers)

- [ ] **Step 8: Run the full suite**

Run: `pytest -v`
Expected: PASS (no regressions in `test_fit_mode.py`/`test_fit_mode_ui.py`, which don't pass `fixed_params` yet)

- [ ] **Step 9: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "feat: support fixing individual fit parameters and report per-parameter uncertainties"
```

---

### Task 3: Marks persist after a successful fit

**Files:**
- Modify: `fit_mode.py:361-382` (the `run_fit` method)
- Test: `tests/test_fit_mode_ui.py`

Currently `run_fit()` calls `self._clear_progress()` after committing a successful fit, wiping the marks. Per the approved design, marks now stay live so the user can immediately change checkboxes/fixed parameters and re-fit -- each successful fit always appends a new entry (never replaces one).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_mode_ui.py`, directly after `test_full_fit_flow_commits_a_fit_result`:

```python
def test_marks_persist_after_a_successful_fit(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    regions = main_window.fit_controller.state.bg_regions
    assert regions[0] == pytest.approx((70.0, 85.0))
    assert regions[1] == pytest.approx((115.0, 130.0))
    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
    assert main_window.fit_controller.state.peak_positions == pytest.approx([100.0])
    assert main_window.fit_button.isEnabled() is True


def test_refitting_same_marks_with_changed_checkbox_appends_a_new_entry(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    assert spectrum.fits[0].link_widths is True
    assert spectrum.fits[1].link_widths is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -v -k "marks_persist or refitting_same_marks"`
Expected: FAIL (`state.bg_regions == []` since `run_fit()` currently clears progress; second test fails with `assert 1 == 2`)

- [ ] **Step 3: Remove the clear-after-fit call**

In `fit_mode.py`, in `run_fit()`, remove this line (currently right after `active.fits.append(result)`):

```python
        self._clear_progress()
```

`run_fit()`'s body should now read (unchanged apart from that one removed line):

```python
    def run_fit(self):
        if not self.state.ready_to_fit():
            return
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        left, right = self.state.ordered_bg_regions()
        x = np.arange(len(active.data), dtype=float)
        y = active.data
        link_widths = not self.main_window.independent_widths_action.isChecked()
        enable_left_tail = self.main_window.left_tail_action.isChecked()
        try:
            result = fit_peaks(
                x, y, left, right, self.state.fit_region, list(self.state.peak_positions),
                link_widths=link_widths, enable_left_tail=enable_left_tail,
            )
        except FitError as exc:
            self._show_status_message(f"Fit failed: {exc}", 5000)
            return
        active.fits.append(result)
        self.main_window._plot_data(preserve_view=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (all tests -- none of the pre-existing tests asserted marks were cleared after a fit, only `test_clear_discards_in_progress_marks_without_committing` checks clearing, and that goes through `clear()`, not `run_fit()`)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: keep marks live after a fit so it can be repeated under changed settings"
```

---

### Task 4: Fit Parameters panel (new)

**Files:**
- Modify: `fit_mode.py` (imports, `FitModeController.__init__`, add `build_parameters_panel`/`update_parameters_panel`/`_on_fix_toggled`/`fixed_params_from_panel`, update `_clear_progress`)
- Modify: `main_window.py:156-158` (wire up the new panel)
- Test: `tests/test_fit_mode_ui.py`

A new dockable "Fit Parameters" panel (same construction pattern as the existing "Fit Results" panel) shows one row per parameter after a fit runs, with a Value cell and a Fix checkbox. This task builds the panel and populates it after each fit; wiring Fix/Value *into* the next `fit_peaks()` call is Task 5.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_mode_ui.py`, directly after `test_refitting_same_marks_with_changed_checkbox_appends_a_new_entry`:

```python
def test_parameters_panel_populates_after_a_fit(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    table = main_window.fit_controller.parameters_table
    assert table.rowCount() == 3  # amp_0, pos_0, sigma (linked default)
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Shared sigma"]


def test_parameters_panel_rebuilds_when_row_set_changes(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared sigma"

    main_window.independent_widths_action.setChecked(True)
    main_window.fit_controller.run_fit()

    assert table.rowCount() == 3  # amp_0, pos_0, sigma_0 (independent now)
    labels = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert labels == ["Peak 1 amplitude", "Peak 1 position", "Peak 1 sigma"]
    # Row set changed, so the earlier Fix checkbox must not have survived.
    assert table.cellWidget(2, 2).isChecked() is False


def test_parameters_panel_updates_values_in_place_when_row_set_is_unchanged(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared sigma"

    main_window.fit_controller.run_fit()  # re-fit with the same checkboxes/marks

    assert table.rowCount() == 3
    assert table.cellWidget(2, 2).isChecked() is True  # Fix state survived


def test_clear_empties_the_parameters_panel(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()

    main_window.fit_controller.clear()

    assert main_window.fit_controller.parameters_table.rowCount() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -v -k "parameters_panel"`
Expected: FAIL with `AttributeError: 'FitModeController' object has no attribute 'parameters_table'`

- [ ] **Step 3: Update imports in `fit_mode.py`**

Replace:

```python
from PySide6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMenu, QToolBar

from peak_fit import FitError, fit_peaks, hypermet_left_tail
```

with:

```python
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QListWidget, QListWidgetItem, QMenu, QTableWidget,
    QTableWidgetItem, QToolBar,
)

from peak_fit import (
    FitError, fit_peaks, fit_result_values_by_name, hypermet_left_tail, parameter_names,
)
```

- [ ] **Step 4: Add a name-to-label helper**

Add to `fit_mode.py`, after the `_MARK_TYPE_LABEL` dict:

```python
def _parameter_label(name):
    """Human-readable row label for a canonical parameter name from
    peak_fit.parameter_names() -- e.g. "amp_0" -> "Peak 1 amplitude"."""
    if name == "sigma":
        return "Shared sigma"
    if name == "tail_fraction":
        return "Tail fraction (r)"
    if name == "tail_beta":
        return "Tail beta (β)"
    prefix, index = name.rsplit("_", 1)
    peak_num = int(index) + 1
    kind = {"amp": "amplitude", "pos": "position", "sigma": "sigma"}[prefix]
    return f"Peak {peak_num} {kind}"
```

- [ ] **Step 5: Track parameter-panel state in `__init__` and update `_clear_progress`**

In `FitModeController.__init__`, add a new attribute after `self._status_message_until = 0.0`:

```python
        self._parameter_names_shown = []
```

In `_clear_progress`, add a line to also empty the parameters table (after `self.state.reset()`):

```python
    def _clear_progress(self):
        for artist in self._progress_artists:
            try:
                artist.remove()
            except NotImplementedError:
                pass
        self._progress_artists = []
        self.state.reset()
        self.parameters_table.setRowCount(0)
        self._parameter_names_shown = []
```

- [ ] **Step 6: Add `build_parameters_panel`/`update_parameters_panel`/`_on_fix_toggled`**

Add to `fit_mode.py`, directly after `build_results_panel`:

```python
    def build_parameters_panel(self):
        mw = self.main_window
        self.parameters_table = QTableWidget(0, 3)
        self.parameters_table.setHorizontalHeaderLabels(["Parameter", "Value", "Fix"])

        self.parameters_dock = QDockWidget("Fit Parameters", mw)
        self.parameters_dock.setWidget(self.parameters_table)
        mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.parameters_dock)

        self.toggle_parameters_panel_action = QAction("Fit Parameters", mw)
        self.toggle_parameters_panel_action.setCheckable(True)
        self.toggle_parameters_panel_action.setToolTip("Show/hide fixable fit parameters")
        self.toggle_parameters_panel_action.toggled.connect(self.parameters_dock.setVisible)
        self.parameters_dock.visibilityChanged.connect(
            self.toggle_parameters_panel_action.setChecked
        )
        self.parameters_dock.setVisible(False)

        parameters_tab_bar = QToolBar("Fit Parameters Tab", mw)
        parameters_tab_bar.setMovable(False)
        parameters_tab_bar.setFloatable(False)
        parameters_tab_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        parameters_tab_bar.addAction(self.toggle_parameters_panel_action)
        mw.addToolBar(Qt.ToolBarArea.RightToolBarArea, parameters_tab_bar)

    def update_parameters_panel(self, names, values_by_name):
        """Rebuilds the Fit Parameters table (resetting every Fix
        checkbox) when the set of parameter names has changed since
        the last fit for these marks; otherwise updates displayed
        values in place, preserving Fix checkbox state and any
        user-edited fixed values."""
        if names != self._parameter_names_shown:
            self.parameters_table.setRowCount(0)
            self._parameter_names_shown = list(names)
            for name in names:
                row = self.parameters_table.rowCount()
                self.parameters_table.insertRow(row)

                label_item = QTableWidgetItem(_parameter_label(name))
                label_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.parameters_table.setItem(row, 0, label_item)

                value_item = QTableWidgetItem(f"{values_by_name[name]:.6g}")
                value_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.parameters_table.setItem(row, 1, value_item)

                fix_checkbox = QCheckBox()
                fix_checkbox.toggled.connect(
                    lambda checked, r=row: self._on_fix_toggled(r, checked)
                )
                self.parameters_table.setCellWidget(row, 2, fix_checkbox)
        else:
            for row, name in enumerate(names):
                fix_checkbox = self.parameters_table.cellWidget(row, 2)
                if not fix_checkbox.isChecked():
                    self.parameters_table.item(row, 1).setText(f"{values_by_name[name]:.6g}")

    def _on_fix_toggled(self, row, checked):
        value_item = self.parameters_table.item(row, 1)
        if checked:
            value_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable
            )
        else:
            value_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
```

- [ ] **Step 7: Populate the panel from `run_fit()`**

In `run_fit()`, add a call right after `active.fits.append(result)`:

```python
        active.fits.append(result)
        names = parameter_names(len(result.peaks), result.link_widths, result.tail_fraction is not None)
        self.update_parameters_panel(names, fit_result_values_by_name(result))
        self.main_window._plot_data(preserve_view=True)
```

- [ ] **Step 8: Wire the panel into `main_window.py`**

In `main_window.py`, in `MainWindow.__init__`, replace:

```python
        self.fit_controller = FitModeController(self)
        self._build_fit_mode_buttons()
        self.fit_controller.build_results_panel()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._update_fit_mode_availability()
```

with:

```python
        self.fit_controller = FitModeController(self)
        self._build_fit_mode_buttons()
        self.fit_controller.build_results_panel()
        self.fit_controller.build_parameters_panel()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._update_fit_mode_availability()
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (all tests)

- [ ] **Step 10: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add fit_mode.py main_window.py tests/test_fit_mode_ui.py
git commit -m "feat: add Fit Parameters panel populated after each fit"
```

---

### Task 5: Wire Fix/Value state into `run_fit()`

**Files:**
- Modify: `fit_mode.py` (`run_fit`; add `fixed_params_from_panel`)
- Test: `tests/test_fit_mode_ui.py`

The panel exists and populates (Task 4); this task makes the Fix checkboxes and their Value cells actually constrain the next fit.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_mode_ui.py`, directly after `test_clear_empties_the_parameters_panel`:

```python
def test_fixing_a_parameter_in_the_panel_holds_it_for_the_next_fit(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared sigma"
    table.item(2, 1).setText("5.0")

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 2
    second = spectrum.fits[1]
    assert second.peaks[0].sigma == 5.0
    assert second.peaks[0].sigma_err == 0.0
    assert second.fixed_params == {"sigma": 5.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k fixing_a_parameter_in_the_panel`
Expected: FAIL (`assert second.peaks[0].sigma == 5.0` fails since `fixed_params` isn't read from the panel yet -- the re-fit is fully free)

- [ ] **Step 3: Add `fixed_params_from_panel` and use it in `run_fit`**

Add to `fit_mode.py`, directly after `_on_fix_toggled`:

```python
    def fixed_params_from_panel(self):
        """Reads the current Fix checkboxes/values from the Fit
        Parameters panel into a {name: value} dict for the next
        fit_peaks() call. Empty when nothing is fixed (including the
        first fit for a fresh set of marks, before the panel has ever
        been populated)."""
        fixed = {}
        for row, name in enumerate(self._parameter_names_shown):
            fix_checkbox = self.parameters_table.cellWidget(row, 2)
            if fix_checkbox is not None and fix_checkbox.isChecked():
                fixed[name] = float(self.parameters_table.item(row, 1).text())
        return fixed
```

In `run_fit()`, replace:

```python
        try:
            result = fit_peaks(
                x, y, left, right, self.state.fit_region, list(self.state.peak_positions),
                link_widths=link_widths, enable_left_tail=enable_left_tail,
            )
```

with:

```python
        try:
            result = fit_peaks(
                x, y, left, right, self.state.fit_region, list(self.state.peak_positions),
                link_widths=link_widths, enable_left_tail=enable_left_tail,
                fixed_params=self.fixed_params_from_panel(),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: wire Fit Parameters panel's Fix/Value state into the next fit"
```

---

### Task 6: Double-click a Fit Results entry to reload it for editing

**Files:**
- Modify: `fit_mode.py` (`build_results_panel`; add `_on_result_double_clicked`)
- Test: `tests/test_fit_mode_ui.py`

Double-clicking an entry in the Fit Results panel copies its regions/peaks/checkbox settings back into the live marking state and repopulates the Fit Parameters panel with its values, with Fix checkboxes restored to match that entry's `fixed_params`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`, directly after `test_fixing_a_parameter_in_the_panel_holds_it_for_the_next_fit`:

```python
def test_double_click_reloads_a_committed_fit_for_editing(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    main_window.fit_controller.run_fit()
    table = main_window.fit_controller.parameters_table
    table.cellWidget(2, 2).setChecked(True)  # fix "Shared sigma"
    main_window.fit_controller.run_fit()  # second entry, sigma fixed

    main_window.fit_controller.clear()
    assert main_window.fit_controller.state.bg_regions == []

    item = main_window.fit_controller.results_list.item(1)
    main_window.fit_controller._on_result_double_clicked(item)

    regions = main_window.fit_controller.state.bg_regions
    assert regions[0] == pytest.approx((70.0, 85.0))
    assert regions[1] == pytest.approx((115.0, 130.0))
    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
    assert main_window.fit_controller.state.peak_positions == pytest.approx([100.0])
    assert main_window.independent_widths_action.isChecked() is False
    assert main_window.left_tail_action.isChecked() is False

    reloaded_table = main_window.fit_controller.parameters_table
    assert reloaded_table.rowCount() == 3
    assert reloaded_table.cellWidget(2, 2).isChecked() is True  # sigma was fixed
    assert reloaded_table.cellWidget(0, 2).isChecked() is False  # amplitude was free
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k double_click_reloads`
Expected: FAIL with `AttributeError: 'FitModeController' object has no attribute '_on_result_double_clicked'`

- [ ] **Step 3: Connect the double-click signal**

In `build_results_panel`, add a line right after `self.results_list.customContextMenuRequested.connect(self._on_results_context_menu)`:

```python
        self.results_list.itemDoubleClicked.connect(self._on_result_double_clicked)
```

- [ ] **Step 4: Implement the reload handler**

Add to `fit_mode.py`, directly after `_on_results_context_menu`:

```python
    def _on_result_double_clicked(self, item):
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        index = self.results_list.row(item)
        result = active.fits[index]

        self._clear_progress()

        # Bypasses the click-pairing API (add_bg_click/add_fit_click)
        # deliberately -- this restores a previously-computed,
        # already-valid state wholesale, not a fresh in-progress click
        # sequence.
        self.state.bg_regions = [result.left_bg_region, result.right_bg_region]
        self.state.fit_region = result.fit_region
        self.state.peak_positions = [peak.position for peak in result.peaks]

        self.main_window.independent_widths_action.setChecked(not result.link_widths)
        self.main_window.left_tail_action.setChecked(result.tail_fraction is not None)

        names = parameter_names(
            len(result.peaks), result.link_widths, result.tail_fraction is not None
        )
        self._parameter_names_shown = []  # force a full rebuild below
        self.update_parameters_panel(names, fit_result_values_by_name(result))
        for row, name in enumerate(names):
            if name in result.fixed_params:
                self.parameters_table.cellWidget(row, 2).setChecked(True)

        self._redraw_progress()
        self.main_window._update_fit_mode_availability()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: double-click a Fit Results entry to reload it for editing"
```

---

## Self-Review Notes

- **Spec coverage:** Repeatability under changed checkboxes (Task 3), fixing any individual parameter (Tasks 2, 4, 5), complete per-parameter uncertainties including `0.0` for fixed ones (Task 2), disk-export field readiness via `fixed_params` on `FitResult` (Task 2, consumed later by sub-project 2), double-click reload restoring marks/checkboxes/fixed state (Task 6) -- all covered. "Clear" button's own bug fix and the "Integration" mode are explicitly out of scope per the spec, and untouched here.
- **Placeholder scan:** No TBD/TODO; every step has complete, runnable code.
- **Type consistency:** `fit_peaks(..., fixed_params=None)` (Task 2) matches its call site in `fit_mode.py`'s `run_fit()` (Task 5, `fixed_params=self.fixed_params_from_panel()`). `parameter_names`/`fit_result_values_by_name` (Task 2) are imported and used identically in Task 4's `run_fit()` update and Task 6's `_on_result_double_clicked`. `PeakResult.amplitude_err`/`sigma_err` (Task 2) are read in Task 4's implicit reliance on `fit_result_values_by_name` and asserted directly in Task 2's/Task 5's tests.
