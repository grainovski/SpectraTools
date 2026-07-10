# Peak Fitting Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v1's drag-based region marking with b/r/p+click (per `docs/superpowers/specs/2026-07-10-peak-fitting-refinement-design.md`), default all peaks in a fit region to a shared FWHM with an "Independent widths" opt-out, and add an optional gf3-style left-tail contribution to the peak shape.

**Architecture:** `peak_fit.py`'s `fit_peaks()` grows two new boolean parameters (`link_widths`, `enable_left_tail`) that select among four flat curve_fit parameter layouts, built/read by three small shared helpers (`_initial_guess`, `_unpack_params`, `_bounds`) so the layout logic lives in exactly one place. `fit_mode.py`'s `FitModeState` drops its rigid step machine for three independent click-pairing primitives (2-slot background ring buffer, 1-slot fit region, peak add/remove toggle); `FitModeController` becomes a `QObject` installing a Qt event filter on the canvas to track held b/r/p keys reliably, replacing matplotlib's own (documented-unreliable) `MouseEvent.key`. `main_window.py` loses the "Fit Peaks" toggle action and its nav-toolbar-disabling/scroll-guard logic, and gains two checkable toolbar actions read at fit time.

**Tech Stack:** Python 3.13, `scipy.optimize.curve_fit` and `scipy.special.erfc` (existing/new imports in `peak_fit.py`), `numpy`, PySide6/matplotlib (existing), `pytest` with the existing `qapp` fixture.

---

## Before you start

Read `docs/superpowers/specs/2026-07-10-peak-fitting-refinement-design.md` in full — every task below implements a specific part of it. The current (pre-refinement) contents of `peak_fit.py`, `fit_mode.py`, and `main_window.py` are reproduced inline in each task's steps as needed, but skim the live files first so the diffs make sense.

Run the full test suite once before starting to confirm a clean baseline:

Run: `pytest -v` from the worktree root.
Expected: all tests pass (77 at last count).

---

### Task 1: `hypermet_left_tail` — the tail-shape math, in isolation

**Files:**
- Modify: `peak_fit.py` (add near the top, after `FWHM_FACTOR`)
- Test: `tests/test_peak_fit.py`

This is gf3's exact Hypermet tail term (left-side only), ported from `srcRW/gf3_subs.c`'s `eval()` and verified numerically during design. It's written and tested standalone first since it's pure math with no dependency on the rest of the fitting machinery.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_peak_fit.py` (near the top, after the existing imports):

```python
from peak_fit import hypermet_left_tail


def test_hypermet_left_tail_biases_left_not_right():
    x = np.array([-10.0, 10.0])  # dx values either side of position=0
    values = hypermet_left_tail(x, position=0.0, sigma=3.0, r=0.1, beta=5.0)
    left_value, right_value = values
    assert left_value > right_value
    # hand-derived: ~14.7x; generous margin so this isn't brittle to
    # floating-point/erfc implementation differences
    assert left_value / right_value > 5.0


def test_hypermet_left_tail_reduces_to_gaussian_when_r_is_zero():
    x = np.array([-6.0, -2.0, 0.0, 3.0])
    values = hypermet_left_tail(x, position=1.0, sigma=2.0, r=0.0, beta=5.0)
    expected = np.exp(-((x - 1.0) ** 2) / (2 * 2.0 ** 2))
    np.testing.assert_allclose(values, expected, rtol=1e-10)


def test_hypermet_left_tail_handles_large_offsets_without_overflow():
    x = np.array([1000.0])  # dx/beta = 200, far past the safety threshold
    values = hypermet_left_tail(x, position=0.0, sigma=3.0, r=0.1, beta=5.0)
    assert np.all(np.isfinite(values))
    assert values[0] < 1e-6


def test_hypermet_left_tail_handles_degenerate_erfc_underflow():
    # sigma vastly larger than beta drives y = sigma/(beta*sqrt(2)) so
    # high that erfc(y) underflows to exactly 0.0 -- must fall back to
    # the plain Gaussian core instead of dividing by (effectively) zero.
    x = np.array([0.0, -5.0, 5.0])
    values = hypermet_left_tail(x, position=0.0, sigma=1000.0, r=0.1, beta=0.1)
    assert np.all(np.isfinite(values))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_peak_fit.py -v -k hypermet`
Expected: FAIL with `ImportError: cannot import name 'hypermet_left_tail'`

- [ ] **Step 3: Implement `hypermet_left_tail`**

Add to `peak_fit.py`, after the `FWHM_FACTOR` constant and before `class FitError`:

```python
from scipy.special import erfc

TAIL_FRACTION_MAX = 0.3
TAIL_BETA_MIN = 0.1


def hypermet_left_tail(x, position, sigma, r, beta):
    """gf3's Hypermet tail term (left-side only; the step-background
    term is out of scope, per the design spec):
        (1-r)*exp(-w^2) + r*exp(dx/beta)*erfc(w+y)/erfc(y)
    where dx = x - position, w = dx/(sigma*sqrt(2)),
    y = sigma/(beta*sqrt(2)). Ported from srcRW/gf3_subs.c's eval();
    verified numerically during design that beta > 0 biases the tail
    toward lower x (left), matching real low-energy detector tailing.
    Public (no leading underscore) because fit_mode.py's committed-fit
    overlay drawing reuses this exact formula to redraw tailed fits.
    """
    dx = x - position
    w = dx / (sigma * np.sqrt(2))
    gaussian_core = np.exp(-w ** 2)

    y = sigma / (beta * np.sqrt(2))
    erfc_y = erfc(y)
    if erfc_y < 1e-300:
        # y is so large that erfc(y) has underflowed to zero -- an
        # extreme, unphysical width/decay ratio that only an
        # unconverged optimizer iterate would ever produce. Treat the
        # tail as vanishing rather than divide by (effectively) zero.
        return (1 - r) * gaussian_core

    # Matches gf3's own overflow guard: exp(dx/beta) grows unbounded
    # for dx/beta > 0, but the true (mathematically bounded) tail
    # contribution there is negligible once |dx/beta| is large -- gf3
    # itself zeroes the tail term entirely past this same threshold
    # rather than risk exp() overflowing before erfc() can suppress it.
    ratio = dx / beta
    tail = np.zeros_like(x, dtype=float)
    safe = np.abs(ratio) <= 12.0
    z = w[safe] + y
    tail[safe] = np.exp(ratio[safe]) * erfc(z) / erfc_y

    return (1 - r) * gaussian_core + r * tail
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v -k hypermet`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "feat: add gf3 Hypermet left-tail shape function"
```

---

### Task 2: `FitResult` gains the new fields

**Files:**
- Modify: `peak_fit.py:19-25` (the `FitResult` dataclass)
- Test: `tests/test_peak_fit.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_peak_fit.py`, directly after `test_fit_result_holds_expected_fields`:

```python
def test_fit_result_new_fields_have_sensible_defaults():
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
    assert result.link_widths is True
    assert result.tail_fraction is None
    assert result.tail_fraction_err is None
    assert result.tail_beta is None
    assert result.tail_beta_err is None


def test_fit_result_accepts_explicit_tail_fields():
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
        link_widths=False,
        tail_fraction=0.1, tail_fraction_err=0.02,
        tail_beta=3.0, tail_beta_err=0.5,
    )
    assert result.link_widths is False
    assert result.tail_fraction == 0.1
    assert result.tail_beta == 3.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_peak_fit.py -v -k "new_fields or explicit_tail"`
Expected: FAIL with `TypeError: FitResult.__init__() got an unexpected keyword argument 'link_widths'`

- [ ] **Step 3: Extend the dataclass**

In `peak_fit.py`, replace:

```python
@dataclass
class FitResult:
    left_bg_region: tuple
    right_bg_region: tuple
    fit_region: tuple
    background_slope: float
    background_intercept: float
    peaks: list
```

with:

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS (all tests, including the two new ones and every pre-existing `FitResult(...)` construction call, which still work unchanged since the new fields all have defaults)

- [ ] **Step 5: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "feat: add link_widths/tail fields to FitResult"
```

---

### Task 3: `fit_peaks()` supports linked widths and the optional left tail

**Files:**
- Modify: `peak_fit.py` (replace `_gaussian_sum`, `_initial_guess`, and the body of `fit_peaks`)
- Test: `tests/test_peak_fit.py`

This is the core of the fitting-model change. `_gaussian_sum` is replaced by three helpers that all share one canonical parameter ordering: `(amplitude_0, position_0, [sigma_0], amplitude_1, position_1, [sigma_1], ..., [shared_sigma], [tail_fraction, tail_beta])` — the per-peak `sigma` is present only when `link_widths=False`, and the trailing pair only when `enable_left_tail=True`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_peak_fit.py`, after `test_fit_single_peak_with_poisson_noise`:

```python
def test_fit_independent_widths_recovers_different_sigmas():
    x, y = _make_spectrum(
        channels=200,
        peaks=[(400.0, 95.0, 2.0), (300.0, 108.0, 5.0)],
        slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(60.0, 75.0),
        right_bg_region=(130.0, 145.0),
        fit_region=(75.0, 130.0),
        peak_positions=[95.0, 108.0],
        link_widths=False,
    )
    assert result.link_widths is False
    sigmas = {round(p.position): p.sigma for p in result.peaks}
    assert sigmas[95] == pytest.approx(2.0, rel=0.2)
    assert sigmas[108] == pytest.approx(5.0, rel=0.2)


def test_fit_linked_widths_forces_equal_sigma_even_with_different_true_widths():
    x, y = _make_spectrum(
        channels=200,
        peaks=[(400.0, 95.0, 2.0), (300.0, 108.0, 5.0)],
        slope=0.0, intercept=20.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(60.0, 75.0),
        right_bg_region=(130.0, 145.0),
        fit_region=(75.0, 130.0),
        peak_positions=[95.0, 108.0],
        link_widths=True,
    )
    assert result.link_widths is True
    assert result.peaks[0].sigma == result.peaks[1].sigma
    assert result.peaks[0].fwhm == result.peaks[1].fwhm


def test_fit_default_link_widths_is_true():
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
    assert result.link_widths is True


def test_fit_without_left_tail_leaves_tail_fields_none():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
    )
    assert result.tail_fraction is None
    assert result.tail_beta is None


def test_fit_with_left_tail_enabled_recovers_known_tail_parameters():
    # Synthetic data WITH a real left tail baked in via the same
    # hypermet_left_tail() the fitter itself uses, so this verifies
    # fit_peaks() can recover known tail parameters, not just that the
    # option runs without crashing.
    x = np.arange(200, dtype=float)
    true_r, true_beta = 0.1, 4.0
    y = 20.0 + 500.0 * hypermet_left_tail(x, position=100.0, sigma=3.0, r=true_r, beta=true_beta)
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(80.0, 120.0),
        peak_positions=[100.0],
        enable_left_tail=True,
    )
    assert result.tail_fraction == pytest.approx(true_r, abs=0.05)
    assert result.tail_beta == pytest.approx(true_beta, rel=0.3)
    assert result.tail_fraction_err is not None
    assert result.tail_beta_err is not None


def test_fit_independent_widths_with_left_tail_both_enabled():
    # Exercises the 4th (3n+2) parameter-count combination.
    x = np.arange(200, dtype=float)
    y = (
        20.0
        + 400.0 * hypermet_left_tail(x, position=95.0, sigma=2.0, r=0.1, beta=4.0)
        + 300.0 * hypermet_left_tail(x, position=115.0, sigma=4.0, r=0.1, beta=4.0)
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(60.0, 75.0),
        right_bg_region=(140.0, 155.0),
        fit_region=(80.0, 130.0),
        peak_positions=[95.0, 115.0],
        link_widths=False,
        enable_left_tail=True,
    )
    assert len(result.peaks) == 2
    assert result.link_widths is False
    assert result.tail_fraction is not None
    sigmas = {round(p.position): p.sigma for p in result.peaks}
    assert sigmas[95] == pytest.approx(2.0, rel=0.3)
    assert sigmas[115] == pytest.approx(4.0, rel=0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_peak_fit.py -v -k "independent_widths or linked_widths or default_link or without_left_tail or with_left_tail_enabled or both_enabled"`
Expected: FAIL with `TypeError: fit_peaks() got an unexpected keyword argument 'link_widths'`

- [ ] **Step 3: Replace `_gaussian_sum`/`_initial_guess` and rewrite `fit_peaks()`**

In `peak_fit.py`, delete the existing `_gaussian_sum` function and `_initial_guess` function entirely, and replace them with:

```python
def _unpack_params(params, n_peaks, link_widths, enable_left_tail):
    """Splits a flat curve_fit parameter array into
    (amplitudes, positions, sigmas, tail_fraction, tail_beta). `sigmas`
    is always length n_peaks (the shared value repeated if linked);
    tail_fraction/tail_beta are None if enable_left_tail is False. This
    is the single canonical parameter ordering shared by the model
    function, the initial guess, the bounds, and result extraction --
    changing the layout means changing it only here."""
    idx = 0
    amplitudes = []
    positions = []
    sigmas = []
    for _ in range(n_peaks):
        amplitudes.append(params[idx]); idx += 1
        positions.append(params[idx]); idx += 1
        if not link_widths:
            sigmas.append(params[idx]); idx += 1
    if link_widths:
        shared_sigma = params[idx]; idx += 1
        sigmas = [shared_sigma] * n_peaks
    tail_fraction = None
    tail_beta = None
    if enable_left_tail:
        tail_fraction = params[idx]; idx += 1
        tail_beta = params[idx]; idx += 1
    return amplitudes, positions, sigmas, tail_fraction, tail_beta


def _make_model(n_peaks, link_widths, enable_left_tail):
    def model(x, *params):
        amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_params(
            params, n_peaks, link_widths, enable_left_tail
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


def _initial_guess(x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail):
    lo, hi = fit_region
    region_width = hi - lo
    n_peaks = len(peak_positions)
    sigma0 = max(region_width / (4 * n_peaks), 1e-6)

    guess = []
    for pos in peak_positions:
        idx = int(np.argmin(np.abs(x_fit - pos)))
        amplitude0 = float(y_sub[idx])
        guess.append(amplitude0)
        guess.append(float(pos))
        if not link_widths:
            guess.append(sigma0)
    if link_widths:
        guess.append(sigma0)
    if enable_left_tail:
        guess.append(0.05)
        guess.append(max(sigma0, TAIL_BETA_MIN))
    return guess


def _bounds(n_peaks, link_widths, enable_left_tail):
    """Only needed when enable_left_tail is True (to keep the tail
    fraction genuinely small and the decay constant away from zero);
    returns None otherwise so the no-tail fits keep using curve_fit's
    default unconstrained method, unchanged from v1's behavior."""
    if not enable_left_tail:
        return None
    lower = []
    upper = []
    for _ in range(n_peaks):
        lower.append(-np.inf); upper.append(np.inf)  # amplitude
        lower.append(-np.inf); upper.append(np.inf)  # position
        if not link_widths:
            lower.append(1e-6); upper.append(np.inf)  # sigma
    if link_widths:
        lower.append(1e-6); upper.append(np.inf)  # shared sigma
    lower.append(0.0); upper.append(TAIL_FRACTION_MAX)  # tail_fraction
    lower.append(TAIL_BETA_MIN); upper.append(np.inf)   # tail_beta
    return (lower, upper)
```

Then replace the entire body of `fit_peaks()` (keep the same `def fit_peaks(...)` signature line but extend its parameters, and replace everything below it):

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
    n_params = (
        2 * n_peaks
        + (1 if link_widths else n_peaks)
        + (2 if enable_left_tail else 0)
    )
    if x_fit.size < n_params:
        raise FitError(
            f"Fit region has {x_fit.size} data points, need at least "
            f"{n_params} for {n_peaks} peak(s)"
        )

    y_sub = y_fit - (slope * x_fit + intercept)
    p0 = _initial_guess(x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail)
    y_err = np.sqrt(np.maximum(y_fit, 1.0))
    model = _make_model(n_peaks, link_widths, enable_left_tail)
    bounds = _bounds(n_peaks, link_widths, enable_left_tail)

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
    amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_params(
        popt, n_peaks, link_widths, enable_left_tail
    )
    amplitude_errs, position_errs, sigma_errs, tail_fraction_err, tail_beta_err = _unpack_params(
        perr, n_peaks, link_widths, enable_left_tail
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

- [ ] **Step 4: Run the full peak_fit test file**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS (all tests — the pre-existing single-peak/multiplet/noise/error tests all still pass unchanged since `link_widths=True` with a single peak is behaviorally identical to the old unconditional per-peak sigma, and the multiplet test's two true peaks already share the same true sigma=3.0)

- [ ] **Step 5: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "feat: support linked/independent widths and optional left tail in fit_peaks"
```

---

### Task 4: `FitModeState` — ring buffer, single-slot region, peak toggle

**Files:**
- Modify: `fit_mode.py:1-68` (delete `STEP_*` constants and the old `FitModeState`, replace with the new one)
- Test: `tests/test_fit_mode.py` (full rewrite)

This is a pure, Qt-free rewrite of the marking state machine: no more rigid step sequence, just three independent click-pairing primitives.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_fit_mode.py` with:

```python
import pytest

from fit_mode import BG_REGION_CAP, FitModeState


def test_initial_state():
    state = FitModeState()
    assert state.bg_regions == []
    assert state.pending_bg_click is None
    assert state.fit_region is None
    assert state.pending_fit_click is None
    assert state.peak_positions == []


def test_bg_click_pairs_into_one_region():
    state = FitModeState()
    assert state.add_bg_click(70) is None
    assert state.pending_bg_click == 70
    completed = state.add_bg_click(85)
    assert completed == (70, 85)
    assert state.pending_bg_click is None
    assert state.bg_regions == [(70, 85)]


def test_bg_click_normalizes_reversed_order():
    state = FitModeState()
    state.add_bg_click(130)
    completed = state.add_bg_click(115)
    assert completed == (115, 130)


def test_bg_region_ring_buffer_evicts_oldest_after_two():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    assert state.bg_regions == [(70, 85), (115, 130)]

    state.add_bg_click(200)
    state.add_bg_click(210)
    assert state.bg_regions == [(115, 130), (200, 210)]


def test_fit_region_click_pairs_and_overwrites_on_new_pair():
    state = FitModeState()
    completed = state.add_fit_click(85)
    assert completed is None
    completed = state.add_fit_click(115)
    assert completed == (85, 115)
    assert state.fit_region == (85, 115)

    state.add_fit_click(90)
    state.add_fit_click(110)
    assert state.fit_region == (90, 110)


def test_peak_toggle_requires_a_fit_region():
    state = FitModeState()
    assert state.toggle_peak(100.0, proximity=1.0) is None
    assert state.peak_positions == []


def test_peak_toggle_rejects_position_outside_fit_region():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.toggle_peak(200.0, proximity=1.0) is None
    assert state.peak_positions == []


def test_peak_toggle_adds_then_removes():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)

    added = state.toggle_peak(100.0, proximity=1.0)
    assert added == ("added", 100.0)
    assert state.peak_positions == [100.0]

    removed = state.toggle_peak(100.4, proximity=1.0)
    assert removed == ("removed", 100.0)
    assert state.peak_positions == []


def test_peak_toggle_respects_proximity_threshold():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    state.toggle_peak(100.0, proximity=1.0)

    added = state.toggle_peak(105.0, proximity=1.0)
    assert added == ("added", 105.0)
    assert state.peak_positions == [100.0, 105.0]


def test_ready_to_fit_requires_two_bg_regions_a_fit_region_and_a_peak():
    state = FitModeState()
    assert state.ready_to_fit() is False
    state.add_bg_click(70)
    state.add_bg_click(85)
    assert state.ready_to_fit() is False
    state.add_bg_click(115)
    state.add_bg_click(130)
    assert state.ready_to_fit() is False
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.ready_to_fit() is False
    state.toggle_peak(100.0, proximity=1.0)
    assert state.ready_to_fit() is True


def test_reset_clears_everything():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.add_fit_click(85)
    state.add_fit_click(115)
    state.toggle_peak(100.0, proximity=1.0)

    state.reset()

    assert state.bg_regions == []
    assert state.fit_region is None
    assert state.peak_positions == []


def test_ordered_bg_regions_regardless_of_marking_order():
    state = FitModeState()
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.add_bg_click(70)
    state.add_bg_click(85)

    left, right = state.ordered_bg_regions()
    assert left == (70, 85)
    assert right == (115, 130)


def test_bg_region_cap_is_two():
    assert BG_REGION_CAP == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode.py -v`
Expected: FAIL with `ImportError: cannot import name 'BG_REGION_CAP'`

- [ ] **Step 3: Replace the state machine in `fit_mode.py`**

In `fit_mode.py`, delete lines 11-68 (the `STEP_*` constants through the end of the old `FitModeState` class) and replace with:

```python
BG_REGION_CAP = 2


class FitModeState:
    """Tracks in-progress background/fit-region/peak marks made via
    independent b/r/p click actions -- order-free, no Qt/matplotlib
    dependency. Each of add_bg_click/add_fit_click counts its own
    pending point independently, so interleaving a different key's
    clicks never disturbs an in-progress pair."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.bg_regions = []
        self.pending_bg_click = None
        self.fit_region = None
        self.pending_fit_click = None
        self.peak_positions = []

    def add_bg_click(self, x):
        """Returns the completed (lo, hi) region if this click
        completed a pair, else None (this click becomes the pending
        first point). A 3rd completed pair evicts the oldest region."""
        if self.pending_bg_click is None:
            self.pending_bg_click = x
            return None
        region = (min(self.pending_bg_click, x), max(self.pending_bg_click, x))
        self.pending_bg_click = None
        self.bg_regions.append(region)
        if len(self.bg_regions) > BG_REGION_CAP:
            self.bg_regions.pop(0)
        return region

    def add_fit_click(self, x):
        """Returns the completed (lo, hi) region if this click
        completed a pair, else None. A newly completed pair always
        replaces any existing fit region."""
        if self.pending_fit_click is None:
            self.pending_fit_click = x
            return None
        region = (min(self.pending_fit_click, x), max(self.pending_fit_click, x))
        self.pending_fit_click = None
        self.fit_region = region
        return region

    def toggle_peak(self, x, proximity):
        """Adds a peak at x, or removes an existing one within
        `proximity` of x. Returns ("added", x), ("removed", old_x), or
        None if there's no fit region yet or x falls outside it."""
        if self.fit_region is None:
            return None
        lo, hi = self.fit_region
        if not (lo <= x <= hi):
            return None
        for i, pos in enumerate(self.peak_positions):
            if abs(pos - x) <= proximity:
                del self.peak_positions[i]
                return ("removed", pos)
        self.peak_positions.append(x)
        return ("added", x)

    def ready_to_fit(self):
        return (
            len(self.bg_regions) == BG_REGION_CAP
            and self.fit_region is not None
            and len(self.peak_positions) > 0
        )

    def ordered_bg_regions(self):
        """Returns (left, right) background regions ordered by mean
        x-coordinate, regardless of which was marked first. Only valid
        once both regions exist."""
        a, b = self.bg_regions
        a_mid = (a[0] + a[1]) / 2
        b_mid = (b[0] + b[1]) / 2
        return (a, b) if a_mid <= b_mid else (b, a)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode.py
git commit -m "feat: replace step-sequence FitModeState with independent click-pairing primitives"
```

---

### Task 5: `FitModeController` — key-tracking event filter, click handling, no more mode toggle

**Files:**
- Modify: `fit_mode.py` (rewrite `FitModeController`; the class after the new `FitModeState`)
- Modify: `main_window.py:31,119-527` (several call sites; see below)
- Test: `tests/test_fit_mode_ui.py` (full rewrite)

This is the largest task: `FitModeController` becomes a `QObject` that installs a Qt event filter on the canvas to reliably track held b/r/p keys (matplotlib's own `MouseEvent.key` is documented as unreliable when the canvas lacked focus at key-press time), replaces the old drag-based `on_press`/`on_release` pair with a single `on_click` handler, and drops the `enabled`/`toggle()` mode entirely. `main_window.py` drops the "Fit Peaks" toggle action, the nav-toolbar-disabling logic, and the `_on_scroll` fit-mode guard; `_update_fit_mode_availability` is repurposed to gate the "Fit" button directly (combining "active spectrum is visible" with "marking is complete").

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_fit_mode_ui.py` with:

```python
import numpy as np
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from peak_fit import FitResult, PeakResult
from spectrum import LoadedSpectrum

_QT_KEY = {"b": Qt.Key.Key_B, "r": Qt.Key.Key_R, "p": Qt.Key.Key_P}


def _click(main_window, xdata, ydata=10.0):
    ax = main_window.axes
    px, py = ax.transData.transform((xdata, ydata))
    event = MouseEvent("button_press_event", main_window.canvas, px, py, button=1)
    main_window.canvas.callbacks.process("button_press_event", event)


def _dispatch(main_window, name, xdata, ydata=10.0):
    ax = main_window.axes
    px, py = ax.transData.transform((xdata, ydata))
    event = MouseEvent(name, main_window.canvas, px, py, button=1)
    main_window.canvas.callbacks.process(name, event)


def _held_key_click(main_window, key_char, xdata, ydata=10.0):
    """Directly sets the controller's held-key state (bypassing real Qt
    key events) then simulates a mouse click -- the fast, simple way
    most tests exercise "given this key is held, what does a click do."
    test_key_event_filter_tracks_held_key below separately verifies the
    actual Qt event-filter wiring that sets this state in real usage."""
    main_window.fit_controller._held_key = key_char
    _click(main_window, xdata, ydata)
    main_window.fit_controller._held_key = None


def _make_active_spectrum(main_window):
    y = np.full(200, 20, dtype=np.int64)
    y[97:104] += (
        500 * np.exp(-((np.arange(97, 104) - 100.0) ** 2) / (2 * 3.0 ** 2))
    ).astype(np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    main_window.spectra.append(spectrum)
    main_window._plot_data()
    return spectrum


def test_key_event_filter_tracks_held_key(qapp):
    main_window = MainWindow()
    canvas = main_window.canvas

    press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_B, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(canvas, press)
    assert main_window.fit_controller._held_key == "b"

    release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_B, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(canvas, release)
    assert main_window.fit_controller._held_key is None


def test_full_fit_flow_commits_a_fit_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    assert main_window.fit_button.isEnabled() is False
    _held_key_click(main_window, "p", 100)
    assert main_window.fit_button.isEnabled() is True

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    result = spectrum.fits[0]
    assert len(result.peaks) == 1
    assert abs(result.peaks[0].position - 100.0) < 1.0


def test_marking_order_is_free(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    # fit region and a peak marked before either background region
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    assert main_window.fit_button.isEnabled() is True

    main_window.fit_controller.run_fit()
    assert len(spectrum.fits) == 1


def test_bg_region_ring_buffer_eviction_via_clicks(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    assert main_window.fit_controller.state.bg_regions == [(70.0, 85.0), (115.0, 130.0)]

    _held_key_click(main_window, "b", 150)
    _held_key_click(main_window, "b", 160)
    assert main_window.fit_controller.state.bg_regions == [(115.0, 130.0), (150.0, 160.0)]


def test_peak_click_toggle_adds_and_removes(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    assert main_window.fit_controller.state.peak_positions == [100.0]

    _held_key_click(main_window, "p", 100)  # same spot -- removes it
    assert main_window.fit_controller.state.peak_positions == []


def test_peak_click_outside_fit_region_is_rejected_with_a_hint(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 200)

    assert main_window.fit_controller.state.peak_positions == []
    assert main_window.statusBar().currentMessage() != ""


def test_plot_data_preserve_view_keeps_current_zoom(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    main_window.axes.set_xlim(80, 120)
    main_window._plot_data(preserve_view=True)

    assert main_window.axes.get_xlim() == (80.0, 120.0)


def test_run_fit_preserves_the_current_zoom(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.axes.set_xlim(80, 120)

    _held_key_click(main_window, "b", 81)
    _held_key_click(main_window, "b", 86)
    _held_key_click(main_window, "b", 114)
    _held_key_click(main_window, "b", 119)
    _held_key_click(main_window, "r", 90)
    _held_key_click(main_window, "r", 110)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    assert main_window.axes.get_xlim() == (80.0, 120.0)


def test_clear_discards_in_progress_marks_without_committing(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.clear()

    assert main_window.fit_controller.state.bg_regions == []
    assert main_window.fit_button.isEnabled() is False
    assert len(spectrum.fits) == 0


def test_fit_button_disabled_when_active_spectrum_hidden(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)
    assert main_window.fit_button.isEnabled() is True

    spectrum.visible = False
    main_window._update_fit_mode_availability()

    assert main_window.fit_button.isEnabled() is False


def test_fit_failure_leaves_marks_intact_and_shows_message(qapp):
    # Fit region deliberately far too narrow (1 data point) for the 2
    # peaks marked in it -- fit_peaks() raises FitError for "too few
    # data points", which run_fit() must catch, show as a status-bar
    # message, and NOT commit a result or reset the in-progress marks.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 99.5)
    _held_key_click(main_window, "r", 100.5)
    _held_key_click(main_window, "p", 99.6)
    _held_key_click(main_window, "p", 100.4)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 0
    assert main_window.fit_controller.state.peak_positions == [99.6, 100.4]
    assert main_window.statusBar().currentMessage() != ""


def test_status_message_not_immediately_clobbered_by_mouse_move(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "p", 100)  # no fit region yet -- rejected with a hint

    message_after_hint = main_window.statusBar().currentMessage()
    assert message_after_hint != ""

    _dispatch(main_window, "motion_notify_event", 50)

    assert main_window.statusBar().currentMessage() == message_after_hint


def test_plot_data_draws_committed_fit_overlay(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0),
            right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 110.0),
            background_slope=0.0,
            background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1,
                    fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0,
                    amplitude=200.0, sigma=2.0,
                )
            ],
        )
    )
    main_window.spectra.append(spectrum)

    main_window._plot_data()

    assert len(main_window.axes.patches) == 3
    assert len(main_window.axes.lines) == 4


def _fit_result_with_one_peak():
    return FitResult(
        left_bg_region=(10.0, 20.0),
        right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0),
        background_slope=0.0,
        background_intercept=20.0,
        peaks=[
            PeakResult(
                position=100.0, position_err=0.1,
                fwhm=5.0, fwhm_err=0.2,
                area=1000.0, area_err=50.0,
                amplitude=200.0, sigma=2.0,
            )
        ],
    )


def test_results_panel_lists_committed_fit(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    spectrum.fits.append(_fit_result_with_one_peak())
    main_window.spectra.append(spectrum)

    main_window.fit_controller.update_results_list()

    assert main_window.fit_controller.results_list.count() == 1
    text = main_window.fit_controller.results_list.item(0).text()
    assert "100.00" in text
    assert "5.00" in text


def test_results_panel_updates_when_active_spectrum_changes(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum_a = LoadedSpectrum("a.txt", y, "#1f77b4")
    spectrum_a.active = True
    spectrum_a.fits.append(_fit_result_with_one_peak())
    spectrum_b = LoadedSpectrum("b.txt", y, "#ff7f0e")
    main_window.spectra.extend([spectrum_a, spectrum_b])
    main_window.fit_controller.update_results_list()
    assert main_window.fit_controller.results_list.count() == 1

    main_window._on_active_toggled("b.txt", True)

    assert main_window.fit_controller.results_list.count() == 0


def test_clear_all_fits_empties_panel_and_spectrum(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    spectrum.fits.append(_fit_result_with_one_peak())
    main_window.spectra.append(spectrum)
    main_window.fit_controller.update_results_list()

    spectrum.fits.clear()
    main_window._plot_data()
    main_window.fit_controller.update_results_list()

    assert main_window.fit_controller.results_list.count() == 0


def test_clear_progress_survives_an_intervening_full_replot(qapp):
    # Reproduces a real crash: mark one background region (creating an
    # in-progress artist), then something else triggers a full
    # axes.clear() (in the real app: the results panel's "Remove
    # Fit"/"Clear All Fits" context menu calling _plot_data() while a
    # fit is still mid-marking) before the in-progress fit is finished.
    # Finishing it afterward (Clear, here) must not raise.
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)

    main_window._plot_data()

    main_window.fit_controller.clear()

    assert main_window.fit_controller.state.bg_regions == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: FAIL (e.g. `AttributeError: 'MainWindow' object has no attribute 'fit_mode_action'` no longer exists is fine since it's not referenced anymore, but tests will fail against the still-old `FitModeController`/`FitModeState`/`main_window.py` — expect import or AttributeError failures such as `AttributeError: 'FitModeController' object has no attribute '_held_key'`)

- [ ] **Step 3: Rewrite `FitModeController` in `fit_mode.py`**

Replace the entire `FitModeController` class in `fit_mode.py` (everything from `class FitModeController:` to the end of the file, i.e. through the old `run_fit`) with:

```python
PEAK_CLICK_PIXEL_PROXIMITY = 8

_KEY_TO_MARK_TYPE = {
    Qt.Key.Key_B: "b",
    Qt.Key.Key_R: "r",
    Qt.Key.Key_P: "p",
}

_MARK_TYPE_LABEL = {
    "b": "background region",
    "r": "fit region",
    "p": "peak",
}


class FitModeController(QObject):
    """Qt/matplotlib-facing wrapper around FitModeState: tracks which of
    b/r/p is currently held via a Qt event filter on the canvas (not
    matplotlib's own MouseEvent.key, which is documented as unreliable
    if the canvas lacked focus when the key was pressed), owns the
    in-progress marking artists, draws committed fits, and drives
    fit_peaks() when the user clicks "Fit". There is no exclusive
    "fit mode" -- marking is always available alongside normal
    pan/zoom, since b/r/p+click never collides with a plain click-drag."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.state = FitModeState()
        self._held_key = None
        self._progress_artists = []
        self._status_message_until = 0.0

        canvas = main_window.canvas
        canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        canvas.installEventFilter(self)
        canvas.mpl_connect("figure_enter_event", lambda event: canvas.setFocus())

    def eventFilter(self, obj, event):
        if obj is self.main_window.canvas:
            if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                mark_type = _KEY_TO_MARK_TYPE.get(event.key())
                if mark_type is not None:
                    self._held_key = mark_type
                    self._show_status_message(
                        f"Marking {_MARK_TYPE_LABEL[mark_type]}: click to place", 60000
                    )
            elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
                mark_type = _KEY_TO_MARK_TYPE.get(event.key())
                if mark_type is not None and self._held_key == mark_type:
                    self._held_key = None
        return False

    def _show_status_message(self, message, duration_ms):
        self.main_window.statusBar().showMessage(message, duration_ms)
        self._status_message_until = time.monotonic() + duration_ms / 1000.0

    def clear(self):
        self._clear_progress()
        self.main_window._update_fit_mode_availability()
        self.main_window.canvas.draw_idle()

    def _clear_progress(self):
        for artist in self._progress_artists:
            try:
                artist.remove()
            except NotImplementedError:
                # The artist may already have been invalidated by an
                # unrelated full-axes clear (main_window._plot_data(),
                # e.g. triggered by the results panel's context menu
                # while a fit was still being marked) -- matplotlib's
                # Axes.clear() sets an artist's _remove_method to None
                # for every child it had, making a later .remove() call
                # on that same (now-stale) reference raise this. Since
                # the artist is already gone from the axes either way,
                # there's nothing left to do for it here.
                pass
        self._progress_artists = []
        self.state.reset()

    def _redraw_progress(self):
        """Clears and fully rebuilds every in-progress marking artist
        from the current FitModeState fields. Trades a little redundant
        redraw work (never more than a handful of artists) for avoiding
        any incremental per-artist bookkeeping -- no risk of a stale
        artist left behind by an evicted background region or a
        removed peak."""
        for artist in self._progress_artists:
            try:
                artist.remove()
            except NotImplementedError:
                pass
        self._progress_artists = []

        axes = self.main_window.axes
        state = self.state

        if state.pending_bg_click is not None:
            self._progress_artists.append(
                axes.axvline(state.pending_bg_click, color="gray", linestyle="--", linewidth=1)
            )
        for region in state.bg_regions:
            self._progress_artists.append(axes.axvspan(*region, color="gray", alpha=0.15))

        if state.pending_fit_click is not None:
            self._progress_artists.append(
                axes.axvline(state.pending_fit_click, color="tab:blue", linestyle="--", linewidth=1)
            )
        if state.fit_region is not None:
            self._progress_artists.append(
                axes.axvspan(*state.fit_region, color="tab:blue", alpha=0.1)
            )

        for x in state.peak_positions:
            self._progress_artists.append(
                axes.axvline(x, color="red", linestyle=":", linewidth=1)
            )

        self.main_window.canvas.draw_idle()

    def _pixel_proximity_to_data(self, event):
        """Converts the fixed PEAK_CLICK_PIXEL_PROXIMITY pixel threshold
        into a data-coordinate distance at the current zoom level, so
        the "close enough to hit an existing peak" tolerance stays
        visually consistent regardless of zoom."""
        axes = self.main_window.axes
        inverse = axes.transData.inverted()
        x0 = inverse.transform((event.x, event.y))[0]
        x1 = inverse.transform((event.x + PEAK_CLICK_PIXEL_PROXIMITY, event.y))[0]
        return abs(x1 - x0)

    def on_click(self, event):
        if event.inaxes != self.main_window.axes or event.xdata is None:
            return
        if event.button != 1:
            return
        key = self._held_key
        if key == "b":
            self.state.add_bg_click(event.xdata)
            self._redraw_progress()
        elif key == "r":
            self.state.add_fit_click(event.xdata)
            self._redraw_progress()
        elif key == "p":
            proximity = self._pixel_proximity_to_data(event)
            result = self.state.toggle_peak(event.xdata, proximity)
            if result is None:
                self._show_status_message(
                    "Mark the fit region (hold R and click twice) before marking peaks", 3000
                )
                return
            self._redraw_progress()
        else:
            return
        self.main_window._update_fit_mode_availability()

    def draw_committed_fits(self, spectrum):
        axes = self.main_window.axes
        # x in data coordinates, y in axes-fraction -- keeps peak labels
        # pinned near the top of the visible plot regardless of the
        # current y-axis scale (linear or log) or zoom level.
        label_transform = axes.get_xaxis_transform()
        for result in spectrum.fits:
            axes.axvspan(*result.left_bg_region, color="gray", alpha=0.15)
            axes.axvspan(*result.right_bg_region, color="gray", alpha=0.15)
            axes.axvspan(*result.fit_region, color="tab:blue", alpha=0.1)

            lo, hi = result.fit_region
            background_lo = result.background_slope * lo + result.background_intercept
            background_hi = result.background_slope * hi + result.background_intercept
            axes.plot([lo, hi], [background_lo, background_hi], color="black",
                       linestyle="--", linewidth=1)

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

            for peak in result.peaks:
                axes.axvline(peak.position, color="red", linestyle=":", linewidth=1)
                axes.annotate(
                    f"pos={peak.position:.1f}\nFWHM={peak.fwhm:.1f}\nvol={peak.area:.0f}",
                    xy=(peak.position, 0.95),
                    xycoords=label_transform,
                    ha="center", va="top",
                    fontsize=7, color="red",
                )

    def build_results_panel(self):
        mw = self.main_window
        self.results_list = QListWidget()
        self.results_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_list.customContextMenuRequested.connect(self._on_results_context_menu)

        self.results_dock = QDockWidget("Fit Results", mw)
        self.results_dock.setWidget(self.results_list)
        mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.results_dock)

        self.toggle_results_panel_action = QAction("Fit Results", mw)
        self.toggle_results_panel_action.setCheckable(True)
        self.toggle_results_panel_action.setToolTip("Show/hide fit results")
        self.toggle_results_panel_action.toggled.connect(self.results_dock.setVisible)
        self.results_dock.visibilityChanged.connect(
            self.toggle_results_panel_action.setChecked
        )
        self.results_dock.setVisible(False)

        results_tab_bar = QToolBar("Fit Results Tab", mw)
        results_tab_bar.setMovable(False)
        results_tab_bar.setFloatable(False)
        results_tab_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        results_tab_bar.addAction(self.toggle_results_panel_action)
        mw.addToolBar(Qt.ToolBarArea.RightToolBarArea, results_tab_bar)

    def update_results_list(self):
        self.results_list.clear()
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        for result in active.fits:
            header = f"Fit region [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]"
            if not result.link_widths:
                header += "  (independent widths)"
            if result.tail_fraction is not None:
                header += (
                    f"  (left tail: r={result.tail_fraction:.2f}"
                    f"±{result.tail_fraction_err:.2f}, "
                    f"β={result.tail_beta:.1f}±{result.tail_beta_err:.1f})"
                )
            lines = [header]
            for i, peak in enumerate(result.peaks, start=1):
                lines.append(
                    f"  Peak {i}: pos={peak.position:.2f}±{peak.position_err:.2f}  "
                    f"FWHM={peak.fwhm:.2f}±{peak.fwhm_err:.2f}  "
                    f"volume={peak.area:.1f}±{peak.area_err:.1f}"
                )
            self.results_list.addItem(QListWidgetItem("\n".join(lines)))

    def _on_results_context_menu(self, position):
        mw = self.main_window
        item = self.results_list.itemAt(position)
        menu = QMenu(mw)
        remove_action = menu.addAction("Remove Fit") if item is not None else None
        clear_action = menu.addAction("Clear All Fits")
        chosen = menu.exec(self.results_list.viewport().mapToGlobal(position))
        active = next((s for s in mw.spectra if s.active), None)
        if active is None:
            return
        if item is not None and chosen == remove_action:
            index = self.results_list.row(item)
            del active.fits[index]
            mw._plot_data(preserve_view=True)
        elif chosen == clear_action:
            active.fits.clear()
            mw._plot_data(preserve_view=True)

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
        self._clear_progress()
        self.main_window._plot_data(preserve_view=True)
```

Also update the top of `fit_mode.py` — replace the existing imports:

```python
import time

import numpy as np
from matplotlib.backend_bases import _Mode
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMenu, QToolBar

from peak_fit import FitError, fit_peaks
```

with:

```python
import time

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMenu, QToolBar

from peak_fit import FitError, fit_peaks, hypermet_left_tail
```

(`_Mode` is no longer used now that the nav-toolbar-disabling logic in `toggle()` is gone; `QObject`/`QEvent` are new; `hypermet_left_tail` is needed by `draw_committed_fits`.)

- [ ] **Step 4: Update `main_window.py`**

In `main_window.py`, replace:

```python
        self.fit_controller = FitModeController(self)
        self._build_fit_mode_buttons()
        self.fit_controller.build_results_panel()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._update_fit_mode_availability()
```

with:

```python
        self.fit_controller = FitModeController(self)
        self._build_fit_mode_buttons()
        self.fit_controller.build_results_panel()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self._update_fit_mode_availability()
```

Replace the `_build_fit_mode_buttons` method:

```python
    def _build_fit_mode_buttons(self):
        self.fit_toolbar = QToolBar("Fit Peaks", self)
        self.fit_toolbar.setMovable(False)

        self.fit_mode_action = QAction("Fit Peaks", self)
        self.fit_mode_action.setCheckable(True)
        self.fit_mode_action.toggled.connect(self.fit_controller.toggle)
        self.fit_toolbar.addAction(self.fit_mode_action)

        self.fit_button = QAction("Fit", self)
        self.fit_button.setEnabled(False)
        self.fit_button.triggered.connect(self.fit_controller.run_fit)
        self.fit_toolbar.addAction(self.fit_button)

        self.clear_fit_button = QAction("Clear", self)
        self.clear_fit_button.setEnabled(False)
        self.clear_fit_button.triggered.connect(self.fit_controller.clear)
        self.fit_toolbar.addAction(self.clear_fit_button)

        self.addToolBar(self.fit_toolbar)
```

with:

```python
    def _build_fit_mode_buttons(self):
        self.fit_toolbar = QToolBar("Fit Peaks", self)
        self.fit_toolbar.setMovable(False)

        self.independent_widths_action = QAction("Independent widths", self)
        self.independent_widths_action.setCheckable(True)
        self.independent_widths_action.setToolTip(
            "Fit each peak's width independently instead of sharing one FWHM"
        )
        self.fit_toolbar.addAction(self.independent_widths_action)

        self.left_tail_action = QAction("Left tail", self)
        self.left_tail_action.setCheckable(True)
        self.left_tail_action.setToolTip(
            "Allow a small low-channel tail contribution to each peak's shape"
        )
        self.fit_toolbar.addAction(self.left_tail_action)

        self.fit_button = QAction("Fit", self)
        self.fit_button.setEnabled(False)
        self.fit_button.triggered.connect(self.fit_controller.run_fit)
        self.fit_toolbar.addAction(self.fit_button)

        self.clear_fit_button = QAction("Clear", self)
        self.clear_fit_button.triggered.connect(self.fit_controller.clear)
        self.fit_toolbar.addAction(self.clear_fit_button)

        self.addToolBar(self.fit_toolbar)
```

Replace:

```python
    def _on_canvas_press(self, event):
        self.fit_controller.on_press(event)

    def _on_canvas_release(self, event):
        self.fit_controller.on_release(event)

    def _update_fit_mode_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        available = active is not None and active.visible
        self.fit_mode_action.setEnabled(available)
        if not available and self.fit_mode_action.isChecked():
            self.fit_mode_action.setChecked(False)

    def _on_scroll(self, event):
        if self.fit_controller.enabled:
            return
        if event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)
```

with:

```python
    def _on_canvas_click(self, event):
        self.fit_controller.on_click(event)

    def _update_fit_mode_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        available = active is not None and active.visible
        self.fit_button.setEnabled(available and self.fit_controller.state.ready_to_fit())

    def _on_scroll(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)
```

- [ ] **Step 5: Run the fit-mode tests**

Run: `pytest tests/test_fit_mode_ui.py tests/test_fit_mode.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: PASS (no regressions in `test_peak_fit.py`, `test_spe_io.py`, `test_spk_io.py`, or any other existing test file)

- [ ] **Step 7: Commit**

```bash
git add fit_mode.py main_window.py tests/test_fit_mode_ui.py
git commit -m "feat: replace drag-based fit-mode toggle with always-on b/r/p+click marking"
```

---

### Task 6: Checkboxes wired into `run_fit()`, tail-aware overlay drawing and results text

Task 5 already wired `independent_widths_action`/`left_tail_action` into `run_fit()` and made `draw_committed_fits`/`update_results_list` tail-aware, since those changes were entangled with the same files being rewritten. This task adds the tests confirming that wiring actually works end-to-end.

**Files:**
- Test: `tests/test_fit_mode_ui.py` (append)
- Test: `tests/test_peak_fit.py` (no changes needed -- covered by Task 3)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fit_mode_ui.py`:

```python
def test_independent_widths_checkbox_is_passed_to_fit_peaks(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.independent_widths_action.setChecked(True)
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    assert spectrum.fits[0].link_widths is False


def test_left_tail_checkbox_is_passed_to_fit_peaks(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.left_tail_action.setChecked(True)
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    assert spectrum.fits[0].tail_fraction is not None


def test_plot_data_draws_committed_fit_overlay_with_left_tail(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0),
            right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 110.0),
            background_slope=0.0,
            background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1,
                    fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0,
                    amplitude=200.0, sigma=2.0,
                )
            ],
            link_widths=True,
            tail_fraction=0.1, tail_fraction_err=0.02,
            tail_beta=3.0, tail_beta_err=0.5,
        )
    )
    main_window.spectra.append(spectrum)

    main_window._plot_data()  # must not raise

    assert len(main_window.axes.patches) == 3
    assert len(main_window.axes.lines) == 4


def test_results_panel_shows_tail_and_width_link_info(qapp):
    main_window = MainWindow()
    y = np.full(200, 20, dtype=np.int64)
    spectrum = LoadedSpectrum("synthetic.txt", y, "#1f77b4")
    spectrum.active = True
    spectrum.fits.append(
        FitResult(
            left_bg_region=(10.0, 20.0),
            right_bg_region=(180.0, 190.0),
            fit_region=(90.0, 110.0),
            background_slope=0.0,
            background_intercept=20.0,
            peaks=[
                PeakResult(
                    position=100.0, position_err=0.1,
                    fwhm=5.0, fwhm_err=0.2,
                    area=1000.0, area_err=50.0,
                    amplitude=200.0, sigma=2.0,
                )
            ],
            link_widths=False,
            tail_fraction=0.1, tail_fraction_err=0.02,
            tail_beta=3.0, tail_beta_err=0.5,
        )
    )
    main_window.spectra.append(spectrum)

    main_window.fit_controller.update_results_list()

    text = main_window.fit_controller.results_list.item(0).text()
    assert "independent widths" in text
    assert "r=0.10" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -v -k "checkbox or with_left_tail or tail_and_width"`
Expected: FAIL — most likely `AttributeError: 'MainWindow' object has no attribute 'independent_widths_action'` if Task 5 wasn't yet applied, or a real assertion failure if Task 5's wiring has a bug (this task exists specifically to prove that wiring). If Task 5 was completed correctly, these should already PASS on the first run; if so, treat this step as verification rather than TDD red, and proceed to step 3 as a no-op confirmation.

- [ ] **Step 3: Confirm the implementation (already in place from Task 5)**

No production code changes needed if Task 5 was applied as written above — `run_fit()` already reads `independent_widths_action`/`left_tail_action`, `draw_committed_fits()` already branches on `result.tail_fraction`, and `update_results_list()` already includes the width-link/tail text. If any test above fails, re-check the corresponding block in `fit_mode.py` from Task 5 against what's shown there.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add tests/test_fit_mode_ui.py
git commit -m "test: cover width-link/left-tail checkbox wiring end-to-end"
```

---

### Task 7: Rebuild the Windows installer

**Files:** none (build only)

The refinement doesn't add any new third-party dependency beyond what v1 already introduced (`scipy`, already present), so no packaging file changes are expected — this task just rebuilds and smoke-tests the installer against the refined code.

- [ ] **Step 1: Rebuild**

Run: `packaging/windows/build.ps1` (or the project's existing build entry point — check `packaging/windows/` for the exact script name used in the v1 plan's Task 11 if this differs)

- [ ] **Step 2: Smoke-test**

Launch the built app from its installed location (or run `python main_window.py` from source if the installer's UAC prompt can't be completed non-interactively in this environment, per the known outstanding issue from the v1 plan) and manually verify:
- Hold `b`, click twice, see a shaded gray region appear
- Hold `r`, click twice, see a shaded blue region appear
- Hold `p`, click once inside it, see a red dashed marker appear; click the same spot again, see it disappear
- Click "Fit" — a result appears in the plot and the "Fit Results" panel
- Toggle "Independent widths" and "Left tail", fit again, confirm no crash and the results panel shows the extra info line

- [ ] **Step 3: Report back**

If the installer's UAC elevation still can't be completed automatically in this environment (as in the v1 plan's Task 11), report this transparently and ask the user to verify the rebuilt installer themselves, same as before.

---

## Self-Review Notes

- **Spec coverage:** Interaction model (Task 4+5: ring buffer, single-slot fit region, peak toggle with pixel proximity, no mode toggle, event-filter key tracking with a real-`QKeyEvent` test) — covered. Fitting model (Task 1: tail formula; Task 3: linked/independent widths, all 4 parameter-count combinations, bounds on `r`/`β`) — covered. Data model (Task 2: new `FitResult` fields) — covered. UI changes (Task 5: toolbar action removal/addition; Task 6: checkbox wiring, tail-aware drawing, results text) — covered. Out-of-scope items (right tail, step term, per-peak overrides, checkbox persistence, auto peak-finding) — none implemented, as intended.
- **Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code.
- **Type consistency:** `FitModeState.add_bg_click`/`add_fit_click`/`toggle_peak` signatures used identically in Task 4's tests and Task 5's `FitModeController.on_click`. `fit_peaks(..., link_widths=, enable_left_tail=)` signature from Task 3 matches its usage in Task 5's `run_fit()`. `hypermet_left_tail` (public, no underscore) defined in Task 1, imported and reused unchanged in Task 5's `draw_committed_fits` and Task 3's tests.
