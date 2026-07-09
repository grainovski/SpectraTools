# Peak Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add interactive Gaussian peak fitting to SpectraTools — manual background/fit-region marking plus per-peak clicks, driving a `scipy.optimize.curve_fit` fit against the Active spectrum, with results drawn on the plot and listed in a new results panel.

**Architecture:** A new pure-logic module `peak_fit.py` (background computation, Gaussian model, initial guesses, the fit itself, uncertainty propagation — no Qt dependency, thoroughly unit-tested) and a new `fit_mode.py` (the interactive state machine, mouse-event handling, plot-overlay drawing, and the results-panel widget), wired into the existing `main_window.py`.

**Tech Stack:** Python 3.13, `scipy.optimize.curve_fit` (new dependency), `numpy`, PySide6/matplotlib (existing), `pytest` (existing tests) plus a new `qapp` fixture for the Qt-dependent tests in this plan.

**Spec:** `docs/superpowers/specs/2026-07-09-peak-fitting-design.md` — read this first; this plan implements it, not re-derives it.

---

### Task 1: Add the `scipy` dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add scipy to requirements.txt**

Current content:
```
# 6.11.x fails with "DLL load failed while importing QtWidgets" on Windows
PySide6>=6.6,<6.9
matplotlib>=3.8
numpy>=1.26
```
Change to:
```
# 6.11.x fails with "DLL load failed while importing QtWidgets" on Windows
PySide6>=6.6,<6.9
matplotlib>=3.8
numpy>=1.26
scipy>=1.11
```

- [ ] **Step 2: Install it into the project venv**

Run: `.venv/Scripts/python.exe -m pip install "scipy>=1.11"`
Expected: installs successfully (no output errors).

- [ ] **Step 3: Verify the import works**

Run: `.venv/Scripts/python.exe -c "from scipy.optimize import curve_fit; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Add scipy dependency for peak fitting"
```

---

### Task 2: `peak_fit.py` — dataclasses and `FitError`

**Files:**
- Create: `peak_fit.py`
- Test: `tests/test_peak_fit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_peak_fit.py`:

```python
from peak_fit import FitError, FitResult, PeakResult


def test_peak_result_holds_expected_fields():
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=5.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=2.0,
    )
    assert peak.position == 100.0
    assert peak.fwhm == 5.0
    assert peak.area == 1000.0


def test_fit_result_holds_expected_fields():
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
    assert result.fit_region == (90.0, 110.0)
    assert result.peaks == [peak]


def test_fit_error_is_an_exception():
    assert issubclass(FitError, Exception)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_peak_fit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'peak_fit'`

- [ ] **Step 3: Write the implementation**

Create `peak_fit.py`:

```python
from dataclasses import dataclass


class FitError(Exception):
    """Raised when a fit cannot be performed or does not converge."""


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


@dataclass
class FitResult:
    left_bg_region: tuple
    right_bg_region: tuple
    fit_region: tuple
    background_slope: float
    background_intercept: float
    peaks: list
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "Add peak_fit.py data model (PeakResult, FitResult, FitError)"
```

---

### Task 3: `peak_fit.py` — background computation

**Files:**
- Modify: `peak_fit.py`
- Modify: `tests/test_peak_fit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_peak_fit.py`:

```python
import numpy as np
import pytest

from peak_fit import _compute_background


def test_compute_background_flat():
    x = np.arange(200, dtype=float)
    y = np.full(200, 15.0)
    slope, intercept = _compute_background(x, y, (10.0, 20.0), (150.0, 160.0))
    assert slope == pytest.approx(0.0, abs=1e-9)
    assert intercept == pytest.approx(15.0, abs=1e-9)


def test_compute_background_sloped():
    x = np.arange(200, dtype=float)
    y = 0.5 * x + 3.0
    slope, intercept = _compute_background(x, y, (10.0, 20.0), (150.0, 160.0))
    assert slope == pytest.approx(0.5, abs=1e-6)
    assert intercept == pytest.approx(3.0, abs=1e-4)


def test_compute_background_rejects_empty_region():
    x = np.arange(200, dtype=float)
    y = np.full(200, 15.0)
    with pytest.raises(FitError):
        _compute_background(x, y, (500.0, 501.0), (150.0, 160.0))


def test_compute_background_rejects_identical_mean_x():
    x = np.arange(200, dtype=float)
    y = np.full(200, 15.0)
    with pytest.raises(FitError):
        _compute_background(x, y, (10.0, 20.0), (10.0, 20.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_peak_fit.py -v -k compute_background`
Expected: FAIL with `ImportError: cannot import name '_compute_background'`

- [ ] **Step 3: Add the implementation**

Add to `peak_fit.py` (needs `numpy` — add the import at the top of the file alongside the existing `dataclasses` import):

```python
from dataclasses import dataclass

import numpy as np
```

Then append at the end of the file:

```python
def _region_centroid(x, y, region):
    lo, hi = region
    mask = (x >= lo) & (x <= hi)
    if not np.any(mask):
        raise FitError(f"Background region {region} contains no data")
    return float(np.mean(x[mask])), float(np.mean(y[mask]))


def _compute_background(x, y, left_bg_region, right_bg_region):
    left_x, left_y = _region_centroid(x, y, left_bg_region)
    right_x, right_y = _region_centroid(x, y, right_bg_region)
    if right_x == left_x:
        raise FitError("Background regions must not share the same mean channel")
    slope = (right_y - left_y) / (right_x - left_x)
    intercept = left_y - slope * left_x
    return slope, intercept
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v`
Expected: all PASS (7 tests: 3 from Task 2 + 4 new)

- [ ] **Step 5: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "Add fixed-line background computation to peak_fit.py"
```

---

### Task 4: `peak_fit.py` — `fit_peaks()` core

**Files:**
- Modify: `peak_fit.py`
- Modify: `tests/test_peak_fit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_peak_fit.py`:

```python
from peak_fit import fit_peaks


def _make_spectrum(channels, peaks, slope, intercept, noise_seed=None):
    x = np.arange(channels, dtype=float)
    y = slope * x + intercept
    for amplitude, position, sigma in peaks:
        y = y + amplitude * np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        y = rng.poisson(np.maximum(y, 0)).astype(float)
    return x, y


def test_fit_single_peak_no_noise():
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
    assert len(result.peaks) == 1
    peak = result.peaks[0]
    assert peak.position == pytest.approx(100.0, abs=0.5)
    assert peak.amplitude == pytest.approx(500.0, rel=0.05)
    assert peak.sigma == pytest.approx(3.0, rel=0.1)
    assert peak.fwhm == pytest.approx(3.0 * 2.3548, rel=0.1)
    assert peak.area == pytest.approx(500.0 * 3.0 * np.sqrt(2 * np.pi), rel=0.1)
    assert result.background_slope == pytest.approx(0.0, abs=0.5)
    assert result.background_intercept == pytest.approx(20.0, abs=2.0)


def test_fit_multiplet_two_peaks_no_noise():
    x, y = _make_spectrum(
        channels=200,
        peaks=[(400.0, 95.0, 3.0), (300.0, 108.0, 3.0)],
        slope=0.1, intercept=10.0,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(60.0, 75.0),
        right_bg_region=(130.0, 145.0),
        fit_region=(75.0, 130.0),
        peak_positions=[95.0, 108.0],
    )
    assert len(result.peaks) == 2
    positions = sorted(p.position for p in result.peaks)
    assert positions[0] == pytest.approx(95.0, abs=1.0)
    assert positions[1] == pytest.approx(108.0, abs=1.0)


def test_fit_single_peak_with_poisson_noise():
    x, y = _make_spectrum(
        channels=200, peaks=[(2000.0, 100.0, 4.0)], slope=0.0, intercept=50.0,
        noise_seed=42,
    )
    result = fit_peaks(
        x, y,
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        peak_positions=[100.0],
    )
    peak = result.peaks[0]
    assert peak.position == pytest.approx(100.0, abs=1.0)
    assert peak.fwhm == pytest.approx(4.0 * 2.3548, rel=0.15)


def test_fit_rejects_no_peaks():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(x, y, (70.0, 85.0), (115.0, 130.0), (85.0, 115.0), [])


def test_fit_rejects_empty_fit_region():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(x, y, (70.0, 85.0), (115.0, 130.0), (500.0, 501.0), [100.0])


def test_fit_rejects_too_few_points_for_peak_count():
    x, y = _make_spectrum(channels=200, peaks=[(500.0, 100.0, 3.0)], slope=0.0, intercept=20.0)
    with pytest.raises(FitError):
        fit_peaks(x, y, (70.0, 85.0), (115.0, 130.0), (99.5, 100.5), [100.0, 101.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_peak_fit.py -v -k fit_peaks or test_fit_`
Expected: FAIL with `ImportError: cannot import name 'fit_peaks'`

- [ ] **Step 3: Add the implementation**

Add to the top of `peak_fit.py`, right after the `import numpy as np` line:

```python
from scipy.optimize import curve_fit

FWHM_FACTOR = 2.3548200450309493  # 2*sqrt(2*ln(2))
```

Then append at the end of the file:

```python
def _gaussian_sum(x, *params):
    result = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amplitude, position, sigma = params[i], params[i + 1], params[i + 2]
        result = result + amplitude * np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
    return result


def _initial_guess(x_fit, y_sub, fit_region, peak_positions):
    lo, hi = fit_region
    region_width = hi - lo
    n_peaks = len(peak_positions)
    sigma0 = max(region_width / (4 * n_peaks), 1e-6)
    guess = []
    for pos in peak_positions:
        idx = int(np.argmin(np.abs(x_fit - pos)))
        amplitude0 = float(y_sub[idx])
        guess.extend([amplitude0, float(pos), sigma0])
    return guess


def fit_peaks(x, y, left_bg_region, right_bg_region, fit_region, peak_positions):
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

    n_params = 3 * len(peak_positions)
    if x_fit.size < n_params:
        raise FitError(
            f"Fit region has {x_fit.size} data points, need at least "
            f"{n_params} for {len(peak_positions)} peak(s)"
        )

    y_sub = y_fit - (slope * x_fit + intercept)
    p0 = _initial_guess(x_fit, y_sub, fit_region, peak_positions)
    y_err = np.sqrt(np.maximum(y_fit, 1.0))

    try:
        popt, pcov = curve_fit(
            _gaussian_sum, x_fit, y_sub, p0=p0, sigma=y_err, absolute_sigma=True
        )
    except RuntimeError as exc:
        raise FitError(f"Fit did not converge: {exc}") from exc

    if pcov is None or not np.all(np.isfinite(pcov)):
        raise FitError("Fit produced a non-finite covariance matrix")

    perr = np.sqrt(np.diag(pcov))

    peaks = []
    for i in range(len(peak_positions)):
        amplitude, position, sigma = popt[3 * i], popt[3 * i + 1], popt[3 * i + 2]
        amplitude_err, position_err, sigma_err = perr[3 * i], perr[3 * i + 1], perr[3 * i + 2]
        sigma = abs(sigma)
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
                position=float(position),
                position_err=float(position_err),
                fwhm=float(fwhm),
                fwhm_err=float(fwhm_err),
                area=float(area),
                area_err=float(area_err),
                amplitude=float(amplitude),
                sigma=float(sigma),
            )
        )

    return FitResult(
        left_bg_region=tuple(left_bg_region),
        right_bg_region=tuple(right_bg_region),
        fit_region=tuple(fit_region),
        background_slope=float(slope),
        background_intercept=float(intercept),
        peaks=peaks,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v`
Expected: all PASS (13 tests: 7 from Tasks 2-3 + 6 new). If
`test_fit_single_peak_with_poisson_noise` fails on a specific tolerance
due to the particular noise draw, widen that assertion's tolerance
rather than removing the test — the point is verifying the fit is
robust to realistic Poisson noise, not pinning an exact value.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `pytest -v` from the worktree root.
Expected: all tests PASS (14 new + all pre-existing `histogram_io`/
`spe_io`/`spk_io` tests).

- [ ] **Step 6: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "Add fit_peaks(): Gaussian multiplet fit against fixed background"
```

---

### Task 5: `spectrum.py` — add `fits` field

**Files:**
- Modify: `spectrum.py`

- [ ] **Step 1: Add the field**

In `spectrum.py`, change:
```python
class LoadedSpectrum:
    def __init__(self, path, data, color):
        self.path = path
        self.data = data
        self.color = color
        self.visible = True
        self.active = False
```
to:
```python
class LoadedSpectrum:
    def __init__(self, path, data, color):
        self.path = path
        self.data = data
        self.color = color
        self.visible = True
        self.active = False
        self.fits = []
```

- [ ] **Step 2: Run the full test suite to confirm no regressions**

Run: `pytest -v`
Expected: all tests PASS (this is an additive change; nothing reads or
depends on the absence of `fits` yet).

- [ ] **Step 3: Commit**

```bash
git add spectrum.py
git commit -m "Add fits field to LoadedSpectrum for committed peak fits"
```

---

### Task 6: `conftest.py` — Qt test fixture

**Files:**
- Modify: `conftest.py`
- Test: `tests/test_qt_setup.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_qt_setup.py`:

```python
def test_qapp_fixture_provides_a_running_application(qapp):
    from PySide6.QtWidgets import QApplication
    assert isinstance(qapp, QApplication)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qt_setup.py -v`
Expected: FAIL with `fixture 'qapp' not found`

- [ ] **Step 3: Add the fixture**

`conftest.py` currently contains:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```
Change it to:
```python
import os

# Must be set before any PySide6/Qt import, including ones triggered by
# importing project modules below -- lets the Qt-dependent tests in this
# suite run headlessly without requiring the environment variable to be
# set externally every time.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_qt_setup.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add conftest.py tests/test_qt_setup.py
git commit -m "Add session-scoped qapp fixture for Qt-dependent tests"
```

---

### Task 7: `fit_mode.py` — `FitModeState` (pure state machine)

**Files:**
- Create: `fit_mode.py`
- Test: `tests/test_fit_mode.py`

This task adds only the pure, Qt-free state-tracking piece: which
step the user is on, the marked regions, and the marked peak
positions. The Qt/matplotlib-facing controller is added in Task 8.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fit_mode.py`:

```python
import pytest

from fit_mode import (
    STEP_FIT_REGION,
    STEP_LEFT_BG,
    STEP_MARKING_PEAKS,
    STEP_RIGHT_BG,
    FitModeState,
)


def test_initial_state():
    state = FitModeState()
    assert state.step == STEP_LEFT_BG
    assert state.left_bg_region is None
    assert state.right_bg_region is None
    assert state.fit_region is None
    assert state.peak_positions == []


def test_region_sequence_advances_steps():
    state = FitModeState()
    state.add_region(70, 85)
    assert state.step == STEP_RIGHT_BG
    assert state.left_bg_region == (70, 85)

    state.add_region(130, 115)  # reversed order gets normalized
    assert state.step == STEP_FIT_REGION
    assert state.right_bg_region == (115, 130)

    state.add_region(85, 115)
    assert state.step == STEP_MARKING_PEAKS
    assert state.fit_region == (85, 115)


def test_add_region_raises_when_not_awaiting_a_region():
    state = FitModeState()
    state.add_region(70, 85)
    state.add_region(115, 130)
    state.add_region(85, 115)
    with pytest.raises(ValueError):
        state.add_region(200, 210)


def test_add_peak_within_fit_region():
    state = FitModeState()
    state.add_region(70, 85)
    state.add_region(115, 130)
    state.add_region(85, 115)

    assert state.add_peak(100.0) is True
    assert state.peak_positions == [100.0]


def test_add_peak_outside_fit_region_is_ignored():
    state = FitModeState()
    state.add_region(70, 85)
    state.add_region(115, 130)
    state.add_region(85, 115)

    assert state.add_peak(200.0) is False
    assert state.peak_positions == []


def test_add_peak_before_fit_region_raises():
    state = FitModeState()
    state.add_region(70, 85)
    with pytest.raises(ValueError):
        state.add_peak(100.0)


def test_ready_to_fit_requires_all_regions_and_a_peak():
    state = FitModeState()
    assert state.ready_to_fit() is False
    state.add_region(70, 85)
    state.add_region(115, 130)
    state.add_region(85, 115)
    assert state.ready_to_fit() is False
    state.add_peak(100.0)
    assert state.ready_to_fit() is True


def test_reset_clears_everything():
    state = FitModeState()
    state.add_region(70, 85)
    state.add_region(115, 130)
    state.add_region(85, 115)
    state.add_peak(100.0)

    state.reset()

    assert state.step == STEP_LEFT_BG
    assert state.left_bg_region is None
    assert state.peak_positions == []


def test_ordered_bg_regions_regardless_of_marking_order():
    state = FitModeState()
    # mark the spatially-RIGHT region first
    state.add_region(115, 130)
    state.add_region(70, 85)
    state.add_region(85, 115)

    left, right = state.ordered_bg_regions()
    assert left == (70, 85)
    assert right == (115, 130)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fit_mode'`

- [ ] **Step 3: Write the implementation**

Create `fit_mode.py`:

```python
STEP_LEFT_BG = "left_bg"
STEP_RIGHT_BG = "right_bg"
STEP_FIT_REGION = "fit_region"
STEP_MARKING_PEAKS = "marking_peaks"


class FitModeState:
    """Tracks the in-progress region/peak marking sequence for one
    fit-mode session. Pure state -- no Qt/matplotlib dependency."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step = STEP_LEFT_BG
        self.left_bg_region = None
        self.right_bg_region = None
        self.fit_region = None
        self.peak_positions = []

    def add_region(self, lo, hi):
        region = (min(lo, hi), max(lo, hi))
        if self.step == STEP_LEFT_BG:
            self.left_bg_region = region
            self.step = STEP_RIGHT_BG
        elif self.step == STEP_RIGHT_BG:
            self.right_bg_region = region
            self.step = STEP_FIT_REGION
        elif self.step == STEP_FIT_REGION:
            self.fit_region = region
            self.step = STEP_MARKING_PEAKS
        else:
            raise ValueError("Not currently awaiting a region selection")

    def add_peak(self, position):
        if self.step != STEP_MARKING_PEAKS:
            raise ValueError("Not currently marking peaks")
        lo, hi = self.fit_region
        if not (lo <= position <= hi):
            return False
        self.peak_positions.append(position)
        return True

    def ready_to_fit(self):
        return (
            self.left_bg_region is not None
            and self.right_bg_region is not None
            and self.fit_region is not None
            and len(self.peak_positions) > 0
        )

    def ordered_bg_regions(self):
        """Returns (left, right) background regions ordered by mean
        x-coordinate, regardless of which was marked first."""
        a, b = self.left_bg_region, self.right_bg_region
        a_mid = (a[0] + a[1]) / 2
        b_mid = (b[0] + b[1]) / 2
        return (a, b) if a_mid <= b_mid else (b, a)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mode.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `pytest -v`

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode.py
git commit -m "Add FitModeState: pure state machine for the fit-marking sequence"
```

---

### Task 8: `fit_mode.py` `FitModeController` + `main_window.py` wiring

**Files:**
- Modify: `fit_mode.py`
- Modify: `main_window.py`
- Test: `tests/test_fit_mode_ui.py`

This task adds the Qt/matplotlib-facing controller (mouse handling,
in-progress marker drawing, running the fit, committing a `FitResult`)
and wires it into `main_window.py`: a "Fit Peaks" toggle, "Fit"/"Clear"
buttons, and canvas mouse events. Committed-fit overlay redrawing (on
`_plot_data()`) and the results panel are added in Tasks 9-10 — until
then, a committed fit is stored on the spectrum but not yet re-drawn
after a replot, which is fine since this task's test checks the
in-memory result, not the drawing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fit_mode_ui.py`:

```python
import numpy as np
from matplotlib.backend_bases import MouseEvent

from main_window import MainWindow
from spectrum import LoadedSpectrum


def _dispatch(main_window, name, xdata, ydata=10.0):
    ax = main_window.axes
    px, py = ax.transData.transform((xdata, ydata))
    event = MouseEvent(name, main_window.canvas, px, py, button=1)
    main_window.canvas.callbacks.process(name, event)


def _drag(main_window, x_start, x_end, y=10.0):
    _dispatch(main_window, "button_press_event", x_start, y)
    _dispatch(main_window, "button_release_event", x_end, y)


def _click(main_window, x, y=10.0):
    _drag(main_window, x, x, y)


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


def test_toggling_fit_mode_disables_nav_toolbar(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    assert main_window.nav_toolbar.isEnabled() is True
    main_window.fit_mode_action.setChecked(True)
    assert main_window.fit_controller.enabled is True
    assert main_window.nav_toolbar.isEnabled() is False

    main_window.fit_mode_action.setChecked(False)
    assert main_window.nav_toolbar.isEnabled() is True


def test_full_fit_flow_commits_a_fit_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.fit_mode_action.setChecked(True)

    _drag(main_window, 70, 85)    # left background region
    _drag(main_window, 115, 130)  # right background region
    _drag(main_window, 85, 115)   # fit region
    assert main_window.fit_button.isEnabled() is False
    _click(main_window, 100)      # peak position
    assert main_window.fit_button.isEnabled() is True

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 1
    result = spectrum.fits[0]
    assert len(result.peaks) == 1
    assert abs(result.peaks[0].position - 100.0) < 1.0


def test_clear_discards_in_progress_marks_without_committing(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.fit_mode_action.setChecked(True)
    _drag(main_window, 70, 85)
    _drag(main_window, 115, 130)
    _drag(main_window, 85, 115)
    _click(main_window, 100)

    main_window.fit_controller.clear()

    assert main_window.fit_controller.state.step == "left_bg"
    assert main_window.fit_button.isEnabled() is False
    assert len(spectrum.fits) == 0


def test_fit_mode_disabled_when_active_spectrum_hidden(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    spectrum.visible = False
    main_window._update_fit_mode_availability()

    assert main_window.fit_mode_action.isEnabled() is False


def test_fit_failure_leaves_marks_intact_and_shows_message(qapp):
    # Fit region deliberately far too narrow (1 data point) for the 2
    # peaks marked in it -- fit_peaks() raises FitError for "too few
    # data points", which run_fit() must catch, show as a status-bar
    # message, and NOT commit a result or reset the in-progress marks.
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    main_window.fit_mode_action.setChecked(True)
    _drag(main_window, 70, 85)
    _drag(main_window, 115, 130)
    _drag(main_window, 99.5, 100.5)
    _click(main_window, 100)
    _click(main_window, 100)

    main_window.fit_controller.run_fit()

    assert len(spectrum.fits) == 0
    assert main_window.fit_controller.state.step == "marking_peaks"
    assert main_window.fit_controller.state.peak_positions == [100.0, 100.0]
    assert main_window.statusBar().currentMessage() != ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: FAIL — `main_window.py` has no `fit_mode_action`/`fit_controller`/
`fit_button`/`_update_fit_mode_availability` yet (`AttributeError`).

- [ ] **Step 3: Add `FitModeController` to `fit_mode.py`**

Append to `fit_mode.py` (it already has the `import`-free `FitModeState`
from Task 7; add these imports at the top of the file first):

```python
import numpy as np
from matplotlib.backend_bases import _Mode
from PySide6.QtGui import QAction

from peak_fit import FitError, fit_peaks
```

Then append at the end of the file:

```python
class FitModeController:
    """Qt/matplotlib-facing wrapper around FitModeState: owns the plot
    artists for in-progress marking (committed-fit drawing and the
    results panel are added on top of this in later tasks), and drives
    fit_peaks() when the user clicks "Fit"."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.enabled = False
        self.state = FitModeState()
        self._drag_start = None
        self._progress_artists = []

    def toggle(self, enabled):
        self.enabled = enabled
        self._clear_progress()
        mw = self.main_window
        mw.nav_toolbar.setEnabled(not enabled)
        if enabled:
            if mw.nav_toolbar.mode == _Mode.PAN:
                mw.nav_toolbar.pan()
            elif mw.nav_toolbar.mode == _Mode.ZOOM:
                mw.nav_toolbar.zoom()
        mw.fit_button.setEnabled(False)
        mw.clear_fit_button.setEnabled(enabled)
        mw.canvas.draw_idle()

    def clear(self):
        self._clear_progress()
        self.main_window.fit_button.setEnabled(False)
        self.main_window.canvas.draw_idle()

    def _clear_progress(self):
        for artist in self._progress_artists:
            artist.remove()
        self._progress_artists = []
        self.state.reset()

    def on_press(self, event):
        if not self.enabled or event.inaxes != self.main_window.axes or event.xdata is None:
            return
        if event.button != 1:
            return
        self._drag_start = event.xdata

    def on_release(self, event):
        if not self.enabled or self._drag_start is None:
            return
        if event.inaxes != self.main_window.axes or event.xdata is None:
            self._drag_start = None
            return
        start = self._drag_start
        end = event.xdata
        self._drag_start = None

        if self.state.step == STEP_MARKING_PEAKS:
            added = self.state.add_peak(end)
            if added:
                artist = self.main_window.axes.axvline(
                    end, color="red", linestyle=":", linewidth=1
                )
                self._progress_artists.append(artist)
                self.main_window.canvas.draw_idle()
            else:
                self.main_window.statusBar().showMessage(
                    "Peak position must be inside the fit region", 3000
                )
        else:
            lo, hi = min(start, end), max(start, end)
            if hi <= lo:
                self.main_window.statusBar().showMessage(
                    "Drag to select a region (a click alone is not enough)", 3000
                )
                return
            color = "tab:blue" if self.state.step == STEP_FIT_REGION else "gray"
            artist = self.main_window.axes.axvspan(lo, hi, color=color, alpha=0.15)
            self._progress_artists.append(artist)
            self.state.add_region(lo, hi)
            self.main_window.canvas.draw_idle()

        self.main_window.fit_button.setEnabled(self.state.ready_to_fit())

    def run_fit(self):
        if not self.state.ready_to_fit():
            return
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        left, right = self.state.ordered_bg_regions()
        x = np.arange(len(active.data), dtype=float)
        y = active.data
        try:
            result = fit_peaks(
                x, y, left, right, self.state.fit_region, list(self.state.peak_positions)
            )
        except FitError as exc:
            self.main_window.statusBar().showMessage(f"Fit failed: {exc}", 5000)
            return
        active.fits.append(result)
        self._clear_progress()
        self.main_window.fit_button.setEnabled(False)
        self.main_window._plot_data()
```

- [ ] **Step 4: Wire `main_window.py`**

Add the import. Change:
```python
from histogram_io import ParseError, load_histogram
from settings import Settings
from spe_io import load_spe
from spectrum import LoadedSpectrum, next_color
from spk_io import load_spk
```
to:
```python
from fit_mode import FitModeController
from histogram_io import ParseError, load_histogram
from settings import Settings
from spe_io import load_spe
from spectrum import LoadedSpectrum, next_color
from spk_io import load_spk
```

In `MainWindow.__init__`, change:
```python
        self._build_zoom_buttons()
```
to:
```python
        self._build_zoom_buttons()

        self.fit_controller = FitModeController(self)
        self._build_fit_mode_buttons()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._update_fit_mode_availability()
```

Add these new methods to `MainWindow` (e.g. right after `_build_zoom_buttons`):

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
```

Update `_on_scroll` to suspend scroll-zoom during fit mode. Change:
```python
    def _on_scroll(self, event):
        if event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)
```
to:
```python
    def _on_scroll(self, event):
        if self.fit_controller.enabled:
            return
        if event.inaxes != self.axes or event.xdata is None:
            return
        factor = (1 / ZOOM_FACTOR) if event.button == "up" else ZOOM_FACTOR
        self._zoom_x(factor, center=event.xdata)
```

Update `_plot_data` to refresh fit-mode availability after every replot
(load, show/hide toggle, log-scale toggle all call this already). Change:
```python
        self.canvas.draw()
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()
```
to:
```python
        self.canvas.draw()
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()
        self._update_fit_mode_availability()
```

Update `_on_active_toggled` to refresh availability when the Active
spectrum changes (this doesn't call `_plot_data()` today, so it needs
its own explicit call). Change:
```python
    def _on_active_toggled(self, path, checked):
        if not checked:
            return
        for spectrum in self.spectra:
            spectrum.active = spectrum.path == path
```
to:
```python
    def _on_active_toggled(self, path, checked):
        if not checked:
            return
        for spectrum in self.spectra:
            spectrum.active = spectrum.path == path
        self._update_fit_mode_availability()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Run the full test suite to confirm no regressions**

Run: `pytest -v`

- [ ] **Step 7: Commit**

```bash
git add fit_mode.py main_window.py tests/test_fit_mode_ui.py
git commit -m "Add FitModeController: interactive fit-marking wired into main_window"
```

---

### Task 9: Committed-fit overlay drawing

**Files:**
- Modify: `fit_mode.py`
- Modify: `main_window.py`
- Modify: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fit_mode_ui.py`:

```python
from peak_fit import FitResult, PeakResult


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

    # 3 shaded spans (left bg, right bg, fit region) = 3 patches;
    # spectrum data line + background dashed line + fitted curve +
    # 1 peak marker = 4 lines.
    assert len(main_window.axes.patches) == 3
    assert len(main_window.axes.lines) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mode_ui.py -v -k draws_committed`
Expected: FAIL — no fit overlay is drawn yet, so `patches`/`lines`
counts won't match (0 patches, 1 line).

- [ ] **Step 3: Add `draw_committed_fits` to `FitModeController`**

Append this method inside the `FitModeController` class in `fit_mode.py`
(anywhere among its other methods, e.g. after `run_fit`):

```python
    def draw_committed_fits(self, spectrum):
        axes = self.main_window.axes
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
                total = total + peak.amplitude * np.exp(
                    -((x_dense - peak.position) ** 2) / (2 * peak.sigma ** 2)
                )
            axes.plot(x_dense, total, color="red", linewidth=1.5)

            for peak in result.peaks:
                axes.axvline(peak.position, color="red", linestyle=":", linewidth=1)
```

- [ ] **Step 4: Call it from `main_window.py`'s `_plot_data`**

Change:
```python
        for spectrum in visible:
            channels = np.arange(len(spectrum.data))
            self.axes.plot(channels, spectrum.data, drawstyle="steps-mid", color=spectrum.color)
```
to:
```python
        for spectrum in visible:
            channels = np.arange(len(spectrum.data))
            self.axes.plot(channels, spectrum.data, drawstyle="steps-mid", color=spectrum.color)
            self.fit_controller.draw_committed_fits(spectrum)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_fit_mode_ui.py -v -k draws_committed`
Expected: PASS

- [ ] **Step 6: Run the full test suite to confirm no regressions**

Run: `pytest -v`

- [ ] **Step 7: Commit**

```bash
git add fit_mode.py main_window.py tests/test_fit_mode_ui.py
git commit -m "Draw committed fit overlays (regions, background, curve, peaks) on replot"
```

---

### Task 10: Results panel

**Files:**
- Modify: `fit_mode.py`
- Modify: `main_window.py`
- Modify: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fit_mode_ui.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fit_mode_ui.py -v -k results_panel or clear_all_fits`
Expected: FAIL — `FitModeController` has no `results_list`/
`update_results_list` yet (`AttributeError`).

- [ ] **Step 3: Add the results panel to `FitModeController`**

Add these imports to the top of `fit_mode.py`, alongside the existing
`from PySide6.QtGui import QAction` line:
```python
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem, QMenu, QToolBar
```

Append these methods inside the `FitModeController` class (e.g. after
`draw_committed_fits`):

```python
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
            lines = [f"Fit region [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]"]
            for i, peak in enumerate(result.peaks, start=1):
                lines.append(
                    f"  Peak {i}: pos={peak.position:.2f}±{peak.position_err:.2f}  "
                    f"FWHM={peak.fwhm:.2f}±{peak.fwhm_err:.2f}  "
                    f"area={peak.area:.1f}±{peak.area_err:.1f}"
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
            mw._plot_data()
            self.update_results_list()
        elif chosen == clear_action:
            active.fits.clear()
            mw._plot_data()
            self.update_results_list()
```

- [ ] **Step 4: Wire it into `main_window.py`**

Change:
```python
        self.fit_controller = FitModeController(self)
        self._build_fit_mode_buttons()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._update_fit_mode_availability()
```
to:
```python
        self.fit_controller = FitModeController(self)
        self._build_fit_mode_buttons()
        self.fit_controller.build_results_panel()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._update_fit_mode_availability()
```

Update `_plot_data` to refresh the results list on every replot. Change:
```python
        self.canvas.draw()
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()
        self._update_fit_mode_availability()
```
to:
```python
        self.canvas.draw()
        self.nav_toolbar.update()
        self.nav_toolbar.push_current()
        self._update_fit_mode_availability()
        self.fit_controller.update_results_list()
```

Update `_on_active_toggled` to refresh the results list when the Active
spectrum changes (the panel is scoped to the Active spectrum, unlike the
plot overlays which show every visible spectrum's fits regardless of
which is Active). Change:
```python
    def _on_active_toggled(self, path, checked):
        if not checked:
            return
        for spectrum in self.spectra:
            spectrum.active = spectrum.path == path
        self._update_fit_mode_availability()
```
to:
```python
    def _on_active_toggled(self, path, checked):
        if not checked:
            return
        for spectrum in self.spectra:
            spectrum.active = spectrum.path == path
        self._update_fit_mode_availability()
        self.fit_controller.update_results_list()
```

Also update `run_fit` in `fit_mode.py` to refresh the results list after
committing a fit. Change:
```python
        active.fits.append(result)
        self._clear_progress()
        self.main_window.fit_button.setEnabled(False)
        self.main_window._plot_data()
```
to:
```python
        active.fits.append(result)
        self._clear_progress()
        self.main_window.fit_button.setEnabled(False)
        self.main_window._plot_data()
        self.update_results_list()
```

(This is a small duplication with `_plot_data`'s own call to
`update_results_list` -- harmless, since the method is idempotent
[it just clears and rebuilds the list from `active.fits`], and keeps
`run_fit` correct even if `_plot_data`'s internals change later.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS (all tests in this file, 9 total across Tasks 8-10)

- [ ] **Step 6: Run the full test suite to confirm no regressions**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 7: Manually verify the "Remove Fit"/"Clear All Fits" context
  menu wiring** (the menu-item selection itself isn't covered by the
  automated tests above, since driving a modal `QMenu.exec()` headlessly
  is disproportionately fragile to automate for this one piece of glue
  code — the underlying list-index-to-fit-removal logic it calls into is
  already covered by the tests above):

```bash
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -c "
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
from spectrum import LoadedSpectrum
from peak_fit import FitResult, PeakResult
import numpy as np

app = QApplication([])
w = MainWindow()
y = np.full(200, 20, dtype=np.int64)
s = LoadedSpectrum('synthetic.txt', y, '#1f77b4')
s.active = True
s.fits.append(FitResult(
    left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
    fit_region=(90.0, 110.0), background_slope=0.0, background_intercept=20.0,
    peaks=[PeakResult(position=100.0, position_err=0.1, fwhm=5.0, fwhm_err=0.2,
                       area=1000.0, area_err=50.0, amplitude=200.0, sigma=2.0)],
))
w.spectra.append(s)
w.fit_controller.update_results_list()
print('before remove:', w.fit_controller.results_list.count())
del s.fits[0]
w._plot_data()
w.fit_controller.update_results_list()
print('after remove:', w.fit_controller.results_list.count())
"
```

Expected: `before remove: 1` then `after remove: 0` (confirms the
list/spectrum state that the context-menu actions manipulate behaves as
expected; the menu itself just calls this same code on a click).

- [ ] **Step 8: Commit**

```bash
git add fit_mode.py main_window.py tests/test_fit_mode_ui.py
git commit -m "Add fit results panel with per-fit removal and clear-all"
```

---

### Task 11: Rebuild and verify the Windows installer

**Files:** none (build artifacts only)

- [ ] **Step 1: Rebuild**

```powershell
powershell -File packaging\windows\build.ps1
```

Expected: build completes successfully, producing
`packaging/windows/output/SpectraToolsSetup.exe`. (Retry once if Inno
Setup's `islzma.dll` access-violation recurs — an established
intermittent packaging flake in this project, not a code issue.)

- [ ] **Step 2: Silent-install, launch, and confirm no crash**

```bash
"packaging/windows/output/SpectraToolsSetup.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /NOICONS
```

Launch the installed `SpectraTools.exe` (typically under
`Program Files (x86)\SpectraTools`), load a spectrum, toggle "Fit
Peaks", mark the three regions and a peak, click "Fit", and confirm a
result appears on the plot and in the "Fit Results" panel without a
crash. Then close it.

- [ ] **Step 3: Clean up the test install**

Run the installed uninstaller silently
(`unins000.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART`), then confirm
the install directory is gone and no stray desktop shortcut was left
behind.

- [ ] **Step 4: Report completion**

Report to the user that peak fitting — the third and final approved
sub-project — is complete: manual background/fit-region/peak marking,
Gaussian-plus-fixed-background fitting via `scipy.curve_fit`, persisted
multi-fit overlays, and a results panel, all verified against the design
spec, with a full automated test suite and a successful installer
rebuild.
