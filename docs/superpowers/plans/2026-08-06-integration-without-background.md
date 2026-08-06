# Integration Without Background Regions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `Ctrl+I` (Integration) run with zero background regions marked (a fit region alone is enough), reporting the raw gross area/centroid/FWHM/skewness with no background subtraction. The existing two-background-region path stays byte-for-byte unchanged. First feature of version 2.1.2.

**Architecture:** Extends the existing `peak_fit.py` → `fit_mode.py` → `fit_export.py` pipeline; no new files. `IntegrationResult` gains a `has_background` property. `integrate_region()`'s background-pooling loop and its final region-tuple construction become conditional on `left_bg_region`/`right_bg_region` being non-`None`. `fit_mode.py`'s state gating, plot drawing, and tooltip, and `fit_export.py`'s JSON/text writers all branch on `has_background` to *suppress* (not zero-out) the background/net breakdown — matching TV's own `FIIntegrateRegion`/`ParseIntPeaks` behavior, researched directly from `tv-1.9.13/` source in the design spec below.

**Tech Stack:** Python 3.13 (Windows dev machine and Windows build), Python 3.9.25 (the AlmaLinux 8 Linux build — see Task 1's implementation note on why dataclass field annotations stay bare `tuple`, not `tuple | None`), PySide6/matplotlib/numpy (existing), `pytest` with the existing `qapp` fixture.

**Design spec:** `docs/superpowers/specs/2026-08-06-integration-without-background-design.md` — read for the full rationale (including the TV source research this was grounded in) if anything below is unclear on the *why*; this plan covers the exact *what*.

---

## Context for the implementer

This is a nuclear-spectroscopy peak-fitting desktop app (SpectraTools). Today, `Ctrl+I` (Integration) requires exactly two background regions (`B` marks) plus a fit region (`R` mark) before it's even enabled. This feature makes it work with **zero** background regions too (still gated to exactly 0 or exactly 2 — never 1), in which case the result reports the raw region sum and centroid with no background subtraction, and the UI shows a single unlabeled value set instead of a Gross/Background/Net breakdown.

All work happens directly on `master`, committing after each task — this matches every prior feature in this repo's history (`git log --oneline` shows Add/Subtract Spectra, the v2.1.0/v2.1.1 Help-page fixes, etc., all committed straight to `master`; no feature branches or worktrees were used). Do not create a branch or worktree for this work. (There is a pre-existing, unrelated worktree at `.claude/worktrees/fit-parameter-fixing` — leave it alone, it predates this plan.)

Run tests with `.venv/Scripts/python.exe -m pytest <path> -v` on Windows (this repo's dev machine) or `.venv/bin/python -m pytest <path> -v` on Linux.

---

## Task 1: `peak_fit.py` — `IntegrationResult.has_background` and `integrate_region()`

**Files:**
- Modify: `peak_fit.py:101-133` (`IntegrationResult` dataclass)
- Modify: `peak_fit.py:536-553` (background-pooling block inside `integrate_region()`)
- Modify: `peak_fit.py:635-650` (`IntegrationResult(...)` construction at the end of `integrate_region()`)
- Test: `tests/test_peak_fit.py`

**Implementation note before you start:** the design spec's architecture section illustrates `left_bg_region`/`right_bg_region` as `tuple | None` (PEP 604 syntax). Do **not** write that literally — this file has no `from __future__ import annotations`, and the Linux build actually runs on **Python 3.9.25** (AlmaLinux 8, confirmed in `docs/superpowers/specs/2026-08-03-linux-native-packages-design.md`), where a bare `tuple | None` class-body annotation raises `TypeError` at import time. Python dataclasses don't validate field types at runtime anyway — this file already has exactly this pattern (`timestamp: str = None`, a field documented as optional via a plain-`str` annotation). Leave `left_bg_region: tuple` and `right_bg_region: tuple` exactly as they are; only add the new property.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_peak_fit.py` right after `test_integration_result_holds_expected_fields()` (around line 990):

```python
def test_integration_result_has_background_true_when_bg_regions_set():
    result = IntegrationResult(
        left_bg_region=(10.0, 20.0), right_bg_region=(180.0, 190.0),
        fit_region=(90.0, 110.0), background_density=5.0,
        gross_area=100.0, gross_area_err=10.0,
        gross_centroid=100.0, gross_centroid_err=1.0,
        gross_fwhm=5.0, gross_fwhm_err=0.5,
        gross_skewness=0.1, gross_skewness_err=0.05,
        background_area=20.0, background_area_err=4.0,
        background_centroid=100.0, background_centroid_err=2.0,
        background_fwhm=3.0, background_fwhm_err=0.3,
        background_skewness=0.0, background_skewness_err=0.02,
        net_area=80.0, net_area_err=11.0,
        net_centroid=100.0, net_centroid_err=1.2,
        net_fwhm=5.0, net_fwhm_err=0.6,
        net_skewness=0.1, net_skewness_err=0.06,
    )
    assert result.has_background is True


def test_integration_result_has_background_false_when_bg_regions_none():
    result = IntegrationResult(
        left_bg_region=None, right_bg_region=None,
        fit_region=(90.0, 110.0), background_density=0.0,
        gross_area=100.0, gross_area_err=10.0,
        gross_centroid=100.0, gross_centroid_err=1.0,
        gross_fwhm=5.0, gross_fwhm_err=0.5,
        gross_skewness=0.1, gross_skewness_err=0.05,
        background_area=0.0, background_area_err=0.0,
        background_centroid=0.0, background_centroid_err=0.0,
        background_fwhm=0.0, background_fwhm_err=0.0,
        background_skewness=0.0, background_skewness_err=0.0,
        net_area=100.0, net_area_err=10.0,
        net_centroid=100.0, net_centroid_err=1.0,
        net_fwhm=5.0, net_fwhm_err=0.5,
        net_skewness=0.1, net_skewness_err=0.05,
    )
    assert result.has_background is False
```

Add to `tests/test_peak_fit.py` right after `test_integrate_region_matches_hand_computed_moments()` (around line 1033):

```python
def test_integrate_region_without_background_omits_bg_regions_and_zeroes_density():
    x, y = _make_spectrum(
        channels=200, peaks=[(5000.0, 100.0, 4.0)], slope=0.0, intercept=50.0,
    )
    result = integrate_region(
        x, y, left_bg_region=None, right_bg_region=None,
        fit_region=(85.0, 115.0),
    )
    assert result.left_bg_region is None
    assert result.right_bg_region is None
    assert result.background_density == 0.0
    assert result.background_area == 0.0
    assert result.has_background is False


def test_integrate_region_without_background_makes_net_equal_gross():
    x, y = _make_spectrum(
        channels=200, peaks=[(5000.0, 100.0, 4.0)], slope=0.0, intercept=50.0,
    )
    result = integrate_region(
        x, y, left_bg_region=None, right_bg_region=None,
        fit_region=(85.0, 115.0),
    )
    # No background to subtract -- net reduces to gross exactly (both the
    # sum and, for this fixture's positive gross sum, every moment too).
    assert result.net_area == result.gross_area
    assert result.net_centroid == result.gross_centroid
    assert result.net_fwhm == result.gross_fwhm
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_peak_fit.py -k "has_background or without_background" -v`

Expected: `test_integration_result_has_background_true_when_bg_regions_set` and `..._false_when_bg_regions_none` both FAIL with `AttributeError: 'IntegrationResult' object has no attribute 'has_background'`. `test_integrate_region_without_background_omits_bg_regions_and_zeroes_density` and `..._makes_net_equal_gross` both FAIL with `TypeError: cannot unpack non-iterable NoneType object` (from the unconditional `for region in (left_bg_region, right_bg_region): blo, bhi = region` loop).

- [ ] **Step 3: Add the `has_background` property**

In `peak_fit.py`, at the end of the `IntegrationResult` dataclass (after `visible: bool = True`, currently line 132):

```python
    timestamp: str = None
    visible: bool = True

    @property
    def has_background(self):
        return self.left_bg_region is not None
```

- [ ] **Step 4: Guard the background-pooling block in `integrate_region()`**

Replace (currently `peak_fit.py:536-553`):

```python
    # ---- background: pooled flat density across BOTH bg regions ----
    bg_chn = 0
    bg_count = 0.0
    bg_dcount = 0.0
    for region in (left_bg_region, right_bg_region):
        blo, bhi = region
        bmask = (x >= blo) & (x <= bhi)
        bg_chn += int(np.sum(bmask))
        bg_y = y[bmask]
        bg_count += float(np.sum(bg_y))
        bg_dcount += float(np.sum(bg_y))

    if bg_chn > 0:
        bg_density = bg_count / bg_chn
        bg_density_var = bg_dcount / (bg_chn * bg_chn)
    else:
        bg_density = 0.0
        bg_density_var = 0.0
```

with:

```python
    # ---- background: pooled flat density across BOTH bg regions ----
    # TV's own branch has no trailing `else` (vsFitInt.c:85,194) -- when
    # no background regions are marked, this block is skipped outright,
    # not "looped over zero regions that happens to sum to zero".
    bg_chn = 0
    bg_count = 0.0
    bg_dcount = 0.0
    if left_bg_region is not None and right_bg_region is not None:
        for region in (left_bg_region, right_bg_region):
            blo, bhi = region
            bmask = (x >= blo) & (x <= bhi)
            bg_chn += int(np.sum(bmask))
            bg_y = y[bmask]
            bg_count += float(np.sum(bg_y))
            bg_dcount += float(np.sum(bg_y))

    if bg_chn > 0:
        bg_density = bg_count / bg_chn
        bg_density_var = bg_dcount / (bg_chn * bg_chn)
    else:
        bg_density = 0.0
        bg_density_var = 0.0
```

- [ ] **Step 5: Guard the `IntegrationResult(...)` construction's region-tuple conversion**

`tuple(None)` raises `TypeError`, so the final `return IntegrationResult(...)` (currently `peak_fit.py:635-650`) needs its first two lines changed. Replace:

```python
    return IntegrationResult(
        left_bg_region=tuple(left_bg_region), right_bg_region=tuple(right_bg_region),
        fit_region=tuple(fit_region), background_density=float(bg_density),
```

with:

```python
    return IntegrationResult(
        left_bg_region=tuple(left_bg_region) if left_bg_region is not None else None,
        right_bg_region=tuple(right_bg_region) if right_bg_region is not None else None,
        fit_region=tuple(fit_region), background_density=float(bg_density),
```

(The rest of the `return IntegrationResult(...)` call — `gross_area=float(gross_area), ...` through `net_skewness_err=float(n_skew_err),` — is unchanged.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_peak_fit.py -v`

Expected: all tests pass, including the 4 new ones and every pre-existing `integrate_region`/`IntegrationResult` test (no regressions to the with-background path).

- [ ] **Step 7: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "feat: allow integrate_region() to run without background regions"
```

---

## Task 2: `fit_mode.py` — gate Integration on 0-or-2 background regions

**Files:**
- Modify: `fit_mode.py:103-113` (`FitModeState.ready_to_integrate()`, `FitModeState.ordered_bg_regions()`)
- Test: `tests/test_fit_mode.py`
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_mode.py` (after `test_ordered_bg_regions_regardless_of_marking_order`, around line 172):

```python
def test_ready_to_integrate_true_with_zero_bg_regions_and_a_fit_region():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.ready_to_integrate() is True


def test_ready_to_integrate_false_with_exactly_one_bg_region():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.ready_to_integrate() is False


def test_ordered_bg_regions_returns_none_pair_when_no_regions_marked():
    state = FitModeState()
    assert state.ordered_bg_regions() == (None, None)
```

Add to `tests/test_fit_mode_ui.py` (after `test_run_integration_appends_an_integration_result`, around line 2319):

```python
def test_ready_to_integrate_allows_zero_background_regions(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    state = main_window.fit_controller.state

    assert state.ready_to_integrate() is False
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    assert state.ready_to_integrate() is True  # no background marks needed


def test_integrate_button_enables_with_zero_background_regions(qapp):
    main_window = MainWindow()
    assert main_window.integrate_button.isEnabled() is False

    _make_active_spectrum(main_window)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    assert main_window.integrate_button.isEnabled() is True  # no background marks needed
```

**Note (found during Task 2's actual execution, not anticipated when this plan was first written):** an earlier draft of this task also added `test_run_integration_with_no_background_produces_a_backgroundless_result` here. That test calls the *full* `run_integration()`, which — after building the result — unconditionally calls `fit_export.append_auto_log()` (crashes on `None` bg regions until Task 4 lands) and then `main_window._plot_data()` → `draw_committed_fits()` (crashes on `None` bg regions until Task 3 lands). It cannot pass until Tasks 2, 3, *and* 4 are all done, so it does not belong in Task 2. It has been moved to Task 4 (the last of the three prerequisite tasks in this plan's ordering) — see Task 4's Step 1 below. Task 2's own scope is fully covered by the 5 tests above, which only exercise `ready_to_integrate()`/`ordered_bg_regions()`/button-enablement and need nothing from Tasks 3-4.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode.py -k "ready_to_integrate_true_with_zero or ordered_bg_regions_returns_none" -v`

Expected: `test_ready_to_integrate_true_with_zero_bg_regions_and_a_fit_region` FAILS (`ready_to_integrate()` currently requires exactly 2 regions, so it returns `False`, but the test asserts `True`). `test_ordered_bg_regions_returns_none_pair_when_no_regions_marked` FAILS with `ValueError: not enough values to unpack (expected 2, got 0)`. (`test_ready_to_integrate_false_with_exactly_one_bg_region` already passes today — 1 region was never valid — this is a regression guard, not new behavior; keep it.)

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode_ui.py -k "allows_zero_background_regions or enables_with_zero_background" -v`

Expected: both FAIL — `ready_to_integrate()` returns `False` with 0 background regions today, so `main_window.integrate_button.isEnabled()` stays `False` too (the button's `setEnabled(...)` call delegates entirely to `ready_to_integrate()`).

- [ ] **Step 3: Implement**

Replace (currently `fit_mode.py:103-113`):

```python
    def ready_to_integrate(self):
        return len(self.bg_regions) == BG_REGION_CAP and self.fit_region is not None

    def ordered_bg_regions(self):
        """Returns (left, right) background regions ordered by mean
        x-coordinate, regardless of which was marked first. Only valid
        once both regions exist."""
        a, b = self.bg_regions
        a_mid = (a[0] + a[1]) / 2
        b_mid = (b[0] + b[1]) / 2
        return (a, b) if a_mid <= b_mid else (b, a)
```

with:

```python
    def ready_to_integrate(self):
        return self.fit_region is not None and len(self.bg_regions) in (0, BG_REGION_CAP)

    def ordered_bg_regions(self):
        """Returns (left, right) background regions ordered by mean
        x-coordinate, regardless of which was marked first. Returns
        (None, None) when no background regions are marked -- a valid,
        deliberate state now that Integration allows a zero-background
        run. Only meaningful when len(bg_regions) is 0 or BG_REGION_CAP;
        any other count (e.g. exactly 1) is not a state
        ready_to_integrate() would ever let a caller reach."""
        if not self.bg_regions:
            return (None, None)
        a, b = self.bg_regions
        a_mid = (a[0] + a[1]) / 2
        b_mid = (b[0] + b[1]) / 2
        return (a, b) if a_mid <= b_mid else (b, a)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode.py tests/test_fit_mode_ui.py -v`

Expected: all tests pass, including the 5 new ones and every pre-existing test in both files (in particular `test_ready_to_integrate_needs_bg_regions_and_fit_region_but_not_peaks` and `test_integrate_button_has_ctrl_i_shortcut_and_is_gated`, which exercise the still-required exactly-2-regions path — unaffected since `BG_REGION_CAP` stays in the allowed set).

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: gate Integration on 0-or-2 background regions, not exactly 2"
```

---

## Task 3: `fit_mode.py` — simplify plot annotation and tooltip without background

**Files:**
- Modify: `fit_mode.py:235-257` (`_integration_tooltip()`)
- Modify: `fit_mode.py:542-574` (the `IntegrationResult` branch inside `draw_committed_fits()`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_mode_ui.py` (after `test_draw_committed_fits_draws_region_shading_and_annotation_for_integration`, around line 1224):

```python
def test_draw_committed_fits_draws_only_fit_region_and_annotation_without_background(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=None, right_bg_region=None,
            fit_region=(85.0, 115.0), background_density=0.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=0.0, background_area_err=0.0,
            background_centroid=0.0, background_centroid_err=0.0,
            background_fwhm=0.0, background_fwhm_err=0.0,
            background_skewness=0.0, background_skewness_err=0.0,
            net_area=1000.0, net_area_err=30.0,
            net_centroid=100.0, net_centroid_err=0.5,
            net_fwhm=8.0, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )

    patches_before = len(main_window.axes.patches)
    lines_before = len(main_window.axes.lines)
    texts_before = len(main_window.axes.texts)
    main_window.fit_controller.draw_committed_fits(spectrum)

    # 1 region span (fit region only, no bg spans), 0 background lines,
    # 1 annotation.
    assert len(main_window.axes.patches) == patches_before + 1
    assert len(main_window.axes.lines) == lines_before + 0
    assert len(main_window.axes.texts) == texts_before + 1
```

Add to `tests/test_fit_mode_ui.py` (after `test_results_table_shows_a_region_row_for_an_integration_result` and its keV-tooltip counterpart, around line 2376):

```python
def test_integration_tooltip_is_a_single_line_without_background(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=None, right_bg_region=None,
            fit_region=(85.0, 115.0), background_density=0.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=0.0, background_area_err=0.0,
            background_centroid=0.0, background_centroid_err=0.0,
            background_fwhm=0.0, background_fwhm_err=0.0,
            background_skewness=0.0, background_skewness_err=0.0,
            net_area=1000.0, net_area_err=30.0,
            net_centroid=100.0, net_centroid_err=0.5,
            net_fwhm=8.0, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )

    main_window.fit_controller.update_results_list()

    tooltip = main_window.fit_controller.results_table.item(0, 0).toolTip()
    assert "\n" not in tooltip
    assert "Gross" not in tooltip
    assert "Background" not in tooltip
    assert "Net" not in tooltip
    assert "Area=1000.0" in tooltip
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode_ui.py -k "without_background and (draw_committed_fits or tooltip)" -v`

Expected: `test_draw_committed_fits_draws_only_fit_region_and_annotation_without_background` FAILS — `draw_committed_fits()` currently does `left_lo, left_hi = result.left_bg_region` unconditionally, raising `TypeError: cannot unpack non-iterable NoneType object`. `test_integration_tooltip_is_a_single_line_without_background` FAILS on `assert "\n" not in tooltip` (the tooltip currently always renders all 3 Gross/Background/Net lines, using `getattr` with harmless 0.0 fallbacks rather than crashing).

- [ ] **Step 3: Implement — `_integration_tooltip()`**

Replace (currently `fit_mode.py:235-257`):

```python
def _integration_tooltip(main_window, result):
    """Full gross/background/net breakdown for an IntegrationResult's Fit
    Results row tooltip -- mirrors the level of detail the existing
    left-tail-info tooltip gives for a Gaussian fit. Centroid/FWHM show
    dual units when calibration is active; area has no energy-axis
    equivalent and stays channel-only, matching the results table."""
    lines = []
    for label, prefix in (("Gross", "gross"), ("Background", "background"), ("Net", "net")):
        area = getattr(result, f"{prefix}_area")
        area_err = getattr(result, f"{prefix}_area_err")
        centroid = getattr(result, f"{prefix}_centroid")
        centroid_err = getattr(result, f"{prefix}_centroid_err")
        fwhm = getattr(result, f"{prefix}_fwhm")
        fwhm_err = getattr(result, f"{prefix}_fwhm_err")
        skewness = getattr(result, f"{prefix}_skewness")
        skewness_err = getattr(result, f"{prefix}_skewness_err")
        lines.append(
            f"{label}: area={area:.1f}±{area_err:.1f}, "
            f"centroid={_dual_unit_value(main_window, centroid, centroid_err, is_width=False)}, "
            f"FWHM={_dual_unit_value(main_window, fwhm, fwhm_err, is_width=True, reference_position=centroid)}, "
            f"skewness={skewness:.3g}±{skewness_err:.3g}"
        )
    return "\n".join(lines)
```

with:

```python
def _integration_tooltip(main_window, result):
    """Full gross/background/net breakdown for an IntegrationResult's Fit
    Results row tooltip -- mirrors the level of detail the existing
    left-tail-info tooltip gives for a Gaussian fit. Centroid/FWHM show
    dual units when calibration is active; area has no energy-axis
    equivalent and stays channel-only, matching the results table. With
    no background marked, gross/background/net collapse to one number
    each (net == gross exactly), so a single unlabeled line is shown
    instead of a three-way breakdown -- reading from gross_* directly
    rather than relying on that equality, matching TV's own choice to
    report the total/gross row in this case."""
    if not result.has_background:
        return (
            f"Area={result.gross_area:.1f}±{result.gross_area_err:.1f}, "
            f"centroid={_dual_unit_value(main_window, result.gross_centroid, result.gross_centroid_err, is_width=False)}, "
            f"FWHM={_dual_unit_value(main_window, result.gross_fwhm, result.gross_fwhm_err, is_width=True, reference_position=result.gross_centroid)}, "
            f"skewness={result.gross_skewness:.3g}±{result.gross_skewness_err:.3g}"
        )
    lines = []
    for label, prefix in (("Gross", "gross"), ("Background", "background"), ("Net", "net")):
        area = getattr(result, f"{prefix}_area")
        area_err = getattr(result, f"{prefix}_area_err")
        centroid = getattr(result, f"{prefix}_centroid")
        centroid_err = getattr(result, f"{prefix}_centroid_err")
        fwhm = getattr(result, f"{prefix}_fwhm")
        fwhm_err = getattr(result, f"{prefix}_fwhm_err")
        skewness = getattr(result, f"{prefix}_skewness")
        skewness_err = getattr(result, f"{prefix}_skewness_err")
        lines.append(
            f"{label}: area={area:.1f}±{area_err:.1f}, "
            f"centroid={_dual_unit_value(main_window, centroid, centroid_err, is_width=False)}, "
            f"FWHM={_dual_unit_value(main_window, fwhm, fwhm_err, is_width=True, reference_position=centroid)}, "
            f"skewness={skewness:.3g}±{skewness_err:.3g}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Implement — `draw_committed_fits()`**

Replace (currently `fit_mode.py:542-574`):

```python
        for result in spectrum.fits:
            if not result.visible:
                continue
            left_lo, left_hi = result.left_bg_region
            axes.axvspan(
                to_display(left_lo), to_display(left_hi), color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA
            )
            right_lo, right_hi = result.right_bg_region
            axes.axvspan(
                to_display(right_lo), to_display(right_hi), color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA
            )
            fit_lo, fit_hi = result.fit_region
            axes.axvspan(
                to_display(fit_lo), to_display(fit_hi), color=FIT_REGION_COLOR, alpha=FIT_REGION_ALPHA
            )

            if isinstance(result, IntegrationResult):
                lo, hi = result.fit_region
                axes.plot(
                    [to_display(lo), to_display(hi)],
                    [result.background_density, result.background_density],
                    color=bg_line_color, linestyle="--", linewidth=1,
                )
                axes.annotate(
                    f"centroid={result.net_centroid:.1f}\n"
                    f"FWHM={result.net_fwhm:.1f}\n"
                    f"full={result.gross_area:.0f}\nnet={result.net_area:.0f}",
                    xy=(to_display(result.net_centroid), 0.95),
                    xycoords=label_transform,
                    ha="center", va="top",
                    fontsize=7, color=fit_color,
                )
                continue
```

with:

```python
        for result in spectrum.fits:
            if not result.visible:
                continue
            if result.left_bg_region is not None:
                left_lo, left_hi = result.left_bg_region
                axes.axvspan(
                    to_display(left_lo), to_display(left_hi), color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA
                )
                right_lo, right_hi = result.right_bg_region
                axes.axvspan(
                    to_display(right_lo), to_display(right_hi), color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA
                )
            fit_lo, fit_hi = result.fit_region
            axes.axvspan(
                to_display(fit_lo), to_display(fit_hi), color=FIT_REGION_COLOR, alpha=FIT_REGION_ALPHA
            )

            if isinstance(result, IntegrationResult):
                if result.has_background:
                    lo, hi = result.fit_region
                    axes.plot(
                        [to_display(lo), to_display(hi)],
                        [result.background_density, result.background_density],
                        color=bg_line_color, linestyle="--", linewidth=1,
                    )
                    axes.annotate(
                        f"centroid={result.net_centroid:.1f}\n"
                        f"FWHM={result.net_fwhm:.1f}\n"
                        f"full={result.gross_area:.0f}\nnet={result.net_area:.0f}",
                        xy=(to_display(result.net_centroid), 0.95),
                        xycoords=label_transform,
                        ha="center", va="top",
                        fontsize=7, color=fit_color,
                    )
                else:
                    axes.annotate(
                        f"centroid={result.gross_centroid:.1f}\n"
                        f"FWHM={result.gross_fwhm:.1f}\n"
                        f"area={result.gross_area:.0f}",
                        xy=(to_display(result.gross_centroid), 0.95),
                        xycoords=label_transform,
                        ha="center", va="top",
                        fontsize=7, color=fit_color,
                    )
                continue
```

`FitResult` rows always have a non-`None` `left_bg_region` (that dataclass's background handling is unrelated to this feature and untouched), so `if result.left_bg_region is not None:` is always `True` for them — the `FitResult` rendering path below this block (background line + peak curves) is completely unaffected.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode_ui.py -v`

Expected: all tests pass, including the 2 new ones and every pre-existing test — in particular `test_draw_committed_fits_draws_region_shading_and_annotation_for_integration` (+3 patches/+1 line/+1 text, unchanged), `test_draw_committed_fits_skips_a_hidden_integration_result`, and `test_results_table_shows_a_region_row_for_an_integration_result` (still asserts `"Gross:"`/`"Background:"`/`"Net:"` present for the with-background case).

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: simplify Integration plot annotation and tooltip without background"
```

---

## Task 4: `fit_export.py` — omit background/net fields without background

**Files:**
- Modify: `fit_export.py:83-98` (`integration_result_to_json_record()`)
- Modify: `fit_export.py:189-217` (`integration_result_to_text_report()`)
- Test: `tests/test_fit_export.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_export.py` right after `_make_integration_result()` (around line 149):

```python
def _make_integration_result_no_bg(timestamp="2026-07-16T12:00:00"):
    return IntegrationResult(
        left_bg_region=None, right_bg_region=None,
        fit_region=(85.0, 115.0), background_density=0.0,
        gross_area=1000.0, gross_area_err=30.0,
        gross_centroid=100.0, gross_centroid_err=0.5,
        gross_fwhm=8.0, gross_fwhm_err=0.4,
        gross_skewness=0.0, gross_skewness_err=0.1,
        background_area=0.0, background_area_err=0.0,
        background_centroid=0.0, background_centroid_err=0.0,
        background_fwhm=0.0, background_fwhm_err=0.0,
        background_skewness=0.0, background_skewness_err=0.0,
        net_area=1000.0, net_area_err=30.0,
        net_centroid=100.0, net_centroid_err=0.5,
        net_fwhm=8.0, net_fwhm_err=0.4,
        net_skewness=0.0, net_skewness_err=0.1,
        timestamp=timestamp,
    )


def test_integration_result_to_json_record_omits_background_and_net_without_background():
    result = _make_integration_result_no_bg()
    record = integration_result_to_json_record(result, "eu.spe")
    assert record["type"] == "integration"
    assert record["gross"]["area"] == 1000.0
    assert "left_bg_region" not in record
    assert "right_bg_region" not in record
    assert "background_density" not in record
    assert "background" not in record
    assert "net" not in record


def test_integration_result_to_text_report_collapses_to_a_single_area_block_without_background():
    result = _make_integration_result_no_bg()
    report = integration_result_to_text_report(result, "eu.spe", fit_number=1)
    assert "Integration" in report
    assert "Gross:" not in report
    assert "Background:" not in report
    assert "Net:" not in report
    assert "Area:" in report
    assert "1000" in report


def test_append_auto_log_handles_an_integration_result_without_background(tmp_path):
    spectrum_path = str(tmp_path / "eu.spe")
    append_auto_log(spectrum_path, _make_integration_result_no_bg())

    log_path = auto_log_path(spectrum_path)
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "integration"
    assert "background" not in record
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_export.py -k without_background -v`

Expected: all 3 new tests FAIL with `TypeError` — `test_integration_result_to_json_record_omits_background_and_net_without_background` on `list(result.left_bg_region)` (`'NoneType' object is not iterable`); `test_integration_result_to_text_report_collapses_to_a_single_area_block_without_background` on `result.left_bg_region[0]` (`'NoneType' object is not subscriptable`); `test_append_auto_log_handles_an_integration_result_without_background` on the same crash, propagated through `append_auto_log`.

- [ ] **Step 3: Implement — `integration_result_to_json_record()`**

Replace (currently `fit_export.py:83-98`):

```python
def integration_result_to_json_record(result, spectrum_path, calibration=None):
    """Converts one IntegrationResult into a plain dict covering the
    full gross/background/net breakdown, ready for json.dumps() -- the
    Integration-mode analog of fit_result_to_json_record."""
    return {
        "type": "integration",
        "timestamp": result.timestamp,
        "spectrum_path": spectrum_path,
        "left_bg_region": list(result.left_bg_region),
        "right_bg_region": list(result.right_bg_region),
        "fit_region": list(result.fit_region),
        "background_density": result.background_density,
        "gross": _integration_layer_record(result, "gross", calibration),
        "background": _integration_layer_record(result, "background", calibration),
        "net": _integration_layer_record(result, "net", calibration),
    }
```

with:

```python
def integration_result_to_json_record(result, spectrum_path, calibration=None):
    """Converts one IntegrationResult into a plain dict covering the
    full gross/background/net breakdown, ready for json.dumps() -- the
    Integration-mode analog of fit_result_to_json_record. When
    `result.has_background` is False, the background-region/density and
    "background"/"net" keys are omitted entirely rather than written as
    null or zero -- matching TV's own choice to suppress, not zero, a
    background breakdown that doesn't exist."""
    record = {
        "type": "integration",
        "timestamp": result.timestamp,
        "spectrum_path": spectrum_path,
        "fit_region": list(result.fit_region),
        "gross": _integration_layer_record(result, "gross", calibration),
    }
    if not result.has_background:
        return record
    record["left_bg_region"] = list(result.left_bg_region)
    record["right_bg_region"] = list(result.right_bg_region)
    record["background_density"] = result.background_density
    record["background"] = _integration_layer_record(result, "background", calibration)
    record["net"] = _integration_layer_record(result, "net", calibration)
    return record
```

- [ ] **Step 4: Implement — `integration_result_to_text_report()`**

Replace (currently `fit_export.py:189-217`):

```python
def integration_result_to_text_report(result, spectrum_path, fit_number=1, calibration=None):
    """Human-readable report for one Integration result -- the
    Integration-mode analog of fit_result_to_text_report."""
    lines = [
        f"Fit {fit_number} (Integration)",
        f"Spectrum: {spectrum_path}",
        f"Timestamp: {result.timestamp}",
        f"Fit region: [{result.fit_region[0]:.2f}, {result.fit_region[1]:.2f}]",
        f"Left background region: [{result.left_bg_region[0]:.2f}, {result.left_bg_region[1]:.2f}]",
        f"Right background region: [{result.right_bg_region[0]:.2f}, {result.right_bg_region[1]:.2f}]",
        f"Background density: {result.background_density:.6g}",
        "",
    ]
    for label, prefix in (("Gross", "gross"), ("Background", "background"), ("Net", "net")):
        centroid = getattr(result, f"{prefix}_centroid")
        lines.append(f"  {label}:")
        lines.append(
            f"    Area:      {_format_err(getattr(result, f'{prefix}_area'), getattr(result, f'{prefix}_area_err'))}"
        )
        lines.append(
            f"    Centroid:  {_format_dual(centroid, getattr(result, f'{prefix}_centroid_err'), calibration, is_width=False)}"
        )
        lines.append(
            f"    FWHM:      {_format_dual(getattr(result, f'{prefix}_fwhm'), getattr(result, f'{prefix}_fwhm_err'), calibration, is_width=True, reference_position=centroid)}"
        )
        lines.append(
            f"    Skewness:  {_format_err(getattr(result, f'{prefix}_skewness'), getattr(result, f'{prefix}_skewness_err'))}"
        )
    return "\n".join(lines)
```

with:

```python
def integration_result_to_text_report(result, spectrum_path, fit_number=1, calibration=None):
    """Human-readable report for one Integration result -- the
    Integration-mode analog of fit_result_to_text_report. When
    `result.has_background` is False, the background-region/density
    lines and the Gross/Background/Net breakdown collapse to a single
    unlabeled Area/Centroid/FWHM/Skewness block, matching TV's own
    suppress-the-row choice."""
    lines = [
        f"Fit {fit_number} (Integration)",
        f"Spectrum: {spectrum_path}",
        f"Timestamp: {result.timestamp}",
        f"Fit region: [{result.fit_region[0]:.2f}, {result.fit_region[1]:.2f}]",
    ]
    if result.has_background:
        lines.append(f"Left background region: [{result.left_bg_region[0]:.2f}, {result.left_bg_region[1]:.2f}]")
        lines.append(f"Right background region: [{result.right_bg_region[0]:.2f}, {result.right_bg_region[1]:.2f}]")
        lines.append(f"Background density: {result.background_density:.6g}")
    lines.append("")
    if not result.has_background:
        centroid = result.gross_centroid
        lines.append(f"  Area:      {_format_err(result.gross_area, result.gross_area_err)}")
        lines.append(f"  Centroid:  {_format_dual(centroid, result.gross_centroid_err, calibration, is_width=False)}")
        lines.append(
            f"  FWHM:      {_format_dual(result.gross_fwhm, result.gross_fwhm_err, calibration, is_width=True, reference_position=centroid)}"
        )
        lines.append(f"  Skewness:  {_format_err(result.gross_skewness, result.gross_skewness_err)}")
        return "\n".join(lines)
    for label, prefix in (("Gross", "gross"), ("Background", "background"), ("Net", "net")):
        centroid = getattr(result, f"{prefix}_centroid")
        lines.append(f"  {label}:")
        lines.append(
            f"    Area:      {_format_err(getattr(result, f'{prefix}_area'), getattr(result, f'{prefix}_area_err'))}"
        )
        lines.append(
            f"    Centroid:  {_format_dual(centroid, getattr(result, f'{prefix}_centroid_err'), calibration, is_width=False)}"
        )
        lines.append(
            f"    FWHM:      {_format_dual(getattr(result, f'{prefix}_fwhm'), getattr(result, f'{prefix}_fwhm_err'), calibration, is_width=True, reference_position=centroid)}"
        )
        lines.append(
            f"    Skewness:  {_format_err(getattr(result, f'{prefix}_skewness'), getattr(result, f'{prefix}_skewness_err'))}"
        )
    return "\n".join(lines)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_export.py -v`

Expected: all tests pass, including the 3 new ones and every pre-existing test — in particular `test_integration_result_to_json_record_includes_gross_background_net` and `test_integration_result_to_text_report_includes_all_three_layers` (with-background case, unchanged), and the two `fwhm_keV`-derivative regression guards.

- [ ] **Step 6: Add the end-to-end regression test relocated from Task 2**

Task 2 originally included a full `run_integration()` end-to-end test, but it couldn't pass there: `run_integration()` also calls `fit_export.append_auto_log()` (crashes on `None` bg regions until this task's Steps 3-4) and `draw_committed_fits()` via `_plot_data()` (crashed until Task 3). By this point in the plan, Tasks 1-3 and this task's own Steps 3-4 are all done, so all three prerequisites are finally satisfied — this is the right place for it.

Add to `tests/test_fit_mode_ui.py` (after `test_run_integration_appends_an_integration_result`, around line 2319 — the same anchor point Task 2 used for its own tests, so this lands right after them):

```python
def test_run_integration_with_no_background_produces_a_backgroundless_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()

    assert len(spectrum.fits) == 1
    result = spectrum.fits[0]
    assert isinstance(result, IntegrationResult)
    assert result.has_background is False
    assert result.left_bg_region is None
    assert result.right_bg_region is None
    assert result.net_area == result.gross_area
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode_ui.py -k with_no_background_produces -v`

Expected: PASSES immediately — Tasks 1-3 plus this task's own Steps 3-4 already provide everything this test needs, so there is no separate implementation step here, just adding and confirming.

- [ ] **Step 7: Commit**

```bash
git add fit_export.py tests/test_fit_export.py tests/test_fit_mode_ui.py
git commit -m "feat: omit background/net fields from Integration export without background"
```

---

## Task 4b: `fit_mode.py` — fix `_on_result_double_clicked()` for zero-background results

**Found during Task 3's code-quality review, not anticipated when this plan was first written.** `_on_result_double_clicked()` restores a previously-computed result's marks for editing via `self.state.bg_regions = [result.left_bg_region, result.right_bg_region]` (`fit_mode.py:1042`), unconditionally. For a zero-background `IntegrationResult`, this produces `bg_regions = [None, None]` — a 2-element list of `None`s, not the empty list `[]` that `FitModeState` treats as "no background" everywhere else (`ordered_bg_regions()`, `ready_to_fit()`, `ready_to_integrate()`). Immediately after, `_redraw_progress()` (`fit_mode.py:455`) does `for lo, hi in state.bg_regions:`, which raises `TypeError: cannot unpack non-iterable NoneType object` when unpacking the first `None`. This is reachable through a completely ordinary workflow — double-clicking a Fit Results row to re-edit it, already exercised for the with-background case by `test_double_click_reloads_a_committed_fit_for_editing`. This task is placed *after* Task 4 rather than alongside Task 3 because its own test needs `run_integration()` to succeed without crashing first (i.e. it needs Task 4's export fix as a prerequisite, the same reason Task 4's Step 6 test was relocated from Task 2).

**Files:**
- Modify: `fit_mode.py:1042`
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py` (after `test_double_click_reloads_a_committed_fit_for_editing`, around line 1013):

```python
def test_double_click_reloads_a_zero_background_integration_result_without_crashing(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()

    main_window.fit_controller.reset_marks()
    assert main_window.fit_controller.state.bg_regions == []

    item = main_window.fit_controller.results_table.item(0, 0)
    main_window.fit_controller._on_result_double_clicked(item)

    assert main_window.fit_controller.state.bg_regions == []
    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode_ui.py -k zero_background_integration_result_without_crashing -v`

Expected: FAILS with `TypeError: cannot unpack non-iterable NoneType object` (from `_redraw_progress()`'s `for lo, hi in state.bg_regions:`, triggered via `_on_result_double_clicked()` setting `bg_regions = [None, None]`).

- [ ] **Step 3: Implement**

Replace (currently `fit_mode.py:1029-1043`):

```python
    def _on_result_double_clicked(self, item):
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        fit_index = self._results_row_fit_index[item.row()]
        result = active.fits[fit_index]

        self._clear_progress()

        # Bypasses the click-pairing API (add_bg_click/add_fit_click)
        # deliberately -- this restores a previously-computed,
        # already-valid state wholesale, not a fresh in-progress click
        # sequence.
        self.state.bg_regions = [result.left_bg_region, result.right_bg_region]
        self.state.fit_region = result.fit_region
```

with:

```python
    def _on_result_double_clicked(self, item):
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        fit_index = self._results_row_fit_index[item.row()]
        result = active.fits[fit_index]

        self._clear_progress()

        # Bypasses the click-pairing API (add_bg_click/add_fit_click)
        # deliberately -- this restores a previously-computed,
        # already-valid state wholesale, not a fresh in-progress click
        # sequence. An empty list (not [None, None]) is bg_regions'
        # own canonical "no background" representation everywhere else
        # in FitModeState (see ordered_bg_regions()) -- restoring a
        # zero-background result must produce [], or _redraw_progress()'s
        # `for lo, hi in state.bg_regions:` unpack crashes on None.
        self.state.bg_regions = (
            [] if result.left_bg_region is None
            else [result.left_bg_region, result.right_bg_region]
        )
        self.state.fit_region = result.fit_region
```

(The rest of the method — the `isinstance(result, IntegrationResult)` branch and everything below it — is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fit_mode_ui.py -v`

Expected: all tests pass, including the new one and every pre-existing test — in particular `test_double_click_reloads_a_committed_fit_for_editing` (with-background double-click restore, unchanged).

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "fix: restore empty bg_regions, not [None, None], when double-clicking a zero-background Integration result"
```

---

## Task 5: `help_content.py` — document the optional background in the HowTo page

**Files:**
- Modify: `help_content.py:236-241` (the "8. Integration" HowTo section)
- Test: `tests/test_help_content.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_help_content.py` (after `test_howto_html_covers_every_operation`, around line 86):

```python
def test_howto_integration_section_mentions_optional_background():
    html = build_howto_html()
    assert "Background regions are optional" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_help_content.py -k optional_background -v`

Expected: FAIL — the current "8. Integration" section has no such sentence.

- [ ] **Step 3: Implement**

Replace (currently `help_content.py:236-241`):

```python
<h3>8. Integration</h3>
<p><kbd>Ctrl+I</kbd> computes gross/background/net counts across the
marked regions directly (background centroid, FWHM, skewness, and area,
all with uncertainties) <i>without</i> fitting a peak shape -- faster,
and useful when a peak is too irregular to fit well, or when you only
need a total count rather than individual peak parameters.</p>
```

with:

```python
<h3>8. Integration</h3>
<p><kbd>Ctrl+I</kbd> computes gross/background/net counts across the
marked regions directly (background centroid, FWHM, skewness, and area,
all with uncertainties) <i>without</i> fitting a peak shape -- faster,
and useful when a peak is too irregular to fit well, or when you only
need a total count rather than individual peak parameters. Background
regions are optional: marking only a fit region and pressing
<kbd>Ctrl+I</kbd> reports the raw area and centroid (plus FWHM and
skewness) with no background subtraction; marking two background
regions first still works exactly as before.</p>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_help_content.py -v`

Expected: all tests pass, including the new one and the existing balanced-HTML-tags check (`_TagBalanceChecker`) and `test_howto_html_covers_every_operation`.

- [ ] **Step 5: Commit**

```bash
git add help_content.py tests/test_help_content.py
git commit -m "docs: document background-optional Integration in the HowTo page"
```

---

## Task 6: Windows verification (primary focus)

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass, including everything added in Tasks 1-5 and Task 4b.

- [ ] **Step 2: Rebuild and manually exercise the app**

```bash
powershell -File packaging\windows\build.ps1
```

Then run `dist\SpectraTools.exe` directly (matching this project's existing smoke-test pattern) and manually verify:
- Load a spectrum. Mark only a fit region (`R`, click twice) — no `B` marks at all. Confirm `Ctrl+I` (the Integrate button) is enabled with zero background regions marked.
- Run Integration. Confirm a new row appears in Fit Results with sensible Position/FWHM/Volume values (no crash, no NaN).
- Hover the row's first cell — confirm the tooltip is a single unlabeled line (`Area=...`, `centroid=...`, `FWHM=...`, `skewness=...`), not a 3-line Gross/Background/Net breakdown.
- Check the plot annotation near the result — confirm it reads `centroid=...\nFWHM=...\narea=...` (one area number), with no dashed background-density line drawn and no green background-region shading (only the blue fit-region span).
- Export this result (`Ctrl+E` or the equivalent single-fit export) and open the resulting `.txt`/check the auto-log `.jsonl` next to the spectrum file — confirm the text report has no "Background:"/"Net:" section or background-region lines, and the JSON record has no `"background"`/`"net"`/`"left_bg_region"`/`"right_bg_region"`/`"background_density"` keys.
- Double-click the zero-background result's row in Fit Results to reload it for editing — confirm this restores the fit region mark with no crash (this exact workflow crashed until Task 4b; deliberately verify it, don't skip it).
- Now mark exactly one background region (`B`, click twice) plus the fit region, and confirm `Ctrl+I` is **disabled** (1 region is still not a valid count).
- Mark a second background region (2 total) plus the fit region and run Integration again — confirm the existing with-background behavior is completely unchanged: 3-line Gross/Background/Net tooltip, dashed background line + green shading on the plot, and full breakdown in the export.
- Double-click that with-background result's row too — confirm it still restores both background regions correctly (regression check for Task 4b's change).
- Open the HowTo page (F1) and confirm the "8. Integration" section now mentions optional background marks, reads correctly, and the rest of the page still renders with no broken HTML.

- [ ] **Step 3: No commit** (verification-only task; fix and re-verify if anything above fails).

---

## Task 7: Linux verification (proportionate)

**Files:** none (verification only)

- [ ] **Step 1: Rebuild via the existing scripts**

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d AlmaLinux-8 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/build.sh"
```
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/build_deb.sh"
```

- [ ] **Step 2: Smoke-test the DEB on Ubuntu-24.04**

Write a verification script to `packaging/linux/output/_task7_verify_ubuntu.sh` and run it via `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u rig -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/_task7_verify_ubuntu.sh"` (per this project's established WSL invocation rules: `MSYS_NO_PATHCONV=1`, a real script file rather than inline `bash -c`, foreground execution):

```bash
#!/usr/bin/env bash
set -uo pipefail

echo "=== reinstalling the freshly built DEB ==="
apt-get install -y --reinstall /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/spectratools_*.deb > /tmp/task7_install_ubuntu.log 2>&1
echo "install exit: $?"
tail -5 /tmp/task7_install_ubuntu.log

echo "=== confirming the app still launches correctly (unrelated to this feature, but a cheap regression check) ==="
rm -f /tmp/task7_stdout.log /tmp/task7_stderr.log
timeout -k 1 8 spectratools >/tmp/task7_stdout.log 2>/tmp/task7_stderr.log
echo "EXIT CODE: $?"
cat /tmp/task7_stderr.log
```

Expected: install succeeds, `EXIT CODE: 124` (genuinely alive, not crashed), stderr only the known-benign Fontconfig/Qt warnings already documented from prior verification (no new errors).

- [ ] **Step 3: Smoke-test the RPM on AlmaLinux-10** (cross-version compatibility check — built on AlmaLinux-8's older glibc, verified running on the newer AlmaLinux-10)

Write a verification script to `packaging/linux/output/_task7_verify_almalinux10.sh` and run it via `MSYS_NO_PATHCONV=1 wsl.exe -d AlmaLinux-10 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/_task7_verify_almalinux10.sh"`:

```bash
#!/usr/bin/env bash
set -uo pipefail

echo "=== installing the freshly built RPM (install if absent, reinstall if the same version is already present) ==="
dnf install -y /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/spectratools-*.rpm > /tmp/task7_install_al10.log 2>&1 \
  || dnf reinstall -y /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/spectratools-*.rpm >> /tmp/task7_install_al10.log 2>&1
echo "install exit: $?"
tail -5 /tmp/task7_install_al10.log

echo "=== confirming the app still launches correctly ==="
rm -f /tmp/task7_stdout_al10.log /tmp/task7_stderr_al10.log
timeout -k 1 8 spectratools >/tmp/task7_stdout_al10.log 2>/tmp/task7_stderr_al10.log
echo "EXIT CODE: $?"
cat /tmp/task7_stderr_al10.log
```

Expected: install succeeds, `EXIT CODE: 124`, no new stderr errors beyond the already-documented benign warnings.

This is a launch-level regression check, not a full manual UI exercise of Integration itself on Linux — per the design spec, that depth of Linux-specific verification isn't warranted for a pure application-code change (no new packaging surface here), and Task 6 already covers the feature's actual behavior thoroughly.

Delete the throwaway scripts when done:
```bash
rm packaging/linux/output/_task7_verify_ubuntu.sh packaging/linux/output/_task7_verify_almalinux10.sh
```

- [ ] **Step 4: No commit** (verification-only task).

---

## Final Step: Hand back to the user

Once all 7 tasks are complete, report back: what was built, the Windows and Linux verification results, and that this is the first piece of v2.1.2 (not yet frozen/released — that's a separate, later step following the established release process, including the CHANGELOG.md update, only after the user has manually confirmed the feature works).
