# Integration Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TV-parity "Integration" mode — a direct background-subtracted
region sum (area, centroid, width, skewness, each for gross/background/net
layers) as an alternative to Gaussian peak fitting, triggered by a new
`Ctrl+I` "Integrate" action alongside the existing "Fit".

**Architecture:** A new pure function `integrate_region()` and
`IntegrationResult` dataclass in `peak_fit.py` (no Qt dependency, mirrors
`fit_peaks()`/`FitResult`), ported line-by-line from TV's own
`FIIntegrateRegion` (`tv-1.9.13/lib/tv/vsFitInt.c`) plus its display-layer
sigma/FWHM conversion (`vsFitFmt.c`). `fit_mode.py` gains a new
`run_integration()` alongside `run_fit()`, reusing the existing
`bg_regions`/`fit_region` marks, and `isinstance(result, IntegrationResult)`
branches in the three places `FitResult`-shaped results are currently
handled (results table, plot rendering, double-click reload).
`fit_export.py` gains parallel serialization functions, dispatched the
same way, so the existing auto-log and "Export This/All Fits" actions
pick up Integration results automatically. `LoadedSpectrum.fits` becomes
a mixed list of `FitResult`/`IntegrationResult` objects (not renamed, per
the design spec).

**Tech Stack:** Python, NumPy, PySide6 (Qt), pytest.

**Design spec:** `docs/superpowers/specs/2026-07-16-integration-mode-design.md`
— every formula below was independently prototyped and numerically
validated against a hand-computable example, a flat-background sanity
check, an analytic-peak-recovery check, and a dedicated check proving
the background-moment "quirk" (net's 2nd moment reused for the
background's own higher-moment uncertainty) is both real and correctly
reproduced, *before* this plan was written — the exact validated code is
what appears in Task 1 below.

---

### Task 1: `IntegrationResult` dataclass and `integrate_region()` in `peak_fit.py`

**Files:**
- Modify: `peak_fit.py` (add dataclass + function; no changes to existing code)
- Test: `tests/test_peak_fit.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_peak_fit.py`:

```python
from peak_fit import FWHM_FACTOR, IntegrationResult, integrate_region


def test_integration_result_holds_expected_fields():
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
    assert result.net_area == 80.0
    assert result.timestamp is None
    assert result.visible is True


def test_integrate_region_flat_background_gives_near_zero_net_area():
    x = np.arange(200, dtype=float)
    y = np.full(200, 20.0)
    result = integrate_region(
        x, y, left_bg_region=(10.0, 30.0), right_bg_region=(170.0, 190.0),
        fit_region=(85.0, 115.0),
    )
    assert result.net_area == pytest.approx(0.0, abs=1e-6)
    assert result.background_centroid == pytest.approx(100.0, abs=0.6)


def test_integrate_region_recovers_analytic_peak_area():
    x, y = _make_spectrum(
        channels=200, peaks=[(5000.0, 100.0, 4.0)], slope=0.0, intercept=50.0,
    )
    result = integrate_region(
        x, y, left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
    )
    analytic_area = 5000.0 * 4.0 * np.sqrt(2 * np.pi)
    assert result.net_area == pytest.approx(analytic_area, rel=0.05)
    assert result.net_centroid == pytest.approx(100.0, abs=1.0)
    assert result.net_fwhm == pytest.approx(FWHM_FACTOR * 4.0, abs=1.0)


def test_integrate_region_matches_hand_computed_moments():
    x = np.arange(10, dtype=float)
    y = np.array([0., 0., 10., 20., 10., 0., 0., 0., 0., 0.])
    result = integrate_region(
        x, y, left_bg_region=(8.0, 8.0), right_bg_region=(9.0, 9.0),
        fit_region=(2.0, 4.0),
    )
    # s=[10,20,10] at i=[2,3,4]: gross_area=40, centroid=(20+60+40)/40=3.0,
    # mom2=((2-3)^2*10+(3-3)^2*20+(4-3)^2*10)/40=20/40=0.5
    assert result.gross_area == 40.0
    assert result.gross_centroid == pytest.approx(3.0)
    assert result.gross_fwhm == pytest.approx(np.sqrt(0.5) * FWHM_FACTOR)
    # Both bg regions are exactly 0, so net == gross exactly.
    assert result.net_area == 40.0
    assert result.net_centroid == pytest.approx(3.0)


def test_integrate_region_all_zero_does_not_raise():
    x = np.arange(50, dtype=float)
    y = np.zeros(50)
    result = integrate_region(
        x, y, left_bg_region=(5.0, 8.0), right_bg_region=(40.0, 43.0),
        fit_region=(20.0, 25.0),
    )
    assert result.gross_area == 0.0
    assert result.net_area == 0.0


def test_integrate_region_rejects_empty_fit_region():
    x = np.arange(200, dtype=float)
    y = np.full(200, 20.0)
    with pytest.raises(FitError):
        integrate_region(
            x, y, left_bg_region=(10.0, 30.0), right_bg_region=(170.0, 190.0),
            fit_region=(500.0, 501.0),
        )


def test_integrate_region_background_uncertainty_scales_linearly_with_region_width():
    """Deliberately verifies TV's own (statistically non-standard)
    background-uncertainty formula is preserved exactly: background_area_err
    squared scales linearly with the fit region's channel count, not
    quadratically as strict error propagation for a scaled mean would
    require -- a future "fix" of this would break this test on purpose."""
    x = np.arange(20, dtype=float)
    y = np.full(20, 5.0)
    y[2] = 100.0
    y[17] = 100.0
    result = integrate_region(
        x, y, left_bg_region=(2.0, 2.0), right_bg_region=(17.0, 17.0),
        fit_region=(8.0, 10.0),
    )
    # bg_chn=2, bg_count=200, bg_density=100, bg_density_var=200/(2*2)=50
    # n_region=3 channels (8,9,10) -> background_area=300,
    # background_area_err^2 = 50*3 = 150 (linear, not 50*3^2=450)
    assert result.background_area == pytest.approx(300.0)
    assert result.background_area_err == pytest.approx(np.sqrt(150.0))


def test_integrate_region_background_moment_uncertainty_reuses_net_second_moment():
    """Deliberately verifies another TV quirk: the background layer's own
    width/skewness uncertainty terms reuse the *net* distribution's 2nd
    moment rather than the background's own -- confirmed against TV's
    source (vsFitInt.c:281,303) and cross-checked numerically against an
    independently-coded "corrected" alternative during design (the two
    differ by more than 30% for this fixture; this test pins the exact
    TV-parity value so a future "fix" would fail loudly)."""
    x = np.arange(60, dtype=float)
    y = np.full(60, 30.0)
    y[5] = 40.0
    y[6] = 20.0
    y[50] = 25.0
    y[51] = 35.0
    y[25:31] += np.array([50, 150, 300, 300, 150, 50])
    result = integrate_region(
        x, y, left_bg_region=(5.0, 6.0), right_bg_region=(50.0, 51.0),
        fit_region=(20.0, 36.0),
    )
    assert result.background_fwhm_err == pytest.approx(0.3305131157646951)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_peak_fit.py -k "integrate_region or integration_result" -v`
Expected: FAIL with `ImportError: cannot import name 'IntegrationResult'`

- [ ] **Step 3: Implement `IntegrationResult` and `integrate_region()`**

Add to `peak_fit.py`, after the existing `FitResult` dataclass (after line 90):

```python
@dataclass
class IntegrationResult:
    left_bg_region: tuple
    right_bg_region: tuple
    fit_region: tuple
    background_density: float
    gross_area: float
    gross_area_err: float
    gross_centroid: float
    gross_centroid_err: float
    gross_fwhm: float
    gross_fwhm_err: float
    gross_skewness: float
    gross_skewness_err: float
    background_area: float
    background_area_err: float
    background_centroid: float
    background_centroid_err: float
    background_fwhm: float
    background_fwhm_err: float
    background_skewness: float
    background_skewness_err: float
    net_area: float
    net_area_err: float
    net_centroid: float
    net_centroid_err: float
    net_fwhm: float
    net_fwhm_err: float
    net_skewness: float
    net_skewness_err: float
    timestamp: str = None
    visible: bool = True
```

Add near the end of `peak_fit.py` (after `fit_result_values_by_name`, before the `_ParamDamping` class — this keeps the public API grouped together, matching the existing file organization):

```python
def integrate_region(x, y, left_bg_region, right_bg_region, fit_region):
    """TV-style direct background-subtracted region sum, ported
    line-by-line from FIIntegrateRegion's no-fitted-background branch
    (tv-1.9.13/lib/tv/vsFitInt.c:19-51,194-309) plus the sigma/FWHM
    conversion from ParseIntPeak (tv-1.9.13/lib/tv/vsFitFmt.c:340-392).
    Several arithmetic choices below look unusual (asymmetric plain-sum
    vs abs(sum) normalization between layers/moments; the background's
    own higher-moment uncertainty terms reusing the *net* distribution's
    2nd moment rather than its own; the background-sum uncertainty
    scaling linearly rather than quadratically with region width) --
    these are deliberate, source-verified TV-parity choices, not bugs,
    per the design spec. Independently validated (hand-computed moments,
    flat-background/analytic-peak sanity checks, and a dedicated
    cross-check proving the moment-reuse quirk is real) before this
    function was written -- see the design spec's Testing section."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lo, hi = fit_region
    mask = (x >= lo) & (x <= hi)
    idx = x[mask]
    s = y[mask]
    ds = s.copy()  # Poisson variance per channel = the count itself
    n = idx.size
    if n == 0:
        raise FitError(f"Fit region {fit_region} contains no data")

    # ---- gross sum & variance (vsFitInt.c:41-43) ----
    gross_sum = float(np.sum(s))
    gross_dsum = float(np.sum(ds))
    gross_area = gross_sum
    gross_area_err = math.sqrt(gross_dsum)

    # ---- gross moments (vsFitInt.c:45-83) ----
    g_M1 = g_DM1 = g_M2 = g_DM2 = g_M3 = g_DM3 = 0.0
    if gross_sum != 0.0:
        g_M1 = float(np.sum(idx * s)) / gross_sum

        dlt = idx - g_M1
        dlt2 = dlt ** 2
        dMom1 = float(np.sum(dlt2 * ds))
        mom2_raw = float(np.sum(dlt2 * s))
        g_DM1 = math.sqrt(dMom1) / abs(gross_sum)
        g_M2 = mom2_raw / gross_sum  # plain sum, not abs

        dlt3 = dlt2 * dlt
        dDlt_m2 = dlt2 - g_M2
        dMom2 = float(np.sum((dDlt_m2 ** 2) * ds))
        mom3_raw = float(np.sum(dlt3 * s))
        g_DM2 = math.sqrt(dMom2) / abs(gross_sum)
        g_M3 = mom3_raw / abs(gross_sum)  # abs this time

        term = dlt * (dlt2 - 3.0 * g_M2) - g_M3
        dMom3 = float(np.sum((term ** 2) * ds))
        g_DM3 = math.sqrt(dMom3) / abs(gross_sum)

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

    background_area = bg_density * n
    background_area_var = bg_density_var * n  # linear, not squared (TV quirk, kept)
    background_area_err = math.sqrt(background_area_var)

    net_sum = gross_sum - background_area
    net_dsum = gross_dsum + background_area_var
    net_area = net_sum
    net_area_err = math.sqrt(net_dsum)

    # ---- background & net moments (vsFitInt.c:223-309) ----
    bg_M1 = bg_DM1 = bg_M2 = bg_DM2 = bg_M3 = bg_DM3 = 0.0
    n_M1 = n_DM1 = n_M2 = n_DM2 = n_M3 = n_DM3 = 0.0
    if net_sum != 0.0 or background_area != 0.0:
        b = bg_density
        db = bg_density_var
        bg_sum = background_area

        mom1_raw = float(np.sum(idx * (s - b)))
        bgmom1_raw = float(np.sum(idx * b))
        if bg_sum != 0.0:
            bg_M1 = bgmom1_raw / abs(bg_sum)
        if net_sum != 0.0:
            n_M1 = mom1_raw / abs(net_sum)

        dlt = idx - n_M1
        dltb = idx - bg_M1
        dMom1 = float(np.sum((dlt ** 2) * (ds + db)))
        mom2_raw = float(np.sum((dlt ** 2) * (s - b)))
        dBgMom1 = float(np.sum((dltb ** 2) * db))
        bgmom2_raw = float(np.sum((dltb ** 2) * b))
        if bg_sum != 0.0:
            bg_DM1 = math.sqrt(dBgMom1) / abs(bg_sum)
            bg_M2 = bgmom2_raw / abs(bg_sum)  # abs (unlike gross's plain-sum M2)
        if net_sum != 0.0:
            n_DM1 = math.sqrt(dMom1) / abs(net_sum)
            n_M2 = mom2_raw / net_sum  # plain sum, matches gross's convention

        dlt2 = dlt ** 2
        dltb2 = dltb ** 2
        dDlt_m2 = dlt2 - n_M2
        dDltb_m2 = dltb2 - n_M2  # CONFIRMED TV QUIRK: net's M2, not bg's own
        dMom2 = float(np.sum((dDlt_m2 ** 2) * (ds + db)))
        mom3_raw = float(np.sum((dlt2 * dlt) * (s - b)))
        dBgMom2 = float(np.sum((dDltb_m2 ** 2) * db))
        bgmom3_raw = float(np.sum((dltb2 * dltb) * b))
        if bg_sum != 0.0:
            bg_DM2 = math.sqrt(dBgMom2) / abs(bg_sum)
            bg_M3 = bgmom3_raw / abs(bg_sum)
        if net_sum != 0.0:
            n_DM2 = math.sqrt(dMom2) / abs(net_sum)
            n_M3 = mom3_raw / abs(net_sum)

        term = dlt * (dlt2 - 3.0 * n_M2) - n_M3
        # CONFIRMED TV QUIRK: net's M2 again, but bg's OWN M3 here.
        termb = dltb * (dltb2 - 3.0 * n_M2) - bg_M3
        dMom3 = float(np.sum((term ** 2) * (ds + db)))
        dBgMom3 = float(np.sum((termb ** 2) * db))
        if bg_sum != 0.0:
            bg_DM3 = math.sqrt(dBgMom3) / abs(bg_sum)
        if net_sum != 0.0:
            n_DM3 = math.sqrt(dMom3) / abs(net_sum)

    def _to_reported(M1, DM1, M2, DM2, M3, DM3):
        centroid, centroid_err = M1, DM1
        sigma = math.sqrt(M2) if M2 >= 0.0 else -math.sqrt(-M2)
        sigma_err = DM2 / abs(sigma) if sigma != 0.0 else 0.0
        fwhm = sigma * FWHM_FACTOR
        fwhm_err = sigma_err * FWHM_FACTOR
        return centroid, centroid_err, fwhm, fwhm_err, M3, DM3

    g_centroid, g_centroid_err, g_fwhm, g_fwhm_err, g_skew, g_skew_err = _to_reported(
        g_M1, g_DM1, g_M2, g_DM2, g_M3, g_DM3
    )
    bg_centroid, bg_centroid_err, bg_fwhm, bg_fwhm_err, bg_skew, bg_skew_err = _to_reported(
        bg_M1, bg_DM1, bg_M2, bg_DM2, bg_M3, bg_DM3
    )
    n_centroid, n_centroid_err, n_fwhm, n_fwhm_err, n_skew, n_skew_err = _to_reported(
        n_M1, n_DM1, n_M2, n_DM2, n_M3, n_DM3
    )

    return IntegrationResult(
        left_bg_region=tuple(left_bg_region), right_bg_region=tuple(right_bg_region),
        fit_region=tuple(fit_region), background_density=float(bg_density),
        gross_area=float(gross_area), gross_area_err=float(gross_area_err),
        gross_centroid=float(g_centroid), gross_centroid_err=float(g_centroid_err),
        gross_fwhm=float(g_fwhm), gross_fwhm_err=float(g_fwhm_err),
        gross_skewness=float(g_skew), gross_skewness_err=float(g_skew_err),
        background_area=float(background_area), background_area_err=float(background_area_err),
        background_centroid=float(bg_centroid), background_centroid_err=float(bg_centroid_err),
        background_fwhm=float(bg_fwhm), background_fwhm_err=float(bg_fwhm_err),
        background_skewness=float(bg_skew), background_skewness_err=float(bg_skew_err),
        net_area=float(net_area), net_area_err=float(net_area_err),
        net_centroid=float(n_centroid), net_centroid_err=float(n_centroid_err),
        net_fwhm=float(n_fwhm), net_fwhm_err=float(n_fwhm_err),
        net_skewness=float(n_skew), net_skewness_err=float(n_skew_err),
    )
```

No new imports needed — `math` and `numpy as np` are already imported at the top of `peak_fit.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_peak_fit.py -k "integrate_region or integration_result" -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `python -m pytest -q`
Expected: PASS (same pass/xfail counts as before this task, plus the 8 new tests)

- [ ] **Step 6: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "feat: add integrate_region() and IntegrationResult (TV-parity direct sum)"
```

---

### Task 2: Marking gate, Ctrl+I trigger, and `run_integration()`

**Files:**
- Modify: `fit_mode.py:28-98` (`FitModeState`), `fit_mode.py:15-18` (imports), end of `FitModeController` class
- Modify: `main_window.py:438-448` (`_build_fit_mode_buttons`), `main_window.py:453-456` (`_update_fit_mode_availability`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_fit_mode_ui.py`:

```python
def test_ready_to_integrate_needs_bg_regions_and_fit_region_but_not_peaks(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)
    state = main_window.fit_controller.state

    assert state.ready_to_integrate() is False
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    assert state.ready_to_integrate() is False
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    assert state.ready_to_integrate() is True  # no peak marks needed


def test_integrate_button_has_ctrl_i_shortcut_and_is_gated(qapp):
    main_window = MainWindow()
    assert main_window.integrate_button.shortcut().toString() == "Ctrl+I"
    assert main_window.integrate_button.isEnabled() is False

    _make_active_spectrum(main_window)
    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    assert main_window.integrate_button.isEnabled() is True


def test_run_integration_appends_an_integration_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)

    main_window.fit_controller.run_integration()

    assert len(spectrum.fits) == 1
    result = spectrum.fits[0]
    assert isinstance(result, IntegrationResult)
    assert result.timestamp is not None


def test_run_integration_ignores_any_marked_peaks(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    _held_key_click(main_window, "p", 100)  # a peak is marked too

    main_window.fit_controller.run_integration()  # must not crash or use it

    assert len(spectrum.fits) == 1
    assert isinstance(spectrum.fits[0], IntegrationResult)


def test_marks_persist_after_a_successful_integration(qapp):
    main_window = MainWindow()
    _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)

    main_window.fit_controller.run_integration()

    regions = main_window.fit_controller.state.bg_regions
    assert regions[0] == pytest.approx((70.0, 85.0))
    assert regions[1] == pytest.approx((115.0, 130.0))
    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
```

Change `tests/test_fit_mode_ui.py`'s existing import line (line 13) from:

```python
from peak_fit import FWHM_FACTOR, FitResult, PeakResult, hypermet_left_tail
```

to:

```python
from peak_fit import FWHM_FACTOR, FitResult, IntegrationResult, PeakResult, hypermet_left_tail
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fit_mode_ui.py -k "integrate or integration" -v`
Expected: FAIL (`ready_to_integrate`/`integrate_button`/`run_integration` don't exist; `IntegrationResult` import error)

- [ ] **Step 3: Add `ready_to_integrate()` to `FitModeState`**

In `fit_mode.py`, immediately after the existing `ready_to_fit` method (after line 88):

```python
    def ready_to_integrate(self):
        return len(self.bg_regions) == BG_REGION_CAP and self.fit_region is not None
```

- [ ] **Step 4: Add the `Ctrl+I` "Integrate" action to `main_window.py`**

In `main_window.py`, inside `_build_fit_mode_buttons` (after line 448, right after the existing `clear_fit_button` block):

```python
        self.integrate_button = QAction("Integrate", self)
        self.integrate_button.setShortcut("Ctrl+I")
        self.integrate_button.setEnabled(False)
        self.integrate_button.triggered.connect(self.fit_controller.run_integration)
        self.addAction(self.integrate_button)
```

And in `_update_fit_mode_availability` (currently lines 453-456), add the new gate:

```python
    def _update_fit_mode_availability(self):
        active = next((s for s in self.spectra if s.active), None)
        available = active is not None and active.visible
        self.fit_button.setEnabled(available and self.fit_controller.state.ready_to_fit())
        self.integrate_button.setEnabled(available and self.fit_controller.state.ready_to_integrate())
```

- [ ] **Step 5: Add `integrate_region`/`IntegrationResult`/`FitError` to `fit_mode.py`'s imports**

`fit_mode.py`'s existing import (lines 15-18):

```python
from peak_fit import (
    FWHM_FACTOR, FitError, fit_peaks, fit_result_values_by_name, hypermet_left_tail,
    parameter_names,
)
```

becomes:

```python
from peak_fit import (
    FWHM_FACTOR, FitError, IntegrationResult, fit_peaks, fit_result_values_by_name,
    hypermet_left_tail, integrate_region, parameter_names,
)
```

- [ ] **Step 6: Add `run_integration()` to `FitModeController`**

Add at the end of the `FitModeController` class in `fit_mode.py` (after the existing `run_fit` method, i.e. at the end of the file):

```python
    def run_integration(self):
        if not self.state.ready_to_integrate():
            return
        active = next((s for s in self.main_window.spectra if s.active), None)
        if active is None:
            return
        left, right = self.state.ordered_bg_regions()
        x = np.arange(len(active.data), dtype=float)
        y = active.data
        try:
            result = integrate_region(x, y, left, right, self.state.fit_region)
        except FitError as exc:
            self._show_status_message(f"Integration failed: {exc}", 5000)
            return
        result.timestamp = datetime.now().isoformat(timespec="seconds")
        for earlier in active.fits:
            if (
                earlier.left_bg_region == result.left_bg_region
                and earlier.right_bg_region == result.right_bg_region
                and earlier.fit_region == result.fit_region
            ):
                earlier.visible = False
        active.fits.append(result)
        self.main_window._plot_data(preserve_view=True)
```

(The auto-log call is deliberately not added yet — `fit_export.append_auto_log` doesn't know how to serialize an `IntegrationResult` until Task 6. Adding the call now would crash on every integration; Task 6 adds both the serialization and this call together.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_fit_mode_ui.py -k "integrate or integration" -v`
Expected: PASS (5 tests)

- [ ] **Step 8: Run the full existing test suite to confirm no regression**

Run: `python -m pytest -q`
Expected: PASS (same counts as Task 1's end, plus these 5 new tests)

- [ ] **Step 9: Commit**

```bash
git add fit_mode.py main_window.py tests/test_fit_mode_ui.py
git commit -m "feat: add Ctrl+I Integrate action and run_integration()"
```

---

### Task 3: Results table row for an Integration result

**Files:**
- Modify: `fit_mode.py:560-599` (`update_results_list`), add a new `_integration_tooltip` helper
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_results_table_shows_a_region_row_for_an_integration_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()

    table = main_window.fit_controller.results_table
    assert table.rowCount() == 1
    result = spectrum.fits[0]
    assert table.item(0, 1).text() == "region"
    assert table.item(0, 2).text() == f"{result.net_centroid:.2f} ± {result.net_centroid_err:.2f}"
    assert table.item(0, 3).text() == f"{result.net_fwhm:.2f} ± {result.net_fwhm_err:.2f}"
    assert table.item(0, 4).text() == f"{result.net_area:.1f} ± {result.net_area_err:.1f}"
    tooltip = table.item(0, 0).toolTip()
    assert "Gross:" in tooltip
    assert "Background:" in tooltip
    assert "Net:" in tooltip
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_fit_mode_ui.py::test_results_table_shows_a_region_row_for_an_integration_result -v`
Expected: FAIL — with the current code, `update_results_list()` calls `result.link_widths`/`result.peaks` unconditionally, raising `AttributeError` for an `IntegrationResult`.

- [ ] **Step 3: Add the `_integration_tooltip` helper and branch `update_results_list`**

Add a module-level function in `fit_mode.py`, near `_parameter_label` (after line 154):

```python
def _integration_tooltip(result):
    """Full gross/background/net breakdown for an IntegrationResult's Fit
    Results row tooltip -- mirrors the level of detail the existing
    left-tail-info tooltip gives for a Gaussian fit."""
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
            f"centroid={centroid:.2f}±{centroid_err:.2f}, "
            f"FWHM={fwhm:.2f}±{fwhm_err:.2f}, "
            f"skewness={skewness:.3g}±{skewness_err:.3g}"
        )
    return "\n".join(lines)
```

In `update_results_list` (`fit_mode.py:560-599`), insert an `isinstance` branch at the top of the `for fit_index, result in enumerate(active.fits):` loop body, before the existing `fit_label = ...` line:

```python
        for fit_index, result in enumerate(active.fits):
            if isinstance(result, IntegrationResult):
                fit_label = f"{fit_index + 1} [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]"
                tooltip = _integration_tooltip(result)
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self._results_row_fit_index.append(fit_index)
                values = [
                    fit_label,
                    "region",
                    f"{result.net_centroid:.2f} ± {result.net_centroid_err:.2f}",
                    f"{result.net_fwhm:.2f} ± {result.net_fwhm_err:.2f}",
                    f"{result.net_area:.1f} ± {result.net_area_err:.1f}",
                ]
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    if col == 0:
                        item.setToolTip(tooltip)
                    if not result.visible:
                        item.setForeground(QColor("gray"))
                    self.results_table.setItem(row, col, item)
                continue

            fit_label = f"{fit_index + 1} [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]"
            tooltip_lines = []
```

(The rest of the existing loop body — the `tooltip_lines`/`for peak_index, peak in ...` block — is unchanged, just now reached only for `FitResult` entries.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_fit_mode_ui.py::test_results_table_shows_a_region_row_for_an_integration_result -v`
Expected: PASS

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: show Integration results as a region row in Fit Results"
```

---

### Task 4: Plot rendering for an Integration result

**Files:**
- Modify: `fit_mode.py:340-400` (`draw_committed_fits`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_draw_committed_fits_draws_region_shading_and_annotation_for_integration(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_density=20.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=200.0, background_area_err=10.0,
            background_centroid=100.0, background_centroid_err=1.0,
            background_fwhm=9.0, background_fwhm_err=0.5,
            background_skewness=0.0, background_skewness_err=0.1,
            net_area=800.0, net_area_err=32.0,
            net_centroid=100.0, net_centroid_err=0.6,
            net_fwhm=7.5, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
        )
    )

    lines_before = len(main_window.axes.lines)
    patches_before = len(main_window.axes.patches)
    texts_before = len(main_window.axes.texts)
    main_window.fit_controller.draw_committed_fits(spectrum)

    # 3 region spans (2 bg + 1 fit) as patches, 1 flat background line,
    # 1 annotation -- no peak curve/decomposition/position-marker lines.
    assert len(main_window.axes.patches) == patches_before + 3
    assert len(main_window.axes.lines) == lines_before + 1
    assert len(main_window.axes.texts) == texts_before + 1


def test_draw_committed_fits_skips_a_hidden_integration_result(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)
    spectrum.fits.append(
        IntegrationResult(
            left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
            fit_region=(85.0, 115.0), background_density=20.0,
            gross_area=1000.0, gross_area_err=30.0,
            gross_centroid=100.0, gross_centroid_err=0.5,
            gross_fwhm=8.0, gross_fwhm_err=0.4,
            gross_skewness=0.0, gross_skewness_err=0.1,
            background_area=200.0, background_area_err=10.0,
            background_centroid=100.0, background_centroid_err=1.0,
            background_fwhm=9.0, background_fwhm_err=0.5,
            background_skewness=0.0, background_skewness_err=0.1,
            net_area=800.0, net_area_err=32.0,
            net_centroid=100.0, net_centroid_err=0.6,
            net_fwhm=7.5, net_fwhm_err=0.4,
            net_skewness=0.0, net_skewness_err=0.1,
            visible=False,
        )
    )

    lines_before = len(main_window.axes.lines)
    patches_before = len(main_window.axes.patches)
    texts_before = len(main_window.axes.texts)
    main_window.fit_controller.draw_committed_fits(spectrum)

    assert len(main_window.axes.lines) == lines_before
    assert len(main_window.axes.patches) == patches_before
    assert len(main_window.axes.texts) == texts_before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_fit_mode_ui.py -k "draw_committed_fits_draws_region or draw_committed_fits_skips_a_hidden_integration" -v`
Expected: FAIL — `draw_committed_fits` currently calls `result.background_slope`/`result.peaks` unconditionally, raising `AttributeError`.

- [ ] **Step 3: Branch `draw_committed_fits` for an `IntegrationResult`**

In `fit_mode.py`, `draw_committed_fits` (lines 340-400), insert an `isinstance` branch right after the three `axvspan` calls that are common to both result types (after line 351, before the existing `lo, hi = result.fit_region` line):

```python
            axes.axvspan(*result.left_bg_region, color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA)
            axes.axvspan(*result.right_bg_region, color=BG_REGION_COLOR, alpha=BG_REGION_ALPHA)
            axes.axvspan(*result.fit_region, color=FIT_REGION_COLOR, alpha=FIT_REGION_ALPHA)

            if isinstance(result, IntegrationResult):
                lo, hi = result.fit_region
                axes.plot(
                    [lo, hi], [result.background_density, result.background_density],
                    color="black", linestyle="--", linewidth=1,
                )
                axes.annotate(
                    f"centroid={result.net_centroid:.1f}\n"
                    f"FWHM={result.net_fwhm:.1f}\nnet={result.net_area:.0f}",
                    xy=(result.net_centroid, 0.95),
                    xycoords=label_transform,
                    ha="center", va="top",
                    fontsize=7, color="red",
                )
                continue

            lo, hi = result.fit_region
```

(The rest of the loop body — background line, curve, decomposition, per-peak annotations — is unchanged, now reached only for `FitResult` entries.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_fit_mode_ui.py -k "draw_committed_fits_draws_region or draw_committed_fits_skips_a_hidden_integration" -v`
Expected: PASS

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: draw region shading, flat background line, and annotation for Integration results"
```

---

### Task 5: Double-click reload for an Integration result

**Files:**
- Modify: `fit_mode.py:651-681` (`_on_result_double_clicked`)
- Test: `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fit_mode_ui.py`:

```python
def test_double_click_reloads_an_integration_result_marks_only(qapp):
    main_window = MainWindow()
    spectrum = _make_active_spectrum(main_window)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()

    main_window.fit_controller.clear()
    assert main_window.fit_controller.state.bg_regions == []

    item = main_window.fit_controller.results_table.item(0, 0)
    main_window.fit_controller._on_result_double_clicked(item)

    regions = main_window.fit_controller.state.bg_regions
    assert regions[0] == pytest.approx((70.0, 85.0))
    assert regions[1] == pytest.approx((115.0, 130.0))
    assert main_window.fit_controller.state.fit_region == pytest.approx((85.0, 115.0))
    assert main_window.fit_controller.state.peak_positions == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_fit_mode_ui.py::test_double_click_reloads_an_integration_result_marks_only -v`
Expected: FAIL — `_on_result_double_clicked` currently calls `result.peaks`/`result.link_widths`/`result.tail_fraction` unconditionally, raising `AttributeError`.

- [ ] **Step 3: Branch `_on_result_double_clicked` for an `IntegrationResult`**

In `fit_mode.py`, `_on_result_double_clicked` (lines 651-681), insert an `isinstance` branch after the two lines common to both result types, before the existing `self.state.peak_positions = ...` line:

```python
        self.state.bg_regions = [result.left_bg_region, result.right_bg_region]
        self.state.fit_region = result.fit_region

        if isinstance(result, IntegrationResult):
            self.state.peak_positions = []
            self._redraw_progress()
            self.main_window._update_fit_mode_availability()
            return

        self.state.peak_positions = [peak.position for peak in result.peaks]
```

(The rest of the method — restoring the Independent-widths/Left-tail checkboxes and the Fit Parameters panel — is unchanged, now reached only for `FitResult` entries.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_fit_mode_ui.py::test_double_click_reloads_an_integration_result_marks_only -v`
Expected: PASS

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py tests/test_fit_mode_ui.py
git commit -m "feat: double-click reload restores marks only for an Integration result"
```

---

### Task 6: Export and auto-log support for Integration results

**Files:**
- Modify: `fit_export.py`
- Modify: `fit_mode.py` (`run_integration`, to add the auto-log call)
- Test: `tests/test_fit_export.py`, `tests/test_fit_mode_ui.py`

- [ ] **Step 1: Write the failing tests**

Change `tests/test_fit_export.py`'s existing import line (line 7) from:

```python
from peak_fit import FitResult, PeakResult
```

to:

```python
from peak_fit import FitResult, IntegrationResult, PeakResult
```

Then add to the end of `tests/test_fit_export.py`:

```python
def _make_integration_result(timestamp="2026-07-16T12:00:00"):
    return IntegrationResult(
        left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0), background_density=20.0,
        gross_area=1000.0, gross_area_err=30.0,
        gross_centroid=100.0, gross_centroid_err=0.5,
        gross_fwhm=8.0, gross_fwhm_err=0.4,
        gross_skewness=0.0, gross_skewness_err=0.1,
        background_area=200.0, background_area_err=10.0,
        background_centroid=100.0, background_centroid_err=1.0,
        background_fwhm=9.0, background_fwhm_err=0.5,
        background_skewness=0.0, background_skewness_err=0.1,
        net_area=800.0, net_area_err=32.0,
        net_centroid=100.0, net_centroid_err=0.6,
        net_fwhm=7.5, net_fwhm_err=0.4,
        net_skewness=0.0, net_skewness_err=0.1,
        timestamp=timestamp,
    )


def test_integration_result_to_json_record_includes_gross_background_net():
    result = _make_integration_result()
    record = integration_result_to_json_record(result, "eu.spe")
    assert record["type"] == "integration"
    assert record["timestamp"] == "2026-07-16T12:00:00"
    assert record["spectrum_path"] == "eu.spe"
    assert record["fit_region"] == [85.0, 115.0]
    assert record["background_density"] == 20.0
    assert record["gross"]["area"] == 1000.0
    assert record["background"]["area"] == 200.0
    assert record["net"]["area"] == 800.0
    assert record["net"]["centroid"] == 100.0


def test_append_auto_log_handles_an_integration_result(tmp_path):
    spectrum_path = str(tmp_path / "eu.spe")
    append_auto_log(spectrum_path, _make_integration_result())

    log_path = auto_log_path(spectrum_path)
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "integration"


def test_integration_result_to_text_report_includes_all_three_layers():
    result = _make_integration_result()
    report = integration_result_to_text_report(result, "eu.spe", fit_number=1)
    assert "Integration" in report
    assert "Gross:" in report
    assert "Background:" in report
    assert "Net:" in report
    assert "800" in report


def test_write_text_report_handles_a_mix_of_fit_and_integration_results(tmp_path):
    from peak_fit import FitResult, PeakResult

    peak = PeakResult(
        position=100.0, position_err=0.1, fwhm=7.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0, amplitude=200.0, sigma=3.0,
    )
    fit = FitResult(
        left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
        peaks=[peak], timestamp="2026-07-16T12:00:00",
    )
    integration = _make_integration_result(timestamp="2026-07-16T12:05:00")
    out_path = tmp_path / "report.txt"

    write_text_report(str(out_path), [(1, fit), (2, integration)], "eu.spe")

    text = out_path.read_text(encoding="utf-8")
    assert text.index("Fit 1") < text.index("Fit 2 (Integration)")
```

Update `tests/test_fit_export.py`'s existing import line (line 3-6) to include the new functions:

```python
from fit_export import (
    append_auto_log, auto_log_path, fit_result_to_json_record,
    fit_result_to_text_report, integration_result_to_json_record,
    integration_result_to_text_report, write_text_report,
)
```

Add to `tests/test_fit_mode_ui.py`:

```python
def test_run_integration_appends_to_the_auto_log(qapp, tmp_path):
    main_window = MainWindow()
    spectrum_path = str(tmp_path / "eu.spe")
    _make_active_spectrum(main_window, path=spectrum_path)

    _held_key_click(main_window, "b", 70)
    _held_key_click(main_window, "b", 85)
    _held_key_click(main_window, "b", 115)
    _held_key_click(main_window, "b", 130)
    _held_key_click(main_window, "r", 85)
    _held_key_click(main_window, "r", 115)
    main_window.fit_controller.run_integration()

    log_path = tmp_path / "eu_fits.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["type"] == "integration"
```

Add `import json` near the top of `tests/test_fit_mode_ui.py` if not already present (check first — it is not currently imported there).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_fit_export.py tests/test_fit_mode_ui.py -k "integration" -v`
Expected: FAIL — `integration_result_to_json_record`/`integration_result_to_text_report` don't exist yet; `run_integration` doesn't call `append_auto_log` yet.

- [ ] **Step 3: Add `IntegrationResult` import and layer-record helper to `fit_export.py`**

At the top of `fit_export.py`, after the existing `import json` / `import os` lines:

```python
from peak_fit import IntegrationResult
```

Add after the existing `_peak_record` function:

```python
def _integration_layer_record(result, prefix):
    return {
        "area": getattr(result, f"{prefix}_area"),
        "area_err": getattr(result, f"{prefix}_area_err"),
        "centroid": getattr(result, f"{prefix}_centroid"),
        "centroid_err": getattr(result, f"{prefix}_centroid_err"),
        "fwhm": getattr(result, f"{prefix}_fwhm"),
        "fwhm_err": getattr(result, f"{prefix}_fwhm_err"),
        "skewness": getattr(result, f"{prefix}_skewness"),
        "skewness_err": getattr(result, f"{prefix}_skewness_err"),
    }
```

- [ ] **Step 4: Add `"type": "fit"` to the existing `fit_result_to_json_record`**

In `fit_export.py`'s existing `fit_result_to_json_record` function, add a `"type"` key as the first entry of the returned dict:

```python
    return {
        "type": "fit",
        "timestamp": result.timestamp,
```

- [ ] **Step 5: Add `integration_result_to_json_record` and dispatch in `append_auto_log`**

Add after `fit_result_to_json_record`:

```python
def integration_result_to_json_record(result, spectrum_path):
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
        "gross": _integration_layer_record(result, "gross"),
        "background": _integration_layer_record(result, "background"),
        "net": _integration_layer_record(result, "net"),
    }


def _to_json_record(result, spectrum_path):
    if isinstance(result, IntegrationResult):
        return integration_result_to_json_record(result, spectrum_path)
    return fit_result_to_json_record(result, spectrum_path)
```

Change `append_auto_log`'s body to use the dispatcher:

```python
def append_auto_log(spectrum_path, result):
    """Appends one JSON-Lines record for `result` to
    `<spectrum_stem>_fits.jsonl`, next to the spectrum file. Raises
    OSError on failure (e.g. read-only directory) -- the caller must
    turn that into a non-blocking status message, since a failed log
    write must never invalidate an already-successful fit."""
    record = _to_json_record(result, spectrum_path)
    path = auto_log_path(spectrum_path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

- [ ] **Step 6: Add `integration_result_to_text_report` and dispatch in `write_text_report`**

Add after `fit_result_to_text_report`:

```python
def integration_result_to_text_report(result, spectrum_path, fit_number=1):
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
        lines.append(f"  {label}:")
        lines.append(
            f"    Area:      {_format_err(getattr(result, f'{prefix}_area'), getattr(result, f'{prefix}_area_err'))}"
        )
        lines.append(
            f"    Centroid:  {_format_err(getattr(result, f'{prefix}_centroid'), getattr(result, f'{prefix}_centroid_err'))}"
        )
        lines.append(
            f"    FWHM:      {_format_err(getattr(result, f'{prefix}_fwhm'), getattr(result, f'{prefix}_fwhm_err'))}"
        )
        lines.append(
            f"    Skewness:  {_format_err(getattr(result, f'{prefix}_skewness'), getattr(result, f'{prefix}_skewness_err'))}"
        )
    return "\n".join(lines)


def _to_text_report(result, spectrum_path, fit_number):
    if isinstance(result, IntegrationResult):
        return integration_result_to_text_report(result, spectrum_path, fit_number)
    return fit_result_to_text_report(result, spectrum_path, fit_number)
```

Change `write_text_report`'s body to use the dispatcher:

```python
def write_text_report(path, results, spectrum_path):
    """Writes a plain-text report for one or more fits/integrations to
    `path`, overwriting any existing file. `results` is a list of
    (fit_number, result) pairs, in the order they should appear in the
    report."""
    blocks = [
        _to_text_report(result, spectrum_path, fit_number)
        for fit_number, result in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")
```

- [ ] **Step 7: Wire the auto-log call into `run_integration()`**

In `fit_mode.py`'s `run_integration()` (added in Task 2), add right after `active.fits.append(result)`:

```python
        active.fits.append(result)
        try:
            fit_export.append_auto_log(active.path, result)
        except OSError as exc:
            self._show_status_message(f"Could not write fit log: {exc}", 5000)
        self.main_window._plot_data(preserve_view=True)
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `python -m pytest tests/test_fit_export.py tests/test_fit_mode_ui.py -k "integration" -v`
Expected: PASS

- [ ] **Step 9: Run the full existing test suite to confirm no regression**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add fit_export.py fit_mode.py tests/test_fit_export.py tests/test_fit_mode_ui.py
git commit -m "feat: export and auto-log support for Integration results"
```

---

### Task 7: Full regression run and holistic review

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest -v`
Expected: every test from before this plan started, plus all new tests added in Tasks 1-6, PASS (pre-existing xfails unaffected).

- [ ] **Step 2: Review the whole branch diff holistically**

Diff the full range of commits created by Tasks 1-6 against the commit
this plan started from. Check specifically for:
- Every place `active.fits`/`spectrum.fits` is iterated in `fit_mode.py`
  (results table, plot rendering, double-click reload, context menu,
  export) correctly handles a mix of `FitResult` and `IntegrationResult`
  — the context menu (`_on_results_context_menu`/`_export_fits`) was
  *not* touched by this plan and should be re-checked: it iterates
  `active.fits` by index only (never accesses `FitResult`-specific
  fields directly), so it should already work unmodified for a mixed
  list — confirm this by inspection, not just by trusting the design.
- No stale references to a "moment quirk" or "TV parity" comment that
  contradicts what the code actually does.
- `IntegrationResult`/`integrate_region` imports are consistent (same
  spelling, same module) across `peak_fit.py`, `fit_mode.py`,
  `fit_export.py`, and all three test files.

- [ ] **Step 3: Fix anything found, with a new commit per fix**

If the review in Step 2 finds an issue, fix it, add a regression test
if one doesn't already cover it, re-run the full suite, and commit the
fix separately (do not amend earlier commits).

- [ ] **Step 4: Build and smoke-test the Windows installer**

Run: `powershell -File packaging\windows\build.ps1`
Expected: builds successfully to `packaging/windows/output/SpectraToolsSetup.exe`.
This only verifies the build and packaged executable run — actual
installation (UAC) and manual UI exercise of Integration mode still
needs the user to verify by hand.
