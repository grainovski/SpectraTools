# v3.1.0 Critical + Important Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 5 Critical and 18 Important findings from the v3.0.0 full-codebase audit (published as an artifact, referenced in project memory `project_v310_audit_of_v300.md`), in preparation for a v3.1.0 release. Minor findings (24) are a separate follow-on plan, explicitly out of scope here.

**Architecture:** No new subsystems. Every task is a targeted fix to existing code, following the audit's own root-cause analysis. Grouped into 6 areas matching the audit's own subsystem boundaries: fitting core, fit_mode.py deduplication, file I/O robustness, main window, matrix panel, documentation/packaging.

**Tech Stack:** Same as the rest of the app (Python, PySide6, matplotlib, numpy/scipy, pytest).

**Source of truth for every finding below:** the published audit artifact and `project_v310_audit_of_v300.md`. Each task cites the exact finding it fixes.

---

### Task 1 (Critical 1): Fix uncaught crash when integrating a region with negative counts

**Files:**
- Modify: `peak_fit.py`
- Test: `tests/test_peak_fit.py`

**Context:** `integrate_region()` calls bare `math.sqrt()` on quantities (`dMom`/variance terms) that are only guaranteed non-negative when the underlying spectrum data (`s`, and `ds = s.copy()` at line 506, which stands in for Poisson variance = the count itself) is non-negative. Subtract Spectra deliberately produces unclamped negative results by design, and running Integration on such a result crashes with an uncaught `ValueError: math domain error`. The one existing call site that already handles this correctly (line 627: `sigma = math.sqrt(M2) if M2 >= 0.0 else -math.sqrt(-M2)`) is for a *reported* value (sign-preserving); the fix here is for *uncertainty* terms, which are conventionally non-negative magnitudes, so wrapping the sqrt argument in `abs()` is the correct, minimal, behavior-preserving-for-normal-data fix (for non-negative data, `dMom >= 0` always, so `abs()` is a no-op there — verify this explicitly in your test).

- [ ] **Step 1: Write the failing test**

```python
def test_integrate_region_does_not_crash_on_negative_counts():
    # Simulates a Subtract-Spectra-derived spectrum: unclamped negative
    # counts in and around the fit region.
    x = np.arange(200, dtype=float)
    y = np.full(200, -5.0)
    y[90:110] += 40.0  # a "peak" sitting on a negative background
    result = integrate_region(x, y, (10, 30), (150, 170), (60, 140))
    assert math.isfinite(result.gross_area_err)
    assert math.isfinite(result.background_area_err)
    assert math.isfinite(result.net_area_err)


def test_integrate_region_negative_counts_errors_are_nonnegative_magnitudes():
    x = np.arange(200, dtype=float)
    y = np.full(200, -5.0)
    y[90:110] += 40.0
    result = integrate_region(x, y, (10, 30), (150, 170), (60, 140))
    assert result.gross_area_err >= 0.0
    assert result.background_area_err >= 0.0
    assert result.net_area_err >= 0.0


def test_integrate_region_positive_counts_unaffected_by_abs_guard():
    # Regression guard: for ordinary non-negative data, abs() around the
    # sqrt argument must be a true no-op -- same result as before the fix.
    x = np.linspace(0, 200, 201)
    y = np.zeros(201)
    y[90:110] = 40.0
    y += 5.0
    result_before_style = integrate_region(x, y, (10, 30), (150, 170), (60, 140))
    assert result_before_style.gross_area_err == pytest.approx(math.sqrt(sum(y[(x >= 60) & (x <= 140)])))
```

Add these to `tests/test_peak_fit.py` (check the file's existing imports already include `math`, `pytest`, `np`, `integrate_region` — add any missing ones).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_peak_fit.py -k negative_counts -v`
Expected: FAIL — `ValueError: math domain error` on the first two tests.

- [ ] **Step 3: Wrap every uncertainty-term `math.sqrt()` call in `integrate_region()` with `abs()`**

In `peak_fit.py`'s `integrate_region()`, change every `math.sqrt(X)` call that computes an *uncertainty* (not the sign-preserving `sigma`/`fwhm` reporting block at line 625-631, which must stay exactly as-is) to `math.sqrt(abs(X))`. This applies to these call sites (verify against current line numbers, they may have shifted slightly from a prior edit in this same task):
- `gross_area_err = math.sqrt(gross_dsum)` → `math.sqrt(abs(gross_dsum))`
- `g_DM1 = math.sqrt(dMom1) / abs(gross_sum)` → `math.sqrt(abs(dMom1)) / abs(gross_sum)`
- `g_DM2 = math.sqrt(dMom2) / abs(gross_sum)` → `math.sqrt(abs(dMom2)) / abs(gross_sum)`
- `g_DM3 = math.sqrt(dMom3) / abs(gross_sum)` → `math.sqrt(abs(dMom3)) / abs(gross_sum)`
- `background_area_err = math.sqrt(background_area_var)` → `math.sqrt(abs(background_area_var))`
- `net_area_err = math.sqrt(net_dsum)` → `math.sqrt(abs(net_dsum))`
- `bg_DM1 = math.sqrt(dBgMom1) / abs(bg_sum)` → `math.sqrt(abs(dBgMom1)) / abs(bg_sum)`
- `n_DM1 = math.sqrt(dMom1) / abs(net_sum)` → `math.sqrt(abs(dMom1)) / abs(net_sum)` (this is the *second* `dMom1` variable, in the background/net moments block, not the gross one above — same name, different scope, both need the fix)
- `bg_DM2 = math.sqrt(dBgMom2) / abs(bg_sum)` → `math.sqrt(abs(dBgMom2)) / abs(bg_sum)`
- `n_DM2 = math.sqrt(dMom2) / abs(net_sum)` → `math.sqrt(abs(dMom2)) / abs(net_sum)` (second `dMom2`, background/net block)
- `bg_DM3 = math.sqrt(dBgMom3) / abs(bg_sum)` → `math.sqrt(abs(dBgMom3)) / abs(bg_sum)`
- `n_DM3 = math.sqrt(dMom3) / abs(net_sum)` → `math.sqrt(abs(dMom3)) / abs(net_sum)` (second `dMom3`, background/net block)

Do NOT touch the sign-preserving block at (current) lines 625-631 (`_to_reported`'s `sigma = math.sqrt(M2) if M2 >= 0.0 else -math.sqrt(-M2)`) — that's already correct and handles a different case (a reported value, not an uncertainty).

Also check `fit_peaks()` (`peak_fit.py:435`) for the sibling `gross_area_err = float(np.sqrt(gross_area))` — this doesn't crash but silently produces `NaN`. Change to `float(np.sqrt(abs(gross_area)))` for consistency, and add one more test:

```python
def test_fit_peaks_gross_area_err_finite_for_negative_region():
    # fit_peaks' own gross_area_err sibling to the integrate_region fix above.
    x = np.linspace(0, 200, 201)
    y = np.full(201, -5.0)
    y[90:110] += 40.0
    result = fit_peaks(x, y, (60, 140), peak_positions=[100.0])
    assert math.isfinite(result.gross_area_err)
```
(Adjust the exact `fit_peaks` call signature to match what's actually in the file — check its real parameter names before writing this test.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS, full file — including every pre-existing test (the `abs()` wrapping must be a true no-op for all normal, non-negative-data cases already covered).

- [ ] **Step 5: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "fix: prevent Integration crash on negative-count regions (Subtract Spectra results)"
```

---

### Task 2 (Critical 2): Fix uncaught crash opening a non-UTF8-decodable file

**Files:**
- Modify: `histogram_io.py`
- Test: `tests/test_histogram_io.py`

**Context:** `load_histogram` opens with no explicit `encoding=`, so `open(path, "r")` uses the platform locale codec. A byte sequence that codec can't decode raises `UnicodeDecodeError` (a `ValueError` subclass, but NOT an `OSError`) during file iteration, outside the narrow `except ValueError` that only wraps the per-line `int()` parse — and `main_window.py`'s loader only catches `ParseError`/`OSError`, so this propagates all the way to an uncaught crash. This loader is the catch-all for any file extension other than `.spe`/`.spk`/`.n42` (including "All files (*)" picks), so this is reachable by an ordinary accidental file selection, not just a crafted input.

- [ ] **Step 1: Write the failing test**

```python
def test_load_histogram_raises_parse_error_not_crash_on_undecodable_bytes(tmp_path):
    path = tmp_path / "not_text.bin"
    # Bytes that are invalid in both UTF-8 and Windows cp1252.
    path.write_bytes(b"\xff\xfe\x00\x01\x8f\x90\x91\x92" * 20)
    with pytest.raises(ParseError):
        load_histogram(str(path))
```

Add to `tests/test_histogram_io.py` (add `import pytest` if not already present; the file already imports `ParseError`/`load_histogram` per its existing tests).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_histogram_io.py -k undecodable -v`
Expected: FAIL with an uncaught `UnicodeDecodeError`, not `ParseError`.

- [ ] **Step 3: Open with an explicit encoding and catch decode errors as ParseError**

In `histogram_io.py`, change:
```python
    values = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values.append(int(stripped))
            except ValueError:
                continue
```
to:
```python
    values = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    values.append(int(stripped))
                except ValueError:
                    continue
    except UnicodeDecodeError as exc:
        raise ParseError(f"File is not valid UTF-8 text: {path}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_histogram_io.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add histogram_io.py tests/test_histogram_io.py
git commit -m "fix: raise ParseError instead of crashing on non-UTF8 histogram files"
```

---

### Task 3 (Critical 3): Fix uncaught crash on an unbounded N42 CountedZeroes run-length

**Files:**
- Modify: `n42_io.py`
- Test: `tests/test_n42_io.py`

**Context:** `_decode_counted_zeroes` does `values.extend([0] * run_length)` with no upper bound, so a file with a huge run-length (e.g. `"5 0 100000000000000"`) triggers an uncaught `MemoryError`. Same bug class already fixed via `MAT_COLMAX` in `spk_io.py` and `_DIM_MAX` in `mtx_io.py`; never applied here. Also fix a related gap in the same function: negative run-lengths are silently accepted (`[0] * -3 == []` in Python), silently shrinking the channel array instead of erroring.

- [ ] **Step 1: Write the failing tests**

```python
def test_load_n42_rejects_oversized_counted_zeroes_run_length(tmp_path):
    xml = """<?xml version="1.0"?>
<RadInstrumentData xmlns="http://physics.nist.gov/N42/2011/N42">
  <Measurement>
    <Spectrum id="s1">
      <ChannelData compressionCode="CountedZeroes">5 0 100000000000000</ChannelData>
    </Spectrum>
  </Measurement>
</RadInstrumentData>"""
    path = tmp_path / "huge_run.n42"
    path.write_text(xml)
    with pytest.raises(ParseError):
        load_n42(str(path))


def test_load_n42_rejects_negative_counted_zeroes_run_length(tmp_path):
    xml = """<?xml version="1.0"?>
<RadInstrumentData xmlns="http://physics.nist.gov/N42/2011/N42">
  <Measurement>
    <Spectrum id="s1">
      <ChannelData compressionCode="CountedZeroes">5 0 -3 7</ChannelData>
    </Spectrum>
  </Measurement>
</RadInstrumentData>"""
    path = tmp_path / "negative_run.n42"
    path.write_text(xml)
    with pytest.raises(ParseError):
        load_n42(str(path))
```

Add to `tests/test_n42_io.py`, matching the exact XML-fixture style the file's other tests already use (check an existing `CountedZeroes` test in that file for the real namespace/element structure and match it precisely rather than trusting the sketch above verbatim).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_n42_io.py -k run_length -v`
Expected: FAIL — oversized case raises uncaught `MemoryError`, negative case returns successfully instead of raising.

- [ ] **Step 3: Bound and validate the run-length in `_decode_counted_zeroes`**

In `n42_io.py`, add a module-level constant near the top (matching `spk_io.py`'s `MAT_COLMAX`/`mtx_io.py`'s `_DIM_MAX` convention):
```python
_MAX_RUN_LENGTH = 1 << 20  # generous bound for a single CountedZeroes run; anything beyond this is corrupt, not a real spectrum
```
Then change:
```python
            run_length = tokens[i + 1]
            values.extend([0] * run_length)
            i += 2
```
to:
```python
            run_length = tokens[i + 1]
            if not (0 <= run_length <= _MAX_RUN_LENGTH):
                raise ParseError(
                    f"N42 CountedZeroes run-length {run_length} is out of range "
                    f"(must be 0-{_MAX_RUN_LENGTH}): {path}"
                )
            values.extend([0] * run_length)
            i += 2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_n42_io.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add n42_io.py tests/test_n42_io.py
git commit -m "fix: bound N42 CountedZeroes run-length to prevent MemoryError on corrupt files"
```

---

### Task 4 (Critical 4): Reject infinite/non-finite factors in Multiply, Add, and Subtract

**Files:**
- Modify: `main_window.py`, `combine_dialog.py`
- Test: `tests/test_operations_menu.py` (Multiply), a combine-dialog test file (check whether `tests/test_combine_dialog.py` exists; if not, check where `CombineDialog` is already tested — likely `tests/test_operations_menu.py` too)

**Context:** Factor validation everywhere is `v > 0` / `factor > 0`, which correctly rejects `NaN` but not `inf`. `multiply(data, inf)` (and the equivalent in Add/Subtract) silently corrupts every channel to the int64 sentinel `-9223372036854775808` with only an invisible `RuntimeWarning`, no error shown to the user. Fix by adding a finiteness check alongside the existing positivity check at both call sites.

- [ ] **Step 1: Read the current validators to confirm exact text before editing**

Read `main_window.py`'s `_open_multiply_dialog` (currently around line 553-561: `validate=lambda v: None if v > 0 else "Factor must be greater than zero."`) and `combine_dialog.py`'s `_on_accept` (currently around line 68-76: `if not (factor > 0): QMessageBox.warning(..., "Factor must be greater than zero.")`). Confirm these match before editing — if either has drifted, adapt the steps below to the real current text rather than assuming.

- [ ] **Step 2: Write the failing tests**

```python
# In tests/test_operations_menu.py, near the existing Multiply-by-Factor tests:
def test_multiply_dialog_validator_rejects_infinity():
    window = MainWindow()
    validators = []
    # Capture the real validate lambda main_window.py actually constructs,
    # by monkeypatching FactorDialog to record its constructor args instead
    # of showing a real dialog -- match whatever pattern the existing
    # Multiply-by-Factor tests in this file already use to test the
    # validator without a real Qt dialog loop (check nearby tests for the
    # established monkeypatch/capture technique before writing this from
    # scratch).
    ...
    assert validate(float("inf")) is not None
    assert validate(float("-inf")) is not None
    assert validate(1e20) is None  # still a legal finite factor, even if large -- only inf/nan are rejected
```

(This test's exact shape depends on how `test_operations_menu.py` already tests `_open_multiply_dialog`'s validator — read that file's existing Multiply-by-Factor validator test first and mirror its exact technique instead of guessing; do not introduce a new testing pattern for this one case.)

```python
# Wherever CombineDialog's factor validation is tested:
def test_combine_dialog_rejects_infinite_factor(qapp):
    dialog = CombineDialog(None, "Add Spectra", spectra=[...], active_spectrum=None)
    dialog._factor_field.setText("inf")
    dialog._on_accept()
    assert dialog.result_factor is None  # rejected, dialog stays open
```

(Match the exact `CombineDialog` constructor signature and existing NaN-rejection test's style — find that test first, likely named something like `test_combine_dialog_rejects_nan_factor`, and write this as its direct sibling.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_operations_menu.py -k infinit -v` (and wherever the CombineDialog test landed)
Expected: FAIL — `inf` currently passes validation.

- [ ] **Step 4: Add the finiteness check**

In `main_window.py`'s `_open_multiply_dialog`, change:
```python
            validate=lambda v: None if v > 0 else "Factor must be greater than zero.",
```
to:
```python
            validate=lambda v: None if (math.isfinite(v) and v > 0) else "Factor must be a finite number greater than zero.",
```
(Add `import math` at the top of `main_window.py` if not already present.)

In `combine_dialog.py`'s `_on_accept`, change:
```python
        if not (factor > 0):
            QMessageBox.warning(self, self.windowTitle(), "Factor must be greater than zero.")
            return
```
to:
```python
        if not (math.isfinite(factor) and factor > 0):
            QMessageBox.warning(self, self.windowTitle(), "Factor must be a finite number greater than zero.")
            return
```
(Add `import math` at the top of `combine_dialog.py`.)

Also check `factor_dialog.py`'s Rebin-by-Factor validator (a second call site using the same `FactorDialog` class) — search `main_window.py` for `_open_rebin_dialog` or similar and confirm whether it has its own separate `validate=` lambda with the same `v > 0` gap; if so, apply the identical fix there too.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_operations_menu.py -v` (and the CombineDialog test file)
Expected: PASS, full file(s).

- [ ] **Step 6: Commit**

```bash
git add main_window.py combine_dialog.py tests/test_operations_menu.py
git commit -m "fix: reject infinite factors in Multiply/Rebin/Add/Subtract, not just NaN"
```

---

### Task 5 (Critical 5): Matrix panel's Fit Results table never populates

**Files:**
- Modify: `matrix_panel.py`
- Test: `tests/test_matrix_panel.py`

**Context:** `MatrixPanel._plot_data()` never calls `fit_controller.update_results_list()`. `MainWindow._plot_data()` and `MainWindow._on_active_toggled()` both call it; `MatrixPanel._plot_data()` (which has no analogous "active toggled" concept, since a matrix panel only ever has one spectrum) never does. The underlying fit data is correct — only the table-refresh wiring is missing.

- [ ] **Step 1: Write the failing test**

```python
def test_matrix_panel_fit_results_table_populates_after_fit(qapp, monkeypatch):
    import fit_mode
    monkeypatch.setattr(fit_mode.fit_export, "append_auto_log", lambda *a, **k: None)
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))

    lo, hi = panel.axes.get_xlim()
    panel.fit_controller._held_key = "r"
    _click(panel, lo + (hi - lo) * 0.2)
    _click(panel, lo + (hi - lo) * 0.8)
    panel.fit_controller._held_key = "p"
    _click(panel, lo + (hi - lo) * 0.5)
    panel.fit_controller._held_key = None

    assert panel.fit_controller.state.ready_to_fit()
    panel.fit_controller.run_fit()

    assert panel.fit_controller.results_table.rowCount() == 1
```

Add near `test_matrix_panel_integrate_action_reachable` in `tests/test_matrix_panel.py`, reusing the file's existing `_click` helper and `FIXTURES`/`qapp` fixtures exactly as its neighboring tests do. Check whether the file already has a similar real-fit test (e.g. `test_matrix_panel_fitting_a_peak_on_the_projection_works`) and match its exact marking sequence/style rather than inventing a new one.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matrix_panel.py -k fit_results_table -v`
Expected: FAIL — `rowCount() == 0`.

- [ ] **Step 3: Call update_results_list() from MatrixPanel._plot_data()**

In `matrix_panel.py`'s `_plot_data`, add the call right after `self.fit_controller.draw_committed_fits(spectrum)`:
```python
        self.fit_controller.draw_committed_fits(spectrum)
        self.fit_controller.update_results_list()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "fix: populate matrix panel's Fit Results table (was never wired to update_results_list)"
```

---

### Task 6 (Important 1): Fix background_skewness_err's wrong moment

**Files:**
- Modify: `peak_fit.py`
- Test: `tests/test_peak_fit.py`

**Context:** Line 617's `termb = dltb * (dltb2 - 3.0 * n_M2) - bg_M3` uses the net distribution's `n_M2` where it should use the background's own `bg_M2` — traced directly against `tv-1.9.13/lib/tv/vsFitInt.c:303`, which uses `bgMom2`, not the net's. The design spec's own citation (`vsFitInt.c:281,303`) bundled two different lines under one claim; only line 281 (the already-correct `dDltb_m2` term feeding `bg_DM2`/`background_fwhm_err`) actually has that quirk.

- [ ] **Step 1: Write the failing test**

```python
def test_background_skewness_err_uses_backgrounds_own_m2_not_nets():
    # Constructed so bg_M2 and n_M2 are numerically different -- an
    # asymmetric background window vs. a peak-dominated net region --
    # so the bug (using n_M2 in termb) produces a different, wrong
    # background_skewness_err than the fix (using bg_M2).
    x = np.arange(300, dtype=float)
    y = np.full(300, 3.0)
    y[140:160] += 100.0  # a sharp peak, skews the NET moments a lot
    # Background regions positioned asymmetrically so bg's own M2 differs
    # noticeably from the net region's M2.
    result = integrate_region(x, y, (10, 15), (280, 295), (100, 200))
    # Hand-computed expected value using bg_M2 (not n_M2) in termb -- see
    # Step 3 for the exact formula being asserted here.
    expected = _hand_compute_background_skewness_err(x, y, (10, 15), (280, 295), (100, 200))
    assert result.background_skewness_err == pytest.approx(expected, rel=1e-9)
```

Where `_hand_compute_background_skewness_err` is a small local helper you write in the test file, implementing the same moment machinery as `integrate_region()` but using `bg_M2` (not `n_M2`) in the `termb` line — a literal, independent re-derivation, not just calling `integrate_region()` again. This proves the fix against an independent calculation, not just "did the number change." Read `peak_fit.py`'s full `integrate_region()` first so the helper's background/net moment computation genuinely matches every other part of the real function (mask construction, `bg_density`, `bg_M1`, `dltb`, `dltb2`) — only the one `termb` line should differ from the real implementation, by design (that's the bug being tested for).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_peak_fit.py -k background_skewness_err_uses -v`
Expected: FAIL — current code uses `n_M2`, giving a different (wrong) result than the hand-computed `bg_M2` version.

- [ ] **Step 3: Fix the formula**

In `peak_fit.py`, change:
```python
        termb = dltb * (dltb2 - 3.0 * n_M2) - bg_M3
```
to:
```python
        termb = dltb * (dltb2 - 3.0 * bg_M2) - bg_M3
```
Update the comment above it (currently `# CONFIRMED TV QUIRK: net's M2 again, but bg's OWN M3 here.`) to reflect the corrected understanding:
```python
        # vsFitInt.c:303 uses the background's OWN M2 here (unlike the
        # DM2 term above at vsFitInt.c:281, which genuinely does cross-use
        # net's M2 -- the two lines are NOT the same quirk, despite an
        # earlier design-spec draft citing them together).
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS, full file. Also check `tests/test_fit_export.py` and any export/report golden-value tests that might assert a specific `background_skewness_err` number from before this fix — if any exist and now fail, update their expected values (with a comment noting why), don't work around the fix.

- [ ] **Step 5: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "fix: background_skewness_err uses the background's own M2, not net's (vsFitInt.c:303)"
```

Also add a one-line addendum to `docs/superpowers/specs/2026-07-16-integration-mode-design.md` right after its `vsFitInt.c:281,303` citation, noting that only line 281 has the cross-moment quirk and line 303 does not — so a future reader of that spec isn't misled the same way this session's audit found the implementation was.

---

### Task 7 (Important 2): Stop computing a full Jacobian for every rejected Marquardt trial step

**Files:**
- Modify: `peak_fit.py`
- Test: `tests/test_peak_fit.py`

**Context:** `measure_and_jac(pt)` always computes a full numeric Jacobian, even for a trial about to be rejected (only the scalar measure is needed for the accept/reject decision). Measured 12.4x evaluation blowup for an 8-peak multiplet vs. a 5-peak one. Fix: read the current Marquardt loop in full first (it's a tight, sensitive numerical routine — don't restructure more than necessary), then split into a cheap residual-only evaluation for the trial/reject path, computing the full Jacobian only once a step is actually accepted.

- [ ] **Step 1: Read the current Marquardt loop in full**

Read `peak_fit.py`'s `measure_and_jac` function and the `while not accepted:` retry loop that calls it (search for both) in full, along with `_numeric_jacobian`, before writing anything — this is the core scientific optimizer and needs to be understood precisely, not guessed at from the audit summary.

- [ ] **Step 2: Write the failing test (a timing/count-based regression guard)**

```python
def test_marquardt_does_not_recompute_jacobian_for_rejected_trials():
    # An overlapping multiplet chosen to force several lambda-growth
    # rounds (i.e. several rejected trials) during a real fit -- the
    # exact scenario the audit measured a 12.4x evaluation blowup on.
    call_count = {"model": 0}
    # Wrap peak_fit's own model-building so every underlying model()
    # evaluation is counted -- read _make_model first to find the right
    # place to instrument (likely by monkeypatching the returned model
    # function, or by counting calls to whatever the real Jacobian
    # helper is) rather than assuming a specific hook exists already.
    ...
    x = np.linspace(0, 100, 400)
    y = _synthetic_multiplet(x, n_peaks=8, spacing_sigma=2.9)  # write this helper to match the audit's own repro shape
    fit_peaks(x, y, fit_region=(0, 100), peak_positions=[...], independent_widths=True)
    # A generous but real ceiling: proves the optimization actually
    # reduced evaluations, not just that the fit still works. Pick the
    # exact threshold only after seeing the REAL pre-fix count on this
    # exact synthetic case (run it once manually, note the number, set
    # the ceiling comfortably below it) -- don't guess a number that
    # might already pass before the fix.
    assert call_count["model"] < <measured_prefix_count> * 0.6
```

Since the precise instrumentation hook depends on `_make_model`'s real internal structure (read in Step 1), design this test around whatever the actual, cleanest counting point is — the important property to preserve is: it fails before the fix (proving the current code really does over-evaluate) and passes comfortably after.

- [ ] **Step 3: Run test to verify it fails (or establish the baseline)**

Run: `pytest tests/test_peak_fit.py -k marquardt_does_not_recompute -v`
Record the actual pre-fix evaluation count for the ceiling in Step 2 if not already set.

- [ ] **Step 4: Split trial evaluation into a cheap residual-only path**

In `peak_fit.py`, add a `measure_only(pt)` helper alongside the existing `measure_and_jac(pt)` — same model-evaluation and residual/measure computation, but skipping `_numeric_jacobian` entirely. In the retry loop, call `measure_only(pt)` for the accept/reject test; only call the full `measure_and_jac(pt)` (or compute the Jacobian directly) once a step is confirmed accepted. Preserve the exact existing lambda-growth/shrink logic and convergence criteria — this is a pure performance refactor, the optimizer's numerical behavior (iteration count, final converged values) must be unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_peak_fit.py -v`
Expected: PASS, full file — including every existing convergence/accuracy test (final fitted parameter values must be numerically identical to before this change, since this only changes WHEN the Jacobian is computed, not the algorithm's math).

- [ ] **Step 6: Commit**

```bash
git add peak_fit.py tests/test_peak_fit.py
git commit -m "perf: skip Jacobian computation for rejected Marquardt trial steps"
```

---

### Task 8 (Important 3): Deduplicate fixed_params_from_panel / initial_guess_overrides_from_panel

**Files:**
- Modify: `fit_mode.py`
- Test: `tests/test_fit_mode_ui.py`

**Context:** These two methods (`fit_mode.py:883-908` and `:910-933`) duplicate the entire row-iteration/checkbox-branch/parse/error logic, differing only in the checkbox condition (`isChecked()` vs `not isChecked()`) and one word of the error message. This project already had a real bug in exactly this code path once (invalid "Fixed value" text crashing instead of showing a status message) — duplication is a real drift risk here, not just style.

- [ ] **Step 1: Confirm current behavior is covered by existing tests**

Run: `pytest tests/test_fit_mode_ui.py -k "fixed_params or initial_guess" -v` and confirm the existing tests for both methods pass before refactoring — these are your regression safety net; no new test is strictly required for a pure refactor, but add one if you find either method's behavior under-tested (e.g. the exact error message wording for each).

- [ ] **Step 2: Extract a shared helper**

In `fit_mode.py`, add a private helper:
```python
    def _read_panel_values(self, want_checked, label_verb):
        """Shared by fixed_params_from_panel/initial_guess_overrides_from_panel
        -- both read every row's current Value cell, filtered by whether
        its Fix checkbox matches `want_checked`, converting through
        _panel_value_to_internal. `label_verb` is "Fixed value"/"Value"
        for the two methods' distinct error-message wording."""
        result = {}
        calibrated = self.main_window._calibration_active and self.main_window._calibration is not None
        for row, name in enumerate(self._parameter_names_shown):
            fix_checkbox = self.parameters_table.cellWidget(row, 2)
            if fix_checkbox is not None and fix_checkbox.isChecked() == want_checked:
                text = self.parameters_table.item(row, 1).text()
                try:
                    value = float(text)
                except ValueError:
                    raise FitError(
                        f"{label_verb} for '{_parameter_label(name, calibrated)}' is not a valid number: {text!r}"
                    )
                result[name] = _panel_value_to_internal(
                    self.main_window, name, value, self._parameters_panel_values_shown
                )
        return result

    def fixed_params_from_panel(self):
        """Reads the current Fix checkboxes/values from the Fit
        Parameters panel into a {name: value} dict for the next
        fit_peaks() call. Empty when nothing is fixed."""
        return self._read_panel_values(want_checked=True, label_verb="Fixed value")

    def initial_guess_overrides_from_panel(self):
        """Mirrors fixed_params_from_panel for unchecked rows: reads
        each unfixed row's current Value cell as the starting guess for
        that parameter in the next fit (the parameter stays free)."""
        return self._read_panel_values(want_checked=False, label_verb="Value")
```
Remove the two old full method bodies, replacing them with the thin wrappers shown above (keep both public method names — callers use them by name).

- [ ] **Step 3: Run tests to verify nothing regressed**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS, full file — identical behavior, including the exact error message wording for both the "Fixed value for..." and "Value for..." cases (check the existing tests assert this precisely; if they do, this is your proof the refactor preserved behavior exactly).

- [ ] **Step 4: Commit**

```bash
git add fit_mode.py
git commit -m "refactor: extract shared _read_panel_values from fixed_params_from_panel/initial_guess_overrides_from_panel"
```

---

### Task 9 (Important 4): Deduplicate the fit-commit block in run_fit / run_integration

**Files:**
- Modify: `fit_mode.py`
- Test: `tests/test_fit_mode_ui.py`

**Context:** The exact same "supersede same-region fits, append, auto-log" block appears verbatim in both `run_fit()` (`fit_mode.py:1183-1195`) and `run_integration()` (`:1216-1228`).

- [ ] **Step 1: Confirm current behavior is covered**

Run: `pytest tests/test_fit_mode_ui.py -k "run_fit or run_integration or supersede or auto_log" -v` — confirm existing coverage before refactoring.

- [ ] **Step 2: Extract a shared _commit_result helper**

In `fit_mode.py`, add:
```python
    def _commit_result(self, active, result):
        """Shared by run_fit/run_integration: supersede any earlier
        committed fit/integration on the exact same region (hidden, not
        deleted -- see the 2026-07-15 spec's reasoning for why exact
        tuple equality is reliable here), append the new result, and
        auto-log it."""
        for earlier in active.fits:
            if (
                earlier.left_bg_region == result.left_bg_region
                and earlier.right_bg_region == result.right_bg_region
                and earlier.fit_region == result.fit_region
            ):
                earlier.visible = False
        active.fits.append(result)
        calibration = self.main_window._calibration if self.main_window._calibration_active else None
        try:
            fit_export.append_auto_log(active.path, result, calibration)
        except OSError as exc:
            self._show_status_message(f"Could not write fit log: {exc}", 5000)
```
Replace the duplicated block in both `run_fit()` and `run_integration()` with a single call: `self._commit_result(active, result)`.

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS, full file.

- [ ] **Step 4: Commit**

```bash
git add fit_mode.py
git commit -m "refactor: extract shared _commit_result from run_fit/run_integration"
```

---

### Task 10 (Important 5): Deduplicate peak-shape computation in draw_committed_fits

**Files:**
- Modify: `fit_mode.py`
- Test: `tests/test_fit_mode_ui.py`

**Context:** `draw_committed_fits()`'s total-curve accumulation (`fit_mode.py:682-694`) and its peak-decomposition overlay (`:696-713`) each reimplement the identical Gaussian/hypermet-tail branch. `background_dense` at line 699 also redundantly recomputes the exact same array as `total`'s own seed value at line 683. Risk: if the peak-shape formula ever changes, the total curve and its own decomposition overlay could silently draw differently.

- [ ] **Step 1: Confirm current visual/data behavior is covered**

Run: `pytest tests/test_fit_mode_ui.py -k draw_committed -v` — check what's already tested (likely artist counts/positions, not pixel values, since this is a matplotlib draw path) so you know what must stay identical.

- [ ] **Step 2: Extract a shared _peak_component helper**

In `fit_mode.py`, add a helper near `draw_committed_fits`:
```python
    @staticmethod
    def _peak_component(x_dense, peak, result):
        """One peak's own shape (Gaussian or hypermet-tail, matching
        result.tail_fraction), evaluated over x_dense -- shared by the
        total-curve accumulation and the per-peak decomposition overlay
        in draw_committed_fits, so the two can never silently diverge."""
        if result.tail_fraction is not None:
            return peak.amplitude * hypermet_left_tail(
                x_dense, peak.position, peak.sigma, result.tail_fraction, result.tail_beta,
            )
        return peak.amplitude * np.exp(-((x_dense - peak.position) ** 2) / (2 * peak.sigma ** 2))
```
Then in `draw_committed_fits`, replace the total-curve loop:
```python
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
```
with:
```python
            background_dense = result.background_slope * x_dense + result.background_intercept
            total = background_dense.copy()
            for peak in result.peaks:
                total = total + self._peak_component(x_dense, peak, result)
```
(note `background_dense` is now computed once here and reused below, removing the redundant second computation), and replace the decomposition loop:
```python
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
```
with (dropping its own now-redundant `background_dense` recomputation, reusing the one computed above):
```python
            for peak in result.peaks:
                component = self._peak_component(x_dense, peak, result)
                axes.plot(
```
(keep the rest of that `axes.plot(...)` call exactly as it currently is).

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_fit_mode_ui.py -v`
Expected: PASS, full file — the total curve and decomposition overlay must draw numerically identical values to before (verify by checking whatever the existing tests assert about `axes.lines`' plotted y-data, not just that "a line exists").

- [ ] **Step 4: Commit**

```bash
git add fit_mode.py
git commit -m "refactor: share peak-shape computation between total curve and decomposition overlay"
```

---

### Task 11 (Important 6): Fix uncaught OverflowError in histogram_io.py for huge integers

**Files:**
- Modify: `histogram_io.py`
- Test: `tests/test_histogram_io.py`

**Context:** `data[: len(values)] = values` has no exception handling; `int(stripped)` has no magnitude limit, so a well-formed but absurdly large integer line raises an uncaught `OverflowError` on assignment into the preallocated int64 array. Same class of guard already present in `spk_io.py`/`n42_io.py`, missing here.

- [ ] **Step 1: Write the failing test**

```python
def test_load_histogram_raises_parse_error_not_crash_on_huge_integer(tmp_path):
    path = tmp_path / "huge_value.txt"
    path.write_text(f"5\n10\n{10**23}\n7\n")
    with pytest.raises(ParseError):
        load_histogram(str(path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_histogram_io.py -k huge_integer -v`
Expected: FAIL with an uncaught `OverflowError`.

- [ ] **Step 3: Guard the array assignment**

In `histogram_io.py` (this is inside the same function you edited in Task 2 — read its current state first, since Task 2 already changed the surrounding `try` structure):
```python
    channel_count = _bucket_channel_count(len(values))
    data = np.zeros(channel_count, dtype=np.int64)
    try:
        data[: len(values)] = values
    except OverflowError as exc:
        raise ParseError(f"File contains an out-of-range integer value: {path}") from exc
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_histogram_io.py -v`
Expected: PASS, full file.

- [ ] **Step 5: Commit**

```bash
git add histogram_io.py tests/test_histogram_io.py
git commit -m "fix: raise ParseError instead of crashing on out-of-range histogram values"
```

---

### Task 12 (Important 7): Fix silent float-to-int64 data corruption in spe_io.py and spk_io.py

**Files:**
- Modify: `spe_io.py`, `spk_io.py`
- Test: `tests/test_spe_io.py`, `tests/test_spk_io.py`

**Context:** Both do `np.round(channels).astype(np.int64)` on a float32 array with no range check. `.astype()` doesn't raise on overflow at all — an extreme value (e.g. `1e30`) silently becomes the int64 sentinel `-9223372036854775808`, with only an invisible `RuntimeWarning`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_spe_io.py
def test_load_spe_raises_parse_error_not_silent_corruption_on_extreme_value(tmp_path):
    # Build a minimal valid .spe file whose one channel value is 1e30 --
    # read load_spe's own save_spe as the easiest way to construct a
    # valid file, then verify the round-trip raises instead of silently
    # producing the int64 sentinel.
    path = tmp_path / "extreme.spe"
    save_spe(str(path), [1e30])
    with pytest.raises(ParseError):
        load_spe(str(path))
```

```python
# tests/test_spk_io.py -- for the oldmat float (lf4/hf4) path specifically
def test_load_spk_raises_parse_error_not_silent_corruption_on_extreme_oldmat_float(tmp_path):
    # Construct a minimal oldmat lf4 file with one extreme value. Check
    # the file's existing oldmat-fixture-construction tests (search for
    # "oldmat_lf4" or similar) for the exact byte layout to reuse rather
    # than re-deriving the oldmat header format from scratch.
    ...
    with pytest.raises(ParseError):
        load_spk(str(path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_spe_io.py tests/test_spk_io.py -k extreme -v`
Expected: FAIL — both currently return the sentinel value instead of raising.

- [ ] **Step 3: Guard both casts**

In `spe_io.py`, change:
```python
    dtype = np.dtype(endian + "f4")
    channels = np.frombuffer(data, dtype=dtype, count=idim1, offset=data_start)
    return np.round(channels).astype(np.int64)
```
to:
```python
    dtype = np.dtype(endian + "f4")
    channels = np.frombuffer(data, dtype=dtype, count=idim1, offset=data_start)
    rounded = np.round(channels)
    if np.any(np.abs(rounded) > np.iinfo(np.int64).max):
        raise ParseError(f"File contains an out-of-range channel value: {path}")
    return rounded.astype(np.int64)
```

In `spk_io.py`, change:
```python
    channels = np.frombuffer(data, dtype=dtype, count=columns, offset=0)
    if np.issubdtype(dtype, np.floating):
        return np.round(channels).astype(np.int64)
    return channels.astype(np.int64)
```
to:
```python
    channels = np.frombuffer(data, dtype=dtype, count=columns, offset=0)
    if np.issubdtype(dtype, np.floating):
        rounded = np.round(channels)
        if np.any(np.abs(rounded) > np.iinfo(np.int64).max):
            raise ParseError(f"File contains an out-of-range channel value: {path}")
        return rounded.astype(np.int64)
    return channels.astype(np.int64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_spe_io.py tests/test_spk_io.py -v`
Expected: PASS, both files in full.

- [ ] **Step 5: Commit**

```bash
git add spe_io.py spk_io.py tests/test_spe_io.py tests/test_spk_io.py
git commit -m "fix: raise ParseError instead of silently corrupting out-of-range float channel values"
```

---

### Task 13 (Important 9): Deduplicate the LC2 codec between spk_io.py and mtx_io.py

**Files:**
- Create: `lc_codec.py`
- Modify: `spk_io.py`, `mtx_io.py`
- Test: `tests/test_lc_codec.py` (new), keep `tests/test_spk_io.py` and `tests/test_mtx_io.py` passing unchanged

**Context:** `spk_io.py:195-253` (`_lc2_uncompress`) and `mtx_io.py:17-83` (`_decode_row`) independently implement the same bit-level tag decoding. `_zigzag_decode` is also defined twice, and the dimension bound is duplicated as two same-valued constants (`spk_io.MAT_COLMAX` / `mtx_io._DIM_MAX`) whose comments already point at each other. Do this task BEFORE Task 14 (mtx_io performance), so the performance fix only needs to be written once, in the new shared location.

- [ ] **Step 1: Read both current implementations in full**

Read `spk_io.py`'s `_lc2_uncompress` and `mtx_io.py`'s `_decode_row`/`_zigzag_decode` in full (`mtx_io.py`'s version was already read in full during this plan's own preparation — reproduced below for reference; re-read `spk_io.py`'s version fresh since it wasn't). Confirm they really are behaviorally identical (same tag dispatch, same "deltas against pre-tag `last`" semantics, same same-run behavior) before merging — if you find a genuine behavioral difference, stop and report it rather than silently picking one.

`mtx_io.py`'s current `_decode_row`/`_zigzag_decode` (to become the canonical shared implementation, since it's already been independently verified against `libmfile-1.0.7` byte-for-byte during the original matrix-analysis feature work):
```python
def _zigzag_decode(n):
    return -((n >> 1) + 1) if (n & 1) else (n >> 1)


def _decode_row(data, num_values, path):
    """[docstring as currently in mtx_io.py]"""
    values = []
    last = 0
    pos = 0
    nleft = num_values
    try:
        while nleft > 0:
            t = data[pos]
            pos += 1
            if t & 0x80:
                n = t & 0x3F
                if n > 59:
                    bytes_extra = n - 59
                    n = 59
                    for i in range(bytes_extra):
                        b = data[pos]
                        pos += 1
                        n += (b + 1) << (i * 8)
                if t & 0x40:
                    diff = n & 1
                    same = (n >> 1) + 3
                    values.append(last + diff)
                    nleft -= same
                    if nleft <= 0:
                        raise ParseError(f"lc matrix file: same-run tag overruns row: {path}")
                    values.extend([last] * same)
                else:
                    last = last + _zigzag_decode(n)
                    values.append(last)
                nleft -= 1
            elif t & 0x40:
                nleft -= 2
                if nleft < 0:
                    raise ParseError(f"lc matrix file: 2-value pack overruns row: {path}")
                a = t & 0x7
                b = (t >> 3) & 0x7
                values.append(last + _zigzag_decode(a))
                last = last + _zigzag_decode(b)
                values.append(last)
            else:
                nleft -= 3
                if nleft < 0:
                    raise ParseError(f"lc matrix file: 3-value pack overruns row: {path}")
                a = t & 0x3
                b = (t >> 2) & 0x3
                c = (t >> 4) & 0x3
                values.append(last + _zigzag_decode(a))
                values.append(last + _zigzag_decode(b))
                last = last + _zigzag_decode(c)
                values.append(last)
    except IndexError as exc:
        raise ParseError(f"lc matrix file: row data ends mid-tag: {path}") from exc
    return values
```

- [ ] **Step 2: Write the failing test for the new module**

```python
# tests/test_lc_codec.py
import pytest
from lc_codec import zigzag_decode, decode_row, DIM_MAX
from histogram_io import ParseError


def test_zigzag_decode_matches_known_values():
    assert zigzag_decode(0) == 0
    assert zigzag_decode(1) == -1
    assert zigzag_decode(2) == 1
    assert zigzag_decode(3) == -2


def test_decode_row_raises_parse_error_on_truncated_data():
    with pytest.raises(ParseError):
        decode_row(b"", 5, "fake_path")


def test_dim_max_is_65536():
    assert DIM_MAX == 1 << 16
```

- [ ] **Step 3: Run test to verify it fails (module doesn't exist yet)**

Run: `pytest tests/test_lc_codec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lc_codec'`.

- [ ] **Step 4: Create lc_codec.py, then update spk_io.py and mtx_io.py to use it**

Create `lc_codec.py` with `zigzag_decode`, `decode_row` (renamed from `_decode_row`, now public — the exact body from Step 1, minus the leading underscore, taking `ParseError` as an import from `histogram_io` same as both callers already do), and `DIM_MAX = 1 << 16`.

In `mtx_io.py`: remove the local `_zigzag_decode`/`_decode_row`/`_DIM_MAX` definitions, add `from lc_codec import decode_row, DIM_MAX`, update all call sites (`_decode_row(...)` → `decode_row(...)`, `_DIM_MAX` → `DIM_MAX`).

In `spk_io.py`: replace the body of `_lc2_uncompress` to delegate to `lc_codec.decode_row` (check its exact current call signature/return-value handling first — it may wrap the result differently than `mtx_io.py` does, e.g. converting to a numpy array immediately vs. returning a plain list; preserve `spk_io.py`'s own existing external behavior exactly, only replace the internal decode logic). Keep `MAT_COLMAX` as a name in `spk_io.py` (other code/tests likely reference it under that name — grep for `MAT_COLMAX` usage across the repo first) but define it as `MAT_COLMAX = lc_codec.DIM_MAX` rather than a separate literal `1 << 16`.

- [ ] **Step 5: Run tests to verify everything passes**

Run: `pytest tests/test_lc_codec.py tests/test_spk_io.py tests/test_mtx_io.py -v`
Expected: PASS, all three files in full — `spk_io.py`'s and `mtx_io.py`'s own test suites must pass completely unchanged (same public behavior, only the internal implementation moved).

- [ ] **Step 6: Commit**

```bash
git add lc_codec.py spk_io.py mtx_io.py tests/test_lc_codec.py
git commit -m "refactor: extract shared LC2 codec (lc_codec.py) from spk_io.py and mtx_io.py"
```

---

### Task 14 (Important 8): Optimize mtx_io.py's row decoder

**Files:**
- Modify: `lc_codec.py` (after Task 13, this is where the decoder now lives)
- Test: `tests/test_mtx_io.py`, `tests/test_lc_codec.py`

**Context:** Real 8192x8192 matrix fixtures take ~5s to load — a pure-Python per-value loop over ~67 million cells. Already-known/deferred per the original matrix-analysis plan; this is the revisit. Do this AFTER Task 13, so the optimization only needs to be written once.

- [ ] **Step 1: Establish the current baseline precisely**

Run a timing check against the real fixtures before changing anything:
```python
import time
from mtx_io import load_mtx
for name in ("tests/fixtures/gg.mtx", "tests/fixtures/gpff.mtx"):
    t0 = time.time()
    load_mtx(name)
    print(name, time.time() - t0)
```
Record the actual numbers on this machine as your improvement target.

- [ ] **Step 2: Write the failing test (a timing regression guard, generous but real)**

```python
# tests/test_mtx_io.py
def test_load_mtx_decodes_real_fixture_reasonably_fast():
    import time
    t0 = time.time()
    load_mtx(os.path.join(FIXTURES, "gg.mtx"))
    elapsed = time.time() - t0
    # Set well below the measured pre-optimization baseline from Step 1
    # (was ~4.7s) but with real margin for slower CI/dev machines --
    # don't just barely clear the target number.
    assert elapsed < 2.0
```
Use the file's existing `_use_cached_load_mtx`-style fixture handling conventions if this would otherwise double-count against the module-level `lru_cache` already in the test suite — check how other tests in this file avoid that before adding a fresh, uncached timing call (you may need to bypass or clear the cache deliberately for this one test to measure a true cold load).

- [ ] **Step 3: Run test to verify it fails against the current decoder**

Run: `pytest tests/test_mtx_io.py -k reasonably_fast -v`
Expected: FAIL (or pass only because of test-suite caching quirks — verify it's actually measuring a cold decode, per Step 2's caution).

- [ ] **Step 4: Vectorize the row decode**

In `lc_codec.py`'s `decode_row` (or a new `decode_row_batch`/optimized variant, your call based on what's cleanest once you're looking at the real code), replace the per-value Python loop with a two-phase approach: (1) a single bulk read of the row's bytes (already the case — `mtx_io.py`'s caller already does one `f.read(row_len)` per row; check whether batching ACROSS rows via one larger read is feasible given the row table's layout, as a secondary win), (2) vectorize the actual tag/delta decode using numpy where the tag structure allows it — the 3-pack/2-pack/same-run cases can likely have their tag *bytes* identified and dispatched via `np.frombuffer`/boolean masking in one pass, even though the running `last` accumulator's serial dependency means the delta-accumulation itself may need to stay a loop (a MUCH shorter one, e.g. `np.cumsum` over pre-computed deltas once the per-tag delta values are extracted in bulk, if the tag structure permits computing all deltas independently before the cumulative sum). This is real numerical/algorithmic work — take the time to understand the tag format precisely (re-read the docstring's cited `lc_c2.c:134-200` behavior) before implementing, and verify correctness via Step 5's exhaustive test before chasing more speed.

If a full vectorization proves too risky to get exactly right in this pass, an acceptable, still-valuable fallback is: keep the per-tag-type branching logic in Python (correctness-critical, low risk) but replace the individual `values.append()`/`values.extend()` calls with pre-sized numpy array writes instead of a Python list, and profile again — even this alone may meaningfully help. Use your judgment on how far to push the full vectorization within this task; correctness must not regress.

- [ ] **Step 5: Run tests to verify they pass — correctness first, then speed**

Run: `pytest tests/test_lc_codec.py tests/test_spk_io.py tests/test_mtx_io.py -v`
Expected: PASS, all three files in full, INCLUDING `test_lc2_compress_recompresses_demo_spk_to_the_identical_bytes` (or whatever the real round-trip test is named) — this is the strongest correctness proof in the whole test suite for this code, do not let it regress. Then confirm the new timing test passes comfortably, and re-run the manual timing check from Step 1 to report the real before/after numbers.

- [ ] **Step 6: Commit**

```bash
git add lc_codec.py tests/test_mtx_io.py
git commit -m "perf: vectorize LC2 row decoding, cutting real-fixture matrix load time"
```

---

### Task 15 (Important 10): Sanitize activated-cut labels so auto-log filenames don't truncate

**Files:**
- Modify: `matrix_panel.py`
- Test: `tests/test_matrix_panel.py`

**Context:** `_activate_cut`'s label embeds `.1f`-formatted region floats directly (e.g. `"gpff.mtx y cut [3004.4, 3859.9]"`). `fit_export.auto_log_path`'s `os.path.splitext` cuts at the last dot, truncating to `..._3859_fits.jsonl`. Confirmed still live via two real stray files in the repo. Fix: sanitize the label before it's used as a path component, without changing what's shown in the Loaded Spectra list in a confusing way.

- [ ] **Step 1: Write the failing test**

```python
def test_activate_cut_auto_log_filename_is_not_truncated_by_embedded_dots(qapp, monkeypatch, tmp_path):
    import fit_mode
    logged = []
    monkeypatch.setattr(fit_mode.fit_export, "append_auto_log", lambda path, *a, **k: logged.append(path))
    # Use a real matrix fixture copied into tmp_path so os.path.dirname
    # resolves somewhere writable and inspectable.
    ...
    main_window = MainWindow()
    panel = MatrixPanel(main_window, str(tmp_matrix_path))
    panel.cut_controller.state.cut_region = (3004.4, 3859.9)
    panel._activate_cut()
    added = main_window.spectra[-1]
    from fit_export import auto_log_path
    log_path = auto_log_path(added.path)
    # The derived log filename must retain the full region info, not
    # truncate at the embedded decimal point.
    assert "3859" in os.path.basename(log_path) and "9" in os.path.basename(log_path).split("3859")[1][:3]
```

(Adjust the exact assertion once you see the real sanitized label format you implement in Step 3 — the key property is: the log filename derived from the new label must not silently drop the `.9` from `3859.9`. Match this file's existing `_activate_cut`-adjacent test setup pattern, e.g. `test_matrix_panel_activate_cut_integration_auto_logs_next_to_source_matrix_file`, rather than writing this from scratch.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matrix_panel.py -k truncat -v`
Expected: FAIL — current label produces a truncated log filename.

- [ ] **Step 3: Sanitize the label**

In `matrix_panel.py`'s `_activate_cut`, change the label construction from:
```python
        label = (
            f"{os.path.basename(self.path)} {self.working_axis} cut "
            f"[{state.cut_region[0]:.1f}, {state.cut_region[1]:.1f}]"
        )
```
to round the region bounds to integers instead of one decimal place, removing the embedded `.` that causes the `splitext` truncation:
```python
        label = (
            f"{os.path.basename(self.path)} {self.working_axis} cut "
            f"[{round(state.cut_region[0])}, {round(state.cut_region[1])}]"
        )
```
This is a minor precision loss in the DISPLAYED label only (the cut computation itself, `compute_cut(...)`, still uses the exact float region — only the label text changes), and directly removes the character class that causes truncation. Note this also changes what's shown in the main window's Loaded Spectra list for future cuts (existing behavior for old log files is unaffected).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py -v`
Expected: PASS, full file — check whether any existing test asserts the exact old `.1f`-formatted label text and update it deliberately (with a comment noting the format changed for this fix), not accidentally.

- [ ] **Step 5: Commit**

```bash
git add matrix_panel.py tests/test_matrix_panel.py
git commit -m "fix: round cut-region bounds in Activate Cut's label to avoid auto-log filename truncation"
```

---

### Task 16 (Important 11): Preserve zoom when toggling, removing, or closing a spectrum

**Files:**
- Modify: `main_window.py`
- Test: `tests/test_multi_spectrum.py` (or wherever multi-spectrum view-preservation is already tested — check `test_operations_menu.py` too, since that's where the existing `preserve_view=True` tests for Multiply/Normalize live)

**Context:** `_on_show_toggled` and `_remove_spectrum` both call plain `self._plot_data()`, resetting the zoom to the full view — inconsistent with every data-mutating operation elsewhere in the file, which explicitly preserves view for exactly this reason.

- [ ] **Step 1: Write the failing tests**

```python
def test_toggling_spectrum_visibility_preserves_zoom(qapp):
    window = MainWindow()
    # Load at least two spectra -- check how existing multi-spectrum
    # tests in this codebase construct a MainWindow with real loaded
    # spectra (likely via _load_files on real fixture paths) and match
    # that pattern.
    ...
    window.axes.set_xlim(10, 20)
    window._on_show_toggled(window.spectra[1].path, False)
    assert window.axes.get_xlim() == pytest.approx((10, 20))


def test_removing_spectrum_preserves_zoom(qapp):
    window = MainWindow()
    ...
    window.axes.set_xlim(10, 20)
    window._remove_spectrum(window.spectra[1].path)
    assert window.axes.get_xlim() == pytest.approx((10, 20))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -k "toggling_spectrum_visibility_preserves_zoom or removing_spectrum_preserves_zoom" -v`
Expected: FAIL — zoom currently resets to full view.

- [ ] **Step 3: Pass preserve_view=True**

In `main_window.py`, change:
```python
    def _on_show_toggled(self, path, checked):
        for spectrum in self.spectra:
            if spectrum.path == path:
                spectrum.visible = checked
                break
        self._plot_data()
```
to:
```python
    def _on_show_toggled(self, path, checked):
        for spectrum in self.spectra:
            if spectrum.path == path:
                spectrum.visible = checked
                break
        self._plot_data(preserve_view=True)
```
and:
```python
    def _remove_spectrum(self, path):
        removed_was_active = any(s.path == path and s.active for s in self.spectra)
        self.spectra = [s for s in self.spectra if s.path != path]
        if removed_was_active and self.spectra:
            self.spectra[0].active = True
        self._update_spectrum_list()
        self._plot_data()
```
to:
```python
    def _remove_spectrum(self, path):
        removed_was_active = any(s.path == path and s.active for s in self.spectra)
        self.spectra = [s for s in self.spectra if s.path != path]
        if removed_was_active and self.spectra:
            self.spectra[0].active = True
        self._update_spectrum_list()
        self._plot_data(preserve_view=True)
```
(`_close_active_spectrum` calls `_remove_spectrum` internally per the audit's own read of the code, so it inherits this fix automatically — confirm this is still true in the current source before assuming it, since call chains can drift.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -k "toggling_spectrum_visibility_preserves_zoom or removing_spectrum_preserves_zoom" -v` then the full suite.
Expected: PASS. Also sanity-check the case where the LAST/only spectrum is removed — `preserve_view=True` with no spectra left should not crash (`_plot_data`'s existing empty-list handling should already cover this, but verify with a quick test if not already covered).

- [ ] **Step 5: Commit**

```bash
git add main_window.py tests/test_multi_spectrum.py
git commit -m "fix: preserve zoom when toggling visibility or removing a spectrum"
```

---

### Task 17 (Important 12): Fix Save Spectrum's missing default file-extension handling

**Files:**
- Modify: `main_window.py`
- Test: `tests/test_operations_menu.py` (or wherever Save Spectrum is currently tested)

**Context:** `QFileDialog.getSaveFileName()` (the static method) has no `setDefaultSuffix` equivalent. On Linux (Qt's own cross-platform dialog, not native GTK auto-suffixing), a user typing a bare filename with no extension gets a file with none — later reload falls through to the plain-text histogram loader for what may be binary data.

- [ ] **Step 1: Read the current save flow in full**

Read `main_window.py`'s `_open_save_spectrum_dialog` and `_write_spectrum` in full (both already partially read during this plan's preparation) to confirm exact current behavior before changing it.

- [ ] **Step 2: Write the failing test**

```python
def test_write_spectrum_appends_extension_when_path_has_none(qapp, tmp_path):
    window = MainWindow()
    # Construct a minimal LoadedSpectrum directly (check how existing
    # _write_spectrum tests in this codebase build one without a real
    # file-open dialog) rather than going through the full Save dialog.
    ...
    bare_path = str(tmp_path / "myspectrum")  # no extension
    window._write_spectrum(spectrum, bare_path, "SPE files (*.spe)")
    assert os.path.exists(bare_path + ".spe") or os.path.exists(bare_path)
    # The real property under test: whatever file actually got written
    # must be loadable again through the normal extension-based dispatch
    # -- if it has no extension, this must NOT silently corrupt via the
    # histogram fallback.
    if os.path.exists(bare_path):
        assert bare_path.lower().endswith((".spe", ".spk", ".txt"))
```

(This test's exact shape depends on `_write_spectrum`'s real current signature and whether it's tested directly today, separately from the full dialog flow — check first and match the established pattern.)

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest -k write_spectrum_appends_extension -v`
Expected: FAIL — a bare path currently stays bare.

- [ ] **Step 4: Enforce a default suffix**

In `main_window.py`'s `_write_spectrum` (or `_open_save_spectrum_dialog`, whichever is the cleaner place given the real current code structure — `_write_spectrum` is likely better since it already knows the chosen filter and is the single place all save paths funnel through), add: if `path` has no recognized extension (doesn't end in `.spe`/`.spk`/`.txt`), append the extension implied by `chosen_filter` before writing. For example, near the top of `_write_spectrum`:
```python
        lower = path.lower()
        if not lower.endswith((".spe", ".spk", ".txt")):
            if chosen_filter.startswith("SPE"):
                path += ".spe"
            elif chosen_filter.startswith("SPK"):
                path += ".spk"
            elif chosen_filter.startswith("Text"):
                path += ".txt"
            lower = path.lower()
```
(Adapt to the exact current filter-string prefixes used in `_open_save_spectrum_dialog`'s filter string — verify `"SPE files (*.spe);;SPK files (*.spk);;Text files (*.txt);;All files (*)"` is still accurate before hardcoding these prefixes, and decide what happens for `chosen_filter == "All files (*)"` with no extension: defaulting to `.txt` — the histogram writer, this app's most format-agnostic one — is a reasonable fallback; state your choice in a comment.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest -v` (relevant test file, then full suite)
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add main_window.py
git commit -m "fix: enforce a real file extension on Save Spectrum, preventing extensionless files on Linux"
```

---

### Task 18 (Important 13): Fix matrix panel and heatmap window toolbar icons not following theme toggle

**Files:**
- Modify: `matrix_panel.py`, `matrix_heatmap.py`, `main_window.py`
- Test: `tests/test_matrix_panel.py`, `tests/test_matrix_heatmap.py`

**Context:** `_style_nav_toolbar_palette`/`_refresh_builtin_toolbar_icons` (`main_window.py:764-809`) fix this exact bug class for the main window's toolbar, but are only ever called from `main_window.py`. The matrix panel and heatmap window each build their own separate `nav_toolbar` and never call either. Confirmed empirically (icon bytes byte-for-byte identical before/after a theme toggle on the matrix panel).

- [ ] **Step 1: Read the exact current theme-refresh call chain**

Read `main_window.py`'s `_apply_theme`, `_style_nav_toolbar_palette`, and `_refresh_builtin_toolbar_icons` in full (already read in part during this plan's preparation). Read `matrix_panel.py`'s constructor (for `self.nav_toolbar`) and whatever theme-related method it currently has (likely none beyond the `_theme` property delegating to `main_window`). Read `matrix_heatmap.py`'s constructor's one-shot `style_axes(self.axes, theme)` call.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_matrix_panel.py
def test_matrix_panel_toolbar_icons_follow_theme_toggle(qapp):
    main_window = MainWindow()
    panel = MatrixPanel(main_window, os.path.join(FIXTURES, "gg.mtx"))
    icon_before = panel.nav_toolbar.actions()[0].icon()
    main_window._apply_theme("dark")
    # The panel's own toolbar palette must reflect the new theme --
    # check whatever main_window.py's own test for this same fix
    # (Windows toolbar icon theme fix, v2.2.2) asserts, and mirror it
    # here for the matrix panel's toolbar.
    ...
```

```python
# tests/test_matrix_heatmap.py
def test_heatmap_window_toolbar_icons_follow_theme_toggle(qapp):
    ...
```

(Match whichever concrete assertion technique `main_window.py`'s own existing icon-theme test uses — e.g. comparing `QIcon` pixmap bytes before/after, or checking `nav_toolbar.palette().color(...)` — reuse that exact technique rather than inventing a new one, since it's already proven to correctly detect this bug class.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_matrix_panel.py tests/test_matrix_heatmap.py -k toolbar_icons_follow_theme -v`
Expected: FAIL.

- [ ] **Step 4: Wire theme refresh into the matrix panel and heatmap window**

In `matrix_panel.py`, add a method mirroring `main_window.py`'s (reusing `main_window.py`'s exact palette-setting logic rather than re-deriving it — consider whether to import and call `main_window`'s helper functions directly, passing `self.nav_toolbar`, since they're already written generically against a `nav_toolbar` argument in spirit even if currently coupled to `self`; adapt as needed):
```python
    def _refresh_toolbar_theme(self):
        from main_window import MainWindow
        MainWindow._style_nav_toolbar_palette(self, self._theme)
        MainWindow._refresh_builtin_toolbar_icons(self)
```
(This works only if those two methods on `MainWindow` reference nothing but `self.nav_toolbar`/`self._theme`/`self.zoom_in_action` etc. that `MatrixPanel` also has under the same names — verify this precisely by re-reading both methods; if they reference something matrix panels don't have, refactor them into free functions taking `nav_toolbar`/`theme` as parameters instead, called from both `MainWindow` and `MatrixPanel`, which is the cleaner fix if the coupling doesn't line up cleanly.)

Call `self._refresh_toolbar_theme()` from wherever `MatrixPanel` currently reacts to a theme change — check whether it has an existing hook, or whether you need to make `main_window._apply_theme` loop over `self._matrix_panels` (and their `_heatmap_windows`) calling a refresh method on each, the same way `_apply_calibration_change` already loops over `self._matrix_panels`. This second approach (main_window driving every open panel/heatmap explicitly) is likely more consistent with this codebase's existing calibration-sharing pattern — prefer it unless you find a good reason not to.

For `matrix_heatmap.py`, add an equivalent `_refresh_theme(self, theme)` method (re-run `style_axes(self.axes, theme)` plus the same toolbar-icon refresh, then `self.canvas.draw()`), called from the same main_window loop.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_matrix_panel.py tests/test_matrix_heatmap.py -v`
Expected: PASS, both files in full.

- [ ] **Step 6: Commit**

```bash
git add matrix_panel.py matrix_heatmap.py main_window.py tests/test_matrix_panel.py tests/test_matrix_heatmap.py
git commit -m "fix: matrix panel and heatmap window toolbar icons now follow theme toggle"
```

---

### Task 19 (Important 14): Document the matrix panel's "Clear Marks" button

**Files:**
- Modify: `help_content.py`
- Test: `tests/test_help_content.py`

**Context:** A real, always-visible button (`matrix_panel.py:241-242,252`) that clears cut/background marks. Appears nowhere in the HowTo page.

- [ ] **Step 1: Read the current "Matrix panel" shortcuts table and section 11 prose**

Read `help_content.py`'s `<h3>Matrix panel</h3>` table and `<h3>11. Matrix analysis</h3>` prose in full to match established formatting exactly.

- [ ] **Step 2: Write the failing test**

```python
def test_howto_html_documents_clear_marks_button():
    html = build_howto_html()
    assert "Clear Marks" in html
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_help_content.py -k clear_marks_button -v`
Expected: FAIL.

- [ ] **Step 4: Add documentation**

Add a short mention in the section 11 prose (near where hold-C/hold-G marking is described) noting that **Clear Marks** clears any in-progress cut/background marks without affecting an already-activated cut. This is a button, not a shortcut, so it doesn't need a shortcuts-table row (check whether the table's convention includes button-only entries anywhere else first — if it does, add a row too; if the table is shortcut-only by design, prose alone is the right place).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_help_content.py -v`
Expected: PASS, full file.

- [ ] **Step 6: Commit**

```bash
git add help_content.py tests/test_help_content.py
git commit -m "docs: document the matrix panel's Clear Marks button"
```

---

### Task 20 (Important 15): Document the "Fix" parameter checkbox

**Files:**
- Modify: `fit_mode.py` (tooltip), `help_content.py` (docs)
- Test: `tests/test_help_content.py`

**Context:** The "Fix" checkbox (`fit_mode.py:858`) has no tooltip and no mention in either help page, unlike its sibling "Independent widths"/"Left tail" checkboxes.

- [ ] **Step 1: Read the current Fit Parameters panel construction and the relevant HowTo/KB sections**

Read `fit_mode.py`'s parameters-table construction (where `fix_checkbox = QCheckBox()` is created) and how "Independent widths"/"Left tail" get their tooltips, for the pattern to match. Read `help_content.py`'s "7. Performing a fit" section and the Knowledge Database's fitting-parameters section.

- [ ] **Step 2: Write the failing test**

```python
def test_howto_html_documents_fix_checkbox():
    html = build_howto_html()
    assert "Fix" in html and ("checkbox" in html.lower() or "fixed" in html.lower())
```
(Write a more specific assertion once you've drafted the real prose — a bare `"Fix" in html` risks false-passing on unrelated text; check for a distinctive phrase you're about to add instead.)

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_help_content.py -k fix_checkbox -v`
Expected: FAIL.

- [ ] **Step 4: Add the tooltip and documentation**

In `fit_mode.py`, add a tooltip to the Fix checkbox matching the style of the existing "Independent widths"/"Left tail" tooltips:
```python
        fix_checkbox.setToolTip(
            "Locks this parameter at its current Value for the next fit. "
            "Leave unchecked to use the Value as a starting guess the fit can still adjust."
        )
```
(Adapt to whatever tooltip-setting convention the sibling checkboxes actually use — direct `.setToolTip()` call vs. some helper — match it exactly.)

In `help_content.py`'s "7. Performing a fit" section, add a short paragraph explaining the Fix checkbox's two modes (locked value vs. movable starting guess), matching the page's established prose style.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_help_content.py -v`
Expected: PASS, full file.

- [ ] **Step 6: Commit**

```bash
git add fit_mode.py help_content.py tests/test_help_content.py
git commit -m "docs: document the Fix parameter checkbox (tooltip + HowTo)"
```

---

### Task 21 (Important 16): Update README.md's stale install instructions

**Files:**
- Modify: `README.md`

**Context:** Still names `SpectraTools-v2.0.0-Setup.exe` and `spectratools_2.0.0_amd64.deb`/`spectratools-2.0.0-1.x86_64.rpm` — six releases stale.

- [ ] **Step 1: Update the Installing section**

In `README.md`, change:
```
Pre-built releases are under `releases/vX.Y.Z/` (the latest is
`releases/v2.0.0/`). Pick your platform:
```
to reference the latest actual release directory (check `releases/` on disk for the real current latest — this plan doesn't freeze a new version itself, so use whatever the latest already-shipped one is, currently `v3.0.0`, but verify against the actual directory listing rather than hardcoding blindly):
```
Pre-built releases are under `releases/vX.Y.Z/` (the latest is
`releases/v3.0.0/`). Pick your platform:
```
Change `Run `SpectraTools-v2.0.0-Setup.exe`` to `Run `SpectraTools-v3.0.0-Setup.exe``, and update the Linux install commands:
```
sudo apt install ./spectratools_3.0.0_amd64.deb      # Debian/Ubuntu
sudo dnf install ./spectratools-3.0.0-1.x86_64.rpm    # RHEL/CentOS/AlmaLinux/Rocky
```

Consider (and decide, don't leave as a TODO) whether to instead phrase this section in version-agnostic terms (e.g. "the latest release directory" with a `<version>` placeholder, matching how `packaging/linux/INSTALL.md` already does it) so it doesn't go stale again on the next release — this is likely the better long-term fix; if you take this approach, make sure the actual commands still work as copy-paste-able examples with a real version number shown once.

- [ ] **Step 2: Verify no test depends on README.md's exact text**

Run: `pytest -k readme -v` (there almost certainly isn't one, but confirm) and `grep -rn "README" tests/` to be sure.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README's stale v2.0.0 install instructions to reflect the current release"
```

---

### Task 22 (Important 17): Soften or verify the untested Fedora support claim

**Files:**
- Modify: `packaging/linux/INSTALL.md`

**Context:** States Fedora support as flat fact; only AlmaLinux 8 and 10 were ever actually tested. The design spec's own internal language is more careful (hedges EL9), but that hedge never reached the user-facing doc.

- [ ] **Step 1: Soften the claim to match the design spec's own honesty**

In `packaging/linux/INSTALL.md`, change the `## RHEL / CentOS / AlmaLinux 9, 10, and current Fedora` section's framing from flat fact to an accurate statement of what's actually been verified vs. expected. Example:
```
## RHEL / CentOS / AlmaLinux 9, 10, and current Fedora

No EPEL needed -- install directly:
```
sudo dnf install ./spectratools-<version>-1.x86_64.rpm
```

AlmaLinux 9 and 10 are directly tested. Fedora and other RHEL-family
distributions aren't separately tested, but are expected to work the same
way (RHEL-family and Fedora share the same core package names this app
depends on). If `dnf install` reports a missing dependency on Fedora,
please open an issue -- it likely means a package name has diverged.

Uninstall:
```
sudo dnf remove spectratools
```
```
(Adjust wording to match the file's existing tone; don't just paste this verbatim without reading the surrounding sections for consistency.)

- [ ] **Step 2: Verify no test depends on this file's exact text**

Run: `grep -rn "INSTALL.md" tests/` to confirm.

- [ ] **Step 3: Commit**

```bash
git add packaging/linux/INSTALL.md
git commit -m "docs: soften untested Fedora support claim in Linux INSTALL.md to match what's actually verified"
```

---

### Task 23 (Important 18): Document the Windows onefile vs. Linux onedir packaging asymmetry

**Files:**
- Modify: `packaging/windows/build.ps1` or `README.md` or a new short note (your call on the best location — a comment at the top of `build.ps1` near the `--onefile` flag is likely the most durable, since that's where a future reader would actually look when touching this)

**Context:** The original onefile-Windows/onedir-Linux split was justified partly by AppImage's AppDir structure, which no longer exists (Linux packaging moved to native .deb/.rpm). Nobody revisited whether onefile still makes sense for Windows. This task is explicitly scoped as documentation, not a packaging-format change — switching Windows to onedir is a bigger decision with real UX tradeoffs (a folder instead of one exe) that wasn't asked for here; don't make that change.

- [ ] **Step 1: Add a note explaining the current tradeoff is a real, considered one (not an oversight)**

Near the `--onefile` flag in `packaging/windows/build.ps1`, add a comment:
```powershell
# --onefile (vs. Linux's --onedir, see build.sh): originally paired with
# Inno Setup convenience and, historically, the now-removed AppImage's
# own AppDir structure on the Linux side. That second reason no longer
# applies (Linux packaging moved to native .deb/.rpm), but onefile is
# kept deliberately for Windows: Inno Setup packages a single exe
# cleanly, and onedir's many loose files would need their own directory
# layout decision in the installer. Tradeoff: onefile's PyInstaller
# bootloader re-extracts the whole bundle to a temp dir on every launch
# (no persistent cache), so Windows cold-starts slower than Linux's
# onedir build does on equivalent hardware, and onefile executables are
# a more common antivirus false-positive target. Revisit if startup
# time or AV false positives become a real user complaint -- see the
# v3.1.0 audit finding that first raised this.
```

- [ ] **Step 2: Verify no test depends on this**

Run: `pytest -k onefile -v` (almost certainly none, confirm quickly).

- [ ] **Step 3: Commit**

```bash
git add packaging/windows/build.ps1
git commit -m "docs: explain the deliberate onefile-Windows/onedir-Linux packaging tradeoff"
```

---

## Self-Review Notes

- **Spec coverage:** all 5 Critical and all 18 Important findings from the v3.1.0 audit are covered, one task each (Tasks 1-23), in Critical-first then Important-grouped-by-subsystem order.
- **Dependency ordering:** Task 13 (LC2 dedup) is placed before Task 14 (mtx_io performance) so the optimization is written once, in the shared location. Tasks 2 and 11 both touch `histogram_io.py`'s `load_histogram` — Task 11 explicitly notes it must read Task 2's already-modified `try` structure rather than assuming the pre-Task-2 shape.
- **Out of scope, deliberately:** all 24 Minor findings (separate follow-on plan, per the user's own "afterwards try to fix the rest" sequencing). No version bump, tag, or release — this plan only fixes code/docs; freezing v3.1.0 happens later, on explicit request, matching this project's standing convention.
- **Verification convention carried forward from every prior plan in this project:** after each task's own tests pass, the full `pytest -q` suite should also be run before considering the task truly done (subagent reviewers in this project have repeatedly reported no Bash access — the controller running the session is expected to independently verify via pytest after every task, not just trust the implementer's own report).
