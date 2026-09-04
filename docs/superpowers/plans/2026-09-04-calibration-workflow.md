# Calibration Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the one-shot energy-calibration dialog into a workflow: assignments persist per spectrum, the fit is shown in a plot window with its coefficients and reduced chi-squared, the result can be exported for CalEnEff, and the Fit Results panel is rearranged.

**Architecture:** Four independent slices over the existing `EnergyAssignDialog`. Two new pure modules (`value_format.py`, `caleneff_export.py`) carry all the logic worth testing without Qt; one new Qt dialog (`calibration_plot_dialog.py`) draws the fit; `spectrum.py` gains a per-spectrum record; `fit_mode.py`'s results table is re-columned. Nothing in `calibration.py` or `peak_fit.py` changes.

**Tech Stack:** Python 3.13, PySide6 (Qt Widgets), matplotlib (Qt5Agg backend via `FigureCanvasQTAgg`), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-calibration-workflow-design.md`

## Global Constraints

- **Never `git add -A`.** ~149 MB of untracked third-party sources and sample spectra sit in the working tree deliberately. Stage explicit paths only, or use `git add -u`.
- Run tests with `./.venv/Scripts/python.exe -m pytest`, never bare `pytest`.
- Existing suite baseline is **1227 passing**. Every task must leave it green.
- Readers raise `histogram_io.ParseError`; calibration errors are `calibration.CalibrationError`; fit errors are `fit_mode.FitError`. Do not invent new exception hierarchies.
- Uncertainties from `peak_fit` may be **NaN** (a parameter the data does not constrain) or **0.0** (a parameter held fixed). Neither is a usable uncertainty. Every formatter and every export must handle both.
- `.sou` intensities are on arbitrary per-file scales. Only `caleneff_export` normalises them; nothing else may assume a scale.
- Match the surrounding comment style: explain *why*, not *what*. Files here carry dense rationale comments; terse code with no reasoning will fail review.
- Do not add `-dev` suffixes or touch `packaging/windows/installer.iss`. Version is already 5.0.0.

---

### Task 1: `value_format.compact` — compact uncertainty notation

**Files:**
- Create: `value_format.py`
- Test: `tests/test_value_format.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `compact(value, error) -> str`. Used by Task 5 (results panel) and Task 3 (plot window).

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the compact uncertainty notation, e.g. 352.7217(14).

The parenthesised digits are the uncertainty expressed in units of the
last decimal place shown, the convention used in nuclear data tables.
"""

import math

import pytest

from value_format import compact


def test_basic_two_significant_digits():
    # 0.00014 -> "14" in the last two decimal places of 352.72172.
    assert compact(352.72172, 0.00014) == "352.72172(14)"


def test_uncertainty_rounded_to_two_significant_digits():
    # 0.00123 -> 0.0012, so the value is shown to 4 decimals.
    assert compact(12.345678, 0.00123) == "12.3457(12)"


def test_large_uncertainty_moves_the_decimal_place():
    # 3.7 -> two sig digits is "3.7", one decimal place.
    assert compact(1234.5678, 3.7) == "1234.6(37)"


def test_uncertainty_of_ten_or_more():
    # 42.0 -> "42", zero decimal places, so no decimal point at all.
    assert compact(1234.5678, 42.0) == "1235(42)"


def test_rounding_carry_in_the_uncertainty():
    # 0.0099 rounds to 0.0099 at two sig digits (not 0.01).
    assert compact(5.0, 0.0099) == "5.0000(99)"


def test_zero_error_gives_a_bare_value():
    """A position held fixed reports exactly 0.0. Printing '(0)' would
    read as a perfectly known value."""
    assert compact(352.7217, 0.0) == "352.7217"


def test_negative_error_gives_a_bare_value():
    assert compact(352.7217, -1.0) == "352.7217"


def test_nan_error_gives_a_bare_value():
    """peak_fit reports NaN for a parameter the data does not constrain."""
    assert compact(352.7217, float("nan")) == "352.7217"


def test_none_error_gives_a_bare_value():
    assert compact(352.7217, None) == "352.7217"


def test_error_larger_than_value_is_still_shown():
    """Unusual but real. Hiding it would be worse than showing it."""
    assert compact(0.5, 2.0) == "0(20)"


def test_non_finite_value_is_an_em_dash():
    assert compact(float("nan"), 0.1) == "—"
    assert compact(float("inf"), 0.1) == "—"


def test_negative_value_keeps_its_sign():
    assert compact(-12.345678, 0.00123) == "-12.3457(12)"


def test_bare_value_of_zero_error_is_not_over_rounded():
    """With no usable uncertainty there is nothing to set the precision,
    so fall back to a readable fixed number of decimals."""
    assert compact(1234.5, 0.0) == "1234.5"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_value_format.py -q`
Expected: FAIL — collection error, `ModuleNotFoundError: No module named 'value_format'`

- [ ] **Step 3: Write the implementation**

Create `value_format.py`:

```python
"""Compact value-with-uncertainty notation, e.g. 352.7217(14).

The digits in parentheses are the uncertainty expressed in units of the
last decimal place shown, which is how nuclear data tables quote a
measurement. It is the notation the Fit Results panel and the
calibration plot window both use, and it lives in its own module so the
two cannot drift apart -- a value rendered one way in the table and
another in the plot of the same fit would be worse than either.
"""

import math

#: Significant digits kept in the uncertainty. Two is the usual choice in
#: nuclear data compilations: one digit throws away real precision when
#: the leading digit is 1, and three implies more than a fit can support.
_UNCERTAINTY_SIG_DIGITS = 2

#: Decimals used when there is no usable uncertainty to set the scale.
_FALLBACK_DECIMALS = 4


def _usable(error):
    """An uncertainty that actually says something.

    peak_fit reports 0.0 for a parameter held fixed and NaN for one the
    data does not constrain. Neither is a measurement, and rendering
    either as '(0)' or '(nan)' would claim something false.
    """
    if error is None:
        return False
    try:
        value = float(error)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def compact(value, error):
    """'352.7217(14)', or the value alone when `error` is unusable.

    Returns an em dash for a non-finite value: there is no number to
    show, and printing 'nan' in a results table reads as data rather
    than as an absent measurement.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"

    if not _usable(error):
        # Trailing zeros trimmed so a value with no uncertainty does not
        # imply precision it never claimed.
        text = f"{number:.{_FALLBACK_DECIMALS}f}".rstrip("0").rstrip(".")
        return text or "0"

    sigma = float(error)
    # Decimal place of the uncertainty's leading digit, then widened to
    # keep _UNCERTAINTY_SIG_DIGITS of it. exponent 0 means units, -2
    # means hundredths.
    exponent = math.floor(math.log10(sigma)) - (_UNCERTAINTY_SIG_DIGITS - 1)
    decimals = max(0, -exponent)

    # Round the uncertainty FIRST, then read its digits off at that same
    # decimal place. Doing it the other way round lets 0.0099 print as
    # '(99)' beside a value rounded as if sigma were 0.01.
    quantum = 10.0 ** exponent
    sigma_rounded = round(sigma / quantum) * quantum
    if sigma_rounded > 0:
        # Rounding can carry (9.99 -> 10.0), which moves the decimal place.
        new_exponent = math.floor(math.log10(sigma_rounded)) - (_UNCERTAINTY_SIG_DIGITS - 1)
        if new_exponent != exponent:
            exponent = new_exponent
            decimals = max(0, -exponent)
            quantum = 10.0 ** exponent
            sigma_rounded = round(sigma / quantum) * quantum

    digits = int(round(sigma_rounded / quantum))
    return f"{number:.{decimals}f}({digits})"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_value_format.py -q`
Expected: PASS, 13 passed

- [ ] **Step 5: Commit**

```bash
git add value_format.py tests/test_value_format.py
git commit -m "feat: compact value(uncertainty) notation"
```

---

### Task 2: `caleneff_export` — the seven-column writer

**Files:**
- Create: `caleneff_export.py`
- Test: `tests/test_caleneff_export.py`

**Interfaces:**
- Consumes: `sou_io.SourceLine` (fields `energy`, `energy_err`, `intensity`, `intensity_err`).
- Produces:
  - `ExportRow` dataclass: `channel, channel_err, area, area_err, energy, intensity_pct, intensity_pct_err`
  - `build_rows(assignments, source_lines) -> (rows, skipped)` where `assignments` is a list of `(channel, channel_err, area, area_err, energy)` tuples and `skipped` is the count with no matching source line.
  - `write_caleneff(path, rows) -> None`, raising `ExportError`.
  - `ExportError(Exception)`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the CalEnEff seven-column export.

Format (from CalEnEff's own ra226_gui.py:18, "ABSOLUTE uncertainties"):
    ch  delta_ch  N  delta_N  E[keV]  I[%]  delta_I[%]
"""

import numpy as np
import pytest

from caleneff_export import ExportError, build_rows, write_caleneff
from sou_io import SourceLine


def _lines():
    # Strongest is 10000 at 344.276, so the scale factor is 100/10000.
    return [
        SourceLine(121.783, 0.002, 5000.0, 50.0),
        SourceLine(344.276, 0.004, 10000.0, 80.0),
        SourceLine(1408.011, 0.010, 2000.0, 30.0),
    ]


def test_intensity_is_normalised_so_the_strongest_line_is_100():
    rows, skipped = build_rows(
        [(100.0, 0.01, 5000.0, 70.0, 344.276)], _lines()
    )
    assert skipped == 0
    assert rows[0].intensity_pct == pytest.approx(100.0)
    assert rows[0].intensity_pct_err == pytest.approx(0.8)


def test_other_lines_scale_by_the_same_factor():
    rows, _ = build_rows(
        [(50.0, 0.01, 900.0, 30.0, 121.783),
         (400.0, 0.02, 300.0, 20.0, 1408.011)],
        _lines(),
    )
    assert rows[0].intensity_pct == pytest.approx(50.0)
    assert rows[0].intensity_pct_err == pytest.approx(0.5)
    assert rows[1].intensity_pct == pytest.approx(20.0)
    assert rows[1].intensity_pct_err == pytest.approx(0.3)


def test_fit_quantities_pass_through_unchanged():
    rows, _ = build_rows(
        [(352.72172, 0.00014, 15787.6, 12.3, 344.276)], _lines()
    )
    row = rows[0]
    assert row.channel == pytest.approx(352.72172)
    assert row.channel_err == pytest.approx(0.00014)
    assert row.area == pytest.approx(15787.6)
    assert row.area_err == pytest.approx(12.3)
    assert row.energy == pytest.approx(344.276)


def test_an_energy_with_no_matching_source_line_is_skipped_and_counted():
    """A hand-typed energy has no intensity, so it cannot be exported."""
    rows, skipped = build_rows(
        [(100.0, 0.01, 900.0, 30.0, 344.276),
         (200.0, 0.01, 500.0, 20.0, 999.999)],
        _lines(),
    )
    assert len(rows) == 1
    assert skipped == 1
    assert rows[0].energy == pytest.approx(344.276)


def test_no_source_lines_skips_everything():
    rows, skipped = build_rows([(100.0, 0.01, 900.0, 30.0, 344.276)], [])
    assert rows == []
    assert skipped == 1


def test_matching_tolerates_float_representation():
    """The energy stored on an assignment came from the same file, so it
    must match even after a round-trip through text."""
    rows, skipped = build_rows(
        [(100.0, 0.01, 900.0, 30.0, float("344.276"))], _lines()
    )
    assert skipped == 0


def test_written_file_reads_back_with_loadtxt(tmp_path):
    """CalEnEff parses with np.loadtxt, so that is what the test uses."""
    rows, _ = build_rows(
        [(352.72172, 0.00014, 15787.6, 12.3, 344.276),
         (500.0, 0.002, 4000.0, 9.0, 121.783)],
        _lines(),
    )
    path = tmp_path / "out.txt"
    write_caleneff(str(path), rows)

    data = np.loadtxt(str(path), ndmin=2)
    assert data.shape == (2, 7)
    assert data[0, 0] == pytest.approx(352.72172)
    assert data[0, 1] == pytest.approx(0.00014)
    assert data[0, 2] == pytest.approx(15787.6)
    assert data[0, 3] == pytest.approx(12.3)
    assert data[0, 4] == pytest.approx(344.276)
    assert data[0, 5] == pytest.approx(100.0)
    assert data[0, 6] == pytest.approx(0.8)


def test_written_file_has_no_header(tmp_path):
    rows, _ = build_rows([(100.0, 0.01, 900.0, 30.0, 344.276)], _lines())
    path = tmp_path / "out.txt"
    write_caleneff(str(path), rows)
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert not first.lstrip().startswith("#")
    assert len(first.split()) == 7


def test_writing_no_rows_is_refused(tmp_path):
    """Better than an empty file CalEnEff would fail to load."""
    path = tmp_path / "out.txt"
    with pytest.raises(ExportError, match="no rows"):
        write_caleneff(str(path), [])
    assert not path.exists()


def test_unwritable_path_raises_export_error(tmp_path):
    rows, _ = build_rows([(100.0, 0.01, 900.0, 30.0, 344.276)], _lines())
    with pytest.raises(ExportError):
        write_caleneff(str(tmp_path / "nope" / "out.txt"), rows)


def test_zero_max_intensity_is_refused():
    """Every line at zero intensity gives no scale to normalise by."""
    lines = [SourceLine(100.0, 0.1, 0.0, 0.0)]
    with pytest.raises(ExportError, match="intensity"):
        build_rows([(10.0, 0.1, 500.0, 20.0, 100.0)], lines)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_caleneff_export.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'caleneff_export'`

- [ ] **Step 3: Write the implementation**

Create `caleneff_export.py`:

```python
"""Writes fitted peaks in CalEnEff's efficiency-calibration format.

CalEnEff (C:\\Users\\RIG\\Documents\\Claude\\efficieny) reads seven
whitespace-separated columns with no header, documented in its own
ra226_gui.py:18 as carrying ABSOLUTE uncertainties:

    ch   delta_ch   N   delta_N   E[keV]   I[%]   delta_I[%]

The first four come from the fit, the last three from the .sou source
line the peak was assigned to.

N is the NET area. CalEnEff computes eff = N / I_pct, so a gross area
would fold the background into the efficiency curve.
"""

import math
import os
from dataclasses import dataclass

#: The strongest line in a source is normalised to this. .sou intensities
#: have no common convention -- the nine sample files peak at values from
#: 1000 to 100000 -- so a scale has to be chosen. Normalising to the
#: strongest line matches CalEnEff's own 226Ra sample, where 609.312 keV
#: carries I = 100. Because eff = N / I_pct, a constant factor rescales
#: the whole efficiency curve without changing its shape, and the
#: relative uncertainties CalEnEff propagates are invariant under it.
_STRONGEST_LINE_PERCENT = 100.0

#: Energies are matched to source lines by value, not by index, because
#: the assignment stores the energy alone. Both sides came from the same
#: file, so this only has to absorb float round-tripping through text.
_ENERGY_MATCH_TOLERANCE = 1e-6


class ExportError(Exception):
    """Raised when the file cannot be produced or would be unusable."""


@dataclass(frozen=True)
class ExportRow:
    channel: float
    channel_err: float
    area: float
    area_err: float
    energy: float
    intensity_pct: float
    intensity_pct_err: float


def build_rows(assignments, source_lines):
    """(rows, skipped) for `assignments`, a list of
    (channel, channel_err, area, area_err, energy) tuples.

    An assignment whose energy matches no source line is SKIPPED and
    counted rather than exported with a zero intensity: CalEnEff refuses
    rows where dN and dI are both zero, because the efficiency
    uncertainty would come out zero, and a made-up intensity would
    silently distort the curve.
    """
    if source_lines:
        maximum = max(line.intensity for line in source_lines)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ExportError(
                "The source has no positive intensity to normalise by, so "
                "I[%] cannot be computed."
            )
        scale = _STRONGEST_LINE_PERCENT / maximum
    else:
        scale = None

    rows, skipped = [], 0
    for channel, channel_err, area, area_err, energy in assignments:
        line = _matching_line(energy, source_lines)
        if line is None or scale is None:
            skipped += 1
            continue
        rows.append(ExportRow(
            channel=float(channel),
            channel_err=float(channel_err),
            area=float(area),
            area_err=float(area_err),
            energy=float(energy),
            intensity_pct=line.intensity * scale,
            intensity_pct_err=line.intensity_err * scale,
        ))
    return rows, skipped


def _matching_line(energy, source_lines):
    for line in source_lines:
        if abs(line.energy - energy) <= _ENERGY_MATCH_TOLERANCE:
            return line
    return None


def write_caleneff(path, rows):
    """Write `rows` as CalEnEff's seven columns.

    Refuses an empty file: CalEnEff would fail to load it, and failing
    here names the reason while the user is still looking at the dialog.
    """
    if not rows:
        raise ExportError(
            "There are no rows to export: no assigned peak had an "
            "intensity from a source file."
        )
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    f"{row.channel:<14.6f} {row.channel_err:<12.6f} "
                    f"{row.area:<12.1f} {row.area_err:<10.2f} "
                    f"{row.energy:<10.3f} {row.intensity_pct:<8.3f} "
                    f"{row.intensity_pct_err:.3f}\n"
                )
    except OSError as exc:
        raise ExportError(f"Cannot write {path}: {exc}") from None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_caleneff_export.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add caleneff_export.py tests/test_caleneff_export.py
git commit -m "feat: CalEnEff seven-column export"
```

---

### Task 3: Reduced chi-squared for a calibration fit

**Files:**
- Create: `calibration_quality.py`
- Test: `tests/test_calibration_quality.py`

**Interfaces:**
- Consumes: `calibration.Calibration` (`apply`, `derivative`, `kind`).
- Produces: `reduced_chi_squared(calibration, channels, energies, channel_errors) -> (value_or_None, reason_or_None)`. Task 4 renders it.

- [ ] **Step 1: Write the failing tests**

```python
"""Reduced chi-squared for an energy calibration, and the three cases
where it does not exist."""

import math

import pytest

from calibration import Calibration, from_points
from calibration_quality import reduced_chi_squared


def test_a_perfect_fit_has_chi_squared_of_zero():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    channels = [100.0, 200.0, 300.0]
    energies = [cal.apply(c) for c in channels]
    value, reason = reduced_chi_squared(cal, channels, energies, [0.1, 0.1, 0.1])
    assert reason is None
    assert value == pytest.approx(0.0, abs=1e-12)


def test_value_matches_an_independent_computation():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    channels = [100.0, 200.0, 300.0, 400.0]
    errors = [0.1, 0.2, 0.1, 0.4]
    energies = [cal.apply(c) + d for c, d in
                zip(channels, (0.05, -0.03, 0.02, 0.10))]

    expected_chi2 = sum(
        ((e - cal.apply(c)) / (s * cal.derivative(c))) ** 2
        for c, e, s in zip(channels, energies, errors)
    )
    expected = expected_chi2 / (len(channels) - 2)

    value, reason = reduced_chi_squared(cal, channels, energies, errors)
    assert reason is None
    assert value == pytest.approx(expected)


def test_quadratic_uses_three_parameters():
    cal = Calibration(kind="quadratic", a=1.0, b=0.5, c=1e-6)
    channels = [100.0, 200.0, 300.0, 400.0]
    errors = [0.1] * 4
    energies = [cal.apply(c) + 0.01 for c in channels]

    expected_chi2 = sum(
        ((e - cal.apply(ch)) / (s * cal.derivative(ch))) ** 2
        for ch, e, s in zip(channels, energies, errors)
    )
    value, _ = reduced_chi_squared(cal, channels, energies, errors)
    assert value == pytest.approx(expected_chi2 / (len(channels) - 3))


def test_unusable_errors_report_an_unweighted_fit():
    """from_points falls back to an unweighted fit when any error is
    zero, negative or non-finite. The weights are then arbitrary and a
    printed chi-squared would look like a goodness of fit."""
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    channels = [100.0, 200.0, 300.0]
    energies = [60.0, 110.0, 160.0]
    for errors in ([0.1, 0.0, 0.1], [0.1, -1.0, 0.1],
                   [0.1, float("nan"), 0.1], [0.1, None, 0.1]):
        value, reason = reduced_chi_squared(cal, channels, energies, errors)
        assert value is None
        assert "unweighted" in reason


def test_exactly_the_minimum_points_has_no_degrees_of_freedom():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    value, reason = reduced_chi_squared(
        cal, [100.0, 200.0], [60.0, 110.0], [0.1, 0.1]
    )
    assert value is None
    assert "degrees of freedom" in reason


def test_missing_errors_entirely_is_unweighted():
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    value, reason = reduced_chi_squared(
        cal, [100.0, 200.0, 300.0], [60.0, 110.0, 160.0], None
    )
    assert value is None
    assert "unweighted" in reason


def test_a_zero_derivative_is_reported_rather_than_dividing_by_zero():
    """A quadratic's vertex has dE/dch == 0, so a point sitting exactly
    there has no finite energy uncertainty."""
    cal = Calibration(kind="quadratic", a=0.0, b=1.0, c=-0.005)
    vertex = 100.0  # -b / (2c)
    channels = [50.0, vertex, 150.0]
    energies = [cal.apply(c) for c in channels]
    value, reason = reduced_chi_squared(cal, channels, energies, [0.1, 0.1, 0.1])
    assert value is None
    assert reason is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_calibration_quality.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'calibration_quality'`

- [ ] **Step 3: Write the implementation**

Create `calibration_quality.py`:

```python
"""How well an energy calibration fits the points it was built from.

Kept out of calibration.py deliberately: a Calibration is a coordinate
transform and is used in places that have no points to judge it by, so
its identity should not carry a goodness-of-fit number. This module is
the judge, and it is pure -- no Qt, no spectra.
"""

import math

#: Free parameters per calibration kind, the p in (n - p).
_PARAMETERS = {"linear": 2, "quadratic": 3}


def _usable_errors(channel_errors, count):
    """The errors as floats, or None if any one of them is unusable.

    All-or-nothing on purpose, matching from_points: it falls back to an
    unweighted fit if ANY error is missing, zero, negative or non-finite,
    so a chi-squared computed from the usable subset would be judging a
    fit that was never performed.
    """
    if channel_errors is None or len(channel_errors) != count:
        return None
    values = []
    for error in channel_errors:
        if error is None:
            return None
        try:
            value = float(error)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        values.append(value)
    return values


def reduced_chi_squared(calibration, channels, energies, channel_errors):
    """(value, None) or (None, reason).

    chi2 = sum( ((E - f(ch)) / (sigma_ch * dE/dch))**2 ) / (n - p)

    The channel uncertainty is converted to an energy uncertainty
    through the calibration's own local slope, which is what makes the
    residual and its uncertainty comparable.
    """
    count = len(channels)
    parameters = _PARAMETERS.get(calibration.kind, 2)

    if count <= parameters:
        return None, "undefined (no degrees of freedom)"

    errors = _usable_errors(channel_errors, count)
    if errors is None:
        return None, "undefined (unweighted fit)"

    total = 0.0
    for channel, energy, sigma in zip(channels, energies, errors):
        slope = calibration.derivative(channel)
        energy_sigma = abs(sigma * slope)
        if not math.isfinite(energy_sigma) or energy_sigma <= 0.0:
            # dE/dch is zero at a quadratic's vertex, where a channel
            # uncertainty maps to no energy uncertainty at all.
            return None, "undefined (a point sits at the calibration's turning point)"
        total += ((energy - calibration.apply(channel)) / energy_sigma) ** 2

    value = total / (count - parameters)
    if not math.isfinite(value):
        return None, "undefined"
    return value, None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_calibration_quality.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add calibration_quality.py tests/test_calibration_quality.py
git commit -m "feat: reduced chi-squared for a calibration fit"
```

---

### Task 4: Per-spectrum assignment persistence

**Files:**
- Modify: `spectrum.py` (add to `LoadedSpectrum.__init__`, around line 35-50)
- Create: `energy_assignments.py`
- Test: `tests/test_energy_assignments.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `EnergyAssignments` frozen dataclass: `source_path` (str or None), `pairs` (tuple of `(channel, energy)`).
  - `restore(assignments, peaks) -> dict[int, float]` mapping row index to energy, where `peaks` is a list of `(channel, fwhm)`.
  - `LoadedSpectrum.energy_assignments`, initialised to `None`.

- [ ] **Step 1: Write the failing tests**

```python
"""Restoring stored energy assignments onto a freshly refitted spectrum."""

import pytest

from energy_assignments import EnergyAssignments, restore


def test_exact_channels_are_restored():
    stored = EnergyAssignments(source_path="eu152.sou",
                               pairs=((100.0, 121.783), (300.0, 344.276)))
    got = restore(stored, [(100.0, 4.0), (200.0, 4.0), (300.0, 4.0)])
    assert got == {0: 121.783, 2: 344.276}


def test_a_small_shift_from_refitting_still_matches():
    """A refit moves a centroid by far less than its own width."""
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    got = restore(stored, [(100.4, 4.0)])
    assert got == {0: 121.783}


def test_a_shift_larger_than_the_peak_width_does_not_match():
    """Past its own width it is a different peak, and a wrong energy
    silently restored would look like a valid calibration."""
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    got = restore(stored, [(106.0, 4.0)])
    assert got == {}


def test_the_boundary_is_the_full_fwhm():
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    assert restore(stored, [(104.0, 4.0)]) == {0: 121.783}
    assert restore(stored, [(104.001, 4.0)]) == {}


def test_two_rows_never_share_one_assignment():
    """The nearer row takes it; the other opens blank."""
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    got = restore(stored, [(101.5, 8.0), (100.2, 8.0)])
    assert got == {1: 121.783}


def test_each_row_takes_its_own_nearest_assignment():
    stored = EnergyAssignments(
        None, ((100.0, 121.783), (200.0, 344.276))
    )
    got = restore(stored, [(100.1, 4.0), (200.1, 4.0)])
    assert got == {0: 121.783, 1: 344.276}


def test_no_stored_assignments_restores_nothing():
    assert restore(None, [(100.0, 4.0)]) == {}
    assert restore(EnergyAssignments(None, ()), [(100.0, 4.0)]) == {}


def test_an_unusable_width_falls_back_to_a_fixed_tolerance():
    """A peak whose FWHM the fit could not determine still deserves a
    chance to match, but a narrow one."""
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    assert restore(stored, [(100.2, float("nan"))]) == {0: 121.783}
    assert restore(stored, [(103.0, float("nan"))]) == {}


def test_the_spectrum_starts_with_no_assignments():
    import numpy as np

    from spectrum import LoadedSpectrum

    spectrum = LoadedSpectrum("x.txt", np.zeros(10), "#fff")
    assert spectrum.energy_assignments is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_energy_assignments.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'energy_assignments'`

- [ ] **Step 3: Write the implementation**

Create `energy_assignments.py`:

```python
"""What the user told the calibration dialog, kept so a refit does not
throw it away.

Session-lived and per spectrum: this rides on the in-memory
LoadedSpectrum, is not written to disk, and does not travel with Save
Fits. The user clears it explicitly or closes the application.
"""

import math
from dataclasses import dataclass

#: Tolerance for a peak whose FWHM the fit could not determine. Narrow
#: on purpose -- with no width to reason about, only a centroid that
#: barely moved should be treated as the same peak.
_FALLBACK_TOLERANCE = 1.0


@dataclass(frozen=True)
class EnergyAssignments:
    """`pairs` is ((channel, energy), ...) as assigned at the time."""
    source_path: str
    pairs: tuple


def _tolerance(fwhm):
    """How far a centroid may move and still be the same peak.

    Its own FWHM: a peak that shifted by more than its width between
    fits is a different peak, and restoring the old energy onto it would
    produce a calibration that looks entirely reasonable and is wrong.
    """
    try:
        width = float(fwhm)
    except (TypeError, ValueError):
        return _FALLBACK_TOLERANCE
    if not math.isfinite(width) or width <= 0.0:
        return _FALLBACK_TOLERANCE
    return width


def restore(assignments, peaks):
    """{row index: energy} for `peaks`, a list of (channel, fwhm).

    Each stored assignment goes to at most one row -- the nearest row
    within tolerance -- so two peaks can never claim the same energy.
    """
    if assignments is None or not assignments.pairs:
        return {}

    # Candidates as (distance, row, energy), then taken in order of
    # increasing distance so the nearest row wins each assignment.
    candidates = []
    for row, (channel, fwhm) in enumerate(peaks):
        tolerance = _tolerance(fwhm)
        for index, (stored_channel, energy) in enumerate(assignments.pairs):
            distance = abs(channel - stored_channel)
            if distance <= tolerance:
                candidates.append((distance, row, index, energy))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    taken_rows, taken_pairs, out = set(), set(), {}
    for _distance, row, index, energy in candidates:
        if row in taken_rows or index in taken_pairs:
            continue
        taken_rows.add(row)
        taken_pairs.add(index)
        out[row] = energy
    return out
```

Modify `spectrum.py`, in `LoadedSpectrum.__init__`, immediately after the
`self.variance = variance` line:

```python
        # What the calibration dialog was last told for this spectrum --
        # an EnergyAssignments, or None. Kept so refitting does not
        # discard the user's identifications; see energy_assignments.py.
        # Session-lived: never written to disk, never saved with fits.
        self.energy_assignments = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_energy_assignments.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
git add energy_assignments.py spectrum.py tests/test_energy_assignments.py
git commit -m "feat: remember energy assignments per spectrum"
```

---

### Task 5: Fit Results panel — new columns and compact notation

**Files:**
- Modify: `fit_mode.py` — `build_results_panel` (~line 953), `_unit_switched_value` (~line 291), `update_results_list` (~line 1199)
- Test: `tests/test_fit_results_layout.py`

**Interfaces:**
- Consumes: `value_format.compact` from Task 1.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Write the failing tests**

```python
"""The rearranged Fit Results panel: # / Position / Volume / FWHM / chi^2,
with compact uncertainty notation and no fit-region column."""

import numpy as np
import pytest

from calibration import Calibration
from main_window import MainWindow
from peak_fit import fit_peaks


def _window_with_one_fit(tmp_path):
    x = np.arange(600, dtype=float)
    clean = np.full(600, 30.0) + 900.0 * np.exp(-((x - 150.0) ** 2) / (2 * 4.0 ** 2))
    y = np.random.default_rng(3).poisson(clean).astype(float)
    path = tmp_path / "s.txt"
    path.write_text("\n".join(str(int(v)) for v in y), encoding="utf-8")

    window = MainWindow()
    window._load_files([str(path)])
    active = window.spectra[0]
    axis = np.arange(len(active.data), dtype=float)
    active.fits.append(
        fit_peaks(axis, active.data, (60, 100), (200, 240), (120, 180), [150.0])
    )
    window.fit_controller.update_results_list()
    return window


def test_columns_are_in_the_new_order(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    table = window.fit_controller.results_table
    headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert headers == ["#", "Position", "Volume", "FWHM", "chi^2"]


def test_the_index_cell_holds_no_fit_region(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    text = window.fit_controller.results_table.item(0, 0).text()
    assert text.strip() == "1"
    assert "[" not in text


def test_the_fit_region_is_still_in_the_tooltip(qapp, tmp_path):
    """Dropped from the column, not lost."""
    window = _window_with_one_fit(tmp_path)
    tooltip = window.fit_controller.results_table.item(0, 0).toolTip()
    assert "region" in tooltip.lower()
    assert "120" in tooltip and "180" in tooltip


def test_position_uses_compact_notation(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    text = window.fit_controller.results_table.item(0, 1).text()
    assert "±" not in text
    assert text.endswith(")")


def test_fwhm_uses_compact_notation(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    text = window.fit_controller.results_table.item(0, 3).text()
    assert "±" not in text


def test_headers_switch_to_kev_when_calibrated(qapp, tmp_path):
    window = _window_with_one_fit(tmp_path)
    window._apply_calibration_change(Calibration(kind="linear", a=0.0, b=2.0), True)
    window.fit_controller.update_results_list()
    table = window.fit_controller.results_table
    headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert headers == ["#", "Position (keV)", "Volume", "FWHM (keV)", "chi^2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_fit_results_layout.py -q`
Expected: FAIL — headers are `["Fit", "Position", "FWHM", "Volume", "chi^2"]`

- [ ] **Step 3: Make the changes**

In `fit_mode.py`, add the import near the other local imports:

```python
from value_format import compact
```

Replace `_unit_switched_value`'s two return statements so it renders
compactly while keeping its unit conversion unchanged:

```python
    converted = _to_energy(main_window, channel_value, channel_err, is_width, reference_position)
    if converted is None:
        return compact(channel_value, channel_err)
    value, err = converted
    return compact(value, err)
```

In `build_results_panel`, change the header list:

```python
        self.results_table.setHorizontalHeaderLabels(
            ["#", "Position", "Volume", "FWHM", "chi^2"]
        )
```

In `update_results_list`, change the header call:

```python
        self.results_table.setHorizontalHeaderLabels([
            "#",
            "Position (keV)" if calibrated else "Position",
            "Volume",
            "FWHM (keV)" if calibrated else "FWHM",
            "chi^2",
        ])
```

In the same function, both `fit_label` assignments become the index
alone, and the region moves into the tooltip. For the `IntegrationResult`
branch:

```python
                fit_label = f"{fit_index + 1}"
                tooltip = "\n".join([
                    f"fit region: [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]",
                    _integration_tooltip(self.main_window, result),
                ])
```

and its `values` list is reordered so Volume precedes FWHM:

```python
                values = [
                    fit_label,
                    _unit_switched_value(
                        self.main_window, result.net_centroid, result.net_centroid_err, is_width=False
                    ),
                    compact(result.net_area, result.net_area_err),
                    _unit_switched_value(
                        self.main_window, result.net_fwhm, result.net_fwhm_err,
                        is_width=True, reference_position=result.net_centroid,
                    ),
                    "—",  # no chi^2 concept for a direct-sum Integration result
                ]
```

For the fit branch, `fit_label` becomes `f"{fit_index + 1}"` and the
region is prepended to `shared_tooltip_lines`:

```python
            fit_label = f"{fit_index + 1}"
            shared_tooltip_lines = [
                f"fit region: [{result.fit_region[0]:.1f}, {result.fit_region[1]:.1f}]",
                f"region full (no bg subtracted): {result.gross_area:.1f} ± {_err_text(result.gross_area_err, '.1f')}",
```

Then reorder that branch's `values` list the same way, with
`compact(peak.area, peak.area_err)` in the Volume slot, immediately after
the Position entry and before the FWHM entry.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_fit_results_layout.py tests/test_fit_mode_ui.py -q`
Expected: PASS. Any existing test asserting the old `"1 [120.0, 180.0]"`
label or a `±` in a results cell must be updated to the new form, not
deleted.

- [ ] **Step 5: Commit**

```bash
git add fit_mode.py tests/test_fit_results_layout.py
git commit -m "feat: rearrange fit results, compact uncertainties, region to tooltip"
```

---

### Task 6: Calibration plot window with Finish

**Files:**
- Create: `calibration_plot_dialog.py`
- Test: `tests/test_calibration_plot.py`

**Interfaces:**
- Consumes: `calibration_quality.reduced_chi_squared` (Task 3), `caleneff_export.build_rows` / `write_caleneff` / `ExportError` (Task 2), `value_format.compact` (Task 1).
- Produces: `CalibrationPlotDialog(parent, calibration, points, source_lines, max_channel, default_path)` where `points` is a list of `(channel, channel_err, area, area_err, energy)`. Public: `.export_to(path) -> str` returning a human-readable summary, `.finish_button`, `.summary_label`.

- [ ] **Step 1: Write the failing tests**

```python
"""The calibration plot window: coefficients, reduced chi-squared, and
the CalEnEff export behind Finish."""

import numpy as np
import pytest

from calibration import Calibration
from calibration_plot_dialog import CalibrationPlotDialog
from sou_io import SourceLine


def _lines():
    return [
        SourceLine(121.783, 0.002, 5000.0, 50.0),
        SourceLine(344.276, 0.004, 10000.0, 80.0),
    ]


def _points():
    return [
        (100.0, 0.05, 9000.0, 95.0, 121.783),
        (300.0, 0.08, 4000.0, 63.0, 344.276),
        (500.0, 0.06, 2000.0, 45.0, 566.0),
    ]


def _dialog(qapp, calibration=None, points=None, lines=None):
    return CalibrationPlotDialog(
        None,
        calibration or Calibration(kind="linear", a=10.0, b=1.1),
        points if points is not None else _points(),
        lines if lines is not None else _lines(),
        max_channel=1024,
        default_path="out_En_Area.txt",
    )


def test_coefficients_and_their_uncertainties_are_shown(qapp):
    cal = Calibration(kind="linear", a=10.0, b=1.1,
                      coefficient_errors=(0.02, 0.0013))
    text = _dialog(qapp, calibration=cal).summary_label.text()
    assert "10.00(20)" in text or "10.0000(200)" in text or "10.000(20)" in text
    assert "b" in text


def test_reduced_chi_squared_is_shown_for_a_weighted_fit(qapp):
    text = _dialog(qapp).summary_label.text()
    assert "chi" in text.lower()
    assert "undefined" not in text.lower()


def test_an_unweighted_fit_says_so_instead_of_printing_a_number(qapp):
    points = [(100.0, 0.0, 9000.0, 95.0, 121.783),
              (300.0, 0.08, 4000.0, 63.0, 344.276),
              (500.0, 0.06, 2000.0, 45.0, 566.0)]
    text = _dialog(qapp, points=points).summary_label.text()
    assert "unweighted" in text.lower()


def test_the_minimum_number_of_points_reports_no_degrees_of_freedom(qapp):
    points = [(100.0, 0.05, 9000.0, 95.0, 121.783),
              (300.0, 0.08, 4000.0, 63.0, 344.276)]
    text = _dialog(qapp, points=points).summary_label.text()
    assert "degrees of freedom" in text.lower()


def test_the_plot_draws_the_points_and_the_curve(qapp):
    dialog = _dialog(qapp)
    axes = dialog.axes
    # One errorbar container for the points, at least one line for the fit.
    assert len(axes.containers) >= 1
    assert len(axes.lines) >= 1


def test_export_writes_only_the_rows_with_intensities(qapp, tmp_path):
    """The 566.0 keV point was typed by hand and has no source line."""
    dialog = _dialog(qapp)
    path = tmp_path / "out.txt"
    summary = dialog.export_to(str(path))

    data = np.loadtxt(str(path), ndmin=2)
    assert data.shape == (2, 7)
    assert "1" in summary  # reports the skipped one
    assert "skip" in summary.lower()


def test_export_normalises_intensity_to_the_strongest_line(qapp, tmp_path):
    dialog = _dialog(qapp)
    path = tmp_path / "out.txt"
    dialog.export_to(str(path))
    data = np.loadtxt(str(path), ndmin=2)
    assert data[:, 5].max() == pytest.approx(100.0)


def test_export_with_nothing_exportable_raises(qapp, tmp_path):
    from caleneff_export import ExportError

    dialog = _dialog(qapp, lines=[])
    with pytest.raises(ExportError):
        dialog.export_to(str(tmp_path / "out.txt"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_calibration_plot.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'calibration_plot_dialog'`

- [ ] **Step 3: Write the implementation**

Create `calibration_plot_dialog.py`:

```python
"""Shows an energy calibration as a fit, not as a status line.

A calibration's coefficients look equally plausible whether or not one
of its points was misidentified. The residual strip is where that shows
up, which is why this window exists at all rather than a message box
reporting a and b.
"""

import os

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from caleneff_export import ExportError, build_rows, write_caleneff
from calibration_quality import reduced_chi_squared
from value_format import compact

#: Points drawn along the fitted curve. Enough that a quadratic reads as
#: a curve rather than a polyline at any zoom the dialog offers.
_CURVE_SAMPLES = 400


class CalibrationPlotDialog(QDialog):
    """`points` is (channel, channel_err, area, area_err, energy) per
    assigned peak. `source_lines` are the .sou lines they were assigned
    from, used only by the export."""

    def __init__(self, parent, calibration, points, source_lines,
                 max_channel, default_path):
        super().__init__(parent)
        self.setWindowTitle("Energy Calibration")
        self._calibration = calibration
        self._points = list(points)
        self._source_lines = list(source_lines or [])
        self._default_path = default_path

        channels = [p[0] for p in self._points]
        errors = [p[1] for p in self._points]
        energies = [p[4] for p in self._points]

        layout = QVBoxLayout(self)

        figure = Figure(figsize=(6.5, 5.0))
        self.canvas = FigureCanvasQTAgg(figure)
        # Residuals share the x axis and get a third of the height: they
        # are read against the main plot, not on their own.
        self.axes, self.residual_axes = figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        self.axes.errorbar(
            channels, energies, xerr=errors, fmt="o", capsize=3,
            label="assigned peaks",
        )
        upper = max(max_channel, max(channels) if channels else 0)
        grid = np.linspace(0.0, float(upper), _CURVE_SAMPLES)
        self.axes.plot(grid, calibration.apply(grid), "-", label="calibration")
        self.axes.set_ylabel("Energy (keV)")
        self.axes.legend(loc="best")

        residuals = [e - calibration.apply(c) for c, e in zip(channels, energies)]
        self.residual_axes.axhline(0.0, linewidth=0.8)
        self.residual_axes.errorbar(channels, residuals, fmt="o", capsize=3)
        self.residual_axes.set_xlabel("Channel")
        self.residual_axes.set_ylabel("Residual (keV)")
        figure.tight_layout()
        layout.addWidget(self.canvas)

        self.summary_label = QLabel(self._summary_text(channels, energies, errors))
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            self.summary_label.textInteractionFlags()
        )
        layout.addWidget(self.summary_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.finish_button = buttons.addButton(
            "Finish and save for CalEnEff...",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.finish_button.clicked.connect(self._on_finish)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _summary_text(self, channels, energies, errors):
        cal = self._calibration
        coefficient_errors = cal.coefficient_errors or ()

        def coefficient(index, value, name):
            error = coefficient_errors[index] if index < len(coefficient_errors) else None
            return f"{name} = {compact(value, error)}"

        parts = [coefficient(0, cal.a, "a"), coefficient(1, cal.b, "b")]
        if cal.kind == "quadratic":
            parts.append(coefficient(2, cal.c, "c"))

        value, reason = reduced_chi_squared(cal, channels, energies, errors)
        parts.append(
            f"reduced chi^2 = {value:.4g}" if reason is None
            else f"reduced chi^2 {reason}"
        )
        parts.append(f"{len(channels)} points")
        return f"{cal.kind} calibration:  " + ",   ".join(parts)

    def export_to(self, path):
        """Write the CalEnEff file. Returns a summary; raises ExportError."""
        assignments = [(c, ce, area, area_err, energy)
                       for c, ce, area, area_err, energy in self._points]
        rows, skipped = build_rows(assignments, self._source_lines)
        write_caleneff(path, rows)
        summary = f"Wrote {len(rows)} rows to {os.path.basename(path)}"
        if skipped:
            summary += (
                f" — skipped {skipped} peak(s) with no source line, so no "
                f"intensity was available for them"
            )
        return summary

    def _on_finish(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save for CalEnEff", self._default_path,
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            summary = self.export_to(path)
        except ExportError as exc:
            QMessageBox.warning(self, "Could not export", str(exc))
            return
        QMessageBox.information(self, "Saved", summary)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_calibration_plot.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add calibration_plot_dialog.py tests/test_calibration_plot.py
git commit -m "feat: calibration plot window with CalEnEff export"
```

---

### Task 7: Wire it together — dialog Clear, restore, and the plot window

**Files:**
- Modify: `energy_assign_dialog.py` (constructor, new Clear button, `_on_accept`)
- Modify: `main_window.py` — `_open_calibrate_from_peaks_dialog` (~line 1070)
- Test: `tests/test_calibration_workflow.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: nothing further.

- [ ] **Step 1: Write the failing tests**

```python
"""End-to-end: assignments survive, Clear empties them, and calibrating
opens the plot window."""

import numpy as np
import pytest

from energy_assign_dialog import EnergyAssignDialog
from energy_assignments import EnergyAssignments
from main_window import MainWindow
from peak_fit import fit_peaks

ENERGY = EnergyAssignDialog.ENERGY_COLUMN


def _window(tmp_path, centres=(150.0, 420.0)):
    x = np.arange(600, dtype=float)
    clean = np.full(600, 30.0)
    for centre in centres:
        clean = clean + 900.0 * np.exp(-((x - centre) ** 2) / (2 * 4.0 ** 2))
    y = np.random.default_rng(5).poisson(clean).astype(float)
    path = tmp_path / "s.txt"
    path.write_text("\n".join(str(int(v)) for v in y), encoding="utf-8")

    window = MainWindow()
    window._load_files([str(path)])
    active = window.spectra[0]
    axis = np.arange(len(active.data), dtype=float)
    for centre in centres:
        active.fits.append(fit_peaks(
            axis, active.data, (centre - 90, centre - 50),
            (centre + 50, centre + 90), (centre - 30, centre + 30), [centre],
        ))
    return window, active


def test_accepting_stores_the_assignments_on_the_spectrum(qapp, tmp_path):
    window, active = _window(tmp_path)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    dialog._on_accept()
    window._store_energy_assignments(active, dialog)

    assert active.energy_assignments is not None
    assert len(active.energy_assignments.pairs) == 2


def test_reopening_restores_what_was_typed(qapp, tmp_path):
    window, active = _window(tmp_path)
    choices = window.fitted_peak_choices()
    active.energy_assignments = EnergyAssignments(
        None, ((choices[0][1], 300.0), (choices[1][1], 840.0))
    )
    dialog = EnergyAssignDialog(
        None, choices, assignments=active.energy_assignments
    )
    assert float(dialog.table.item(0, ENERGY).text()) == pytest.approx(300.0)
    assert float(dialog.table.item(1, ENERGY).text()) == pytest.approx(840.0)


def test_clear_empties_the_table_and_the_record(qapp, tmp_path):
    window, active = _window(tmp_path)
    choices = window.fitted_peak_choices()
    active.energy_assignments = EnergyAssignments(
        "eu152.sou", ((choices[0][1], 300.0),)
    )
    dialog = EnergyAssignDialog(
        None, choices, assignments=active.energy_assignments
    )
    assert dialog.table.item(0, ENERGY).text() != ""

    dialog.clear_button.click()
    assert dialog.table.item(0, ENERGY).text() == ""
    assert dialog.source_lines is None
    assert dialog.cleared is True


def test_clear_does_not_touch_the_active_calibration(qapp, tmp_path):
    from calibration import Calibration

    window, active = _window(tmp_path)
    window._apply_calibration_change(Calibration(kind="linear", a=1.0, b=2.0), True)
    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.clear_button.click()
    assert window._calibration is not None
    assert window._calibration_active is True


def test_calibrating_opens_the_plot_window(qapp, tmp_path, monkeypatch):
    import main_window as module

    window, active = _window(tmp_path)
    opened = {}

    class _Spy:
        def __init__(self, *args, **kwargs):
            opened["args"] = args
        def show(self):
            opened["shown"] = True

    monkeypatch.setattr(module, "CalibrationPlotDialog", _Spy)

    dialog = EnergyAssignDialog(None, window.fitted_peak_choices())
    dialog.table.item(0, ENERGY).setText("300")
    dialog.table.item(1, ENERGY).setText("840")
    dialog._on_accept()
    window._show_calibration_plot(active, dialog)

    assert opened.get("shown") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_calibration_workflow.py -q`
Expected: FAIL — `EnergyAssignDialog.__init__() got an unexpected keyword argument 'assignments'`

- [ ] **Step 3: Make the changes**

In `energy_assign_dialog.py`:

Add imports:

```python
from energy_assignments import EnergyAssignments, restore
```

Extend the constructor signature and store the flag:

```python
    def __init__(self, parent, peaks, quadratic=False, settings=None,
                 assignments=None):
```

After `self._suggested = {}` and before the layout is built, add:

```python
        #: Set by Clear so the caller knows to erase the spectrum's
        #: stored record, not merely to skip writing a new one.
        self.cleared = False
```

After the table's rows are created and `itemChanged` is connected,
restore any stored assignments:

```python
        if assignments is not None:
            self._restore_assignments(assignments)
```

Add the two methods:

```python
    def _restore_assignments(self, assignments):
        """Refill the table from a previous visit.

        Matching is by channel within each peak's own FWHM, so a refit
        that nudged a centroid keeps its energy while a peak that moved
        further than its own width opens blank rather than inheriting
        an identification that belonged to something else.
        """
        peaks = [(peak.channel, peak.fwhm) for peak in self._peaks]
        self._writing = True
        try:
            for row, energy in restore(assignments, peaks).items():
                self.table.item(row, self.ENERGY_COLUMN).setText(repr(energy))
        finally:
            self._writing = False
        if assignments.source_path and os.path.exists(assignments.source_path):
            self.load_source(assignments.source_path)
        self._update_suggest_availability()

    def _on_clear(self):
        """Empty every energy, drop the source, and mark the stored
        record for deletion. Deliberately does NOT touch the active
        calibration: discarding the identifications and un-calibrating
        the spectrum are different actions."""
        self._writing = True
        try:
            for row in range(len(self._peaks)):
                item = self.table.item(row, self.ENERGY_COLUMN)
                item.setText("")
                item.setBackground(QBrush())
                item.setToolTip("")
        finally:
            self._writing = False
        self._suggested.clear()
        self.source_lines = None
        self._source_name = None
        self.source_label.setText("No source loaded")
        self.cleared = True
        self.status.setText("Cleared all assignments.")
        self._update_suggest_availability()
```

`self._peaks` currently holds `(label, channel, channel_err)` tuples and
is unpacked in three places. Growing it to six positional fields would
mean six-value unpackings that are easy to get wrong and impossible to
read, so replace the tuple with a named record and normalise ONCE.

At module level:

```python
@dataclass(frozen=True)
class _PeakRow:
    """One dialog row's peak. A record rather than a widening tuple:
    callers pass anything from (label, channel) to the full six fields,
    and every field beyond the first two is optional, so positional
    unpacking at each use site would be a standing bug."""
    label: str
    channel: float
    channel_err: float = None
    fwhm: float = None
    area: float = 0.0
    area_err: float = 0.0
```

Add `from dataclasses import dataclass` to the imports. Replace the
constructor's list comprehension with:

```python
        # Accepts (label, channel) through the full six fields. Older
        # callers and every existing test pass the short form.
        self._peaks = [_PeakRow(*entry[:6]) for entry in peaks]
```

Then update the three existing unpackings in this file. In
`assignments()`:

```python
        for row, peak in enumerate(self._peaks):
            text = self._energy_text(row).strip()
            if not text:
                continue
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"{peak.label}: {text!r} is not a number") from None
            if not math.isfinite(value):
                raise ValueError(f"{peak.label}: {text!r} is not a finite energy") from None
            energies.append(value)
            channels.append(peak.channel)
            errors.append(peak.channel_err)
```

In the constructor's row-building loop, replace
`for row, (label, channel, channel_err) in enumerate(self._peaks):` with
`for row, peak in enumerate(self._peaks):` and use `peak.label`,
`peak.channel`, `peak.channel_err` in the three `QTableWidgetItem` calls.

In `_on_suggest`, replace
`for row, (_label, channel, _err) in enumerate(self._peaks):` with
`for row, peak in enumerate(self._peaks):` and use `peak.channel`.

Add the Clear button beside Load source, in `source_row`:

```python
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip(
            "Forget every assigned energy and the loaded source for this "
            "spectrum. Does not change the active calibration."
        )
        self.clear_button.clicked.connect(self._on_clear)
        source_row.addWidget(self.clear_button)
```

- [ ] **Step 4: Run the dialog tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_calibration_workflow.py -q -k "clear or restores"`
Expected: the Clear and restore tests PASS. The two tests needing
main_window wiring still FAIL; Task 8 makes them pass.

- [ ] **Step 5: Commit**

```bash
git add energy_assign_dialog.py tests/test_calibration_workflow.py
git commit -m "feat: Clear button and restored assignments in the calibration dialog"
```

---

### Task 8: Wire the main window to the record and the plot

**Files:**
- Modify: `main_window.py` - `fitted_peak_choices`, `_open_calibrate_from_peaks_dialog` (~line 1070)
- Test: `tests/test_calibration_workflow.py` (already written in Task 7)

**Interfaces:**
- Consumes: `EnergyAssignDialog` (Task 7), `CalibrationPlotDialog` (Task 6),
  `EnergyAssignments` (Task 4).
- Produces: nothing further.

- [ ] **Step 1: Confirm the two remaining tests fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_calibration_workflow.py -q -k "stores or opens"`
Expected: FAIL - MainWindow has no `_store_energy_assignments`.

- [ ] **Step 2: Make the changes**

In `main_window.py`, add the import at the top:

```python
from calibration_plot_dialog import CalibrationPlotDialog
```

`fitted_peak_choices` must now also yield each peak's FWHM as a fourth
element so the dialog can match on width. Find it and append
`peak.fwhm` to each tuple it builds.

Replace the tail of `_open_calibrate_from_peaks_dialog` (everything from
`calibration = dialog.result_calibration` onward) with:

```python
        calibration = dialog.result_calibration
        self._apply_calibration_change(calibration, True)
        active = active_spectrum(self.spectra)
        if active is not None:
            self._store_energy_assignments(active, dialog)
            self._show_calibration_plot(active, dialog)
```

and pass the stored record in when constructing the dialog:

```python
        dialog = EnergyAssignDialog(
            self, choices,
            quadratic=(self._calibration is not None
                       and self._calibration.kind == "quadratic"),
            settings=self.settings,
            assignments=(active_spectrum(self.spectra).energy_assignments
                         if active_spectrum(self.spectra) else None),
        )
```

Add the two helpers to `MainWindow`:

```python
    def _store_energy_assignments(self, spectrum, dialog):
        """Remember what the dialog was told, so a refit does not throw
        it away. Clear erases the record rather than leaving the old one
        in place."""
        if dialog.cleared:
            spectrum.energy_assignments = None
            return
        channels, energies, _errors = dialog.assignments()
        spectrum.energy_assignments = EnergyAssignments(
            source_path=dialog.source_path(),
            pairs=tuple(zip(channels, energies)),
        )

    def _show_calibration_plot(self, spectrum, dialog):
        """The coefficients alone look equally plausible whether or not
        a line was misidentified; the plot's residual strip is where
        that shows."""
        points = dialog.export_points()
        stem = os.path.splitext(os.path.basename(spectrum.path.split("::", 1)[0]))[0]
        default_path = os.path.join(
            os.path.dirname(spectrum.path.split("::", 1)[0]),
            f"{stem}_En_Area.txt",
        )
        self._calibration_plot = CalibrationPlotDialog(
            self, dialog.result_calibration, points,
            dialog.source_lines, len(spectrum.data) - 1, default_path,
        )
        self._calibration_plot.show()
```

Add `EnergyAssignments` to `main_window.py`'s imports.

Finally, add to `energy_assign_dialog.py` the two accessors those helpers
call:

```python
    def source_path(self):
        """The loaded .sou path, or None. Stored so reopening the dialog
        can load the same source again."""
        return getattr(self, "_source_path", None)

    def export_points(self):
        """(channel, channel_err, area, area_err, energy) per assigned
        row, for the plot window and the CalEnEff export."""
        points = []
        for row, peak in enumerate(self._peaks):
            text = self._energy_text(row).strip()
            if not text:
                continue
            points.append((
                peak.channel, peak.channel_err or 0.0,
                peak.area, peak.area_err, float(text),
            ))
        return points
```

Set `self._source_path = path` inside `load_source` on success, beside
the existing `self._source_name` assignment, and initialise
`self._source_path = None` in the constructor.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_calibration_workflow.py tests/test_energy_assign.py tests/test_energy_assign_source.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add energy_assign_dialog.py main_window.py tests/test_calibration_workflow.py
git commit -m "feat: persist assignments, Clear button, open the calibration plot"
```

---

### Task 9: Documentation

**Files:**
- Modify: `help_content.py` — HowTo section 11 and the Knowledge Database calibration section
- Test: `tests/test_help_content.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_help_documents_the_calibration_workflow():
    howto = build_howto_html()
    assert "Clear" in howto
    assert "CalEnEff" in howto
    kb = build_knowledge_database_html()
    # The three facts a user cannot infer from the UI.
    assert "within that peak" in kb and "FWHM" in kb
    assert "strongest line" in kb and "100" in kb
    assert "unweighted" in kb


def test_help_documents_the_new_results_columns():
    kb = build_knowledge_database_html()
    assert "352.7217(14)" in kb or "(14)" in kb
    assert "tooltip" in kb.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_help_content.py -q`
Expected: FAIL on the two new tests.

- [ ] **Step 3: Write the documentation**

**Verify every claim against the code before writing it.** This project
has repeatedly found plausible help text to be wrong.

Append to the HowTo's section 11, after the source-file paragraph:

```html
<p><b>Your assignments are remembered.</b> Everything you type stays with
that spectrum for the session, so refitting and reopening the dialog
brings it back rather than making you retype it. A peak whose centroid
moved further than its own width comes back blank, because past that
distance it is a different peak. <b>Clear</b> forgets every assignment
and the loaded source. It deliberately leaves the active calibration
alone: discarding your identifications and un-calibrating the spectrum
are separate actions.</p>
<p><b>The fit opens in its own window</b> showing the assigned points
with their uncertainties, the calibration curve across the whole
spectrum, a residual strip, the coefficients with their errors, and the
reduced chi-squared. <b>Finish and save for CalEnEff...</b> writes a
seven-column file for the efficiency-calibration program: channel and
its error, net area and its error, energy, and relative intensity with
its error.</p>
```

Append to the Knowledge Database's calibration section:

```html
<h3>What the dialog remembers</h3>
<p>Assignments are stored per spectrum for the session. They are not
written to disk and do not travel with Save Fits. When the dialog
reopens, each row takes the stored energy whose channel is nearest,
provided it lies <b>within that peak&rsquo;s own FWHM</b>. A peak that
shifted by more than its width between fits is treated as a different
peak and opens blank. Restoring an energy onto the wrong peak would
produce a calibration whose coefficients and residuals both look
reasonable while being wrong, which is the failure this rule exists to
prevent. No two rows can claim the same stored assignment: the nearer
one takes it.</p>

<h3>Reduced chi-squared, and when there is none</h3>
<p>The plot window divides each point&rsquo;s residual by that
point&rsquo;s energy uncertainty, obtained from its channel uncertainty
through the calibration&rsquo;s local slope dE/dch, squares and sums
them, and divides by n&minus;p. <i>p</i> is 2 for a line and 3 for a
quadratic.</p>
<p>It is reported as <b>undefined</b> in three cases, each naming its
reason rather than printing a number. When any centroid uncertainty is
unusable the calibration was fitted <b>unweighted</b>, so the weights a
chi-squared would divide by are arbitrary and the value would look like
a goodness of fit while meaning nothing. When there are exactly as many
points as parameters there are <b>no degrees of freedom</b> and the fit
passes through them exactly. And a point sitting on a quadratic&rsquo;s
turning point has dE/dch of zero, so its channel uncertainty maps to no
energy uncertainty at all.</p>

<h3>The CalEnEff export</h3>
<p><b>Finish and save for CalEnEff...</b> writes seven
whitespace-separated columns with no header, all uncertainties
<b>absolute</b>: channel, its error, net area, its error, energy in keV,
relative intensity in percent, and its error.</p>
<p>The first four come from the fit and the last three from the source
file. <b>N is the net area</b>, background subtracted, because CalEnEff
computes efficiency as N divided by I: a gross area would fold the
background into the efficiency curve.</p>
<p>Source files carry intensities on no common scale, so the strongest
line in the loaded source is normalised to <b>100</b> and every other
line scaled by the same factor. Because efficiency is a ratio to I, a
constant factor rescales the whole curve without changing its shape, and
the relative uncertainties are unaffected.</p>
<p>A peak whose energy you typed by hand has no intensity, so it cannot
be exported and is <b>skipped</b>, with the count reported. Writing a
zero instead would be worse: CalEnEff rejects rows whose uncertainties
are all zero, because the efficiency uncertainty would come out zero.</p>

<h3>Reading the results columns</h3>
<p>The Fit Results panel shows <b>#</b>, <b>Position</b>, <b>Volume</b>,
<b>FWHM</b> and <b>chi^2</b>. Position and FWHM use the compact notation
of nuclear data tables: <code>352.7217(14)</code> means 352.7217 with an
uncertainty of 0.0014, the parenthesised digits being the uncertainty in
the last digits shown. A value with no usable uncertainty, such as a
position held fixed, is printed alone with no parentheses.</p>
<p>The fit region is no longer a column. It has moved into the
row&rsquo;s tooltip, along with the gross and net areas, so hovering
still shows it.</p>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_help_content.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add help_content.py tests/test_help_content.py
git commit -m "docs: document the calibration workflow and results layout"
```

---

### Task 10: Full-suite verification

- [ ] **Step 1: Run the whole suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`

Expected: every test passes. The count must be **1227 plus the new
tests** (roughly 1275). A count *lower* than 1227 means tests were
deleted or silently skipped, which is a failure even if the run is green.

- [ ] **Step 2: Fix anything the rearrangement broke**

Existing tests asserting the old `"1 [120.0, 180.0]"` results label, the
old column order, or a `±` in a results cell must be updated to the new
form. Update them to assert the new behaviour; do not delete them.

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "test: update fit-results assertions for the new layout"
```
