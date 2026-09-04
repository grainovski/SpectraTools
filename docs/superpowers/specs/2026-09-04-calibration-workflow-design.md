# Calibration workflow: persistence, plot window, CalEnEff export, results layout

Design for the v5.0.0 calibration work. Four changes, approved 2026-09-04.

They share one thread: the energy calibration stops being a one-shot dialog
that reports a status line, and becomes a workflow you can return to,
inspect, and hand to CalEnEff.

## Background

`Operations > Calibrate from Fitted Peaks...` currently opens
`EnergyAssignDialog`, takes typed or `.sou`-suggested energies, fits with
`calibration.from_points`, applies the result, and reports one status line.
Everything the user entered is discarded when the dialog closes.

Terms used below: **assignment** is one (fitted peak, known energy) pair.
**Source** is a loaded `.sou` file. **Net area** is `peak.area`, the
background-subtracted counts, as opposed to `peak.full_area`.

---

## 1. Assignments persist per spectrum, with a Clear button

### What

`LoadedSpectrum` gains one attribute:

```python
self.energy_assignments = None   # or EnergyAssignments(...)
```

`EnergyAssignments` is a frozen dataclass holding:

- `source_path`: the `.sou` file last loaded, or `None`
- `pairs`: list of `(channel, energy)`, the channel being the fitted
  centroid the energy was attached to at the time

It is written when the dialog is accepted, and read when the dialog opens.
Session-lived: it lives on the in-memory spectrum object, is not written to
disk, and does not travel with Save Fits.

### Matching after a refit

The point of persisting is to survive refitting, and a refit moves
centroids. Each dialog row takes the stored energy whose channel is
**nearest, and within that peak's own FWHM**.

A peak that has moved by more than its own width is a different peak, and
its row opens blank. This is deliberately conservative: a wrong energy
silently restored would produce a plausible-looking calibration that is
wrong, which is the same failure mode the suggestion rule already guards
against.

When two rows compete for one stored assignment, the nearer row takes it
and the other opens blank. No row ever receives an assignment already
claimed by another row.

### Clear

A **Clear** button in the dialog empties every energy cell, drops the
loaded source, and erases the spectrum's stored assignments. It is the
only way to discard them short of closing the application.

Clear does not touch the active calibration. Removing the assignments and
silently un-calibrating the spectrum are different actions, and the user
asked for the first.

---

## 2. Calibration plot window

### What

On a successful calibration, `CalibrationPlotDialog` opens instead of the
present one-line status message. It is non-modal so the spectrum stays
usable behind it.

Contents:

- **Main axes**: assigned points as (channel, energy) with horizontal
  error bars from each centroid uncertainty, and the fitted curve drawn
  across the spectrum's full channel range, so extrapolation beyond the
  assigned points is visible rather than implied.
- **Residual strip** below, sharing the x axis: `E_assigned − f(channel)`
  in keV, with a zero line. This is where a misassigned line shows itself
  while `a` and `b` still look reasonable.
- **Coefficient panel**: `a`, `b`, and for a quadratic `c`, each with its
  uncertainty from `Calibration.coefficient_errors`; the point count; and
  reduced chi-squared.
- **Finish** button (section 3) and **Close**.

### Reduced chi-squared, and when it does not exist

```
chi2 = sum( ( (E_i - f(ch_i)) / (sigma_ch_i * dE/dch|ch_i) )**2 )
red  = chi2 / (n - p)
```

`p` is 2 for a linear calibration and 3 for a quadratic. `dE/dch` comes
from `Calibration.derivative`, which converts a channel uncertainty into
an energy uncertainty at that point.

Three cases produce no number, and each is reported as text rather than as
a value:

| Case | Shown |
|---|---|
| Any centroid uncertainty unusable, so `from_points` fitted unweighted | `undefined (unweighted fit)` |
| `n == p`, an interpolation through the minimum number of points | `undefined (no degrees of freedom)` |
| Non-finite result | `undefined` |

`from_points` already falls back to an unweighted fit when any error is
zero, negative, or non-finite. In that case the weights are arbitrary and
a printed chi-squared would look like a goodness of fit while meaning
nothing. Reporting the reason is the point.

`Calibration` carries no chi-squared today, so this is computed in the
plot module from the calibration, the channels, the energies and the
channel errors. It does not change `calibration.py`.

---

## 3. Finish, writing the CalEnEff file

### Format

Seven whitespace-separated columns, no header, matching
`C:\Users\RIG\Documents\Claude\efficieny\226Ra_En_Area.txt`. CalEnEff's own
`ra226_gui.py:18` documents them as **absolute uncertainties**:

```
ch   delta_ch   N   delta_N   E[keV]   I[%]   delta_I[%]
```

| Column | Source |
|---|---|
| `ch`, `delta_ch` | `peak.position`, `peak.position_err` |
| `N`, `delta_N` | `peak.area`, `peak.area_err` (net, not gross) |
| `E[keV]` | the assigned energy |
| `I[%]`, `delta_I[%]` | the `.sou` line's intensity and its error, each scaled by `100 / max_intensity_in_that_file` |

Net area is required because CalEnEff computes `eff = N / I_pct`. Gross
area would fold the background into the efficiency.

### The intensity scale

`.sou` intensities have no common convention. Measured across the nine
files in the working directory, the maxima are 1000, 6800, 9934, 10000,
10000, 13100, 13620, 35700 and 100000.

The file's strongest line is therefore normalised to 100 and every other
line scaled by the same factor, matching the sample file where 609.312 keV
carries `I = 100`. Since `eff = N / I_pct`, a constant factor rescales the
whole efficiency curve without changing its shape, and
`deff = eff * sqrt((dN/N)**2 + (dI/I)**2)` is invariant under it, so the
relative uncertainties are unaffected by the choice.

`delta_I` is scaled by the same factor, keeping it absolute in the new
units as the format requires.

### Peaks that cannot be exported

An energy typed by hand has no intensity, so such a row cannot produce
`I[%]`. Those rows are **skipped**, and the dialog reports how many and
why. Writing a zero would be worse than skipping: CalEnEff explicitly
rejects rows where `dN` and `dI` are both zero, because the efficiency
uncertainty would be zero.

If no row survives, the file is not written and the reason is shown.

### Where the file goes

A save dialog defaulting to the spectrum's own folder and a name derived
from the spectrum's stem plus `_En_Area.txt`, so it lands beside the data
it came from and reads like the CalEnEff sample.

---

## 4. Results panel layout

### Columns

From `["Fit", "Position", "FWHM", "Volume", "chi^2"]` to:

```
#    Position    Volume    FWHM    chi^2
```

- `#` is the fit index alone. The fit region leaves the cell.
- `Volume` keeps its label, per the decision, and moves ahead of `FWHM`.
- `Position` and `FWHM` switch to keV headers when a calibration is
  active, exactly as now.

The fit region is still available: it stays in the row tooltip, so nothing
is lost, it simply stops consuming a column.

### Compact uncertainty notation

`Position` and `FWHM` render as `352.7217(14)`, where the parenthesised
digits are the uncertainty in the final digits shown.

A new module `value_format.py` holds one function so the results panel and
the plot window cannot drift apart:

```python
def compact(value, error):
    """'352.7217(14)'. Falls back to a plain number when `error` is not a
    usable uncertainty."""
```

Rules:

- The uncertainty is quoted to **two significant digits**.
- The value is rendered to the same decimal place as that uncertainty.
- An error that is zero, negative, or non-finite yields the value alone
  with no parentheses, matching how the panel already declines to print
  `0.000` for a position that was held fixed.
- An error larger than the value is still rendered; it is unusual but
  real, and hiding it would be worse than showing it.
- A value that is not finite yields an em dash.

`FWHM` takes the same treatment. The approved sketch listed it bare, and
this is the one deliberate deviation: consistency across the row costs no
width and discards no information. It is one call site if it should change.

`_unit_switched_value` keeps its unit conversion and delegates rendering to
`compact`.

---

## Files

| File | Change |
|---|---|
| `value_format.py` | new, the compact formatter |
| `calibration_plot_dialog.py` | new, plot window and Finish |
| `caleneff_export.py` | new, the seven-column writer and intensity scaling |
| `spectrum.py` | `LoadedSpectrum.energy_assignments`, the assignment record |
| `energy_assign_dialog.py` | Clear button, restore on open, save on accept |
| `fit_mode.py` | results panel columns, compact rendering, region to tooltip |
| `main_window.py` | open the plot window instead of the status line |
| `help_content.py` | document all four changes on both pages |

## Testing

- **`value_format`**: two-significant-digit rounding, a carry that changes
  the decimal place, zero and non-finite errors, error exceeding value,
  non-finite value.
- **Persistence**: assignments restored after reopening; restored after a
  refit that moves a centroid slightly; **not** restored when the centroid
  moves further than its FWHM; two rows never share one assignment; Clear
  empties the record and leaves the calibration alone.
- **Plot window**: coefficients and their errors displayed; each of the
  three undefined chi-squared cases reports its reason rather than a
  number; a known-good calibration reproduces a chi-squared computed
  independently in the test.
- **Export**: a written file read back with `numpy.loadtxt` has seven
  columns and the expected values; intensity normalised so the strongest
  line reads 100; hand-typed rows skipped and counted; nothing written
  when no row survives.
- **Results panel**: five columns in the new order; the index cell holds
  no region; the region is still in the tooltip; compact rendering in
  both channel and keV modes.

Each measurement-style test carries a control that must fail, per the
project's standing practice.
