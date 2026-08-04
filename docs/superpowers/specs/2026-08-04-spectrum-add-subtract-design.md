# Spectrum Add/Subtract with Scaling Factor — Design Spec

Date: 2026-08-04

## Purpose

First feature of version 2.1.0. User request: two new Operations-menu commands that combine two already-loaded, user-selected spectra of the same length (calibrated or uncalibrated) — one added to or subtracted from the other, with the second spectrum scaled by a user-provided factor first. Applies to both the Windows and Linux distributions; per the user, Windows is the primary focus for verification depth, Linux gets a proportionate (not exhaustive) check.

## Researched Before Designing: How TV Does This

This project's foundation is behavioral parity with TV (`tv-1.9.13/`), so TV's own `add`/`subtract`/`multiply` commands were read directly from source before designing anything here (not assumed from the user manual's prose, which turned out to have at least one real discrepancy from the code — see below).

**Headline finding: TV has no single command matching this request.** TV's `add` and `subtract` (`src/VsSpectra.c:195,942`, `Spc_Add`/`Spc_Subtract`) take **no factor at all** — they read only a destination index (`Idx_ReadOne`, no `UF_RealRead`) and are pure `dest[i] += src[i]` / `dest[i] -= src[i]`. The only TV command that takes a numeric factor is `multiply` (`Spc_Multiply`), which scales **one** spectrum in place and never touches a second spectrum. To reproduce "A + factor·B" in TV requires manually chaining three commands: `copy B to scratch`, `multiply scratch by factor`, `add A += scratch`. So this feature is a genuine, convenient enhancement beyond what TV exposes as a single command — there's no TV command structure to directly copy, only TV's underlying *arithmetic* to match for parity where it applies:

- **Core arithmetic** (`SpcBufAdd`/`SpcBufSub`, `lib/tv/vsSpectra.c:417-441,525-549`): unrounded, unclamped, double-precision `a[i] += b[i]` / `a[i] -= b[i]`. No special-casing of negative results.
- **Same-length requirement is a hard error** (`SpcBufNum(t) != SpcBufNum(f)` → `SPCDRES`, `"different resolutions"`, `lib/tv/vsSpectra.c:422-423`) — no truncation/resampling fallback, matching exactly what the user asked for ("of same length").
- **Calibration**: TV only remaps channels by physical position when *both* spectra being combined carry a real position calibration; with the default/uncalibrated state it degenerates to plain index-for-index arithmetic (`lib/tv/vsCal.c:970-979,1047-1082`). Not directly relevant here anyway — SpectraTools has a single calibration object shared by every loaded spectrum (`main_window.py`'s `_calibration`), not a per-spectrum one like TV, so there's no cross-spectrum calibration mismatch that could ever arise in the first place.
- One doc/code discrepancy found and **not** carried forward: the shipped manual's `ls-file` section claims calibration is applied before `ls-file add/subtract`, but that code path (`Spc_MAdd`/`Spc_MSubtract` → `Spc_FileExec` → `SpcReadAdd`/`SpcReadSub`) never receives a `Cal_t` at all in the traced code — flagged as an observed inconsistency in TV itself, not something to replicate.
- TV's `multiply` factor has no implicit default (required argument) and short-circuits at `1.0` (no-op) and `0.0` (clears the buffer) — informed the factor-validation choice below, though this feature's UI default (see below) differs from TV's own "no default" on purpose.

## Resolved Decisions (through discussion with the user)

- **Result target**: creates a **new**, third spectrum. Neither of the two source spectra is modified. (TV itself would overwrite the destination buffer in place, and this codebase's own Multiply/Rebin already overwrite the active spectrum — but the user explicitly chose the safer "always create new" option for this operation over matching either of those precedents.)
- **Negative values**: **not clamped** — `result[i]` can go negative and is kept as-is, matching TV's own `SpcBufSub` exactly.
- **Factor field default**: the dialog's factor field starts pre-filled with `"1"` (a straight, unscaled combine is the common case), unlike the existing `FactorDialog` (used by Multiply/Rebin), which starts blank — this is a property of the *new* dialog only, `FactorDialog` itself is untouched.

## Architecture

Follows the exact pattern already established three times in this codebase (Multiply/Rebin/Normalize): a pure-logic function in `spectrum_operations.py`, a thin Qt dialog, and wiring in `main_window.py`. No new architectural pattern is introduced.

**`spectrum_operations.py`** gains two new functions (not one function with a mode string — matching this file's existing convention of small, single-purpose functions rather than a parameterized do-everything function, and directly mirroring both TV's separate `add`/`subtract` commands and this feature's two separate menu items):

```python
def add(data_a, data_b, factor):
    """result[i] = A[i] + factor * B[i], rounded to nearest integer
    (matches multiply()'s rounding convention). A and B are assumed
    already validated as equal length by the caller. Unlike multiply()/
    rebin(), negative results are NOT clamped -- matches TV's own
    SpcAdd, which never clamps (tv-1.9.13/lib/tv/vsSpectra.c)."""
    return np.round(data_a + factor * data_b).astype(np.int64)


def subtract(data_a, data_b, factor):
    """result[i] = A[i] - factor * B[i], rounded to nearest integer.
    Same assumptions and TV-parity notes as add() above."""
    return np.round(data_a - factor * data_b).astype(np.int64)
```

**New dialog**, `combine_dialog.py` (own file, mirroring `factor_dialog.py`'s naming — not an extension of `FactorDialog`, since that class's whole point is being a generic single-numeric-field dialog shared verbatim by Multiply/Rebin, and bolting two spectrum-pickers onto it would compromise that reusability):

- Two `QComboBox` dropdowns, labeled "Spectrum A" and "Spectrum B", each populated with every loaded spectrum's `os.path.basename(spectrum.path)` (unfiltered — both dropdowns always show every loaded spectrum, no dynamic filtering of one based on the other, since this codebase doesn't use that pattern anywhere else and it adds real complexity for marginal benefit).
  - Default selection: **Spectrum A** defaults to the currently active spectrum (or the first loaded spectrum if none is active); **Spectrum B** defaults to the first *other* loaded spectrum (skipping A's index), so the two dropdowns don't start pointing at the same spectrum by default.
- One `QLineEdit` factor field, defaulting to `"1"`, validated as a float `> 0` on accept (matching `FactorDialog`'s existing `parse=float` / `> 0` validation used by Multiply by Factor — a negative factor on "Add" would just be a backdoor Subtract, so keeping the factor positive keeps the two operations non-overlapping).
- On accept (mirroring `FactorDialog._on_accept`'s parse → validate → show-warning-and-stay-open pattern exactly): if the two selected spectra have different `len(data)`, show a `QMessageBox` warning naming the actual channel counts (e.g. *"Spectrum A has 4096 channels; Spectrum B has 2048 channels. Add/Subtract requires both to have the same length."*) and do **not** close the dialog — same recoverable-error UX as an invalid factor today. Only a same-length pair with a valid factor closes the dialog and returns a result.

**`main_window.py` wiring**: two new `QAction`s, "Add Spectra..." and "Subtract Spectra...", added to the Operations menu after the existing Multiply/Rebin/Normalize entries. Disabled whenever fewer than 2 spectra are loaded (wired into whichever existing mechanism already toggles Multiply/Rebin/Normalize's enabled state as spectra are loaded/closed — the plan will pin down the exact call site). Each opens the new dialog; on a successful result, builds the new `LoadedSpectrum` (next color in the theme's cycle, `active = True`, empty `fits`) and appends it to `self.spectra`, then replots.

**New spectrum naming** (used as its `.path`, which is also what `os.path.basename()`-based display uses elsewhere in this codebase): `"{basename(A.path)} + {basename(B.path)}"` when factor is exactly `1`, or `"{basename(A.path)} + {factor}x{basename(B.path)}"` otherwise (subtract uses `-` in place of `+`); an ASCII `x` is used rather than `×` specifically so this string stays safe if it's ever offered as a suggested filename by Save Spectrum, not just displayed in the spectrum list.

## Error Handling Summary

- **Fewer than 2 spectra loaded**: the two menu actions are disabled outright — the dialog can't be opened at all.
- **Selected pair has different lengths**: caught at dialog-accept time, `QMessageBox` warning with the actual channel counts, dialog stays open for correction. The pure `add()`/`subtract()` functions never see a mismatched pair — they assume equal length was already validated by the caller, matching this file's existing convention for `multiply()`/`rebin()`.
- **Factor ≤ 0 or unparseable**: same existing `FactorDialog`-style warning-and-stay-open behavior, reused via the same `parse`/`validate` shape.
- **Same spectrum picked for both A and B**: allowed, not blocked — mathematically well-defined (`A×(1+factor)` for Add, `A×(1-factor)` for Subtract), not worth the complexity of special-casing.

## Testing Plan

- **Unit tests** (`tests/test_spectrum_operations.py`, extended; pytest, identical on both platforms): `add()`/`subtract()` — rounding behavior, negative results preserved unclamped, various factors including the boundary `factor == 1` naming case. Dialog-level tests for the new `combine_dialog.py`: length-mismatch warning fires with the correct message and doesn't close the dialog; factor ≤ 0 rejected; factor field starts pre-filled `"1"`; default A/B selection logic.
- **Windows** (primary focus, real manual verification — matching every other feature this project has shipped): build via `build.ps1`, actually run the app, load real spectra of matching and mismatched lengths, exercise both Add and Subtract with multiple factors including the `1` default, confirm the new spectrum appears correctly in the list and plot, is fittable and calibrated normally, and confirm the length-mismatch error surfaces correctly against real files.
- **Linux** (proportionate, not the full multi-distro exercise the packaging work required): rebuild via the existing `build.sh`/`build_deb.sh` and smoke-test on Ubuntu WSL specifically — the one Linux environment already confirmed working end-to-end by the user. This is a pure application-code change with no packaging-script involvement, so re-running the AlmaLinux 8/10 cross-version verification isn't warranted the way it was for the packaging infrastructure itself. A full Linux package re-release (rebuilding and re-archiving `releases/v2.1.0/...`) is a separate, later decision once the feature is confirmed working, following the existing release-freeze process (including the now-standing `CHANGELOG.md` update step).

## Out of Scope

- TV's physical-position calibration remap when combining spectra with *different* calibrations — doesn't apply here, since this app has one calibration shared by every spectrum, not a per-spectrum one.
- Per-channel uncertainty/error propagation (TV tracks a parallel variance buffer and combines it correctly on add/subtract/multiply) — this app's `LoadedSpectrum`/data model has no analogous error-buffer concept today, so there's nothing to propagate; noted here only so it isn't mistaken for an oversight.
- Dynamically filtering the Spectrum B dropdown to only show length-compatible spectra, or any other proactive prevention of a mismatched pick — validation is at confirm time only, matching this codebase's existing `FactorDialog` pattern.
- Blocking the same spectrum being picked for both A and B.
- Any changes to `FactorDialog` itself, or to Multiply/Rebin/Normalize's existing behavior.
