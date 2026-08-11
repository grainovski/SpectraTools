# N42 Reader (Histogram + Energy Calibration) — Design Spec

Date: 2026-08-11

## Purpose

Add read support for ANSI/IEEE N42.42-2011 files (`.n42`), extracting only the raw histogram and, when present, the embedded energy calibration — no other N42 content (instrument settings, ROIs, timestamps, etc.), and no write/save support, per explicit request.

## Researched Before Designing

**Directly from the 5 real sample files the user provided** (`316-2_*.n42`, all from the same CAEN DT5770 digital MCA, structurally identical):
- XML, namespace `http://physics.nist.gov/N42/2011/N42` (confirms this is the 2011/2012 revision of the standard, not the older 2006 tabular format).
- `<EnergyCalibration id="...">` holds `<CoefficientValues>a b c</CoefficientValues>`. Verified the ordering numerically against the file's own `CalibrationPoints` (channel 421.52 → 59.5 keV, channel 4004.3 → 661.7 keV): `E = a + b·channel + c·channel²` with `a=-11.3498272291349, b=0.16808176890571, c=0` reproduces both points exactly. This is the same `(a, b, c)` convention already used by this app's own `Calibration` dataclass (`calibration.py`) — no reordering needed.
- `<RadMeasurement>` contains one `<Spectrum energyCalibrationReference="...">`, which contains `<ChannelData compressionCode="None"> ...16384 whitespace-separated integers... </ChannelData>`.

**From web research** (ANSI N42.42-2011/2012 standard; `compressionCode` is an `enumSpectrumCompressionType` attribute):
- `"None"`: channel values listed directly, as in all 5 samples.
- `"CountedZeroes"`: a real, standard-defined run-length encoding — whenever a `0` appears in the token stream, the *next* token is the total count of consecutive zero channels starting at that first zero (i.e. `0 3` expands to `0 0 0`). Not present in any sample file, but a genuine part of the format, not a vendor-specific quirk — a "generic" reader that only handled `"None"` would silently mishandle real files from other instruments.
- N42 files can, per the standard, contain multiple `RadMeasurement`/`Spectrum` elements (e.g. portal monitors doing continuous timed acquisitions) — none of the 5 samples do this (each has exactly one), and it's out of scope per the user's own request shape ("raw histogram" singular) and their explicit confirmation to reject rather than guess.

**From this app's existing file-loading architecture**, read directly:
- `histogram_io.py`, `spe_io.py`, `spk_io.py` each expose a `load_X(path) -> np.ndarray` function and share one `ParseError` exception (defined in `histogram_io.py`, imported by the other two — confirmed by reading `spe_io.py:5`). None of the three ever returns or extracts a calibration — calibration is entirely separate in this app: one global `Calibration` object (`calibration.py`, `kind: "linear"|"quadratic"`, `a`, `b`, `c`) shared across every loaded spectrum, stored on `MainWindow` as `self._calibration`/`self._calibration_active`, normally set via the `Ctrl+L` dialog. N42 is the first format this app will support that carries its own embedded calibration.
- `MainWindow._apply_calibration_change(self, new_calibration, new_active)` (`main_window.py:465`) already does exactly the work needed to apply a calibration and keep everything in sync — replots preserving the current view region, and updates `self.calibration_toggle_action` (toolbar) and `self.calibration_active_menu_action` (Operations menu) enabled/checked state together. It's already shared by both the Calibration dialog's OK handler and the toolbar's quick-toggle action; reusing it for N42's auto-apply means no new UI-sync code is needed.
- `MainWindow._try_load_spectrum(path)` (`main_window.py:741`) dispatches by extension to a loader, calls it, wraps `ParseError`/`OSError` into a user-facing message, and constructs a `LoadedSpectrum`. `LoadedSpectrum` (`spectrum.py:35`) has no calibration field — calibration truly never flows through per-spectrum state anywhere in this app.
- `.spe`/`.spk` (both structured formats with an explicit, unambiguous data boundary — a header-declared channel count for `.spe`, an internal format marker for `.spk`) do **not** pad or round their channel count. Only `.txt` (`load_histogram`, which has no such explicit boundary — just a bare list of numbers) pads to the next `4096 × 2^n` bucket. N42's `ChannelData` has an unambiguous XML element boundary, closer in spirit to `.spe`/`.spk` than to bare `.txt` — so the reader should use the exact extracted channel count, no bucket-padding.
- `tests/fixtures/` already contains at least one **real, committed sample file** used directly in a test (`tests/test_spk_io.py`'s `test_parses_real_demo_spk()` loads `tests/fixtures/demo.spk` for an end-to-end sanity check with hand-verified values), alongside synthetic byte sequences built in-test for exercising specific parsing branches (`_lc_header()` helper, etc.). This is a more useful convention than "synthetic fixtures only" — worth following for N42 too.

## Resolved Decisions (confirmed with the user)

- **Calibration auto-apply, but non-destructive**: when an N42 file's calibration is successfully extracted, it's applied and activated automatically **only if no calibration is currently active** (`self._calibration_active is False`). If a calibration is already active, the N42 file's own embedded calibration is simply not applied — the spectrum still loads normally.
- **Multiple spectra in one file → reject, don't guess**: if a file contains anything other than exactly one `Spectrum` element (zero or more than one), `load_n42()` raises `ParseError` with a message naming the count found, matching how every other loader in this app already refuses to guess at ambiguous input rather than silently picking one interpretation.

## Architecture

New file `n42_io.py` (read-only — no `save_n42`), plus small integration changes in `main_window.py`.

### `n42_io.py`

```python
import xml.etree.ElementTree as ET

import numpy as np

from calibration import Calibration
from histogram_io import ParseError

_NS = "{http://physics.nist.gov/N42/2011/N42}"


def _decode_counted_zeroes(tokens):
    """Expands N42's CountedZeroes run-length encoding: a "0" token is
    immediately followed by the total number of consecutive zero
    channels starting at that first zero (e.g. "0 3" -> three zeros)."""
    values = []
    i = 0
    while i < len(tokens):
        value = tokens[i]
        if value == 0:
            if i + 1 >= len(tokens):
                raise ParseError("N42 CountedZeroes data ends mid run-length pair")
            run_length = tokens[i + 1]
            values.extend([0] * run_length)
            i += 2
        else:
            values.append(value)
            i += 1
    return values


def _parse_channel_data(spectrum_el, path):
    channel_data_el = spectrum_el.find(f"{_NS}ChannelData")
    if channel_data_el is None or channel_data_el.text is None:
        raise ParseError(f"N42 file has no ChannelData: {path}")

    try:
        tokens = [int(tok) for tok in channel_data_el.text.split()]
    except ValueError as exc:
        raise ParseError(f"N42 ChannelData contains non-integer values: {path}") from exc

    compression = channel_data_el.get("compressionCode", "None")
    if compression == "None":
        values = tokens
    elif compression == "CountedZeroes":
        values = _decode_counted_zeroes(tokens)
    else:
        raise ParseError(f"Unsupported N42 ChannelData compressionCode {compression!r}: {path}")

    if not values:
        raise ParseError(f"N42 file's ChannelData is empty: {path}")
    return np.array(values, dtype=np.int64)


def _parse_calibration(root, spectrum_el):
    ref = spectrum_el.get("energyCalibrationReference")
    if ref is None:
        return None
    cal_el = root.find(f".//{_NS}EnergyCalibration[@id='{ref}']")
    if cal_el is None:
        return None
    values_el = cal_el.find(f"{_NS}CoefficientValues")
    if values_el is None or values_el.text is None:
        return None
    try:
        coefficients = [float(tok) for tok in values_el.text.split()]
    except ValueError:
        return None
    if len(coefficients) == 2:
        a, b = coefficients
        return Calibration(kind="linear", a=a, b=b)
    if len(coefficients) == 3:
        a, b, c = coefficients
        return Calibration(kind="quadratic", a=a, b=b, c=c)
    return None  # a polynomial order this app's Calibration model can't represent


def load_n42(path):
    """Reads an N42 (ANSI/IEEE N42.42-2011) file, returning
    (data, calibration) -- calibration is None when the file has none.
    Only the raw histogram and energy calibration are extracted;
    everything else in the file (instrument settings, ROIs, timestamps,
    detector metadata) is ignored. Files with anything other than
    exactly one Spectrum element are rejected rather than guessed at."""
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ParseError(f"Not a valid XML file: {path} ({exc})") from exc
    root = tree.getroot()

    spectra = root.findall(f".//{_NS}Spectrum")
    if len(spectra) != 1:
        raise ParseError(
            f"N42 file has {len(spectra)} Spectrum elements; only single-spectrum "
            f"files are supported: {path}"
        )
    spectrum_el = spectra[0]

    data = _parse_channel_data(spectrum_el, path)
    calibration = _parse_calibration(root, spectrum_el)
    return data, calibration
```

### `main_window.py` integration

`_open_file_dialog`'s filter string gains `*.n42`, e.g. `"Spectrum files (*.txt *.spe *.spk *.n42);;...;;N42 files (*.n42);;All files (*)"`.

`_try_load_spectrum` restructures from a uniform `loader = ...; data = loader(path)` to accommodate N42's different return shape:

```python
def _try_load_spectrum(self, path):
    lower = path.lower()
    calibration = None
    try:
        if lower.endswith(".spe"):
            data = load_spe(path)
        elif lower.endswith(".spk"):
            data = load_spk(path)
        elif lower.endswith(".n42"):
            data, calibration = load_n42(path)
        else:
            data = load_histogram(path)
    except ParseError as exc:
        return None, f"{os.path.basename(path)}: {exc}"
    except OSError as exc:
        return None, f"{os.path.basename(path)}: {exc}"

    if calibration is not None and not self._calibration_active:
        self._apply_calibration_change(calibration, True)

    color_index = self._next_color_index
    ...  # unchanged from here
```

(Exact surrounding code confirmed against the current file when this becomes a plan — shown here to establish the shape, not as a literal diff yet.)

## Error Handling Summary

- Malformed XML → `ParseError` (wraps the underlying `xml.etree.ElementTree.ParseError`).
- Zero or multiple `Spectrum` elements → `ParseError` naming the count found.
- Missing/empty `ChannelData`, non-integer channel values, or an unrecognized `compressionCode` → `ParseError` with a specific message, matching the specificity of `.spe`/`.spk`'s own error messages.
- `CountedZeroes` data ending mid run-length pair (malformed) → `ParseError`, not a silent truncation.
- No `EnergyCalibration` referenced, referenced ID not found, or a coefficient count this app's `Calibration` model can't represent (0, 1, or 4+ values) → `calibration = None`, not an error — the histogram still loads.
- All `ParseError`/`OSError` cases surface through the exact same `_try_load_spectrum` → `_load_files` → `QMessageBox.warning` path every other format already uses; nothing new needed there.

## Testing Plan

- **`tests/fixtures/`**: add one of the real user-provided sample files (e.g. `316-2_160V_0785uA.n42`) as a committed fixture, matching the `demo.spk` precedent — a real end-to-end test loading it and checking the known channel count (16384), a few known values, and the exact calibration coefficients verified above.
- **`tests/test_n42_io.py`**: synthetic minimal XML strings (via `tempfile`/`tmp_path`, matching this project's existing test-file conventions) covering: `compressionCode="None"` (basic case), `compressionCode="CountedZeroes"` (a token stream with more than one zero-run, e.g. `"5 0 3 7 0 2"` decoding to `[5, 0, 0, 0, 7, 0, 0]`), a `CountedZeroes` stream ending on an unpaired `0` (raises `ParseError`), an unrecognized `compressionCode` (raises `ParseError`), zero `Spectrum` elements (raises `ParseError`), two `Spectrum` elements (raises `ParseError`), a file with 2 coefficients (linear `Calibration`), a file with 3 coefficients (quadratic `Calibration`), a file with 4 coefficients (calibration is `None`, no error), a file with no `EnergyCalibration` at all (calibration is `None`), a `Spectrum` with an `energyCalibrationReference` pointing at a nonexistent id (calibration is `None`), malformed (non-well-formed) XML (raises `ParseError`), empty `ChannelData` (raises `ParseError`).
- **`tests/test_main_window.py`** (or wherever `_try_load_spectrum`/calibration-related UI tests currently live — confirmed exact file during planning): a test that opening an N42 file with a calibration and no prior active calibration results in `self._calibration_active is True` and the toolbar/menu toggle states matching; a test that opening one while a calibration is already active leaves the existing calibration untouched.
- **Docs**: `tests/test_help_content.py` gets a test confirming HowTo's format list mentions `.n42` (exact wording pinned down when the HowTo text itself is finalized during planning).

## Out of Scope

- No write/save support (`save_n42`) — explicitly not requested.
- No support for files with multiple `RadMeasurement`/`Spectrum` elements (portal-monitor-style continuous acquisitions) — rejected with a clear error per the user's own confirmation.
- No extraction of anything beyond histogram + calibration: instrument metadata, ROIs, timestamps, live/real time, detector info, background configuration, etc. are all ignored, even though several are visible in the sample files.
- No support for the older N42.42-2006 (pre-XML-schema, tabular) format, or non-ANSI-standard "N42-like" vendor variants — only the 2011/2012 XML schema (namespace `http://physics.nist.gov/N42/2011/N42`) confirmed present in the actual sample files.
- No new third-party dependency — `xml.etree.ElementTree` is Python stdlib.
