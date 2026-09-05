"""Save fits to a file and read them back.

Until this existed, closing the app threw away every fit -- the analysis
had to be redone from scratch next session, which is the difference
between a viewer and a tool you can come back to.

Modelled on HDTV's fitxml.py, and specifically on the part of it that is
easy to skip: it stamps a schema VERSION into every file and keeps a
dedicated reader for each historical version (RestoreFromXml_v1_3 back
through _v0), so a file written years ago still opens. That discipline
starts here at version 1 rather than being retrofitted after the first
schema change breaks somebody's saved work.

Two ways to bring a fit back, as HDTV also offers:
  * RESTORE the stored values, which reproduces exactly what was reported
    when the file was written.
  * REFIT from the stored marks, which re-runs the fit with today's code.
    These differ whenever the fitting has changed -- and across v4.0.0 it
    changed a great deal -- so which one is wanted is the user's call, not
    a default to guess at.
"""

import json

from peak_fit import FitResult, PeakResult, fit_peaks

SCHEMA_VERSION = 1


class FitFileError(Exception):
    """Raised when a fit file cannot be read or is not one."""


_PEAK_FIELDS = (
    "position", "position_err", "fwhm", "fwhm_err", "area", "area_err",
    "amplitude", "amplitude_err", "sigma", "sigma_err",
    "full_area", "full_area_err",
)

_RESULT_FIELDS = (
    "left_bg_region", "right_bg_region", "fit_region",
    "background_slope", "background_intercept", "link_widths",
    "tail_fraction", "tail_fraction_err", "tail_beta", "tail_beta_err",
    "step_fraction", "step_fraction_err",
    "fixed_params", "visible", "timestamp",
    "gross_area", "gross_area_err", "net_area", "net_area_err",
    "reduced_chi2", "fit_background",
)


def _clean(value):
    """NaN and infinity are not JSON. An undetermined uncertainty is
    stored as null and read back as NaN, which is the same convention
    fit_export._json_safe already uses for the auto-log -- so a file
    stays valid JSON that any other tool can parse."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return value


def _restore_float(value):
    return float("nan") if value is None else value


def to_record(result):
    """One fit as a plain JSON-safe dict."""
    record = {name: _clean(getattr(result, name, None)) for name in _RESULT_FIELDS}
    for name in ("left_bg_region", "right_bg_region", "fit_region"):
        if record[name] is not None:
            record[name] = list(record[name])
    record["peaks"] = [
        {field: _clean(getattr(peak, field, None)) for field in _PEAK_FIELDS}
        for peak in result.peaks
    ]
    # The background error model travels too, so a restored fit can redraw
    # its uncertainty band without the spectrum it came from.
    record["background_anchors"] = [
        [_clean(v) for v in anchor] for anchor in (result.background_anchors or ())
    ]
    record["background_covariance"] = [
        _clean(v) for v in (result.background_covariance or ())
    ]
    return record


def savable(results):
    """Just the fits out of a spectrum's result list.

    A spectrum's results hold integrations alongside fits, and an
    IntegrationResult has no peaks at all -- it reports a region's
    gross/background/net totals rather than fitted peak parameters, so it
    does not fit this schema. Filtered here rather than at the call site so
    every caller agrees on what a fit file contains.
    """
    return [result for result in results if getattr(result, "peaks", None) is not None]


def save(path, results, spectrum_path):
    """Writes the fits among `results` to `path`. Integration results are
    skipped -- see savable()."""
    document = {
        "format": "spectratools-fits",
        "schema_version": SCHEMA_VERSION,
        "spectrum": str(spectrum_path),
        "fits": [to_record(result) for result in savable(results)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
        f.write("\n")


def _from_record_v1(record):
    peaks = [
        PeakResult(**{
            field: _restore_float(peak.get(field, 0.0)) for field in _PEAK_FIELDS
        })
        for peak in record.get("peaks", [])
    ]
    values = {}
    for name in _RESULT_FIELDS:
        value = record.get(name)
        if name in ("left_bg_region", "right_bg_region", "fit_region"):
            values[name] = tuple(value) if value is not None else (0.0, 0.0)
        elif name in ("tail_fraction", "tail_fraction_err", "tail_beta", "tail_beta_err",
                      "step_fraction", "step_fraction_err"):
            # These are None when the fit had no tail or no step at all,
            # which is different from an undetermined uncertainty -- so
            # None here must stay None rather than becoming NaN. A file
            # written before the step existed simply has no key, and
            # record.get returns None, which is the right answer.
            values[name] = value
        elif name == "fixed_params":
            values[name] = dict(value or {})
        elif name in ("link_widths", "visible", "fit_background"):
            values[name] = bool(value) if value is not None else (name != "fit_background")
        elif name == "timestamp":
            values[name] = value
        elif name == "reduced_chi2":
            values[name] = value
        else:
            values[name] = _restore_float(value)

    result = FitResult(peaks=peaks, **values)
    result.background_anchors = tuple(
        tuple(anchor) for anchor in record.get("background_anchors") or ()
    )
    result.background_covariance = tuple(record.get("background_covariance") or ())
    return result


#: One reader per schema version, kept forever. When the schema changes,
#: add a new entry -- never edit an old one, or files written by an older
#: build stop opening.
_READERS = {1: _from_record_v1}


def load(path):
    """Reads a fit file, returning (results, spectrum_path)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            document = json.load(f)
    except json.JSONDecodeError as exc:
        raise FitFileError(f"Not a valid fit file: {path} ({exc})") from exc
    except UnicodeDecodeError as exc:
        # A binary file (an installer, a .mtx, a .root) picked in the Load
        # Fits dialog fails DECODING before json ever sees it -- and the
        # dialog catches only FitFileError, so this used to crash the app.
        # Same guard calibration.read_coefficients_file already carries.
        raise FitFileError(f"Not a valid fit file (not UTF-8 text): {path}") from exc
    except OSError as exc:
        raise FitFileError(f"Could not read {path}: {exc}") from exc

    if not isinstance(document, dict) or document.get("format") != "spectratools-fits":
        raise FitFileError(f"Not a SpectraTools fit file: {path}")

    version = document.get("schema_version")
    reader = _READERS.get(version)
    if reader is None:
        raise FitFileError(
            f"Fit file uses schema version {version!r}, which this build does not "
            f"understand (it knows {sorted(_READERS)}): {path}"
        )
    try:
        results = [reader(record) for record in document.get("fits", [])]
    except (TypeError, ValueError, KeyError) as exc:
        raise FitFileError(f"Fit file is malformed: {path} ({exc})") from exc
    return results, document.get("spectrum")


def refit(result, x, y, variance=None):
    """Re-runs `result`'s fit against `x`/`y` using its stored MARKS, and
    returns the new FitResult.

    The point of offering this alongside a plain restore: the marks are
    what the user chose, while the numbers are what a particular build
    computed from them. Re-running picks up every later improvement to the
    fitting -- and across v4.0.0 the peak area, its uncertainty, the
    background's uncertainty and the width seeding all changed -- so a fit
    saved before them can be brought up to date without re-marking it by
    hand.
    """
    return fit_peaks(
        x, y,
        result.left_bg_region, result.right_bg_region, result.fit_region,
        [peak.position for peak in result.peaks],
        link_widths=result.link_widths,
        enable_left_tail=result.tail_fraction is not None,
        fixed_params=dict(result.fixed_params or {}),
        variance=variance,
        fit_background=result.fit_background,
        enable_step=result.step_fraction is not None,
    )
