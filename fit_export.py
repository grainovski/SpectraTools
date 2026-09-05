import csv
import json
import math
import os

from peak_fit import IntegrationResult


def auto_log_path(spectrum_path):
    """Derives the auto-log path from a spectrum's file path -- e.g.
    ".../eu.spe" -> ".../eu_fits.jsonl", next to the spectrum file."""
    directory = os.path.dirname(spectrum_path)
    stem = os.path.splitext(os.path.basename(spectrum_path))[0]
    return os.path.join(directory, f"{stem}_fits.jsonl")


def _peak_record(peak, calibration):
    record = {
        "position": peak.position, "position_err": peak.position_err,
        "fwhm": peak.fwhm, "fwhm_err": peak.fwhm_err,
        "amplitude": peak.amplitude, "amplitude_err": peak.amplitude_err,
        "sigma": peak.sigma, "sigma_err": peak.sigma_err,
        "area": peak.area, "area_err": peak.area_err,
        "full_area": peak.full_area, "full_area_err": peak.full_area_err,
    }
    if calibration is not None:
        record["position_keV"] = calibration.apply(peak.position)
        record["position_err_keV"] = abs(calibration.derivative(peak.position)) * peak.position_err
        record["fwhm_keV"] = abs(calibration.derivative(peak.position)) * peak.fwhm
        record["fwhm_err_keV"] = abs(calibration.derivative(peak.position)) * peak.fwhm_err
    return record


def _integration_layer_record(result, prefix, calibration):
    record = {
        "area": getattr(result, f"{prefix}_area"),
        "area_err": getattr(result, f"{prefix}_area_err"),
        "centroid": getattr(result, f"{prefix}_centroid"),
        "centroid_err": getattr(result, f"{prefix}_centroid_err"),
        "fwhm": getattr(result, f"{prefix}_fwhm"),
        "fwhm_err": getattr(result, f"{prefix}_fwhm_err"),
        "skewness": getattr(result, f"{prefix}_skewness"),
        "skewness_err": getattr(result, f"{prefix}_skewness_err"),
    }
    if calibration is not None:
        centroid = record["centroid"]
        record["centroid_keV"] = calibration.apply(centroid)
        record["centroid_err_keV"] = abs(calibration.derivative(centroid)) * record["centroid_err"]
        record["fwhm_keV"] = abs(calibration.derivative(centroid)) * record["fwhm"]
        record["fwhm_err_keV"] = abs(calibration.derivative(centroid)) * record["fwhm_err"]
    return record


def fit_result_to_json_record(result, spectrum_path, calibration=None):
    """Converts one FitResult into a plain dict covering every
    parameter and uncertainty, ready for json.dumps() -- used by the
    automatic per-fit log. `spectrum_path` is recorded so a later
    re-read of the log can be traced back to its source spectrum.
    `calibration` (a calibration.Calibration or None) adds *_keV fields
    to each peak record when given; omitted entirely otherwise, so
    existing consumers of the auto-log aren't broken by new
    always-required fields."""
    return {
        "type": "fit",
        "timestamp": result.timestamp,
        "spectrum_path": spectrum_path,
        "left_bg_region": list(result.left_bg_region),
        "right_bg_region": list(result.right_bg_region),
        "fit_region": list(result.fit_region),
        "background_slope": result.background_slope,
        "background_intercept": result.background_intercept,
        "link_widths": result.link_widths,
        "tail_fraction": result.tail_fraction,
        "tail_fraction_err": result.tail_fraction_err,
        "step_fraction": result.step_fraction,
        "step_fraction_err": result.step_fraction_err,
        "tail_beta": result.tail_beta,
        "tail_beta_err": result.tail_beta_err,
        "fixed_params": dict(result.fixed_params),
        "peaks": [_peak_record(peak, calibration) for peak in result.peaks],
        "gross_area": result.gross_area, "gross_area_err": result.gross_area_err,
        "net_area": result.net_area, "net_area_err": result.net_area_err,
        "reduced_chi2": result.reduced_chi2,
    }


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
    }
    if result.has_background:
        record["left_bg_region"] = list(result.left_bg_region)
        record["right_bg_region"] = list(result.right_bg_region)
    record["fit_region"] = list(result.fit_region)
    if result.has_background:
        record["background_density"] = result.background_density
    record["gross"] = _integration_layer_record(result, "gross", calibration)
    if result.has_background:
        record["background"] = _integration_layer_record(result, "background", calibration)
        record["net"] = _integration_layer_record(result, "net", calibration)
    return record


def _to_json_record(result, spectrum_path, calibration=None):
    if isinstance(result, IntegrationResult):
        return integration_result_to_json_record(result, spectrum_path, calibration)
    return fit_result_to_json_record(result, spectrum_path, calibration)


def append_auto_log(spectrum_path, result, calibration=None):
    """Appends one JSON-Lines record for `result` to
    `<spectrum_stem>_fits.jsonl`, next to the spectrum file. Raises
    OSError on failure (e.g. read-only directory) -- the caller must
    turn that into a non-blocking status message, since a failed log
    write must never invalidate an already-successful fit."""
    record = _to_json_record(result, spectrum_path, calibration)
    path = auto_log_path(spectrum_path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(record)) + "\n")


def _json_safe(value):
    """Replaces NaN/inf with None (-> JSON `null`) throughout a record.

    An uncertainty is NaN when the fit could not determine that
    parameter (peak_fit._covariance_from). json.dumps would happily
    write a bare `NaN` literal, which Python itself reads back but which
    is NOT valid JSON -- jq, JavaScript's JSON.parse and most other
    parsers reject it outright. Since this log is a .jsonl file users
    are expected to process with other tools, `null` is the only
    portable way to say "no value"."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _format_err(value, err):
    """`value ± err`, with an undetermined (NaN) uncertainty spelled out
    rather than printed as the bare "nan" float repr."""
    if not math.isfinite(err):
        return f"{value:.6g} ± n/a"
    return f"{value:.6g} ± {err:.6g}"


def _format_dual(value, err, calibration, is_width, reference_position=None):
    """Channel value+error, plus a keV parenthetical when `calibration`
    is given. `is_width=True` (e.g. FWHM) scales by the calibration's
    local derivative evaluated at `reference_position` -- the peak's own
    position, NOT the width's numeric value, which has no location of
    its own on the calibration curve; required when is_width is True.
    `is_width=False` (a position, e.g. peak centroid) converts through
    the full calibration, using its own value as the derivative's
    evaluation point."""
    text = _format_err(value, err)
    if calibration is None:
        return text
    if is_width:
        slope = abs(calibration.derivative(reference_position))
        energy_value = slope * value
    else:
        slope = abs(calibration.derivative(value))
        energy_value = calibration.apply(value)
    energy_err = slope * err
    return f"{text} ch ({_format_err(energy_value, energy_err)} keV)"


def fit_result_to_text_report(result, spectrum_path, fit_number=1, calibration=None):
    """Human-readable report for one fit -- one block, used for both a
    single-fit export and (joined by write_text_report) an all-fits
    export. `fit_number` is the fit's 1-based index in the Fit Results
    list, for the block's heading only. `calibration` adds a keV
    parenthetical to Position/FWHM lines when given."""
    lines = [
        f"Fit {fit_number}",
        f"Spectrum: {spectrum_path}",
        f"Timestamp: {result.timestamp}",
        f"Fit region: [{result.fit_region[0]:.2f}, {result.fit_region[1]:.2f}]",
        f"Left background region: [{result.left_bg_region[0]:.2f}, {result.left_bg_region[1]:.2f}]",
        f"Right background region: [{result.right_bg_region[0]:.2f}, {result.right_bg_region[1]:.2f}]",
        f"Background: slope={result.background_slope:.6g}, intercept={result.background_intercept:.6g}",
        f"Independent widths: {not result.link_widths}",
        f"Reduced chi^2: {result.reduced_chi2:.4g}" if result.reduced_chi2 is not None
        else "Reduced chi^2: undefined (zero degrees of freedom)",
    ]
    if result.step_fraction is not None:
        lines.append(
            f"Step: {_format_err(result.step_fraction, result.step_fraction_err)} "
            f"of peak height (background -- not counted in the volume)"
        )
    if result.tail_fraction is not None:
        lines.append(
            f"Left tail: r={_format_err(result.tail_fraction, result.tail_fraction_err)}, "
            f"beta={_format_err(result.tail_beta, result.tail_beta_err)}"
        )
    if result.fixed_params:
        fixed_text = ", ".join(
            f"{name}={value:.6g}" for name, value in sorted(result.fixed_params.items())
        )
        lines.append(f"Fixed parameters: {fixed_text}")
    else:
        lines.append("Fixed parameters: none")
    lines.append(f"Region full area (no background subtracted): {_format_err(result.gross_area, result.gross_area_err)}")
    lines.append(f"Region net area (background subtracted):     {_format_err(result.net_area, result.net_area_err)}")
    lines.append("")
    for i, peak in enumerate(result.peaks):
        lines.append(f"  Peak {i + 1}:")
        lines.append(f"    Position:  {_format_dual(peak.position, peak.position_err, calibration, is_width=False)}")
        lines.append(f"    FWHM:      {_format_dual(peak.fwhm, peak.fwhm_err, calibration, is_width=True, reference_position=peak.position)}")
        lines.append(f"    Amplitude: {_format_err(peak.amplitude, peak.amplitude_err)}")
        lines.append(f"    Sigma:     {_format_err(peak.sigma, peak.sigma_err)}")
        lines.append(f"    Full area (no background subtracted): {_format_err(peak.full_area, peak.full_area_err)}")
        lines.append(f"    Net area (background subtracted):     {_format_err(peak.area, peak.area_err)}")
    return "\n".join(lines)


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


def _to_text_report(result, spectrum_path, fit_number, calibration=None):
    if isinstance(result, IntegrationResult):
        return integration_result_to_text_report(result, spectrum_path, fit_number, calibration)
    return fit_result_to_text_report(result, spectrum_path, fit_number, calibration)


def write_text_report(path, results, spectrum_path, calibration=None):
    """Writes a plain-text report for one or more fits/integrations to
    `path`, overwriting any existing file. `results` is a list of
    (fit_number, result) pairs, in the order they should appear in the
    report."""
    blocks = [
        _to_text_report(result, spectrum_path, fit_number, calibration)
        for fit_number, result in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + "\n")


# --- tabular exports (X8) ----------------------------------------------
#
# One row per PEAK rather than per fit: a fit with three peaks is three
# lines of a table, which is what goes into a paper or a spreadsheet.
# Integration results have no peaks and are skipped by both writers --
# their gross/background/net breakdown is a different shape entirely and
# forcing it into these columns would produce rows whose numbers mean
# something other than the header says.

_TABLE_COLUMNS = (
    "fit", "peak", "position", "position_err", "fwhm", "fwhm_err",
    "area", "area_err", "full_area", "full_area_err", "reduced_chi2",
)


def _table_rows(results, calibration=None):
    """(header, rows) for the tabular writers, values already converted to
    keV where a calibration is active.

    Position and FWHM are converted through the SAME helpers the on-screen
    panel uses, so an exported table and the numbers the user was looking
    at cannot disagree. Area is never converted -- it is a count, and
    counts have no energy equivalent."""
    header = list(_TABLE_COLUMNS)
    if calibration is not None:
        header = [
            f"{name}_keV" if name.startswith(("position", "fwhm")) else name
            for name in header
        ]

    rows = []
    for fit_number, result in results:
        if not getattr(result, "peaks", None):
            continue
        for index, peak in enumerate(result.peaks, start=1):
            position = peak.position
            position_err = peak.position_err
            fwhm = peak.fwhm
            fwhm_err = peak.fwhm_err
            if calibration is not None:
                position = calibration.apply(peak.position)
                position_err = abs(calibration.derivative(peak.position)) * peak.position_err
                # A width has no position of its own on the calibration
                # curve, so it is scaled at its PEAK's position -- the same
                # rule fit_mode._to_energy follows.
                scale = abs(calibration.derivative(peak.position))
                fwhm = scale * peak.fwhm
                fwhm_err = scale * peak.fwhm_err
            rows.append([
                fit_number, index, position, position_err, fwhm, fwhm_err,
                peak.area, peak.area_err, peak.full_area, peak.full_area_err,
                result.reduced_chi2,
            ])
    return header, rows


def _number(value, digits=4):
    if value is None:
        return ""
    try:
        if math.isnan(value):
            return ""
    except TypeError:
        return str(value)
    return f"{value:.{digits}g}"


def write_csv(path, results, spectrum_path, calibration=None):
    """One row per fitted peak, for a spreadsheet.

    An undetermined uncertainty is written as an EMPTY cell rather than
    "nan" or "n/a": a spreadsheet reads an empty cell as missing and a text
    cell as text, which would silently turn the whole column into strings
    and break any average taken over it.
    """
    header, rows = _table_rows(results, calibration)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"# spectrum: {spectrum_path}"])
        writer.writerow(header)
        for row in rows:
            # Eight significant digits, not the four the human-readable
            # writers use: this file is meant to be computed with, and
            # rounding an area of 1234.5 to "1234" on the way out would
            # lose real precision the fit actually determined.
            writer.writerow([
                value if isinstance(value, int) else _number(value, digits=8)
                for value in row
            ])


def _latex_escape(text):
    """LaTeX has characters that are syntax rather than text, and a
    spectrum path is very likely to contain underscores.

    Escaped in ONE pass over the input, not sequential str.replace
    calls: the backslash's own replacement contains braces, and the
    later brace-escaping pass mangled it into \\textbackslash\\{\\} --
    the classic ordering bug. A single pass never rescans its own
    output, so no ordering exists to get wrong."""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _latex_cell(value, err, digits=4):
    """A value with its uncertainty, or just the value when the
    uncertainty is undetermined -- writing "$\\pm$ nan" into a paper would
    be worse than writing nothing."""
    text = _number(value, digits)
    if not text:
        return "--"
    err_text = _number(err, 2)
    return f"${text} \\pm {err_text}$" if err_text else f"${text}$"


def write_latex(path, results, spectrum_path, calibration=None):
    """A LaTeX `tabular` of one row per fitted peak, ready to paste into a
    paper -- which is what these numbers are usually for, and where
    retyping them by hand is exactly where transcription errors enter.

    Emitted as a bare table environment rather than a whole document, so
    it drops into an existing paper without stripping a preamble.
    """
    _, rows = _table_rows(results, calibration)
    unit = " (keV)" if calibration is not None else " (ch)"
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Generated by SpectraTools from "
                f"{_latex_escape(str(spectrum_path))}\n")
        f.write("\\begin{table}[htbp]\n  \\centering\n")
        f.write("  \\begin{tabular}{rrccrr}\n    \\hline\n")
        f.write(f"    Fit & Peak & Position{unit} & FWHM{unit} & "
                "Net area & Full area \\\\\n    \\hline\n")
        for row in rows:
            (fit_number, index, position, position_err, fwhm, fwhm_err,
             area, area_err, full_area, full_area_err, _chi2) = row
            f.write(
                f"    {fit_number} & {index} & "
                f"{_latex_cell(position, position_err)} & "
                f"{_latex_cell(fwhm, fwhm_err)} & "
                f"{_latex_cell(area, area_err, digits=6)} & "
                f"{_latex_cell(full_area, full_area_err, digits=6)} \\\\\n"
            )
        f.write("    \\hline\n  \\end{tabular}\n")
        f.write("  \\caption{Fit results.}\n  \\label{tab:fits}\n\\end{table}\n")
