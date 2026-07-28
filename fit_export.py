import json
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
        f.write(json.dumps(record) + "\n")


def _format_err(value, err):
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
