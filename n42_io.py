import xml.etree.ElementTree as ET

import numpy as np

from calibration import Calibration
from histogram_io import ParseError

_NS = "{http://physics.nist.gov/N42/2011/N42}"


def _decode_counted_zeroes(tokens, path):
    """Expands N42's CountedZeroes run-length encoding: a "0" token is
    immediately followed by the total number of consecutive zero
    channels starting at that first zero (e.g. "0 3" -> three zeros)."""
    values = []
    i = 0
    while i < len(tokens):
        value = tokens[i]
        if value == 0:
            if i + 1 >= len(tokens):
                raise ParseError(f"N42 CountedZeroes data ends mid run-length pair: {path}")
            run_length = tokens[i + 1]
            values.extend([0] * run_length)
            i += 2
        else:
            values.append(value)
            i += 1
    return values


def _parse_channel_data(spectrum_el, path):
    channel_data_el = spectrum_el.find(f"{_NS}ChannelData")
    if channel_data_el is None or channel_data_el.text is None or not channel_data_el.text.strip():
        raise ParseError(f"N42 file's ChannelData is missing or empty: {path}")

    try:
        tokens = [int(tok) for tok in channel_data_el.text.split()]
    except ValueError as exc:
        raise ParseError(f"N42 ChannelData contains non-integer values: {path}") from exc

    compression = channel_data_el.get("compressionCode", "None")
    if compression == "None":
        values = tokens
    elif compression == "CountedZeroes":
        values = _decode_counted_zeroes(tokens, path)
    else:
        raise ParseError(f"Unsupported N42 ChannelData compressionCode {compression!r}: {path}")

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
