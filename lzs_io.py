"""Reader for labZY / nanoMCA `.lzs` spectrum files.

An `.lzs` file is XML: one `<spectrum>` element carrying the histogram as
one integer count per line inside `<data>`, plus acquisition metadata
(`<time>`) and a two-point energy calibration (`<calibration>`).

Read-only, like every other reader here except histogram_io/spe_io/spk_io
-- the app writes .txt/.spe/.spk and nothing else.

THE FILES ARE NOT WELL-FORMED XML, so this does not parse the document as
a whole. Every real file produced by firmware 30.20 closes the
`<volatile>` section's `<slowadvc>` element with `</slowadc>`, and
`xml.etree.ElementTree.parse()` rejects the entire document over it --
all four sample files fail identically at that line. Nothing in the
section matters to this app, so rather than repairing the text (guessing
at a vendor's typo) or hand-rolling a tolerant XML parser, each top-level
section this reader actually needs is located by name and parsed ON ITS
OWN, where it is well-formed. A future firmware that fixes the typo reads
identically; a future firmware that breaks a section we ignore stays
readable.

Two traps in the format, both found by measuring real files rather than
by reading the structure:

* `<registers>` contains its OWN `<data>` element (128 values of hardware
  register state). Searching the document for "the data element" finds
  the wrong one, or both. Everything here is scoped inside `<spectrum>`.

* `<softsize>` is NOT the number of valid channels and must not be used
  to truncate. In `18-082025-AlphaSpec-Mix-Source.lzs` it reads 8192
  against a `<hardsize>` of 16384 -- and 99.97% of that file's counts
  (53,149,554 of 53,163,795) lie ABOVE channel 8192. Truncating there
  would silently discard almost the entire spectrum. The `<data>` block
  holds `hardsize` values and all of them are real.
"""

import re
import xml.etree.ElementTree as ET

import numpy as np

from calibration import CalibrationError, from_points
from histogram_io import ParseError

#: `<units>` value meaning keV. The only value observed across the sample
#: files, and confirmed keV by the numbers themselves: an energy of 5156
#: against a 5.156 MeV alpha line. The app's calibration is keV
#: throughout, so any OTHER units value is left uncalibrated rather than
#: relabelled as keV -- being wrong about the scale of an axis is worse
#: than not having one.
_KEV_UNITS = "2"

#: Accepted spellings of "on" in this format, which is not consistent
#: with itself: `<enabled>` uses YES/NO while `<useA>`/`<useB>` use 1/0.
_TRUE_VALUES = {"1", "YES", "TRUE"}


def _section(text, name, path):
    """The `<name>...</name>` top-level block, parsed on its own, or None.

    See the module docstring for why sections are parsed individually
    instead of parsing the document.
    """
    match = re.search(r"<" + name + r"\b[^>]*>.*?</" + name + r">", text, re.S)
    if match is None:
        return None
    try:
        return ET.fromstring(match.group(0))
    except ET.ParseError as exc:
        raise ParseError(
            f"The <{name}> section of this .lzs file is malformed: {path} ({exc})"
        ) from exc


def _text(element, tag):
    """`element`'s child `tag` as a stripped string, or "" if absent."""
    if element is None:
        return ""
    return (element.findtext(tag) or "").strip()


def _parse_calibration(text, path):
    """The file's two-point energy calibration, or None if it does not
    carry a usable one.

    Unusable is the normal case, not an error: an acquisition taken
    without calibrating has `<enabled>NO</enabled>`, and one point may be
    flagged unused. Every such case returns None and the spectrum still
    loads -- matching n42_io, which treats a degenerate calibration the
    same way.
    """
    block = _section(text, "calibration", path)
    if block is None:
        return None
    if _text(block, "enabled").upper() not in _TRUE_VALUES:
        return None
    if _text(block, "units") != _KEV_UNITS:
        return None

    channels, energies = [], []
    for point in ("A", "B"):
        if _text(block, "use" + point).upper() not in _TRUE_VALUES:
            continue
        try:
            channel = float(_text(block, "channel" + point))
            energy = float(_text(block, "energy" + point))
        except ValueError:
            continue
        channels.append(channel)
        energies.append(energy)

    if len(channels) < 2:
        return None
    try:
        return from_points(channels, energies)
    except CalibrationError:
        # Two points at the same channel, or an energy pair a line cannot
        # be drawn through. Same outcome as a disabled calibration.
        return None


def load_lzs(path):
    """Reads a labZY/nanoMCA `.lzs` file, returning (data, calibration) --
    exactly as load_n42 does, so main_window can dispatch to either.
    `calibration` is None when the file carries no usable one.

    Only the histogram and the energy calibration are read; acquisition
    metadata (live/real/dead time, the date, hardware registers, firmware
    details) is ignored, matching this app's other readers.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except UnicodeDecodeError as exc:
        raise ParseError(f"File is not valid UTF-8 text: {path}") from exc

    spectrum = _section(text, "spectrum", path)
    if spectrum is None:
        raise ParseError(f"No <spectrum> section found in .lzs file: {path}")

    # Scoped to <spectrum> -- <registers> has a <data> element of its own.
    data_element = spectrum.find("data")
    if data_element is None or not (data_element.text or "").strip():
        raise ParseError(f".lzs file's spectrum data is missing or empty: {path}")

    try:
        values = [int(token) for token in data_element.text.split()]
    except ValueError as exc:
        raise ParseError(f".lzs spectrum data contains non-integer values: {path}") from exc

    try:
        data = np.array(values, dtype=np.int64)
    except OverflowError as exc:
        raise ParseError(
            f".lzs spectrum data contains an out-of-range channel value: {path}"
        ) from exc

    return data, _parse_calibration(text, path)
