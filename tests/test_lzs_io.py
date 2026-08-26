"""Tests for the labZY / nanoMCA .lzs reader.

The real sample file is committed as a fixture, the way test_n42_io.py
commits a real .n42: the format's two traps (a second <data> element, and
a <softsize> that must not truncate) are properties of real acquisitions,
and a synthetic file that did not reproduce them would prove nothing.
Synthetic files cover the edge cases the samples happen not to contain.
"""

import os

import numpy as np
import pytest

from histogram_io import ParseError
from lzs_io import load_lzs

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
REAL_FILE = os.path.join(FIXTURES, "18-082025-AlphaSpec-Mix-Source.lzs")


def _lzs(
    data_values,
    hardsize=None,
    softsize=None,
    calibration=True,
    enabled="YES",
    units="2",
    use_a="1",
    use_b="1",
    channel_a="100.0000",
    energy_a="1000.0000",
    channel_b="200.0000",
    energy_b="2000.0000",
    registers=8,
    malformed_volatile=True,
):
    """A synthetic .lzs document.

    `malformed_volatile` defaults to True because every real file is
    malformed in exactly that way -- see lzs_io's module docstring.
    """
    hardsize = len(data_values) if hardsize is None else hardsize
    softsize = hardsize if softsize is None else softsize
    body = "\n".join(str(v) for v in data_values)
    register_block = "\n".join("0" for _ in range(registers))
    calibration_block = ""
    if calibration:
        calibration_block = (
            "<calibration>\n"
            "  <enabled>" + enabled + "</enabled>\n"
            "  <units>" + units + "</units>\n"
            "  <channelA>" + channel_a + "</channelA>\n"
            "  <energyA>" + energy_a + "</energyA>\n"
            "  <useA>" + use_a + "</useA>\n"
            "  <channelB>" + channel_b + "</channelB>\n"
            "  <energyB>" + energy_b + "</energyB>\n"
            "  <useB>" + use_b + "</useB>\n"
            "  <useall>NO</useall>\n"
            "</calibration>\n"
        )
    if malformed_volatile:
        volatile = (
            "<volatile>\n  <firmware>30.20</firmware>\n"
            "  <slowadvc> 0.74</slowadc>\n</volatile>\n"
        )
    else:
        volatile = "<volatile>\n  <firmware>30.20</firmware>\n</volatile>\n"
    return (
        '<?xml version="1.0"?>\n'
        "<nanoMCA>\n"
        " <serialnumber>13199</serialnumber>\n"
        "<spectrum>\n"
        "  <tag>labZY-MCA spectrum</tag>\n"
        "  <hardsize>" + str(hardsize) + "</hardsize>\n"
        "  <softsize>" + str(softsize) + "</softsize>\n"
        "  <data>\n" + body + "\n  </data>\n"
        "</spectrum>\n"
        "<time>\n"
        "  <real> 1400.0000</real>\n"
        "  <live> 1399.6155</live>\n"
        "  <dead>    0.0275</dead>\n"
        "  <date>08/03/2024  13:42:08</date>\n"
        "</time>\n"
        "<registers>\n"
        "  <size>" + str(registers) + "</size>\n"
        "  <data>\n" + register_block + "\n  </data>\n"
        "</registers>\n"
        + calibration_block + volatile + "</nanoMCA>\n"
    )


def _write(tmp_path, text, name="test.lzs"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# --- the real file -----------------------------------------------------


def test_load_real_sample_file():
    data, calibration = load_lzs(REAL_FILE)
    assert data.dtype == np.int64
    assert data.size == 16384
    assert int(data.sum()) == 53163795
    assert int(data.max()) == 1194883
    assert calibration.kind == "linear"
    # The file's own two points must map back to their own energies.
    assert calibration.apply(12939.04) == pytest.approx(5156.0, abs=1e-6)
    assert calibration.apply(14564.32) == pytest.approx(5804.0, abs=1e-6)


def test_softsize_does_not_truncate_the_spectrum():
    """The trap this format sets. The real fixture declares softsize 8192
    against hardsize 16384, and 99.97% of its counts lie ABOVE channel
    8192 -- truncating there would discard almost the entire spectrum
    while still producing a plausible-looking result.
    """
    data, _ = load_lzs(REAL_FILE)
    assert data.size == 16384
    above = data[8192:]
    assert int(above.sum()) == 53149554
    assert int((above != 0).sum()) == 2196


def test_the_real_file_is_not_well_formed_xml():
    """Guards the reason this reader parses section by section: a
    document-level parse of any real file fails on the firmware's
    <slowadvc>...</slowadc> typo. If a future firmware fixes that, this
    test fails loudly rather than the reader quietly depending on it.
    """
    import xml.etree.ElementTree as ET

    with pytest.raises(ET.ParseError):
        ET.parse(REAL_FILE)


# --- the second <data> element -----------------------------------------


def test_register_data_is_not_read_as_spectrum_data(tmp_path):
    """<registers> carries its own <data>. Reading "the data element"
    would take the wrong one, or concatenate both.
    """
    path = _write(tmp_path, _lzs([5, 6, 7, 8], registers=64))
    data, _ = load_lzs(path)
    assert list(data) == [5, 6, 7, 8]


# --- calibration -------------------------------------------------------


def test_two_point_calibration_is_read(tmp_path):
    path = _write(tmp_path, _lzs([1, 2, 3]))
    _, calibration = load_lzs(path)
    # (100 -> 1000) and (200 -> 2000) is E = 10 * channel.
    assert calibration.apply(100.0) == pytest.approx(1000.0)
    assert calibration.apply(150.0) == pytest.approx(1500.0)


def test_disabled_calibration_is_not_returned(tmp_path):
    """An uncalibrated acquisition is normal, not an error: the spectrum
    still loads.
    """
    path = _write(tmp_path, _lzs([1, 2, 3], enabled="NO"))
    data, calibration = load_lzs(path)
    assert list(data) == [1, 2, 3]
    assert calibration is None


def test_calibration_needs_both_points_flagged_used(tmp_path):
    path = _write(tmp_path, _lzs([1, 2, 3], use_b="0"))
    _, calibration = load_lzs(path)
    assert calibration is None


def test_non_kev_units_are_left_uncalibrated(tmp_path):
    """The app's axis is keV. Relabelling some other unit as keV would be
    worse than showing channels.
    """
    path = _write(tmp_path, _lzs([1, 2, 3], units="3"))
    _, calibration = load_lzs(path)
    assert calibration is None


def test_degenerate_calibration_points_are_rejected(tmp_path):
    """Both points on the same channel describe no line."""
    path = _write(tmp_path, _lzs([1, 2, 3], channel_a="100.0", channel_b="100.0"))
    _, calibration = load_lzs(path)
    assert calibration is None


def test_unparseable_calibration_numbers_do_not_fail_the_load(tmp_path):
    path = _write(tmp_path, _lzs([1, 2, 3], channel_b="not-a-number"))
    data, calibration = load_lzs(path)
    assert list(data) == [1, 2, 3]
    assert calibration is None


def test_a_file_with_no_calibration_section_still_loads(tmp_path):
    path = _write(tmp_path, _lzs([4, 5, 6], calibration=False))
    data, calibration = load_lzs(path)
    assert list(data) == [4, 5, 6]
    assert calibration is None


# --- malformed input ---------------------------------------------------


def test_a_well_formed_volatile_section_also_loads(tmp_path):
    """The reader tolerates the firmware typo but must not DEPEND on it."""
    path = _write(tmp_path, _lzs([1, 2, 3], malformed_volatile=False))
    data, _ = load_lzs(path)
    assert list(data) == [1, 2, 3]


def test_missing_spectrum_section_is_refused(tmp_path):
    path = _write(tmp_path, '<?xml version="1.0"?>\n<nanoMCA>\n</nanoMCA>\n')
    with pytest.raises(ParseError, match="No <spectrum> section"):
        load_lzs(path)


def test_empty_spectrum_data_is_refused(tmp_path):
    path = _write(tmp_path, _lzs([]))
    with pytest.raises(ParseError, match="missing or empty"):
        load_lzs(path)


def test_non_integer_spectrum_data_is_refused(tmp_path):
    path = _write(tmp_path, _lzs(["1", "2.5", "3"]))
    with pytest.raises(ParseError, match="non-integer"):
        load_lzs(path)


def test_out_of_range_counts_are_refused(tmp_path):
    path = _write(tmp_path, _lzs(["1", str(2 ** 70), "3"]))
    with pytest.raises(ParseError, match="out-of-range"):
        load_lzs(path)


def test_a_malformed_spectrum_section_is_refused(tmp_path):
    text = _lzs([1, 2, 3]).replace("</hardsize>", "</hardsizeX>")
    path = _write(tmp_path, text)
    with pytest.raises(ParseError, match="malformed"):
        load_lzs(path)


def test_a_binary_file_is_refused_not_crashed_on(tmp_path):
    path = tmp_path / "binary.lzs"
    path.write_bytes(b"\xff\xfe\x00\x01" + bytes(range(256)))
    with pytest.raises(ParseError, match="not valid UTF-8"):
        load_lzs(str(path))


# --- the app actually reaches it ---------------------------------------


def test_main_window_opens_an_lzs_file_and_applies_its_calibration(qapp):
    """A reader the file dialog cannot dispatch to is not a shipped
    feature -- see this app's own history of computation-layer tests
    passing while the UI could not reach the code.
    """
    from main_window import MainWindow

    main_window = MainWindow()
    main_window._load_files([REAL_FILE])

    assert len(main_window.spectra) == 1
    assert main_window.spectra[0].data.size == 16384
    assert main_window._calibration is not None
    assert main_window._calibration_active is True
    assert main_window._calibration.apply(12939.04) == pytest.approx(5156.0, abs=1e-6)


def test_the_open_dialog_offers_lzs(qapp):
    """The filter string is what makes the format selectable at all."""
    import inspect

    from main_window import MainWindow

    source = inspect.getsource(MainWindow._open_file_dialog)
    assert "*.lzs" in source
