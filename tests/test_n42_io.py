import os

import numpy as np
import pytest

from histogram_io import ParseError
from n42_io import load_n42

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

_MINIMAL_N42 = """<?xml version="1.0"?>
<RadInstrumentData xmlns="http://physics.nist.gov/N42/2011/N42">
  {calibration_block}
  <RadMeasurement id="RadMeasurement-1">
    {spectrum_block}
  </RadMeasurement>
</RadInstrumentData>
"""

_CALIBRATION_BLOCK = """<EnergyCalibration id="EnergyCalibration-1">
    <CoefficientValues>{coefficients}</CoefficientValues>
  </EnergyCalibration>"""


def _spectrum_block(channel_data, compression="None", cal_ref="EnergyCalibration-1"):
    ref_attr = f' energyCalibrationReference="{cal_ref}"' if cal_ref else ""
    return (
        f'<Spectrum id="RadMeasurement-1Spectrum-1"{ref_attr}>'
        f'<ChannelData compressionCode="{compression}">{channel_data}</ChannelData>'
        f"</Spectrum>"
    )


def _write_n42(tmp_path, xml_text, name="test.n42"):
    path = tmp_path / name
    path.write_text(xml_text, encoding="utf-8")
    return str(path)


def test_load_n42_real_sample_file():
    data, calibration = load_n42(os.path.join(FIXTURES, "316-2_160V_0785uA.n42"))
    assert len(data) == 16384
    assert list(data[:16]) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    assert int(data.sum()) == 599187
    assert int(data[3488]) == 1428
    assert calibration.kind == "quadratic"
    assert calibration.a == pytest.approx(-11.3498272291349)
    assert calibration.b == pytest.approx(0.16808176890571)
    assert calibration.c == pytest.approx(0.0)


def test_load_n42_basic_none_compression(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block=_CALIBRATION_BLOCK.format(coefficients="1.0 2.0 3.0"),
        spectrum_block=_spectrum_block("0 1 2 3"),
    )
    data, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert list(data) == [0, 1, 2, 3]
    assert calibration.kind == "quadratic"
    assert calibration.a == 1.0
    assert calibration.b == 2.0
    assert calibration.c == 3.0


def test_load_n42_calibration_reference_with_xpath_metacharacters(tmp_path):
    """XML attributes may legally contain apostrophes and brackets.
    The lookup used to interpolate the reference into an XPath predicate
    (.//EnergyCalibration[@id='{ref}']), which made ElementTree raise
    SyntaxError on such a file -- escaping the loader's ParseError/OSError
    handling and crashing the app on well-formed input. Matching is now
    plain Python equality: a weird id still RESOLVES when it matches."""
    weird_id = "cal's [1]"
    calibration_block = (
        f'<EnergyCalibration id="{weird_id}">'
        "<CoefficientValues>1.0 2.0</CoefficientValues>"
        "</EnergyCalibration>"
    )
    xml_text = _MINIMAL_N42.format(
        calibration_block=calibration_block,
        spectrum_block=_spectrum_block("0 1 2 3", cal_ref=weird_id),
    )
    data, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert list(data) == [0, 1, 2, 3]
    assert calibration.kind == "linear"
    assert calibration.a == 1.0
    assert calibration.b == 2.0


def test_load_n42_unresolvable_weird_reference_is_no_calibration_not_a_crash(tmp_path):
    """Same metacharacters, no matching calibration: the spectrum loads
    with calibration None -- the function's usual answer for an unusable
    calibration -- instead of raising SyntaxError."""
    xml_text = _MINIMAL_N42.format(
        calibration_block=_CALIBRATION_BLOCK.format(coefficients="1.0 2.0"),
        spectrum_block=_spectrum_block("0 1 2 3", cal_ref="no'such[ref]"),
    )
    data, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert list(data) == [0, 1, 2, 3]
    assert calibration is None


def test_load_n42_counted_zeroes_compression(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block="",
        spectrum_block=_spectrum_block("5 0 3 7 0 2", compression="CountedZeroes", cal_ref=None),
    )
    data, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert list(data) == [5, 0, 0, 0, 7, 0, 0]
    assert calibration is None


def test_load_n42_counted_zeroes_unpaired_zero_raises(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block="",
        spectrum_block=_spectrum_block("5 0", compression="CountedZeroes", cal_ref=None),
    )
    with pytest.raises(ParseError):
        load_n42(_write_n42(tmp_path, xml_text))


def test_load_n42_rejects_oversized_counted_zeroes_run_length(tmp_path):
    # A huge run-length would previously hit `values.extend([0] * run_length)`
    # with no upper bound, crashing with an uncaught MemoryError. Same bug
    # class already fixed via MAT_COLMAX (spk_io.py) / DIM_MAX (mtx_io.py).
    xml_text = _MINIMAL_N42.format(
        calibration_block="",
        spectrum_block=_spectrum_block(
            "5 0 100000000000000", compression="CountedZeroes", cal_ref=None
        ),
    )
    with pytest.raises(ParseError):
        load_n42(_write_n42(tmp_path, xml_text))


def test_load_n42_rejects_negative_counted_zeroes_run_length(tmp_path):
    # `[0] * -3 == []` in Python, so a negative run-length was previously
    # silently accepted and silently shrank the channel array instead of
    # erroring.
    xml_text = _MINIMAL_N42.format(
        calibration_block="",
        spectrum_block=_spectrum_block(
            "5 0 -3 7", compression="CountedZeroes", cal_ref=None
        ),
    )
    with pytest.raises(ParseError):
        load_n42(_write_n42(tmp_path, xml_text))


def test_load_n42_unrecognized_compression_raises(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block="",
        spectrum_block=_spectrum_block("0 1 2", compression="RLE", cal_ref=None),
    )
    with pytest.raises(ParseError):
        load_n42(_write_n42(tmp_path, xml_text))


def test_load_n42_zero_spectra_raises(tmp_path):
    xml_text = _MINIMAL_N42.format(calibration_block="", spectrum_block="")
    with pytest.raises(ParseError):
        load_n42(_write_n42(tmp_path, xml_text))


def test_load_n42_multiple_spectra_raises(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block="",
        spectrum_block=_spectrum_block("0 1", cal_ref=None) + _spectrum_block("2 3", cal_ref=None),
    )
    with pytest.raises(ParseError):
        load_n42(_write_n42(tmp_path, xml_text))


def test_load_n42_two_coefficients_gives_linear_calibration(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block=_CALIBRATION_BLOCK.format(coefficients="10.0 0.5"),
        spectrum_block=_spectrum_block("0 1 2"),
    )
    _, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert calibration.kind == "linear"
    assert calibration.a == 10.0
    assert calibration.b == 0.5


def test_load_n42_four_coefficients_gives_no_calibration(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block=_CALIBRATION_BLOCK.format(coefficients="1 2 3 4"),
        spectrum_block=_spectrum_block("0 1 2"),
    )
    data, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert list(data) == [0, 1, 2]
    assert calibration is None


def test_load_n42_zero_b_coefficient_linear_gives_no_calibration(tmp_path):
    # b == 0 makes Calibration.__post_init__ raise CalibrationError
    # (calibration.py:35-36) -- a device's "not yet calibrated"
    # placeholder should degrade to no calibration, not a load failure.
    xml_text = _MINIMAL_N42.format(
        calibration_block=_CALIBRATION_BLOCK.format(coefficients="5.0 0.0"),
        spectrum_block=_spectrum_block("0 1 2"),
    )
    data, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert list(data) == [0, 1, 2]
    assert calibration is None


def test_load_n42_zero_b_coefficient_quadratic_gives_no_calibration(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block=_CALIBRATION_BLOCK.format(coefficients="5.0 0.0 0.0"),
        spectrum_block=_spectrum_block("0 1 2"),
    )
    data, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert list(data) == [0, 1, 2]
    assert calibration is None


def test_load_n42_no_energy_calibration_element_gives_none(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block="",
        spectrum_block=_spectrum_block("0 1 2", cal_ref=None),
    )
    _, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert calibration is None


def test_load_n42_dangling_calibration_reference_gives_none(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block="",
        spectrum_block=_spectrum_block("0 1 2", cal_ref="DoesNotExist"),
    )
    _, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert calibration is None


def test_load_n42_malformed_xml_raises(tmp_path):
    path = tmp_path / "broken.n42"
    path.write_text("<RadInstrumentData><unclosed>", encoding="utf-8")
    with pytest.raises(ParseError):
        load_n42(str(path))


def test_load_n42_empty_channel_data_raises(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block="",
        spectrum_block=_spectrum_block("", cal_ref=None),
    )
    with pytest.raises(ParseError):
        load_n42(_write_n42(tmp_path, xml_text))


def test_load_n42_rejects_value_overflowing_int64(tmp_path):
    # A channel value beyond int64 range parses fine as a Python int()
    # but raises OverflowError from np.array(..., dtype=np.int64). This
    # must raise ParseError instead, not crash with an unhandled
    # OverflowError -- matching spk_io.py's identical guard for the same
    # risk (see test_lc_rejects_value_overflowing_int64).
    xml_text = _MINIMAL_N42.format(
        calibration_block="",
        spectrum_block=_spectrum_block("0 1 99999999999999999999999999999999", cal_ref=None),
    )
    with pytest.raises(ParseError):
        load_n42(_write_n42(tmp_path, xml_text))


def test_load_n42_non_finite_coefficient_gives_no_calibration(tmp_path):
    # "nan"/"inf" pass float() cleanly, so before the finiteness check in
    # Calibration.__post_init__ this auto-activated a calibration that
    # made every displayed energy NaN (v3.1.0 audit, Minor). Expected
    # outcome matches the other unusable-calibration cases above:
    # degrade to no calibration, not a load failure.
    xml_text = _MINIMAL_N42.format(
        calibration_block=_CALIBRATION_BLOCK.format(coefficients="nan 0.5"),
        spectrum_block=_spectrum_block("0 1 2"),
    )
    data, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert list(data) == [0, 1, 2]
    assert calibration is None


def test_load_n42_inf_coefficient_gives_no_calibration(tmp_path):
    xml_text = _MINIMAL_N42.format(
        calibration_block=_CALIBRATION_BLOCK.format(coefficients="10.0 inf 0.001"),
        spectrum_block=_spectrum_block("0 1 2"),
    )
    data, calibration = load_n42(_write_n42(tmp_path, xml_text))
    assert list(data) == [0, 1, 2]
    assert calibration is None
