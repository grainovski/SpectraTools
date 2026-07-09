import struct
from pathlib import Path

import pytest

from histogram_io import ParseError
from spe_io import load_spe

FIXTURES = Path(__file__).parent / "fixtures"


def _build_spe_bytes(endian, values):
    channel_count = len(values)
    record1_payload = struct.pack(endian + "8s4i", b"test    ", channel_count, 1, 1, 1)
    record1 = (
        struct.pack(endian + "i", 24) + record1_payload + struct.pack(endian + "i", 24)
    )
    record2_payload = struct.pack(endian + f"{channel_count}f", *values)
    payload_size = channel_count * 4
    record2 = (
        struct.pack(endian + "i", payload_size)
        + record2_payload
        + struct.pack(endian + "i", payload_size)
    )
    return record1 + record2


def test_parses_eu_spe_matches_test_txt_values():
    # eu.spe is the real binary source test.txt/test1.txt were exported
    # from -- confirmed by manual byte-level inspection: same channel
    # count, same leading values.
    data = load_spe(str(FIXTURES / "eu.spe"))
    assert len(data) == 4096
    assert list(data[:5]) == [4, 0, 1, 0, 1]


def test_parses_little_endian_spe(tmp_path):
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    file_path = tmp_path / "little.spe"
    file_path.write_bytes(_build_spe_bytes("<", values))

    data = load_spe(str(file_path))

    assert len(data) == 5
    assert list(data) == [1, 2, 3, 4, 5]


def test_parses_big_endian_spe(tmp_path):
    values = [10.0, 20.0, 30.0]
    file_path = tmp_path / "big.spe"
    file_path.write_bytes(_build_spe_bytes(">", values))

    data = load_spe(str(file_path))

    assert len(data) == 3
    assert list(data) == [10, 20, 30]


def test_raises_parse_error_on_malformed_spe(tmp_path):
    file_path = tmp_path / "bad.spe"
    file_path.write_bytes(b"not a valid spe file at all, too short and wrong")
    with pytest.raises(ParseError):
        load_spe(str(file_path))


def test_raises_parse_error_on_truncated_spe(tmp_path):
    good_bytes = _build_spe_bytes("<", [1.0, 2.0, 3.0])
    file_path = tmp_path / "truncated.spe"
    file_path.write_bytes(good_bytes[:-4])  # cut off the final trailing marker
    with pytest.raises(ParseError):
        load_spe(str(file_path))
