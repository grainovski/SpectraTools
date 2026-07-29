import numpy as np
import pytest
from pathlib import Path

from histogram_io import ParseError, load_histogram, save_histogram

FIXTURES = Path(__file__).parent / "fixtures"


def test_parses_test_txt_to_4096_channels():
    data = load_histogram(str(FIXTURES / "test.txt"))
    assert len(data) == 4096
    assert list(data[:5]) == [4, 0, 1, 0, 1]


def test_parses_test1_txt_same_as_test_txt():
    data = load_histogram(str(FIXTURES / "test1.txt"))
    assert len(data) == 4096
    assert list(data[:5]) == [4, 0, 1, 0, 1]


def test_handles_file_without_header_line():
    data = load_histogram(str(FIXTURES / "no_header.txt"))
    assert len(data) == 4096
    assert list(data[:6]) == [4, 0, 1, 0, 1, 0]
    assert list(data[6:]) == [0] * (4096 - 6)


def test_handles_blank_lines_interspersed():
    data = load_histogram(str(FIXTURES / "blank_lines.txt"))
    assert len(data) == 4096
    assert list(data[:3]) == [4, 0, 1]


def test_raises_parse_error_on_no_numeric_data():
    with pytest.raises(ParseError):
        load_histogram(str(FIXTURES / "no_numeric_data.txt"))


def test_bucket_under_4096_pads_to_4096(tmp_path):
    file_path = tmp_path / "short.txt"
    values = list(range(10))
    file_path.write_text("\n".join(str(v) for v in values) + "\n")

    data = load_histogram(str(file_path))

    assert len(data) == 4096
    assert list(data[:10]) == values
    assert list(data[10:]) == [0] * (4096 - 10)


def test_bucket_between_4096_and_8192_pads_to_8192(tmp_path):
    file_path = tmp_path / "medium.txt"
    values = [i % 10 for i in range(5000)]
    file_path.write_text("\n".join(str(v) for v in values) + "\n")

    data = load_histogram(str(file_path))

    assert len(data) == 8192
    assert list(data[:5000]) == values
    assert list(data[5000:]) == [0] * (8192 - 5000)


def test_save_histogram_round_trips_through_load_histogram(tmp_path):
    data = np.array([4, 0, 1, 0, 1, 7, 200])
    path = tmp_path / "out.txt"

    save_histogram(str(path), data)
    result = load_histogram(str(path))

    assert list(result[:len(data)]) == list(data)


def test_save_histogram_writes_one_value_per_line(tmp_path):
    data = [1, 2, 3]
    path = tmp_path / "out.txt"

    save_histogram(str(path), data)

    lines = path.read_text().splitlines()
    assert lines == ["1", "2", "3"]
