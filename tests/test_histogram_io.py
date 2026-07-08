from pathlib import Path

from histogram_io import load_histogram

FIXTURES = Path(__file__).parent / "fixtures"


def test_parses_test_txt_to_4096_channels():
    data = load_histogram(str(FIXTURES / "test.txt"))
    assert len(data) == 4096
    assert list(data[:5]) == [4, 0, 1, 0, 1]


def test_parses_test1_txt_same_as_test_txt():
    data = load_histogram(str(FIXTURES / "test1.txt"))
    assert len(data) == 4096
    assert list(data[:5]) == [4, 0, 1, 0, 1]
