import numpy as np


class ParseError(Exception):
    """Raised when a file contains no parseable histogram data."""


def _bucket_channel_count(n: int) -> int:
    size = 4096
    while size < n:
        size *= 2
    return size


def load_histogram(path: str) -> np.ndarray:
    values = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values.append(int(stripped))
            except ValueError:
                continue

    if not values:
        raise ParseError(f"No histogram data found in file: {path}")

    channel_count = _bucket_channel_count(len(values))
    data = np.zeros(channel_count, dtype=np.int64)
    data[: len(values)] = values
    return data
