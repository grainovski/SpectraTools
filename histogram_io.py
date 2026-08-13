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
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    values.append(int(stripped))
                except ValueError:
                    continue
    except UnicodeDecodeError as exc:
        raise ParseError(f"File is not valid UTF-8 text: {path}") from exc

    if not values:
        raise ParseError(f"No histogram data found in file: {path}")

    channel_count = _bucket_channel_count(len(values))
    data = np.zeros(channel_count, dtype=np.int64)
    data[: len(values)] = values
    return data


def save_histogram(path: str, data) -> None:
    """Writes `data` as one integer count per line -- the exact inverse
    of load_histogram (which ignores blank lines and pads to the next
    bucket size on read; re-loading a saved file naturally re-pads)."""
    with open(path, "w") as f:
        for value in data:
            f.write(f"{int(value)}\n")
