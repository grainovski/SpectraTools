"""Reader for `.sou` source-description files, and the matcher that
attaches their lines to fitted peaks.

A `.sou` file lists the known gamma lines of one calibration nuclide,
one per line, as four whitespace-separated numbers with no header:

    energy (keV)   energy error   intensity   intensity error

The intensity is relative, normalised so the strongest line reads 10000
-- though not every file honours that: co56.sou peaks at 100000 and
na24.sou at 1000, which is why the intensity is displayed and never
relied upon. Every sample file in the working directory has exactly this
shape, so unlike `.lzs` there is no format trap to work around, and the
reader is strict rather than tolerant: anything that is
not four finite numbers on a non-blank line is refused with the line
named, rather than skipped -- a silently dropped line would leave the
user calibrating against a source that is quietly missing a peak.

Read-only, like every reader here except histogram_io/spe_io/spk_io.
"""

import math
from dataclasses import dataclass

from histogram_io import ParseError

#: Columns per line. Named rather than inlined so the error message and
#: the check cannot drift apart.
_COLUMNS = 4


@dataclass(frozen=True)
class SourceLine:
    energy: float
    energy_err: float
    intensity: float
    intensity_err: float


def load_sou(path):
    """Every line of `path` as a SourceLine, in file order.

    Raises ParseError, naming the line, for a wrong column count, a token
    that is not a number, a non-finite value, a non-positive energy, an
    empty file, or bytes that are not UTF-8. Order is preserved rather
    than sorted: the reader reports what the file says.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except UnicodeDecodeError as exc:
        raise ParseError(f"{path} is not valid UTF-8: {exc}") from None
    except OSError as exc:
        raise ParseError(f"Cannot read {path}: {exc}") from None

    lines = []
    for number, raw in enumerate(text.splitlines(), start=1):
        tokens = raw.split()
        if not tokens:
            continue
        if len(tokens) != _COLUMNS:
            raise ParseError(
                f"{path} line {number}: expected {_COLUMNS} columns "
                f"(energy, error, intensity, error), found {len(tokens)}"
            )
        values = []
        for token in tokens:
            try:
                value = float(token)
            except ValueError:
                raise ParseError(
                    f"{path} line {number}: {token!r} is not a number"
                ) from None
            # float() accepts "nan" and "inf"; either would pass through
            # to the matcher and compare as garbage without complaint.
            if not math.isfinite(value):
                raise ParseError(
                    f"{path} line {number}: {token!r} is not a finite number"
                )
            values.append(value)
        energy = values[0]
        if energy <= 0.0:
            raise ParseError(
                f"{path} line {number}: energy must be positive, got {energy!r}"
            )
        lines.append(SourceLine(*values))

    if not lines:
        raise ParseError(f"{path} contains no source lines")
    return lines


def match_line(energy, lines):
    """The source line that `energy` unambiguously belongs to, or None.

    "Unambiguously" means the nearest line is closer than HALF the
    distance to the runner-up. That is the whole rule, and it has no
    tunable constant: it scales itself to every source, suggesting freely
    where lines are sparse (Y-88's two lines, 900 keV apart) and holding
    back where they crowd (Eu-152's 22). A lone line is always
    unambiguous; duplicate energies never are, since the runner-up sits
    at the same distance as the nearest.

    The caller is expected to pass ALL of a source's lines, including
    ones already assigned: they still count as runners-up, which is what
    stops a peak far from every remaining line being handed the last one
    standing. Whether the winner is already taken is the caller's
    question, not this function's.
    """
    if not lines or not math.isfinite(energy):
        return None
    distances = sorted(
        (abs(line.energy - energy), index) for index, line in enumerate(lines)
    )
    nearest_distance, nearest_index = distances[0]
    if len(distances) == 1:
        return lines[nearest_index]
    runner_up_distance = distances[1][0]
    if nearest_distance * 2.0 < runner_up_distance:
        return lines[nearest_index]
    return None
