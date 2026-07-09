import struct

import numpy as np

from histogram_io import ParseError

RECORD1_PAYLOAD_SIZE = 24


def load_spe(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 4:
        raise ParseError(f"File too small to be a valid .spe file: {path}")

    # No explicit endianness field -- detect it from record 1's leading
    # length marker, which must equal the fixed 24-byte record-1 payload
    # size regardless of byte order.
    if struct.unpack_from("<i", data, 0)[0] == RECORD1_PAYLOAD_SIZE:
        endian = "<"
    elif struct.unpack_from(">i", data, 0)[0] == RECORD1_PAYLOAD_SIZE:
        endian = ">"
    else:
        raise ParseError(f"Not a valid .spe file (unexpected record marker): {path}")

    header_end = 4 + RECORD1_PAYLOAD_SIZE + 4
    if len(data) < header_end:
        raise ParseError(f"File too short for a complete .spe header: {path}")

    trailing1 = struct.unpack_from(endian + "i", data, 4 + RECORD1_PAYLOAD_SIZE)[0]
    if trailing1 != RECORD1_PAYLOAD_SIZE:
        raise ParseError(f"Corrupt .spe file (record 1 marker mismatch): {path}")

    # name(8s), idim1, idim2, ired1, ired2 -- only idim1 (channel count) is
    # actually used; the rest are read to advance past them but otherwise
    # unvalidated (see design spec).
    _name, idim1, _idim2, _ired1, _ired2 = struct.unpack_from(endian + "8s4i", data, 4)

    if idim1 <= 0:
        raise ParseError(f"Invalid channel count in .spe file: {path}")

    record2_offset = header_end
    expected_payload = idim1 * 4
    data_start = record2_offset + 4
    data_end = data_start + expected_payload

    if len(data) < data_end + 4:
        raise ParseError(f"File too short for .spe record 2 data: {path}")

    record2_leading = struct.unpack_from(endian + "i", data, record2_offset)[0]
    if record2_leading != expected_payload:
        raise ParseError(f"Corrupt .spe file (record 2 marker mismatch): {path}")

    record2_trailing = struct.unpack_from(endian + "i", data, data_end)[0]
    if record2_trailing != expected_payload:
        raise ParseError(f"Corrupt .spe file (record 2 trailing marker mismatch): {path}")

    dtype = np.dtype(endian + "f4")
    channels = np.frombuffer(data, dtype=dtype, count=idim1, offset=data_start)
    return np.round(channels).astype(np.int64)
