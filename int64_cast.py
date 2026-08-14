"""Shared int64-overflow guard for float channel/count data."""

import numpy as np


def checked_round_to_int64(rounded, make_error):
    """Casts an already-rounded float array to int64, raising the
    exception make_error() returns instead of silently wrapping to a
    meaningless sentinel value. numpy's .astype() does not raise on
    overflow, and checking np.abs(rounded) > np.iinfo(np.int64).max by
    hand is not a safe guard on its own -- that bound isn't exactly
    representable in float32/float64, so the comparison rounds it up to
    2**63, letting the genuinely out-of-range value 2**63 itself (and
    NaN, via IEEE754's always-false '>' comparison) silently slip
    through and still wrap to the sentinel. Delegating to numpy's own
    cast machinery under errstate(invalid="raise") avoids re-deriving
    int64's bounds by hand. `make_error` is a zero-arg callable
    returning the exception each caller wants raised, so callers keep
    their own exception type and message."""
    with np.errstate(invalid="raise"):
        try:
            return rounded.astype(np.int64)
        except FloatingPointError as exc:
            raise make_error() from exc
