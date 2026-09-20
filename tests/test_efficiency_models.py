"""The two efficiency curve shapes, ported from CalEnEff.

These are pure functions of energy and parameters. Everything later in the
feature -- the fit, the Monte Carlo, the correction applied to a spectrum --
is built on them being right, so they are pinned against values computed
directly from the published formulae rather than against our own output.
"""

import numpy as np
import pytest

from efficiency import f_kfr, f_radware, f_radware_5p, RADWARE_C, RADWARE_G


def test_kfr_matches_its_formula():
    """eps(E) = (aE + b/E) * exp(cE + d/E), evaluated by hand."""
    a, b, c, d = 0.5, 1000.0, -1e-3, -50.0
    E = 500.0
    expected = (a * E + b / E) * np.exp(c * E + d / E)
    assert f_kfr(E, a, b, c, d) == pytest.approx(expected, rel=1e-12)


def test_kfr_is_vectorised():
    E = np.array([100.0, 500.0, 1000.0])
    out = f_kfr(E, 0.5, 1000.0, -1e-3, -50.0)
    assert out.shape == E.shape
    assert np.all(np.isfinite(out))


def test_radware_5p_is_the_7p_model_with_c_and_g_fixed():
    """The 5-parameter form is not a different model -- it is the 7-parameter
    one with Radford's own two defaults held fixed. If these ever disagree,
    one of them has been edited in isolation."""
    E = np.array([100.0, 344.0, 1408.0])
    p = (-3.5, 1.5, -0.9, -0.5, -0.02)          # a1, a2, a4, a5, a6
    full = f_radware(E, p[0], p[1], RADWARE_C, p[2], p[3], p[4], RADWARE_G)
    assert f_radware_5p(E, *p) == pytest.approx(full, rel=1e-12)


def test_radware_uses_log_efficiencies_that_may_be_negative():
    """f1 and f2 are ln(eps) and are normally NEGATIVE. An earlier version of
    the reference clipped them positive with np.maximum(..., 1e-30), which
    made the loss landscape pathological and stalled the Monte Carlo. This
    pins the un-clipped behaviour: with strongly negative polynomials the
    model must return a small positive efficiency, not 1.0 or a constant."""
    E = np.array([100.0, 1000.0])
    out = f_radware_5p(E, -5.0, 1.0, -4.0, -0.5, 0.0)
    assert np.all(out > 0.0)
    assert np.all(out < 1.0), out
    assert out[0] != pytest.approx(out[1]), "clipped model would flatten"


def test_radware_survives_extreme_parameters():
    """The (1 + r**g)**(-1/g) evaluation overflows without the logaddexp form
    and the +-700 clip. The spectrum correction extrapolates this curve well
    outside the fitted range, so overflow here becomes inf counts there."""
    E = np.array([1.0, 1e5])
    out = f_radware_5p(E, 400.0, 50.0, -400.0, -50.0, 10.0)
    assert np.all(np.isfinite(out)), out


def test_a_constant_model_would_fail_these():
    """Control. Every assertion above must be able to fail -- a model that
    ignored its parameters would otherwise sail through the finiteness and
    positivity checks."""
    E = np.array([100.0, 1000.0])
    constant = np.ones_like(E)
    assert not np.allclose(f_kfr(E, 0.5, 1000.0, -1e-3, -50.0), constant)
    assert not np.allclose(f_radware_5p(E, -3.5, 1.5, -0.9, -0.5, -0.02), constant)
