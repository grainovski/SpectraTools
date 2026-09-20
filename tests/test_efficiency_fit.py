"""Our efficiency fit against CalEnEff's, on CalEnEff's own data.

This is an EXTERNAL oracle, which is rare and is the reason to trust the
rest of the feature. The expected numbers below were produced by running
the reference engine (ra226_gui.CalibrationEngine) on these same three
files; they are not our own output fed back in.

What is compared, and why in this order:

  curve values  The thing that actually matters. Fit parameters can trade
                against each other, so two genuinely equivalent fits can
                carry different parameters; the curve they describe is what
                the spectrum correction divides by.
  chi-squared   The quantity being minimised, robust to that degeneracy.
  parameters    Loosest of the three, and informational -- it catches gross
                divergence without failing on a harmless reparameterisation.
"""

import os

import numpy as np
import pytest

from efficiency import fit_efficiency

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "caleneff")

#: Produced by the reference implementation, 2026-09-20.
GOLDEN = {
    "226Ra_En_Area.txt": {
        "n": 23,
        "kfr": [6.274371440450e-01, 7.844365398647e+05,
                -7.258867102388e-04, -1.467068952279e+02],
        "kfr_chi2": 117.5085777268, "kfr_ndf": 19,
        "rw": [-3.765575465541e+01, 7.438801254775e+01, 6.386965185696e+00,
               -7.193596739758e-01, -2.424310667484e-02],
        "rw_chi2": 97.8603554131, "rw_ndf": 18,
    },
    "demo1.txt": {
        "n": 9,
        "kfr": [1.138577308963e+03, 2.962038674033e+07,
                -5.801324494336e-03, -1.015250508519e+02],
        "kfr_chi2": 3.9694346123, "kfr_ndf": 5,
        "rw": [12.192811084709, -0.257477640308, 8.924249211442,
               -2.565507279293, -0.599464105736],
        "rw_chi2": 2.9281520555, "rw_ndf": 4,
    },
    "demo2.txt": {
        "n": 19,
        "kfr": [8.896741862449e+00, 7.089236616431e+06,
                -1.081881901237e-03, -1.160635814659e+02],
        "kfr_chi2": 47.6128442300, "kfr_ndf": 15,
        "rw": [3.012598062741e+00, 3.897772490500e+01, 8.473633125424e+00,
               -8.688228294051e-01, -3.357195834004e-02],
        "rw_chi2": 42.1858733891, "rw_ndf": 14,
    },
}


def _load(name):
    """The reference's 7-column format: ch dch N dN E I dI."""
    data = np.loadtxt(os.path.join(FIXTURES, name), ndmin=2)
    return data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_our_fit_reproduces_caleneff(name):
    from efficiency import f_kfr, f_radware_5p

    want = GOLDEN[name]
    N, dN, E, I, dI = _load(name)
    assert len(E) == want["n"]

    got = fit_efficiency(E, N, dN, I, dI)

    # 1. The curves agree where it matters.
    for f, key in ((f_kfr, "kfr"), (f_radware_5p, "rw")):
        ours = f(E, *getattr(got, key + "_params"))
        theirs = f(E, *want[key])
        assert ours == pytest.approx(theirs, rel=1e-6), (
            "%s: our %s curve differs from CalEnEff's at the data energies"
            % (name, key)
        )

    # 2. The minimised quantity agrees.
    assert got.kfr_chi2 == pytest.approx(want["kfr_chi2"], rel=1e-6)
    assert got.rw_chi2 == pytest.approx(want["rw_chi2"], rel=1e-6)
    assert got.kfr_ndf == want["kfr_ndf"]
    assert got.rw_ndf == want["rw_ndf"]

    # 3. Parameters, loosely -- degeneracy makes this the weakest check.
    assert np.asarray(got.kfr_params) == pytest.approx(
        np.asarray(want["kfr"]), rel=1e-4)
    assert np.asarray(got.rw_params) == pytest.approx(
        np.asarray(want["rw"]), rel=1e-4)


def test_the_oracle_can_fail():
    """Control. Without this the comparison above would pass just as happily
    against a curve that is wrong everywhere, if approx were mis-set or the
    golden values were accidentally derived from our own output."""
    from efficiency import f_kfr

    want = GOLDEN["226Ra_En_Area.txt"]
    _, _, E, _, _ = _load("226Ra_En_Area.txt")
    wrong = list(want["kfr"])
    wrong[0] *= 1.10                      # a 10% error in one parameter
    assert f_kfr(E, *wrong) != pytest.approx(f_kfr(E, *want["kfr"]), rel=1e-6)


def test_birge_is_one_when_there_is_no_redundancy():
    """ndf <= 0 carries no scatter information, so the reference returns 1.0
    rather than dividing by zero. Reachable here: six points against
    Radware's five free parameters leaves ndf = 1, and five would leave 0."""
    from efficiency import birge

    assert birge(10.0, 0) == 1.0
    assert birge(10.0, -1) == 1.0
    assert birge(8.0, 2) == pytest.approx(2.0)


def test_efficiency_points_and_their_errors():
    """eps = N/I with the two relative errors added in quadrature."""
    from efficiency import efficiency_points

    N = np.array([1000.0]); dN = np.array([30.0])
    I = np.array([50.0]); dI = np.array([2.0])
    eff, deff = efficiency_points(N, dN, I, dI)
    assert eff[0] == pytest.approx(20.0)
    assert deff[0] == pytest.approx(
        20.0 * np.sqrt((30.0 / 1000.0) ** 2 + (2.0 / 50.0) ** 2))
