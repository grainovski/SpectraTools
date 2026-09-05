"""Restoring stored energy assignments onto a freshly refitted spectrum."""

import pytest

from energy_assignments import EnergyAssignments, restore


def test_exact_channels_are_restored():
    stored = EnergyAssignments(source_path="eu152.sou",
                               pairs=((100.0, 121.783), (300.0, 344.276)))
    got = restore(stored, [(100.0, 4.0), (200.0, 4.0), (300.0, 4.0)])
    assert got == {0: 121.783, 2: 344.276}


def test_a_small_shift_from_refitting_still_matches():
    """A refit moves a centroid by far less than its own width."""
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    got = restore(stored, [(100.4, 4.0)])
    assert got == {0: 121.783}


def test_a_shift_larger_than_the_peak_width_does_not_match():
    """Past its own width it is a different peak, and a wrong energy
    silently restored would look like a valid calibration."""
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    got = restore(stored, [(106.0, 4.0)])
    assert got == {}


def test_the_boundary_is_the_full_fwhm():
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    assert restore(stored, [(104.0, 4.0)]) == {0: 121.783}
    assert restore(stored, [(104.001, 4.0)]) == {}


def test_two_rows_never_share_one_assignment():
    """The nearer row takes it; the other opens blank.

    Separated clearly enough to decide: 3.4 channels apart against a
    tolerance of 4.0, so the runner-up is not within _AMBIGUITY_MARGIN
    of the winner and the match is made.
    """
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    got = restore(stored, [(100.1, 4.0), (103.5, 4.0)])
    assert got == {0: 121.783}


def test_an_ambiguous_pair_is_left_blank_rather_than_guessed():
    """Two peaks 1.3 channels apart with a FWHM of 8 are 0.16 x FWHM
    apart -- not separable, so which one owns a stored line is a coin
    flip. Guessing produces a calibration that passes through every
    assigned point and is wrong; a blank costs one retyped energy.

    This is a deliberate change from the greedy behaviour, which handed
    the line to whichever peak happened to be nearer.
    """
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    assert restore(stored, [(101.5, 8.0), (100.2, 8.0)]) == {}


def test_a_row_exactly_on_the_stored_channel_is_not_a_coin_flip():
    """The automatic calibration stores the very centroids it fitted,
    so the row at distance zero IS the stored peak. A twin 0.01 channels
    away -- the same peak fitted again by hand -- must not turn that into
    an ambiguity that blanks both rows."""
    stored = EnergyAssignments(None, ((829.45, 344.276),))
    assert restore(stored, [(829.45, 5.8), (829.46, 5.8)]) == {0: 344.276}
    # Off by any amount at all and the coin-flip rule is back.
    assert restore(stored, [(829.451, 5.8), (829.46, 5.8)]) == {}


def test_each_row_takes_its_own_nearest_assignment():
    stored = EnergyAssignments(
        None, ((100.0, 121.783), (200.0, 344.276))
    )
    got = restore(stored, [(100.1, 4.0), (200.1, 4.0)])
    assert got == {0: 121.783, 1: 344.276}


def test_no_stored_assignments_restores_nothing():
    assert restore(None, [(100.0, 4.0)]) == {}
    assert restore(EnergyAssignments(None, ()), [(100.0, 4.0)]) == {}


def test_an_unusable_width_falls_back_to_a_fixed_tolerance():
    """A peak whose FWHM the fit could not determine still deserves a
    chance to match, but a narrow one."""
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    assert restore(stored, [(100.2, float("nan"))]) == {0: 121.783}
    assert restore(stored, [(103.0, float("nan"))]) == {}


def test_the_spectrum_starts_with_no_assignments():
    import numpy as np

    from spectrum import LoadedSpectrum

    spectrum = LoadedSpectrum("x.txt", np.zeros(10), "#fff")
    assert spectrum.energy_assignments is None


# --- v5.0.1: optimal matching, and refusing to guess ---------------------


def test_no_misassignment_once_the_peaks_are_cleanly_resolved():
    """The guarantee v5.0.1 actually delivers, pinned over a fixed-seed
    sweep with a DIFFERENT width per peak -- which is what real fits
    produce, and what the first version of this sweep got wrong by
    giving every peak in a trial the same FWHM.

    From 1.5 x FWHM apart the restore is never wrong. Below that it is
    reduced, not eliminated: the greedy version this replaced scored
    16,453 misassignments over these same inputs against 1,523 now, but
    "cleanly resolved" is the only band where zero is guaranteed.
    """
    import random

    rng = random.Random(2024)
    wrong = restored = 0
    for _ in range(4000):
        count = rng.randint(2, 5)
        ratio = rng.choice([1.5, 4.0])           # cleanly resolved only
        base = rng.uniform(2.0, 9.0)
        fwhms = [base * rng.uniform(0.5, 2.0) for _ in range(count)]
        truth, x = [], 200.0
        for width in fwhms:
            truth.append(x)
            x += width * ratio
        energies = [100.0 + 37.0 * i for i in range(count)]
        peaks = [(c + rng.uniform(-f * 0.45, f * 0.45), f)
                 for c, f in zip(truth, fwhms)]
        got = restore(EnergyAssignments(None, tuple(zip(truth, energies))), peaks)
        restored += len(got)
        wrong += sum(1 for row, e in got.items() if e != energies[row])

    assert wrong == 0, f"{wrong} energies landed on the wrong peak"
    # A guard that restored nothing would also report zero wrong.
    assert restored > 10000, f"only {restored} restored -- the sweep is not exercising the path"


def test_a_tight_doublet_is_reduced_but_not_guaranteed():
    """Stated so nobody reads the test above as a promise it does not
    make. Below one FWHM the pairing is improved and still fallible;
    the honest bound there is a rate, not zero."""
    import random

    rng = random.Random(7)
    wrong = total = 0
    for _ in range(4000):
        count = rng.randint(2, 5)
        base = rng.uniform(2.0, 9.0)
        fwhms = [base * rng.uniform(0.5, 2.0) for _ in range(count)]
        truth, x = [], 200.0
        for width in fwhms:
            truth.append(x)
            x += width * 0.8
        energies = [100.0 + 37.0 * i for i in range(count)]
        peaks = [(c + rng.uniform(-f * 0.45, f * 0.45), f)
                 for c, f in zip(truth, fwhms)]
        got = restore(EnergyAssignments(None, tuple(zip(truth, energies))), peaks)
        total += count
        wrong += sum(1 for row, e in got.items() if e != energies[row])

    rate = wrong / total
    assert rate < 0.02, f"misassignment at 0.8 x FWHM rose to {rate:.2%}"



def test_restore_returns_plain_int_row_numbers():
    """They index Qt rows and get compared against range(); numpy's
    int64 works until something asks for an exact type."""
    peaks = [(100.0, 4.0), (300.0, 4.0)]
    out = restore(EnergyAssignments("s.sou", ((100.0, 121.78), (300.0, 344.28))), peaks)
    assert out
    for row in out:
        assert type(row) is int
