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
    """The nearer row takes it; the other opens blank."""
    stored = EnergyAssignments(None, ((100.0, 121.783),))
    got = restore(stored, [(101.5, 8.0), (100.2, 8.0)])
    assert got == {1: 121.783}


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
