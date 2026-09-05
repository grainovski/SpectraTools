"""Calibration spectra built from real `.sou` source files, with the
calibration that made them known -- so a test can say outright whether
an assignment is right, which no real spectrum lets it do.

The shape is what a germanium detector records: a falling exponential
continuum, Gaussian lines whose width grows with channel, amplitudes in
proportion to the source's own relative intensities, and Poisson noise
on top. Realistic enough that the peak search and the fitter meet the
same difficulties they meet on real data -- unresolved close lines, weak
lines on a steep continuum, fits that run away -- which is what makes
the tests built on it able to fail.
"""

import os

import numpy as np

from sou_io import load_sou

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

#: sigma = SIGMA_AT_ZERO + SIGMA_PER_CHANNEL * channel, in channels. The
#: widening with channel matters: a rule that held every peak to one
#: width would pass on a spectrum where widths are constant and fail on
#: any real one.
SIGMA_AT_ZERO = 1.5
SIGMA_PER_CHANNEL = 0.0012


def fixture_sou(name):
    """Path of one of the tracked sample sources in tests/fixtures."""
    return os.path.join(FIXTURES, name)


def true_fwhm(channel):
    """The width a well-fitted peak at `channel` really has."""
    return 2.3548 * (SIGMA_AT_ZERO + SIGMA_PER_CHANNEL * channel)


def spectrum_from_source(sou_path, offset, gain, channels=8192, seed=1,
                         peak_counts=60000.0, continuum=2500.0):
    """Poisson counts for a spectrum of the nuclide in `sou_path`, laid
    out so that energy = offset + gain * channel.

    Lines that fall off the spectrum, or that would be too weak to see,
    are simply not drawn -- as in a real measurement.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(channels, dtype=float)
    clean = continuum * np.exp(-x / 1500.0) + 30.0
    for line in load_sou(sou_path):
        centre = (line.energy - offset) / gain
        if not (40 < centre < channels - 40):
            continue
        sigma = SIGMA_AT_ZERO + SIGMA_PER_CHANNEL * centre
        amplitude = peak_counts * (line.intensity / 10000.0)
        if amplitude < 3:
            continue
        clean = clean + amplitude * np.exp(-((x - centre) ** 2) / (2 * sigma * sigma))
    return rng.poisson(clean).astype(float)


def wrong_assignments(pairs, positions, offset, gain, tolerance_kev=3.0):
    """Indices of the (peak index, energy) pairs whose energy is not the
    line the peak at `positions[index]` truly is.

    Three keV: a correct assignment differs from the truth by the
    centroid's own noise, a fraction of a keV, while the failures this
    exists to count are ten to twenty keV off.
    """
    return [k for k, energy in pairs
            if abs((offset + gain * positions[k]) - energy) > tolerance_kev]
