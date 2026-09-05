"""Finding peaks in a spectrum, and choosing regions to fit them in.

Everything else in this application fits peaks the user has already
marked. This module is the missing first step: given only the counts, say
where the peaks are, how wide they are, and which of them are close
enough that they must be fitted together rather than one at a time.

Qt-free and pure, like peak_fit itself, so the search can be tested
against synthetic spectra with a known answer.

## What counts as a peak

Prominence divided by the noise it stands in. A peak's prominence is its
height above the higher of the two saddles bounding it, which is already
a local measurement and needs no separate continuum model. The noise on
a channel holding N counts is sqrt(N), so

    significance = prominence / sqrt(max(continuum, 1))

is the familiar "how many sigma above the background" and means the same
thing at 100 counts and at 100000. A fixed count threshold does not: it
would find nothing in a weak spectrum and everything in a strong one.
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

#: Default significance a peak must reach to be reported. Three sigma is
#: too loose here -- a 16k-channel spectrum has enough channels that
#: three-sigma noise fluctuations are common -- and eight is strict
#: enough to miss real calibration lines. Five is the usual compromise
#: in gamma spectroscopy and finds the lines a person would point at.
DEFAULT_SENSITIVITY = 5.0

#: Channels smoothed over before searching. Wide enough to stop single-
#: channel noise spikes registering as peaks, narrow enough not to merge
#: a close doublet. The position and width are read off the smoothed
#: copy too -- a half-maximum crossing on raw Poisson counts is anyone's
#: guess -- and are within a quarter of the truth on synthetic peaks,
#: which is all grouping and region placement need; the fit that follows
#: measures them properly. Only the height is read from the raw counts.
_SMOOTH_WINDOW = 9
_SMOOTH_ORDER = 2

#: Peaks closer than this many FWHM are fitted together. A doublet fitted
#: as two independent single peaks gets both areas and both centroids
#: wrong, because each fit treats the other's tail as background.
GROUP_WITHIN_FWHM = 3.0

#: The fit region spans this many FWHM either side of the outermost peak
#: in a group, and the background regions sit between the inner and outer
#: multiples beyond it.
_FIT_HALF_WIDTH_FWHM = 3.0
_BG_INNER_FWHM = 4.0
_BG_OUTER_FWHM = 6.5

#: A width narrower than this is a noise spike, not a photopeak; wider
#: than this at any realistic gain is a step or a broad structure.
_MIN_FWHM = 1.2
_MAX_FWHM = 60.0


class FoundPeak:
    """One candidate, in channels."""

    __slots__ = ("channel", "fwhm", "prominence", "significance", "height")

    def __init__(self, channel, fwhm, prominence, significance, height):
        self.channel = channel
        self.fwhm = fwhm
        self.prominence = prominence
        self.significance = significance
        self.height = height

    def __repr__(self):
        return (f"FoundPeak(channel={self.channel:.2f}, fwhm={self.fwhm:.2f}, "
                f"significance={self.significance:.1f})")


def _smoothed(counts):
    """Savitzky-Golay keeps a peak's height while removing the spikes;
    a plain moving average would flatten the very features being looked
    for. Falls back to the raw counts for a spectrum too short to
    filter."""
    if len(counts) < _SMOOTH_WINDOW:
        return np.asarray(counts, dtype=float)
    return savgol_filter(np.asarray(counts, dtype=float),
                         _SMOOTH_WINDOW, _SMOOTH_ORDER)


def search(counts, sensitivity=DEFAULT_SENSITIVITY):
    """Candidate peaks in `counts`, weakest first removed, sorted by
    channel.

    `sensitivity` is the number of standard deviations a peak must stand
    above the continuum around it. Lower finds more.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.size < _SMOOTH_WINDOW or not np.any(counts > 0):
        return []

    smooth = _smoothed(counts)
    # A tiny absolute floor keeps find_peaks from returning every ripple
    # in an empty region before the significance test can see them.
    indices, props = find_peaks(smooth, prominence=1.0)
    if indices.size == 0:
        return []

    widths, _h, left_ips, right_ips = peak_widths(smooth, indices, rel_height=0.5)
    prominences = props["prominences"]
    # The continuum under a peak is its height less its prominence: what
    # would be there if the peak were removed. Clamped at 1 so a peak
    # standing on nothing does not divide by zero.
    continuum = np.maximum(smooth[indices] - prominences, 1.0)
    significance = prominences / np.sqrt(continuum)

    found = []
    for i, index in enumerate(indices):
        fwhm = float(widths[i])
        if not (_MIN_FWHM <= fwhm <= _MAX_FWHM):
            continue
        if significance[i] < sensitivity:
            continue
        found.append(FoundPeak(
            channel=float(index),
            fwhm=fwhm,
            prominence=float(prominences[i]),
            significance=float(significance[i]),
            height=float(counts[index]),
        ))
    return found


def group(peaks, within=GROUP_WITHIN_FWHM):
    """Split `peaks` into groups that must be fitted together.

    Two peaks belong together when the gap between them is smaller than
    `within` times the wider one's FWHM -- at that separation each sits
    inside the other's tail, and fitting them separately makes each
    treat the other as background.
    """
    if not peaks:
        return []
    ordered = sorted(peaks, key=lambda p: p.channel)
    groups = [[ordered[0]]]
    for peak in ordered[1:]:
        previous = groups[-1][-1]
        reach = within * max(previous.fwhm, peak.fwhm)
        if peak.channel - previous.channel <= reach:
            groups[-1].append(peak)
        else:
            groups.append([peak])
    return groups


def regions(peak_group, channel_count):
    """(bg_left, bg_right, fit_region, positions) for one group, or None
    when the group sits too close to an edge for a background region to
    fit beside it.

    Refusing is deliberate: a fit whose background region is clipped to
    the spectrum edge reports a slope fitted to almost nothing, and the
    peak areas that follow inherit it.
    """
    widest = max(p.fwhm for p in peak_group)
    first = min(p.channel for p in peak_group)
    last = max(p.channel for p in peak_group)

    fit_lo = first - _FIT_HALF_WIDTH_FWHM * widest
    fit_hi = last + _FIT_HALF_WIDTH_FWHM * widest
    bg_left = (first - _BG_OUTER_FWHM * widest, first - _BG_INNER_FWHM * widest)
    bg_right = (last + _BG_INNER_FWHM * widest, last + _BG_OUTER_FWHM * widest)

    if bg_left[0] < 0 or bg_right[1] > channel_count - 1:
        return None
    return bg_left, bg_right, (fit_lo, fit_hi), [p.channel for p in peak_group]
