"""Finding peaks in a spectrum, and choosing regions to fit them in.

Everything else in this application fits peaks the user has already
marked. This module is the missing first step: given only the counts, say
where the peaks are, how wide they are, which of them are not photopeaks
at all, and where beside each one the background can honestly be read.

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

## Peaks are fitted one at a time

Each found peak gets its own fit, with its own markers: a fit window
around it alone, and two background windows chosen for that peak. This
replaced fitting nearby peaks together as a group, which on a real
Eu-152 spectrum chained the strongest line in the spectrum -- 121.78 keV
-- to a broad Compton feature 8 channels away whose spurious 23-channel
width gave the group a 68-channel reach; the four-peak fit that resulted
produced nothing usable and the strongest line went unassigned. Broad
features are now rejected before anything is fitted (see reject_broad),
and what remains is fitted individually.
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
#: which is all region placement needs; the fit that follows measures
#: them properly. Only the height is read from the raw counts.
_SMOOTH_WINDOW = 9
_SMOOTH_ORDER = 2

#: A found peak wider than this multiple of the width expected at its
#: channel is not a photopeak. On a real Eu-152 spectrum the genuine
#: peaks sat between 0.23 and 1.83 times the trend and the Compton edges,
#: backscatter bumps and steps between 2.4 and 11 times it -- eight of 76
#: candidates -- so two separates them with room on both sides.
BROAD_FEATURE_RATIO = 2.0

#: The fit window reaches this many widths either side of the peak, but
#: never past the midpoint to a neighbouring found peak: a neighbour
#: inside the window would be absorbed into this peak's area.
FIT_HALF_WIDTH_FWHM = 3.0
#: ...and never narrower than this, so a very close neighbour leaves
#: enough of the peak to fit. Below this the two are one blob to any fit.
MIN_FIT_HALF_WIDTH_FWHM = 1.2

#: Background windows: each is BG_WINDOW_FWHM wide, the search for one
#: starts BG_INNER_FWHM out from the peak (just clear of the widest fit
#: window) and gives up BG_MAX_REACH_FWHM out. A window is only
#: considered if it keeps PEAK_CLEARANCE_FWHM of every other found
#: peak's own width away from that peak -- broad features included, since
#: they are exactly what a background must not be read from.
#:
#: 1.5 widths of clearance, not more: a Gaussian is at 0.2% of its
#: height 1.5 FWHM from its centre, so beyond that a neighbour
#: contributes nothing a background window can see. And 1.5 widths of
#: window -- seven or eight channels on a typical HPGe spectrum -- is
#: enough to fix a straight line through. Both were larger at first
#: (2.0 and 2.5), and on a real Eu-152 spectrum with a peak every 54
#: channels that left 35 of 68 photopeaks with no clear window within
#: reach and unfitted.
BG_WINDOW_FWHM = 1.5
BG_INNER_FWHM = 3.25
BG_STEP_FWHM = 0.5
BG_MAX_REACH_FWHM = 25.0
PEAK_CLEARANCE_FWHM = 1.5

#: When no window within reach is clear of EVERY found peak, the search
#: tries again avoiding only the neighbours that would actually bias
#: this peak's background: broad features always, and peaks whose
#: prominence is at least this fraction of the target's. A neighbour a
#: twentieth as high perturbs a background window by less than the
#: window's own Poisson noise; refusing to fit the peak at all would lose
#: far more than admitting it.
WEAK_NEIGHBOUR_FRACTION = 0.05

#: A window whose flatness score is below this is taken as soon as it is
#: found, so backgrounds are read as near the peak as an honest reading
#: allows. Above it, the search continues and the best-scoring window in
#: reach is used. The score is a reduced chi-squared of a straight line
#: through the window plus a tilt term, so 1 is Poisson-flat and 3 means
#: something structured is in there.
FLAT_SCORE = 3.0

#: A width narrower than this is a noise spike, not a photopeak; wider
#: than this at any realistic gain is a step or a broad structure.
_MIN_FWHM = 1.2
_MAX_FWHM = 60.0


class FoundPeak:
    """One candidate, in channels. `broad` is set by reject_broad on a
    candidate judged not to be a photopeak; it is still kept in the list
    so that fit and background windows steer clear of it."""

    __slots__ = ("channel", "fwhm", "prominence", "significance", "height", "broad")

    def __init__(self, channel, fwhm, prominence, significance, height, broad=False):
        self.channel = channel
        self.fwhm = fwhm
        self.prominence = prominence
        self.significance = significance
        self.height = height
        self.broad = broad

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


# ------------------------------------------------------------- widths


def expected_widths(channels, fwhms):
    """The FWHM a well-fitted peak should have at each channel, from a
    robust straight line through the fitted widths.

    Resolution worsens with energy, so one width for the whole spectrum
    would be too loose at the bottom and too tight at the top -- on the
    spectra this was developed on the widths span a factor of seven from
    end to end, so a flat median would have put the lowest peaks at a
    third of it and the highest at nearly twice, where the outlier test
    would have thrown them out. A straight line is the right shape over
    that range.

    The line is Siegel's repeated median rather than least squares,
    because the widths this exists to set aside -- runaway components ten
    times too wide -- are precisely the points that would pull a
    least-squares line towards themselves. It survives up to half the
    peaks being junk; Theil-Sen's one third is not enough at four or five
    peaks, where a single outlier is already a quarter of them.

    Falls back to the median width when there are too few usable widths
    to define a slope, and never returns less than a fifth of the median,
    so a line steep enough to cross zero cannot leave a peak with no
    tolerance at all. A fifth and not a half: the lowest peaks of a real
    spectrum sit at a third of the median width, and a floor above them
    would loosen exactly the tolerances that need to be tightest. NaN
    where a channel is not finite, and everywhere when no width is
    usable.
    """
    channels = np.asarray(channels, dtype=float)
    fwhms = np.asarray(fwhms, dtype=float)
    good = np.isfinite(channels) & np.isfinite(fwhms) & (fwhms > 0.0)
    if not np.any(good):
        return np.full(channels.shape, np.nan)
    c, w = channels[good], fwhms[good]
    median = float(np.median(w))
    if c.size < 3 or np.ptp(c) == 0.0:
        return np.full(channels.shape, median)

    dc = c[:, None] - c[None, :]
    dw = w[:, None] - w[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(dc != 0.0, dw / dc, np.nan)
    row_medians = [np.nanmedian(row) for row in slopes if np.any(np.isfinite(row))]
    slope = float(np.median(row_medians))
    intercept = float(np.median(w - slope * c))
    return np.maximum(intercept + slope * channels, 0.2 * median)


def reject_broad(found, ratio=BROAD_FEATURE_RATIO):
    """(photopeaks, broad features): the found peaks split by width
    against the trend the spectrum as a whole establishes.

    A broad feature is not fitted and never assigned an energy, but it
    is still kept in view -- a background window must not be read from
    one, and a fit window must not reach into one -- which is why callers
    pass the FULL found list to `regions` and only the photopeaks to the
    fitter.

    With fewer than three candidates there is no trend to judge against
    and everything is kept.
    """
    if len(found) < 3:
        return list(found), []
    expected = expected_widths([p.channel for p in found], [p.fwhm for p in found])
    peaks, broad = [], []
    for peak, width in zip(found, expected):
        is_peak = np.isfinite(width) and peak.fwhm <= ratio * width
        peak.broad = not is_peak
        (peaks if is_peak else broad).append(peak)
    return peaks, broad


# ------------------------------------------------------------ regions


def fit_window(peak, found):
    """(lo, hi) around `peak`, clipped at the midpoint to any other found
    peak that would otherwise fall inside it."""
    half = FIT_HALF_WIDTH_FWHM * peak.fwhm
    lo, hi = peak.channel - half, peak.channel + half
    for other in found:
        if other is peak or other.channel == peak.channel:
            continue
        midpoint = 0.5 * (peak.channel + other.channel)
        if lo < other.channel < peak.channel:
            lo = max(lo, midpoint)
        elif peak.channel < other.channel < hi:
            hi = min(hi, midpoint)
    floor = MIN_FIT_HALF_WIDTH_FWHM * peak.fwhm
    return min(lo, peak.channel - floor), max(hi, peak.channel + floor)


def flatness(counts, lo, hi):
    """How much a window looks like a background: the reduced chi-squared
    of a straight line through it with Poisson weights, plus the line's
    relative change across the window.

    1 means the window is as flat as counting statistics allow; a hidden
    peak, a Compton edge or a step drives it up. Infinite for a window
    too short to fit a line through.
    """
    lo, hi = int(round(lo)), int(round(hi))
    y = np.asarray(counts[lo:hi + 1], dtype=float)
    if y.size < 4:
        return float("inf")
    x = np.arange(y.size, dtype=float)
    weight = 1.0 / np.maximum(y, 1.0)
    design = np.vstack([x, np.ones_like(x)]).T * np.sqrt(weight)[:, None]
    slope, intercept = np.linalg.lstsq(design, y * np.sqrt(weight), rcond=None)[0]
    residual = y - (slope * x + intercept)
    chi2 = float(np.sum(weight * residual * residual)) / (y.size - 2)
    tilt = abs(slope) * (y.size - 1) / max(float(np.mean(y)), 1.0)
    return chi2 + 2.0 * tilt


def _clear_window(counts, peak, direction, avoid):
    """The nearest window on one side of `peak` that is clear of every
    peak in `avoid` and flatter than FLAT_SCORE; failing that, the
    flattest clear window within reach; None when nothing clear fits
    between the peak and the spectrum edge."""
    counts = np.asarray(counts, dtype=float)
    width = BG_WINDOW_FWHM * peak.fwhm
    exclusions = [
        (other.channel - PEAK_CLEARANCE_FWHM * other.fwhm,
         other.channel + PEAK_CLEARANCE_FWHM * other.fwhm)
        for other in avoid
    ]
    best = None
    distance = BG_INNER_FWHM * peak.fwhm
    limit = BG_MAX_REACH_FWHM * peak.fwhm
    while distance <= limit:
        if direction < 0:
            hi = peak.channel - distance
            lo = hi - width
        else:
            lo = peak.channel + distance
            hi = lo + width
        if lo < 0 or hi > counts.size - 1:
            break
        distance += BG_STEP_FWHM * peak.fwhm
        if any(not (hi < a or lo > b) for a, b in exclusions):
            continue
        score = flatness(counts, lo, hi)
        if score < FLAT_SCORE:
            return lo, hi
        if best is None or score < best[0]:
            best = (score, (lo, hi))
    return None if best is None else best[1]


def background_window(counts, peak, direction, found):
    """(lo, hi) of a background window on one side of `peak`, or None.

    Searched outward from just beyond the fit window, in two tiers. First
    a window clear of EVERY other found peak, broad features included --
    a stretch where no peak was identified, which is what a background
    should be read from. Only when a crowded spectrum offers none within
    reach does the search relax to avoiding just the neighbours that
    would actually bias the reading (see WEAK_NEIGHBOUR_FRACTION). None
    when even that fails, in which case the peak is skipped rather than
    fitted against a background read from something that is not one.
    """
    others = [other for other in found if other is not peak]
    window = _clear_window(counts, peak, direction, others)
    if window is not None:
        return window
    matter = [other for other in others
              if other.broad
              or other.prominence >= WEAK_NEIGHBOUR_FRACTION * peak.prominence]
    return _clear_window(counts, peak, direction, matter)


def regions(peak, counts, found):
    """(bg_left, bg_right, fit_region, [position]) for fitting `peak`
    alone, or None when a background cannot be read on both sides.

    `found` is the FULL found list, broad features included: both the
    fit window and the background windows steer clear of everything in
    it. Refusing rather than clipping to the edge is deliberate -- a
    background region read from almost nothing reports a slope fitted to
    almost nothing, and the peak's area inherits it.
    """
    left = background_window(counts, peak, -1, found)
    right = background_window(counts, peak, +1, found)
    if left is None or right is None:
        return None
    return left, right, fit_window(peak, found), [peak.channel]
