"""Deciding which found peak is which source line, with no calibration.

This is the circular part of an automatic calibration: knowing a peak's
energy needs a calibration, and building the calibration needs the
energies. It is broken the way star trackers and point-cloud registration
break it -- guess a correspondence, see how much of the rest it explains,
and keep the guess that explains the most.

Two peaks paired with two source lines fix a straight line exactly. Every
such pairing is tried; each one predicts an energy for every other peak;
a peak whose prediction lands on a real line within the spectrum's
resolution is an inlier. The pairing with the most inliers wins, ties
broken by the tighter residual.

Only the STRONGEST peaks and lines are used to propose the pairing, since
trying every combination of everything is needless -- the anchors only
have to be right, and a strong peak in a calibration spectrum is far more
likely to be a strong line of the source than a contaminant. Verification
then runs against every peak and every line.

Qt-free and pure, so a match can be tested against a spectrum whose
answer is known.
"""

import math

import numpy as np

import peak_search

#: Strongest peaks and lines considered when PROPOSING a pairing. The
#: cost is quadratic in each, and 10 x 15 is 4725 candidate lines --
#: fast, and enough that the true pairing is among them whenever the
#: source dominates the spectrum, which is what a calibration source is
#: for. Verification is never restricted this way.
ANCHOR_PEAKS = 10
ANCHOR_LINES = 15

#: How near a predicted energy must fall to a real line to count as an
#: inlier, in units of the FWHM expected at that channel converted to
#: energy. Half a width is generous enough to absorb the nonlinearity a
#: straight line cannot follow, and tight enough that a wrong pairing
#: does not collect inliers by accident.
INLIER_FWHM = 0.5

#: A peak whose fitted width exceeds this multiple of the width expected
#: at its channel is set aside: not an anchor, not an inlier, never
#: assigned. Real widths scatter a few tens of percent about the trend;
#: the fits this exists for were ten times it. Three rather than two
#: because widths are shared within a fit: the real half of a doublet
#: whose other half ran away inherits its partner's width, and at twice
#: the trend it was being thrown out along with the junk, while the junk
#: itself never came in under nine.
#:
#: The failure this guards, seen on every synthetic spectrum the matcher
#: was developed on: one to three assignments per spectrum wrong by 10 to
#: 18 keV, each a fit component that had run away to a width of 120-150
#: channels, absorbing background in a crowded region. The inlier
#: tolerance used to be that peak's OWN fitted width, so a bad fit bought
#: itself a tolerance of 20 keV and was matched to a line that far away.
#: Now the tolerance comes from the resolution the spectrum as a whole
#: establishes (see expected_widths), and a width this far outside it
#: disqualifies the peak altogether.
WIDTH_OUTLIER_RATIO = 3.0

#: Fewer inliers than this is not a calibration, it is a coincidence.
#: Three points can be fitted by a quadratic exactly, so four is the
#: smallest number that can disagree with the fit and thereby support it.
MIN_INLIERS = 4

#: A rival solution whose gain differs by more than this is a genuinely
#: DIFFERENT calibration, not a refinement of the winner. If such a rival
#: explains as many peaks, the evidence does not choose between them and
#: the match is refused.
#:
#: This is the failure this guard exists for, seen while developing it: a
#: spectrum yielding only four peaks was "matched" with all four
#: assignments wrong and an rms of 0.5 keV, because two of the four were
#: the anchors and the other two confirmed nothing that a coincidence
#: could not. A wrong calibration passes through every point it was
#: fitted to, so nothing downstream would have questioned it.
AMBIGUOUS_GAIN_RATIO = 0.05

#: Confirmations required BEYOND the two anchor peaks. The anchors are
#: fitted exactly by construction and so are not evidence for anything.
MIN_CONFIRMATIONS = 2

#: The strongest peaks in a calibration spectrum ARE the source's lines
#: -- that is what a calibration source is. So an assignment that leaves
#: most of them unexplained has found a coincidence among the weak ones,
#: however self-consistent it looks.
#:
#: The failure this catches, seen while developing it: a Co-56 spectrum
#: offered the Am-241 file matched 5 of 24 peaks at a gain of 0.028
#: instead of 0.5, every assignment wrong. Am-241's eight lines all lie
#: below 103 keV, so a small enough gain lets them drift across the
#: spectrum until some weak peaks happen to sit near them. None of the
#: strong peaks was among them.
#:
#: Three of five rather than five of five leaves room for a genuine
#: contaminant -- a background K-40 or Tl-208 line is often among the
#: strongest peaks present and belongs to no calibration source -- and
#: for a strong doublet the fitter could not resolve.
STRONG_PEAKS_CHECKED = 5
STRONG_PEAKS_REQUIRED = 3

#: Sanity bounds on the provisional straight line. A gain outside this
#: range or an offset this large is not a spectrometer, it is a pairing
#: that happens to be arithmetically consistent.
_MIN_GAIN, _MAX_GAIN = 1e-3, 1e3
_MAX_OFFSET = 500.0


class MatchResult:
    """What the search concluded.

    `pairs` is [(peak index, energy)] -- indices into the peaks list as
    given. `reason` is set only when the search failed, and says why in
    words meant for the status line. `unusable` counts the peaks set
    aside before matching for an implausible width or area; they are
    never assigned, and the status line should say so rather than let
    them pass as merely unidentified.
    """

    __slots__ = ("pairs", "gain", "offset", "rms", "reason", "considered",
                 "unusable")

    def __init__(self, pairs=(), gain=0.0, offset=0.0, rms=float("inf"),
                 reason=None, considered=0, unusable=0):
        self.pairs = list(pairs)
        self.gain = gain
        self.offset = offset
        self.rms = rms
        self.reason = reason
        self.considered = considered
        self.unusable = unusable

    @property
    def ok(self):
        return self.reason is None and len(self.pairs) >= MIN_INLIERS

    def __repr__(self):
        if self.reason:
            return f"MatchResult(failed: {self.reason})"
        return (f"MatchResult({len(self.pairs)} pairs, gain={self.gain:.5g}, "
                f"offset={self.offset:.4g}, rms={self.rms:.3g} keV)")


def _strongest(values, count):
    """Indices of the `count` largest, in their original order."""
    if len(values) <= count:
        return list(range(len(values)))
    keep = np.argsort(np.asarray(values, dtype=float))[-count:]
    return sorted(int(i) for i in keep)


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


def _usable(fwhms, weights, expected):
    """Which peaks may take part at all.

    A width the fit could not determine, a width far outside the trend,
    or a weight that is not positive -- a fitted area that came out
    negative is a component modelling background, not a line -- each
    mean the centroid is not the centroid of a peak.
    """
    return (np.isfinite(fwhms) & (fwhms > 0.0)
            & np.isfinite(weights) & (weights > 0.0)
            & np.isfinite(expected) & (fwhms <= WIDTH_OUTLIER_RATIO * expected))


def _inliers(channels, tolerances, energies, gain, offset):
    """Greedy one-to-one nearest matching under a provisional line.

    Greedy rather than optimal on purpose: this runs thousands of times,
    and it only has to RANK pairings. The winner is re-matched properly
    afterwards, where being exact is worth the cost.

    A negative tolerance means the peak is not to be matched at all.
    """
    predicted = offset + gain * channels
    order = np.argsort(np.abs(predicted - energies[np.clip(
        np.searchsorted(energies, predicted), 0, len(energies) - 1)]))
    taken_lines, pairs, total = set(), [], 0.0
    for k in order:
        target = predicted[k]
        j = int(np.searchsorted(energies, target))
        # nearest line, checking the neighbour on each side
        best_j, best_d = -1, float("inf")
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < len(energies) and cand not in taken_lines:
                d = abs(energies[cand] - target)
                if d < best_d:
                    best_j, best_d = cand, d
        if best_j < 0 or best_d > tolerances[k]:
            continue
        taken_lines.add(best_j)
        pairs.append((int(k), best_j))
        total += best_d * best_d
    rms = math.sqrt(total / len(pairs)) if pairs else float("inf")
    return pairs, rms


def match(peak_channels, peak_fwhms, peak_weights, line_energies,
          line_intensities=None, inlier_fwhm=INLIER_FWHM):
    """Assign source lines to peaks with no prior calibration.

    `peak_weights` ranks peaks for anchor selection -- area or height.
    A peak whose weight is not positive is never assigned; see _usable.
    `line_intensities` does the same for lines; without it the first
    ANCHOR_LINES are used.

    Returns a MatchResult. Failure is reported, never guessed at: a
    calibration built on a wrong pairing looks entirely reasonable, since
    it passes through every point it was fitted to.
    """
    channels = np.asarray(peak_channels, dtype=float)
    fwhms = np.asarray(peak_fwhms, dtype=float)
    weights = np.asarray(peak_weights, dtype=float)
    energies = np.asarray(sorted(line_energies), dtype=float)

    if channels.size < MIN_INLIERS:
        return MatchResult(reason=(
            f"only {channels.size} peak(s) were fitted; at least "
            f"{MIN_INLIERS} are needed to establish a calibration"))
    if energies.size < 2:
        return MatchResult(reason="the source file has fewer than two lines")

    expected = expected_widths(channels, fwhms)
    usable = _usable(fwhms, weights, expected)
    usable_count = int(np.count_nonzero(usable))
    unusable = int(channels.size - usable_count)
    if usable_count < MIN_INLIERS:
        return MatchResult(reason=(
            f"only {usable_count} of {channels.size} fitted peaks have a "
            f"plausible width and area; at least {MIN_INLIERS} are needed "
            f"to establish a calibration"), unusable=unusable)

    # The tolerance in channels, before the gain is known. From the
    # resolution the spectrum as a whole establishes, never from the
    # peak's own width -- see WIDTH_OUTLIER_RATIO for the failure that
    # taught this. Negative for a peak that must not be matched.
    unit_tolerance = np.where(usable, expected * inlier_fwhm, -1.0)

    anchor_p = [i for i in _strongest(np.where(usable, weights, -np.inf), ANCHOR_PEAKS)
                if usable[i]]
    if line_intensities is None:
        anchor_l = list(range(min(ANCHOR_LINES, energies.size)))
    else:
        order = np.argsort(np.asarray(line_energies, dtype=float))
        sorted_int = np.asarray(line_intensities, dtype=float)[order]
        anchor_l = _strongest(sorted_int, ANCHOR_LINES)

    best = MatchResult(reason="no consistent assignment was found")
    # Best solution whose gain is incompatible with the winner's, kept so
    # an unresolvable tie can be recognised rather than broken arbitrarily.
    rival = MatchResult()
    considered = 0
    for ai in range(len(anchor_p)):
        for aj in range(ai + 1, len(anchor_p)):
            i, j = anchor_p[ai], anchor_p[aj]
            if channels[j] == channels[i]:
                continue
            lo_c, hi_c = (i, j) if channels[i] < channels[j] else (j, i)
            for bi in range(len(anchor_l)):
                for bj in range(bi + 1, len(anchor_l)):
                    p, q = anchor_l[bi], anchor_l[bj]
                    lo_e, hi_e = (p, q) if energies[p] < energies[q] else (q, p)
                    span_c = channels[hi_c] - channels[lo_c]
                    span_e = energies[hi_e] - energies[lo_e]
                    if span_c <= 0 or span_e <= 0:
                        continue
                    gain = span_e / span_c
                    if not (_MIN_GAIN <= gain <= _MAX_GAIN):
                        continue
                    offset = energies[lo_e] - gain * channels[lo_c]
                    if abs(offset) > _MAX_OFFSET:
                        continue
                    considered += 1
                    tolerances = gain * unit_tolerance
                    pairs, rms = _inliers(channels, tolerances, energies,
                                          gain, offset)
                    better = (len(pairs) > len(best.pairs)
                              or (len(pairs) == len(best.pairs) and rms < best.rms))
                    if better:
                        # the outgoing winner becomes a rival if its gain
                        # is incompatible with the new one
                        if (best.pairs and gain > 0 and best.gain > 0
                                and abs(best.gain - gain) / gain > AMBIGUOUS_GAIN_RATIO
                                and len(best.pairs) > len(rival.pairs)):
                            rival = best
                        best = MatchResult(
                            pairs=[(k, float(energies[m])) for k, m in pairs],
                            gain=gain, offset=offset, rms=rms, considered=considered)
                    elif (pairs and best.gain > 0
                          and abs(best.gain - gain) / best.gain > AMBIGUOUS_GAIN_RATIO
                          and (len(pairs) > len(rival.pairs)
                               or (len(pairs) == len(rival.pairs) and rms < rival.rms))):
                        rival = MatchResult(
                            pairs=[(k, float(energies[m])) for k, m in pairs],
                            gain=gain, offset=offset, rms=rms)

    best.considered = considered
    best.unusable = unusable
    if len(best.pairs) < MIN_INLIERS or len(best.pairs) < 2 + MIN_CONFIRMATIONS:
        return MatchResult(
            reason=(f"no assignment matched more than {len(best.pairs)} peak(s) "
                    f"to the source; the spectrum may not be this nuclide, or "
                    f"too few of its lines were found"),
            considered=considered, unusable=unusable)
    # Over every peak, usable or not: a strong peak set aside for its
    # width still counts against the match, since the ratio below
    # already allows for a couple of them.
    strongest = _strongest(peak_weights, STRONG_PEAKS_CHECKED)
    matched_rows = {k for k, _e in best.pairs}
    strong_hits = sum(1 for k in strongest if k in matched_rows)
    if strong_hits < min(STRONG_PEAKS_REQUIRED, len(strongest)):
        return MatchResult(
            reason=(f"only {strong_hits} of the {len(strongest)} strongest peaks "
                    f"could be identified; the spectrum does not look like this "
                    f"source"),
            considered=considered, unusable=unusable)
    if len(rival.pairs) >= len(best.pairs):
        return MatchResult(
            reason=(f"two different calibrations explain the spectrum equally "
                    f"well ({len(best.pairs)} peaks each, gains "
                    f"{best.gain:.4g} and {rival.gain:.4g} keV/channel); "
                    f"assign two peaks by hand to settle it"),
            considered=considered, unusable=unusable)
    return best


# --------------------------------------------------------------- fitting


class AutoFitOutcome:
    """What the automatic fitting pass produced.

    `results` are FitResults ready to commit to a spectrum, in channel
    order. `peaks` is (result index, PeakResult) flattened in the same
    order, which is what the matcher and the calibration dialog see.
    `attempted`/`failed` are for the status line: a fit that will not
    converge is normal in a crowded spectrum and must not stop the rest.
    """

    __slots__ = ("results", "peaks", "attempted", "failed", "skipped_edge")

    def __init__(self, results, peaks, attempted, failed, skipped_edge):
        self.results = results
        self.peaks = peaks
        self.attempted = attempted
        self.failed = failed
        self.skipped_edge = skipped_edge


def fit_found_peaks(x, y, found, link_widths=True, variance=None):
    """Fit every group of found peaks, skipping what will not fit.

    Groups rather than individual peaks: a doublet fitted as two separate
    single peaks gets both centroids and both areas wrong, because each
    fit treats the other peak's tail as background. peak_search.group
    decides what belongs together.

    A group whose fit raises is dropped and counted, never allowed to
    abort the pass -- one unconvergeable multiplet in a crowded region
    should not cost the user every other peak in the spectrum.

    `variance` is the spectrum's propagated per-channel variance, or None
    for counts read from a file, exactly as the interactive Fit passes
    it: a matrix cut or an Add/Subtract result is not Poisson in its own
    counts, and fitting it as though it were misweights every channel.
    """
    from peak_fit import FitError, fit_peaks

    results, peaks = [], []
    attempted = failed = skipped_edge = 0
    for cluster in peak_search.group(found):
        region = peak_search.regions(cluster, len(y))
        if region is None:
            skipped_edge += 1
            continue
        bg_left, bg_right, fit_region, positions = region
        attempted += 1
        try:
            result = fit_peaks(x, y, bg_left, bg_right, fit_region, positions,
                               link_widths=link_widths, variance=variance)
        except (FitError, ValueError, RuntimeError):
            failed += 1
            continue
        results.append(result)
        for peak in result.peaks:
            peaks.append((len(results) - 1, peak))
    return AutoFitOutcome(results, peaks, attempted, failed, skipped_edge)


# ------------------------------------------------------------ the whole run


def _plural(count, noun):
    return f"{count} {noun}{'' if count == 1 else 's'}"


class AutoCalibration:
    """Everything one automatic run produced: the peaks found, the fits
    made from them, and what the matcher made of those.

    The fits are kept whether or not the match succeeded -- they are
    real fits of real peaks, and the user can still assign them by hand.
    """

    __slots__ = ("found", "fits", "match")

    def __init__(self, found, fits, match):
        self.found = found
        self.fits = fits
        self.match = match

    @property
    def results(self):
        """FitResults to commit to the spectrum, in channel order."""
        return self.fits.results

    @property
    def peaks(self):
        """Every fitted PeakResult, in the order the matcher indexed."""
        return [peak for _index, peak in self.fits.peaks]

    @property
    def pairs(self):
        """[(channel, energy)] on the FITTED centroids -- what the
        calibration dialog restores from, and since these are the very
        centroids it will list, each lands on its own row exactly."""
        peaks = self.peaks
        return [(peaks[k].position, energy) for k, energy in self.match.pairs]

    def summary(self, existing_fits, source_name):
        """One line for the status bar, honest about every count: what
        was found, what could not be fitted, what was identified, and
        that nothing already on the spectrum was touched."""
        fits = self.fits
        text = (f"{len(self.found)} peaks found; fits: {fits.attempted} attempted, "
                f"{fits.failed} failed, {fits.skipped_edge} skipped at the edge; "
                f"{len(self.peaks)} peaks fitted, ")
        if self.match.ok:
            text += f"{len(self.match.pairs)} identified in {source_name}"
        else:
            text += f"none identified: {self.match.reason}"
        # Set aside is not the same as unidentified, and a user counting
        # rows against the source deserves to know which it was.
        if self.match.unusable:
            text += (f" ({self.match.unusable} set aside for an implausible "
                     f"width or area)")
        if existing_fits:
            text += f"; {_plural(existing_fits, 'fit')} already on the spectrum "
            text += "were kept." if existing_fits != 1 else "was kept."
        else:
            text += "; no fits were on the spectrum before."
        return text


def calibrate(x, counts, lines, sensitivity=peak_search.DEFAULT_SENSITIVITY,
              variance=None, link_widths=True):
    """search -> fit -> match, as the dialog runs it. `lines` are the
    SourceLines of the loaded .sou file.

    Total: every outcome, including finding nothing, comes back as an
    AutoCalibration whose match carries the reason, so the caller has one
    path and the status line one source of truth.
    """
    found = peak_search.search(counts, sensitivity)
    if not found:
        return AutoCalibration(found, AutoFitOutcome([], [], 0, 0, 0), MatchResult(
            reason="no peaks were found at this sensitivity; lower it"))
    fits = fit_found_peaks(x, counts, found, link_widths=link_widths,
                           variance=variance)
    peaks = [peak for _index, peak in fits.peaks]
    if not peaks:
        return AutoCalibration(found, fits, MatchResult(
            reason=f"none of the {len(found)} peaks found could be fitted"))
    result = match([p.position for p in peaks], [p.fwhm for p in peaks],
                   [p.area for p in peaks],
                   [line.energy for line in lines],
                   [line.intensity for line in lines])
    return AutoCalibration(found, fits, result)
