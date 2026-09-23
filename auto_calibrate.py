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
broken by the tighter residual -- and then the winner has to pass a
second, independent test: the peak areas it implies must follow a
detector efficiency curve (see efficiency_scatter).

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
from peak_search import expected_widths  # noqa: F401  (re-exported; it lived here first)

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

#: A fitted peak whose width exceeds this multiple of the width expected
#: at its channel is set aside: not an anchor, not an inlier, never
#: assigned, never exported. Real single-peak widths scatter between a
#: quarter and 1.8 times the trend on a real Eu-152 spectrum; a fit that
#: has slid onto a neighbour or absorbed background comes back at 2.3
#: times it and more, and the runaways this first existed for were ten
#: times it. Two separates them with room on both sides.
#:
#: This was three while nearby peaks were fitted together, because the
#: real half of a doublet whose other half ran away inherited its
#: partner's width. Every fit is now a single peak, so nothing
#: legitimately inherits a width -- and three let through two fits on the
#: real spectrum, one of them 674.64 keV fitted 6.3 channels off its line
#: at 2.6 times the expected width, that would have reached CalEnEff as
#: measurements.
WIDTH_OUTLIER_RATIO = 2.0

#: A refitted centroid this many expected widths from the found peak it
#: was seeded on has not fitted that peak: it has slid onto a neighbour.
#: The 674.64 keV case above landed on the 678.62 keV peak six channels
#: away, inside a fit window that had been clipped to keep it out -- a
#: window bounds the data fitted, not where the centroid may go.
REFIT_MAX_SHIFT_FWHM = 1.0

#: Two fitted centroids closer together than this many expected widths
#: are one peak fitted twice, not two peaks. Nothing is lost by dropping
#: the weaker of them: lines half a width apart are not resolved by a
#: single-peak fit anyway, so both fits describe the same blob, and
#: keeping both lets the matcher hand one physical peak two energies.
DUPLICATE_FWHM = 0.5

#: Fewer inliers than this is not a calibration, it is a coincidence.
#: Three points can be fitted by a quadratic exactly, so four is the
#: smallest number that can disagree with the fit and thereby support it.
MIN_INLIERS = 4

#: A rival solution whose gain differs by more than this is a genuinely
#: DIFFERENT calibration, not a refinement of the winner. If such a rival
#: explains as many peaks and the efficiency test cannot separate them,
#: the evidence does not choose and the match is refused.
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

#: The efficiency test. For a correct assignment, area / intensity is
#: the detector's full-energy-peak efficiency at that energy times a
#: constant, and an HPGe efficiency curve is smooth -- a quadratic in
#: log-log is the standard parametrisation and follows the low-energy
#: turnover as well as the power-law fall above it. For a WRONG
#: assignment the intensities belong to other lines and the ratios
#: scatter by orders of magnitude.
#:
#: Measured on a real Eu-152 spectrum: the 32 correct assignments
#: scatter about the curve by 0.12 dex (a factor 1.3); shifting every
#: assignment to the neighbouring line gives 0.76 dex (a factor 5.7).
#: The limit below is a factor of ~2.8, three times the real scatter and
#: less than half the wrong one.
EFFICIENCY_MAX_SCATTER = 0.45
#: A point this many scatters from the curve is SUSPECT -- a wrong line,
#: a misfitted area, or a genuine doublet -- and is handed back for the
#: user to see hollow on the plot rather than silently included in the
#: fit. Never tighter than 0.3 dex (a factor 2), so a tightly consistent
#: spectrum does not flag its own noise.
EFFICIENCY_SUSPECT_SIGMA = 3.0
EFFICIENCY_SUSPECT_FLOOR = 0.3
#: Fewer points than this and the curve is not constrained enough to
#: judge anything; the test is skipped, not failed.
EFFICIENCY_MIN_POINTS = 5

#: Sanity bounds on the provisional straight line. A gain outside this
#: range or an offset this large is not a spectrometer, it is a pairing
#: that happens to be arithmetically consistent.
_MIN_GAIN, _MAX_GAIN = 1e-3, 1e3
_MAX_OFFSET = 500.0

#: The refit pass looks for a found peak within this many expected
#: widths of where the calibration says a source line should be. Half a
#: width: the calibration is already good to a fraction of a channel
#: (0.12 keV rms on the real Eu-152 spectrum, a fifth of a channel), so
#: a peak further off than that is a different peak, and claiming it
#: would export a wrong area under this line's intensity.
REFIT_SEARCH_FWHM = 0.5

#: How near an unticked energy has to be to a source line to be taken as
#: that line, in keV.
#:
#: The energies come back from the calibration dialog as the user left
#: them: normally the line's own value to full precision, since that is
#: what the automatic pass and Suggest put in the cell, but a hand-typed
#: row may be a rounded reading of it. Three tenths of a keV covers a
#: couple of decimal places and is still under half the closest spacing
#: in the sample sources -- 0.69 keV, between the 963.37 and 964.06 keV
#: lines of Eu-152 -- so it cannot reach a neighbouring line by mistake.
EXCLUDED_MATCH_KEV = 0.3


class MatchResult:
    """What the search concluded.

    `pairs` is [(peak index, energy)] -- indices into the peaks list as
    given. `reason` is set only when the search failed, and says why in
    words meant for the status line. `unusable` counts the peaks set
    aside before matching for an implausible width or area. `suspect` is
    the subset of `pairs` whose area does not sit on the efficiency curve
    the others define; `efficiency_scatter` is that curve's residual rms
    in dex, or None when there were too few points to fit one.
    """

    __slots__ = ("pairs", "gain", "offset", "rms", "reason", "considered",
                 "unusable", "suspect", "efficiency_scatter")

    def __init__(self, pairs=(), gain=0.0, offset=0.0, rms=float("inf"),
                 reason=None, considered=0, unusable=0, suspect=(),
                 efficiency_scatter=None):
        self.pairs = list(pairs)
        self.gain = gain
        self.offset = offset
        self.rms = rms
        self.reason = reason
        self.considered = considered
        self.unusable = unusable
        self.suspect = list(suspect)
        self.efficiency_scatter = efficiency_scatter

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


def _usable(fwhms, weights, expected):
    """Which fitted peaks may take part at all.

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
    count = len(energies)
    predicted = offset + gain * channels
    insertion = np.searchsorted(energies, predicted)
    order = np.argsort(np.abs(
        predicted - energies[np.clip(insertion, 0, count - 1)]))

    # Every peak's three candidate lines and their distances, computed
    # once for the whole array rather than per peak inside the loop
    # below. This function runs thousands of times per match and was two
    # thirds of the matcher's time, almost all of it in 130,000 scalar
    # searchsorted calls. The candidates and the ordering are exactly
    # what the scalar version produced -- an out-of-range neighbour is
    # given an infinite distance rather than being skipped, which comes
    # to the same thing under the minimum.
    neighbours = np.stack([insertion - 1, insertion, insertion + 1])
    inside = (neighbours >= 0) & (neighbours < count)
    safe = np.clip(neighbours, 0, count - 1)
    distances = np.where(inside, np.abs(energies[safe] - predicted), np.inf)

    taken_lines, pairs, total = set(), [], 0.0
    for k in order:
        # nearest line, checking the neighbour on each side
        best_j, best_d = -1, float("inf")
        for row in range(3):
            cand = int(safe[row, k])
            if inside[row, k] and cand not in taken_lines and distances[row, k] < best_d:
                best_j, best_d = cand, float(distances[row, k])
        if best_j < 0 or best_d > tolerances[k]:
            continue
        taken_lines.add(best_j)
        pairs.append((int(k), best_j))
        total += best_d * best_d
    rms = math.sqrt(total / len(pairs)) if pairs else float("inf")
    return pairs, rms


def efficiency_scatter(energies, areas, intensities):
    """(rms in dex, per-point residuals in dex) of log10(area/intensity)
    about a quadratic in log10(energy) -- or (None, None) with fewer than
    EFFICIENCY_MIN_POINTS usable points.

    This is what makes a set of assignments physically consistent rather
    than merely arithmetically consistent: the ratios must trace one
    smooth efficiency curve. Points with a non-positive energy, area or
    intensity are left out of the fit and get NaN residuals.
    """
    e = np.asarray(energies, dtype=float)
    a = np.asarray(areas, dtype=float)
    i = np.asarray(intensities, dtype=float)
    ok = np.isfinite(e) & np.isfinite(a) & np.isfinite(i) & (e > 0) & (a > 0) & (i > 0)
    residuals = np.full(e.shape, np.nan)
    if int(ok.sum()) < EFFICIENCY_MIN_POINTS:
        return None, residuals
    x = np.log10(e[ok])
    y = np.log10(a[ok] / i[ok])
    degree = 2 if ok.sum() >= 6 else 1
    coefficients = np.polyfit(x, y, degree)
    residuals[ok] = y - np.polyval(coefficients, x)
    return float(np.sqrt(np.mean(residuals[ok] ** 2))), residuals


def match(peak_channels, peak_fwhms, peak_weights, line_energies,
          line_intensities=None, inlier_fwhm=INLIER_FWHM):
    """Assign source lines to peaks with no prior calibration.

    `peak_weights` ranks peaks for anchor selection and, being the fitted
    areas, feeds the efficiency test. A peak whose weight is not positive
    is never assigned; see _usable. `line_intensities` rank the lines
    and feed the efficiency test too; without them the first
    ANCHOR_LINES are used and the efficiency test is skipped.

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
    intensity_of = None
    if line_intensities is None:
        anchor_l = list(range(min(ANCHOR_LINES, energies.size)))
    else:
        order = np.argsort(np.asarray(line_energies, dtype=float))
        sorted_int = np.asarray(line_intensities, dtype=float)[order]
        anchor_l = _strongest(sorted_int, ANCHOR_LINES)
        intensity_of = dict(zip(energies.tolist(), sorted_int.tolist()))

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

    def scatter_of_pairs(pairs):
        if intensity_of is None or not pairs:
            return None, None
        idx = [k for k, _e in pairs]
        e = [en for _k, en in pairs]
        return efficiency_scatter(e, weights[idx], [intensity_of[en] for en in e])

    def scatter_of(result):
        return scatter_of_pairs(result.pairs)

    best_scatter, best_residuals = scatter_of(best)

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
        # Equal counts. The efficiency test is independent evidence: if
        # one candidate's areas trace an efficiency curve and the other's
        # do not, that one is right. If neither is clearly better, refuse.
        rival_scatter, _r = scatter_of(rival)
        decided = False
        if best_scatter is not None and rival_scatter is not None:
            if best_scatter < 0.5 * rival_scatter:
                decided = True
            elif rival_scatter < 0.5 * best_scatter:
                best, best_scatter, best_residuals = rival, rival_scatter, _r
                best.considered, best.unusable = considered, unusable
                decided = True
        if not decided:
            return MatchResult(
                reason=(f"two different calibrations explain the spectrum equally "
                        f"well ({len(best.pairs)} peaks each, gains "
                        f"{best.gain:.4g} and {rival.gain:.4g} keV/channel); "
                        f"assign two peaks by hand to settle it"),
                considered=considered, unusable=unusable)

    if best_scatter is not None:
        if best_scatter > EFFICIENCY_MAX_SCATTER:
            return MatchResult(
                reason=(f"the peak areas do not follow a detector efficiency curve "
                        f"under this assignment (scatter a factor of "
                        f"{10 ** best_scatter:.1f} about the curve); the lines are "
                        f"probably paired wrongly"),
                considered=considered, unusable=unusable)
        limit = max(EFFICIENCY_SUSPECT_SIGMA * best_scatter, EFFICIENCY_SUSPECT_FLOOR)
        best.suspect = [pair for pair, r in zip(best.pairs, best_residuals)
                        if np.isfinite(r) and abs(r) > limit]
        # Report the scatter of the curve that SURVIVES, not the one the
        # doubted points distorted. A suspect point is already kept out
        # of the calibration fit and out of the export -- it opens
        # unticked -- so a number computed with it in describes a curve
        # nothing downstream uses.
        #
        # This is not cosmetic. On a real Eu-152 spectrum one peak in a
        # crowded triplet was flagged suspect, correctly, and still
        # carried the reported scatter from 0.045 to 0.078 on its own --
        # which is what made a better-fitting peak shape look worse than
        # the one it would replace.
        #
        # Deliberately NOT iterated: dropping a point lowers the scatter,
        # which lowers the limit, which can flag another. One pass is
        # enough to stop a known-bad point inflating the number, and
        # chasing the sequence risks trimming a spectrum down to whatever
        # happens to be self-consistent.
        kept = [pair for pair in best.pairs if pair not in best.suspect]
        kept_scatter, _kept_residuals = scatter_of_pairs(kept)
        best.efficiency_scatter = (
            best_scatter if kept_scatter is None else kept_scatter
        )
    return best


# --------------------------------------------------------------- fitting


class AutoFitOutcome:
    """What the automatic fitting pass produced.

    `results` are FitResults ready to commit to a spectrum, one per
    fitted peak, in channel order. `peaks` is (result index, PeakResult)
    in the same order, which is what the matcher and the calibration
    dialog see. `broad` counts the found peaks set aside as not being
    photopeaks; `skipped` the photopeaks that had no clear background on
    both sides; `failed` the fits that did not converge; `runaway` those
    that converged on something other than the peak they were seeded on;
    `duplicate` those that landed on a peak another fit had already
    taken. None of those stops the rest.
    """

    __slots__ = ("results", "peaks", "attempted", "failed", "skipped", "broad",
                 "runaway", "duplicate")

    def __init__(self, results, peaks, attempted, failed, skipped, broad,
                 runaway=0, duplicate=0):
        self.results = results
        self.peaks = peaks
        self.attempted = attempted
        self.failed = failed
        self.skipped = skipped
        self.broad = broad
        self.runaway = runaway
        self.duplicate = duplicate


def _fit_one(x, y, peak, found, variance):
    """(FitResult or None, reason) for one peak fitted on its own, with
    its own markers: a fit window around it alone and two background
    windows read from flat, peak-free stretches beside it."""
    from peak_fit import FitError, fit_peaks

    region = peak_search.regions(peak, y, found)
    if region is None:
        return None, "skipped"
    bg_left, bg_right, fit_region, positions = region
    try:
        return fit_peaks(x, y, bg_left, bg_right, fit_region, positions,
                         variance=variance), None
    except (FitError, ValueError, RuntimeError):
        return None, "failed"


def _settle(fitted):
    """(FitResults worth keeping, runaways dropped, duplicates dropped).

    A single-peak fit is free to walk out of the window it was seeded in,
    and on a crowded spectrum it does. Of 67 fits attempted on a real
    Eu-152 spectrum, ten came back as something other than the peak they
    were seeded on: components up to eighteen times the width of that
    spectrum's own peaks, three of them with a negative area, two of
    them sitting on a peak another fit had already caught. Committed,
    each is a peak in Fit Results that is not in the spectrum, and the
    matcher is free to name it -- while the fit window was still
    widening onto close neighbours, two fits of one peak gave it two
    different energies, 1084.00 and 1085.84 keV.

    So a fit is dropped when it converged on something other than the
    peak it was seeded on -- displaced by more than REFIT_MAX_SHIFT_FWHM
    of the expected width, wider than WIDTH_OUTLIER_RATIO of it, or with
    no positive area -- and, of two fits that landed on the same peak,
    the one with the smaller area goes. This is the same guard the refit
    pass applies, for the same reason and against the same trend through
    the FITTED widths rather than each peak's own width.
    """
    if not fitted:
        return [], 0, 0
    widths = expected_widths([result.peaks[0].position for _p, result in fitted],
                             [result.peaks[0].fwhm for _p, result in fitted])
    sound, runaway = [], 0
    for (seed, result), expected in zip(fitted, widths):
        one = result.peaks[0]
        if (not math.isfinite(float(expected)) or not math.isfinite(one.fwhm)
                or not math.isfinite(one.position) or not (one.area > 0.0)
                or one.fwhm > WIDTH_OUTLIER_RATIO * expected
                or abs(one.position - seed.channel) > REFIT_MAX_SHIFT_FWHM * expected):
            runaway += 1
            continue
        sound.append((result, one, float(expected)))

    # Strongest first, so the fit that actually caught the peak is the
    # one kept and the stray that landed beside it is the one dropped.
    kept, duplicate = [], 0
    for entry in sorted(sound, key=lambda s: -s[1].area):
        if any(abs(entry[1].position - other.position) < DUPLICATE_FWHM * entry[2]
               for _r, other, _e in kept):
            duplicate += 1
            continue
        kept.append(entry)
    kept.sort(key=lambda s: s[1].position)
    return [result for result, _one, _expected in kept], runaway, duplicate


def fit_found_peaks(x, y, found, link_widths=True, variance=None):
    """Fit every photopeak among `found`, one at a time.

    Individual fits, each with its own markers -- that is what the user
    does by hand, and it keeps one bad neighbour from spoiling a good
    peak. Broad features are set aside first and never fitted, but they
    stay in the list every fit is told about, so no fit window reaches
    into one and no background is read from one.

    A fit that raises is dropped and counted, never allowed to abort the
    pass. `link_widths` is accepted for the caller's convenience and has
    no effect on a single-peak fit. `variance` is the spectrum's
    propagated per-channel variance, or None for counts read from a
    file, exactly as the interactive Fit passes it.
    """
    peaks_to_fit, broad = peak_search.reject_broad(found)
    fitted = []
    attempted = failed = skipped = 0
    for peak in sorted(peaks_to_fit, key=lambda p: p.channel):
        result, reason = _fit_one(x, y, peak, found, variance)
        if reason == "skipped":
            skipped += 1
            continue
        attempted += 1
        if result is None or not result.peaks:
            failed += 1
            continue
        fitted.append((peak, result))

    results, runaway, duplicate = _settle(fitted)
    peaks = [(index, one)
             for index, result in enumerate(results) for one in result.peaks]
    return AutoFitOutcome(results, peaks, attempted, failed, skipped, len(broad),
                          runaway, duplicate)


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

    @property
    def suspect_energies(self):
        """Energies of the assignments the efficiency test doubts. The
        dialog opens with these UNTICKED and hollow on the plot: still
        assigned, so the user sees what was doubted and why, but out of
        the fit until they decide otherwise."""
        return tuple(energy for _k, energy in self.match.suspect)

    def summary(self, existing_fits, source_name):
        """One line for the status bar, honest about every count: what
        was found, what was set aside, what could not be fitted, what was
        identified, and that nothing already on the spectrum was
        touched."""
        fits = self.fits
        text = (f"{len(self.found)} peaks found, {fits.broad} broad features set aside; "
                f"fits: {fits.attempted} attempted, {fits.failed} failed, "
                f"{fits.skipped} skipped for want of a clear background, "
                f"{fits.runaway} that strayed and {fits.duplicate} that "
                f"repeated another dropped; "
                f"{len(self.peaks)} peaks fitted, ")
        if self.match.ok:
            text += f"{len(self.match.pairs)} identified in {source_name}"
            if self.match.suspect:
                text += (f", {len(self.match.suspect)} of them left out of the fit as "
                         f"suspect (area off the efficiency curve)")
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
        return AutoCalibration(found, AutoFitOutcome([], [], 0, 0, 0, 0), MatchResult(
            reason="no peaks were found at this sensitivity; lower it"))
    fits = fit_found_peaks(x, counts, found, link_widths=link_widths,
                           variance=variance)
    peaks = [peak for _index, peak in fits.peaks]
    if not peaks:
        # Say which way they were lost. "Could not be fitted" reads as a
        # solver failure, and it usually is not: on a crowded spectrum
        # the fits succeed and are then dropped for landing somewhere
        # other than the peak they were seeded on.
        why = ", ".join(
            f"{count} {what}" for count, what in (
                (fits.broad, "set aside as broad features"),
                (fits.skipped, "with no clear background"),
                (fits.failed, "that would not fit"),
                (fits.runaway, "whose fit strayed"),
                (fits.duplicate, "repeating another fit"),
            ) if count
        )
        return AutoCalibration(found, fits, MatchResult(
            reason=(f"none of the {len(found)} peaks found could be fitted"
                    + (f": {why}" if why else ""))))
    result = match([p.position for p in peaks], [p.fwhm for p in peaks],
                   [p.area for p in peaks],
                   [line.energy for line in lines],
                   [line.intensity for line in lines])
    return AutoCalibration(found, fits, result)


# --------------------------------------------- the refit for CalEnEff


class RefitOutcome:
    """What refitting the source's lines produced, once the calibration
    is known.

    `results` are one FitResult per line fitted; `pairs` the matching
    [(fitted channel, energy)]. The line is always `result.peaks[0]`; any
    further peak in a result is a listed line fitted beside it
    (`alongside`), never exported. A blended line's share of its peak is
    not decided here but where every efficiency point is made,
    caleneff_export.share_blended_areas, so the automatic run and the
    hand-assigned dialog cannot disagree about it.

    The counts say what happened to every other line: `outside` the
    spectrum's range, `invisible` with no found peak where the
    calibration puts it, `alongside` with no found peak of its own but
    inside a stronger line's fit window, so fitted there as a second
    peak, `blended` into a stronger line that the same found peak
    already accounts for, `skipped` for want of a clear background,
    `failed` to converge, `runaway` when the fit converged on something
    other than the peak it was seeded on -- too wide, displaced, or with
    no positive area -- which is dropped rather than exported as a
    measurement of that line, and `excluded` when the user unticked it
    in the calibration dialog.
    """

    __slots__ = ("results", "pairs", "outside", "invisible", "alongside",
                 "blended", "skipped", "failed", "runaway", "excluded")

    def __init__(self):
        self.results, self.pairs = [], []
        self.outside = self.invisible = self.alongside = self.blended = 0
        self.skipped = self.failed = self.runaway = self.excluded = 0

    def summary(self, source_name):
        parts = [f"{_plural(len(self.results), 'line')} of {source_name} refitted for CalEnEff"]
        for count, what in ((self.outside, "outside the spectrum"),
                            (self.invisible, "with no visible peak"),
                            (self.alongside, "fitted beside a stronger neighbour"),
                            (self.blended, "blended into a stronger neighbour, "
                                           "the area shared by intensity"),
                            (self.skipped, "without a clear background"),
                            (self.failed, "that would not fit"),
                            (self.runaway, "whose fit ran onto a neighbour or the background"),
                            (self.excluded, "left unticked in the calibration")):
            if count:
                parts.append(f"{count} {what}")
        return "; ".join(parts) + "."


def _unticked(lines, excluded):
    """The energies of `lines` the user unticked, matched to the nearest
    line within EXCLUDED_MATCH_KEV. An excluded energy that matches no
    line is ignored rather than guessed at."""
    chosen = set()
    for value in excluded or ():
        nearest = min(lines, key=lambda line: abs(line.energy - value), default=None)
        if nearest is not None and abs(nearest.energy - value) <= EXCLUDED_MATCH_KEV:
            chosen.add(nearest.energy)
    return chosen


def refit_source_lines(x, counts, lines, calibration,
                       sensitivity=peak_search.DEFAULT_SENSITIVITY, variance=None,
                       excluded=()):
    """Fit, individually, every source line that is visibly present.

    The calibration says where each line should be; a found photopeak
    within REFIT_SEARCH_FWHM of that is fitted on its own with its own
    markers, seeded at the found centroid rather than the predicted one.
    Lines the calibration puts outside the spectrum, lines with nothing
    above the noise where they should be, and lines that land on a peak
    already claimed by a stronger line are counted, not exported -- an
    area fitted where there is no peak is a number, not a measurement,
    and it would sit on the efficiency curve as if it were one.

    Not exported is not the same as not there, though, and two kinds of
    neighbour used to hand their counts to the line being fitted:

    - A weaker line close enough to land on the SAME found peak is part
      of that peak, unresolved, and its counts are in the fitted area.
      It is counted here as blended; the area is shared by intensity when
      the efficiency points are made, by caleneff_export.share_blended_areas
      -- the one place that does it for the hand-assigned route as well.
    - A listed line with no found peak of its own but inside this line's
      fit window is added to the fit as a second peak, its position held
      where the calibration puts it and its width tied to the line's.
      It is modelled only so that its counts are not counted as the
      line's; it is not exported. On a real Eu-152 spectrum the 566.44
      keV line put 18% on top of 563.99 keV this way, and on a real
      Ra-226 one the Bi-214 line at 386.77 keV put 80% on top of 388.89
      keV. Should that fit not converge, the single-peak fit stands,
      which is what this did before.

    `excluded` are the energies the user unticked in the calibration
    dialog. They are not fitted and not exported: the reason a point is
    unticked in an automatic run is that its area does not sit on the
    efficiency curve the others trace, which is exactly the number
    CalEnEff would be fed. An unticked line still claims its peak, so a
    weaker line blended into it does not inherit the peak's whole area,
    and an unticked line blended into a stronger one still takes its
    share.
    """
    from peak_fit import FitError, fit_peaks

    outcome = RefitOutcome()
    unticked = _unticked(lines, excluded)
    counts = np.asarray(counts, dtype=float)
    found = peak_search.search(counts, sensitivity)
    photopeaks, _broad = peak_search.reject_broad(found)
    if not photopeaks:
        outcome.invisible = len(lines) - len(unticked)
        outcome.excluded = len(unticked)
        return outcome
    photopeaks.sort(key=lambda p: p.channel)
    centres = np.array([p.channel for p in photopeaks])
    widths = expected_widths(centres, [p.fwhm for p in photopeaks])

    claims = {}
    unclaimed = []
    for line in lines:
        try:
            channel = float(calibration.invert(line.energy))
        except Exception:
            outcome.outside += 1
            continue
        if not math.isfinite(channel) or channel < 0 or channel > counts.size - 1:
            outcome.outside += 1
            continue
        k = int(np.argmin(np.abs(centres - channel)))
        if abs(centres[k] - channel) > REFIT_SEARCH_FWHM * widths[k]:
            unclaimed.append((line, channel))
            continue
        claims.setdefault(k, []).append(line)

    fitted, beside_a_fit = [], set()
    for k in sorted(claims):
        candidates = claims[k]
        line = max(candidates, key=lambda l: l.intensity)
        outcome.blended += len(candidates) - 1
        if line.energy in unticked:
            outcome.excluded += 1
            continue
        region = peak_search.regions(photopeaks[k], counts, found)
        if region is None:
            outcome.skipped += 1
            continue
        bg_left, bg_right, fit_region, positions = region
        beside = [(i, other, channel) for i, (other, channel) in enumerate(unclaimed)
                  if fit_region[0] <= channel <= fit_region[1]]
        result = None
        if beside:
            first = len(positions)
            try:
                result = fit_peaks(
                    x, counts, bg_left, bg_right, fit_region,
                    list(positions) + [channel for _i, _o, channel in beside],
                    variance=variance,
                    fixed_params={f"pos_{first + j}": channel
                                  for j, (_i, _o, channel) in enumerate(beside)},
                )
                beside_a_fit.update(i for i, _o, _c in beside)
            except (FitError, ValueError, RuntimeError):
                result = None
        if result is None:
            try:
                result = fit_peaks(x, counts, bg_left, bg_right, fit_region, positions,
                                   variance=variance)
            except (FitError, ValueError, RuntimeError):
                outcome.failed += 1
                continue
        fitted.append((photopeaks[k], result, line))
    outcome.alongside = len(beside_a_fit)
    outcome.invisible = len(unclaimed) - outcome.alongside

    # The same judgement the identification pass applies before it trusts
    # a fit, applied here before a fit is EXPORTED -- and against the same
    # trend it uses, the one through the FITTED widths. Judging against
    # the search's widths instead, as this did, is 1.8 times looser: the
    # search reads a width off a smoothed copy, which on a real Eu-152
    # spectrum came out 5.03 channels where the fits say 2.83. Nothing
    # wrong had reached an export, but this is the more consequential of
    # the two paths and had the weaker gate.
    kept, runaway, _duplicate = _settle([(seed, result) for seed, result, _l in fitted])
    outcome.runaway += runaway
    by_id = {id(result): line for _seed, result, line in fitted}
    for result in kept:
        outcome.results.append(result)
        outcome.pairs.append((result.peaks[0].position, by_id[id(result)].energy))
    return outcome
