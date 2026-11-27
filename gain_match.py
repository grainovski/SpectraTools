"""Gain matching one detector onto another: CH1 mapped onto CH0's channels.

A two-detector gamma-gamma matrix has one detector per axis, each with its
own gain and offset. Symmetrizing it, or simply comparing the two axes,
needs both on one scale. The decision (v7 plan, 2026-11-27) is that CH0 is
the reference and stays untouched, and CH1 is mapped onto CH0's ADC
channel scale by a fitted polynomial

    ch0' = P(ch1) = a0 + a1*ch1 (+ a2*ch1**2)

made from peaks that both detectors see. Getting there has four steps,
one function each, so the dialog can show and override every one:

1. **find_peaks** -- search and fit each singles spectrum, reusing
   peak_search and auto_calibrate.fit_found_peaks unchanged.
2. **pair_peaks** -- decide which CH1 peak is which CH0 peak. This is the
   step that can go wrong silently, so it reports failure rather than
   guessing (auto_calibrate's rule): a gain fitted to a wrong pairing
   passes through every point it was fitted to and looks entirely
   reasonable.
3. **fit_gain** -- a weighted polynomial fit through the included pairs.
4. **GainMatch.apply** -- map integer CH1 ADC values event by event, with
   a seeded dither so a non-integer gain leaves no picket fence.

Qt-free and pure, like auto_calibrate, so every step can be tested
against spectra whose answer is known.

## Channel coordinates (read this before touching apply)

Every centroid here is measured the way the rest of the application
measures it: on x = arange(n), so a count in channel i sits at x = i and
channel i covers [i - 1/2, i + 1/2). P is fitted on those centroids, so it
maps CENTRE coordinates. An event recorded in ADC channel E therefore had
a true value somewhere in [E - 1/2, E + 1/2), and its mapped channel is

    ch0' = floor(P(E - 1/2 + u) + 1/2),   u ~ U(0, 1), seeded.

The plan writes this as floor(P(E + u)), which is the same rule in EDGE
coordinates (channel E covering [E, E + 1)) -- but only if P is fitted in
edge coordinates too. Applying it to a P fitted on centres shifts every
event by about (P' - 1)/2 channels: 0.19 channels for a gain of 1.37, too
small to see on one peak and systematic over all of them. With the
identity map both forms give back E exactly.
"""

import datetime
import json
import math
from dataclasses import dataclass

import numpy as np

import auto_calibrate
import peak_search
from matrix_symmetrize import DEFAULT_SEED
from peak_search import expected_widths

#: The DT5780's 14-bit ADC, which is what the v7 sorter bins at.
DEFAULT_NCHAN = 16384

#: Compression factors tried when searching a spectrum, in order of
#: preference. peak_search only reports peaks between _MIN_FWHM (1.2) and
#: _MAX_FWHM (60) channels wide, which is right for the 4k-8k spectra it
#: was built on and wrong at both ends of a 16k digitizer spectrum: the
#: real Na-22 run's scintillators have a 66-channel FWHM at 511 keV and
#: 105 at 1275 keV, so at full resolution NEITHER line is found -- only
#: a low-energy bump and two one-channel spikes. Compressing by two finds
#: both, plus K-40. An HPGe spectrum at 16k has 3-6 channel peaks, which
#: compression by four would push below the minimum. So each factor is
#: tried and the one that captures the peaks is kept; see find_peaks.
COMPRESSIONS = (1, 2, 4, 8)

#: A compression is accepted when its fitted peaks hold at least this
#: fraction of the largest total fitted area any factor achieved. The
#: smallest such factor wins, because it keeps the most resolution. On
#: the real Na-22 run the totals were 4.2k / 49.1k / 49.2k / 44.8k counts
#: for factors 1 / 2 / 4 / 8: a factor that loses a strong line loses
#: most of the area, while two factors that both see every line differ
#: by a percent or two.
COMPRESSION_AREA_FRACTION = 0.9

#: A peak narrower than this many COMPRESSED channels is under-sampled:
#: its fitted area is unreliable and is not counted when choosing the
#: compression. Measured on a synthetic HPGe-like spectrum with 2.4-4.6
#: channel peaks: compressed by four they are about one channel wide and
#: their fitted areas came out up to 58% high, enough to make the factor
#: that had destroyed the resolution look like the one that found most.
UNDERSAMPLED_FWHM = 2.0

#: The straight line that pairs the peaks only finds pairs within
#: INLIER_FWHM of it, and a scintillator pair's relative gain can curve
#: by more than that across 16k channels (a synthetic 3e-6 quadratic
#: term: 476 channels at the top). Once four or more pairs are known,
#: further pairs are looked for under a quadratic through them, at most
#: this many rounds.
_EXTEND_ROUNDS = 5

#: Polynomial order 2 needs at least this many included pairs: three
#: points fix a quadratic exactly, so four is the fewest that can
#: disagree with it (auto_calibrate.MIN_INLIERS, for the same reason).
MIN_PAIRS_QUADRATIC = 4

#: A centroid error at or below zero (a fit whose covariance could not
#: be estimated) is floored here rather than given infinite weight.
_MIN_CENTROID_ERR = 1e-3

#: Iterations of the weighted fit. The weights depend on the slope, which
#: depends on the weights; the slope is already right to a part in 1e4
#: after the first pass, so three is plenty.
_WEIGHT_ITERATIONS = 3

FORMAT_NAME = "spectratools-gain-match"
FORMAT_VERSION = 1


class GainMatchError(ValueError):
    """A gain match that cannot be made, with a message meant for the
    user."""


# ------------------------------------------------------------------ peaks


@dataclass(frozen=True)
class GainPeak:
    """One fitted peak, in FULL-resolution channels whatever compression
    it was found at."""

    centroid: float
    centroid_err: float
    fwhm: float
    area: float
    area_err: float
    compression: int = 1


def _compress(counts, factor):
    """Sum `factor` adjacent channels; a tail shorter than `factor` is
    dropped (at most 7 channels of 16384, at the very top)."""
    if factor == 1:
        return counts
    n = (counts.size // factor) * factor
    return counts[:n].reshape(-1, factor).sum(axis=1)


def _peaks_at(counts, factor, sensitivity):
    """Fitted peaks of `counts` compressed by `factor`, converted back to
    full-resolution channels.

    Compressed bin j holds full channels j*k ... j*k + k - 1, whose
    centre is j*k + (k - 1)/2; a centroid x in compressed channels is
    therefore k*x + (k - 1)/2 in full ones. Widths and errors scale by k;
    areas are counts and do not scale.
    """
    y = _compress(counts, factor)
    found = peak_search.search(y, sensitivity)
    if not found:
        return []
    outcome = auto_calibrate.fit_found_peaks(np.arange(y.size, dtype=float), y, found)
    k = float(factor)
    return [GainPeak(centroid=k * p.position + (k - 1.0) / 2.0,
                     centroid_err=k * p.position_err,
                     fwhm=k * p.fwhm, area=p.area, area_err=p.area_err,
                     compression=factor)
            for _index, p in outcome.peaks]


def find_peaks(spectrum, sensitivity=peak_search.DEFAULT_SENSITIVITY,
               compression=None):
    """Fitted peaks of a singles spectrum, sorted by centroid.

    `compression` None tries every factor in COMPRESSIONS and keeps the
    smallest whose total fitted area reaches COMPRESSION_AREA_FRACTION of
    the best -- total area because it is what a wrong factor loses, a
    line too wide to be found or too narrow to survive compression. Peaks
    under UNDERSAMPLED_FWHM compressed channels wide are left out of the
    total, since compression inflates exactly their areas. A
    count of peaks would not do: at full resolution the real Na-22 CH1
    spectrum yields three "peaks", two of them one-channel spikes of 11
    and 18 counts, and none of them 511 or 1275 keV.

    An integer `compression` uses that factor alone.
    """
    counts = np.asarray(spectrum, dtype=float)
    if counts.ndim != 1:
        raise GainMatchError("A singles spectrum must be one-dimensional.")
    factors = COMPRESSIONS if compression is None else (int(compression),)
    if any(f < 1 for f in factors):
        raise GainMatchError("The compression factor must be at least 1.")
    candidates = [(f, _peaks_at(counts, f, sensitivity)) for f in factors]
    totals = [sum(p.area for p in peaks
                  if p.area > 0 and p.fwhm >= UNDERSAMPLED_FWHM * f)
              for f, peaks in candidates]
    best = max(totals) if totals else 0.0
    if best <= 0:
        # Nothing well sampled anywhere: the least compressed answer is
        # the one that has thrown away the least, so it is given as is.
        return sorted(candidates[0][1], key=lambda p: p.centroid)
    for (_factor, peaks), total in zip(candidates, totals):
        if total >= COMPRESSION_AREA_FRACTION * best:
            return sorted(peaks, key=lambda p: p.centroid)
    return []  # unreachable: the best factor always qualifies


# ---------------------------------------------------------------- pairing


@dataclass
class GainPair:
    """One CH1 peak and the CH0 peak it was matched to. `energy` is the
    source line both were identified as, or None without a source.
    `included` is the box the dialog shows; a pair the matcher doubts
    starts unticked."""

    ch1: float
    ch1_err: float
    ch0: float
    ch0_err: float
    energy: float = None
    included: bool = True
    fwhm1: float = float("nan")
    fwhm0: float = float("nan")
    area1: float = float("nan")
    area0: float = float("nan")

    def to_dict(self):
        return {"ch1": self.ch1, "ch1_err": self.ch1_err,
                "ch0": self.ch0, "ch0_err": self.ch0_err,
                "energy": self.energy, "included": bool(self.included),
                "fwhm1": _json_float(self.fwhm1), "fwhm0": _json_float(self.fwhm0),
                "area1": _json_float(self.area1), "area0": _json_float(self.area0)}

    @classmethod
    def from_dict(cls, d):
        def num(key):
            value = d.get(key)
            return float("nan") if value is None else float(value)
        energy = d.get("energy")
        return cls(ch1=float(d["ch1"]), ch1_err=float(d["ch1_err"]),
                   ch0=float(d["ch0"]), ch0_err=float(d["ch0_err"]),
                   energy=None if energy is None else float(energy),
                   included=bool(d.get("included", True)),
                   fwhm1=num("fwhm1"), fwhm0=num("fwhm0"),
                   area1=num("area1"), area0=num("area0"))


def _json_float(value):
    """JSON has no NaN; None stands for 'not known'."""
    return None if value is None or not math.isfinite(value) else float(value)


class PairResult:
    """What pair_peaks concluded.

    `pairs` is set only on success, sorted by CH1 centroid. `reason` is
    set only on failure, in words meant for the status line. `method`
    says how the pairing was decided: "match" (auto_calibrate.match, four
    or more pairs with its full set of checks), "few" (two or three
    pairs, the exhaustive check below) or "source" (both channels
    identified by energy). `confirmations` is how many pairs there are
    beyond the two that any straight line passes through exactly -- 0
    means the pairing is the only one possible but nothing confirms it,
    which `note` then says in words.
    """

    __slots__ = ("pairs", "reason", "method", "confirmations", "note")

    def __init__(self, pairs=(), reason=None, method=None, confirmations=0, note=None):
        self.pairs = sorted(pairs, key=lambda p: p.ch1)
        self.reason = reason
        self.method = method
        self.confirmations = confirmations
        self.note = note

    @property
    def ok(self):
        return self.reason is None and len(self.pairs) >= 2

    def __repr__(self):
        if self.reason:
            return f"PairResult(failed: {self.reason})"
        return (f"PairResult({len(self.pairs)} pairs by {self.method}, "
                f"{self.confirmations} confirmation(s))")


class _Assignment:
    """[(peak index, reference value)] plus the indices to start unticked,
    or a reason. Internal: one channel's peaks against one reference
    list, which is either CH0's centroids or a source's energies."""

    __slots__ = ("pairs", "suspect", "reason", "method")

    def __init__(self, pairs=(), suspect=(), reason=None, method=None):
        self.pairs = list(pairs)
        self.suspect = set(suspect)
        self.reason = reason
        self.method = method


def _candidates(channels, fwhms, weights, refs, ref_weights):
    """Every provisional straight line auto_calibrate.match would try,
    with the inliers it explains: [(gain, offset, pairs, rms)].

    The same anchors (the ANCHOR_PEAKS strongest usable peaks, the
    ANCHOR_LINES strongest references), the same sanity bounds and the
    same inlier rule (auto_calibrate._inliers with a tolerance of
    INLIER_FWHM expected widths), so that the count found here is the
    count match would find. `refs` must be sorted.
    """
    expected = expected_widths(channels, fwhms)
    usable = auto_calibrate._usable(fwhms, weights, expected)
    unit_tolerance = np.where(usable, expected * auto_calibrate.INLIER_FWHM, -1.0)
    anchor_p = [i for i in auto_calibrate._strongest(np.where(usable, weights, -np.inf),
                                                     auto_calibrate.ANCHOR_PEAKS)
                if usable[i]]
    if ref_weights is None:
        anchor_l = list(range(min(auto_calibrate.ANCHOR_LINES, refs.size)))
    else:
        anchor_l = auto_calibrate._strongest(ref_weights, auto_calibrate.ANCHOR_LINES)
    out = []
    for ai in range(len(anchor_p)):
        for aj in range(ai + 1, len(anchor_p)):
            i, j = anchor_p[ai], anchor_p[aj]
            lo_c, hi_c = (i, j) if channels[i] < channels[j] else (j, i)
            span_c = channels[hi_c] - channels[lo_c]
            if span_c <= 0:
                continue
            for bi in range(len(anchor_l)):
                for bj in range(bi + 1, len(anchor_l)):
                    p, q = anchor_l[bi], anchor_l[bj]
                    lo_e, hi_e = (p, q) if refs[p] < refs[q] else (q, p)
                    span_e = refs[hi_e] - refs[lo_e]
                    if span_e <= 0:
                        continue
                    gain = span_e / span_c
                    if not (auto_calibrate._MIN_GAIN <= gain <= auto_calibrate._MAX_GAIN):
                        continue
                    offset = refs[lo_e] - gain * channels[lo_c]
                    if abs(offset) > auto_calibrate._MAX_OFFSET:
                        continue
                    pairs, rms = auto_calibrate._inliers(
                        channels, gain * unit_tolerance, refs, gain, offset)
                    out.append((gain, offset, pairs, rms))
    return out, usable


def _assign(channels, fwhms, weights, refs, ref_weights=None):
    """Which peak is which reference, or why that cannot be said.

    Four or more pairs: auto_calibrate.match decides, with every check it
    has -- confirmations beyond the anchors, the strongest peaks
    explained, an incompatible rival refused, and the efficiency test.

    Two or three: match refuses these outright (MIN_INLIERS = 4), and a
    Na-22 gain match has exactly two lines, so they are decided here by a
    stricter rule of the same kind. A candidate is only credible if it
    pairs the TWO strongest usable peaks -- in a gain match the strong
    peaks are the lines both detectors see -- and it is accepted only if
    no other credible candidate explains as many peaks with a DIFFERENT
    assignment. That is stricter than match's rival test, which compares
    gains: with a few peaks two assignments can share a gain and differ
    by an offset (peaks at 3000 and 3400 against 3000, 3400 and 3800 fit
    gain 1 at offset 0 or at offset 400 equally well), and choosing
    between them would be a guess.
    """
    channels = np.asarray(channels, dtype=float)
    fwhms = np.asarray(fwhms, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(np.asarray(refs, dtype=float), kind="stable")
    refs_sorted = np.asarray(refs, dtype=float)[order]
    rw_sorted = None if ref_weights is None else np.asarray(ref_weights, dtype=float)[order]

    if channels.size < 2:
        return _Assignment(reason=f"only {channels.size} peak(s) were fitted; "
                                  f"at least two are needed")
    if refs_sorted.size < 2:
        return _Assignment(reason=f"only {refs_sorted.size} reference peak(s) or "
                                  f"line(s); at least two are needed")

    candidates, usable = _candidates(channels, fwhms, weights, refs_sorted, rw_sorted)
    if int(np.count_nonzero(usable)) < 2:
        return _Assignment(reason=(f"only {int(np.count_nonzero(usable))} of "
                                   f"{channels.size} fitted peaks have a plausible "
                                   f"width and area; at least two are needed"))
    most = max((len(c[2]) for c in candidates), default=0)

    if most >= auto_calibrate.MIN_INLIERS and channels.size >= auto_calibrate.MIN_INLIERS:
        result = auto_calibrate.match(channels, fwhms, weights, refs_sorted.tolist(),
                                      None if rw_sorted is None else rw_sorted.tolist())
        if not result.ok:
            return _Assignment(reason=result.reason)
        return _Assignment(pairs=result.pairs,
                           suspect={k for k, _e in result.suspect}, method="match")

    strongest = auto_calibrate._strongest(np.where(usable, weights, -np.inf), 2)
    strongest = [k for k in strongest if usable[k]]
    credible = [c for c in candidates
                if all(k in {pk for pk, _m in c[2]} for k in strongest)]
    if not credible:
        return _Assignment(reason=("no straight line pairs the two strongest peaks "
                                   "with the reference; the spectra may not show "
                                   "the same lines"))
    most = max(len(c[2]) for c in credible)
    if most < 2:
        return _Assignment(reason="no two peaks could be paired consistently")
    top = [c for c in credible if len(c[2]) == most]
    assignments = {tuple(sorted(c[2])) for c in top}
    if len(assignments) > 1:
        return _Assignment(reason=(
            f"{len(assignments)} different pairings explain {most} peaks equally "
            f"well; with this few peaks the evidence does not choose between "
            f"them -- add a calibration run with more lines, or a source file"))
    best = min(top, key=lambda c: c[3])
    return _Assignment(pairs=[(k, float(refs_sorted[m])) for k, m in best[2]],
                       method="few")


def _peak_arrays(peaks):
    return ([p.centroid for p in peaks], [p.fwhm for p in peaks],
            [p.area for p in peaks])


def _confirmation_note(count):
    if count >= 3:
        return None
    if count == 2:
        return ("Two pairs fix a straight line exactly, so nothing confirms "
                "this pairing; it is the only one the peaks allow.")
    return "One pair beyond the two that fix the line confirms this pairing."


def pair_peaks(peaks_ch1, peaks_ch0, source_lines=None):
    """Match CH1's peaks to CH0's, or report why not.

    Without `source_lines`: CH0's fitted centroids play the source lines
    and its areas the line intensities, and auto_calibrate's machinery
    finds the straight CH1 -> CH0 relation that explains the most peaks
    (see _assign); with four or more pairs found, pairs the straight
    line was too stiff to reach are added under a quadratic (_extend).
    The efficiency test stays on: for a peak both
    detectors see, area(CH1)/area(CH0) is the ratio of the two detectors'
    efficiencies, a smooth function of energy, while a wrong pairing puts
    unrelated areas side by side and scatters it. It only judges five or
    more pairs, so it never fires in the two-and-three-pair regime. Its
    one blind spot here: a peak cut by a different threshold in each
    detector breaks the smoothness at the bottom of the range, so such a
    pair may come back flagged suspect (unticked) rather than wrong.

    With `source_lines` (SourceLines from a .sou file): each channel is
    identified against the source's energies separately, by the same
    rules, and peaks with the same energy are paired. This is the route
    for a calibration source with lines across the whole range, where the
    gain is too non-linear for one straight line to pair the peaks.
    """
    peaks_ch1 = list(peaks_ch1)
    peaks_ch0 = list(peaks_ch0)
    if source_lines is None:
        c1, f1, a1 = _peak_arrays(peaks_ch1)
        c0, _f0, a0 = _peak_arrays(peaks_ch0)
        found = _assign(c1, f1, a1, c0, a0)
        if found.reason:
            return PairResult(reason=found.reason)
        index_of = {}
        for k, p in enumerate(peaks_ch0):
            index_of.setdefault(float(p.centroid), k)
        matched = [(k, index_of[ref]) for k, ref in found.pairs]
        if len(matched) >= MIN_PAIRS_QUADRATIC:
            matched = _extend(matched, peaks_ch1, peaks_ch0)
        pairs = [_pair(peaks_ch1[k1], peaks_ch0[k0], None, k1 not in found.suspect)
                 for k1, k0 in matched]
        count = len(pairs)
        return PairResult(pairs, method=found.method, confirmations=count - 2,
                          note=_confirmation_note(count))

    lines = list(source_lines)
    energies = [line.energy for line in lines]
    intensities = [line.intensity for line in lines]
    identified = []
    for name, peaks in (("CH1", peaks_ch1), ("CH0", peaks_ch0)):
        c, f, a = _peak_arrays(peaks)
        found = _assign(c, f, a, energies, intensities)
        if found.reason:
            return PairResult(reason=f"{name} could not be identified: {found.reason}")
        identified.append({energy: (k, k not in found.suspect)
                           for k, energy in found.pairs})
    by_energy1, by_energy0 = identified
    common = sorted(set(by_energy1) & set(by_energy0))
    if len(common) < 2:
        return PairResult(reason=(
            f"only {len(common)} line(s) were identified in both channels; at "
            f"least two are needed"))
    pairs = []
    for energy in common:
        k1, ok1 = by_energy1[energy]
        k0, ok0 = by_energy0[energy]
        pairs.append(_pair(peaks_ch1[k1], peaks_ch0[k0], energy, ok1 and ok0))
    return PairResult(pairs, method="source", confirmations=len(pairs) - 2,
                      note=_confirmation_note(len(pairs)))


def _extend(matched, peaks_ch1, peaks_ch0):
    """`matched` [(CH1 index, CH0 index)] grown by the pairs a quadratic
    through it reveals.

    The assignment is already settled and confirmed by the time this
    runs (four or more pairs, through every check of match); this only
    collects pairs a straight line was too stiff to reach. A CH1 peak is
    added only when its mapped position lands within INLIER_FWHM of the
    expected CH0 width of an unpaired CH0 peak, that CH0 peak's own
    nearest unpaired CH1 peak is this one (mutual), and the runner-up
    CH0 peak is more than twice as far (sou_io.match_line's rule for an
    unambiguous nearest). Anything less is left unpaired, not guessed.
    """
    c1 = np.array([p.centroid for p in peaks_ch1], dtype=float)
    c0 = np.array([p.centroid for p in peaks_ch0], dtype=float)
    ok1 = np.array([p.area > 0 and math.isfinite(p.fwhm) for p in peaks_ch1])
    ok0 = np.array([p.area > 0 and math.isfinite(p.fwhm) for p in peaks_ch0])
    width0 = expected_widths(c0, [p.fwhm for p in peaks_ch0])
    matched = list(matched)
    for _round in range(_EXTEND_ROUNDS):
        x = c1[[k1 for k1, _k0 in matched]]
        y = c0[[k0 for _k1, k0 in matched]]
        coefficients = np.polynomial.polynomial.polyfit(x, y, 2)
        free1 = [k for k in range(c1.size) if ok1[k] and k not in {a for a, _b in matched}]
        free0 = [k for k in range(c0.size) if ok0[k] and k not in {b for _a, b in matched}]
        if not free1 or not free0:
            break
        predicted = np.polynomial.polynomial.polyval(c1[free1], coefficients)
        added = []
        for i, k1 in enumerate(free1):
            d = np.abs(c0[free0] - predicted[i])
            order = np.argsort(d)
            j = free0[order[0]]
            if d[order[0]] > auto_calibrate.INLIER_FWHM * width0[j]:
                continue
            if order.size > 1 and not d[order[1]] > 2.0 * d[order[0]]:
                continue
            back = np.abs(predicted - c0[j])
            if free1[int(np.argmin(back))] != k1:
                continue
            added.append((k1, j))
        if not added:
            break
        matched.extend(added)
    return sorted(matched)


def _pair(q1, q0, energy, included):
    return GainPair(ch1=q1.centroid, ch1_err=q1.centroid_err,
                    ch0=q0.centroid, ch0_err=q0.centroid_err,
                    energy=energy, included=bool(included),
                    fwhm1=q1.fwhm, fwhm0=q0.fwhm, area1=q1.area, area0=q0.area)


# ---------------------------------------------------------------- fitting


class GainMatch:
    """The fitted map ch0' = P(ch1), coefficients in increasing power.

    `covariance` is the coefficients' covariance from the weighted fit,
    NOT scaled by the reduced chi-square: it is what the centroid errors
    imply, and `reduced_chi2` (None with no degrees of freedom) says
    whether they were right. `residuals` are ch0 - P(ch1) for every pair,
    included or not, so the dialog can plot the excluded ones too.
    """

    def __init__(self, coefficients, pairs=(), covariance=None, residuals=None,
                 rms=float("nan"), reduced_chi2=None, source=None, date=None,
                 nchan=DEFAULT_NCHAN):
        self.coefficients = [float(c) for c in coefficients]
        if len(self.coefficients) not in (2, 3):
            raise GainMatchError("A gain map has order 1 or 2.")
        self.pairs = list(pairs)
        n = len(self.coefficients)
        self.covariance = (np.zeros((n, n)) if covariance is None
                           else np.asarray(covariance, dtype=float))
        self.residuals = [] if residuals is None else [float(r) for r in residuals]
        self.rms = float(rms)
        self.reduced_chi2 = None if reduced_chi2 is None else float(reduced_chi2)
        self.source = source
        self.date = date or datetime.datetime.now().isoformat(timespec="seconds")
        self.nchan = int(nchan)

    @property
    def order(self):
        return len(self.coefficients) - 1

    @property
    def errors(self):
        return [float(math.sqrt(max(v, 0.0))) for v in np.diag(self.covariance)]

    def map(self, ch1_channels):
        """P at `ch1_channels` (centre coordinates), as floats."""
        return np.polynomial.polynomial.polyval(
            np.asarray(ch1_channels, dtype=float), self.coefficients)

    def derivative(self, ch1_channels):
        d = np.polynomial.polynomial.polyder(self.coefficients)
        return np.polynomial.polynomial.polyval(np.asarray(ch1_channels, dtype=float), d)

    def apply(self, energies_int, seed=DEFAULT_SEED, nchan=None):
        """(CH0-scale channels as int64, in-range mask) for integer CH1 ADC
        values.

        Each event is placed uniformly within its channel, u ~ U(0, 1)
        from `seed`, mapped, and binned: floor(P(E - 1/2 + u) + 1/2) (see
        the module docstring for why not floor(P(E + u))). A channel's
        whole width is mapped continuously, so the output has no picket
        fence however non-integer the gain; the seed makes the sort
        reproducible. Channels outside 0 ... nchan-1 are returned as they
        are and marked False in the mask, so the caller can count what
        fell off the edge.
        """
        e = np.asarray(energies_int)
        n = self.nchan if nchan is None else int(nchan)
        rng = np.random.default_rng(seed)
        u = rng.random(e.shape)
        mapped = self.map(e.astype(float) - 0.5 + u)
        channels = np.floor(mapped + 0.5)
        finite = np.isfinite(channels)
        channels = np.where(finite, channels, -1.0).astype(np.int64)
        mask = finite & (channels >= 0) & (channels < n)
        return channels, mask

    def to_dict(self):
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "mapping": "ch0 = sum(coefficients[k] * ch1**k)",
            "reference": "CH0",
            "mapped": "CH1",
            "channel_convention": ("centroids on x = arange(n); channel i "
                                   "covers [i - 1/2, i + 1/2)"),
            "order": self.order,
            "coefficients": self.coefficients,
            "covariance": self.covariance.tolist(),
            "rms": _json_float(self.rms),
            "reduced_chi2": self.reduced_chi2,
            "nchan": self.nchan,
            "source": self.source,
            "date": self.date,
            "pairs": [p.to_dict() for p in self.pairs],
            "residuals": self.residuals,
        }

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict) or d.get("format") != FORMAT_NAME:
            raise GainMatchError("This is not a SpectraTools gain-match file.")
        if int(d.get("version", 0)) > FORMAT_VERSION:
            raise GainMatchError(
                f"This gain-match file is version {d.get('version')}; this "
                f"version of SpectraTools reads up to {FORMAT_VERSION}.")
        try:
            coefficients = [float(c) for c in d["coefficients"]]
        except (KeyError, TypeError, ValueError):
            raise GainMatchError("The gain-match file has no usable coefficients.") from None
        if not all(math.isfinite(c) for c in coefficients):
            raise GainMatchError("The gain-match file has a non-finite coefficient.")
        rms = d.get("rms")
        return cls(coefficients,
                   pairs=[GainPair.from_dict(p) for p in d.get("pairs", [])],
                   covariance=d.get("covariance"),
                   residuals=d.get("residuals"),
                   rms=float("nan") if rms is None else rms,
                   reduced_chi2=d.get("reduced_chi2"),
                   source=d.get("source"), date=d.get("date"),
                   nchan=d.get("nchan", DEFAULT_NCHAN))

    def save(self, path):
        with open(path, "w", encoding="utf-8", newline="") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GainMatchError(f"Cannot read {path}: {exc}") from None
        return cls.from_dict(data)

    def __repr__(self):
        terms = ", ".join(f"{c:.6g}" for c in self.coefficients)
        return f"GainMatch(order {self.order}: [{terms}], rms={self.rms:.3g} ch)"


def fit_gain(pairs, order=None, source=None, nchan=DEFAULT_NCHAN):
    """Weighted least-squares P through the included pairs.

    `pairs` is a PairResult or a list of GainPairs. `order` None is 2
    with at least MIN_PAIRS_QUADRATIC included pairs and 1 otherwise; an
    explicit 2 with fewer is refused rather than quietly lowered, since
    the caller asked for something the pairs cannot support.

    Both centroids carry errors. The fit treats ch1 as exact and gives
    each point the effective variance

        sigma^2 = sigma_ch0^2 + (P'(ch1) * sigma_ch1)^2,

    the first-order propagation of the ch1 error through the slope
    (the "effective variance" method). It is exact for a straight line
    when the errors are small against the curvature scale, which a
    centroid error of a fraction of a channel always is; the weights are
    recomputed from the current slope _WEIGHT_ITERATIONS times.

    A map that is not increasing anywhere on 0 ... nchan is refused: a
    quadratic turning over inside the ADC range would send two CH1
    energies to the same CH0 channel.
    """
    if isinstance(pairs, PairResult):
        if not pairs.ok:
            raise GainMatchError(pairs.reason or "There are no pairs to fit.")
        pairs = pairs.pairs
    pairs = list(pairs)
    used = [p for p in pairs if p.included]
    if len(used) < 2:
        raise GainMatchError(
            f"{len(used)} pair(s) are included; at least two are needed to "
            f"fit a gain.")
    if order is None:
        order = 2 if len(used) >= MIN_PAIRS_QUADRATIC else 1
    if order not in (1, 2):
        raise GainMatchError("The polynomial order must be 1 or 2.")
    if order == 2 and len(used) < MIN_PAIRS_QUADRATIC:
        raise GainMatchError(
            f"Order 2 needs at least {MIN_PAIRS_QUADRATIC} included pairs; "
            f"there are {len(used)}. Use order 1.")
    x = np.array([p.ch1 for p in used], dtype=float)
    y = np.array([p.ch0 for p in used], dtype=float)
    sx = np.maximum(np.nan_to_num(np.array([p.ch1_err for p in used], dtype=float),
                                  nan=_MIN_CENTROID_ERR), _MIN_CENTROID_ERR)
    sy = np.maximum(np.nan_to_num(np.array([p.ch0_err for p in used], dtype=float),
                                  nan=_MIN_CENTROID_ERR), _MIN_CENTROID_ERR)
    if np.unique(x).size <= order:
        raise GainMatchError("The included pairs do not span enough distinct channels.")

    design = np.vander(x, order + 1, increasing=True)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(_WEIGHT_ITERATIONS):
        slope = np.polynomial.polynomial.polyval(
            x, np.polynomial.polynomial.polyder(coefficients))
        w = 1.0 / (sy ** 2 + (slope * sx) ** 2)
        normal = design.T @ (design * w[:, None])
        covariance = np.linalg.inv(normal)
        coefficients = covariance @ (design.T @ (w * y))

    result = GainMatch(coefficients, pairs=pairs, covariance=covariance,
                       source=source, nchan=nchan)
    edges = np.linspace(0.0, float(nchan), 257)
    if np.any(result.derivative(edges) <= 0):
        raise GainMatchError(
            "The fitted map is not increasing over the whole ADC range, so two "
            "CH1 channels would land on one CH0 channel. Use order 1 or "
            "exclude the outlying pair.")
    all_x = np.array([p.ch1 for p in pairs], dtype=float)
    all_y = np.array([p.ch0 for p in pairs], dtype=float)
    result.residuals = (all_y - result.map(all_x)).tolist()
    used_res = y - result.map(x)
    result.rms = float(np.sqrt(np.mean(used_res ** 2)))
    ndf = len(used) - (order + 1)
    if ndf > 0:
        chi2 = float(np.sum(w * used_res ** 2))
        result.reduced_chi2 = chi2 / ndf
    return result


def gain_match(spectrum_ch1, spectrum_ch0, source_lines=None, order=None,
               sensitivity=peak_search.DEFAULT_SENSITIVITY, source=None,
               nchan=DEFAULT_NCHAN):
    """find_peaks on both, pair_peaks, fit_gain: the whole step in one
    call, for a script or a test. Raises GainMatchError with the reason
    if any step fails; the dialog calls the steps one by one instead so
    the user can see and override each."""
    peaks1 = find_peaks(spectrum_ch1, sensitivity)
    peaks0 = find_peaks(spectrum_ch0, sensitivity)
    paired = pair_peaks(peaks1, peaks0, source_lines)
    if not paired.ok:
        raise GainMatchError(f"The peaks could not be paired: {paired.reason}")
    return fit_gain(paired, order=order, source=source, nchan=nchan)
