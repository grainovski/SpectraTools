import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfcx

FWHM_FACTOR = 2.3548200450309493  # 2*sqrt(2*ln(2))
SQRT_2PI = 2.5066282746310002  # sqrt(2*pi)

TAIL_FRACTION_MAX = 0.3
TAIL_BETA_MIN = 0.1

#: tail_beta is bounded ABOVE at this many sigma, as well as below at
#: TAIL_BETA_MIN.
#:
#: Without an upper bound, a fit on data with no real tail drives
#: tail_fraction to ~0, at which point the tail term stops contributing to
#: the model at all and d(model)/d(tail_beta) vanishes -- so beta is
#: unconstrained and wanders. Observed reaching 7e18. The fitted CURVE stays
#: perfectly good (reduced chi-square 0.88-1.12 across the cases measured),
#: which is what makes this so easy to miss, but hypermet_area's tail term
#: is 2*r*beta/erfcx(y) and tends to 2*r*beta as beta grows. The product is
#: enormous even when r is negligible: the reported area came out 6e18
#: counts for a peak of amplitude 3800. In a randomised sweep, 16 of ~150
#: tailed fits reported an area more than 100x their own Gaussian core.
#:
#: 20 sigma is chosen to be permissive rather than tight. A real detector
#: tail has beta of order sigma; fitting data generated FROM the hypermet
#: shape recovers a true beta of 1 sigma as 0.94, 2 sigma as 1.88 and
#: 10 sigma as 10.1, all with areas correct to better than 0.7%, so the
#: bound does not bind on genuine tails -- it only stops the runaway when
#: there is no tail there to fit.
#:
#: This is not free, and the trade was made deliberately. The unbounded fit
#: does reach a slightly lower chi-square in about 12% of tailed fits (worst
#: observed increase in reduced chi-square: 0.05), because a beta of 1e18 is
#: effectively a flat pedestal under the peak and absorbs background
#: mismatch. Absorbing background mismatch is the background's job, not the
#: tail's, and paying 0.05 in reduced chi-square to stop reporting areas
#: wrong by fifteen orders of magnitude is the right way round.
TAIL_BETA_MAX_SIGMA = 20.0

_CUR_RATIO = 1e-12
_CUR_LAMBDA_START = 1e-3
_CUR_MIN_LAMBDA = 1e-20
_CUR_MAX_LAMBDA = 1e30
_CUR_INC_LAMBDA = 10.0
_CUR_MAX_ITERATIONS = 50


def hypermet_left_tail(x, position, sigma, r, beta):
    """gf3's Hypermet tail term (left-side only; the step-background
    term is out of scope, per the design spec):
        (1-r)*exp(-w^2) + r*exp(dx/beta)*erfc(w+y)/erfc(y)
    where dx = x - position, w = dx/(sigma*sqrt(2)),
    y = sigma/(beta*sqrt(2)). Ported from srcRW/gf3_subs.c's eval();
    verified numerically during design that beta > 0 biases the tail
    toward lower x (left), matching real low-energy detector tailing.
    Public (no leading underscore) because fit_mode.py's committed-fit
    overlay drawing reuses this exact formula to redraw tailed fits.

    Re-verified term-by-term against the real gf3 source on 2026-08-16
    (srcRW/gf3_subs.c:2914-2966), which had never been possible before
    -- the vendored tree is untracked and only reached this machine
    then. Every term matches, including one that looks like it does
    not: gf3 writes y as FWHM/(beta*3.33021838), and since
    3.33021838 == 2.35482*sqrt(2) that is exactly sigma/(beta*sqrt(2))
    above. The |dx/beta| > 12 cutoff below is gf3's own (:2949-2953).

    THREE of gf3's cutoffs are deliberately NOT replicated. All three
    are float32-era numerical safeguards; this port computes in float64,
    where the exact value is both computable and more accurate:
      * `|w| > 4  -> gaussian core := 0` (:2918-2921). Differs by ~1e-7
        of peak amplitude.
      * `|w+y| > 4 -> erfc(w+y) := 0 or 2` (:2957-2962). Up to ~1e-3 of
        peak amplitude on the right flank, where gf3 truncates a
        genuinely small but non-zero huge*tiny product.
      * `y > 4 -> tail disabled entirely` (:2896-2898, gf3's `notail`).
        This one is NOT a precision detail: it switches the model to a
        pure Gaussian, and measured up to 0.28 of peak amplitude of
        difference. It is reachable in practice -- a fit of genuinely
        short-tailed data was observed converging to y ~ 20 with beta
        pinned at TAIL_BETA_MIN.

        DECIDED 2026-08-16, by the user, after being shown that
        measurement: keep the true Hypermet function here and do NOT
        replicate gf3's switch. This port computes in float64, so the
        numerical danger the safeguard exists to avoid does not apply,
        and collapsing to a Gaussian would discard real tail shape in
        exactly the short-tail fits where it is most visible.

        This is therefore a deliberate divergence from the reference,
        not an unported detail -- do not "fix" it in a future parity
        pass. tests/test_peak_fit.py::
        test_tail_stays_active_where_gf3_would_disable_it locks it in.
    """
    x = np.asarray(x, dtype=float)
    dx = x - position
    w = dx / (sigma * np.sqrt(2))
    gaussian_core = np.exp(-w ** 2)
    y = sigma / (beta * np.sqrt(2))

    # Evaluated via the scaled complementary error function, which is
    # algebraically identical to the formula in the docstring and free of
    # the huge*tiny product that made a cutoff necessary:
    #
    #   exp(dx/beta) * erfc(w+y) / erfc(y)
    #     = exp(dx/beta) * erfcx(w+y)exp(-(w+y)^2) / erfcx(y)exp(-y^2)
    #     = exp(dx/beta - w^2 - 2wy) * erfcx(w+y)/erfcx(y)
    #     = exp(-w^2) * erfcx(w+y)/erfcx(y)
    #
    # because 2wy = 2*(dx/(sigma*sqrt2))*(sigma/(beta*sqrt2)) = dx/beta
    # exactly -- the same cancellation that makes the area integrable in
    # closed form (see hypermet_area).
    #
    # This REPLACES gf3's `|dx/beta| > 12 -> tail := 0` cutoff
    # (:2949-2953), which was previously ported. gf3 needs it because it
    # evaluates exp(dx/beta) directly, which overflows long before erfc()
    # can suppress it; nothing here evaluates that quantity at all. The
    # cutoff was not free: it discarded up to 65% of the tail's own area
    # for a short tail (measured at sigma=8, beta=0.3, r=1 -- and 14% at
    # beta=1), which would have left the reported area disagreeing with
    # the curve actually fitted and drawn. Dropping it is the same
    # reasoning already applied to gf3's other three float32-era
    # safeguards listed above, and the same reasoning behind the user's
    # decision to keep the true Hypermet function rather than gf3's
    # collapse-to-Gaussian switch.
    #
    # erfcx(y) needs no guard of its own: beta >= TAIL_BETA_MIN > 0 and
    # sigma > 0, so y > 0, where erfcx lies in (0, 1] -- it can neither
    # overflow nor underflow, which is why the old erfc(y) < 1e-300 check
    # is gone too.
    #
    # erfcx(z) grows like 2exp(z^2) and overflows for strongly negative z,
    # which is the deep left flank -- exactly where a LEFT tail carries
    # its weight, so it cannot simply be floored to zero. exp(-w^2) is
    # underflowing to zero there at the same time; only their product is
    # well scaled. Reflecting with erfcx(z) = 2exp(z^2) - erfcx(-z) gives
    #
    #   exp(-w^2) * erfcx(z) = 2exp(z^2 - w^2) - exp(-w^2)*erfcx(-z)
    #                        = 2exp(dx/beta + y^2) - exp(-w^2)*erfcx(-z)
    #
    # using z^2 - w^2 = 2wy + y^2 = dx/beta + y^2. In that branch the
    # first term is computed directly (its exponent is always strongly
    # negative there: z < -25 forces dx/beta + y^2 < -50y - y^2 < 0, so it
    # underflows gracefully instead of overflowing) and the second is
    # dropped -- at the crossover it is ~1e-275 of the peak amplitude
    # against a tail value of ~3e-2, i.e. 273 orders of magnitude down.
    #
    # A first attempt floored the tail to zero for z < -25 instead. That
    # is wrong for a LONG tail: with sigma=1.2, beta=10 the crossover
    # falls at dx = -43, where the true tail is still 3% of the peak
    # amplitude, and the area came out 0.4% low. Caught by
    # test_hypermet_area_matches_numeric_integration_of_the_fitted_shape.
    z = w + y
    erfcx_y = erfcx(y)
    near = z > -25.0
    # np.where evaluates BOTH branches for every element and only then
    # selects, so each branch has to stay finite on the other's inputs or
    # it raises overflow warnings for values that are immediately thrown
    # away. erfcx's argument is pinned inside the near branch's domain, and
    # the far branch's exponent is clamped at 0 -- which is a no-op where
    # that branch is actually used, since z <= -25 forces
    # dx/beta + y^2 <= -50y - y^2 < 0, and merely tames the discarded
    # near-branch values where dx/beta can be large and positive.
    far_exponent = np.minimum(dx / beta + y * y, 0.0)
    tail = np.where(
        near,
        gaussian_core * erfcx(np.where(near, z, 0.0)) / erfcx_y,
        2.0 * np.exp(far_exponent) / erfcx_y,
    )

    return (1 - r) * gaussian_core + r * tail


def hypermet_area(amplitude, sigma, r, beta):
    """Exact analytic integral of `amplitude * hypermet_left_tail(...)`
    over all x:

        A * [ (1-r) * sigma*sqrt(2*pi) + 2*r*beta / erfcx(y) ]

    with y = sigma/(beta*sqrt(2)) as everywhere else.

    Derivation: the Gaussian core integrates to sigma*sqrt(2*pi) in the
    usual way. For the tail, integrating by parts leaves
    (2b/(a*sqrt(pi))) * integral of exp(a*u - (b*u+c)^2), and completing
    the square there cancels the linear term EXACTLY -- because
    c = a/(2b) is precisely what y = sigma/(beta*sqrt(2)) encodes. What
    remains is elementary and gives 2*beta*exp(-y^2)/erfc(y), written
    above as 2*beta/erfcx(y) so that neither factor underflows for large
    y. Verified against dense numeric integration of the untruncated
    shape across sigma 1.2-8, beta 0.3-40 and r 0.05-1: agreement to
    4.3e-16, i.e. machine precision.

    Two limits worth knowing, both of which the tests pin:
      * r = 0 gives sigma*sqrt(2*pi), the plain Gaussian area.
      * y -> large (short tail) gives erfcx(y) -> 1/(y*sqrt(pi)) and
        hence 2*r*beta*y*sqrt(pi) = r*sigma*sqrt(2*pi), so the total
        tends to sigma*sqrt(2*pi) from above -- the tail degenerates
        toward the core rather than diverging.

    This replaces `amplitude * sigma * sqrt(2*pi)`, which integrated the
    Gaussian core ALONE and so understated every tailed peak's area by
    whatever the tail held: measured 5.0% at r=0.1, 13.6% at r=0.3, and
    41.1% for a long tail (sigma=3, beta=10, r=0.3).
    """
    y = sigma / (beta * np.sqrt(2.0))
    return amplitude * ((1.0 - r) * sigma * SQRT_2PI + 2.0 * r * beta / erfcx(y))


class FitError(Exception):
    """Raised when a fit cannot be performed or does not converge."""


@dataclass
class PeakResult:
    position: float
    position_err: float
    fwhm: float
    fwhm_err: float
    area: float
    area_err: float
    amplitude: float
    sigma: float
    amplitude_err: float = 0.0
    sigma_err: float = 0.0
    full_area: float = 0.0
    full_area_err: float = 0.0


@dataclass
class FitResult:
    left_bg_region: tuple
    right_bg_region: tuple
    fit_region: tuple
    background_slope: float
    background_intercept: float
    peaks: list
    link_widths: bool = True
    tail_fraction: float = None
    tail_fraction_err: float = None
    tail_beta: float = None
    tail_beta_err: float = None
    fixed_params: dict = field(default_factory=dict)
    visible: bool = True
    timestamp: str = None
    gross_area: float = 0.0
    gross_area_err: float = 0.0
    net_area: float = 0.0
    net_area_err: float = 0.0
    reduced_chi2: float = None
    # ((x1, var1), (x2, var2)) from background_anchors() -- enough to
    # reproduce the background's uncertainty band without keeping the
    # spectrum around, so a redraw does not have to re-sum the regions and
    # a restored fit can still draw it. Empty for a result built before
    # this field existed (or by hand in a test), which
    # background_level_error() reports as zero error rather than crashing.
    background_anchors: tuple = ()
    # (reference, var_c0, cov_c0c1, var_c1) when the background was fitted
    # jointly with the peaks (F5). Takes precedence over the anchors, since
    # in that mode the two marked regions only seeded the line -- the fit
    # decided it.
    background_covariance: tuple = ()
    fit_background: bool = False

    def background_level_error(self, at):
        """Standard error of this fit's background line at position(s)
        `at`, in counts. Zero everywhere if neither error model is
        available."""
        at = np.asarray(at, dtype=float)
        if self.background_covariance:
            # HDTV's PolyBg::EvalError for a linear background:
            # sqrt(v C v) with v = [1, x - reference].
            reference, var_c0, cov, var_c1 = self.background_covariance
            dx = at - reference
            return np.sqrt(np.maximum(var_c0 + 2.0 * cov * dx + var_c1 * dx * dx, 0.0))
        if self.background_anchors:
            return anchor_error(self.background_anchors, at)
        return np.zeros_like(at)


@dataclass
class IntegrationResult:
    left_bg_region: tuple
    right_bg_region: tuple
    fit_region: tuple
    background_density: float
    gross_area: float
    gross_area_err: float
    gross_centroid: float
    gross_centroid_err: float
    gross_fwhm: float
    gross_fwhm_err: float
    gross_skewness: float
    gross_skewness_err: float
    background_area: float
    background_area_err: float
    background_centroid: float
    background_centroid_err: float
    background_fwhm: float
    background_fwhm_err: float
    background_skewness: float
    background_skewness_err: float
    net_area: float
    net_area_err: float
    net_centroid: float
    net_centroid_err: float
    net_fwhm: float
    net_fwhm_err: float
    net_skewness: float
    net_skewness_err: float
    timestamp: str = None
    visible: bool = True

    @property
    def has_background(self):
        return self.left_bg_region is not None


def _region_centroid(x, y, region):
    lo, hi = region
    mask = (x >= lo) & (x <= hi)
    if not np.any(mask):
        raise FitError(f"Background region {region} contains no data")
    return float(np.mean(x[mask])), float(np.mean(y[mask]))


def _region_mean_variance(x, y, region, variance=None):
    """Variance of a background region's MEAN level.

    The mean of n channels has variance sum(var_i)/n**2. With no supplied
    variance the channels are Poisson, so var_i is the count itself,
    floored at zero -- a region of an already-subtracted spectrum can hold
    negative counts, which carry no Poisson variance to read off.
    """
    lo, hi = region
    mask = (x >= lo) & (x <= hi)
    n = int(np.count_nonzero(mask))
    if n == 0:
        raise FitError(f"Background region {region} contains no data")
    if variance is None:
        total = float(np.sum(np.maximum(y[mask], 0.0)))
    else:
        total = float(np.sum(np.asarray(variance, dtype=float)[mask]))
    return total / float(n * n)


def background_anchors(x, y, left_bg_region, right_bg_region, variance=None):
    """((x1, var1), (x2, var2)) -- the two independent anchors the
    background line is drawn through, each with the variance of its own
    region mean.

    Returned as plain numbers so a FitResult can carry them and reproduce
    its own uncertainty band later without holding on to the spectrum.
    """
    left_x, _ = _region_centroid(x, y, left_bg_region)
    right_x, _ = _region_centroid(x, y, right_bg_region)
    if right_x == left_x:
        raise FitError("Background regions must not share the same mean channel")
    return (
        (left_x, _region_mean_variance(x, y, left_bg_region, variance)),
        (right_x, _region_mean_variance(x, y, right_bg_region, variance)),
    )


def anchor_error(anchors, at):
    """Standard error of the two-point background line at position(s)
    `at`. Single source of the formula, shared by background_error() and
    FitResult.background_level_error()."""
    (left_x, var_left), (right_x, var_right) = anchors
    t = (np.asarray(at, dtype=float) - left_x) / (right_x - left_x)
    return np.sqrt((1.0 - t) ** 2 * var_left + t ** 2 * var_right)


def background_error(x, y, left_bg_region, right_bg_region, at, variance=None):
    """Standard error of the fitted background level at position(s) `at`.

    Our background is not a least-squares polynomial but a line through
    the MEANS of the two marked regions, which makes its uncertainty
    exact rather than approximate. Writing t for the fractional position
    between the two region centroids,

        bg(x) = (1 - t) * y1 + t * y2,   t = (x - x1) / (x2 - x1)

    so bg is a linear combination of two independent measurements -- the
    regions are disjoint, so there is no covariance term -- and

        var(bg(x)) = (1 - t)**2 var(y1) + t**2 var(y2)

    This is the counterpart of HDTV's PolyBg::EvalError, which evaluates
    sqrt(sum_ij cov(c_i,c_j) x^i x^j) over a fitted polynomial's
    covariance matrix (src/fit/PolyBg.cxx:238-262). A two-point line has
    no covariance matrix to read, but it does not need one: the two
    anchors ARE the independent parameters.

    Note the shape this implies. The error is smallest at the two
    centroids and grows linearly outside them, so extrapolating the
    background under a peak far from both regions is exactly where it is
    least certain -- which is the useful thing for a user to see.
    """
    return anchor_error(
        background_anchors(x, y, left_bg_region, right_bg_region, variance), at
    )


_channel_indices_cache = {}


def channel_indices(length):
    """`[0, 1, ... length-1]` as float64 -- the channel-number axis for a
    spectrum of `length` channels.

    Cached by length. Safe to cache because the array depends ONLY on
    the channel count, never on the counts themselves: a spectrum whose
    data is mutated in place (Multiply, Add/Subtract) keeps the exact
    same channel axis, so unlike a cached fit result this can never go
    stale. A spectrum whose LENGTH changes (Rebin) simply lands on a
    different key.

    Returned read-only so a caller that tries to modify it fails loudly
    instead of silently corrupting the axis for every other caller
    sharing the same cached array.
    """
    cached = _channel_indices_cache.get(length)
    if cached is None:
        cached = np.arange(length, dtype=float)
        cached.flags.writeable = False
        _channel_indices_cache[length] = cached
    return cached


def compute_background(x, y, left_bg_region, right_bg_region):
    left_x, left_y = _region_centroid(x, y, left_bg_region)
    right_x, right_y = _region_centroid(x, y, right_bg_region)
    if right_x == left_x:
        raise FitError("Background regions must not share the same mean channel")
    slope = (right_y - left_y) / (right_x - left_x)
    intercept = left_y - slope * left_x
    return slope, intercept


def _parameter_names(n_peaks, link_widths, enable_left_tail, fit_background=False):
    """Canonical, ordered list of every fittable parameter's name for
    a given (n_peaks, link_widths, enable_left_tail, fit_background)
    configuration.
    This is the single source of truth for parameter identity, shared
    by the initial guess, bounds, model function, and (from Task 2)
    fixed-parameter handling in fit_peaks(). Public (no leading
    underscore) because fit_mode.py's Fit Parameters panel needs this
    same ordering to know which rows to show."""
    names = []
    for i in range(n_peaks):
        names.append(f"amp_{i}")
        names.append(f"pos_{i}")
        if not link_widths:
            names.append(f"sigma_{i}")
    if link_widths:
        names.append("sigma")
    if enable_left_tail:
        names.append("tail_fraction")
        names.append("tail_beta")
    if fit_background:
        # Appended last, as HDTV appends its internal-background
        # coefficients after the peak parameters. bg_c0 is the level at the
        # fit region's CENTRE, not at channel 0 -- see _make_model.
        names.append("bg_c0")
        names.append("bg_c1")
    return names


def _unpack_named(values_by_name, n_peaks, link_widths, enable_left_tail):
    """Extracts (amplitudes, positions, sigmas, tail_fraction, tail_beta)
    from a {name: value} mapping covering every name _parameter_names()
    would produce for this configuration. `sigmas` is always length
    n_peaks (the shared value repeated if linked); tail_fraction/
    tail_beta are None if enable_left_tail is False.

    Background coefficients, when present, are read directly by
    _make_model rather than returned here -- every existing caller unpacks
    exactly five values."""
    amplitudes = [values_by_name[f"amp_{i}"] for i in range(n_peaks)]
    positions = [values_by_name[f"pos_{i}"] for i in range(n_peaks)]
    if link_widths:
        sigmas = [values_by_name["sigma"]] * n_peaks
    else:
        sigmas = [values_by_name[f"sigma_{i}"] for i in range(n_peaks)]
    tail_fraction = values_by_name.get("tail_fraction")
    tail_beta = values_by_name.get("tail_beta")
    return amplitudes, positions, sigmas, tail_fraction, tail_beta


def _make_model(names, free_names, fixed_params, n_peaks, link_widths, enable_left_tail,
                background_reference=None):
    """The model the optimiser sees: the sum of the peaks, plus a linear
    background when `background_reference` is given.

    `background_reference` is the channel the background's constant term is
    measured AT -- the fit region's centre, not channel 0. Fitting
    bg(x) = c0 + c1*x directly would make c0 the extrapolated level at
    channel 0, which for a peak at channel 3000 is a huge number almost
    perfectly anti-correlated with c1; the curvature matrix becomes
    near-singular for a reason that has nothing to do with the data.
    Centring on the region removes that. HDTV fits raw powers of x
    (PolyBg::_Eval is a Horner scheme in x) and so does not do this, but it
    is inverting a small matrix per fit where we are stepping a Marquardt
    iteration.
    """
    def evaluate(x, values_by_name, peak_indices=None, include_background=True):
        """The model, optionally restricted to a subset of its terms.

        `peak_indices=None` means every peak, which is the model proper.
        Naming a subset is what lets the numeric Jacobian avoid recomputing
        peaks that a perturbed parameter cannot possibly have moved -- see
        _parameter_influence.
        """
        amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_named(
            values_by_name, n_peaks, link_widths, enable_left_tail
        )
        chosen = range(n_peaks) if peak_indices is None else peak_indices
        result = np.zeros_like(x, dtype=float)
        for i in chosen:
            amplitude, position, sigma = amplitudes[i], positions[i], sigmas[i]
            if enable_left_tail:
                result = result + amplitude * hypermet_left_tail(
                    x, position, sigma, tail_fraction, tail_beta
                )
            else:
                result = result + amplitude * np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
        if include_background and background_reference is not None:
            result = result + (
                values_by_name["bg_c0"]
                + values_by_name["bg_c1"] * (x - background_reference)
            )
        return result

    def model(x, *free_values):
        values_by_name = dict(fixed_params)
        values_by_name.update(zip(free_names, free_values))
        return evaluate(x, values_by_name)

    # Attached rather than returned as a pair so every existing caller --
    # and every test that builds a model by hand -- keeps working unchanged.
    model.evaluate = evaluate
    return model


def _parameter_influence(free_names, n_peaks, link_widths):
    """Which terms of the model each free parameter can move, as a list
    parallel to `free_names`.

    Each entry is either None ("everything -- recompute the whole model")
    or a `(peak_indices, include_background)` pair naming the only terms
    that can have changed.

    This is what turns the numeric Jacobian from O(peaks**2) into O(peaks).
    A central difference of the FULL model when only one peak moved
    computes every other peak twice and then subtracts it from itself --
    work that is not merely wasted but actively lossy, since the unchanged
    terms are large and cancel, eating precision out of the small
    difference that is actually wanted.

    The dispatch is on the same name prefixes _parameter_names() already
    establishes as the single source of truth, and mirrors
    _build_param_damping's structure deliberately: an unrecognised name
    raises rather than being guessed at, because a wrong answer here would
    silently zero out part of a Jacobian column.

    Note which parameters are genuinely global. A LINKED sigma is shared by
    every peak, and both tail parameters are, so those still cost a full
    evaluation -- there is no shortcut to take. With linked widths that is
    one parameter in 2n+1; the other 2n become single-peak.
    """
    influence = []
    for name in free_names:
        if name.startswith(("amp_", "pos_")):
            influence.append(([int(name.split("_", 1)[1])], False))
        elif name.startswith("sigma_"):
            influence.append(([int(name.split("_", 1)[1])], False))
        elif name == "sigma":
            # Shared by every peak when widths are linked.
            influence.append(None)
        elif name in ("tail_fraction", "tail_beta"):
            influence.append(None)
        elif name in ("bg_c0", "bg_c1"):
            influence.append(([], True))
        else:
            raise FitError(f"No influence rule for parameter {name!r}")
    assert len(influence) == len(free_names), "influence list must be parallel to free_names"
    return influence


def _integral_sigma(x_fit, y_sub, peak_positions):
    """One common width seed for every peak, derived from the region's
    INTEGRAL rather than from measuring a peak. Ported from HDTV's
    TheuerkaufFitter::_Fit initial-parameter estimation.

    Take the total background-subtracted volume over the fit region and
    the sum of the peaks' amplitudes; if the peaks are Gaussians of a
    common width, volume = amplitude * sigma * sqrt(2*pi) summed over
    them, so

        sigma = sum(volume) / (sum(amplitude) * sqrt(2*pi))

    Returns None when the region cannot support the estimate (a
    non-positive volume or amplitude sum -- reachable on an
    already-subtracted spectrum), leaving the caller to fall back.

    Why this rather than measuring a width. Measured against two peaks of
    a known sigma=4, comparing what each approach actually SEEDS -- TV's
    halved measurement against this estimate:

        separation   old seed   this   truth
           0.5 sigma     2.07   2.13     4.0
           1.0 sigma     2.29   2.49     4.0
           2.0 sigma     3.43   3.52     4.0
           3.0 sigma     1.99   3.96     4.0
           6.0 sigma     2.00   4.00     4.0

    The gain is not overlap robustness, which was the expectation going
    in: under heavy overlap BOTH err narrow and land in much the same
    place. The gain is that this estimate is simply CORRECT once the peaks
    are resolved, where the halved measurement is a systematic factor of
    two too narrow no matter how clean the data is. Starting at the right
    width is what converges more reliably.

    Both estimators are biased under overlap, in opposite directions: a
    width measurement is INFLATED by the blend (6.86 against a true 4.0 at
    2 sigma separation), while this one is DEFLATED, because amplitudes
    sampled at each peak include their neighbours' contribution and so
    inflate sum(amplitude). Deflated is the forgiving direction -- starting
    narrow stops neighbouring peaks being swallowed before the optimiser
    can resolve them, which is the same reasoning that made TV's halved
    seed better than the unhalved one it replaced in v3.1.0. HDTV's own
    comment concedes its amplitude estimates degrade under heavy overlap.

    ADOPTED ON MEASUREMENT, per the v4.0.0 plan's gate. Paired
    (McNemar) sweep over 1200 randomised hard multiplets -- 2-5 peaks,
    sigma 1.2-10 ch, separations 0.5-2 sigma, Poisson noise: convergence
    failures fell from 41 to 12, with 33 cases converging only under this
    seed against 4 only under the old one (two-sided p < 0.0001).
    Accuracy on the 1155 cases both seeds fitted is a dead tie -- mean
    worst-peak position error 2.069 ch vs 2.066 ch, better in 571 and
    worse in 584 -- so this buys robustness, not precision. On an easier
    sweep (1-4 peaks, separations 1-4 sigma) both seeds converged
    essentially always and the two were indistinguishable, which is why
    the gate had to be measured in the hard regime to say anything at
    all.
    """
    total_volume = float(np.sum(y_sub))
    total_amplitude = 0.0
    for position in peak_positions:
        index = int(np.argmin(np.abs(x_fit - position)))
        total_amplitude += float(y_sub[index])
    if total_volume <= 0.0 or total_amplitude <= 0.0:
        return None
    sigma = total_volume / (total_amplitude * SQRT_2PI)
    if not math.isfinite(sigma) or sigma <= 0.0:
        return None
    return sigma


def _initial_guess(
    free_names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail,
    initial_guess_overrides=None, background_seed=None,
):
    lo, hi = fit_region
    region_width = hi - lo
    n_peaks = len(peak_positions)
    fallback_sigma0 = max(region_width / (4 * n_peaks), 1e-6)
    # `None` as the fallback so a genuine measurement can be told apart
    # from the heuristic below: TV's halving applies only to the former.
    measured_sigma = _measure_width(x_fit, y_sub, peak_positions, None)

    # Preferred: HDTV's integral-based estimate (see _integral_sigma).
    # Falls back to TV's halved width measurement, then to the region
    # heuristic, so a region whose counts cannot support the integral form
    # -- an already-subtracted spectrum summing to <= 0, say -- still gets
    # a usable seed.
    integral_sigma = _integral_sigma(x_fit, y_sub, peak_positions)
    if integral_sigma is not None:
        sigma0 = integral_sigma
    elif measured_sigma is None:
        sigma0 = fallback_sigma0
    else:
        # TV seeds HALF the sigma its own width measurement implies:
        # vsFitSetup.c:384 is `width = w * FWHM_TO_SIGMA`, where `w`
        # (the search loop at vsFitSetup.c:369-383) is the distance from
        # the peak centre out to the half-maximum crossing -- a HALF
        # width -- while FWHM_TO_SIGMA (vsCal.h:12) is 1/2.3548, a
        # FWHM-to-sigma factor. TV's `width` really is a sigma: its
        # Gaussian is exp(-dx*dx*0.5*ss) with ss = 1/sig^2
        # (vsFit.c:988, :1070). This port had been seeding the full
        # measured sigma, i.e. 2x TV.
        #
        # Restored to TV's scale because it is measurably better, not
        # only for parity: across 200 randomised fits (1-4 peaks, sigma
        # 1.2-10 ch, separations 1-4 sigma) the 2x seed failed to
        # converge 10 times where TV's scale never failed, and among the
        # fits both completed, mean position error was 1.04 ch vs 0.48
        # ch and worst-case 19.3 ch vs 10.2 ch. Median error was
        # identical (0.099 ch): the seeds only diverge on hard
        # multiplets, where starting narrow stops neighbouring peaks
        # being swallowed before the optimiser can resolve them.
        #
        # Applied here rather than inside _measure_width so that
        # function keeps meaning what its name and tests say -- the
        # peak's actual measured sigma. The fallback heuristic above is
        # this port's own, not TV's, so the halving does not apply to
        # it.
        sigma0 = max(measured_sigma / 2.0, 1e-6)

    guess_by_name = {}
    for i, pos in enumerate(peak_positions):
        idx = int(np.argmin(np.abs(x_fit - pos)))
        guess_by_name[f"amp_{i}"] = float(y_sub[idx])
        guess_by_name[f"pos_{i}"] = float(pos)
        if not link_widths:
            guess_by_name[f"sigma_{i}"] = sigma0
    if link_widths:
        guess_by_name["sigma"] = sigma0
    if enable_left_tail:
        guess_by_name["tail_fraction"] = 0.05
        # Seeded from a FULL width estimate, never the halved one:
        # tail_beta is a decay length, not a width, and TV's halving is
        # a quirk of its width-parameter initialisation specifically --
        # nothing in vsFitSetup.c propagates it to the tail parameter.
        # The integral estimate is already a full sigma, so it serves
        # here directly when available.
        tail_beta_reference = (
            integral_sigma if integral_sigma is not None
            else measured_sigma if measured_sigma is not None
            else fallback_sigma0
        )
        guess_by_name["tail_beta"] = max(tail_beta_reference, TAIL_BETA_MIN)
    if background_seed is not None:
        # Seeded from the two-point construction, which is a perfectly good
        # starting point even when it is not the final answer -- the same
        # role HDTV's "constant at the lowest bin" estimate plays, but
        # better informed since we already have the marked regions.
        guess_by_name["bg_c0"], guess_by_name["bg_c1"] = background_seed
    if initial_guess_overrides:
        guess_by_name.update(initial_guess_overrides)
    return [guess_by_name[name] for name in free_names]


def _build_param_damping(free_names, fixed_params, link_widths):
    """Translates the free-parameter name list into _ParamDamping
    metadata for _marquardt_fit, dispatching on each name's prefix --
    the same naming _parameter_names() already establishes as the
    single source of truth for parameter identity."""
    free_index = {name: i for i, name in enumerate(free_names)}
    damping = []
    for name in free_names:
        if name.startswith("amp_"):
            damping.append(_ParamDamping("amp"))
        elif name.startswith("pos_"):
            peak_idx = name.split("_", 1)[1]
            sigma_name = "sigma" if link_widths else f"sigma_{peak_idx}"
            if sigma_name in free_index:
                damping.append(_ParamDamping("pos", sigma_index=free_index[sigma_name]))
            else:
                damping.append(_ParamDamping("pos", sigma_value=fixed_params[sigma_name]))
        elif name == "sigma" or name.startswith("sigma_"):
            damping.append(_ParamDamping("sigma"))
        elif name == "tail_fraction":
            damping.append(_ParamDamping("tail_fraction"))
        elif name == "tail_beta":
            # Every peak width, however it is held, so _clamp_trial can bound
            # this against the widest of them. Collected here rather than in
            # the clamp because this is where the free/fixed split is known.
            width_names = (
                ["sigma"] if link_widths
                else [n for n in (free_names + list(fixed_params)) if n.startswith("sigma_")]
            )
            damping.append(_ParamDamping(
                "tail_beta",
                sigma_indices=tuple(
                    free_index[n] for n in width_names if n in free_index
                ),
                sigma_values=tuple(
                    fixed_params[n] for n in width_names if n in fixed_params
                ),
            ))
        elif name in ("bg_c0", "bg_c1"):
            damping.append(_ParamDamping("bg"))
        else:
            raise FitError(f"No damping rule for parameter {name!r}")
    # _apply_step_damping walks `damping` and `p` by the same index, so a
    # name that matched no branch above would not merely go undamped -- it
    # would shift every later parameter's rule onto the wrong parameter.
    # Silently, and only for configurations exercising that name.
    assert len(damping) == len(free_names), "damping list must be parallel to free_names"
    return damping


def fit_peaks(
    x, y, left_bg_region, right_bg_region, fit_region, peak_positions,
    link_widths=True, enable_left_tail=False, fixed_params=None,
    initial_guess_overrides=None, variance=None, fit_background=False,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fixed_params = dict(fixed_params) if fixed_params else {}

    if not peak_positions:
        raise FitError("At least one peak must be marked")

    slope, intercept = compute_background(x, y, left_bg_region, right_bg_region)

    lo, hi = fit_region
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size == 0:
        raise FitError(f"Fit region {fit_region} contains no data")

    n_peaks = len(peak_positions)
    names = _parameter_names(n_peaks, link_widths, enable_left_tail, fit_background)
    unknown_fixed = set(fixed_params) - set(names)
    if unknown_fixed:
        raise FitError(f"Unknown fixed parameter name(s): {sorted(unknown_fixed)}")
    free_names = [name for name in names if name not in fixed_params]

    if free_names and x_fit.size < len(free_names):
        raise FitError(
            f"Fit region has {x_fit.size} data points, need at least "
            f"{len(free_names)} free parameter(s) for {n_peaks} peak(s)"
        )

    y_sub = y_fit - (slope * x_fit + intercept)
    # Poisson unless the caller supplied a propagated variance. sqrt(N)
    # is right for raw counts and wrong for anything derived: a matrix
    # cut or an Add/Subtract result has variance `pos + f**2 * bg`, which
    # exceeds its own counts, and can go negative -- where the max(y, 1.0)
    # floor below would silently assign it an error of exactly 1 count.
    # Weighting those channels as though they were Poisson misweights
    # every one of them and makes the reported parameter errors and
    # chi-square optimistic. HDTV avoids this by carrying ROOT's own
    # propagated bin errors through every projection and cut; this is the
    # same information arriving by a different route.
    #
    # The same floor is applied either way, for the same reason: a
    # channel with (near-)zero variance would otherwise take an infinite
    # weight and dominate the fit on its own.
    if variance is None:
        y_err = np.sqrt(np.maximum(y_fit, 1.0))
    else:
        variance = np.asarray(variance, dtype=float)
        if variance.shape != y.shape:
            raise FitError(
                f"variance has length {variance.size}, expected {y.size} to match the spectrum"
            )
        y_err = np.sqrt(np.maximum(variance[mask], 1.0))

    # After the variance has been validated above, so a wrong-length array
    # is reported as a FitError rather than mis-indexing in here.
    bg_anchors = background_anchors(x, y, left_bg_region, right_bg_region, variance)

    # F5: with fit_background the background stops being a fixed two-point
    # line and becomes two more fitted parameters, so the model is compared
    # against the RAW counts rather than a pre-subtracted residual.
    #
    # This is the statistically honest arrangement. Subtracting a background
    # first and then fitting as though it were exact throws away the
    # peak-background correlation, which makes every peak uncertainty come
    # out too small: the data cannot actually tell a slightly taller peak on
    # a slightly lower background apart from the reverse, and pretending
    # otherwise claims information nobody has. HDTV offers exactly this
    # choice -- an external pre-fitted Background, or an "internal"
    # polynomial whose coefficients join the peak parameters
    # (TheuerkaufFitter::_Fit, fIntBgDeg).
    #
    # Off by default. It is not strictly better in every case: two extra
    # free parameters cost degrees of freedom and can go degenerate against
    # a peak width on a short fit region, and TV -- which this app ports --
    # has no such mode, so leaving it on by default would silently move
    # every result away from TV without being asked.
    background_reference = float(0.5 * (lo + hi)) if fit_background else None
    if fit_background:
        target = y_fit
        background_seed = (
            float(slope * background_reference + intercept),
            float(slope),
        )
    else:
        target = y_sub
        background_seed = None

    def bg_level_err_at(at):
        """The background's own standard error, so a peak's
        background-included area can carry it (see full_area_err below).
        Rebound after the fit when the background was fitted jointly, so it
        reports the fitted covariance instead of the two-point anchors."""
        return anchor_error(bg_anchors, at)

    model_named = _make_model(
        names, free_names, fixed_params, n_peaks, link_widths, enable_left_tail,
        background_reference=background_reference,
    )

    if not free_names:
        # Every parameter is fixed -- nothing to optimize. Evaluate
        # directly at the fixed values instead of calling curve_fit
        # with an empty parameter vector. No fit was performed, so
        # every value's uncertainty is exactly 0.0.
        values_by_name = dict(fixed_params)
        err_by_name = {name: 0.0 for name in names}
        # No fit was performed, so there is no covariance to propagate an
        # area uncertainty from -- and none is needed, since every
        # parameter's uncertainty is exactly 0 here rather than unknown.
        pcov = None
    else:
        # Seeding always works from y_sub -- the amplitude and width
        # estimates want the background out of the way whether or not the
        # background itself is being fitted afterwards.
        p0 = _initial_guess(
            free_names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail,
            initial_guess_overrides=initial_guess_overrides,
            background_seed=background_seed,
        )
        model = lambda xx, pp: model_named(xx, *pp)
        damping = _build_param_damping(free_names, fixed_params, link_widths)

        # One evaluator per free parameter, each computing only the model
        # terms that parameter can move. See _parameter_influence.
        def _restrict(peak_indices, include_background):
            def evaluate(xx, pp):
                values = dict(fixed_params)
                values.update(zip(free_names, pp))
                return model_named.evaluate(
                    xx, values, peak_indices, include_background
                )
            return evaluate

        restricted = [
            None if scope is None else _restrict(*scope)
            for scope in _parameter_influence(free_names, n_peaks, link_widths)
        ]

        try:
            popt, pcov = _marquardt_fit(
                model, x_fit, target, y_err, p0, damping, fit_region_bounds=fit_region,
                restricted=restricted,
            )
        except np.linalg.LinAlgError as exc:
            raise FitError(f"Fit did not converge: {exc}") from exc

        if not np.all(np.isfinite(popt)):
            # The VALUES themselves are unusable -- there is no fit to
            # report at all. Distinct from an unusable covariance, which
            # is handled below by reporting the fit with undetermined
            # uncertainties rather than discarding it.
            raise FitError("Fit did not converge: the fit produced non-finite parameter values")

        # A covariance diagonal holds variances, so a negative or
        # non-finite entry means that parameter is not determined by
        # this data. This used to reject the WHOLE fit, which threw away
        # good position/sigma/amplitude values because a nuisance
        # parameter was undetermined -- most often the tail pair, once
        # tail_fraction is driven to ~0 and d(model)/d(tail_beta)
        # vanishes. Now the undetermined parameters alone get NaN and
        # the rest keep real uncertainties (see _uncertainties_from).
        perr = _uncertainties_from(pcov, len(popt))

        # An undetermined TAIL parameter is worth reporting around: the
        # peak's own position/width/area are what the user reads, and
        # they are usually solid even when the tail is not. An
        # undetermined PEAK parameter is different -- it means the marks
        # themselves are degenerate (duplicate peaks at one position,
        # where any split of amplitude between them fits equally well),
        # so the values carry no more information than the uncertainties
        # do and reporting them would dress a meaningless answer up as a
        # successful fit.
        undetermined = {
            name for name, err in zip(free_names, perr) if math.isnan(err)
        }
        degenerate_peak_params = {
            name for name in undetermined
            if not name.startswith("tail_")
        }
        if degenerate_peak_params:
            # Reported as peak NUMBERS, not internal parameter names.
            # This used to list them raw -- "could not determine amp_0,
            # amp_1, amp_2, amp_3, pos_0, ... sigma_3" -- which is twelve
            # pieces of jargon for what is really "peaks 1-4", and
            # amp_0/sigma_2 mean nothing to someone marking peaks on a
            # spectrum. The parameter names are an implementation detail
            # of _parameter_names(); the peak index is the only part the
            # user chose.
            peak_numbers = sorted(
                {int(name.rsplit("_", 1)[1]) + 1
                 for name in degenerate_peak_params
                 if name.rsplit("_", 1)[-1].isdigit()}
            )
            if peak_numbers:
                if len(peak_numbers) == 1:
                    which = f"peak {peak_numbers[0]}"
                else:
                    which = "peaks " + ", ".join(str(n) for n in peak_numbers)
            else:
                # The shared "sigma" of a linked-width fit carries no peak
                # index at all, so there is no number to report.
                which = "the peak width"
            detail = (
                f"the fit could not determine {which} -- the marked peaks or regions "
                "are degenerate for this data. Peaks marked at (or very near) the "
                "same position, or a fit region too narrow to separate them, are the "
                "usual cause"
            )
            if enable_left_tail:
                detail += "; if the peak has no real tail, try unchecking Left tail"
            raise FitError(detail[0].upper() + detail[1:])
        values_by_name = dict(fixed_params)
        values_by_name.update(zip(free_names, popt))
        err_by_name = {name: 0.0 for name in fixed_params}
        err_by_name.update(zip(free_names, perr))

    # Reduced chi-square of the final (fitted or fully-fixed) model against
    # the same data the solver itself compared against -- background-
    # subtracted normally, raw counts when the background is part of the
    # model. Using y_sub in the latter case would score the fit against a
    # residual it never tried to reproduce. Weighted as the solver weighted
    # its residuals. None (undisplayable) rather than a divide-by-zero when
    # there are exactly as many data points as free parameters (zero degrees
    # of freedom) -- mirrors TV's own refusal to fit at all in that case
    # (vsCurFit.c's CurFreedom <= 0 check), except this app already allows
    # the fit to proceed and only the chi-square reporting is affected.
    free_values = [values_by_name[name] for name in free_names]
    model_curve = model_named(x_fit, *free_values)
    chi2 = float(np.sum(((target - model_curve) / y_err) ** 2))
    dof = x_fit.size - len(free_names)
    reduced_chi2 = chi2 / dof if dof > 0 else None

    # With the background fitted, the reported line and its uncertainty come
    # from the fit rather than from the two marked regions. Converted back
    # out of the centred parametrisation (see _make_model) so
    # background_slope/background_intercept keep meaning the same thing to
    # every caller that draws or evaluates the line.
    background_covariance = ()
    if fit_background:
        c0 = values_by_name["bg_c0"]
        c1 = values_by_name["bg_c1"]
        slope = float(c1)
        intercept = float(c0 - c1 * background_reference)
        indices = [free_names.index(n) for n in ("bg_c0", "bg_c1") if n in free_names]
        if pcov is not None and len(indices) == 2:
            block = np.asarray(pcov)[np.ix_(indices, indices)]
            if np.all(np.isfinite(block)):
                # Stored with the reference channel, since the variances are
                # for the CENTRED coefficients. This is HDTV's
                # PolyBg::EvalError in the two-coefficient case: the error
                # band is sqrt(v C v) with v = [1, x - reference].
                background_covariance = (
                    background_reference,
                    float(block[0, 0]), float(block[0, 1]), float(block[1, 1]),
                )

    # Whichever error model applies, the per-peak full_area_err below must
    # use it -- otherwise a jointly-fitted background would still be
    # credited with the two-point regions' uncertainty.
    _bg_error_model = FitResult(
        left_bg_region=tuple(left_bg_region), right_bg_region=tuple(right_bg_region),
        fit_region=tuple(fit_region), background_slope=slope,
        background_intercept=intercept, peaks=[],
        background_anchors=bg_anchors, background_covariance=background_covariance,
    )

    def bg_level_err_at(at):  # noqa: F811 -- deliberately rebound, see above
        return _bg_error_model.background_level_error(at)

    amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_named(
        values_by_name, n_peaks, link_widths, enable_left_tail
    )
    amplitude_errs, position_errs, sigma_errs, tail_fraction_err, tail_beta_err = _unpack_named(
        err_by_name, n_peaks, link_widths, enable_left_tail
    )

    peaks = []
    for i in range(n_peaks):
        amplitude = amplitudes[i]
        position = positions[i]
        sigma = abs(sigmas[i])
        amplitude_err = amplitude_errs[i]
        position_err = position_errs[i]
        sigma_err = sigma_errs[i]
        fwhm = FWHM_FACTOR * sigma
        fwhm_err = FWHM_FACTOR * sigma_err
        # Exact integral of the FULL fitted shape, tail included -- see
        # hypermet_area. This used to integrate the Gaussian core alone
        # and so understated every tailed peak's area by whatever the
        # tail held (measured 5.0% at r=0.1, 13.6% at r=0.3, 41.1% for a
        # long tail). r=0 recovers the plain Gaussian area exactly, so
        # untailed fits are unaffected; beta is then irrelevant and only
        # needs to be positive to keep erfcx(y) finite.
        area, area_err = _area_with_error(
            amplitude,
            sigma,
            tail_fraction if tail_fraction is not None else 0.0,
            tail_beta if tail_beta is not None else 1.0,
            _area_param_names(i, link_widths, enable_left_tail),
            free_names,
            pcov,
        )
        # Full (background-included) area for this one peak: its own net
        # area plus the background "under" it, using the standard
        # gamma-spectroscopy convention of the local linear-background
        # level at the peak's own center times that peak's own FWHM --
        # the per-peak analog of Integration mode's flat
        # density-times-width background area.
        #
        # The background term used to be treated as exact -- full_area_err
        # was set to area_err on the stated grounds that a two-point line
        # is "uncertainty-free". It is not: both anchors are means of real
        # counts and carry Poisson error, which background_error()
        # propagates exactly. Combined in quadrature here, as HDTV does
        # for its own background-subtracted integral
        # (TH1BgsubIntegral::GetBinError2 = eh**2 + eb**2).
        background_under_peak = (slope * position + intercept) * fwhm
        background_area_err = float(fwhm * bg_level_err_at(position))
        full_area = area + background_under_peak
        full_area_err = float(math.sqrt(area_err ** 2 + background_area_err ** 2))
        peaks.append(
            PeakResult(
                position=float(position), position_err=float(position_err),
                fwhm=float(fwhm), fwhm_err=float(fwhm_err),
                area=float(area), area_err=float(area_err),
                amplitude=float(amplitude), amplitude_err=float(amplitude_err),
                sigma=float(sigma), sigma_err=float(sigma_err),
                full_area=float(full_area), full_area_err=float(full_area_err),
            )
        )

    # Region-level totals, analogous to Integration's gross/net split:
    # gross is the raw (background-included) count total over the fit
    # region -- same Poisson formula as integrate_region()'s gross_area.
    # net is the total of the fitted peaks' own (background-excluded)
    # areas, with their already-fit-propagated uncertainties combined in
    # quadrature.
    #
    # The background's own uncertainty is deliberately NOT added here,
    # even though background_error() now provides it and each peak's
    # full_area_err does include it. Net area excludes the background by
    # definition, so there is no background term to add; what a rigorous
    # treatment would capture instead is the peaks' SENSITIVITY to the
    # background level, since they were fitted against data from which it
    # had already been subtracted as though exact. That sensitivity is not
    # 1-times-width in general and cannot be recovered post hoc -- it
    # falls out naturally only from fitting the background jointly with
    # the peaks, which is a separate change (F5 in the v4.0.0 plan). Until
    # then net_area_err is conditional on the background, and saying so
    # here is better than adding a term that looks rigorous and is not.
    gross_area = float(np.sum(y_fit))
    if variance is None:
        # Deliberately no guard here, matching TV's own lack of one: unlike
        # integrate_region() (which now rejects a negative-count region
        # outright, see the FitError raised above `ds = s.copy()`), this is
        # one auxiliary summary field on an ALREADY-SUCCESSFUL fit -- a
        # negative gross_area (e.g. fitting a Subtract Spectra result)
        # produces a NaN here without discarding the rest of a valid fit.
        gross_area_err = float(np.sqrt(gross_area))
    else:
        # The same reasoning the per-channel weights above are built on:
        # sqrt(N) is the Poisson error of raw counts and is simply wrong
        # for a derived spectrum, whose variance exceeds its own counts and
        # whose channels can go negative. Reporting sqrt(N) here while the
        # peaks in the same fit were weighted by the real variance left two
        # numbers in one results panel disagreeing about how well the same
        # data is known -- measured at a factor of two low for a
        # background-subtracted matrix cut. No max(..., 0) is needed: a
        # variance array is non-negative by construction everywhere it is
        # produced (matrix_cut and spectrum_operations both floor it).
        gross_area_err = float(np.sqrt(np.sum(variance[mask])))
    net_area = float(sum(peak.area for peak in peaks))
    # Propagated through the whole covariance rather than summed in
    # quadrature -- see _total_area_error for the measurement that forced
    # this. net_area itself is still the plain sum of the peak areas, so
    # only the uncertainty changes.
    net_area_err = float(_total_area_error(
        values_by_name, n_peaks, link_widths, enable_left_tail, free_names, pcov,
    ))

    return FitResult(
        left_bg_region=tuple(left_bg_region), right_bg_region=tuple(right_bg_region),
        fit_region=tuple(fit_region), background_slope=float(slope),
        background_intercept=float(intercept), peaks=peaks,
        link_widths=link_widths,
        tail_fraction=float(tail_fraction) if tail_fraction is not None else None,
        tail_fraction_err=float(tail_fraction_err) if tail_fraction_err is not None else None,
        tail_beta=float(tail_beta) if tail_beta is not None else None,
        tail_beta_err=float(tail_beta_err) if tail_beta_err is not None else None,
        fixed_params=dict(fixed_params),
        gross_area=gross_area, gross_area_err=gross_area_err,
        net_area=net_area, net_area_err=net_area_err,
        reduced_chi2=reduced_chi2,
        background_anchors=bg_anchors,
        background_covariance=background_covariance,
        fit_background=fit_background,
    )


def parameter_names(n_peaks, link_widths, enable_left_tail, fit_background=False):
    """Public alias of _parameter_names -- fit_mode.py's Fit Parameters
    panel needs this exact ordering to know which rows to display."""
    return _parameter_names(n_peaks, link_widths, enable_left_tail, fit_background)


def fit_result_values_by_name(result):
    """Reconstructs the {name: value} mapping (matching
    parameter_names()'s canonical naming) from an already-committed
    FitResult -- used by the UI layer to populate the Fit Parameters
    panel and to pre-fill fixed_params when reloading an older fit.
    """
    n_peaks = len(result.peaks)
    link_widths = result.link_widths
    enable_left_tail = result.tail_fraction is not None
    values = {}
    for i, peak in enumerate(result.peaks):
        values[f"amp_{i}"] = peak.amplitude
        values[f"pos_{i}"] = peak.position
        if not link_widths:
            values[f"sigma_{i}"] = peak.sigma
    if link_widths and result.peaks:
        values["sigma"] = result.peaks[0].sigma
    if enable_left_tail:
        values["tail_fraction"] = result.tail_fraction
        values["tail_beta"] = result.tail_beta
    if result.fit_background:
        # Back into the CENTRED form the fit used, so the panel's values
        # round-trip through fixed_params/initial_guess_overrides. The
        # reference travels on background_covariance; without it (a fit whose
        # background covariance came back unusable) fall back to the fit
        # region's centre, which is how the reference is chosen in the first
        # place.
        reference = (result.background_covariance[0] if result.background_covariance
                    else 0.5 * (result.fit_region[0] + result.fit_region[1]))
        values["bg_c1"] = result.background_slope
        values["bg_c0"] = result.background_intercept + result.background_slope * reference
    return values


def integrate_region(x, y, left_bg_region, right_bg_region, fit_region,
                     variance=None):
    """TV-style direct background-subtracted region sum, ported
    line-by-line from FIIntegrateRegion's no-fitted-background branch
    (tv-1.9.13/lib/tv/vsFitInt.c:19-51,194-309) plus the sigma/FWHM
    conversion from ParseIntPeak (tv-1.9.13/lib/tv/vsFitFmt.c:340-392).
    Several arithmetic choices below look unusual (asymmetric plain-sum
    vs abs(sum) normalization between layers/moments; the background's
    own FWHM uncertainty term specifically -- not its skewness term --
    reusing the *net* distribution's 2nd moment rather than its own;
    the background-sum uncertainty
    scaling linearly rather than quadratically with region width) --
    these are deliberate, source-verified TV-parity choices, not bugs,
    per the design spec. Independently validated (hand-computed moments,
    flat-background/analytic-peak sanity checks, and a dedicated
    cross-check proving the moment-reuse quirk is real) before this
    function was written -- see the design spec's Testing section.

    `variance` is this port's one addition to TV, and it is opt-in: pass a
    per-channel variance and it is used everywhere TV reads a variance off
    the counts, leaving every other line of the algorithm untouched.
    Without it the behaviour is exactly TV's, including the refusal below.

    The reason it exists is that Fit and Integration were disagreeing about
    the same data. A matrix cut carries a propagated variance -- `pos +
    factor**2 * bg`, which exceeds its own counts -- and fit_peaks has
    weighted by it since v4.0.0, while integrating the same spectrum
    silently assumed raw Poisson counts and reported an uncertainty too
    small. Two numbers from one spectrum should not imply different things
    about how well it is known.

    Supplying a variance also lifts the negative-count refusal, because
    that refusal exists only to stop a variance being read off counts that
    cannot support one. With a real variance in hand there is nothing to
    read off, and a background-subtracted cut -- precisely the thing that
    carries a variance -- legitimately goes negative.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if variance is not None:
        variance = np.asarray(variance, dtype=float)
        if variance.shape != y.shape:
            raise FitError(
                f"variance has length {variance.size}, expected {y.size} to "
                "match the spectrum"
            )

    lo, hi = fit_region
    mask = (x >= lo) & (x <= hi)
    idx = x[mask]
    s = y[mask]
    n = idx.size
    if n == 0:
        raise FitError(f"Fit region {fit_region} contains no data")
    if variance is None and np.any(s < 0.0):
        raise FitError(
            "Cannot integrate a region containing negative counts (e.g. "
            "from a Subtract Spectra result) -- the uncertainty "
            "calculation used here assumes non-negative Poisson counts, "
            "matching TV's own convention."
        )
    # Poisson variance per channel = the count itself, unless the caller
    # knows better.
    ds = s.copy() if variance is None else variance[mask]

    # ---- gross sum & variance (vsFitInt.c:41-43) ----
    gross_sum = float(np.sum(s))
    gross_dsum = float(np.sum(ds))
    gross_area = gross_sum
    gross_area_err = math.sqrt(gross_dsum)

    # ---- gross moments (vsFitInt.c:45-83) ----
    g_M1 = g_DM1 = g_M2 = g_DM2 = g_M3 = g_DM3 = 0.0
    if gross_sum != 0.0:
        g_M1 = float(np.sum(idx * s)) / gross_sum

        dlt = idx - g_M1
        dlt2 = dlt ** 2
        dMom1 = float(np.sum(dlt2 * ds))
        mom2_raw = float(np.sum(dlt2 * s))
        g_DM1 = math.sqrt(dMom1) / abs(gross_sum)
        g_M2 = mom2_raw / gross_sum  # plain sum, not abs

        dlt3 = dlt2 * dlt
        dDlt_m2 = dlt2 - g_M2
        dMom2 = float(np.sum((dDlt_m2 ** 2) * ds))
        mom3_raw = float(np.sum(dlt3 * s))
        g_DM2 = math.sqrt(dMom2) / abs(gross_sum)
        g_M3 = mom3_raw / abs(gross_sum)  # abs this time

        term = dlt * (dlt2 - 3.0 * g_M2) - g_M3
        dMom3 = float(np.sum((term ** 2) * ds))
        g_DM3 = math.sqrt(dMom3) / abs(gross_sum)

    # ---- background: pooled flat density across BOTH bg regions ----
    # TV's own branch has no trailing `else` (vsFitInt.c:85,194) -- when
    # no background regions are marked, this block is skipped outright,
    # not "looped over zero regions that happens to sum to zero".
    bg_chn = 0
    bg_count = 0.0
    bg_dcount = 0.0
    if left_bg_region is not None and right_bg_region is not None:
        for region in (left_bg_region, right_bg_region):
            blo, bhi = region
            bmask = (x >= blo) & (x <= bhi)
            bg_chn += int(np.sum(bmask))
            bg_y = y[bmask]
            if variance is None and np.any(bg_y < 0.0):
                raise FitError(
                    "Cannot integrate a region containing negative counts "
                    "(e.g. from a Subtract Spectra result) -- the "
                    "uncertainty calculation used here assumes "
                    "non-negative Poisson counts, matching TV's own "
                    "convention."
                )
            bg_count += float(np.sum(bg_y))
            # TV's two accumulators read the same numbers because its
            # variance IS the count; they part company once a real variance
            # is supplied, and only the second one is a variance.
            bg_dcount += float(
                np.sum(bg_y) if variance is None else np.sum(variance[bmask])
            )

    if bg_chn > 0:
        bg_density = bg_count / bg_chn
        bg_density_var = bg_dcount / (bg_chn * bg_chn)
    else:
        bg_density = 0.0
        bg_density_var = 0.0

    background_area = bg_density * n
    background_area_var = bg_density_var * n  # linear, not squared (TV quirk, kept)
    background_area_err = math.sqrt(background_area_var)

    net_sum = gross_sum - background_area
    net_dsum = gross_dsum + background_area_var
    net_area = net_sum
    net_area_err = math.sqrt(net_dsum)

    # ---- background & net moments (vsFitInt.c:223-309) ----
    bg_M1 = bg_DM1 = bg_M2 = bg_DM2 = bg_M3 = bg_DM3 = 0.0
    n_M1 = n_DM1 = n_M2 = n_DM2 = n_M3 = n_DM3 = 0.0
    if net_sum != 0.0 or background_area != 0.0:
        b = bg_density
        db = bg_density_var
        bg_sum = background_area

        mom1_raw = float(np.sum(idx * (s - b)))
        bgmom1_raw = float(np.sum(idx * b))
        if bg_sum != 0.0:
            bg_M1 = bgmom1_raw / abs(bg_sum)
        if net_sum != 0.0:
            n_M1 = mom1_raw / abs(net_sum)

        dlt = idx - n_M1
        dltb = idx - bg_M1
        dMom1 = float(np.sum((dlt ** 2) * (ds + db)))
        mom2_raw = float(np.sum((dlt ** 2) * (s - b)))
        dBgMom1 = float(np.sum((dltb ** 2) * db))
        bgmom2_raw = float(np.sum((dltb ** 2) * b))
        if bg_sum != 0.0:
            bg_DM1 = math.sqrt(dBgMom1) / abs(bg_sum)
            bg_M2 = bgmom2_raw / abs(bg_sum)  # abs (unlike gross's plain-sum M2)
        if net_sum != 0.0:
            n_DM1 = math.sqrt(dMom1) / abs(net_sum)
            n_M2 = mom2_raw / abs(net_sum)  # abs, like every other bg/net moment (vsFitInt.c:264)

        dlt2 = dlt ** 2
        dltb2 = dltb ** 2
        dDlt_m2 = dlt2 - n_M2
        dDltb_m2 = dltb2 - n_M2  # CONFIRMED TV QUIRK: net's M2, not bg's own
        dMom2 = float(np.sum((dDlt_m2 ** 2) * (ds + db)))
        mom3_raw = float(np.sum((dlt2 * dlt) * (s - b)))
        dBgMom2 = float(np.sum((dDltb_m2 ** 2) * db))
        bgmom3_raw = float(np.sum((dltb2 * dltb) * b))
        if bg_sum != 0.0:
            bg_DM2 = math.sqrt(dBgMom2) / abs(bg_sum)
            bg_M3 = bgmom3_raw / abs(bg_sum)
        if net_sum != 0.0:
            n_DM2 = math.sqrt(dMom2) / abs(net_sum)
            n_M3 = mom3_raw / abs(net_sum)

        term = dlt * (dlt2 - 3.0 * n_M2) - n_M3
        # vsFitInt.c:303 uses the background's OWN M2 here (unlike the
        # DM2 term above at vsFitInt.c:281, which genuinely does cross-use
        # net's M2 -- the two lines are NOT the same quirk, despite an
        # earlier design-spec draft citing them together).
        termb = dltb * (dltb2 - 3.0 * bg_M2) - bg_M3
        dMom3 = float(np.sum((term ** 2) * (ds + db)))
        dBgMom3 = float(np.sum((termb ** 2) * db))
        if bg_sum != 0.0:
            bg_DM3 = math.sqrt(dBgMom3) / abs(bg_sum)
        if net_sum != 0.0:
            n_DM3 = math.sqrt(dMom3) / abs(net_sum)

    def _to_reported(M1, DM1, M2, DM2, M3, DM3):
        centroid, centroid_err = M1, DM1
        sigma = math.sqrt(M2) if M2 >= 0.0 else -math.sqrt(-M2)
        sigma_err = DM2 / abs(sigma) if sigma != 0.0 else 0.0
        fwhm = sigma * FWHM_FACTOR
        fwhm_err = sigma_err * FWHM_FACTOR
        return centroid, centroid_err, fwhm, fwhm_err, M3, DM3

    g_centroid, g_centroid_err, g_fwhm, g_fwhm_err, g_skew, g_skew_err = _to_reported(
        g_M1, g_DM1, g_M2, g_DM2, g_M3, g_DM3
    )
    bg_centroid, bg_centroid_err, bg_fwhm, bg_fwhm_err, bg_skew, bg_skew_err = _to_reported(
        bg_M1, bg_DM1, bg_M2, bg_DM2, bg_M3, bg_DM3
    )
    n_centroid, n_centroid_err, n_fwhm, n_fwhm_err, n_skew, n_skew_err = _to_reported(
        n_M1, n_DM1, n_M2, n_DM2, n_M3, n_DM3
    )

    return IntegrationResult(
        left_bg_region=tuple(left_bg_region) if left_bg_region is not None else None,
        right_bg_region=tuple(right_bg_region) if right_bg_region is not None else None,
        fit_region=tuple(fit_region), background_density=float(bg_density),
        gross_area=float(gross_area), gross_area_err=float(gross_area_err),
        gross_centroid=float(g_centroid), gross_centroid_err=float(g_centroid_err),
        gross_fwhm=float(g_fwhm), gross_fwhm_err=float(g_fwhm_err),
        gross_skewness=float(g_skew), gross_skewness_err=float(g_skew_err),
        background_area=float(background_area), background_area_err=float(background_area_err),
        background_centroid=float(bg_centroid), background_centroid_err=float(bg_centroid_err),
        background_fwhm=float(bg_fwhm), background_fwhm_err=float(bg_fwhm_err),
        background_skewness=float(bg_skew), background_skewness_err=float(bg_skew_err),
        net_area=float(net_area), net_area_err=float(net_area_err),
        net_centroid=float(n_centroid), net_centroid_err=float(n_centroid_err),
        net_fwhm=float(n_fwhm), net_fwhm_err=float(n_fwhm_err),
        net_skewness=float(n_skew), net_skewness_err=float(n_skew_err),
    )


class _ParamDamping:
    """Per-parameter step-damping rule for _marquardt_fit, ported from
    TV's CHANGE_TRY macro (tv-1.9.13/src/VsFitFct.c). `kind` selects the
    rule: "amp" (no damping), "pos" (step capped at the peak's own
    current sigma; needs sigma_index or sigma_value to know that sigma),
    "sigma" (step capped at 2x current value, floored away from zero),
    "tail_fraction"/"tail_beta" (nudged away from exactly zero, then
    hard-clamped to this project's existing bounds).

    `sigma_indices`/`sigma_values` are for "tail_beta" alone, which is
    bounded above at TAIL_BETA_MAX_SIGMA times a peak width and so needs to
    know the widths. Both are sequences because tail_beta is shared by every
    peak while the widths need not be (unlinked fits have one sigma each),
    and because a width may be free (an index into the parameter vector) or
    held fixed (a value, since it is not in the vector at all)."""

    __slots__ = ("kind", "sigma_index", "sigma_value", "sigma_indices",
                 "sigma_values")

    def __init__(self, kind, sigma_index=None, sigma_value=None,
                 sigma_indices=(), sigma_values=()):
        self.kind = kind
        self.sigma_index = sigma_index
        self.sigma_value = sigma_value
        self.sigma_indices = tuple(sigma_indices)
        self.sigma_values = tuple(sigma_values)


def _tail_beta_upper_bound(meta, p):
    """The widest peak width this fit knows about, times
    TAIL_BETA_MAX_SIGMA -- or None when no width is available at all.

    The WIDEST rather than the narrowest, because one tail_beta is shared by
    every peak: bounding against a narrow peak would bind on a broad peak's
    genuine tail. None (no bound) rather than a guess when there is nothing
    to scale by, which keeps this from inventing a limit out of nothing.
    """
    widths = [abs(p[index]) for index in meta.sigma_indices]
    widths += [abs(value) for value in meta.sigma_values if value is not None]
    widths = [w for w in widths if math.isfinite(w) and w > 0.0]
    if not widths:
        return None
    return TAIL_BETA_MAX_SIGMA * max(widths)


def _current_sigma_for(meta, p):
    if meta.sigma_index is not None:
        return p[meta.sigma_index]
    return meta.sigma_value


_MIN_PLAUSIBLE_FWHM_CHANNELS = 2.0


def _measure_width(x_fit, y_sub, peak_positions, fallback):
    """TV-style initial width estimate, ported from
    tv-1.9.13/lib/tv/vsFitSetup.c's FSInitWidth: starting from the
    marked peak with the largest background-subtracted amplitude, walk
    to the true local maximum, then walk outward until counts drop
    below half that maximum, and convert the resulting FWHM to a sigma.
    Falls back to the caller-supplied heuristic value if the data is
    too flat/noisy to find a clear peak, or the measured FWHM is
    implausibly small to be a real detector peak (a simple plausibility
    guard, not a rigorous significance test -- a pure-noise region can
    occasionally still produce a small but "clean" half-max crossing;
    this is an accepted, rare edge case since real usage always marks
    an actual visible peak)."""
    if len(x_fit) == 0:
        return fallback

    indices = [int(np.argmin(np.abs(x_fit - pos))) for pos in peak_positions]
    amplitudes = [y_sub[i] for i in indices]
    idx = indices[int(np.argmax(amplitudes))]

    n = len(x_fit)
    while True:
        moved = False
        if idx + 1 < n and y_sub[idx + 1] > y_sub[idx]:
            idx += 1
            moved = True
        elif idx - 1 >= 0 and y_sub[idx - 1] > y_sub[idx]:
            idx -= 1
            moved = True
        if not moved:
            break

    peak_value = y_sub[idx]
    if peak_value <= 0:
        return fallback
    half = peak_value / 2.0

    left = idx
    while left - 1 >= 0 and y_sub[left - 1] >= half:
        left -= 1
    right = idx
    while right + 1 < n and y_sub[right + 1] >= half:
        right += 1

    def _interp_crossing(i_inside, i_outside):
        y_in, y_out = y_sub[i_inside], y_sub[i_outside]
        if y_in == y_out:
            return float(x_fit[i_inside])
        frac = (half - y_in) / (y_out - y_in)
        return float(x_fit[i_inside] + frac * (x_fit[i_outside] - x_fit[i_inside]))

    # Measure each side's half-width independently and take the smaller
    # one (doubled, to get a full FWHM) rather than the raw
    # (x_right - x_left) span. A neighboring peak on one side (as with
    # closely-spaced multiplets) slows that side's descent below half-
    # max, inflating a naive two-sided span well past this peak's own
    # true width -- exactly the scenario TV's FSInitWidth (same source
    # file) guards against by using whichever side crosses half-max
    # first. Taking the min keeps the estimate anchored to the
    # uncontaminated side; for an isolated, symmetric peak both sides
    # agree anyway.
    half_widths = []
    if left > 0:
        half_widths.append(x_fit[idx] - _interp_crossing(left, left - 1))
    if right < n - 1:
        half_widths.append(_interp_crossing(right, right + 1) - x_fit[idx])
    if not half_widths:
        return fallback

    fwhm = 2.0 * min(half_widths)
    if fwhm < _MIN_PLAUSIBLE_FWHM_CHANNELS:
        return fallback
    return max(fwhm / FWHM_FACTOR, 1e-6)


def _numeric_jacobian(model, x, p, restricted=None):
    """Central-difference Jacobian -- matches TV's own fallback
    (CurNumericDerivation) for fit-function modules without an analytic
    derivative; avoids hand-deriving hypermet's erfc-based partials.

    `restricted` is an optional list parallel to `p`, holding either None
    or a callable `f(x, p)` that evaluates ONLY the model terms the
    corresponding parameter can move (see _parameter_influence). The
    difference of two such restricted evaluations equals the difference of
    two full ones, because every omitted term is identical in both and
    cancels exactly -- so this is the same derivative, reached without
    recomputing peaks that did not move.

    Omitted entirely (the default) it evaluates the whole model per
    column, which is what every caller that builds a model by hand does.
    """
    n = len(p)
    J = np.zeros((len(x), n))
    for i in range(n):
        h = max(abs(p[i]), 1e-6) * 1e-6
        p_hi = p.copy(); p_hi[i] += h
        p_lo = p.copy(); p_lo[i] -= h
        evaluate = model if restricted is None or restricted[i] is None else restricted[i]
        J[:, i] = (evaluate(x, p_hi) - evaluate(x, p_lo)) / (2 * h)
    return J


def _apply_step_damping(p, delta, damping, fit_region_bounds):
    """TV's CHANGE_TRY, per parameter kind. Returns a new delta array;
    does not mutate its inputs."""
    delta = delta.copy()
    for i, meta in enumerate(damping):
        if meta.kind == "amp":
            continue
        elif meta.kind == "pos":
            sigma = abs(_current_sigma_for(meta, p))
            if sigma > 0 and abs(delta[i]) > sigma:
                delta[i] = math.copysign(sigma, delta[i])
            if fit_region_bounds is not None:
                lo, hi = fit_region_bounds
                old = p[i]
                if lo <= old <= hi:
                    new = old + delta[i]
                    if new < lo:
                        delta[i] = (lo - old) * 0.5
                    elif new > hi:
                        delta[i] = (hi - old) * 0.5
        elif meta.kind == "sigma":
            width = abs(p[i])
            cap = 2.0 * width
            if width > 0 and abs(delta[i]) > cap:
                delta[i] = math.copysign(cap, delta[i])
            if p[i] + delta[i] == 0.0:
                delta[i] *= 0.5
        elif meta.kind in ("tail_fraction", "tail_beta"):
            if p[i] + delta[i] == 0.0:
                delta[i] *= 0.5
        elif meta.kind == "bg":
            # Undamped, deliberately. TV's CHANGE_TRY has no rule for a
            # background coefficient because TV never fits one, and the
            # rules above all exist to stop a NONLINEAR parameter taking a
            # step that leaves the basin (a position jumping past its
            # neighbour, a width collapsing through zero). The background
            # enters the model linearly, so its exact minimum is one step
            # away and capping that step only slows convergence.
            continue
    return delta


def _clamp_trial(p_try, damping):
    """Hard bound enforcement after damping -- this project's existing
    sigma/tail_fraction/tail_beta bounds (previously enforced via
    scipy's curve_fit(..., bounds=...), now folded in here since the
    custom solver has no separate bounds mechanism)."""
    p_try = p_try.copy()
    for i, meta in enumerate(damping):
        if meta.kind == "sigma":
            if abs(p_try[i]) < 1e-6:
                p_try[i] = 1e-6 if p_try[i] >= 0 else -1e-6
        elif meta.kind == "tail_fraction":
            p_try[i] = min(max(p_try[i], 0.0), TAIL_FRACTION_MAX)
        elif meta.kind == "tail_beta":
            p_try[i] = max(p_try[i], TAIL_BETA_MIN)
            # Bounded above as well as below -- see TAIL_BETA_MAX_SIGMA. The
            # bound is recomputed every trial because it scales with the
            # current width, which the fit is still moving.
            upper = _tail_beta_upper_bound(meta, p_try)
            if upper is not None and upper > TAIL_BETA_MIN:
                p_try[i] = min(p_try[i], upper)
    return p_try


def _marquardt_fit(model, x, y, y_err, p0, damping, fit_region_bounds=None,
                   restricted=None):
    """Damped Levenberg-Marquardt solver replicating TV's fit procedure
    (tv-1.9.13/lib/tv/vsCurFit.c's CurFit + VsFitFct.c's CHANGE_TRY).
    `model(x, p)` takes the full parameter array (not *args, unlike
    scipy's curve_fit convention) and returns the model y-values.
    `damping` is a list of _ParamDamping, one per entry in p0, telling
    the solver how to limit each parameter's per-iteration step.
    `restricted` is passed straight through to _numeric_jacobian and is
    purely an optimisation -- omitting it changes nothing but the cost.
    Returns (popt, pcov) with the same meaning as
    scipy.optimize.curve_fit(..., absolute_sigma=True)."""
    p = np.array(p0, dtype=float)
    n = len(p)
    weights = 1.0 / y_err

    # A trial's accept/reject test (below) only needs the scalar measure,
    # not the Jacobian -- and most trials in a damped Marquardt loop get
    # rejected (that's what the lambda-growth retries are for). Splitting
    # the two means a rejected trial costs one model evaluation instead of
    # 2*n+1 (2*n for the numeric Jacobian's central differences, +1 for
    # the measure's own evaluation), while an accepted step still gets
    # its full Jacobian, computed once.
    def measure_only(pt):
        r = (y - model(x, pt)) * weights
        measure = float(np.sum(r ** 2))
        return measure, r

    def jac_only(pt):
        return _numeric_jacobian(model, x, pt, restricted) * weights[:, None]

    def measure_and_jac(pt):
        measure, r = measure_only(pt)
        return measure, r, jac_only(pt)

    measure, r, J = measure_and_jac(p)
    lam = _CUR_LAMBDA_START
    iterations = 0
    give_up = False

    while iterations < _CUR_MAX_ITERATIONS and not give_up:
        old_measure = measure
        alpha = J.T @ J
        beta = J.T @ r
        accepted = False

        while not accepted:
            damped = alpha.copy()
            diag = np.diag(damped).copy()
            diag_safe = np.where(diag == 0, 1.0, diag)
            damped[np.diag_indices(n)] = diag_safe * (1 + lam)
            try:
                raw_delta = np.linalg.solve(damped, beta)
            except np.linalg.LinAlgError:
                raw_delta, *_ = np.linalg.lstsq(damped, beta, rcond=None)

            delta = _apply_step_damping(p, raw_delta, damping, fit_region_bounds)
            p_try = _clamp_trial(p + delta, damping)
            try_measure, try_r = measure_only(p_try)

            if try_measure < measure:
                improvement = measure - try_measure
                p, measure, r, J = p_try, try_measure, try_r, jac_only(p_try)
                lam = max(0.001 * improvement, _CUR_MIN_LAMBDA)
                accepted = True
            else:
                lam *= _CUR_INC_LAMBDA
                if lam > _CUR_MAX_LAMBDA:
                    # Last-resort "half step back": lambda has run away,
                    # so either the minimum is already reached or this
                    # step overshot it. TV tries the reversed half step
                    # (vsCurFit.c:130-134) -- and critically, pushes it
                    # back through the SAME CurChangeTry damping pass the
                    # forward step used, rather than applying it raw.
                    #
                    # Re-damping is not redundant here. Halving alone can
                    # only shrink a step, but NEGATING it can violate a
                    # constraint the forward step satisfied: the
                    # position rule keeps a peak inside the marked fit
                    # region, and a reversed step can leave that region
                    # by the opposite edge. _clamp_trial does not cover
                    # that case -- it only bounds sigma and the tail
                    # parameters -- so without this the fallback could
                    # accept a fit whose peak sits outside the region the
                    # user marked.
                    back_delta = _apply_step_damping(
                        p, -0.5 * delta, damping, fit_region_bounds
                    )
                    p_try2 = _clamp_trial(p + back_delta, damping)
                    try_measure2, try_r2 = measure_only(p_try2)
                    if try_measure2 < measure:
                        p, measure, r, J = p_try2, try_measure2, try_r2, jac_only(p_try2)
                    accepted = True
                    give_up = True

        iterations += 1
        if give_up or measure <= 0:
            break
        ratio = (old_measure - measure) / measure
        if ratio <= _CUR_RATIO:
            break

    # The tail pair is what may be dropped to rescue an otherwise-good
    # fit; every other parameter is one the user actually reads.
    optional = [
        i for i, meta in enumerate(damping)
        if meta.kind in ("tail_fraction", "tail_beta")
    ]
    return p, _covariance_from(J.T @ J, n, optional)


def _area_with_error(amplitude, sigma, r, beta, area_params, free_names, pcov):
    """A peak's area and its uncertainty.

    The area is hypermet_area()'s closed form. The uncertainty is the
    full first-order propagation

        var = g^T C g,    g_k = d(area)/d(param_k)

    over the covariance block for whichever of this peak's parameters
    were free. That matters because it keeps the OFF-DIAGONAL terms.
    The previous code added relative errors in quadrature:

        area_err = area * sqrt((dA/A)^2 + (dsigma/sigma)^2)

    which is what g^T C g reduces to only if amplitude and sigma are
    uncorrelated. In a peak fit they are strongly ANTI-correlated -- a
    wider peak with a lower amplitude fits almost as well -- and both
    partials are positive, so a negative covariance makes the true
    variance SMALLER than the quadrature sum. The old figure was
    therefore too large, which is the safe direction to be wrong in but
    wrong nonetheless, and it got worse the more the fit relied on the
    tail parameters, whose correlations it ignored entirely.

    Undetermined parameters (NaN or negative variance, see
    _uncertainties_from) are left out of the block rather than poisoning
    the result to NaN. That is not a shortcut: _covariance_from already
    drops those parameters and re-inverts, so the matrix here is the
    covariance CONDITIONAL on them, and it is the same matrix every other
    uncertainty this function reports is derived from. Including a NaN row
    would make the area the only quantity that degrades to "n/a" when a
    nuisance tail parameter is unconstrained, while position and width --
    computed from the same block -- still report numbers.

    Gradients are central differences. The area is a smooth scalar
    function of at most four parameters, so this is accurate to ~1e-10
    relative, and it avoids four hand-derived partials involving
    erfcx' -- the same trade _numeric_jacobian makes for the model.
    """
    area = float(hypermet_area(amplitude, sigma, r, beta))
    if pcov is None:
        # Nothing was fitted (every parameter fixed), so there is no
        # uncertainty rather than an unknown one.
        return area, 0.0

    values = {"amp": amplitude, "sigma": sigma, "r": r, "beta": beta}
    diag = np.diag(pcov)

    indices, partials = [], []
    for kind, name in area_params:
        if name not in free_names:
            continue  # held fixed: contributes no uncertainty
        index = free_names.index(name)
        if not (math.isfinite(diag[index]) and diag[index] >= 0.0):
            continue  # undetermined -- see the docstring
        value = values[kind]
        step = 1e-6 * max(abs(value), 1e-3)
        hi = value + step
        lo = value - step
        if kind in ("sigma", "beta") and lo <= 0.0:
            # Keep the perturbation inside the domain; sigma and beta are
            # strictly positive and erfcx(y) changes character across 0.
            lo = value * 0.5
        def _at(**override):
            v = dict(values, **override)
            return float(hypermet_area(v["amp"], v["sigma"], v["r"], v["beta"]))
        partials.append((_at(**{kind: hi}) - _at(**{kind: lo})) / (hi - lo))
        indices.append(index)

    if not indices:
        return area, float("nan")

    gradient = np.asarray(partials, dtype=float)
    block = np.asarray(pcov)[np.ix_(indices, indices)]
    variance = float(gradient @ block @ gradient)
    if not math.isfinite(variance) or variance < 0.0:
        return area, float("nan")
    return area, math.sqrt(variance)


def _total_area_error(values_by_name, n_peaks, link_widths, enable_left_tail,
                      free_names, pcov):
    """Uncertainty of the SUM of every peak's area, propagated through the
    whole covariance matrix at once:

        var = g^T C g,   g_k = d(sum of areas)/d(param_k)

    This replaces adding the per-peak area errors in quadrature, which is
    valid only if the peaks' areas are independent. They are not. Two
    things couple them, and they pull in opposite directions:

      * A linked-width fit gives every peak the SAME sigma, so their areas
        move together -- positive correlation, which quadrature
        understates.
      * Overlapping peaks have strongly ANTI-correlated amplitudes,
        because the data constrains the pair's total far better than it
        constrains the split between them. Quadrature overstates that, and
        for a real doublet this term dominates by a wide margin.

    Measured against a 400-realisation Monte Carlo -- refitting the same
    spectrum under fresh Poisson noise and taking the actual spread of the
    fitted total, which is what this number is supposed to estimate:

        separation   true spread   quadrature   this
          1.0 sigma       232.9       2323.9    238.3
          1.5 sigma       256.8        819.6    238.4
          2.0 sigma       237.8        437.5    238.6
          3.0 sigma       253.6        268.9    239.0

    The true spread barely moves with separation, which is the physics:
    the region's total is well measured however the fit divides it. This
    tracks that to within 7% throughout, where quadrature was out by a
    factor of ten at one sigma.

    A single-peak fit has one term and is unaffected. Well-separated peaks
    are very nearly unaffected too (0.14% at 25 sigma apart), so this
    changes reported numbers only for genuine multiplets.

    Gradients are central differences over EVERY free parameter, including
    the ones the area does not depend on -- a position or background
    coefficient simply gets a zero partial and contributes nothing, which
    is both correct and cheaper to write than enumerating which parameters
    matter. Undetermined parameters are dropped exactly as
    _area_with_error drops them, and for the same reason.
    """
    if pcov is None:
        # Nothing was fitted, so there is no uncertainty rather than an
        # unknown one -- matching _area_with_error's own fixed-fit branch.
        return 0.0

    def total_at(values):
        amplitudes, _positions, sigmas, r, beta = _unpack_named(
            values, n_peaks, link_widths, enable_left_tail
        )
        return float(sum(
            hypermet_area(
                amplitude,
                # abs() here rather than on the stored parameter so the
                # central difference sees the same |sigma| the reported
                # areas were computed from, sign flip included.
                abs(sigma),
                r if r is not None else 0.0,
                beta if beta is not None else 1.0,
            )
            for amplitude, sigma in zip(amplitudes, sigmas)
        ))

    diag = np.diag(pcov)
    indices, partials = [], []
    for index, name in enumerate(free_names):
        if not (math.isfinite(diag[index]) and diag[index] >= 0.0):
            continue  # undetermined -- see the docstring
        value = values_by_name[name]
        step = 1e-6 * max(abs(value), 1e-3)
        hi = value + step
        lo = value - step
        if name == "tail_beta" and lo <= 0.0:
            # Keep the perturbation inside the domain, as _area_with_error
            # does: beta is strictly positive and erfcx(y) changes
            # character across zero. sigma needs no such guard here because
            # total_at takes its absolute value.
            lo = value * 0.5
        high = dict(values_by_name); high[name] = hi
        low = dict(values_by_name); low[name] = lo
        partials.append((total_at(high) - total_at(low)) / (hi - lo))
        indices.append(index)

    if not indices:
        return float("nan")

    gradient = np.asarray(partials, dtype=float)
    block = np.asarray(pcov)[np.ix_(indices, indices)]
    variance = float(gradient @ block @ gradient)
    if not math.isfinite(variance) or variance < 0.0:
        return float("nan")
    return math.sqrt(variance)


def _area_param_names(index, link_widths, enable_left_tail):
    """(kind, parameter-name) pairs for the parameters a peak's area
    depends on. `kind` is hypermet_area's own argument name, so
    _area_with_error can perturb the right one without re-deriving which
    fitted parameter is which."""
    params = [
        ("amp", f"amp_{index}"),
        ("sigma", "sigma" if link_widths else f"sigma_{index}"),
    ]
    if enable_left_tail:
        params += [("r", "tail_fraction"), ("beta", "tail_beta")]
    return params


def _uncertainties_from(pcov, n):
    """Per-parameter standard errors from a covariance matrix, NaN where
    the variance is missing (NaN from _covariance_from) or invalid (a
    negative variance means the same thing: not determined by this
    data). Computed with an explicit mask rather than sqrt-ing the whole
    diagonal, so an undetermined parameter does not raise a numpy
    invalid-value warning on every fit."""
    if pcov is None:
        return np.full(n, np.nan)
    diag = np.diag(pcov)
    perr = np.full(n, np.nan)
    determined = np.isfinite(diag) & (diag >= 0)
    perr[determined] = np.sqrt(diag[determined])
    return perr


def _invert_or_none(alpha):
    """inv(alpha) if the result is a usable covariance, else None. A
    non-finite entry or a negative variance both mean "not determined by
    this data"; only the diagonal is checked for sign, since negative
    off-diagonal covariances are perfectly legitimate for correlated
    parameters."""
    try:
        inverse = np.linalg.inv(alpha)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(inverse)) or np.any(np.diag(inverse) < 0):
        return None
    return inverse


def _covariance_from(alpha, n, optional=()):
    """Inverts the curvature matrix, falling back to dropping `optional`
    parameters (by index) rather than losing the whole fit.

    The full matrix is tried FIRST and returned unchanged when it works,
    so every fit that already produced uncertainties keeps exactly the
    ones it had -- this is deliberately not a general re-derivation of
    how uncertainties are computed.

    Only when that fails do the `optional` parameters get dropped and the
    remaining submatrix inverted on its own. They are the tail pair:
    once tail_fraction reaches ~0 the tail term stops contributing, so
    d(model)/d(tail_beta) vanishes, alpha loses rank, and a plain inv()
    takes the peak's own well-determined position/width/amplitude down
    with it. Dropped parameters come back NaN -- genuinely unknown --
    while the peak parameters keep real uncertainties.

    Deliberately NOT a pseudo-inverse: pinv would return a minimum-norm
    variance near ZERO for an undetermined parameter, which reads as
    "measured perfectly" -- the most dangerous possible answer here. NaN
    is the honest one.
    """
    if not np.all(np.isfinite(alpha)):
        return np.full((n, n), np.nan)

    full = _invert_or_none(alpha)
    if full is not None:
        return full

    pcov = np.full((n, n), np.nan)
    keep = np.array([i for i in range(n) if i not in set(optional)], dtype=int)
    if keep.size == 0 or keep.size == n:
        return pcov
    sub = _invert_or_none(alpha[np.ix_(keep, keep)])
    if sub is None:
        return pcov
    pcov[np.ix_(keep, keep)] = sub
    return pcov
