import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

FWHM_FACTOR = 2.3548200450309493  # 2*sqrt(2*ln(2))

TAIL_FRACTION_MAX = 0.3
TAIL_BETA_MIN = 0.1

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
    """
    dx = x - position
    w = dx / (sigma * np.sqrt(2))
    gaussian_core = np.exp(-w ** 2)

    y = sigma / (beta * np.sqrt(2))
    erfc_y = erfc(y)
    if erfc_y < 1e-300:
        # y is so large that erfc(y) has underflowed to zero -- an
        # extreme, unphysical width/decay ratio that only an
        # unconverged optimizer iterate would ever produce. Treat the
        # tail as vanishing rather than divide by (effectively) zero.
        return (1 - r) * gaussian_core

    # Matches gf3's own overflow guard: exp(dx/beta) grows unbounded
    # for dx/beta > 0, but the true (mathematically bounded) tail
    # contribution there is negligible once |dx/beta| is large -- gf3
    # itself zeroes the tail term entirely past this same threshold
    # rather than risk exp() overflowing before erfc() can suppress it.
    ratio = dx / beta
    tail = np.zeros_like(x, dtype=float)
    safe = np.abs(ratio) <= 12.0
    z = w[safe] + y
    tail[safe] = np.exp(ratio[safe]) * erfc(z) / erfc_y

    return (1 - r) * gaussian_core + r * tail


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


def compute_background(x, y, left_bg_region, right_bg_region):
    left_x, left_y = _region_centroid(x, y, left_bg_region)
    right_x, right_y = _region_centroid(x, y, right_bg_region)
    if right_x == left_x:
        raise FitError("Background regions must not share the same mean channel")
    slope = (right_y - left_y) / (right_x - left_x)
    intercept = left_y - slope * left_x
    return slope, intercept


def _parameter_names(n_peaks, link_widths, enable_left_tail):
    """Canonical, ordered list of every fittable parameter's name for
    a given (n_peaks, link_widths, enable_left_tail) configuration.
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
    return names


def _unpack_named(values_by_name, n_peaks, link_widths, enable_left_tail):
    """Extracts (amplitudes, positions, sigmas, tail_fraction, tail_beta)
    from a {name: value} mapping covering every name _parameter_names()
    would produce for this configuration. `sigmas` is always length
    n_peaks (the shared value repeated if linked); tail_fraction/
    tail_beta are None if enable_left_tail is False."""
    amplitudes = [values_by_name[f"amp_{i}"] for i in range(n_peaks)]
    positions = [values_by_name[f"pos_{i}"] for i in range(n_peaks)]
    if link_widths:
        sigmas = [values_by_name["sigma"]] * n_peaks
    else:
        sigmas = [values_by_name[f"sigma_{i}"] for i in range(n_peaks)]
    tail_fraction = values_by_name.get("tail_fraction")
    tail_beta = values_by_name.get("tail_beta")
    return amplitudes, positions, sigmas, tail_fraction, tail_beta


def _make_model(names, free_names, fixed_params, n_peaks, link_widths, enable_left_tail):
    def model(x, *free_values):
        values_by_name = dict(fixed_params)
        values_by_name.update(zip(free_names, free_values))
        amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_named(
            values_by_name, n_peaks, link_widths, enable_left_tail
        )
        result = np.zeros_like(x, dtype=float)
        for amplitude, position, sigma in zip(amplitudes, positions, sigmas):
            if enable_left_tail:
                result = result + amplitude * hypermet_left_tail(
                    x, position, sigma, tail_fraction, tail_beta
                )
            else:
                result = result + amplitude * np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
        return result
    return model


def _initial_guess(
    free_names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail,
    initial_guess_overrides=None,
):
    lo, hi = fit_region
    region_width = hi - lo
    n_peaks = len(peak_positions)
    fallback_sigma0 = max(region_width / (4 * n_peaks), 1e-6)
    sigma0 = _measure_width(x_fit, y_sub, peak_positions, fallback_sigma0)

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
        guess_by_name["tail_beta"] = max(sigma0, TAIL_BETA_MIN)
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
            damping.append(_ParamDamping("tail_beta"))
    return damping


def fit_peaks(
    x, y, left_bg_region, right_bg_region, fit_region, peak_positions,
    link_widths=True, enable_left_tail=False, fixed_params=None,
    initial_guess_overrides=None,
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
    names = _parameter_names(n_peaks, link_widths, enable_left_tail)
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
    y_err = np.sqrt(np.maximum(y_fit, 1.0))
    model_named = _make_model(names, free_names, fixed_params, n_peaks, link_widths, enable_left_tail)

    if not free_names:
        # Every parameter is fixed -- nothing to optimize. Evaluate
        # directly at the fixed values instead of calling curve_fit
        # with an empty parameter vector. No fit was performed, so
        # every value's uncertainty is exactly 0.0.
        values_by_name = dict(fixed_params)
        err_by_name = {name: 0.0 for name in names}
    else:
        p0 = _initial_guess(
            free_names, x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail,
            initial_guess_overrides=initial_guess_overrides,
        )
        model = lambda xx, pp: model_named(xx, *pp)
        damping = _build_param_damping(free_names, fixed_params, link_widths)

        try:
            popt, pcov = _marquardt_fit(
                model, x_fit, y_sub, y_err, p0, damping, fit_region_bounds=fit_region,
            )
        except np.linalg.LinAlgError as exc:
            raise FitError(f"Fit did not converge: {exc}") from exc

        if pcov is None or not np.all(np.isfinite(pcov)) or np.any(np.diag(pcov) < 0):
            # A covariance matrix's diagonal holds variances, which
            # become perr via sqrt() below -- a negative diagonal entry
            # is never physically valid (it would silently corrupt the
            # UI with NaN uncertainties), even though the matrix itself
            # is finite in that case (hence the message below covers
            # both, distinctly, rather than calling a negative variance
            # "non-finite"). Off-diagonal negative entries are untouched
            # by this check -- those are legitimate for correlated
            # parameters.
            if pcov is not None and np.all(np.isfinite(pcov)):
                detail = (
                    "the fit converged but one or more parameters are not "
                    "well-determined by this data (invalid negative "
                    "uncertainty)"
                )
                if enable_left_tail:
                    detail += (
                        "; if the peak has no real tail, this is often the "
                        "tail parameters specifically -- try unchecking "
                        "Left tail"
                    )
            else:
                detail = "the fit produced a non-finite covariance matrix"
            raise FitError(detail[0].upper() + detail[1:])

        perr = np.sqrt(np.diag(pcov))
        values_by_name = dict(fixed_params)
        values_by_name.update(zip(free_names, popt))
        err_by_name = {name: 0.0 for name in fixed_params}
        err_by_name.update(zip(free_names, perr))

    # Reduced chi-square of the final (fitted or fully-fixed) model
    # against the background-subtracted data, weighted the same way the
    # solver itself weighted residuals. None (undisplayable) rather than
    # a divide-by-zero when there are exactly as many data points as free
    # parameters (zero degrees of freedom) -- mirrors TV's own refusal to
    # fit at all in that case (vsCurFit.c's CurFreedom <= 0 check), except
    # this app already allows the fit to proceed and only the chi-square
    # reporting is affected.
    free_values = [values_by_name[name] for name in free_names]
    model_curve = model_named(x_fit, *free_values)
    chi2 = float(np.sum(((y_sub - model_curve) / y_err) ** 2))
    dof = x_fit.size - len(free_names)
    reduced_chi2 = chi2 / dof if dof > 0 else None

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
        # This is the analytic integral of the plain Gaussian core only.
        # When enable_left_tail is True, the fitted shape also carries a
        # tail term (see hypermet_left_tail) whose own contribution to the
        # true integral is not included here -- a known approximation.
        area = amplitude * sigma * np.sqrt(2 * np.pi)
        rel_err_sq = 0.0
        if amplitude != 0:
            rel_err_sq += (amplitude_err / amplitude) ** 2
        if sigma != 0:
            rel_err_sq += (sigma_err / sigma) ** 2
        area_err = abs(area) * np.sqrt(rel_err_sq)
        # Full (background-included) area for this one peak: its own net
        # area plus the background "under" it, using the standard
        # gamma-spectroscopy convention of the local linear-background
        # level at the peak's own center times that peak's own FWHM --
        # the per-peak analog of Integration mode's flat
        # density-times-width background area. The background term is a
        # deterministic function of the (unweighted, uncertainty-free)
        # linear fit, so it adds no uncertainty of its own: full_area_err
        # is exactly area_err.
        background_under_peak = (slope * position + intercept) * fwhm
        full_area = area + background_under_peak
        full_area_err = area_err
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
    # quadrature -- unlike Integration, there's no separate "background
    # area uncertainty" to add in here, since the linear background used
    # by this fit path is a deterministic two-point line, not a
    # statistically-fit quantity with its own propagated uncertainty.
    gross_area = float(np.sum(y_fit))
    # Deliberately no guard here, matching TV's own lack of one: unlike
    # integrate_region() (which now rejects a negative-count region
    # outright, see the FitError raised above `ds = s.copy()`), this is
    # one auxiliary summary field on an ALREADY-SUCCESSFUL fit -- a
    # negative gross_area (e.g. fitting a Subtract Spectra result)
    # produces a NaN here without discarding the rest of a valid fit.
    gross_area_err = float(np.sqrt(gross_area))
    net_area = float(sum(peak.area for peak in peaks))
    net_area_err = float(np.sqrt(sum(peak.area_err ** 2 for peak in peaks)))

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
    )


def parameter_names(n_peaks, link_widths, enable_left_tail):
    """Public alias of _parameter_names -- fit_mode.py's Fit Parameters
    panel needs this exact ordering to know which rows to display."""
    return _parameter_names(n_peaks, link_widths, enable_left_tail)


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
    return values


def integrate_region(x, y, left_bg_region, right_bg_region, fit_region):
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
    function was written -- see the design spec's Testing section."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lo, hi = fit_region
    mask = (x >= lo) & (x <= hi)
    idx = x[mask]
    s = y[mask]
    n = idx.size
    if n == 0:
        raise FitError(f"Fit region {fit_region} contains no data")
    if np.any(s < 0.0):
        raise FitError(
            "Cannot integrate a region containing negative counts (e.g. "
            "from a Subtract Spectra result) -- the uncertainty "
            "calculation used here assumes non-negative Poisson counts, "
            "matching TV's own convention."
        )
    ds = s.copy()  # Poisson variance per channel = the count itself

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
            if np.any(bg_y < 0.0):
                raise FitError(
                    "Cannot integrate a region containing negative counts "
                    "(e.g. from a Subtract Spectra result) -- the "
                    "uncertainty calculation used here assumes "
                    "non-negative Poisson counts, matching TV's own "
                    "convention."
                )
            bg_count += float(np.sum(bg_y))
            bg_dcount += float(np.sum(bg_y))

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
    hard-clamped to this project's existing bounds)."""

    __slots__ = ("kind", "sigma_index", "sigma_value")

    def __init__(self, kind, sigma_index=None, sigma_value=None):
        self.kind = kind
        self.sigma_index = sigma_index
        self.sigma_value = sigma_value


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


def _numeric_jacobian(model, x, p):
    """Central-difference Jacobian -- matches TV's own fallback
    (CurNumericDerivation) for fit-function modules without an analytic
    derivative; avoids hand-deriving hypermet's erfc-based partials."""
    n = len(p)
    J = np.zeros((len(x), n))
    for i in range(n):
        h = max(abs(p[i]), 1e-6) * 1e-6
        p_hi = p.copy(); p_hi[i] += h
        p_lo = p.copy(); p_lo[i] -= h
        J[:, i] = (model(x, p_hi) - model(x, p_lo)) / (2 * h)
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
    return p_try


def _marquardt_fit(model, x, y, y_err, p0, damping, fit_region_bounds=None):
    """Damped Levenberg-Marquardt solver replicating TV's fit procedure
    (tv-1.9.13/lib/tv/vsCurFit.c's CurFit + VsFitFct.c's CHANGE_TRY).
    `model(x, p)` takes the full parameter array (not *args, unlike
    scipy's curve_fit convention) and returns the model y-values.
    `damping` is a list of _ParamDamping, one per entry in p0, telling
    the solver how to limit each parameter's per-iteration step. Returns
    (popt, pcov) with the same meaning as
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
        return _numeric_jacobian(model, x, pt) * weights[:, None]

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
                    p_try2 = _clamp_trial(p - 0.5 * delta, damping)
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

    alpha_final = J.T @ J
    try:
        pcov = np.linalg.inv(alpha_final)
    except np.linalg.LinAlgError:
        pcov = np.full((n, n), np.inf)
    return p, pcov
