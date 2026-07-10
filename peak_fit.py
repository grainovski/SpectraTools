from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

FWHM_FACTOR = 2.3548200450309493  # 2*sqrt(2*ln(2))

TAIL_FRACTION_MAX = 0.3
TAIL_BETA_MIN = 0.1


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


def _region_centroid(x, y, region):
    lo, hi = region
    mask = (x >= lo) & (x <= hi)
    if not np.any(mask):
        raise FitError(f"Background region {region} contains no data")
    return float(np.mean(x[mask])), float(np.mean(y[mask]))


def _compute_background(x, y, left_bg_region, right_bg_region):
    left_x, left_y = _region_centroid(x, y, left_bg_region)
    right_x, right_y = _region_centroid(x, y, right_bg_region)
    if right_x == left_x:
        raise FitError("Background regions must not share the same mean channel")
    slope = (right_y - left_y) / (right_x - left_x)
    intercept = left_y - slope * left_x
    return slope, intercept


def _gaussian_sum(x, *params):
    result = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amplitude, position, sigma = params[i], params[i + 1], params[i + 2]
        result = result + amplitude * np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
    return result


def _initial_guess(x_fit, y_sub, fit_region, peak_positions):
    lo, hi = fit_region
    region_width = hi - lo
    n_peaks = len(peak_positions)
    sigma0 = max(region_width / (4 * n_peaks), 1e-6)
    guess = []
    for pos in peak_positions:
        idx = int(np.argmin(np.abs(x_fit - pos)))
        amplitude0 = float(y_sub[idx])
        guess.extend([amplitude0, float(pos), sigma0])
    return guess


def fit_peaks(x, y, left_bg_region, right_bg_region, fit_region, peak_positions):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not peak_positions:
        raise FitError("At least one peak must be marked")

    slope, intercept = _compute_background(x, y, left_bg_region, right_bg_region)

    lo, hi = fit_region
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size == 0:
        raise FitError(f"Fit region {fit_region} contains no data")

    n_params = 3 * len(peak_positions)
    if x_fit.size < n_params:
        raise FitError(
            f"Fit region has {x_fit.size} data points, need at least "
            f"{n_params} for {len(peak_positions)} peak(s)"
        )

    y_sub = y_fit - (slope * x_fit + intercept)
    p0 = _initial_guess(x_fit, y_sub, fit_region, peak_positions)
    y_err = np.sqrt(np.maximum(y_fit, 1.0))

    try:
        popt, pcov = curve_fit(
            _gaussian_sum, x_fit, y_sub, p0=p0, sigma=y_err, absolute_sigma=True
        )
    except (RuntimeError, ValueError) as exc:
        raise FitError(f"Fit did not converge: {exc}") from exc

    if pcov is None or not np.all(np.isfinite(pcov)):
        raise FitError("Fit produced a non-finite covariance matrix")

    perr = np.sqrt(np.diag(pcov))

    peaks = []
    for i in range(len(peak_positions)):
        amplitude, position, sigma = popt[3 * i], popt[3 * i + 1], popt[3 * i + 2]
        amplitude_err, position_err, sigma_err = perr[3 * i], perr[3 * i + 1], perr[3 * i + 2]
        sigma = abs(sigma)
        fwhm = FWHM_FACTOR * sigma
        fwhm_err = FWHM_FACTOR * sigma_err
        area = amplitude * sigma * np.sqrt(2 * np.pi)
        rel_err_sq = 0.0
        if amplitude != 0:
            rel_err_sq += (amplitude_err / amplitude) ** 2
        if sigma != 0:
            rel_err_sq += (sigma_err / sigma) ** 2
        area_err = abs(area) * np.sqrt(rel_err_sq)
        peaks.append(
            PeakResult(
                position=float(position),
                position_err=float(position_err),
                fwhm=float(fwhm),
                fwhm_err=float(fwhm_err),
                area=float(area),
                area_err=float(area_err),
                amplitude=float(amplitude),
                sigma=float(sigma),
            )
        )

    return FitResult(
        left_bg_region=tuple(left_bg_region),
        right_bg_region=tuple(right_bg_region),
        fit_region=tuple(fit_region),
        background_slope=float(slope),
        background_intercept=float(intercept),
        peaks=peaks,
    )
