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


def _unpack_params(params, n_peaks, link_widths, enable_left_tail):
    """Splits a flat curve_fit parameter array into
    (amplitudes, positions, sigmas, tail_fraction, tail_beta). `sigmas`
    is always length n_peaks (the shared value repeated if linked);
    tail_fraction/tail_beta are None if enable_left_tail is False. This
    is the single canonical parameter ordering shared by the model
    function, the initial guess, the bounds, and result extraction --
    changing the layout means changing it only here."""
    idx = 0
    amplitudes = []
    positions = []
    sigmas = []
    for _ in range(n_peaks):
        amplitudes.append(params[idx]); idx += 1
        positions.append(params[idx]); idx += 1
        if not link_widths:
            sigmas.append(params[idx]); idx += 1
    if link_widths:
        shared_sigma = params[idx]; idx += 1
        sigmas = [shared_sigma] * n_peaks
    tail_fraction = None
    tail_beta = None
    if enable_left_tail:
        tail_fraction = params[idx]; idx += 1
        tail_beta = params[idx]; idx += 1
    return amplitudes, positions, sigmas, tail_fraction, tail_beta


def _make_model(n_peaks, link_widths, enable_left_tail):
    def model(x, *params):
        amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_params(
            params, n_peaks, link_widths, enable_left_tail
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


def _initial_guess(x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail):
    lo, hi = fit_region
    region_width = hi - lo
    n_peaks = len(peak_positions)
    sigma0 = max(region_width / (4 * n_peaks), 1e-6)
    if not link_widths and n_peaks > 1:
        # With independent per-peak sigmas, a starting width this wide
        # (derived from the whole fit region) makes neighboring peaks
        # overlap heavily and lets curve_fit's unconstrained
        # Levenberg-Marquardt solver settle into a spurious local
        # minimum (e.g. a negative-amplitude "correction" peak) instead
        # of recovering each peak's own width. Cap the guess at a
        # quarter of the closest peak spacing so peaks start out
        # reasonably well-separated; linked-width and single-peak fits
        # aren't prone to this failure mode, so they keep the original
        # region-based guess unchanged.
        sorted_positions = sorted(peak_positions)
        min_spacing = min(
            b - a for a, b in zip(sorted_positions, sorted_positions[1:])
        )
        sigma0 = max(min(sigma0, min_spacing / 4), 1e-6)

    guess = []
    for pos in peak_positions:
        idx = int(np.argmin(np.abs(x_fit - pos)))
        amplitude0 = float(y_sub[idx])
        guess.append(amplitude0)
        guess.append(float(pos))
        if not link_widths:
            guess.append(sigma0)
    if link_widths:
        guess.append(sigma0)
    if enable_left_tail:
        guess.append(0.05)
        guess.append(max(sigma0, TAIL_BETA_MIN))
    return guess


def _bounds(n_peaks, link_widths, enable_left_tail):
    """Only needed when enable_left_tail is True (to keep the tail
    fraction genuinely small and the decay constant away from zero);
    returns None otherwise so the no-tail fits keep using curve_fit's
    default unconstrained method, unchanged from v1's behavior."""
    if not enable_left_tail:
        return None
    lower = []
    upper = []
    for _ in range(n_peaks):
        lower.append(-np.inf); upper.append(np.inf)  # amplitude
        lower.append(-np.inf); upper.append(np.inf)  # position
        if not link_widths:
            lower.append(1e-6); upper.append(np.inf)  # sigma
    if link_widths:
        lower.append(1e-6); upper.append(np.inf)  # shared sigma
    lower.append(0.0); upper.append(TAIL_FRACTION_MAX)  # tail_fraction
    lower.append(TAIL_BETA_MIN); upper.append(np.inf)   # tail_beta
    return (lower, upper)


def fit_peaks(
    x, y, left_bg_region, right_bg_region, fit_region, peak_positions,
    link_widths=True, enable_left_tail=False,
):
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

    n_peaks = len(peak_positions)
    n_params = (
        2 * n_peaks
        + (1 if link_widths else n_peaks)
        + (2 if enable_left_tail else 0)
    )
    if x_fit.size < n_params:
        raise FitError(
            f"Fit region has {x_fit.size} data points, need at least "
            f"{n_params} for {n_peaks} peak(s)"
        )

    y_sub = y_fit - (slope * x_fit + intercept)
    p0 = _initial_guess(x_fit, y_sub, fit_region, peak_positions, link_widths, enable_left_tail)
    y_err = np.sqrt(np.maximum(y_fit, 1.0))
    model = _make_model(n_peaks, link_widths, enable_left_tail)
    bounds = _bounds(n_peaks, link_widths, enable_left_tail)

    try:
        if bounds is not None:
            popt, pcov = curve_fit(
                model, x_fit, y_sub, p0=p0, sigma=y_err, absolute_sigma=True, bounds=bounds
            )
        else:
            popt, pcov = curve_fit(
                model, x_fit, y_sub, p0=p0, sigma=y_err, absolute_sigma=True
            )
    except (RuntimeError, ValueError) as exc:
        raise FitError(f"Fit did not converge: {exc}") from exc

    if pcov is None or not np.all(np.isfinite(pcov)):
        raise FitError("Fit produced a non-finite covariance matrix")

    perr = np.sqrt(np.diag(pcov))
    amplitudes, positions, sigmas, tail_fraction, tail_beta = _unpack_params(
        popt, n_peaks, link_widths, enable_left_tail
    )
    amplitude_errs, position_errs, sigma_errs, tail_fraction_err, tail_beta_err = _unpack_params(
        perr, n_peaks, link_widths, enable_left_tail
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
        area = amplitude * sigma * np.sqrt(2 * np.pi)
        rel_err_sq = 0.0
        if amplitude != 0:
            rel_err_sq += (amplitude_err / amplitude) ** 2
        if sigma != 0:
            rel_err_sq += (sigma_err / sigma) ** 2
        area_err = abs(area) * np.sqrt(rel_err_sq)
        peaks.append(
            PeakResult(
                position=float(position), position_err=float(position_err),
                fwhm=float(fwhm), fwhm_err=float(fwhm_err),
                area=float(area), area_err=float(area_err),
                amplitude=float(amplitude), sigma=float(sigma),
            )
        )

    return FitResult(
        left_bg_region=tuple(left_bg_region), right_bg_region=tuple(right_bg_region),
        fit_region=tuple(fit_region), background_slope=float(slope),
        background_intercept=float(intercept), peaks=peaks,
        link_widths=link_widths,
        tail_fraction=float(tail_fraction) if tail_fraction is not None else None,
        tail_fraction_err=float(tail_fraction_err) if tail_fraction_err is not None else None,
        tail_beta=float(tail_beta) if tail_beta is not None else None,
        tail_beta_err=float(tail_beta_err) if tail_beta_err is not None else None,
    )
