from dataclasses import dataclass

import numpy as np


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
