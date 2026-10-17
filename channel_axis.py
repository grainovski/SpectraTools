"""A spectrum's channel axis: where each of its bins sits on the detector's
own channel scale.

Almost every spectrum this app reads has one bin per ADC channel, so bin i
IS channel i and nothing here applies. A ROOT histogram is different: its
axis carries real coordinates, and a CoMPASS energy-energy matrix is
written with 1023 bins spanning ADC 0-16383, about 16 ADC channels per
bin. ROOT's browser and HDTV both draw such a histogram against those
coordinates -- the axis reads 0-16383 and a peak sits at the same ADC
value it has in the full-resolution 1D spectrum -- while a plain bin
index would put it at 1/16 of that. This module is how this app does what
they do.

The rule. A ChannelAxis maps bin index i to the detector channel

    channel = offset + width * i

The fitting engine and every stored region, mark and fit result stay in
BIN units, untouched. The axis is applied wherever a channel number is
drawn, read off the cursor, typed in, reported or exported, and an energy
calibration is always a function of the CHANNEL, never of the bin -- so a
calibration made on a 16k spectrum applies unchanged to a 1k cut of the
same detector.

`None` stands for the identity axis everywhere, so the ordinary spectrum
costs nothing and cannot be affected.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from calibration import Calibration


@dataclass(frozen=True)
class ChannelAxis:
    offset: float
    width: float

    def __post_init__(self):
        if not (math.isfinite(self.offset) and math.isfinite(self.width)):
            raise ValueError(f"channel axis must be finite, got {self!r}")
        if self.width <= 0.0:
            raise ValueError(f"channel axis width must be positive, got {self.width!r}")

    def to_channel(self, index):
        """Bin index (scalar or array) -> detector channel."""
        return self.offset + self.width * index

    def to_index(self, channel):
        """Detector channel (scalar or array) -> fractional bin index."""
        return (channel - self.offset) / self.width


def from_bin_edges(edges):
    """The ChannelAxis a ROOT axis with these bin edges implies, or None
    when it is the ordinary one-bin-per-channel axis.

    A bin's channel is the mean of the integer ADC values it collects --
    the digitizer writes integers, and a channel IS one of those integers.
    Bin [a, b) holds the integers ceil(a) .. ceil(b) - 1, so its mean is

        (ceil(a) + ceil(b) - 1) / 2

    and the axis offset is that mean, less width * bin, averaged over the
    bins. For a 1:1 axis over 0-N, bin [i, i+1) holds exactly the integer
    i, so the offset is 0 -- channel i, as this app has always read it.
    For a CoMPASS energy-energy axis (1023 bins over 0-16383, 16.0147 wide)
    the edges fall between integers, so each bin's integers average its
    CENTRE: offset 8.006.

    6.3.0 used "centre - 1/2" for every axis. That is right for the 1:1
    case it was taken from and half a channel low for the CoMPASS one: a
    400-fit sweep of integer ADC data put every position on a 1023-bin
    cut 0.53 channels below the same line in the 16k spectrum (mean pull
    -5.4). With the integer mean the offset is 0.03 channels.

    What a linear axis cannot carry is the bin-to-bin wobble: a 16.0147
    bin holds 16 or 17 integers, whose mean sits up to half a channel
    either side of the line. That leaves a position systematic of about
    0.27 channels RMS that only shows at high statistics -- documented in
    the Knowledge Database, and shared by ROOT and HDTV, which use the
    bin centre.

    Below one channel per bin the integer model means nothing (a bin can
    hold no integer at all), so such an axis keeps "centre - 1/2", which
    agrees with the 1:1 case.

    Raises ValueError on non-uniform binning, which no linear axis can
    describe.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        return None
    widths = np.diff(edges)
    width = float(widths[0])
    if not np.allclose(widths, width, rtol=1e-9, atol=0.0):
        raise ValueError("non-uniform bin widths cannot be expressed as a channel axis")
    if not (math.isfinite(width) and width > 0.0):
        raise ValueError(f"unusable bin width ({width})")
    if width >= 1.0:
        # Tolerance on the ceilings: edges built as low + width * i carry
        # float error, and an edge meant to be the integer k must not
        # count as k + epsilon (whose ceiling is k + 1).
        tol = 1e-6
        ceilings = np.ceil(edges - tol)
        means = (ceilings[:-1] + ceilings[1:] - 1.0) / 2.0
        offset = float(np.mean(means - width * np.arange(means.size)))
    else:
        offset = float(edges[0]) + width / 2.0 - 0.5
    if np.isclose(offset, 0.0, rtol=0.0, atol=1e-9) and np.isclose(width, 1.0, rtol=1e-12, atol=0.0):
        return None
    return ChannelAxis(offset, width)


def axis_of(spectrum):
    """The spectrum's ChannelAxis, or None for the identity. Tolerates any
    object without the attribute (a test double, an older pickle)."""
    return getattr(spectrum, "axis", None)


def to_channel(axis, index):
    return index if axis is None else axis.to_channel(index)


def to_index(axis, channel):
    return channel if axis is None else axis.to_index(channel)


def width_to_channel(axis, width):
    """A width or an uncertainty, bins -> channels: scales, never shifts."""
    return width if axis is None else axis.width * width


def same_axis(a, b):
    """Whether two axes describe the same channel scale (None == identity)."""
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a.offset, b.offset, rel_tol=1e-9, abs_tol=1e-9) and math.isclose(
        a.width, b.width, rel_tol=1e-9)


def rebinned(axis, factor):
    """The axis of a spectrum after summing every `factor` bins into one.

    New bin k holds old bins [k*n, k*n + n-1], so it sits at the centre
    of that group -- old index k*n + (n-1)/2 -- the same anchor
    Calibration.rescaled uses and for the same reason.
    """
    current = axis if axis is not None else ChannelAxis(0.0, 1.0)
    return ChannelAxis(
        current.offset + current.width * (factor - 1) / 2.0, current.width * factor)


def as_calibration(axis):
    """The axis written as a linear Calibration (channel = a + b*index),
    or None for the identity."""
    if axis is None:
        return None
    return Calibration(kind="linear", a=axis.offset, b=axis.width)


def compose(calibration, axis):
    """calibration(axis(index)) as one Calibration in BIN units.

    E = a + b*ch + c*ch**2 with ch = o + w*i expands to

        a' = a + b*o + c*o**2,   b' = w*(b + 2*c*o),   c' = c*w**2

    -- still linear or quadratic, so everything that takes a Calibration
    (apply, derivative, invert) works on it unchanged. Coefficient errors
    do not carry over: they describe the channel-space fit, and nothing
    reads them off a composed calibration.
    """
    if axis is None or calibration is None:
        return calibration
    a, b, c = calibration.a, calibration.b, calibration.c
    o, w = axis.offset, axis.width
    return Calibration(
        kind=calibration.kind,
        a=a + b * o + c * o * o,
        b=w * (b + 2.0 * c * o),
        c=c * w * w,
    )


def decompose(calibration, axis):
    """Inverse of compose: a calibration in BIN units -> the same
    calibration as a function of the CHANNEL. Used where a calibration is
    derived from bin-space positions and has to be stored the way every
    other calibration is."""
    if axis is None or calibration is None:
        return calibration
    a, b, c = calibration.a, calibration.b, calibration.c
    o, w = axis.offset, axis.width
    return Calibration(
        kind=calibration.kind,
        a=a - b * o / w + c * o * o / (w * w),
        b=b / w - 2.0 * c * o / (w * w),
        c=c / (w * w),
        coefficient_errors=calibration.coefficient_errors,
    )


def _param_to_channel(axis, name, value):
    if name.startswith("pos_"):
        return axis.to_channel(value)
    if name == "sigma" or name.startswith("sigma_") or name == "tail_beta":
        return axis.width * value
    if name == "bg_c1":
        return value / axis.width
    return value


def _region(axis, region):
    if region is None:
        return None
    return tuple(axis.to_channel(v) for v in region)


def result_in_channels(result, axis):
    """A copy of a FitResult or IntegrationResult with every quantity that
    has a channel unit re-expressed on `axis`; the result itself unchanged.

    For reports and exports, which print what they are given. The stored
    result stays in bins because it is what the fit engine, the plot and
    a refit all read.

    Positions shift and scale; widths, uncertainties and the tail's beta
    (a length on the x axis) only scale; areas, amplitudes, chi2 and the
    background LEVEL are counts per bin and do not change -- the bins are
    the same bins. The background line is re-expressed so it still
    evaluates to the same level at the same place.
    """
    if axis is None:
        return result
    # Imported here so this module stays importable without the fit engine.
    from peak_fit import FitResult, IntegrationResult

    w = axis.width
    if isinstance(result, IntegrationResult):
        changes = {
            "left_bg_region": _region(axis, result.left_bg_region),
            "right_bg_region": _region(axis, result.right_bg_region),
            "fit_region": _region(axis, result.fit_region),
            # Counts per BIN; the bins are the same bins.
            "background_density": result.background_density,
        }
        for prefix in ("gross", "background", "net"):
            changes[f"{prefix}_centroid"] = axis.to_channel(getattr(result, f"{prefix}_centroid"))
            for name in (f"{prefix}_centroid_err", f"{prefix}_fwhm", f"{prefix}_fwhm_err"):
                changes[name] = w * getattr(result, name)
        return dataclasses.replace(result, **changes)

    if not isinstance(result, FitResult):
        return result
    peaks = [
        dataclasses.replace(
            peak,
            position=axis.to_channel(peak.position),
            position_err=w * peak.position_err,
            fwhm=w * peak.fwhm,
            fwhm_err=w * peak.fwhm_err,
            sigma=w * peak.sigma,
            sigma_err=w * peak.sigma_err,
        )
        for peak in result.peaks
    ]
    slope = result.background_slope / w
    intercept = result.background_intercept - result.background_slope * axis.offset / w
    anchors = tuple(
        (axis.to_channel(x), var) for x, var in result.background_anchors
    )
    covariance = result.background_covariance
    if covariance:
        reference, var_c0, cov, var_c1 = covariance
        covariance = (axis.to_channel(reference), var_c0, cov / w, var_c1 / (w * w))
    return dataclasses.replace(
        result,
        left_bg_region=_region(axis, result.left_bg_region),
        right_bg_region=_region(axis, result.right_bg_region),
        fit_region=_region(axis, result.fit_region),
        background_slope=slope,
        background_intercept=intercept,
        peaks=peaks,
        tail_beta=None if result.tail_beta is None else w * result.tail_beta,
        tail_beta_err=None if result.tail_beta_err is None else w * result.tail_beta_err,
        fixed_params={
            name: _param_to_channel(axis, name, value)
            for name, value in result.fixed_params.items()
        },
        background_anchors=anchors,
        background_covariance=covariance,
    )


def fit_marks_onto_axis(result, from_axis, to_axis):
    """A FitResult saved on one binning, carried onto another so that its
    marks and fixed parameters sit on the same detector CHANNELS.

    Bin b on `from_axis` is channel o1 + w1*b, which is bin
    (o1 - o2)/w2 + (w1/w2)*b on `to_axis` -- the same affine kind of map
    as an axis itself, so result_in_channels does the positions, widths and
    regions. What it leaves alone, because a report keeps the same bins,
    has to change here: anything in counts PER BIN. A bin w2/w1 times as
    wide holds w2/w1 times the counts, so a fixed amplitude or background
    level scales by that, and the background slope (per bin, per bin)
    once more.

    Only the marks of the result are meant to be used -- it is refitted
    on the new spectrum, never restored, since its fitted values describe
    the old bins.
    """
    def parts(axis):
        return (0.0, 1.0) if axis is None else (axis.offset, axis.width)

    (o1, w1), (o2, w2) = parts(from_axis), parts(to_axis)
    scale = w1 / w2
    moved = result_in_channels(result, ChannelAxis((o1 - o2) / w2, scale))
    per_bin = 1.0 / scale
    fixed = {}
    for name, value in moved.fixed_params.items():
        # Amplitude and level are counts per bin; the slope is per bin per
        # bin, and result_in_channels has already done one of its two.
        if name.startswith("amp_") or name in ("bg_c0", "bg_c1"):
            value = value * per_bin
        fixed[name] = value
    return dataclasses.replace(moved, fixed_params=fixed)
