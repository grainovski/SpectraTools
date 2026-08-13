"""Original annotated matplotlib figures for the Knowledge Database help
page -- rendered from this app's own fit math (peak_fit.hypermet_left_tail,
calibration.Calibration), not stock images. Each function returns PNG
bytes; help_content.py embeds them as base64 data URIs so the generated
help page has no external file dependencies."""

from io import BytesIO

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from calibration import Calibration
from peak_fit import FWHM_FACTOR, hypermet_left_tail

_DATA_COLOR = "#1f77b4"
_FIT_COLOR = "#d62728"
_BG_COLOR = "#2ca02c"
_REGION_BG_COLOR = "#9467bd"
_REGION_FIT_COLOR = "#ff7f0e"


def _figure_to_png_bytes(fig):
    canvas = FigureCanvasAgg(fig)
    buf = BytesIO()
    canvas.print_png(buf)
    return buf.getvalue()


def anatomy_of_a_fit_figure():
    """A single peak with both background regions, the fit region, the
    peak position, and the fitted curve -- the reference figure the
    HowTo page's "performing a fit" steps point back to."""
    x = np.linspace(60.0, 140.0, 400)
    position, sigma, amplitude, r, beta = 100.0, 4.0, 480.0, 0.15, 6.0
    bg_slope, bg_intercept = 0.05, 15.0
    background = bg_intercept + bg_slope * x
    y = background + amplitude * hypermet_left_tail(x, position, sigma, r, beta)
    rng = np.random.default_rng(0)
    y_noisy = rng.poisson(np.clip(y, 1, None)).astype(float)

    left_bg = (65.0, 75.0)
    right_bg = (125.0, 135.0)
    fit_region = (80.0, 120.0)
    fwhm = sigma * FWHM_FACTOR

    fig = Figure(figsize=(7.5, 5.0), dpi=110)
    ax = fig.add_subplot(111)
    ax.step(x, y_noisy, where="mid", color=_DATA_COLOR, linewidth=1.0, label="Spectrum data")
    ax.plot(x, y, color=_FIT_COLOR, linewidth=1.8, label="Fitted curve")
    ax.plot(x, background, color=_BG_COLOR, linestyle="--", linewidth=1.3, label="Background")

    ax.axvspan(*left_bg, color=_REGION_BG_COLOR, alpha=0.25)
    ax.axvspan(*right_bg, color=_REGION_BG_COLOR, alpha=0.25)
    ax.axvspan(*fit_region, color=_REGION_FIT_COLOR, alpha=0.10)

    ax.axvline(position, color="black", linestyle=":", linewidth=1.0)
    ax.axvline(position - fwhm / 2, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(position + fwhm / 2, color="gray", linestyle=":", linewidth=0.8)

    ax.text(
        0.5, -0.13,
        "purple = background regions (mark with B)   "
        "orange = fit region (mark with R)   "
        "dotted black = peak position (mark with P), dotted gray = FWHM span",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, wrap=True,
    )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts")
    ax.set_title("Anatomy of a fit")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _figure_to_png_bytes(fig)


def tail_effect_figure():
    """The same peak shape with tail_fraction r=0 (pure Gaussian) vs.
    r=0.25, so the low-energy tail that r and beta control is visually
    obvious rather than just a formula."""
    x = np.linspace(70.0, 130.0, 400)
    position, sigma, amplitude = 100.0, 4.0, 1.0

    y_gauss = amplitude * hypermet_left_tail(x, position, sigma, 0.0, 6.0)
    y_tailed = amplitude * hypermet_left_tail(x, position, sigma, 0.25, 6.0)

    fig = Figure(figsize=(7.5, 4.5), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(x, y_gauss, color=_DATA_COLOR, linewidth=1.8, label="tail_fraction r = 0 (pure Gaussian)")
    ax.plot(x, y_tailed, color=_FIT_COLOR, linewidth=1.8, label="tail_fraction r = 0.25 (visible tail)")
    ax.fill_between(x, y_gauss, y_tailed, where=(y_tailed > y_gauss), color=_FIT_COLOR, alpha=0.15)

    ax.annotate(
        "low-energy tail\n(controlled by r and beta)",
        xy=(position - sigma * 2.2, amplitude * 0.08),
        xytext=(position - sigma * 6.5, amplitude * 0.35),
        arrowprops=dict(arrowstyle="->", color="black"), fontsize=8,
    )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("The tail effect")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _figure_to_png_bytes(fig)


def sigma_fwhm_figure():
    """A single Gaussian core with sigma and the FWHM (full width at
    half maximum) both marked -- the reference figure for the "Sigma
    and FWHM" section, which every width this program reports (fitted
    or integrated) converts through."""
    x = np.linspace(70.0, 130.0, 400)
    position, sigma = 100.0, 8.0
    y = np.exp(-((x - position) ** 2) / (2 * sigma ** 2))
    fwhm = sigma * FWHM_FACTOR

    fig = Figure(figsize=(7.0, 4.5), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(x, y, color=_DATA_COLOR, linewidth=1.8)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.annotate(
        "", xy=(position - fwhm / 2, 0.5), xytext=(position + fwhm / 2, 0.5),
        arrowprops=dict(arrowstyle="<->", color=_FIT_COLOR),
    )
    ax.text(position, 0.54, "FWHM", color=_FIT_COLOR, ha="center", fontsize=9)

    ax.annotate(
        "", xy=(position, 0.03), xytext=(position + sigma, 0.03),
        arrowprops=dict(arrowstyle="<->", color=_BG_COLOR),
    )
    ax.text(position + sigma / 2, 0.07, "sigma", color=_BG_COLOR, ha="center", fontsize=10)

    ax.text(
        0.5, -0.16,
        "FWHM = 2 * sqrt(2 * ln2) * sigma  ~=  2.3548 * sigma",
        transform=ax.transAxes, ha="center", va="top", fontsize=10,
    )
    ax.set_xlabel("Channel")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Sigma and FWHM")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return _figure_to_png_bytes(fig)


def multiplet_figure():
    """Three peaks fit together with one shared FWHM -- two of them
    close enough to overlap, illustrating why multiplets link width
    across peaks instead of fitting each independently."""
    x = np.linspace(70.0, 160.0, 500)
    sigma = 3.2
    peaks = [(100.0, 300.0), (112.0, 180.0), (140.0, 220.0)]
    r, beta = 0.1, 6.0

    fig = Figure(figsize=(7.5, 4.5), dpi=110)
    ax = fig.add_subplot(111)

    total = np.zeros_like(x)
    for i, (position, amplitude) in enumerate(peaks):
        component = amplitude * hypermet_left_tail(x, position, sigma, r, beta)
        total += component
        ax.plot(x, component, linestyle="--", linewidth=1.0, label=f"Peak {i + 1} component")

    ax.plot(x, total, color=_FIT_COLOR, linewidth=1.8, label="Combined fit (shared FWHM)")

    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts")
    ax.set_title("Multiplet: peaks sharing one FWHM")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _figure_to_png_bytes(fig)


def integration_background_figure():
    """The same simulated peak integrated two ways side by side: with
    two background regions marked (background line drawn, net shaded
    separately from gross) and with none (gross only, no subtraction)
    -- the reference figure for "How integration computes gross,
    background, and net"."""
    x = np.linspace(60.0, 140.0, 400)
    position, sigma, amplitude = 100.0, 4.0, 400.0
    bg_slope, bg_intercept = 0.05, 15.0
    background = bg_intercept + bg_slope * x
    gross = background + amplitude * np.exp(-((x - position) ** 2) / (2 * sigma ** 2))

    rng = np.random.default_rng(1)
    gross_noisy = rng.poisson(np.clip(gross, 1, None)).astype(float)

    fit_region = (80.0, 120.0)
    left_bg = (65.0, 75.0)
    right_bg = (125.0, 135.0)
    mask = (x >= fit_region[0]) & (x <= fit_region[1])

    fig = Figure(figsize=(9.5, 4.5), dpi=110)

    ax1 = fig.add_subplot(121)
    ax1.step(x, gross_noisy, where="mid", color=_DATA_COLOR, linewidth=1.0, label="Gross (data)")
    ax1.plot(x, background, color=_BG_COLOR, linestyle="--", linewidth=1.3, label="Background")
    ax1.fill_between(
        x[mask], background[mask], gross_noisy[mask], step="mid",
        color=_FIT_COLOR, alpha=0.25, label="Net",
    )
    ax1.axvspan(*left_bg, color=_REGION_BG_COLOR, alpha=0.25)
    ax1.axvspan(*right_bg, color=_REGION_BG_COLOR, alpha=0.25)
    ax1.axvspan(*fit_region, color=_REGION_FIT_COLOR, alpha=0.10)
    ax1.set_title("With background regions")
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("Counts")
    ax1.legend(loc="upper left", fontsize=7)

    ax2 = fig.add_subplot(122, sharey=ax1)
    ax2.step(x, gross_noisy, where="mid", color=_DATA_COLOR, linewidth=1.0, label="Gross (data)")
    ax2.fill_between(
        x[mask], 0, gross_noisy[mask], step="mid",
        color=_FIT_COLOR, alpha=0.25, label="Gross area",
    )
    ax2.axvspan(*fit_region, color=_REGION_FIT_COLOR, alpha=0.10)
    ax2.set_title("Without background regions")
    ax2.set_xlabel("Channel")
    ax2.legend(loc="upper left", fontsize=7)

    fig.suptitle("Integration: with vs. without background subtraction")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _figure_to_png_bytes(fig)


def calibration_curve_figure():
    """Linear vs. quadratic calibration through the same 5 points, with
    a residuals panel -- deliberately exaggerated curvature (real
    detectors are usually far more linear than this) so the difference
    is visually obvious rather than realistically subtle."""
    channels = np.array([120.0, 340.0, 610.0, 890.0, 1150.0])
    true_energies = 0.95 * channels + 0.00035 * channels ** 2 + np.array(
        [3.0, -2.0, 4.0, -3.0, 2.0]
    )

    lin_b, lin_a = np.polyfit(channels, true_energies, 1)
    quad_c, quad_b, quad_a = np.polyfit(channels, true_energies, 2)
    linear_cal = Calibration(kind="linear", a=lin_a, b=lin_b)
    quadratic_cal = Calibration(kind="quadratic", a=quad_a, b=quad_b, c=quad_c)

    x = np.linspace(0, 1300, 300)
    fig = Figure(figsize=(7.5, 5.0), dpi=110)
    ax_curve = fig.add_subplot(211)
    ax_resid = fig.add_subplot(212, sharex=ax_curve)

    ax_curve.scatter(channels, true_energies, color="black", zorder=3, label="Calibration points")
    ax_curve.plot(x, linear_cal.apply(x), color=_DATA_COLOR, label="Linear fit")
    ax_curve.plot(x, quadratic_cal.apply(x), color=_FIT_COLOR, linestyle="--", label="Quadratic fit")
    ax_curve.set_ylabel("Energy (keV)")
    ax_curve.set_title("Calibration curve: channel -> energy")
    ax_curve.legend(loc="upper left", fontsize=8)

    lin_resid = true_energies - linear_cal.apply(channels)
    quad_resid = true_energies - quadratic_cal.apply(channels)
    ax_resid.axhline(0, color="gray", linewidth=0.8)
    ax_resid.scatter(channels, lin_resid, color=_DATA_COLOR, label="Linear residuals")
    ax_resid.scatter(channels, quad_resid, color=_FIT_COLOR, marker="x", label="Quadratic residuals")
    ax_resid.set_xlabel("Channel")
    ax_resid.set_ylabel("Residual (keV)")
    ymin, ymax = ax_resid.get_ylim()
    ax_resid.set_ylim(ymin, ymax + 0.4 * (ymax - ymin))
    ax_resid.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    return _figure_to_png_bytes(fig)


def matrix_projection_cut_figure():
    """A small schematic 2D coincidence matrix with a cut (orange) and
    background (purple) band marked on one axis, alongside the resulting
    1D projection -- the reference figure the Knowledge Database's 2D
    spectra section points back to. Same region colors as
    anatomy_of_a_fit_figure's R/B marks (_REGION_FIT_COLOR /
    _REGION_BG_COLOR), reused here for the analogous cut/background
    concept."""
    rng = np.random.default_rng(1)
    size = 60
    matrix = rng.poisson(3.0, size=(size, size)).astype(float)
    # A diagonal ridge of coincidence counts, the kind of structure a
    # real gamma-gamma matrix shows for genuinely correlated peaks.
    for i in range(size):
        j = min(size - 1, i + 5)
        matrix[i, :] += 40 * np.exp(-((np.arange(size) - j) ** 2) / 8.0)

    fig = Figure(figsize=(8.0, 4.2), dpi=110)
    ax_matrix = fig.add_subplot(121)
    ax_matrix.imshow(matrix, origin="lower", cmap="viridis", aspect="auto")
    cut_lo, cut_hi = 20, 30
    bg_lo, bg_hi = 40, 46
    ax_matrix.axvspan(cut_lo, cut_hi, color=_REGION_FIT_COLOR, alpha=0.35)
    ax_matrix.axvspan(bg_lo, bg_hi, color=_REGION_BG_COLOR, alpha=0.35)
    ax_matrix.set_xlabel("X channel")
    ax_matrix.set_ylabel("Y channel")
    ax_matrix.set_title("2D matrix")

    projection = matrix.sum(axis=0)
    ax_proj = fig.add_subplot(122)
    ax_proj.step(np.arange(size), projection, where="mid", color=_DATA_COLOR, linewidth=1.0)
    ax_proj.axvspan(cut_lo, cut_hi, color=_REGION_FIT_COLOR, alpha=0.25)
    ax_proj.axvspan(bg_lo, bg_hi, color=_REGION_BG_COLOR, alpha=0.25)
    ax_proj.set_xlabel("X channel")
    ax_proj.set_ylabel("Counts")
    ax_proj.set_title("X projection")

    fig.text(
        0.5, 0.01,
        "orange = cut region (the gate)   purple = background region -- "
        "summing along Y gives the projection shown on the right",
        ha="center", va="bottom", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _figure_to_png_bytes(fig)
