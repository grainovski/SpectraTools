"""The two efficiency output files.

Both carry BOTH curves so the file is a complete record of the calibration
and the two models can be compared after the fact. Neither carries a channel
column: the energy calibration is available wherever these are read.
"""

import numpy as np

MODEL_NAMES = {"kfr": "KFR", "rw": "Radware"}

#: Above this many energies the curves are evaluated on this many evenly
#: spaced knots and interpolated. See _both for the measurements behind it.
FILE_KNOTS = 2048


def _header_lines(result, extra=()):
    fit = result.fit
    lines = [
        "SpectraTools relative efficiency",
        "",
        "applied model      : %s" % MODEL_NAMES[result.model],
        "normalisation      : %.6g   (1 / peak of the applied curve)" %
        result.normalisation,
        "fitted energy range: %.4f .. %.4f keV" % (fit.E.min(), fit.E.max()),
        "peaks              : %d" % len(fit.E),
        "KFR  params        : %s" % ", ".join("%.12g" % v
                                              for v in fit.kfr_params),
        "KFR  chi2/ndf      : %.6f / %d   Birge %.4f   RMS %.6g" %
        (fit.kfr_chi2, fit.kfr_ndf, fit.kfr_birge, fit.kfr_rms),
    ]
    if fit.rw_params is None:
        lines.append("Radware            : did not converge")
    else:
        lines += [
            # The divisor is part of the answer: the parameters describe
            # eff*rw_scale, so the curve is f_radware_5p(E, *params) over
            # it. Writing the parameters without it would be writing a
            # curve a reader cannot reconstruct.
            "Radware params     : %s" % ", ".join("%.12g" % v
                                                  for v in fit.rw_params),
            "Radware scale      : %.12g" % fit.rw_scale,
            "Radware chi2/ndf   : %.6f / %d   Birge %.4f   RMS %.6g" %
            (fit.rw_chi2, fit.rw_ndf, fit.rw_birge, fit.rw_rms),
        ]
    if result.mc is not None:
        lines.append(
            "Monte Carlo        : KFR %d accepted, Radware %d accepted, "
            "%d samples rejected" %
            (result.mc.kfr_accepted, result.mc.rw_accepted,
             result.mc.rejected))
        # Stated because the file is read on its own, away from the Help.
        # Without it a reader sees "Birge 0.6398" beside the deff columns
        # and reasonably assumes the band was scaled by it. It was not:
        # the scaling is inflate-only.
        applied = (fit.kfr_birge if result.model == "kfr" else fit.rw_birge)
        lines.append(
            "band scaling       : x%.4f   (Birge %.4f, inflate-only: a "
            "ratio below 1 never narrows the band)"
            % (max(1.0, applied), applied))
    if result.calibration is not None:
        lines.append("energy calibration : %s" % (result.calibration,))
    if result.source:
        lines.append("source             : %s" % result.source)
    lines.extend(extra)
    return "".join("# %s\n" % line if line else "#\n" for line in lines)


def _both(result, energies):
    """Normalised value and 1-sigma half-width for each model at `energies`.

    Two things here are about cost, and both were measured rather than
    guessed. Written the obvious way -- curve() then band() per model,
    exactly at every channel -- writing a 16,384-bin file took **91.6 s** on
    a full 10,000-sample Monte Carlo. That is not a slow save; it is
    something a user kills believing it has hung.

    1. The MC mean is computed ONCE per model and handed to the band as its
       centre. Otherwise band() recomputes it, doubling the most expensive
       step.
    2. Above FILE_KNOTS energies the curves are evaluated on that many knots
       and interpolated. They are smooth analytic functions of energy, so
       the price is a maximum relative error of 1.9e-6 against an MC
       uncertainty on the very same numbers of 5.7e-3 -- roughly 3000 times
       smaller than what the value already does not know about itself.
       Writing the exact figure instead would be false precision bought with
       a minute and a half of the user's time.

    Together: 91.6 s -> 4.6 s. The header records that interpolation was
    used and its error bound, so the file does not quietly imply more
    precision than it has.
    """
    energies = np.asarray(energies, dtype=float)
    interpolated = len(energies) > FILE_KNOTS
    knots = (np.linspace(energies[0], energies[-1], FILE_KNOTS)
             if interpolated else energies)

    out = []
    for model in ("kfr", "rw"):
        if model == "rw" and result.fit.rw_params is None:
            nan = np.full(len(energies), float("nan"))
            out += [nan, nan]
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            centre = result._mc_mean_raw(knots, model)
            value = centre * result.normalisation
            lo, hi = result.band(knots, model, centre=centre)
        half = (hi - lo) / 2.0
        if interpolated:
            value = np.interp(energies, knots, value)
            half = np.interp(energies, knots, half)
        out += [value, half]
    return out


def write_per_peak(path, result, energy_errors):
    """One row per calibration peak: E dE eff_kfr deff_kfr eff_rw deff_rw."""
    E = result.fit.E
    dE = np.asarray(energy_errors, dtype=float)
    if len(dE) != len(E):
        raise ValueError(
            "got %d energy errors for %d peaks" % (len(dE), len(E)))
    k, dk, r, dr = _both(result, E)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_header_lines(
            result, ["", "columns: E  dE  eff_kfr  deff_kfr  eff_rw  deff_rw"]))
        for row in zip(E, dE, k, dk, r, dr):
            fh.write("%14.6f %12.6f %14.8g %14.8g %14.8g %14.8g\n" % row)


def write_per_bin(path, result, calibration, channels):
    """One row per channel: E eff_kfr deff_kfr eff_rw deff_rw.

    No dE column -- a sampled curve point has no energy uncertainty, and a
    zero column would be a meaningless number written to disk.
    """
    energies = np.asarray(
        calibration.apply(np.arange(int(channels), dtype=float)), dtype=float)
    k, dk, r, dr = _both(result, energies)
    notes = ["", "one row per channel, evaluated through the energy "
             "calibration above"]
    if len(energies) > FILE_KNOTS:
        notes.append(
            "curves interpolated from %d knots; max relative error ~2e-6, "
            "far below the Monte Carlo uncertainty in the deff columns"
            % FILE_KNOTS)
    notes.append("columns: E  eff_kfr  deff_kfr  eff_rw  deff_rw")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_header_lines(result, notes))
        for row in zip(energies, k, dk, r, dr):
            fh.write("%14.6f %14.8g %14.8g %14.8g %14.8g\n" % row)
