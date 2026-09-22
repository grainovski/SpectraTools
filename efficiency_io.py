"""The two efficiency output files, and reading them back.

Both carry BOTH curves so the file is a complete record of the calibration
and the two models can be compared after the fact. Neither carries a channel
column: the energy calibration is available wherever these are read.
"""

import os
import re

import numpy as np

MODEL_NAMES = {"krf": "KRF", "rw": "Radware"}

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
        "KRF  params        : %s" % ", ".join("%.12g" % v
                                              for v in fit.krf_params),
        "KRF  chi2/ndf      : %.6f / %d   Birge %.4f   RMS %.6g" %
        (fit.krf_chi2, fit.krf_ndf, fit.krf_birge, fit.krf_rms),
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
            "Monte Carlo        : KRF %d accepted, Radware %d accepted, "
            "%d samples rejected" %
            (result.mc.krf_accepted, result.mc.rw_accepted,
             result.mc.rejected))
        # Stated because the file is read on its own, away from the Help.
        # Without it a reader sees "Birge 0.6398" beside the deff columns
        # and reasonably assumes the band was scaled by it. It was not:
        # the scaling is inflate-only.
        applied = (fit.krf_birge if result.model == "krf" else fit.rw_birge)
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
       and interpolated, EXCEPT in any knot interval where interpolation is
       checked and found to miss the exact curve -- those are evaluated
       exactly at every channel. See _untrusted_intervals.

    The original version assumed both curves were smooth and claimed a
    maximum error of 1.9e-6 without checking. Measured over every channel in
    2026-09: 4e-4 for KRF, and 527% for Radware on the demo2 reference set,
    whose Monte Carlo mean falls off a cliff at the lowest calibration line.
    Checked and refined, both are within 1.1e-5 at every channel, for
    6.1 s against 4.6 s -- still far from the 91.6 s of evaluating every
    channel exactly. The header states the bound that is now guaranteed.
    """
    energies = np.asarray(energies, dtype=float)
    interpolated = len(energies) > FILE_KNOTS
    knots = (np.linspace(energies[0], energies[-1], FILE_KNOTS)
             if interpolated else energies)

    out = []
    for model in ("krf", "rw"):
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
            v_all = np.interp(energies, knots, value)
            h_all = np.interp(energies, knots, half)
            bad = _untrusted_intervals(result, model, knots, value)
            if bad.any():
                interval = np.clip(
                    np.searchsorted(knots, energies, side="right") - 1,
                    0, len(knots) - 2)
                exact = bad[interval]
                if exact.any():
                    e_sel = energies[exact]
                    with np.errstate(over="ignore", invalid="ignore",
                                     divide="ignore"):
                        c_sel = result._mc_mean_raw(e_sel, model)
                        lo_s, hi_s = result.band(e_sel, model, centre=c_sel)
                    v_all[exact] = c_sel * result.normalisation
                    h_all[exact] = (hi_s - lo_s) / 2.0
            value, half = v_all, h_all
        out += [value, half]
    return out


#: Relative error at which a knot interval stops being trusted.
REFINE_TOL = 1e-5


def _untrusted_intervals(result, model, knots, value):
    """Knot intervals where linear interpolation misses the exact curve.

    The knots were introduced on the grounds that both curves are smooth
    functions of energy. KRF is. The Radware MONTE CARLO MEAN is not always:
    on the demo2 reference set it falls from 1.0 at 122 keV to 0.03 at
    119 keV, a cliff right at the lowest calibration line where Radware's
    two branches trade places. Interpolating straight across that wrote an
    efficiency 18% wrong into the saved file, under a header claiming the
    error was 2e-6.

    So every interval is checked against the exact curve at its thirds --
    two points, because a single midpoint probe is blind to a step sitting
    exactly halfway -- and one that misses by more than REFINE_TOL, or is
    finite where the exact curve is not or vice versa, is flagged for exact
    evaluation at every channel it covers.
    """
    a, b = knots[:-1], knots[1:]
    probes = np.concatenate([a + (b - a) / 3.0, a + 2.0 * (b - a) / 3.0])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exact = result._mc_mean_raw(probes, model) * result.normalisation
        guess = np.interp(probes, knots, value)
        finite = np.isfinite(exact) & np.isfinite(guess)
        rel = np.where(finite, np.abs(guess - exact)
                       / np.maximum(np.abs(exact), 1e-300), 0.0)
    wrong = (finite & (rel > REFINE_TOL)) | (np.isfinite(exact)
                                              != np.isfinite(guess))
    return wrong[:len(a)] | wrong[len(a):]


def write_per_peak(path, result, energy_errors):
    """One row per calibration peak: E dE eff_krf deff_krf eff_rw deff_rw."""
    E = result.fit.E
    dE = np.asarray(energy_errors, dtype=float)
    if len(dE) != len(E):
        raise ValueError(
            "got %d energy errors for %d peaks" % (len(dE), len(E)))
    k, dk, r, dr = _both(result, E)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_header_lines(
            result, ["", "columns: E  dE  eff_krf  deff_krf  eff_rw  deff_rw"]))
        for row in zip(E, dE, k, dk, r, dr):
            fh.write("%14.6f %12.6f %14.8g %14.8g %14.8g %14.8g\n" % row)


def write_per_bin(path, result, calibration, channels):
    """One row per channel: E eff_krf deff_krf eff_rw deff_rw.

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
            "curves interpolated from %d knots and evaluated exactly "
            "wherever interpolation missed by more than %.0e; far below the "
            "Monte Carlo uncertainty in the deff columns"
            % (FILE_KNOTS, REFINE_TOL))
    notes.append("columns: E  eff_krf  deff_krf  eff_rw  deff_rw")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_header_lines(result, notes))
        for row in zip(energies, k, dk, r, dr):
            fh.write("%14.6f %14.8g %14.8g %14.8g %14.8g\n" % row)


# --- Reading a saved efficiency back ----------------------------------------
#
# The files above are the only place a finished calibration is saved, and
# they record everything needed to use it again: both models, the applied
# one, the normalisation -- and, in the header, the exact ENERGY calibration
# the efficiency was made under. They are also the only place any energy
# calibration the program produced is written down, so reading them back
# restores both.

#: First line of every file written above. Anything without it is not ours.
FILE_MARKER = "SpectraTools relative efficiency"


class SavedEfficiencyError(Exception):
    """A file that cannot be read back as a saved efficiency, with a reason
    written for the person who picked it."""


#: The energy-calibration header line is the dataclass repr, e.g.
#:   Calibration(kind='quadratic', a=-0.35, b=0.5, c=2e-08, coefficient_errors=())
#: The fields are plain floats today, so the repr is exact and round-trips.
#: An optional np.float64(...) wrapper is accepted as well, because NumPy 2
#: prints its own scalars that way and one reaching the dataclass would
#: otherwise make a correct file unreadable.
_NUM = r"(?:np\.float64\()?([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\)?"
_CAL_RE = re.compile(
    r"Calibration\(kind='(linear|quadratic)',\s*a=" + _NUM
    + r",\s*b=" + _NUM + r",\s*c=" + _NUM)


def _canonical(text):
    """Header keys and column names with the pre-6.1.1 spelling mapped.

    The four-parameter model was labelled KFR up to 6.1.0 and KRF since. A
    file saved by an older version is exactly what this reader exists for,
    so both spellings must read identically.
    """
    return text.replace("KFR", "KRF").replace("kfr", "krf")


def _header(path):
    """({key: value} from a file's '#' lines, raw header text)."""
    if not isinstance(path, (str, os.PathLike)):
        raise SavedEfficiencyError("No file was given (got %r)." % (path,))
    fields, raw = {}, []
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                raw.append(line)
                body = line[1:].strip()
                if ":" not in body:
                    continue
                key, _, value = body.partition(":")
                fields[" ".join(_canonical(key).split())] = value.strip()
    except OSError as exc:
        raise SavedEfficiencyError(
            "Could not read %s: %s" % (path, exc.strerror or exc)) from exc
    except UnicodeDecodeError as exc:
        raise SavedEfficiencyError(
            "%s is not a text file." % os.path.basename(path)) from exc
    if not raw or FILE_MARKER not in raw[0]:
        raise SavedEfficiencyError(
            "%s is not an efficiency saved by SpectraTools. Load the "
            "_bins or _peaks file written by the efficiency window's Save "
            "button." % os.path.basename(path))
    return fields, "".join(raw)


def parse_energy_calibration(header_text):
    """The energy calibration recorded in a saved efficiency, or None if the
    file does not carry one.

    Raises SavedEfficiencyError if the line is there but cannot be turned
    into a valid calibration, rather than quietly dropping it.
    """
    from calibration import Calibration, CalibrationError

    line = next((l for l in header_text.splitlines()
                 if "energy calibration" in l), None)
    if line is None:
        return None
    match = _CAL_RE.search(line)
    if match is None:
        raise SavedEfficiencyError(
            "The energy calibration in this file could not be read:\n%s"
            % line.strip())
    kind, a, b, c = match.groups()
    try:
        return Calibration(kind=kind, a=float(a), b=float(b), c=float(c))
    except (ValueError, CalibrationError) as exc:
        raise SavedEfficiencyError(
            "The energy calibration in this file is not valid: %s" % exc
        ) from exc


def read_energy_calibration(path):
    """Just the energy calibration from a saved efficiency file."""
    _fields, raw = _header(path)
    calibration = parse_energy_calibration(raw)
    if calibration is None:
        raise SavedEfficiencyError(
            "%s does not record an energy calibration."
            % os.path.basename(path))
    return calibration


def is_saved_efficiency(path):
    """Whether `path` starts like a file written by this module."""
    # A bool is an int, and open() treats an int as a file descriptor:
    # open(False) is standard input. A Qt clicked() signal wired straight to
    # a slot with an optional path argument delivers exactly that False.
    if not isinstance(path, (str, os.PathLike)):
        return False
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return FILE_MARKER in fh.readline()
    except (OSError, UnicodeDecodeError):
        return False


def _bins_path_for(path):
    """The _bins file that belongs with `path`.

    The CURVE has to come from the _bins file. The _peaks file only holds
    the curve at the calibration lines, and rebuilding it from the saved
    parameters instead is not a faithful substitute: what the program
    applies is the Monte Carlo MEAN, and for Radware the best fit differs
    from that mean by up to 10% inside the fitted range (measured on the
    demo2 reference set), because the mean averages across a flat direction
    that the best fit sits at one point on. So a _peaks file is resolved to
    the _bins file saved beside it.
    """
    stem, ext = os.path.splitext(path)
    if stem.endswith("_peaks"):
        return stem[:-len("_peaks")] + "_bins" + ext
    return path


class SavedEfficiency:
    """An efficiency read back from disk, usable wherever a fitted one is.

    Carries what applying needs -- `model`, `curve(energies, model)` and
    `calibration` -- so efficiency_apply and the main window treat it
    exactly like a fresh result. It has no Monte Carlo samples and no
    measured points, because the files do not hold them, so it can be
    applied but not refitted or re-plotted as a fit.
    """

    def __init__(self, path, fields, raw_header, table, columns):
        from efficiency_apply import ZEROED_BELOW_KEV

        self.path = path
        self.source = fields.get("source")
        self.calibration = parse_energy_calibration(raw_header)
        self._fields = fields
        self._tables = {}
        E = table[:, columns.index("E")]
        for model in ("krf", "rw"):
            name = "eff_" + model
            if name not in columns:
                continue
            eff = table[:, columns.index(name)]
            # Only rows that can be interpolated honestly. Below
            # ZEROED_BELOW_KEV the saved curve is an extrapolation that can
            # reach 1e+195 -- never applied, but the neighbour of the first
            # valid row, so interpolating across it would drag that into
            # the lowest applied bins.
            keep = (np.isfinite(E) & np.isfinite(eff) & (eff > 0.0)
                    & (E >= ZEROED_BELOW_KEV))
            if keep.sum() < 2:
                continue
            order = np.argsort(E[keep], kind="stable")
            e_sorted, eff_sorted = E[keep][order], eff[keep][order]
            # A quadratic calibration that folds inside the spectrum can
            # repeat an energy, and np.interp needs it increasing.
            e_sorted, first = np.unique(e_sorted, return_index=True)
            self._tables[model] = (e_sorted, eff_sorted[first])
        if not self._tables:
            raise SavedEfficiencyError(
                "%s holds no usable efficiency curve."
                % os.path.basename(path))
        applied = _canonical(fields.get("applied model", "KRF")).lower()
        self._model = "rw" if applied.startswith("radware") else "krf"
        if self._model not in self._tables:
            self._model = next(iter(self._tables))

    @property
    def available_models(self):
        return tuple(m for m in ("krf", "rw") if m in self._tables)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if value not in self._tables:
            raise ValueError("this file has no %r curve" % (value,))
        self._model = value

    @property
    def energy_range(self):
        """(lowest, highest) energy the selected curve is available for."""
        e = self._tables[self._model][0]
        return float(e[0]), float(e[-1])

    def field(self, key):
        """A header value by its label, e.g. 'fitted energy range'."""
        return self._fields.get(key)

    def statistic(self, model, name):
        """A header figure such as 'chi2/ndf' for `model`, as text."""
        label = "KRF" if model == "krf" else "Radware"
        return self._fields.get("%s %s" % (label, name))

    def curve(self, energies, model=None):
        """The saved (normalised, Monte Carlo mean) efficiency at `energies`.

        NaN outside the energies the file covers -- the caller already
        zeroes a bin whose efficiency is not finite, which is the honest
        outcome for an energy this curve was never evaluated at.
        """
        e_tab, eff_tab = self._tables[model or self._model]
        e = np.asarray(energies, dtype=float)
        out = np.interp(e, e_tab, eff_tab, left=np.nan, right=np.nan)
        return np.where(np.isfinite(e), out, np.nan)


def read_saved_efficiency(path):
    """Read a saved efficiency: a _bins file, or a _peaks file whose _bins
    file sits beside it."""
    _header(path)                            # rejects non-efficiency files
    bins = _bins_path_for(path)
    if bins != path and not os.path.isfile(bins):
        raise SavedEfficiencyError(
            "%s holds the efficiency only at the calibration lines. The "
            "curve itself is in %s, which is not in the same folder -- load "
            "that file instead." % (os.path.basename(path),
                                    os.path.basename(bins)))
    fields, raw = _header(bins)
    columns = _canonical(fields.get("columns", "")).split()
    if not columns:
        raise SavedEfficiencyError(
            "%s has no column line; it may be truncated."
            % os.path.basename(bins))
    if "dE" in columns:
        raise SavedEfficiencyError(
            "%s holds the efficiency only at the calibration lines. Load "
            "the _bins file saved beside it." % os.path.basename(bins))
    try:
        table = np.loadtxt(bins, comments="#", ndmin=2)
    except ValueError as exc:
        raise SavedEfficiencyError(
            "The table in %s could not be read: %s"
            % (os.path.basename(bins), exc)) from exc
    if table.shape[1] != len(columns):
        raise SavedEfficiencyError(
            "%s has %d columns of numbers but its header names %d."
            % (os.path.basename(bins), table.shape[1], len(columns)))
    return SavedEfficiency(bins, fields, raw, table, columns)
