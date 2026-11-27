"""Reading a 1-D spectrum file of any format the app opens, by extension.

One dispatch for every place that reads a spectrum from disk: File > Open
(MainWindow._try_load_spectrum) and the gain-matching step's calibration
spectra (gain_match_dialog). The gain dialog used to carry its own copy of
this dispatch, which would have drifted the first time a format was added
to one and not the other.
"""

import os

from histogram_io import load_histogram
from lzs_io import load_lzs
from n42_io import load_n42
from spe_io import load_spe
from spk_io import load_spk

#: The file-dialog filter for the formats load_spectrum_file reads.
SPECTRUM_FILTER = ("Spectrum files (*.txt *.spe *.spk *.n42 *.lzs);;"
                   "Text files (*.txt);;SPE files (*.spe);;SPK files (*.spk);;"
                   "N42 files (*.n42);;LZS files (*.lzs);;All files (*)")


def load_spectrum_file(path):
    """(counts, calibration) of the 1-D spectrum in `path`.

    The format is chosen by extension, case-insensitively: .spe, .spk,
    .n42, .lzs, and anything else as a plain-text histogram. `calibration`
    is the energy calibration the file carries (.n42 and .lzs can), else
    None. Raises ParseError for a file that is not a spectrum and OSError
    when it cannot be read.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".spe":
        return load_spe(path), None
    if ext == ".spk":
        return load_spk(path), None
    if ext == ".n42":
        return load_n42(path)
    if ext == ".lzs":
        return load_lzs(path)
    return load_histogram(path), None
