"""The `.sou` files ship with the app from 5.2.3.

These tests guard the part that is easy to get wrong: the packaging and
the code have to agree on one directory name, across two build paths that
are configured separately (the Windows spec's `datas`, and the --add-data
flag in packaging/linux/build.sh). A change to one of those that misses
the other produces a build that works on the machine it was made on.
"""

import os
import re

import bundled_sources


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_ten_sources_are_found_from_a_checkout():
    directory = bundled_sources.bundled_sources_dir()
    assert directory is not None
    assert os.path.isdir(directory)
    names = [os.path.basename(p) for p in bundled_sources.bundled_source_files()]
    assert names == [
        "am241.sou", "am243.sou", "ba133.sou", "co56.sou", "eu152.sou",
        "na24.sou", "ra226.sou", "se75.sou", "ta182.sou", "y88.sou",
    ]


def test_every_bundled_file_actually_parses():
    """A packaged file that the reader rejects would ship a dialog that
    fails on its own default contents."""
    from sou_io import load_sou

    for path in bundled_sources.bundled_source_files():
        lines = load_sou(path)
        assert lines, path


def test_a_frozen_build_looks_beside_the_unpacked_bundle(monkeypatch, tmp_path):
    """PyInstaller sets sys._MEIPASS; the directory must be resolved from
    there rather than from the source tree the app was built from."""
    bundle = tmp_path / "bundle"
    (bundle / "sources").mkdir(parents=True)
    (bundle / "sources" / "eu152.sou").write_text("121.7817 0.0003 10000.0 56.1\n")

    monkeypatch.setattr(bundled_sources.sys, "_MEIPASS", str(bundle), raising=False)

    assert bundled_sources.bundled_sources_dir() == str(bundle / "sources")
    assert [os.path.basename(p) for p in bundled_sources.bundled_source_files()] == [
        "eu152.sou"
    ]


def test_a_missing_directory_is_none_not_an_error(monkeypatch, tmp_path):
    """Running from a checkout without the directory must still work."""
    monkeypatch.setattr(bundled_sources.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert bundled_sources.bundled_sources_dir() is None
    assert bundled_sources.bundled_source_files() == []


# --- the packaging must carry what the code expects --------------------


def test_the_windows_spec_packages_the_directory():
    spec = open(os.path.join(REPO, "packaging", "windows", "SpectraTools.spec"),
                encoding="utf-8").read()
    assert "_sources" in spec, "the spec no longer builds a sources datas entry"
    assert re.search(r"datas\s*=\s*datas_awk\s*\+\s*_sources", spec), (
        "the sources entry is built but not passed to Analysis(datas=...)"
    )


def test_the_linux_build_stages_and_packages_the_directory():
    build = open(os.path.join(REPO, "packaging", "linux", "build.sh"),
                 encoding="utf-8").read()
    # Linux does not use the spec file -- it calls PyInstaller directly in a
    # staging dir that receives only the root *.py, so the directory has to
    # be copied in AND declared.
    assert re.search(r'cp\s+-a\s+"\$ROOT_DIR/sources"', build), (
        "build.sh does not copy sources/ into its staging directory"
    )
    assert "--add-data" in build and "sources:sources" in build, (
        "build.sh does not pass sources/ to PyInstaller"
    )


def test_both_build_paths_use_the_name_the_code_looks_for():
    """The one name, in all three places."""
    name = bundled_sources.DIRECTORY_NAME
    spec = open(os.path.join(REPO, "packaging", "windows", "SpectraTools.spec"),
                encoding="utf-8").read()
    build = open(os.path.join(REPO, "packaging", "linux", "build.sh"),
                 encoding="utf-8").read()
    assert "'%s'" % name in spec or '"%s"' % name in spec
    assert "%s:%s" % (name, name) in build


# --- the dialogs must actually reach the bundled files -----------------
#
# Packaging a directory that no dialog ever opens on would ship the data
# and leave the feature exactly as unusable as before, which is the whole
# reason for the change. These drive the real "Load source..." handler and
# read the directory it hands QFileDialog.


class _RecordingDialog:
    """Captures the directory argument QFileDialog is opened with."""

    def __init__(self):
        self.directory = None

    def __call__(self, parent, caption, directory, filters):
        self.directory = directory
        return "", ""


class _Settings:
    def __init__(self, source_folder=""):
        self.source_folder = source_folder

    def last_folder(self):
        return "C:/some/spectra/folder"

    def set_last_folder(self, folder):
        pass

    def last_source_folder(self):
        return self.source_folder

    def set_last_source_folder(self, folder):
        self.source_folder = folder


def _open_directory_for(dialog, module, monkeypatch):
    recorder = _RecordingDialog()
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", recorder)
    dialog._on_load_source_clicked()
    return recorder.directory


def test_auto_calibrate_dialog_opens_on_the_bundled_sources(qapp, monkeypatch):
    import auto_calibrate_dialog as mod
    import numpy as np

    counts = np.random.default_rng(1).poisson(50.0, 512).astype(float)
    dialog = mod.AutoCalibrateDialog(None, counts, settings=_Settings(""))
    assert _open_directory_for(dialog, mod, monkeypatch) == \
        bundled_sources.bundled_sources_dir()


def test_energy_assign_dialog_opens_on_the_bundled_sources(qapp, monkeypatch):
    import energy_assign_dialog as mod

    dialog = mod.EnergyAssignDialog(None, [], settings=_Settings(""))
    assert _open_directory_for(dialog, mod, monkeypatch) == \
        bundled_sources.bundled_sources_dir()


def test_a_remembered_source_folder_wins_over_the_bundled_one(qapp, monkeypatch):
    """The default is a starting point, not an override: once the user has
    loaded a source of their own, the dialog goes back there."""
    import energy_assign_dialog as mod

    dialog = mod.EnergyAssignDialog(None, [], settings=_Settings("D:/my/nuclides"))
    assert _open_directory_for(dialog, mod, monkeypatch) == "D:/my/nuclides"


def test_the_spectrum_folder_is_not_used_for_sources(qapp, monkeypatch):
    """Control for the above: last_folder() is a plausible-looking value
    that must NOT be what the source dialog opens on, or the bundled files
    would be unreachable on any install that has opened a spectrum."""
    import energy_assign_dialog as mod

    dialog = mod.EnergyAssignDialog(None, [], settings=_Settings(""))
    assert _open_directory_for(dialog, mod, monkeypatch) != "C:/some/spectra/folder"
