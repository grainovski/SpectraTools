"""Locates the `.sou` source-description files shipped with the app.

The calibration features need a `.sou` file naming a nuclide's known gamma
lines, and until 5.2.3 the packages carried none: the feature was present
but arrived without the data it needs, so a new user could not calibrate
until they found source files of their own.

The ten in the repository's `sources/` are now packaged, which means the
same directory has to be found in two quite different layouts:

- **From source** every module sits at the repository root, so `sources/`
  is a sibling of this file.
- **Frozen** PyInstaller unpacks `datas` under `sys._MEIPASS`, which for a
  onedir build is the `_internal/` folder beside the executable.

Both are just "a `sources/` directory next to where the code lives", which
is what `_base_dir` resolves. Nothing here parses a `.sou`; that is
`sou_io`.
"""

import os
import sys

#: Name of the packaged directory, as used by the PyInstaller spec's
#: `datas` entry AND by packaging/linux/build.sh's --add-data flag. The two
#: build paths are configured separately -- Windows from the spec file,
#: Linux from the command line -- so this name appearing in three places is
#: a fact worth stating rather than a coincidence.
DIRECTORY_NAME = "sources"


def _base_dir():
    """Where the running code lives: the unpacked bundle, or this file's
    directory when running from a checkout."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return bundle
    return os.path.dirname(os.path.abspath(__file__))


def bundled_sources_dir():
    """Absolute path of the packaged `sources/` directory, or None when it
    is absent.

    None is a normal answer, not a fault: a developer running from a
    checkout that never fetched the directory still gets a working file
    dialog, just one that does not start anywhere in particular.
    """
    path = os.path.join(_base_dir(), DIRECTORY_NAME)
    return path if os.path.isdir(path) else None


def bundled_source_files():
    """The packaged `.sou` files, sorted by name. Empty when none ship."""
    directory = bundled_sources_dir()
    if directory is None:
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".sou")
    )
