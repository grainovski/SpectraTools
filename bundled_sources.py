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


def _sources_path():
    """The packaged directory if it exists at all, without asking what is
    in it. Separate from bundled_sources_dir so that one can consult
    bundled_source_files without the two calling each other."""
    path = os.path.join(_base_dir(), DIRECTORY_NAME)
    return path if os.path.isdir(path) else None


def bundled_source_files():
    """The packaged `.sou` files, sorted by name. Empty when none ship."""
    directory = _sources_path()
    if directory is None:
        return []
    try:
        names = os.listdir(directory)
    except OSError:
        # Readable a moment ago when isdir answered; gone or unreadable
        # now. An empty answer is the same outcome as shipping none.
        return []
    return sorted(
        os.path.join(directory, name)
        for name in names
        if name.lower().endswith(".sou")
    )


def bundled_sources_dir():
    """Where a `.sou` file dialog should start, or None when there is no
    better answer than wherever the dialog would go on its own.

    None is a normal answer, not a fault: a developer running from a
    checkout that never fetched the directory still gets a working file
    dialog, just one that does not start anywhere in particular.

    A directory that exists but holds no `.sou` counts as absent. Opening
    the dialog on an empty folder is worse than not steering it at all --
    the user is shown nothing and has to navigate out of it.
    """
    return _sources_path() if bundled_source_files() else None
