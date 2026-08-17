import os

# Must be set before any PySide6/Qt import, including ones triggered by
# importing project modules below -- lets the Qt-dependent tests in this
# suite run headlessly without requiring the environment variable to be
# set externally every time.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _isolated_matrix_cache(tmp_path_factory, monkeypatch):
    """Point the on-disk matrix cache at a scratch directory for every
    test.

    Two things go wrong without this, and both bit during development.
    A test that patches load_mtx to observe the decode is silently
    satisfied by a cache entry some earlier test wrote, so it passes or
    fails depending on the order tests happen to run in. And a suite run
    leaves real cache files -- hundreds of megabytes, for the larger
    fixtures -- in the developer's own profile.

    Autouse rather than opt-in: any test that opens a matrix touches the
    cache, whether or not it mentions caching, so opting in would mean
    remembering on every future test.
    """
    import matrix_cache

    monkeypatch.setenv(
        matrix_cache.CACHE_DIR_ENV,
        str(tmp_path_factory.mktemp("matrix-cache")),
    )
