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


def close_all_windows():
    """Close every top-level window, releasing what it holds.

    One QApplication serves the whole session, so a window a test leaves
    open stays open for every test that follows. That is not a leak in
    the app -- MatrixPanel.closeEvent drops its decoded matrix and
    MainWindow.closeEvent closes its panels first -- but those handlers
    only run if something closes the window, and tests overwhelmingly do
    not: test_matrix_panel.py alone opens 75 panels and closes none.

    Each abandoned panel keeps an 8192x8192 matrix (512 MB) reachable
    through the Qt signal closures made in its __init__, which capture
    `self`. Measured before this existed: a four-file Qt run climbed from
    1.6 GB to 3.4 GB over 57 minutes, and a full run died at 52% with a
    Windows access violation inside a MainWindow constructor.

    close() rather than deleteLater() alone: it is close() that triggers
    the closeEvent handlers doing the releasing, and deleting the widget
    without closing it would skip them.
    """
    app = QApplication.instance()
    if app is None:
        return
    for widget in list(app.topLevelWidgets()):
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            # Already destroyed on the C++ side -- closing a MainWindow
            # cascades to its panels, so a widget listed a moment ago can
            # be gone by the time this loop reaches it.
            pass
    # deleteLater only takes effect when the event loop runs, and the
    # suite never starts one.
    app.processEvents()


@pytest.fixture(autouse=True)
def _close_windows_after_each_test():
    """Teardown, so a test still sees whatever it opened while it runs --
    several assert on windows they deliberately leave open."""
    yield
    close_all_windows()


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
