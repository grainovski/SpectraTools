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
