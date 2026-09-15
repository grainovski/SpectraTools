"""The suite must not write into the user's real settings.

`settings.Settings` is `QSettings(ORG_NAME, APP_NAME)` with no injection
point, so before conftest redirected it every test that opened a dialog
wrote to the developer's own per-user store -- the registry on Windows.

That is not hypothetical. It put pytest temp paths into `last_folder`,
`last_source_folder` and `recent_files` on this machine, and the stale
`last_source_folder` then sent the *installed* application's "Load
source..." dialog into a pytest directory instead of the `.sou` files
shipped with it, breaking a hand-verification of v5.2.3 that existed to
catch exactly what the tests cannot.

These fail if the redirect is ever lost.
"""

import os

from PySide6.QtCore import QSettings

from settings import APP_NAME, ORG_NAME, Settings


_NATIVE_PREFIX = "\\HKEY"


def _backing_file():
    """Where the object the application actually uses stores its values.

    Deliberately reached through Settings() rather than by constructing a
    QSettings here: conftest replaces the constructor *settings.py* calls,
    so a QSettings built directly in this file is NOT redirected and would
    make the test pass or fail for the wrong reason.
    """
    return Settings()._settings.fileName()


def _native_store():
    """The real per-user store. Reading does not create it."""
    return QSettings(
        QSettings.Format.NativeFormat, QSettings.Scope.UserScope,
        ORG_NAME, APP_NAME,
    )


def test_settings_do_not_resolve_to_the_native_user_store():
    where = _backing_file()
    assert "spectratools-qsettings-" in where, where
    assert not where.upper().startswith(_NATIVE_PREFIX), where


def test_an_unredirected_qsettings_would_have_gone_to_the_native_store():
    """The control. Without it the test above could pass on a machine
    where the native store simply happens to look like a temp path -- and
    it also pins the exact mistake the helper above avoids.
    """
    direct = QSettings(ORG_NAME, APP_NAME).fileName()
    assert "spectratools-qsettings-" not in direct, direct


def test_a_written_value_lands_in_the_redirected_file(tmp_path):
    settings = Settings()
    settings.set_last_folder(str(tmp_path))
    assert settings.last_folder() == str(tmp_path)
    assert os.path.exists(_backing_file())


def test_the_write_does_not_reach_the_real_user_store():
    sentinel = "spectratools-isolation-sentinel"
    settings = Settings()
    settings.set_last_source_folder(sentinel)
    assert settings.last_source_folder() == sentinel
    assert _native_store().value("last_source_folder", "") != sentinel


def test_recent_files_are_isolated_too():
    """recent_files was the first casualty: it had collected pytest ROOT
    paths long before last_source_folder existed."""
    marker = "C:/nowhere/isolation-check.spe"
    settings = Settings()
    settings.add_recent_file(marker)
    assert marker in settings.recent_files()

    real = _native_store().value("recent_files", []) or []
    if isinstance(real, str):
        real = [real]
    assert marker not in real
