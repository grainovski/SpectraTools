"""Which Qt platform the app runs on under WSL (v4.1.0).

The app used to force xcb on WSL to dodge a compositor bug that rendered
the main window at 0x0. That workaround turned out to cause the dialog
flicker reported on Linux -- going through XWayland makes a file dialog
appear, vanish for about a second and come back a few seconds later. The
flicker lives below the X protocol: a trace of a dialog's whole lifetime
showed one MAP, one EXPOSE and one UNMAP, with none of the extra Expose
events a repainting compositor would produce. Running on Wayland removes
the XWayland layer and the flicker with it.

Wayland is therefore preferred now, with xcb kept as an automatic
fallback for compositors that still have the 0x0 bug.
"""

import sys

import pytest

import main as main_module


def test_module_import_does_not_force_a_platform(monkeypatch):
    """Importing must not pin QT_QPA_PLATFORM: that is what sent every WSL
    user through XWayland."""

    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(
        main_module, "is_wsl", lambda: True
    )
    # Re-importing would re-run the module body; assert on the source
    # instead, since the module has already been imported by the suite.
    import inspect

    source = inspect.getsource(main_module)
    body_before_main = source.split("def main(", 1)[0]
    assert 'os.environ["QT_QPA_PLATFORM"] = "xcb"' not in body_before_main, (
        "the module body must not force xcb at import time"
    )


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (0, 0, True),        # the bug exactly as it presents
        (0, 600, True),      # one axis is enough to be unusable
        (900, 0, True),
        (1, 1, True),
        (900, 600, False),   # a normal window
        (2, 2, False),       # the smallest thing we still accept
    ],
)
def test_needs_platform_fallback(width, height, expected):
    assert main_module.needs_platform_fallback(width, height) is expected


def test_fallback_command_when_frozen(monkeypatch):
    """PyInstaller: sys.executable IS the app, and argv[0] already names
    it, so the interpreter must not be prepended."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/spectratools/SpectraTools")
    monkeypatch.setattr(sys, "argv", ["/opt/spectratools/SpectraTools", "--flag"])
    program, argv = main_module.fallback_command()
    assert program == "/opt/spectratools/SpectraTools"
    assert argv == ["/opt/spectratools/SpectraTools", "--flag"]


def test_fallback_command_from_source(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(sys, "argv", ["main.py", "--flag"])
    program, argv = main_module.fallback_command()
    assert program == "/usr/bin/python3"
    assert argv == ["/usr/bin/python3", "main.py", "--flag"]


def test_is_wsl_reads_proc_version(monkeypatch, tmp_path):
    import builtins

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/version":
            return real_open(tmp_path / "version", *args, **kwargs)
        return real_open(path, *args, **kwargs)

    (tmp_path / "version").write_text(
        "Linux version 6.18.33.2-microsoft-standard-WSL2"
    )
    monkeypatch.setattr(builtins, "open", fake_open)
    assert main_module.is_wsl() is True

    (tmp_path / "version").write_text("Linux version 6.8.0-generic")
    assert main_module.is_wsl() is False


def test_is_wsl_is_false_when_proc_version_is_unreadable(monkeypatch):
    import builtins

    def boom(path, *args, **kwargs):
        if path == "/proc/version":
            raise OSError("nope")
        raise AssertionError("unexpected open")

    monkeypatch.setattr(builtins, "open", boom)
    assert main_module.is_wsl() is False
