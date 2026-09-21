import os
import sys
import threading

#: Set on the re-exec below, so the fallback can only ever happen once.
_FALLBACK_ENV = "SPECTRATOOLS_PLATFORM_FALLBACK"

#: A window smaller than this in either direction is not a window the user
#: can use -- it is the WSLg 0x0-configure bug described in
#: needs_platform_fallback().
_MIN_USABLE_WINDOW = 2


def is_wsl():
    """True when running under Windows Subsystem for Linux."""
    try:
        # Closed explicitly rather than left to refcounting: CPython would
        # collect it at once, but this is startup code and a ResourceWarning
        # here is noise in exactly the place it is hardest to notice.
        with open("/proc/version") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def needs_platform_fallback(width, height):
    """Whether the just-shown main window came up unusably small, meaning
    this WSLg build still has the 0x0 compositor bug.

    Older WSLg sends an xdg_toplevel.configure event with size (0, 0) when
    a Qt window is created, and Qt's Wayland plugin applies that literally
    instead of falling back to MainWindow's own resize() -- the window
    exists in the taskbar but renders at zero size, so nothing is ever
    visible. This app used to force the X11 platform (xcb, via WSLg's
    XWayland) unconditionally to sidestep it.

    That workaround is no longer free. Going through XWayland is what
    caused dialogs to appear, vanish for a second and reappear a few
    seconds later on WSLg: the flicker happens in XWayland's surface
    presentation, below the X protocol entirely -- a trace of a file
    dialog's whole lifetime shows exactly one MAP, one EXPOSE and one
    UNMAP, with no re-expose a repainting compositor would have produced.
    Running on Wayland directly removes that layer and the flicker with
    it, confirmed by the user on the build where it reproduced.

    So Wayland is now preferred and xcb is kept only as an automatic
    fallback, chosen by MEASURING the window rather than by trying to
    detect a WSLg version -- the bug is a property of the compositor's
    behaviour, and the measurement is exactly the symptom.
    """
    return width < _MIN_USABLE_WINDOW or height < _MIN_USABLE_WINDOW


def fallback_command():
    """(program, argv) to re-launch this process under xcb.

    Frozen by PyInstaller, sys.executable IS the application and argv[0]
    already points at it. Running from source it is the interpreter, which
    has to be put in front of the script.
    """
    if getattr(sys, "frozen", False):
        return sys.executable, list(sys.argv)
    return sys.executable, [sys.executable] + list(sys.argv)


# Deliberately NOT forcing a platform here any more. Qt picks Wayland when
# WSLg offers it, which is what removes the dialog flicker; the fallback in
# main() catches the older compositors that need xcb. An explicitly set
# QT_QPA_PLATFORM still wins, so the choice can be overridden by hand.

import matplotlib

matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication

from dialog_utils import RaiseOnClickFilter
from main_window import MainWindow


def _warm_up_scipy_modules():
    """Import the scipy-backed modules on a background thread.

    scipy is deliberately NOT on the startup path -- v5.2.5 took it off and
    measured 209 -> 145 DLLs, 35.3 MB of them, and a warm start of 1218 ->
    1035 ms. The cost does not vanish, it moves: the first action needing
    scipy pays all of it at once. Measured on this machine, opening the
    Automatic Calibration dialog for the first time spends 1.10 s importing
    and 0.01 s building the dialog, against 0.00 s every later time. The
    user sees a frozen window and no explanation.

    Starting the import here keeps both halves of that. The main window is
    already on screen, so nothing the user watches gets slower, and by the
    time they reach a fit or a calibration the import has usually finished.

    A click that beats the thread is not penalised: Python's import lock
    makes the second importer wait for the first, so it waits exactly as
    long as it would have with no warm-up -- never longer.
    """
    def load():
        try:
            import auto_calibrate_dialog  # noqa: F401 - pulls in scipy
        except Exception:
            # A warm-up must never take the app down. The real import runs
            # again on first use and reports its failure properly there.
            pass

    threading.Thread(target=load, name="scipy-warmup", daemon=True).start()


def main():
    app = QApplication(sys.argv)
    # Click any of the app's overlapping windows to bring it forward. Held
    # on `app` because an event filter is not kept alive by the object it
    # is installed on, and a garbage-collected filter stops filtering
    # silently.
    app.raise_on_click_filter = RaiseOnClickFilter(app)
    app.installEventFilter(app.raise_on_click_filter)
    window = MainWindow()
    window.show()

    # Let the compositor deliver its first configure before measuring: on
    # Wayland the size is not known until then, so checking immediately
    # after show() would read the pre-configure value and re-exec every
    # single launch.
    app.processEvents()
    if (
        is_wsl()
        and "QT_QPA_PLATFORM" not in os.environ
        and _FALLBACK_ENV not in os.environ
        and needs_platform_fallback(window.width(), window.height())
    ):
        # This compositor still has the 0x0 bug. Restart on xcb, which
        # renders correctly there, rather than leaving the user with an
        # invisible window -- a flicker is a nuisance, nothing on screen
        # at all is unusable.
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        os.environ[_FALLBACK_ENV] = "1"
        program, argv = fallback_command()
        os.execv(program, argv)

    # After the re-exec check, not before: on WSL the branch above replaces
    # this process outright, and a thread started first would be spawned
    # only to be thrown away.
    _warm_up_scipy_modules()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
