import os
import sys

# WSLg's bundled Wayland compositor sends an xdg_toplevel.configure event
# with size (0, 0) when a Qt window is created; Qt's Wayland platform
# plugin applies that literally instead of falling back to our own
# resize() call in MainWindow.__init__, so the window exists (visible in
# the taskbar) but renders at zero size -- nothing is ever visible.
# Forcing the X11 platform (xcb, via WSLg's built-in XWayland) sidesteps
# this specific compositor bug -- confirmed via a direct window-geometry
# query that xcb renders at the correct, intended 900x600 while Wayland
# renders at 0x0. Scoped to WSL specifically (not all Linux) since this
# hasn't been tested on a real native Wayland desktop, where forcing
# xcb-via-XWayland would be a strictly worse experience if Wayland
# already works fine there. Respects an existing QT_QPA_PLATFORM if one
# is already set, so it can still be overridden manually.
if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
    try:
        is_wsl = "microsoft" in open("/proc/version").read().lower()
    except OSError:
        is_wsl = False
    if is_wsl:
        os.environ["QT_QPA_PLATFORM"] = "xcb"

import matplotlib

matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
