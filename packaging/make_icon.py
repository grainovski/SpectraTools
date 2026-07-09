"""Build helper: render the app's icon (assets/icon.png, assets/icon.ico)
from the exact same drawing code used for the app's own runtime window icon
(main_window._app_icon_pixmap), so all three (window icon, .exe icon,
Linux AppImage icon) are guaranteed to look identical.

assets/icon.ico is a minimal hand-rolled ICO container wrapping PNG-format
entries -- supported natively since Windows Vista -- rather than depending
on Pillow just for icon conversion.
"""
import struct
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtCore import QBuffer
from PySide6.QtGui import QGuiApplication

from main_window import _APP_ICON_SIZES, _app_icon_pixmap

ASSETS_DIR = ROOT_DIR / "assets"


def _pixmap_to_png_bytes(pixmap):
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    data = bytes(buffer.data())
    buffer.close()
    return data


def _write_ico(sizes_and_png_bytes, ico_path):
    count = len(sizes_and_png_bytes)
    header = struct.pack("<HHH", 0, 1, count)
    entries = b""
    images = b""
    offset = 6 + 16 * count
    for size, png_bytes in sizes_and_png_bytes:
        wh = size if size < 256 else 0
        entries += struct.pack("<BBBBHHII", wh, wh, 0, 0, 1, 32, len(png_bytes), offset)
        images += png_bytes
        offset += len(png_bytes)
    ico_path.write_bytes(header + entries + images)


def main():
    app = QGuiApplication([])
    ASSETS_DIR.mkdir(exist_ok=True)

    sizes_and_png_bytes = [(size, _pixmap_to_png_bytes(_app_icon_pixmap(size))) for size in _APP_ICON_SIZES]

    png_path = ASSETS_DIR / "icon.png"
    png_path.write_bytes(dict(sizes_and_png_bytes)[256])

    ico_path = ASSETS_DIR / "icon.ico"
    _write_ico(sizes_and_png_bytes, ico_path)

    print(f"Wrote {png_path} and {ico_path}")


if __name__ == "__main__":
    main()
