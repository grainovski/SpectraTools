#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Build inside $HOME, not directly on the repo path: under WSL the repo lives on
# the /mnt/c 9p-mounted Windows filesystem, which is far too slow for PyInstaller's
# heavy file scanning of PySide6/Qt. Only the small final artifacts are copied back.
BUILD_DIR="$HOME/.histogram-viewer-build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cp "$ROOT_DIR"/main.py "$ROOT_DIR"/main_window.py "$ROOT_DIR"/histogram_io.py \
   "$ROOT_DIR"/settings.py "$ROOT_DIR"/requirements.txt "$ROOT_DIR"/requirements-dev.txt \
   "$BUILD_DIR/"
cd "$BUILD_DIR"

# --without-pip --system-site-packages works around this WSL image's python3-pip
# package being in a broken (dpkg "failed-config") state, which makes venv's normal
# ensurepip bootstrap segfault. The system pip (already installed) works fine, so
# the venv borrows it via --system-site-packages instead of bootstrapping its own.
python3 -m venv --without-pip --system-site-packages .venv
.venv/bin/python3 -m pip install --quiet -r requirements-dev.txt

.venv/bin/python3 -m PyInstaller --noconfirm --onedir --windowed --name HistogramViewer main.py

APPDIR="$BUILD_DIR/HistogramViewer.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp -r dist/HistogramViewer/* "$APPDIR/usr/bin/"

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/HistogramViewer" "$@"
EOF
chmod +x "$APPDIR/AppRun"

cp "$ROOT_DIR/packaging/linux/histogramviewer.desktop" "$APPDIR/histogramviewer.desktop"
cp "$ROOT_DIR/packaging/linux/icon.png" "$APPDIR/histogramviewer.png"

APPIMAGETOOL="$ROOT_DIR/packaging/linux/tools/appimagetool"
if [ ! -x "$APPIMAGETOOL" ]; then
    mkdir -p "$ROOT_DIR/packaging/linux/tools"
    curl -L -o "$APPIMAGETOOL" \
        https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$APPIMAGETOOL"
fi

mkdir -p "$ROOT_DIR/packaging/linux/output"
"$APPIMAGETOOL" --appimage-extract-and-run "$APPDIR" "$ROOT_DIR/packaging/linux/output/HistogramViewer-x86_64.AppImage"

echo "AppImage built at packaging/linux/output/HistogramViewer-x86_64.AppImage"
