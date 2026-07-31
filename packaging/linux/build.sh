#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Build inside $HOME, not directly on the repo path: under WSL the repo lives on
# the /mnt/c 9p-mounted Windows filesystem, which is far too slow for PyInstaller's
# heavy file scanning of PySide6/Qt. Only the small final artifacts are copied back.
BUILD_DIR="$HOME/.spectratools-build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
# Glob rather than a hand-maintained file list: an explicit list here previously
# went stale (still only 5 of the app's ~17 modules) as the app grew, breaking
# the build with ModuleNotFoundError deep in PyInstaller's analysis. A glob
# can't go stale the same way.
cp "$ROOT_DIR"/*.py "$ROOT_DIR"/requirements.txt "$ROOT_DIR"/requirements-dev.txt "$BUILD_DIR/"
cd "$BUILD_DIR"

# --without-pip --system-site-packages works around this WSL image's python3-pip
# package being in a broken (dpkg "failed-config") state, which makes venv's normal
# ensurepip bootstrap segfault. The system pip (already installed) works fine, so
# the venv borrows it via --system-site-packages instead of bootstrapping its own.
python3 -m venv --without-pip --system-site-packages .venv
.venv/bin/python3 -m pip install --quiet -r requirements-dev.txt

if [ ! -f "$ROOT_DIR/assets/icon.png" ]; then
    .venv/bin/python3 "$ROOT_DIR/packaging/make_icon.py"
fi

# Single shared source of truth for the version across both platforms is
# installer.iss (Windows-specific file, but the value itself isn't) -- mirrors
# packaging/windows/build.ps1's own stamping step. Escaped the same way that
# script escapes it: AppVersion is developer-edited, and a stray backslash or
# quote would otherwise produce a build_info.py with a Python syntax error.
VERSION="$(grep '^AppVersion=' "$ROOT_DIR/packaging/windows/installer.iss" | head -1 | sed 's/^AppVersion=//' | tr -d '\r')"
if [ -z "$VERSION" ]; then
    echo "Could not find AppVersion in packaging/windows/installer.iss" >&2
    exit 1
fi
VERSION_ESCAPED="$(printf '%s' "$VERSION" | sed 's/\\/\\\\/g; s/"/\\"/g')"
BUILD_DATE="$(date +%Y-%m-%d)"
cat > "$BUILD_DIR/build_info.py" <<EOF
VERSION = "$VERSION_ESCAPED"
BUILD_DATE = "$BUILD_DATE"
EOF
echo "Stamped build_info.py: VERSION=$VERSION BUILD_DATE=$BUILD_DATE"

.venv/bin/python3 -m PyInstaller --noconfirm --onedir --windowed --name SpectraTools main.py

APPDIR="$BUILD_DIR/SpectraTools.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp -r dist/SpectraTools/* "$APPDIR/usr/bin/"

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/SpectraTools" "$@"
EOF
chmod +x "$APPDIR/AppRun"

cp "$ROOT_DIR/packaging/linux/spectratools.desktop" "$APPDIR/spectratools.desktop"
cp "$ROOT_DIR/assets/icon.png" "$APPDIR/spectratools.png"

APPIMAGETOOL="$ROOT_DIR/packaging/linux/tools/appimagetool"
if [ ! -x "$APPIMAGETOOL" ]; then
    mkdir -p "$ROOT_DIR/packaging/linux/tools"
    curl -L -o "$APPIMAGETOOL" \
        https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$APPIMAGETOOL"
fi

mkdir -p "$ROOT_DIR/packaging/linux/output"
"$APPIMAGETOOL" --appimage-extract-and-run "$APPDIR" "$ROOT_DIR/packaging/linux/output/SpectraTools-x86_64.AppImage"

echo "AppImage built at packaging/linux/output/SpectraTools-x86_64.AppImage"
