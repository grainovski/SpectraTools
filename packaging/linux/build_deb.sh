#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DIST_ONEDIR="$ROOT_DIR/packaging/linux/output/dist-onedir/SpectraTools"
if [ ! -d "$DIST_ONEDIR" ]; then
    echo "No PyInstaller build found at packaging/linux/output/dist-onedir/SpectraTools -- run packaging/linux/build.sh (on AlmaLinux 8) first." >&2
    exit 1
fi

# || true: without it, a failed/empty grep here triggers errexit under
# pipefail before the -z check below ever runs (verified in build.sh's
# identical pattern) -- this lets that check do its job as intended.
VERSION="$(grep '^AppVersion=' "$ROOT_DIR/packaging/windows/installer.iss" | head -1 | sed 's/^AppVersion=//' | tr -d '\r')" || true
if [ -z "$VERSION" ]; then
    echo "Could not find AppVersion in packaging/windows/installer.iss" >&2
    exit 1
fi

BUILD_DIR="$HOME/.spectratools-deb-build"
rm -rf "$BUILD_DIR"
PKGROOT="$BUILD_DIR/pkgroot"
mkdir -p "$PKGROOT/DEBIAN" "$PKGROOT/opt/spectratools" "$PKGROOT/usr/bin" "$PKGROOT/usr/share/applications" "$PKGROOT/usr/share/pixmaps"

cp -a "$DIST_ONEDIR/." "$PKGROOT/opt/spectratools/"
ln -s "/opt/spectratools/SpectraTools" "$PKGROOT/usr/bin/spectratools"
cp "$ROOT_DIR/packaging/linux/spectratools.desktop" "$PKGROOT/usr/share/applications/spectratools.desktop"
cp "$ROOT_DIR/assets/icon.png" "$PKGROOT/usr/share/pixmaps/spectratools.png"

cat > "$PKGROOT/DEBIAN/control" <<EOF
Package: spectratools
Version: $VERSION
Section: science
Priority: optional
Architecture: amd64
Depends: libgl1, libegl1, fontconfig, libxcb-image0, libxcb-cursor0, libxkbcommon-x11-0, libxcb-icccm4, libxcb-keysyms1, libxcb-render-util0, libatomic1, libcrypt1
Maintainer: SpectraTools <grainovski@googlemail.com>
Description: Spectrum viewer and peak-fitting tool
 A cross-platform desktop app for opening histogram files, plotting
 them, and fitting peaks.
EOF

chmod 0755 "$PKGROOT/DEBIAN"
chmod 0644 "$PKGROOT/DEBIAN/control"

mkdir -p "$ROOT_DIR/packaging/linux/output"
OUT_DEB="$ROOT_DIR/packaging/linux/output/spectratools_${VERSION}_amd64.deb"
dpkg-deb --build --root-owner-group "$PKGROOT" "$OUT_DEB"

echo "DEB built at packaging/linux/output/spectratools_${VERSION}_amd64.deb"
