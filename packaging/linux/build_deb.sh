#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# This is a separate script, run on a separate WSL host (Ubuntu-24.04),
# rather than a third stage appended to build.sh (which runs on
# AlmaLinux-8). Debian packaging needs dpkg-deb, which requires a
# Debian-family system -- AlmaLinux (RHEL-family) doesn't have it. It
# consumes build.sh's already-built onedir output below rather than
# rebuilding anything itself.
#
# VERSION is re-derived independently from installer.iss below rather
# than handed off from build.sh, matching the existing precedent of
# build.ps1/build.sh each independently reading the same source-of-truth
# file instead of introducing a hand-off mechanism for one value.
#
# BUILD_DIR is placed under $HOME (not the repo path) below for the same
# reason build.sh does this: the repo's /mnt/c path is a slow 9p mount --
# even though this script's own file operations (copying an already-built
# onedir tree, building a .deb) are far lighter than PyInstaller's, the
# same slow-mount concern still applies.

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
# (No hyphen check here, unlike build.sh's RPM path -- Debian's Version:
# field tolerates an embedded hyphen; only RPM's Version: field rejects
# it outright.)

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
Depends: libgl1, libegl1, fontconfig, libxcb-image0, libxcb-cursor0, libxkbcommon-x11-0, libxcb-icccm4, libxcb-keysyms1, libxcb-render-util0, libatomic1, libcrypt1, xdg-utils
Maintainer: SpectraTools <grainovski@googlemail.com>
Description: Spectrum viewer and peak-fitting tool
 A cross-platform desktop app for opening histogram files, plotting
 them, and fitting peaks.
EOF

chmod 0755 "$PKGROOT/DEBIAN"
chmod 0644 "$PKGROOT/DEBIAN/control"

mkdir -p "$ROOT_DIR/packaging/linux/output"
OUT_DEB="$ROOT_DIR/packaging/linux/output/spectratools_${VERSION}_amd64.deb"
# --root-owner-group makes dpkg-deb record every file's owner as root:root
# inside the built package, without needing fakeroot or a manual chown
# pass first -- Debian package contents are conventionally root-owned
# regardless of which user account runs the build.
dpkg-deb --build --root-owner-group "$PKGROOT" "$OUT_DEB"

echo "DEB built at $OUT_DEB"
