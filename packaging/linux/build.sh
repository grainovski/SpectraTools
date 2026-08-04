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

# Build-host prerequisites, installed unconditionally every run (dnf install
# on an already-installed package is a fast no-op, verified) so this script
# is self-sufficient regardless of what state the host is already in --
# python39 for the build itself; binutils because PyInstaller's own
# dependency scanner (needs objdump) fails *silently*, producing a build
# that looks fine but is subtly broken, if binutils is missing -- it does
# not error loudly the way a missing python3.9 would; epel-release plus the
# runtime libs because PyInstaller's own analysis needs to actually resolve
# these libraries at build time too, not just at install time on the target
# machine (xcb-util-cursor specifically needs EPEL on EL8 -- see the design
# spec's "EPEL" note).
dnf install -y epel-release >/dev/null
dnf install -y python39 python39-devel python39-pip binutils \
    mesa-libGL mesa-libEGL fontconfig xcb-util-image xcb-util-cursor \
    libxkbcommon-x11 xcb-util-wm xcb-util-keysyms xcb-util-renderutil libatomic \
    >/dev/null

# Built on AlmaLinux 8 (RHEL 8-compatible, glibc 2.28) rather than whatever
# python3 happens to resolve to, so the shipped binary runs on both old AND
# current RHEL-family releases (glibc is forward-compatible, never backward --
# see docs/superpowers/specs/2026-08-03-linux-native-packages-design.md).
# python39 (the AppStream module package) provides /usr/bin/python3.9, and its
# venv bootstraps its own pip cleanly on this base (verified) -- no
# --without-pip workaround needed here, unlike the old Ubuntu-based build.
python3.9 -m venv .venv
# AlmaLinux 8's python39-pip RPM is frozen at 20.2.4 (RHEL module streams
# don't rebase pip mid-lifecycle -- confirmed via its rpm changelog, last
# touched 2024 for an unrelated patch) and python39's ensurepip ships no
# bundled wheel of its own, so venv creation always inherits that exact
# 20.2.4. That version predates PEP 600 (pip 20.3+) and so cannot see the
# manylinux_2_28-tagged wheels PySide6 has shipped since 6.3 -- it silently
# falls back to the last manylinux1 release it CAN see (6.2.4) and then
# fails the >=6.6 requirement outright. Upgrading pip first (a universal,
# tag-less wheel itself, so any pip version can install it) fixes this
# before it ever touches PySide6.
.venv/bin/python3 -m pip install --quiet --upgrade pip
.venv/bin/python3 -m pip install --quiet -r requirements-dev.txt

if [ ! -f "$ROOT_DIR/assets/icon.png" ]; then
    .venv/bin/python3 "$ROOT_DIR/packaging/make_icon.py"
fi

# Single shared source of truth for the version across every platform and
# package format is installer.iss (Windows-specific file, but the value
# itself isn't) -- mirrors packaging/windows/build.ps1's own stamping step.
# || true: without it, a failed/empty grep here triggers errexit under
# pipefail before the -z check below ever runs (verified) -- this lets
# that check do its job as intended.
VERSION="$(grep '^AppVersion=' "$ROOT_DIR/packaging/windows/installer.iss" | head -1 | sed 's/^AppVersion=//' | tr -d '\r')" || true
if [ -z "$VERSION" ]; then
    echo "Could not find AppVersion in packaging/windows/installer.iss" >&2
    exit 1
fi
# RPM's Version: field rejects hyphens outright (hyphens are reserved as the
# name-version-release separator) -- fail loudly here rather than let
# rpmbuild produce a confusing parse error during the packaging stage.
case "$VERSION" in
    *-*)
        echo "AppVersion '$VERSION' contains a hyphen, which RPM's Version: field cannot contain. Use a plain X.Y.Z version for Linux package builds." >&2
        exit 1
        ;;
esac
# RPM package release number (the "-1" in name-version-release). One
# variable, referenced from the spec's Release: field, its %changelog
# entry, and the final .rpm filename below, so those three spots can't
# drift out of sync.
RELEASE="1"
VERSION_ESCAPED="$(printf '%s' "$VERSION" | sed 's/\\/\\\\/g; s/"/\\"/g')"
BUILD_DATE="$(date +%Y-%m-%d)"
cat > "$BUILD_DIR/build_info.py" <<EOF
VERSION = "$VERSION_ESCAPED"
BUILD_DATE = "$BUILD_DATE"
EOF
echo "Stamped build_info.py: VERSION=$VERSION BUILD_DATE=$BUILD_DATE"

.venv/bin/python3 -m PyInstaller --noconfirm --onedir --windowed --name SpectraTools main.py

# Hand-off point for build_deb.sh (run separately, on the Ubuntu-24.04 WSL
# host): copy the PyInstaller output to the repo's own (gitignored) output
# dir, which every WSL distro sees identically via its own /mnt/c mount of
# this same Windows path -- no WSL-to-WSL bridging needed.
DIST_OUT="$ROOT_DIR/packaging/linux/output/dist-onedir"
rm -rf "$DIST_OUT"
mkdir -p "$DIST_OUT"
cp -a dist/SpectraTools "$DIST_OUT/"

echo "PyInstaller onedir build ready at packaging/linux/output/dist-onedir/SpectraTools/"

# --- RPM packaging (same host, same PyInstaller output — no rebuild) ---

dnf install -y rpm-build >/dev/null

RPM_ROOT="$BUILD_DIR/rpmbuild"
rm -rf "$RPM_ROOT"
mkdir -p "$RPM_ROOT"/{SPECS,SOURCES,BUILD,RPMS,SRPMS,BUILDROOT}

SPEC_FILE="$RPM_ROOT/SPECS/spectratools.spec"
cat > "$SPEC_FILE" <<EOF
Name: spectratools
Version: $VERSION
Release: $RELEASE
Summary: Spectrum viewer and peak-fitting tool
License: Proprietary
BuildArch: x86_64
# rpmbuild's default post-install processing auto-generates a
# /usr/lib/.build-id/xx/yyyy... symlink for every ELF file under the
# buildroot -- with PyInstaller's onedir bundle containing hundreds of
# .so files, this silently shipped 538 undeclared files never listed in
# %files (found via rpm -qpl during review, not anticipated up front).
# debug_package %%{nil} alone does NOT suppress this (verified) -- both
# macros together are required. (%% here escapes the literal % so rpm's
# macro processor doesn't expand the real, built-in "nil" macro even
# inside this comment -- confirmed rpmbuild emits a "Macro expanded in
# comment" warning without the extra %.)
%global _build_id_links none
%global debug_package %{nil}
# rpmbuild's default __os_install_post macro chain also runs brp-strip
# ("strip -g") unconditionally on every ELF file under the buildroot that
# isn't already stripped -- including PyInstaller's bundled _internal/*.so
# files. For numpy's vendored libscipy_openblas64_*.so specifically, that
# in-place strip corrupts the file's ELF segment alignment, breaking it at
# dynamic-link time (ImportError: ... ELF load command address/offset not
# properly aligned) -- found via Task 5's real installed-binary launch,
# not a metadata-only check. This is a SEPARATE mechanism from the
# build-id symlinks handled above: __os_install_post %%{nil} alone does
# NOT suppress those (538 build-id links still appeared when tested
# alone, verified) -- so this line is additive to, not a replacement
# for, the two macros above; all three are required together. None of
# __os_install_post's other normal side effects (doc compression, .py
# bytecompile, shebang mangling, ldconfig cache refresh) apply to a
# self-contained PyInstaller bundle, so disabling the whole chain costs
# nothing here.
%global __os_install_post %{nil}
# Without this, rpmbuild's automatic dependency scanner walks every file
# under %files, including PyInstaller's hundreds of bundled _internal/*.so
# files, and can auto-generate spurious extra Requires/Provides from the
# app's own vendored libraries -- a well-known friction point when
# RPM-packaging PyInstaller onedir bundles. The Requires: lines below are
# the complete, deliberately-curated list; manually-declared Requires
# still work normally with AutoReqProv: no (verified).
AutoReqProv: no
Requires: mesa-libGL
Requires: mesa-libEGL
Requires: fontconfig
Requires: xcb-util-image
Requires: xcb-util-cursor
Requires: libxkbcommon-x11
Requires: xcb-util-wm
Requires: xcb-util-keysyms
Requires: xcb-util-renderutil
Requires: libatomic
Requires: libcrypt.so.1()(64bit)

%description
A cross-platform desktop app for opening histogram files, plotting
them, and fitting peaks.

%install
rm -rf "%{buildroot}"
mkdir -p "%{buildroot}/opt/spectratools"
cp -a "$BUILD_DIR/dist/SpectraTools/." "%{buildroot}/opt/spectratools/"
mkdir -p "%{buildroot}/usr/bin"
ln -s "/opt/spectratools/SpectraTools" "%{buildroot}/usr/bin/spectratools"
mkdir -p "%{buildroot}/usr/share/applications"
cp "$ROOT_DIR/packaging/linux/spectratools.desktop" "%{buildroot}/usr/share/applications/spectratools.desktop"
mkdir -p "%{buildroot}/usr/share/pixmaps"
cp "$ROOT_DIR/assets/icon.png" "%{buildroot}/usr/share/pixmaps/spectratools.png"

%files
/opt/spectratools
/usr/bin/spectratools
/usr/share/applications/spectratools.desktop
/usr/share/pixmaps/spectratools.png

%changelog
* $(date +"%a %b %d %Y") SpectraTools <grainovski@googlemail.com> - $VERSION-$RELEASE
- $VERSION release
EOF

rpmbuild --define "_topdir $RPM_ROOT" -bb "$SPEC_FILE"

mkdir -p "$ROOT_DIR/packaging/linux/output"
cp "$RPM_ROOT/RPMS/x86_64/spectratools-$VERSION-$RELEASE.x86_64.rpm" "$ROOT_DIR/packaging/linux/output/"

echo "RPM built at packaging/linux/output/spectratools-$VERSION-$RELEASE.x86_64.rpm"
