# Linux `.deb` + `.rpm` Packages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken `SpectraTools-v2.0.0-x86_64.AppImage` (fails on WSL/Ubuntu with a missing-FUSE error, fails on CentOS with a GLIBC version mismatch) with native `.deb` and `.rpm` packages whose built-in dependency resolution installs missing runtime libraries automatically, and which install and run correctly on both the RHEL 8-vintage baseline and current RHEL-family releases (verified on AlmaLinux 10.2).

**Architecture:** Build the app once on `AlmaLinux-8` via WSL (Python 3.9, PyInstaller `--onedir`), producing a single compiled output used by both packaging steps — no separate rebuild per format. `.rpm` is built natively via `rpmbuild` on the same AlmaLinux-8 host. `.deb` is built via `dpkg-deb` on the existing Ubuntu-24.04 WSL host, wrapping the AlmaLinux-8-built binary (handed off via the repo's own Windows-mounted path, which every WSL distro sees identically at `/mnt/c/...`). One dependency — `libcrypt.so.1`, needed by Python 3.9's own shared library — is declared as an RPM SONAME capability (`Requires: libcrypt.so.1()(64bit)`) rather than a fixed package name, because the providing package differs between AlmaLinux 8 (`libxcrypt`, present by default) and AlmaLinux 10 (`libxcrypt-compat`, not installed by default, and not even available in AlmaLinux 8's repos under that name).

**Tech Stack:** Bash (build scripts), `rpmbuild`/`rpm` (RPM packaging), `dpkg-deb`/`dpkg` (Debian packaging), PyInstaller 6.x `--onedir`, WSL2 (AlmaLinux-8, AlmaLinux-10, Ubuntu-24.04 distros) as the build/test hosts from Windows.

---

## Context Every Task Needs

**Design spec:** [`docs/superpowers/specs/2026-08-03-linux-native-packages-design.md`](../specs/2026-08-03-linux-native-packages-design.md) — read this first. It has the full root-cause analysis and the verified dependency table this plan implements.

**Repo root (Windows path):** `C:\Users\RIG\Documents\Claude\PeakFinderFitting`. Every WSL distro sees this identically at `/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting` — this is how the AlmaLinux-8 build hands its output to the Ubuntu-24.04 `.deb` step and how the AlmaLinux-10 cross-version test reaches the AlmaLinux-8-built `.rpm`, with no network transfer or WSL-to-WSL bridging needed.

**Existing files this plan touches:**
- `packaging/linux/build.sh` — currently builds a PyInstaller onedir app and wraps it as an AppImage. Rewritten by Tasks 1–2 to produce a `.rpm` instead (PyInstaller stage unchanged in spirit, RPM packaging replaces the AppDir/appimagetool section).
- `packaging/linux/spectratools.desktop` — existing desktop entry, fixed by Task 3.
- `packaging/windows/installer.iss` — untouched, but read by every build script: `AppVersion=2.0.0` (line 3) is the single source of truth for the version number across all platforms and formats.
- `assets/icon.png` — existing 256×256 RGBA PNG, already built by `packaging/make_icon.py`. Reused as-is.

**New files this plan creates:**
- `packaging/linux/build_deb.sh` (Task 4)
- `packaging/linux/INSTALL.md` (Task 8)

**Files this plan removes:**
- `packaging/linux/tools/` (the `appimagetool` download — gitignored, never committed, just delete the local copy)
- `packaging/linux/output/SpectraTools-x86_64.AppImage` (stale build artifact)
- `releases/v2.0.0/SpectraTools-v2.0.0-x86_64.AppImage` (the broken release artifact — replaced by the new `.deb`/`.rpm` in Task 9)

### The WSL Invocation Pattern (every task's verification steps use this)

All verification in this plan runs inside WSL distros from a Windows Git-Bash/PowerShell environment. Three rules, all learned the hard way earlier in this project and confirmed again during this plan's own preparation — **do not deviate from them**:

1. **Always prefix `wsl.exe` calls with `MSYS_NO_PATHCONV=1`** when run from Git Bash, or `/mnt/c/...`-style path arguments get mangled before `wsl.exe` sees them.
   ```bash
   MSYS_NO_PATHCONV=1 wsl.exe -d <DistroName> -u root -- bash "/mnt/c/path/to/script.sh"
   ```
2. **Write real `.sh` script files and invoke those — never inline multi-line `bash -c '...'` strings**, especially ones with loops or `$variable` references. Inline strings passed through Git Bash → `wsl.exe` have repeatedly produced garbled output (empty variables, stale results) in this project. Write the script to a file under `packaging/linux/output/` (already gitignored, so it's safe to scatter throwaway scripts there) and delete it when the task's steps are done.
3. **To check whether a launched GUI process is genuinely alive, run it in the foreground under `timeout -k`, never in the background with `disown` + a later separate check.** The background+disown pattern was tried during this plan's own preparation and silently failed (the process didn't survive past the launching `wsl.exe` invocation, leaving a 0-byte log and no running process — no error, just silent failure). The reliable pattern:
   ```bash
   rm -f /tmp/stdout.log /tmp/stderr.log
   timeout -k 1 8 env QT_QPA_PLATFORM=offscreen /path/to/binary >/tmp/stdout.log 2>/tmp/stderr.log
   echo "EXIT CODE: $?"
   ```
   Exit code **124** means `timeout` killed it after 8 seconds *because it was still running* — this is the unambiguous "genuinely alive" signal. Any other exit code means the process exited on its own before the timeout, which for a GUI app is a crash signal, not success. Always redirect stdout/stderr to **separate** files and `cat` them separately afterward — interleaved output from a single combined stream has been ambiguous to read in the past.
   The script must NOT use `set -e` when it includes a `timeout` call you expect to hit — `-e` would abort the script at the (expected, non-zero) `timeout` exit before later cleanup steps run. Use `set -uo pipefail` (no `-e`) in verification scripts for this reason.

---

## Task 1: `build.sh` — PyInstaller onedir build stage on AlmaLinux 8

**Files:**
- Modify: `packaging/linux/build.sh`

This task replaces the build-tooling portion of the existing script (Python version, dependency install) with the AlmaLinux-8-specific versions verified during design, and adds the hand-off copy step `build_deb.sh` (Task 4) will consume. The AppImage-specific back half of the current file (AppDir, AppRun, appimagetool) is removed here; Task 2 adds the RPM packaging in its place.

- [ ] **Step 1: Replace the file**

```bash
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
.venv/bin/python3 -m pip install --quiet -r requirements-dev.txt

if [ ! -f "$ROOT_DIR/assets/icon.png" ]; then
    .venv/bin/python3 "$ROOT_DIR/packaging/make_icon.py"
fi

# Single shared source of truth for the version across every platform and
# package format is installer.iss (Windows-specific file, but the value
# itself isn't) -- mirrors packaging/windows/build.ps1's own stamping step.
VERSION="$(grep '^AppVersion=' "$ROOT_DIR/packaging/windows/installer.iss" | head -1 | sed 's/^AppVersion=//' | tr -d '\r')"
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
```

- [ ] **Step 2: Run it for real on AlmaLinux 8 and verify the onedir output**

Write this verification script to `packaging/linux/output/_task1_verify.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting
bash packaging/linux/build.sh
echo "=== onedir contents ==="
ls -la packaging/linux/output/dist-onedir/SpectraTools/ | head -20
echo "=== binary is executable ==="
test -x packaging/linux/output/dist-onedir/SpectraTools/SpectraTools && echo "OK: executable"
```

Run:
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d AlmaLinux-8 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/_task1_verify.sh"
```

Expected: the script completes with `PyInstaller onedir build ready at ...` printed by `build.sh` itself, then `OK: executable` at the end. The script installs its own prerequisites (`python39`, `binutils`, `epel-release`, the runtime libs) automatically, so it should succeed on a bare AlmaLinux-8 image with no manual setup. If it still fails at the `AppVersion` check, verify `packaging/windows/installer.iss` line 3 is intact.

Delete the throwaway verify script when done:
```bash
rm packaging/linux/output/_task1_verify.sh
```

- [ ] **Step 3: Commit**

```bash
git add packaging/linux/build.sh
git commit -m "feat: build Linux onedir on AlmaLinux 8 for glibc compatibility

Switches the PyInstaller build stage from whatever python3 resolves to
on Ubuntu, to AlmaLinux 8's python3.9 explicitly. Building on an older
glibc baseline is what lets the same binary run on both old and current
RHEL-family releases (glibc is forward-compatible, never backward).
The AppImage-specific packaging half of this script is replaced in the
next commit."
```

---

## Task 2: `build.sh` — RPM packaging stage

**Files:**
- Modify: `packaging/linux/build.sh` (append to the file from Task 1)

This appends the RPM packaging stage to the same script, using the dependency table verified in the design spec, with the `libcrypt.so.1` SONAME-capability `Requires:` line that makes the single RPM installable on both AlmaLinux 8 and AlmaLinux 10.

- [ ] **Step 1: Append the RPM packaging stage**

Add this to the end of `packaging/linux/build.sh` (after the `echo "PyInstaller onedir build ready..."` line from Task 1):

```bash

# --- RPM packaging (same host, same PyInstaller output — no rebuild) ---

dnf install -y rpm-build >/dev/null

RPM_ROOT="$BUILD_DIR/rpmbuild"
rm -rf "$RPM_ROOT"
mkdir -p "$RPM_ROOT"/{SPECS,SOURCES,BUILD,RPMS,SRPMS,BUILDROOT}

SPEC_FILE="$RPM_ROOT/SPECS/spectratools.spec"
cat > "$SPEC_FILE" <<EOF
Name: spectratools
Version: $VERSION
Release: 1
Summary: Spectrum viewer and peak-fitting tool
License: Proprietary
BuildArch: x86_64
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
rm -rf %{buildroot}
mkdir -p %{buildroot}/opt/spectratools
cp -a $BUILD_DIR/dist/SpectraTools/. %{buildroot}/opt/spectratools/
mkdir -p %{buildroot}/usr/bin
ln -s /opt/spectratools/SpectraTools %{buildroot}/usr/bin/spectratools
mkdir -p %{buildroot}/usr/share/applications
cp $ROOT_DIR/packaging/linux/spectratools.desktop %{buildroot}/usr/share/applications/spectratools.desktop
mkdir -p %{buildroot}/usr/share/pixmaps
cp $ROOT_DIR/assets/icon.png %{buildroot}/usr/share/pixmaps/spectratools.png

%files
/opt/spectratools
/usr/bin/spectratools
/usr/share/applications/spectratools.desktop
/usr/share/pixmaps/spectratools.png

%changelog
* $(date +"%a %b %d %Y") SpectraTools <grainovski@googlemail.com> - $VERSION-1
- $VERSION release
EOF

rpmbuild --define "_topdir $RPM_ROOT" -bb "$SPEC_FILE"

mkdir -p "$ROOT_DIR/packaging/linux/output"
cp "$RPM_ROOT/RPMS/x86_64/spectratools-$VERSION-1.x86_64.rpm" "$ROOT_DIR/packaging/linux/output/"

echo "RPM built at packaging/linux/output/spectratools-$VERSION-1.x86_64.rpm"
```

**Why the `libcrypt.so.1` line is a capability, not a package name:** PyInstaller bundles Python 3.9's own shared library, which links against `libcrypt.so.1`. On AlmaLinux 8, the default `libxcrypt` package provides it directly. On AlmaLinux 10, that SONAME moved to a separate `libxcrypt-compat` package — one that doesn't exist at all in AlmaLinux 8's repos. A hardcoded `Requires: libxcrypt-compat` would make the RPM uninstallable on EL8. `Requires: libcrypt.so.1()(64bit)` lets `dnf`/`rpm` resolve to whichever package actually provides that library on the target system. See the design spec's "Compatibility Target" and "Verified Before Writing This Spec" sections for the full investigation.

- [ ] **Step 2: Run it for real and verify the RPM's metadata and dependencies**

Write this verification script to `packaging/linux/output/_task2_verify.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting
bash packaging/linux/build.sh
RPM_FILE=$(ls packaging/linux/output/spectratools-*.rpm | head -1)
echo "=== rpm -qip ==="
rpm -qip "$RPM_FILE"
echo "=== rpm -qp --requires (must list all 11 deps below) ==="
rpm -qp --requires "$RPM_FILE"
```

Run:
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d AlmaLinux-8 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/_task2_verify.sh"
```

Expected: `rpm -qip` shows `Name: spectratools`, `Version:` matching `installer.iss`'s `AppVersion`, `License: Proprietary`. `rpm -qp --requires` lists `mesa-libGL`, `mesa-libEGL`, `fontconfig`, `xcb-util-image`, `xcb-util-cursor`, `libxkbcommon-x11`, `xcb-util-wm`, `xcb-util-keysyms`, `xcb-util-renderutil`, `libatomic`, and `libcrypt.so.1()(64bit)` (plus some `rpmlib(...)` lines, which are normal RPM internals, not app dependencies).

Delete the throwaway verify script when done:
```bash
rm packaging/linux/output/_task2_verify.sh
```

- [ ] **Step 3: Commit**

```bash
git add packaging/linux/build.sh
git commit -m "feat: package Linux build as a native .rpm instead of an AppImage

rpmbuild runs on the same AlmaLinux 8 host as the PyInstaller build, no
rebuild. Requires: declares the dependency table verified in the design
spec, with libcrypt.so.1 as a SONAME capability rather than a fixed
package name so the same RPM installs on both AlmaLinux 8 (libxcrypt)
and AlmaLinux 10 (libxcrypt-compat) without distro-conditional logic."
```

---

## Task 3: Fix `spectratools.desktop`'s `Exec=` line

**Files:**
- Modify: `packaging/linux/spectratools.desktop`

**Why this is needed:** the design's planned launcher symlink is `/usr/bin/spectratools` (lowercase, matching the `spectratools` package name convention). The current desktop file has `Exec=SpectraTools` (capital S-T, matching the PyInstaller binary name). Desktop-entry `Exec=` with a bare command name is resolved via `$PATH`, and `$PATH` lookups are case-sensitive on Linux — `SpectraTools` would never match a lowercase `spectratools` symlink, silently breaking launches from the desktop menu (the terminal/CLI path via the lowercase symlink would still work; only the desktop-menu entry would be broken). The fix: point `Exec=` at the real binary's absolute path instead of relying on `$PATH` at all — the standard, unambiguous pattern for installed desktop files.

- [ ] **Step 1: Fix the file**

Current content:
```
[Desktop Entry]
Type=Application
Name=SpectraTools
Exec=SpectraTools
Icon=spectratools
Categories=Science;
```

New content:
```
[Desktop Entry]
Type=Application
Name=SpectraTools
Exec=/opt/spectratools/SpectraTools
Icon=spectratools
Categories=Science;
```

- [ ] **Step 2: Verify**

```bash
cat packaging/linux/spectratools.desktop
```

Expected: `Exec=/opt/spectratools/SpectraTools` present. (This file is already protected against CRLF corruption by `.gitattributes`'s `*.desktop text eol=lf` rule — no further action needed there.)

- [ ] **Step 3: Commit**

```bash
git add packaging/linux/spectratools.desktop
git commit -m "fix: use absolute Exec= path in desktop entry

Exec=SpectraTools (bare, PATH-resolved) would never match the planned
lowercase /usr/bin/spectratools launcher symlink on a case-sensitive
filesystem, silently breaking desktop-menu launches. An absolute path
sidesteps PATH/case-matching entirely, which is the standard pattern
for installed desktop files anyway."
```

---

## Task 4: `build_deb.sh` — Debian packaging on Ubuntu 24.04

**Files:**
- Create: `packaging/linux/build_deb.sh`

Wraps the SAME AlmaLinux-8-built onedir output (left at `packaging/linux/output/dist-onedir/SpectraTools/` by Task 1's `build.sh`) into a `.deb`, using `dpkg-deb` on the Ubuntu-24.04 WSL host. Independently re-reads `installer.iss`'s `AppVersion`, matching the existing precedent of `build.ps1`/`build.sh` each independently reading the same source-of-truth file rather than introducing a new hand-off mechanism for one value.

- [ ] **Step 1: Create the file**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DIST_ONEDIR="$ROOT_DIR/packaging/linux/output/dist-onedir/SpectraTools"
if [ ! -d "$DIST_ONEDIR" ]; then
    echo "No PyInstaller build found at packaging/linux/output/dist-onedir/SpectraTools -- run packaging/linux/build.sh (on AlmaLinux 8) first." >&2
    exit 1
fi

VERSION="$(grep '^AppVersion=' "$ROOT_DIR/packaging/windows/installer.iss" | head -1 | sed 's/^AppVersion=//' | tr -d '\r')"
if [ -z "$VERSION" ]; then
    echo "Could not find AppVersion in packaging/windows/installer.iss" >&2
    exit 1
fi

BUILD_DIR="$HOME/.spectratools-deb-build"
rm -rf "$BUILD_DIR"
PKGROOT="$BUILD_DIR/pkgroot"
mkdir -p "$PKGROOT/DEBIAN" "$PKGROOT/opt/spectratools" "$PKGROOT/usr/bin" "$PKGROOT/usr/share/applications" "$PKGROOT/usr/share/pixmaps"

cp -a "$DIST_ONEDIR/." "$PKGROOT/opt/spectratools/"
ln -s /opt/spectratools/SpectraTools "$PKGROOT/usr/bin/spectratools"
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
```

- [ ] **Step 2: Make it executable and run it for real**

```bash
chmod +x packaging/linux/build_deb.sh
```

Write this verification script to `packaging/linux/output/_task4_verify.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting
bash packaging/linux/build_deb.sh
DEB_FILE=$(ls packaging/linux/output/spectratools_*.deb | head -1)
echo "=== dpkg -I ==="
dpkg -I "$DEB_FILE"
echo "=== dpkg -c (contents) ==="
dpkg -c "$DEB_FILE"
```

Run (this consumes the `dist-onedir` output Task 1's verification run already left at `packaging/linux/output/dist-onedir/`, visible to Ubuntu-24.04 via the same `/mnt/c` path):
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/_task4_verify.sh"
```

Expected: `dpkg -I` shows `Package: spectratools`, `Version:` matching `installer.iss`, `Depends:` listing all 11 packages (`libgl1, libegl1, fontconfig, libxcb-image0, libxcb-cursor0, libxkbcommon-x11-0, libxcb-icccm4, libxcb-keysyms1, libxcb-render-util0, libatomic1, libcrypt1`). `dpkg -c` shows `./opt/spectratools/SpectraTools`, `./usr/bin/spectratools` (as a symlink), `./usr/share/applications/spectratools.desktop`, `./usr/share/pixmaps/spectratools.png`.

Delete the throwaway verify script when done:
```bash
rm packaging/linux/output/_task4_verify.sh
```

- [ ] **Step 3: Commit**

```bash
git add packaging/linux/build_deb.sh
git commit -m "feat: add build_deb.sh to package the Linux build as a native .deb

Wraps the same AlmaLinux-8-built onedir output used for the .rpm --
runs on the existing Ubuntu-24.04 WSL host via dpkg-deb, no separate
rebuild. Depends: uses the Debian-side dependency names verified
against real apt-cache/dpkg -S output in the design spec."
```

---

## Task 5: Verify automatic dependency resolution for the RPM (AlmaLinux 8)

**Files:** none (verification only)

This is the task that actually proves the user's original ask — "automatic installation of any necessary libs" — works, by removing the runtime dependencies and confirming `dnf install ./spectratools.rpm` pulls them back in in the same transaction, then confirming the installed app genuinely runs, then confirming clean uninstall.

- [ ] **Step 1: Write and run the verification script**

Write this to `packaging/linux/output/_task5_verify.sh`:

```bash
#!/usr/bin/env bash
set -uo pipefail
# No -e: the timeout call below is expected to return a non-zero exit code
# (124) when the app is genuinely alive, which would otherwise abort the
# script before the cleanup steps run.

echo "=== Removing runtime deps to test automatic dependency resolution ==="
dnf remove -y mesa-libGL mesa-libEGL fontconfig xcb-util-image xcb-util-cursor \
    libxkbcommon-x11 xcb-util-wm xcb-util-keysyms xcb-util-renderutil libatomic \
    > /tmp/task5_remove.log 2>&1
tail -20 /tmp/task5_remove.log

echo "=== Installing the RPM (dnf should auto-pull everything just removed) ==="
RPM_FILE=$(ls /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/spectratools-*.rpm | head -1)
echo "Installing: $RPM_FILE"
dnf install -y "$RPM_FILE" > /tmp/task5_install.log 2>&1
echo "dnf install exit code: $?"
cat /tmp/task5_install.log

echo "=== Confirming package registered ==="
rpm -q spectratools

echo "=== Confirming files landed correctly ==="
ls -la /usr/bin/spectratools
readlink -f /usr/bin/spectratools
cat /usr/share/applications/spectratools.desktop
ls -la /usr/share/pixmaps/spectratools.png

echo "=== Running the installed binary via its real entry point ==="
rm -f /tmp/task5_stdout.log /tmp/task5_stderr.log
timeout -k 1 8 env QT_QPA_PLATFORM=offscreen /usr/bin/spectratools >/tmp/task5_stdout.log 2>/tmp/task5_stderr.log
echo "REAL EXIT CODE: $?"
echo "--- stdout ---"
cat /tmp/task5_stdout.log
echo "--- stderr ---"
cat /tmp/task5_stderr.log

echo "=== Uninstalling cleanly ==="
dnf remove -y spectratools > /tmp/task5_remove_pkg.log 2>&1
cat /tmp/task5_remove_pkg.log
ls /usr/bin/spectratools 2>&1 || echo "CONFIRMED: symlink removed"
```

Run:
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d AlmaLinux-8 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/_task5_verify.sh"
```

Expected:
- The `dnf install` transaction summary shows `Installing dependencies:` with (at minimum) `mesa-libGL`, `mesa-libEGL`, `fontconfig`, `xcb-util-image`, `xcb-util-cursor`, `libxkbcommon-x11`, `xcb-util-wm`, `xcb-util-keysyms`, `xcb-util-renderutil`, `libatomic` listed — proving `dnf` pulled them in automatically, not that they were already present. (`libcrypt.so.1()(64bit)` may or may not trigger a new install here specifically, since AlmaLinux 8's base `libxcrypt` package already provides it — that's expected and correct; the point of this task is the other 10 libraries plus confirming the RPM installs cleanly at all.)
- `rpm -q spectratools` prints the installed NVR (e.g. `spectratools-2.0.0-1.x86_64`).
- `readlink -f /usr/bin/spectratools` prints `/opt/spectratools/SpectraTools`.
- `REAL EXIT CODE: 124` — the app launched and was still running when `timeout` killed it 8 seconds later. Any other exit code means it crashed; read `/tmp/task5_stderr.log` for a missing-library error (`error while loading shared libraries: ...`) and cross-check the missing `.so` against the design spec's dependency table.
- After `dnf remove -y spectratools`, `/usr/bin/spectratools` no longer exists.

If any dependency fails to resolve, first re-check that EPEL is enabled on this AlmaLinux-8 instance (`dnf repolist | grep -i epel`) — `xcb-util-cursor` specifically needs it there.

Delete the throwaway verify script when done:
```bash
rm packaging/linux/output/_task5_verify.sh
```

- [ ] **Step 2: No commit** (verification-only task, no source files changed). If a real bug was found and fixed as a result, that fix belongs in Task 2's script and should be committed there — re-run this task's verification afterward to confirm the fix.

---

## Task 6: Verify automatic dependency resolution for the DEB (Ubuntu 24.04)

**Files:** none (verification only)

Same proof as Task 5, for the `.deb` on a Debian-family system via `apt`.

- [ ] **Step 1: Write and run the verification script**

Write this to `packaging/linux/output/_task6_verify.sh`:

```bash
#!/usr/bin/env bash
set -uo pipefail

echo "=== Removing runtime deps to test automatic dependency resolution ==="
apt-get remove -y libgl1 libegl1 fontconfig libxcb-image0 libxcb-cursor0 \
    libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 libxcb-render-util0 libatomic1 \
    > /tmp/task6_remove.log 2>&1
tail -20 /tmp/task6_remove.log

echo "=== Installing the DEB (apt should auto-pull everything just removed) ==="
DEB_FILE=$(ls /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/spectratools_*.deb | head -1)
echo "Installing: $DEB_FILE"
apt-get install -y "$DEB_FILE" > /tmp/task6_install.log 2>&1
echo "apt-get install exit code: $?"
cat /tmp/task6_install.log

echo "=== Confirming package registered ==="
dpkg -s spectratools | head -5

echo "=== Confirming files landed correctly ==="
ls -la /usr/bin/spectratools
readlink -f /usr/bin/spectratools
cat /usr/share/applications/spectratools.desktop
ls -la /usr/share/pixmaps/spectratools.png

echo "=== Running the installed binary via its real entry point ==="
rm -f /tmp/task6_stdout.log /tmp/task6_stderr.log
timeout -k 1 8 env QT_QPA_PLATFORM=offscreen /usr/bin/spectratools >/tmp/task6_stdout.log 2>/tmp/task6_stderr.log
echo "REAL EXIT CODE: $?"
echo "--- stdout ---"
cat /tmp/task6_stdout.log
echo "--- stderr ---"
cat /tmp/task6_stderr.log

echo "=== Uninstalling cleanly ==="
apt-get remove -y spectratools > /tmp/task6_remove_pkg.log 2>&1
cat /tmp/task6_remove_pkg.log
ls /usr/bin/spectratools 2>&1 || echo "CONFIRMED: symlink removed"
```

Run:
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/_task6_verify.sh"
```

Expected: same shape as Task 5 — `apt-get install` shows it pulling in the removed libraries automatically, `dpkg -s spectratools` shows `Status: install ok installed`, `REAL EXIT CODE: 124`, and clean removal at the end.

Delete the throwaway verify script when done:
```bash
rm packaging/linux/output/_task6_verify.sh
```

- [ ] **Step 2: No commit** (verification-only task; any real fix found belongs in Task 4 and should be committed there).

---

## Task 7: Cross-version verification — install the AlmaLinux-8-built RPM on a clean AlmaLinux 10

**Files:** none (verification only)

This is the task that specifically proves the compatibility requirement driving this whole plan: the same RPM, built on AlmaLinux 8, must install and run correctly on a **current** RHEL-family release. This was verified manually during design (copying files around and installing packages by hand); this task re-verifies it the real way — installing the actual packaged `.rpm` via `dnf` — since that's the first point the SONAME-capability `Requires:` line is exercised by dnf's own dependency resolver rather than by hand.

- [ ] **Step 1: Write and run the verification script**

Write this to `packaging/linux/output/_task7_verify.sh`:

```bash
#!/usr/bin/env bash
set -uo pipefail

echo "=== Baseline: remove all target packages if present, for a clean slate ==="
dnf remove -y mesa-libGL mesa-libEGL fontconfig xcb-util-image xcb-util-cursor \
    libxkbcommon-x11 xcb-util-wm xcb-util-keysyms xcb-util-renderutil libatomic \
    libxcrypt-compat spectratools \
    > /tmp/task7_baseline_remove.log 2>&1
tail -20 /tmp/task7_baseline_remove.log

echo "=== Installing the EL8-built RPM on this EL10 system ==="
RPM_FILE=$(ls /mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/spectratools-*.rpm | head -1)
echo "Installing: $RPM_FILE"
dnf install -y "$RPM_FILE" > /tmp/task7_install.log 2>&1
echo "dnf install exit code: $?"
cat /tmp/task7_install.log

echo "=== Confirming libxcrypt-compat was specifically pulled in (the EL10-only dependency) ==="
rpm -q libxcrypt-compat

echo "=== Confirming package registered ==="
rpm -q spectratools

echo "=== Running the installed binary via its real entry point ==="
rm -f /tmp/task7_stdout.log /tmp/task7_stderr.log
timeout -k 1 8 env QT_QPA_PLATFORM=offscreen /usr/bin/spectratools >/tmp/task7_stdout.log 2>/tmp/task7_stderr.log
echo "REAL EXIT CODE: $?"
echo "--- stdout ---"
cat /tmp/task7_stdout.log
echo "--- stderr ---"
cat /tmp/task7_stderr.log

echo "=== Uninstalling cleanly ==="
dnf remove -y spectratools > /tmp/task7_remove_pkg.log 2>&1
cat /tmp/task7_remove_pkg.log
```

Run:
```bash
MSYS_NO_PATHCONV=1 wsl.exe -d AlmaLinux-10 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/output/_task7_verify.sh"
```

Expected: the `dnf install` transaction summary explicitly lists `libxcrypt-compat` under `Installing dependencies:` (proving the SONAME-capability `Requires:` line correctly resolved to the EL10-specific package — this exact mechanism was confirmed during design using a throwaway test RPM; this step re-confirms it with the real SpectraTools RPM). `rpm -q libxcrypt-compat` and `rpm -q spectratools` both succeed. `REAL EXIT CODE: 124`, stderr shows no missing-library errors (a `Fontconfig warning: ... unknown element "reset-dirs"` and `This plugin does not support propagateSizeHints()` are known-benign and expected — not failures).

Delete the throwaway verify script when done:
```bash
rm packaging/linux/output/_task7_verify.sh
```

- [ ] **Step 2: No commit** (verification-only task).

---

## Task 8: Write `packaging/linux/INSTALL.md`

**Files:**
- Create: `packaging/linux/INSTALL.md`

- [ ] **Step 1: Create the file**

```markdown
# Installing SpectraTools on Linux

SpectraTools ships as native `.deb` and `.rpm` packages. Both automatically
install any missing runtime libraries as part of the same install command —
no separate dependency-hunting required.

## Debian / Ubuntu

```
sudo apt install ./spectratools_<version>_amd64.deb
```

(Or double-click the file in a file manager with a software-installer GUI.)

Uninstall:
```
sudo apt remove spectratools
```

## RHEL / CentOS / AlmaLinux 8

First enable EPEL (one-time, if not already enabled):
```
sudo dnf install epel-release
```

Then:
```
sudo dnf install ./spectratools-<version>-1.x86_64.rpm
```

Uninstall:
```
sudo dnf remove spectratools
```

## RHEL / CentOS / AlmaLinux 9, 10, and current Fedora

No EPEL needed — install directly:
```
sudo dnf install ./spectratools-<version>-1.x86_64.rpm
```

Uninstall:
```
sudo dnf remove spectratools
```

## After installing

SpectraTools appears in your desktop's application menu (under Science), or
run `spectratools` from a terminal.
```

- [ ] **Step 2: Verify**

```bash
cat packaging/linux/INSTALL.md
```

Expected: renders as valid Markdown, matches the design spec's "Install Instructions" section.

- [ ] **Step 3: Commit**

```bash
git add packaging/linux/INSTALL.md
git commit -m "docs: add Linux install instructions for the .deb/.rpm packages"
```

---

## Task 9: Clean up AppImage artifacts and archive the v2.0.0 Linux release

**Files:**
- Delete: `packaging/linux/tools/` (local only — gitignored, was never tracked)
- Delete: `packaging/linux/output/SpectraTools-x86_64.AppImage` (stale, local only)
- Delete: `releases/v2.0.0/SpectraTools-v2.0.0-x86_64.AppImage`
- Create: `releases/v2.0.0/spectratools_2.0.0_amd64.deb`
- Create: `releases/v2.0.0/spectratools-2.0.0-1.x86_64.rpm`
- Create: `releases/v2.0.0/INSTALL.md` (copy)

**Archived under their native build-output names, not renamed to the `SpectraTools-vX.Y.Z-...` pattern used by the Windows installer and the old AppImage:** `INSTALL.md`'s install commands (Task 8) reference the exact lowercase filenames `dpkg-deb`/`rpmbuild` produce (`spectratools_2.0.0_amd64.deb`, `spectratools-2.0.0-1.x86_64.rpm`). A user copy-pasting `sudo apt install ./spectratools_2.0.0_amd64.deb` needs that filename to match what's actually in the folder — renaming to match the Windows convention would silently break the exact commands the instructions tell them to run. (The Windows `.exe` doesn't have this problem: it's launched by double-click, not by typing its name into a command.)

`releases/` and `packaging/linux/output/`/`packaging/linux/tools/` are all gitignored (confirmed in `.gitignore`), so none of this touches git history — it's just local file cleanup and archiving, same as the existing v1.0.0/v2.0.0 Windows installer archiving already does.

- [ ] **Step 1: Remove the dead AppImage tooling and stale artifact**

```bash
rm -rf packaging/linux/tools
rm -f packaging/linux/output/SpectraTools-x86_64.AppImage
```

- [ ] **Step 2: Re-run the full build to produce fresh, final artifacts**

```bash
rm -rf packaging/linux/output/dist-onedir packaging/linux/output/*.rpm packaging/linux/output/*.deb
```

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d AlmaLinux-8 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/build.sh"
```

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash "/mnt/c/Users/RIG/Documents/Claude/PeakFinderFitting/packaging/linux/build_deb.sh"
```

Expected: `packaging/linux/output/spectratools-2.0.0-1.x86_64.rpm` and `packaging/linux/output/spectratools_2.0.0_amd64.deb` both exist afterward (adjust the version numbers if `installer.iss`'s `AppVersion` has changed since this plan was written).

- [ ] **Step 3: Archive into `releases/v2.0.0/`**

```bash
rm -f releases/v2.0.0/SpectraTools-v2.0.0-x86_64.AppImage
cp packaging/linux/output/spectratools-2.0.0-1.x86_64.rpm releases/v2.0.0/
cp packaging/linux/output/spectratools_2.0.0_amd64.deb releases/v2.0.0/
cp packaging/linux/INSTALL.md releases/v2.0.0/INSTALL.md
```

- [ ] **Step 4: Verify**

```bash
ls releases/v2.0.0/
```

Expected: `SpectraTools-v2.0.0-Setup.exe` (unchanged, Windows), `spectratools-2.0.0-1.x86_64.rpm`, `spectratools_2.0.0_amd64.deb`, `INSTALL.md`. No `.AppImage` file. The `.rpm`/`.deb` filenames are lowercase and un-prefixed (matching `INSTALL.md`'s commands exactly) even though the Windows installer alongside them uses the branded `SpectraTools-v2.0.0-...` pattern — this is intentional, see the note under Task 9's file list above.

- [ ] **Step 5: No commit needed for `releases/`** (gitignored). If any non-gitignored files changed in this task (none expected), commit those separately.

---

## Final Step: Hand back to the user

Once all 9 tasks are complete, report the final `releases/v2.0.0/` contents and the outcome of Tasks 5–7's verification (automatic dependency resolution confirmed on both formats, cross-version install confirmed on AlmaLinux 10) back to the user. Do not tag a new version or push anything — this plan fixes the existing v2.0.0 Linux release artifact, it doesn't cut a new version.
