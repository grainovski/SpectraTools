# Linux `.deb` + `.rpm` Packages — Design Spec

Date: 2026-08-03

## Purpose

The v2.0.0 AppImage release ([[project_versioning_and_releases.md]]) doesn't work for the user: WSL/Ubuntu is missing `libfuse.so.2` (AppImage's default execution mode needs FUSE to mount itself), and CentOS refuses to load it at all (`GLIBC_2.38' not found` — the AppImage was built on Ubuntu 24.04, whose glibc is newer than CentOS ships). User asked to fix it and produce "a self-sufficient installation package (including checking for dependencies and automatic installation of any necessary libs) with instructions how to install."

## Root Causes (diagnosed and independently verified before designing)

1. **FUSE**: AppImage's default double-click/direct-run execution mounts the embedded squashfs via FUSE. No FUSE2 on this WSL image. `--appimage-extract-and-run` works around it, but that's not a real fix for an end user.
2. **glibc**: PyInstaller bundles Python and compiled extensions, but never bundles glibc itself (by design — glibc is meant to be system-provided). A binary built on Ubuntu 24.04 requires glibc ≥2.38 at runtime; older distros (CentOS Stream 9 ships 2.34) can't satisfy that. Building on an older baseline is the only real fix — there's no way to "patch" this after the fact without bundling glibc itself, which is fragile and not standard practice.

## Resolved Decisions (through discussion with the user)

- **Package format**: native `.deb` and `.rpm`, not a hand-rolled shell installer or AppImage. Chosen specifically because native package managers resolve declared dependencies automatically (`apt install ./x.deb` / `dnf install ./x.rpm` pulls in anything missing from the same transaction) — this satisfies "automatic installation of dependencies" using standard, idiomatic tooling rather than a custom detection script.
- **AppImage is fully replaced**, not kept alongside. It doesn't solve a problem the native packages don't solve better, and the FUSE issue means it wasn't actually a working no-install option anyway.
- **Build baseline**: `AlmaLinux-8` (RHEL 8 binary-compatible, glibc 2.28) instead of Ubuntu 24.04. The *same* compiled binary is wrapped by both package formats — no separate rebuild per format, since rebuilding on Ubuntu would reintroduce the glibc problem for the `.deb` too.

## Verified Before Writing This Spec (not assumed)

Installed a real `AlmaLinux-8` WSL distro and actually built the app there, end to end, resolving each failure empirically rather than guessing:

- **Python**: AlmaLinux 8's base repos don't include `python3` by default; `python39` (Python 3.9.25) is available via the AppStream module and is what the build uses. `pip install "PySide6>=6.6,<6.9"` succeeds with no glibc-related error at all — confirming the *wheel itself* isn't tied to a newer glibc than 2.28 (this was the single biggest open risk in this design; it's now closed).
- **Build tooling**: `binutils` (provides `objdump`) must be installed before running PyInstaller on this base — PyInstaller's binary-dependency analysis needs it and doesn't fail loudly if it's missing versus just producing a broken build silently, so this must be an explicit step.
- **Runtime dependencies**: resolved one at a time by actually trying to construct a `QApplication` and installing whatever `ImportError` named, then rebuilt the full app with PyInstaller and cross-checked its own dependency-scan warnings, then **ran the actual built binary** (not just built it) to confirm it stays alive (`timeout -k 1 6` returning exit 124 — killed while still running — rather than a suspiciously-immediate exit 0, which the [[reference_wsl_invocation_from_git_bash.md]] memory already flagged as a false-negative trap). Final verified mapping, both distro families:

| Missing `.so` | RPM package (AlmaLinux/RHEL 8, needs `Requires:`) | Debian package (Ubuntu/Debian, needs `Depends:`) |
|---|---|---|
| `libGL.so.1` | `mesa-libGL` | `libgl1` |
| `libEGL.so.1` | `mesa-libEGL` | `libegl1` |
| `libfontconfig.so.1` | `fontconfig` | `fontconfig` |
| `libxcb-image.so.0` | `xcb-util-image` | `libxcb-image0` |
| `libxcb-cursor.so.0` | `xcb-util-cursor` **(needs EPEL — see below)** | `libxcb-cursor0` |
| `libxkbcommon-x11.so.0` | `libxkbcommon-x11` | `libxkbcommon-x11-0` |
| `libxcb-icccm.so.4` | `xcb-util-wm` | `libxcb-icccm4` |
| `libxcb-keysyms.so.1` | `xcb-util-keysyms` | `libxcb-keysyms1` |
| `libxcb-render-util.so.0` | `xcb-util-renderutil` | `libxcb-render-util0` |
| `libatomic.so.1` | `libatomic` | `libatomic1` |

- **EPEL**: `xcb-util-cursor` is not in AlmaLinux 8's base/AppStream repos, only in EPEL (`dnf install epel-release`). This is extremely standard for RHEL-family desktop software — EPEL is a near-universal prerequisite for anything beyond core server packages — but it must be called out explicitly in the install instructions as a one-time prerequisite, since a fresh minimal RHEL/CentOS/AlmaLinux install won't have it enabled by default and `dnf install ./spectratools.rpm` will otherwise fail to resolve that one dependency.
- Debian-side names cross-checked directly against `apt-cache show`/`dpkg -S` on the real Ubuntu 24.04 WSL image (not assumed from memory) — confirmed `libgl1`/`libegl1` are correct (not `libglvnd0`, a different, lower-level package that was an easy wrong guess).

## Architecture

**Build stage** (on `AlmaLinux-8` via WSL):
1. Install `python39`, `python39-devel`, `python39-pip`, `binutils`, plus the runtime libs above (needed on the *build* machine too, since PyInstaller's dependency scanner needs to actually resolve them to embed correct references — though the shipped binary relies on the *target* machine having them via the declared package dependencies, not by bundling them).
2. `python3.9 -m venv`, `pip install -r requirements-dev.txt`.
3. Stamp `build_info.py` from `installer.iss`'s `AppVersion` (same mechanism as Windows/the old AppImage build — see [[project_help_menu_feature.md]] for why this exists).
4. `pyinstaller --onedir --windowed --name SpectraTools main.py`.

**Packaging stage** (two separate wrapping steps around the *same* build output — no rebuilding):
- **`.rpm`**: built natively via `rpmbuild` on the AlmaLinux-8 host. `Requires:` lists the RPM column above.
- **`.deb`**: built via `dpkg-deb` on the existing Ubuntu-24.04 WSL host, packaging the AlmaLinux-8-built binary (copied over, not rebuilt). `Depends:` lists the Debian column above.
- Both install to `/opt/spectratools/` (standard convention for vendor-distributed apps outside the distro's own repos), with a launcher symlink at `/usr/bin/spectratools` and the existing `packaging/linux/spectratools.desktop` + icon (already built for the AppImage attempt, reused as-is) placed in the standard `/usr/share/applications/` and `/usr/share/icons/` locations for desktop-menu integration.

**Version**: both packages' own version metadata reads from `installer.iss`'s `AppVersion`, matching the existing convention — filenames `spectratools_2.0.0_amd64.deb` / `spectratools-2.0.0-1.x86_64.rpm` (adjusting to whatever `installer.iss` says at build time, not hardcoded).

**Replaces**: `packaging/linux/build.sh` is rewritten to produce these two packages instead of an AppImage. `packaging/linux/tools/` (the old `appimagetool` download) and any AppImage-specific logic are removed. `releases/vX.Y.Z/` gets a `.deb` and `.rpm` instead of a `.AppImage`.

## Install Instructions (to ship alongside the release, e.g. a short `INSTALL.md` or release notes section)

- **Debian/Ubuntu**: `sudo apt install ./spectratools_2.0.0_amd64.deb` (or double-click in a file manager with a software-installer GUI). Uninstall: `sudo apt remove spectratools`.
- **RHEL/CentOS/AlmaLinux/Fedora**: first ensure EPEL is enabled (`sudo dnf install epel-release` — skip if already enabled, e.g. via a distro that already includes it), then `sudo dnf install ./spectratools-2.0.0-1.x86_64.rpm`. Uninstall: `sudo dnf remove spectratools`.

## Testing / Verification Requirements

- Build succeeds on AlmaLinux-8, `rpmbuild`/`dpkg-deb` both produce valid packages (`rpm -qip`/`dpkg -I` sanity checks).
- **Actually install** each package on a real system (not just build it) and confirm: dependency resolution pulls in the declared libs correctly, the app launches from the desktop menu entry and from the `/usr/bin/spectratools` symlink, `apt remove`/`dnf remove` cleanly uninstalls.
- Re-run the same "construct QApplication, run the real binary, confirm it stays alive (not a fast clean exit)" check used during this investigation, on a *clean* container/VM of each target distro family if practical — the build-machine testing so far has been on the same machine used to resolve the dependencies, which risks missing a dependency that happened to already be present for unrelated reasons.

## Out of Scope

- Any distro other than Debian/Ubuntu-family (`.deb`) and RHEL/Fedora-family (`.rpm`) — e.g. no Arch/`pacman`, no openSUSE/`zypper` package, no Flatpak/Snap.
- A GUI installer/wizard — native package manager UX (CLI or the distro's own software-center GUI opening the file) is the interface, nothing custom-built.
- Auto-detecting *which* package format to offer — the user downloads the right file for their distro, same as virtually all cross-distro Linux software distribution.
- Code-signing/repository hosting (a real `apt`/`dnf` repo with GPG-signed metadata) — these are locally-built, manually-distributed package files, not a hosted repository.
