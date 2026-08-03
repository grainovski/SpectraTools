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

## Compatibility Target

The user's explicit requirement: the packages must work not just on the RHEL 8-vintage baseline chosen to fix the original glibc bug, but on **current** RHEL-family releases too (user's own example: AlmaLinux 10.2). glibc's forward-compatibility guarantee means an EL8-built binary *runs* on EL10 without rebuilding — but that says nothing about whether EL10 still ships the same runtime library **packages** the RPM declares as `Requires:`. RHEL-family distros do reorganize packaging between major versions, so this was verified empirically rather than assumed (see below) — and it surfaced one real, non-obvious cross-version packaging difference.

## Verified Before Writing This Spec (not assumed)

Installed a real `AlmaLinux-8` WSL distro and actually built the app there, end to end, resolving each failure empirically rather than guessing. Then, separately, installed a clean `AlmaLinux-10` (10.2 "Lavender Lion", glibc 2.39) WSL distro and verified the EL8-built binary against it too — both the package names (`dnf list available` / `dnf provides`) and an actual run of the real built binary (not just a dependency-listing check).

- **Python**: AlmaLinux 8's base repos don't include `python3` by default; `python39` (Python 3.9.25) is available via the AppStream module and is what the build uses. `pip install "PySide6>=6.6,<6.9"` succeeds with no glibc-related error at all — confirming the *wheel itself* isn't tied to a newer glibc than 2.28 (this was the single biggest open risk in this design; it's now closed).
- **Build tooling**: `binutils` (provides `objdump`) must be installed before running PyInstaller on this base — PyInstaller's binary-dependency analysis needs it and doesn't fail loudly if it's missing versus just producing a broken build silently, so this must be an explicit step.
- **Runtime dependencies**: resolved one at a time by actually trying to construct a `QApplication` and installing whatever `ImportError` named, then rebuilt the full app with PyInstaller and cross-checked its own dependency-scan warnings, then **ran the actual built binary** (not just built it) to confirm it stays alive (`timeout -k 1 6` returning exit 124 — killed while still running — rather than a suspiciously-immediate exit 0, which the [[reference_wsl_invocation_from_git_bash.md]] memory already flagged as a false-negative trap). This exact check (real binary, `timeout -k`, exit 124 = genuinely alive) was repeated on the clean AlmaLinux-10.2 install too, after installing the packages below there — same result, no missing-library errors, only two cosmetic warnings (a fontconfig config-file quirk, a harmless Qt offscreen-plugin notice). Final verified mapping, both distro families, cross-checked EL8 vs. EL10:

| Missing `.so` | RPM package (AlmaLinux/RHEL 8 **and** 10 — see note) | Debian package (Ubuntu/Debian) |
|---|---|---|
| `libGL.so.1` | `mesa-libGL` | `libgl1` |
| `libEGL.so.1` | `mesa-libEGL` | `libegl1` |
| `libfontconfig.so.1` | `fontconfig` | `fontconfig` |
| `libxcb-image.so.0` | `xcb-util-image` | `libxcb-image0` |
| `libxcb-cursor.so.0` | `xcb-util-cursor` **(needs EPEL on EL8 only — see below)** | `libxcb-cursor0` |
| `libxkbcommon-x11.so.0` | `libxkbcommon-x11` | `libxkbcommon-x11-0` |
| `libxcb-icccm.so.4` | `xcb-util-wm` | `libxcb-icccm4` |
| `libxcb-keysyms.so.1` | `xcb-util-keysyms` | `libxcb-keysyms1` |
| `libxcb-render-util.so.0` | `xcb-util-renderutil` | `libxcb-render-util0` |
| `libatomic.so.1` | `libatomic` | `libatomic1` |
| `libcrypt.so.1` (Python 3.9's own shared lib) | **not a fixed package name — see below** | `libcrypt1` |

All 10 package names in the first block of rows were confirmed **identical** on both AlmaLinux 8 and AlmaLinux 10 (`dnf list available <name>` succeeds on both, and the end-to-end binary run on EL10 above found nothing missing after installing them under these same names) — RHEL-family desktop-library naming turned out to be far more stable across major versions than the packaging split below suggests it might be.

- **`libcrypt.so.1` — a genuine EL8-vs-EL10 packaging split, not just a naming difference**: PyInstaller bundles Python 3.9's own shared library (`libpython3.9.so.1.0`), which links against `libcrypt.so.1`. On AlmaLinux 8, the **default** `libxcrypt` package (present on any base install) provides `libcrypt.so.1` directly — confirmed via `rpm -ql libxcrypt` and `ldconfig -p`. On AlmaLinux 10, the default `libxcrypt` package only provides the *new* `libcrypt.so.2` SONAME; the old `.so.1` compat library moved to a separate `libxcrypt-compat` package, confirmed via `rpm -ql libxcrypt-compat` showing `/usr/lib64/libcrypt.so.1`. Critically, **`libxcrypt-compat` does not exist at all in AlmaLinux 8's repos** (`dnf install libxcrypt-compat` on EL8 fails with "No match for argument") — so a single RPM `Requires: libxcrypt-compat` line would install fine on EL10 but make the package **uninstallable** on EL8, which is exactly the kind of cross-version breakage this compatibility pass exists to catch. The fix is to depend on the **SONAME capability**, not a package name: `Requires: libcrypt.so.1()(64bit)`. This is RPM's native mechanism for "I need something that provides this shared library, whatever it's called on this system" — `dnf`/`rpm` resolve it to `libxcrypt` on EL8 and to `libxcrypt-compat` on EL10 automatically, with no distro-conditional spec-file logic needed. (Debian's `libcrypt1` package name, by contrast, was stable — no equivalent split found there.)
- **EPEL**: on AlmaLinux 8, `xcb-util-cursor` is not in the base/AppStream repos, only in EPEL (`dnf install epel-release`). On AlmaLinux 10, the same package **is** directly in the base AppStream repo — no EPEL needed (confirmed via `dnf provides "*/libxcb-cursor.so.0"` listing a non-EPEL AppStream package on EL10). So EPEL is an EL8-specific prerequisite, not a blanket RHEL-family one; the install instructions call this out precisely rather than telling EL10 users to add a repo they don't need. (EL9 wasn't separately tested — this spec doesn't assume its behavior either way.)
- Debian-side names cross-checked directly against `apt-cache show`/`dpkg -S` on the real Ubuntu 24.04 WSL image (not assumed from memory) — confirmed `libgl1`/`libegl1` are correct (not `libglvnd0`, a different, lower-level package that was an easy wrong guess). `libcrypt1` (added to this table after the EL10 pass surfaced the dependency) was checked the same way: `dpkg -S libcrypt.so.1` on Ubuntu 24.04 confirms `libcrypt1:amd64` owns it.

## Architecture

**Build stage** (on `AlmaLinux-8` via WSL):
1. Install `python39`, `python39-devel`, `python39-pip`, `binutils`, plus the runtime libs above (needed on the *build* machine too, since PyInstaller's dependency scanner needs to actually resolve them to embed correct references — though the shipped binary relies on the *target* machine having them via the declared package dependencies, not by bundling them).
2. `python3.9 -m venv`, `pip install -r requirements-dev.txt`.
3. Stamp `build_info.py` from `installer.iss`'s `AppVersion` (same mechanism as Windows/the old AppImage build — see [[project_help_menu_feature.md]] for why this exists).
4. `pyinstaller --onedir --windowed --name SpectraTools main.py`.

**Packaging stage** (two separate wrapping steps around the *same* build output — no rebuilding):
- **`.rpm`**: built natively via `rpmbuild` on the AlmaLinux-8 host. `Requires:` lists the 10 fixed RPM package names above, **plus** `Requires: libcrypt.so.1()(64bit)` as a SONAME capability (not a package name) for the reason documented above — this is the one dependency that can't be written as a single package name valid on both EL8 and EL10. The spec file also sets `AutoReqProv: no`: `rpmbuild`'s automatic dependency scanner otherwise walks every file under `%files`, including PyInstaller's bundled `_internal/*.so` tree, and can auto-generate spurious extra `Requires:`/`Provides:` from the app's own vendored libraries — a well-known friction point when RPM-packaging PyInstaller onedir bundles. Verified directly (a throwaway test package) that manually-declared `Requires:` lines, including the SONAME-capability one, still work normally with `AutoReqProv: no` set — so this closes the risk outright rather than leaving it to discover during the first real build.
- **`.deb`**: built via `dpkg-deb` on the existing Ubuntu-24.04 WSL host, packaging the AlmaLinux-8-built binary (copied over, not rebuilt). `Depends:` lists the Debian column above (including `libcrypt1`, which — unlike its RPM counterpart — was a single stable package name with no cross-version split found).
- Both install to `/opt/spectratools/` (standard convention for vendor-distributed apps outside the distro's own repos), with a launcher symlink at `/usr/bin/spectratools` and the existing `packaging/linux/spectratools.desktop` + icon (already built for the AppImage attempt, reused as-is) placed in the standard `/usr/share/applications/` and `/usr/share/icons/` locations for desktop-menu integration.

**Version**: both packages' own version metadata reads from `installer.iss`'s `AppVersion`, matching the existing convention — filenames `spectratools_2.0.0_amd64.deb` / `spectratools-2.0.0-1.x86_64.rpm` (adjusting to whatever `installer.iss` says at build time, not hardcoded).

**Replaces**: `packaging/linux/build.sh` is rewritten to produce these two packages instead of an AppImage. `packaging/linux/tools/` (the old `appimagetool` download) and any AppImage-specific logic are removed. `releases/vX.Y.Z/` gets a `.deb` and `.rpm` instead of a `.AppImage`.

## Install Instructions (to ship alongside the release, e.g. a short `INSTALL.md` or release notes section)

- **Debian/Ubuntu**: `sudo apt install ./spectratools_2.0.0_amd64.deb` (or double-click in a file manager with a software-installer GUI). Uninstall: `sudo apt remove spectratools`.
- **RHEL/CentOS/AlmaLinux 8**: first enable EPEL (`sudo dnf install epel-release`), then `sudo dnf install ./spectratools-2.0.0-1.x86_64.rpm`. Uninstall: `sudo dnf remove spectratools`.
- **RHEL/CentOS/AlmaLinux 9/10 and current Fedora**: `sudo dnf install ./spectratools-2.0.0-1.x86_64.rpm` — no EPEL needed (verified directly on AlmaLinux 10.2; EL9 not separately tested but the package in question ships in AppStream from EL10 onward, and RHEL doesn't typically un-promote a package from AppStream back to EPEL-only on the next major version, so EL9 is expected, not just assumed blind, to behave like EL10 here). Uninstall: `sudo dnf remove spectratools`.

## Testing / Verification Requirements

- Build succeeds on AlmaLinux-8, `rpmbuild`/`dpkg-deb` both produce valid packages (`rpm -qip`/`dpkg -I` sanity checks).
- **Actually install** each package on a real system (not just build it) and confirm: dependency resolution pulls in the declared libs correctly, the app launches from the desktop menu entry and from the `/usr/bin/spectratools` symlink, `apt remove`/`dnf remove` cleanly uninstalls.
- Re-run the same "construct QApplication, run the real binary, confirm it stays alive (not a fast clean exit)" check used during this investigation, on a *clean* container/VM of each target distro family if practical — the build-machine testing so far has been on the same machine used to resolve the dependencies, which risks missing a dependency that happened to already be present for unrelated reasons.
- **Done, pre-implementation**: this cross-version check was run on a clean AlmaLinux-10.2 WSL install (separate from the AlmaLinux-8 build machine) — installed only the declared packages (no build tooling), copied over the EL8-built binary, ran it under `timeout -k`, confirmed exit 124 (genuinely alive) with no missing-library errors. This is what surfaced the `libcrypt.so.1`/`libxcrypt-compat` split above; **the implementation must still verify the same on the actual packaged `.rpm`/`.deb` themselves** (installed via `dnf`/`apt`, not via manually-copied files + manually-installed packages) since that's the first point where the SONAME-capability `Requires:` line is exercised for real.

## Out of Scope

- Any distro other than Debian/Ubuntu-family (`.deb`) and RHEL/Fedora-family (`.rpm`) — e.g. no Arch/`pacman`, no openSUSE/`zypper` package, no Flatpak/Snap.
- A GUI installer/wizard — native package manager UX (CLI or the distro's own software-center GUI opening the file) is the interface, nothing custom-built.
- Auto-detecting *which* package format to offer — the user downloads the right file for their distro, same as virtually all cross-distro Linux software distribution.
- Code-signing/repository hosting (a real `apt`/`dnf` repo with GPG-signed metadata) — these are locally-built, manually-distributed package files, not a hosted repository.
