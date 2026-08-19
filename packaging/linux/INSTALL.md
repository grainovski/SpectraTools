# Installing SpectraTools on Linux

SpectraTools ships as native `.deb` and `.rpm` packages. Both automatically
install any missing runtime libraries as part of the same install command —
no separate dependency-hunting required.

**Minimum supported OS: RHEL/CentOS/AlmaLinux/Rocky 8 or later, or any
current Debian/Ubuntu release.** SpectraTools does **not** run on RHEL/CentOS
7 or older — this isn't a packaging limitation that could be fixed with a
different build, it's because PySide6 (the Qt6 binding this app uses) and
current numpy/scipy no longer publish wheels compatible with RHEL 7's glibc
(2.17) at all; the last PySide6 release that did was 6.2.4 in 2021. If
you're on RHEL/CentOS 7, the practical fix is upgrading that machine to a
current release (RHEL/CentOS/AlmaLinux/Rocky 8+) rather than downgrading
SpectraTools to match — RHEL 7 reached end-of-life in June 2024.

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

AlmaLinux 10 is the version this has actually been installed and run on.
AlmaLinux 9, Fedora, and other RHEL-family distributions aren't
separately tested, but are expected to work the same way — RHEL-family
and Fedora share the same core package names this app depends on. If
`dnf install` reports a missing dependency, please open an issue — it
likely means a package name has diverged.

Uninstall:
```
sudo dnf remove spectratools
```

## After installing

SpectraTools appears in your desktop's application menu (under Science), or
run `spectratools` from a terminal.

## Running under WSLg (Windows Subsystem for Linux)

SpectraTools runs on Wayland directly when WSLg provides it, and falls back
to X11 (xcb, via WSLg's XWayland) only on older WSLg builds that need it.
That choice is made automatically at startup — see below if you want to
override it.

**Dialogs that vanished and came back are fixed as of 4.1.0.** Up to 4.0.1
the app always forced X11 under WSL, to work around an older WSLg bug that
rendered the main window at zero size. Going through XWayland had a cost
that was not understood at the time: a file dialog would appear, disappear
after about a second, and return a few seconds later. The flicker happens
in XWayland's surface presentation, below the X protocol itself — tracing a
dialog's entire lifetime shows one map, one expose and one unmap, without
any of the repeated expose events a repainting compositor would produce,
which is why it looked for a long time like something with no fix. Running
on Wayland removes that layer, and the flicker with it.

Earlier versions of this file described that flicker as cosmetic, unfixable
and curable by `wsl --shutdown`. All three claims were wrong: it is fixed
in the application, and `wsl --shutdown` does not affect it.

**If the main window does not appear (only a taskbar icon).** Some older
WSLg builds send a zero-size configure event that Qt applies literally, so
the window exists but renders at 0x0. SpectraTools detects that at startup
and automatically restarts itself on X11, where it renders correctly, so
this should no longer be visible. If it ever is, launch with the platform
forced by hand:

```bash
QT_QPA_PLATFORM=xcb spectratools
```

An explicitly set `QT_QPA_PLATFORM` always wins over the automatic choice,
so the same variable can be used to force Wayland (`wayland`) for testing.

Neither behaviour has been observed on a native Linux desktop — both are
specific to WSLg's own compositor.
