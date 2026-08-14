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

## Known issues on WSLg (Windows Subsystem for Linux)

If you're running SpectraTools inside WSL rather than on a native Linux
desktop, WSLg's compositor has a couple of cosmetic quirks that aren't bugs
in SpectraTools itself:

- **Main window doesn't appear after launch (only a taskbar icon).** This is
  a known WSLg compositor issue, not specific to SpectraTools. Fix: from a
  Windows terminal (not from inside WSL), run `wsl --shutdown`, wait a few
  seconds, then relaunch.
- **A secondary window (a dialog, e.g. Multiply by Factor or the file-open
  dialog) briefly disappears right after opening, then reappears on its
  own.** Same underlying WSLg compositor behavior as above, just triggered
  by dialog creation instead of the main window. It's purely visual and
  self-resolves within a moment — the dialog works normally once it
  reappears. No action needed; if it becomes persistently annoying, the same
  `wsl --shutdown` fix applies.

Neither of these has been observed on a native Linux desktop (only through
WSLg), and neither has an application-side fix — both are WSLg's own
Wayland/XWayland compositor timing behavior.
