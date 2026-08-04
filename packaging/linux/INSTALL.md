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
