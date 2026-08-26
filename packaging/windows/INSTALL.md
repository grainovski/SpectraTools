# Installing SpectraTools on Windows

Requires **Windows 10 or later** — Qt6, which this app is built on, does not
support Windows 7 or 8. There are no other prerequisites: the Python runtime
and every library the app needs are bundled inside the installer.

## Windows will warn you that the publisher is unknown

It will, and the warning is accurate. Expect a blue **"Windows protected
your PC"** dialog naming an unknown publisher, and possibly a similar
warning from your browser while downloading.

**This is not a virus alert.** SmartScreen shows it for *any* program that
is not code-signed, regardless of what the program does. SpectraTools is
not signed: a code-signing certificate costs a few hundred euros a year and
must be tied to a verified legal identity, which has not been worth it for
a research tool distributed to a handful of machines. Nothing about the
build is unusual — it is PyInstaller output wrapped in an Inno Setup
installer, both entirely standard.

To proceed: click **More info**, then **Run anyway**.

If your browser refuses to keep the download at all, or the file arrives
blocked, right-click the file → **Properties** → tick **Unblock** at the
bottom → **OK**.

### Verify the download instead of trusting the dialog

The warning tells you nothing about whether the file is genuine, so do the
check it cannot do. Every release ships a `SHA256SUMS.txt` listing the
exact checksum of each file. In PowerShell, in the folder you downloaded to:

```powershell
Get-FileHash -Algorithm SHA256 .\SpectraTools-v4.2.1-Setup.exe
```

Compare the printed hash with the line for that file in `SHA256SUMS.txt`.
If they match, the file is byte-for-byte the one that was built and
published — which is a far stronger guarantee than a signature dialog, and
the only one that would catch a corrupted or tampered download.

If they do **not** match, delete the file and download it again. If it still
does not match, do not run it — say so, and it will be investigated.

## Installing

Run `SpectraTools-v<version>-Setup.exe` and follow the prompts.

**You do not need administrator rights.** Setup asks up front whether to
install for everyone on the machine or just for you:

- **Install for all users** — needs an administrator prompt, and installs to
  `C:\Program Files\SpectraTools`.
- **Install for me only** — no prompt at all, and installs to
  `%LocalAppData%\Programs\SpectraTools`.

Either way you get a Start-menu entry, an optional desktop shortcut, and the
option to launch the app when Setup finishes.

To skip that question — when scripting an install, say — pass the mode on
the command line instead:

```powershell
.\SpectraTools-v4.2.1-Setup.exe /CURRENTUSER    # just me, no admin prompt
.\SpectraTools-v4.2.1-Setup.exe /ALLUSERS       # everyone, needs admin
```

### Upgrading from 4.2.0 or earlier: uninstall the old version first

From 4.2.1 onward, Setup recognises an existing SpectraTools installation
and offers to remove it before installing the new one. It cannot see
installations made by **4.2.0 or earlier**, because those registered
themselves under a different identifier — so if you simply run the new
installer over an old version, you will end up with **two** entries in
Installed apps and two Start-menu shortcuts.

Uninstall the old version through **Settings -> Apps -> Installed apps**
before installing 4.2.1. This is a one-time step; upgrades from 4.2.1
onward are detected automatically.

## The first launch is slower than the ones after it

This is expected and one-time, not a performance problem with the app.

The first time it runs, two things happen that never happen again: Windows
Defender scans the newly installed program in full, and matplotlib (the
plotting library) builds its font cache by enumerating every font on the
system. Together those can add several seconds. Every later launch skips
both.

If startup stays slow on *every* launch, that is worth reporting — it means
something other than these two.

## Uninstalling

Use **Settings -> Apps -> Installed apps -> SpectraTools -> Uninstall**, or
the **Uninstall SpectraTools** entry in the Start menu. It removes the whole
install directory.

Your own files are untouched: SpectraTools never writes into its install
directory, and spectra, fits and exports stay wherever you saved them.
Window preferences (theme, recent files) live in the registry under
`HKCU\Software\PeakFinderFitting\SpectraTools` and are left behind
deliberately, so reinstalling restores your settings.
