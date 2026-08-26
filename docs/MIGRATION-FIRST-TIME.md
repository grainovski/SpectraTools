# Migrating to a computer the project has never been on

Use this when the target machine has **nothing**: no clone, no reference
sources, no sample data, no toolchain.

If the machine has had this project before — the repo directory,
`tv-1.9.13/`, `srcRW/`, `libmfile-1.0.7/` and the spectra/matrices are
already sitting there — use **`MIGRATION-RETURNING.md`** instead. It is a
much shorter path and it deliberately avoids re-copying the ~149 MB you
already have.

`SETUP.md` is the reference for *why* each environment step exists and
carries the deeper notes. This document is the *procedure*: what to package
on the source machine, and the order to do things on the target.

---

## The one thing to understand first

Git carries more than you might expect, and less than you need.

| | Carried by git? | Consequence |
|---|---|---|
| Application code, tests, packaging scripts, docs | **Yes** | A clone is a working, testable checkout |
| `tests/fixtures/` (60 MB, incl. both real matrices) | **Yes** | The full 1129-test suite runs from a bare clone with zero hand-carried files |
| Reference C sources (`tv-1.9.13/`, `srcRW/`, `libmfile-1.0.7/`) | No | No test needs them; **TV/gf3 parity work is impossible without them** |
| Sample spectra/matrices in the repo root | No | Only for exercising the app by hand |
| `releases/` | No | `v1.0.0` and `v2.0.0` exist **nowhere else** — see Step 1c |
| Claude's project memory | No | Lives under `~/.claude`, outside the repo entirely |

Verified: the four test files that mention `tv-1.9.13`/`libmfile`/`srcRW`
reference them only in comments and source citations — there is no
filesystem access to those paths anywhere in `tests/`.

---

## Step 1 — On the SOURCE machine: package what git cannot carry

Run these from the repository root.

### 1a. Reference C sources (~88 MB)

The authority for every TV/gf3 parity decision in `peak_fit.py`,
`matrix_cut.py` and `calibration.py`. The code cites specific files and line
numbers (`vsFitInt.c:303`, `vsCurFit.c`, `gf3_subs.c`). Without them you
cannot check a parity claim, only trust it.

```bash
tar -czf reference-sources.tar.gz tv-1.9.13 srcRW libmfile-1.0.7
```

(Add `hdtv` to that list if the machine has it — it arrived during the
v4.0.0 ROOT work and is not present on every machine.)

### 1b. Sample spectra and matrices (~61 MB)

Measurement data for driving the app by hand. Not needed by the suite.

```bash
tar -czf sample-data.tar.gz *.n42 *.spk *.spe *.lzs *.mtx *.root demo.txt test.txt test1.txt
```

### 1c. Released artifacts — check before assuming they are disposable

`releases/` is gitignored and local-only. **GitHub Releases only began at
v2.1.0**, so `releases/v1.0.0/` and `releases/v2.0.0/` have no remote copy
anywhere. Everything from v2.1.0 onward can be re-downloaded with
`gh release download`.

```bash
tar -czf releases-local-only.tar.gz releases/v1.0.0 releases/v2.0.0
```

Never blanket-delete `releases/` on the source machine after copying — verify
first, asset by asset. A published release existing is **not** proof its
assets are complete: v3.0.0 was found in 2026-08 to be missing three of its
five assets, including the Windows installer.

### 1d. Claude's project memory (~490 KB, ~53 notes)

The reasoning behind decisions that are not derivable from the code — TV
parity choices, WSLg behaviour, release-process traps, per-version history.
It lives at a path derived from the project's own location:

```
~/.claude/projects/<project-path-with-separators-as-dashes>/memory/
```

On the source machine that is, for example,
`C:\Users\RIG\.claude\projects\C--Users-RIG-Documents-Claude-PeakFinderFitting\memory\`.

```bash
tar -czf claude-memory.tar.gz -C ~/.claude/projects/<mangled-path> memory
```

### 1e. Confirm nothing is stranded

```bash
git status --short | grep '^??'   # everything untracked -- is it in an archive above?
git log --oneline origin/master..master   # unpushed commits? push them first
```

---

## Step 2 — On the TARGET machine: install prerequisites

Windows 11 ships `winget`. Reopen the terminal after each install so `PATH`
updates.

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.13 -e
```

Verify:

```bash
git --version
python --version      # 3.13.x
```

If you use the GUI Python installer instead, tick **"Add python.exe to
PATH"** on the first screen — everything below depends on it.

---

## Step 3 — Clone, and restore what git did not carry

```bash
git clone https://github.com/grainovski/SpectraTools.git PeakFinderFitting
cd PeakFinderFitting
```

Unpack the archives from Step 1 into the repository root:

```bash
tar -xzf /path/to/reference-sources.tar.gz
tar -xzf /path/to/sample-data.tar.gz
tar -xzf /path/to/releases-local-only.tar.gz   # if you brought it
```

> **`git add -A` must never be used in this repository.** These files are
> *untracked, not gitignored*, so `-A` sweeps ~149 MB of third-party C source
> and measurement data straight into the index. This has actually happened and
> was caught on the commit's own `--stat`. Use **`git add -u`**, which stages
> only already-tracked files — the correct tool here, since every real change
> is to a file git already knows about.

Check what you have against the last release:

```bash
git log --oneline $(git describe --tags --abbrev=0)..master
```

Anything listed is on `master` but in no released package. `master` is
deliberately allowed to sit ahead of the newest tag.

---

## Step 4 — Python environment

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

`requirements.txt` has deliberate upper bounds on PySide6 (`<6.9`) and
matplotlib (`<3.12`), each with a comment explaining why. Do not raise either
without re-reading that comment.

---

## Step 5 — Verify

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expect **1129 passed, 0 failed** (as of v4.1.2).

Two things worth knowing before you start it:

- **It takes hours, not minutes.** Just over three hours on the machine last
  measured, peaking near 5 GB — the matrix fixtures decode into 536 MB
  arrays. While iterating, run one file (`-q tests/test_mtx_io.py`, seconds).
- **A green run does not prove the whole suite ran.** `test_root_io.py` and
  `test_root_ui.py` begin with `pytest.importorskip("uproot")`. A virtualenv
  missing `uproot` silently collects **1099** instead of 1129 and still
  reports all-passed. If you see 1099, that is the cause; re-run Step 4.
  The Windows build needs `uproot` too — `build.ps1` passes
  `--collect-all awkward_cpp`, which fails outright without it.

Then run the app:

```bash
.venv/Scripts/python.exe main.py
```

Open a sample spectrum, mark a fit, and check Help → Knowledge Database
renders.

---

## Step 6 — Claude's project memory

Unpack into the directory matching **the new machine's** project path, not
the old one — the directory name encodes the absolute path with separators
replaced by dashes, so it changes when the project moves:

```
~/.claude/projects/C--Users-<you>-Documents-Claude-PeakFinderFitting/memory/
```

If the path cannot match, put the folder anywhere and tell the session
directly: *"Read the memory files in `<path>` before we start — this project
has history there."*

---

## Step 7 — Release toolchain

Only needed if this machine will **build installers or publish releases**.
Skip for development-only use.

### Windows

```powershell
winget install --id JRSoftware.InnoSetup -e
```

`packaging/windows/build.ps1` looks for `ISCC.exe` in
`%LOCALAPPDATA%\Programs\Inno Setup 6` first, then the Program Files
locations.

```bash
powershell -File packaging/windows/build.ps1
```

Do **not** redirect its stderr (`2>&1`): Windows PowerShell 5.1 turns
PyInstaller's ordinary INFO output into a fatal `NativeCommandError`.

### Linux — three WSL distros

| Distro | Role |
|---|---|
| **AlmaLinux-8** | Builds the RPM *and* the onedir the DEB consumes. Its glibc 2.28 sets the compatibility floor — this is why the app cannot run on RHEL/CentOS 7. |
| **Ubuntu-24.04** | Builds the DEB (`dpkg-deb` is not available on RHEL-family). |
| **AlmaLinux-10** | Smoke-test target. **Always test here, never only on the build host.** |

```powershell
wsl --install AlmaLinux-8  --no-launch
wsl --install AlmaLinux-10 --no-launch
wsl --install Ubuntu-24.04 --no-launch
```

`--no-launch` skips the interactive first-run account prompt entirely; the
build scripts run as root anyway.

Build in this order — the DEB build consumes the onedir the RPM build
produces:

```bash
wsl -d AlmaLinux-8  -u root -- bash -lc 'cd /mnt/c/<path>/PeakFinderFitting && bash packaging/linux/build.sh'
wsl -d Ubuntu-24.04 -u root -- bash -lc 'cd /mnt/c/<path>/PeakFinderFitting && bash packaging/linux/build_deb.sh'
```

`-u root` is required: the build scripts call `dnf`/`apt` without `sudo`.

**A freshly installed Ubuntu has empty apt lists**, so installing the DEB
fails with a misleading `Depends: libxcb-cursor0 but it is not installable`.
The dependency is fine; apt just has no index. Run `apt-get update` once.

### GitHub CLI

```powershell
winget install --id GitHub.cli -e --scope user
gh auth login --hostname github.com --git-protocol https --web
```

The device-code step **requires a human** to enter a code at
github.com/login/device. Two mechanical notes: pipe `gh`'s output to a
*file*, not through `grep`/`tr` (a pipe buffers the code so it never
appears), and feed it `printf '\n' |` for the "press Enter" prompt.

---

## Known gotchas on any new machine

**The app's window never appears under WSL** — a taskbar icon shows up but
clicking or Alt+Tab does nothing. This is a WSLg compositor bug, not an app
fault, and it has recurred across many releases. Fix from a **Windows**
terminal (never from inside WSL):

```powershell
wsl --shutdown
```

Then relaunch. To confirm objectively rather than by eye, check the Windows
side for the window: a working launch shows a process named `msrdc` with the
title `SpectraTools (<distro>)`.

**The version lives in exactly one place** — `packaging/windows/installer.iss`'s
`AppVersion`. Every platform reads it from there and stamps `build_info.py`,
which is generated and gitignored.
