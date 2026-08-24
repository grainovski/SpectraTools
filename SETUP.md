# Setting SpectraTools up on a new machine

Cloning the repository is not enough to reach a working development
environment. Several things this project depends on are deliberately *not*
in git, and the release toolchain is external. This is the full list, in the
order it is worth doing.

Written when moving from the machine that produced v4.1.0 (2026-08-19);
refreshed after v4.1.1 (2026-08-22).

## 1. What the clone gives you

```bash
git clone https://github.com/grainovski/SpectraTools.git
cd SpectraTools
```

That brings the application, the tests, the packaging scripts, the docs, and
all 17 tags (`v1.0.0` … `v4.1.1`). The repository is the source of truth for
everything under version control.

**`master` is not always the released state.** Work is sometimes committed
without cutting a release, so the newest tag can be behind `master`. Check
before assuming a clone matches the last published build:

```bash
git log --oneline $(git describe --tags --abbrev=0)..master
```

Anything listed there is on `master` but not in any released package.

## 2. What the clone does NOT give you

Four categories, none of which git carries:

| Missing | Why it is not tracked | How to restore |
|---|---|---|
| Reference C sources (`tv-1.9.13/`, `srcRW/`, `libmfile-1.0.7/`, `hdtv/`) | Third-party upstream code, ~105 MB unpacked | copy `reference-sources.tar.gz`, or re-download from upstream |
| Sample spectra and matrices (`*.n42`, `*.spk`, `*.spe`, `*.mtx`, `*.root`, `demo.txt`, `test*.txt`) | Measurement data, not source | copy `sample-data.tar.gz` |
| Claude's project memory | Lives under `~/.claude`, outside the repo | copy `claude-memory.tar.gz` — see §6 |
| `.venv/`, `build_info.py`, `releases/` | Generated | rebuilt by the steps below |

**The reference sources matter more than their size suggests.** They are the
authority for every TV/gf3 parity decision in `peak_fit.py`,
`matrix_cut.py` and `calibration.py` — the code cites specific files and
line numbers (`vsFitInt.c`, `vsCurFit.c`, `vsFitSetup.c`, `gf3_subs.c`,
`VMatrix.cxx`). Without them you cannot check a parity claim, only trust it.
A previous machine move left them behind and that was felt immediately.

Unpack all three archives into the repository root:

```bash
tar -xzf reference-sources.tar.gz -C /path/to/SpectraTools
tar -xzf sample-data.tar.gz -C /path/to/SpectraTools
```

Note what they are not: these directories and the sample data are **merely
untracked, not `.gitignore`d**. That is exactly why **`git add -A` must never
be used in this repository** — it sweeps all ~105 MB of third-party C source
and every measurement file straight into the index. This has happened; it was
caught on the commit's own `--stat` and reset. Use **`git add -u`**, which
stages only already-tracked files, and is the correct tool here because every
change to this project is a change to a file git already knows about.

## 3. Python environment

Python 3.13 on Windows is what the last release was built with (3.13.15).

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

`requirements.txt` carries deliberate upper bounds on PySide6 (`<6.9`) and
matplotlib (`<3.12`); both have comments explaining why, and neither should
be raised without re-checking what the comment names.

Verify:

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expect **1129 passed, 0 failed** (as of v4.1.2). No test needs deselecting
any more: the decode-speed guard used to assert an absolute wall-clock bound
calibrated on one machine and failed on slower hardware with no regression
present, but it now times the decoder against a frozen copy of the
pre-optimization implementation in the same process and asserts a ratio,
which holds anywhere.

**Budget real time for it.** The Qt and matrix tests dominate: a full run
took just over three hours on the machine this was last measured on, and
the matrix fixtures decode into 536 MB arrays, so peak memory reaches ~5 GB.
Running a single file (`-q tests/test_mtx_io.py`) is seconds, so prefer that
while iterating.

**A green run does not by itself mean the whole suite ran.**
`tests/test_root_io.py` and `tests/test_root_ui.py` open with
`pytest.importorskip("uproot")`, so a virtualenv predating v4.0.0 — when
`uproot` was added to `requirements.txt` — silently collects 30 fewer tests
and still reports all-passed. If the count comes out at 1099 rather than
1129, that is this, and the fix is to re-run the install step above. Note
the Windows build needs `uproot` too: `build.ps1` passes
`--collect-all awkward_cpp`, which fails outright without it.

## 4. Release toolchain (only needed to build installers)

**Windows** — [Inno Setup 6](https://jrsoftware.org/isdl.php):

```bash
winget install --id JRSoftware.InnoSetup -e
```

`packaging/windows/build.ps1` looks for `ISCC.exe` in
`%LOCALAPPDATA%\Programs\Inno Setup 6`, then the two Program Files
locations. Build with:

```bash
powershell -File packaging/windows/build.ps1
```

Do **not** redirect its stderr (`2>&1`): Windows PowerShell 5.1 turns
PyInstaller's ordinary INFO output into a fatal `NativeCommandError`.

**Linux** — three WSL distros, in this order:

| Distro | Role |
|---|---|
| AlmaLinux 8 | builds the RPM *and* the onedir the DEB consumes; its glibc 2.28 sets the compatibility floor |
| Ubuntu 24.04 | builds the DEB (`dpkg-deb` is not available on RHEL-family) |
| AlmaLinux 10 | smoke-test target — always test here, not only on the build host |

```bash
wsl -d AlmaLinux-8  -u root -- bash -lc 'cd /mnt/c/…/SpectraTools && bash packaging/linux/build.sh'
wsl -d Ubuntu-24.04 -u root -- bash -lc 'cd /mnt/c/…/SpectraTools && bash packaging/linux/build_deb.sh'
```

`-u root` is required — the build scripts call `dnf`/`apt` without `sudo`,
and the distros do not log in as root by default.

The version lives in exactly one place, `packaging/windows/installer.iss`'s
`AppVersion`. Every platform reads it from there and stamps `build_info.py`,
which is generated and gitignored.

## 5. Released artifacts

Not in the repository (`releases/` is gitignored). Older versions are
pruned locally once verified recoverable, so only the current and previous
release are usually present. Every published build is on GitHub:

```bash
gh release download v4.1.1 --dir releases/v4.1.1
```

`gh` needs `gh auth login` on the new machine; the device-authorisation step
needs a human.

## 6. Claude's project memory

`claude-memory.tar.gz` holds ~52 notes covering the reasoning behind
decisions that are not derivable from the code: TV parity choices, WSLg
compositor behaviour, release-process traps, and per-version histories.

It belongs at a path derived from the project's own location:

```
~/.claude/projects/<mangled-project-path>/memory/
```

The directory name is the absolute project path with separators replaced by
dashes — on the previous machine,
`C--Users-grain-Documents-Claude-Projects-PeakFinderFitting`. **If the
project lands at a different path on the new machine, that directory name
changes too**, so unpack into the new path's directory rather than restoring
the old name verbatim.

## 7. Sanity check

```bash
.venv/Scripts/python.exe main.py
```

Open a sample spectrum, mark a fit, and check Help → Knowledge Database
renders. On Linux under WSL the app now selects Wayland automatically and
falls back to X11 only if the window comes up unusably small — see
`packaging/linux/INSTALL.md` for the detail.
