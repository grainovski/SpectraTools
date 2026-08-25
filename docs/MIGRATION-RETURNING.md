# Migrating back to a computer the project has been on before

Use this when the target machine **already has**:

- the repository directory
- `tv-1.9.13/`, `srcRW/`, `libmfile-1.0.7/`
- the sample spectra and matrices in the repo root

If any of that is missing, use **`MIGRATION-FIRST-TIME.md`** instead.

The whole point of this path is that **you do not re-copy the ~149 MB of
untracked assets.** They do not change. Copying them again is wasted time and
risks overwriting a newer local copy with an older one.

What you *do* have to move is the two things git does not carry between
machines: **new commits** (via `git pull`, not file copying) and **Claude's
project memory** (by hand — it lives outside the repo).

---

## The short version

```bash
cd /path/to/PeakFinderFitting
git status --short | grep -v '^??'        # 1. any uncommitted work here?
git log --oneline origin/master..master   # 2. any unpushed commits here?
git fetch origin && git status -sb        # 3. how far behind?
git pull                                  # 4. only once 1-3 are understood
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt   # 5. refresh deps
.venv/Scripts/python.exe -m pytest -q tests/test_mtx_io.py        # 6. quick sanity
```

Then sync memory (Step 5 below), which git will never do for you.

---

## Step 1 — Check this machine for work you would otherwise destroy

**Do this before pulling.** A machine you left weeks ago may hold commits
that were never pushed, or edits never committed. This repository has had
multiple sessions running against it at once, so the local state is not
always what you remember leaving.

```bash
git status --short | grep -v '^??'        # uncommitted changes to tracked files
git log --oneline origin/master..master   # local commits not on origin
git stash list                            # forgotten stashes
git branch                                # stale local branches
```

- **Uncommitted changes** → commit or stash them before pulling.
- **Unpushed commits** → this is a divergence, not a clean pull. Push them if
  they are wanted (`git push origin master`), or discard deliberately. Do not
  paper over it with a force operation.
- **Nothing listed** → proceed.

Ignore the `??` untracked entries: those are the reference sources, sample
data and `releases/` that are supposed to be there.

---

## Step 2 — Pull

```bash
git fetch origin
git status -sb          # "behind N" is the normal case
git pull
```

If `git pull` reports a divergence rather than a fast-forward, stop and read
Step 1 again — something on this machine was never pushed.

Then see what arrived that is not yet in any release:

```bash
git log --oneline $(git describe --tags --abbrev=0)..master
```

`master` is deliberately allowed to sit ahead of the newest tag.

---

## Step 3 — Refresh the Python environment

The `.venv` is **not** carried by git and does not update itself. A venv left
over from an earlier visit is the single most likely thing to be wrong on a
returning machine.

```bash
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

If the venv is old enough to predate a dependency change, this is what fixes
it. The concrete case that has bitten before: **`uproot` was added in v4.0.0**,
and a venv created before that silently skips the 30 ROOT tests *and* breaks
the Windows build (`build.ps1` passes `--collect-all awkward_cpp`, which fails
without it).

If the venv is broken or its base interpreter has been uninstalled, delete and
rebuild it — nothing is lost:

```bash
rm -rf .venv
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

---

## Step 4 — Verify

A single file is enough for a returning machine and takes seconds:

```bash
.venv/Scripts/python.exe -m pytest -q tests/test_mtx_io.py
```

Run the full suite only when you actually need the guarantee — **it takes
just over three hours** and peaks near 5 GB:

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expect **1129 passed, 0 failed** (as of v4.1.2).

**Check the count, not just the colour.** If it says **1099**, `uproot` is
missing from the venv — the ROOT tests `importorskip` it, vanish silently, and
the run still reports all-passed. Go back to Step 3.

---

## Step 5 — Sync Claude's project memory

**Git does not carry this.** It lives outside the repository, so a machine you
have not used for a while holds a stale copy — it will confidently act on
conclusions that were superseded elsewhere.

The folder is at a path derived from the project's own location, with
separators replaced by dashes:

```
~/.claude/projects/C--Users-<you>-Documents-Claude-PeakFinderFitting/memory/
```

Copy the newer machine's `memory/` folder over. `MEMORY.md` is the index —
one line per note. If both machines have diverged, merge rather than
overwrite: the individual `.md` files are independent, so the union is
normally correct, and only `MEMORY.md` itself needs the two indexes combining.

If the paths do not match between machines, put the folder anywhere and tell
the session: *"Read the memory files in `<path>` before we start — this
project has history there."*

---

## Step 6 — Confirm the toolchain survived

Only if this machine builds installers or publishes releases.

```powershell
wsl --list --verbose        # AlmaLinux-8, AlmaLinux-10, Ubuntu-24.04
gh auth status              # token still valid?
ls "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
```

- **Missing WSL distros** → reinstall per `MIGRATION-FIRST-TIME.md` Step 7.
- **`gh` logged out** → `gh auth login --web`; the device-code step needs a
  human.
- **`releases/` mostly empty** → expected. Old versions get pruned locally
  once verified recoverable. Re-fetch any you need:
  `gh release download v4.1.2 --dir releases/v4.1.2`. The exceptions are
  `v1.0.0` and `v2.0.0`, which predate GitHub Releases and exist **only** as
  local copies — if this machine has them, do not delete them.

---

## What NOT to do on a returning machine

- **Do not re-copy** `tv-1.9.13/`, `srcRW/`, `libmfile-1.0.7/`, the spectra or
  the matrices. They are already there and do not change.
- **Do not `git add -A`.** Those same directories are *untracked, not
  gitignored*, so `-A` sweeps ~149 MB of third-party C source and measurement
  data into the index. Use **`git add -u`**.
- **Do not delete `releases/v1.0.0` or `releases/v2.0.0`.** No remote copy
  exists.
- **Do not assume the newest tag equals `master`.** Check.

---

## If the app starts but shows no window under WSL

A taskbar icon appears and clicking does nothing. This is the WSLg compositor,
not the app, and it recurs regularly. From a **Windows** terminal (not inside
WSL):

```powershell
wsl --shutdown
```

Relaunch afterwards. To verify objectively rather than by eye, a working
launch shows a Windows process named `msrdc` titled `SpectraTools (<distro>)`.
