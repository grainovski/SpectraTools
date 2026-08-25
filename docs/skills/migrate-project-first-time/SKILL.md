---
name: migrate-project-first-time
description: Set up a project on a computer that has never had it before - install prerequisites, clone, restore the untracked assets git cannot carry, build the environment, and verify against a recorded baseline. Use this whenever the user is setting up a project on a new or freshly reimaged machine, has just bought or been given a computer, is onboarding to a codebase for the first time on this box, asks why a fresh clone does not work or is missing files, or has migration archives and a manifest they need restored. If the machine has had this project before, use migrate-project-returning instead - it is far shorter.
---

# Setting up a project on a machine that has never had it

Run this on the **receiving** machine when it is starting from nothing: no
clone, no dependencies, no toolchain.

If the machine has had this project before — the directory is still there,
with its untracked assets and reference data — stop and use
`migrate-project-returning`. That path is much shorter and deliberately
avoids re-copying things that are already present.

If a `MIGRATION-MANIFEST.json` came with the archives (written by
`prepare-project-move`), read it first. It records the stack, the toolchain,
what each archive restores, and — most usefully — the **baseline test
result** to verify against at the end.

## The shape of the problem

A clone gives a working checkout of everything tracked. What it cannot give:
untracked assets, gitignored-but-irreplaceable artifacts, state living
outside the repo (agent memory, credentials), and the external toolchain.

Two things are worth establishing early, because they change how much work
this is:

- **Can the test suite run from a bare clone?** Often yes — test fixtures
  are usually tracked. If so, the untracked assets matter for *doing work*,
  not for *verifying the setup*, and you can verify long before the archives
  finish copying. Check rather than assume: see whether anything under the
  test directory actually reads the untracked paths, or whether they are
  only mentioned in comments.
- **Is there a remote at all?** If the manifest says no remote, the
  repository itself is part of the hand-carried payload and there is nothing
  to clone.

## Procedure

### 1. Confirm the scenario and survey the machine

Confirm the project is genuinely absent. Then find out what is already
installed rather than blindly installing:

```bash
git --version
# plus the language runtime the manifest names, e.g.
python --version   /   node --version   /   cargo --version   /   go version
```

On Windows check `winget`, on macOS `brew`, on Linux the native package
manager. Knowing which exists decides whether you can install things
yourself or must hand the user a download link.

### 2. Install prerequisites — do everything you can

Install non-interactively where the platform allows:

```bash
winget install --id Git.Git -e                    # Windows
brew install git                                   # macOS
sudo apt-get install -y git                        # Debian/Ubuntu
```

Reopen the shell afterwards so `PATH` updates, and re-check the version
rather than assuming the install worked.

**Hand back what you genuinely cannot do**, with precise instructions
rather than vague pointers. Typically:

- Installers requiring UAC/admin elevation or a GUI wizard
- Anything needing an account login, licence key, or 2FA
- Device-code authentication flows (`gh auth login`, cloud CLIs) — these
  need a human to open a URL and type a code
- Interactive first-run prompts (some WSL distros, some database installers)

For each, say exactly what to click or type, and what "done" looks like so
they can tell you when to resume.

### 3. Get the code

```bash
git clone <remote-url> <target-dir>
cd <target-dir>
```

If the manifest records a commit or tag, check that the clone matches, and
mention if `HEAD` sits ahead of the newest tag — a clone of the default
branch is not necessarily the last released state.

If there is no remote, restore the repository from the hand-carried copy
instead, then confirm it is intact (`git fsck --connectivity-only`,
`git log -1`).

### 4. Restore the archives, and verify the transfer

```bash
sha256sum <each archive>        # compare against the manifest
tar -xzf project-untracked-assets.tar.gz -C <repo root>
tar -xzf project-local-only.tar.gz -C <repo root>
```

Checksums matter here: a truncated copy off a USB stick or an interrupted
cloud sync produces files that unpack partially and fail much later, in a
way that looks like a code problem rather than a transfer problem.

After unpacking, warn the user about a specific hazard if the restored
files are *untracked rather than gitignored*: `git add -A` will sweep all of
them into the index. Recommend `git add -u` as the habit for such a
project, since it stages only already-tracked files. This is not
hypothetical — it is a common and annoying mistake to undo.

### 5. Build the environment

Use the manifest's `install_cmd` if present; otherwise derive it from the
detected manifest file:

| Stack | Typical setup |
|---|---|
| Python | `python -m venv .venv` then the venv's pip with `requirements*.txt` / `pyproject.toml` / `poetry install` |
| Node | `npm ci` (lockfile present) or `npm install`; `yarn`/`pnpm` per lockfile |
| Rust | `cargo fetch` / `cargo build` |
| Go | `go mod download` |
| Ruby / Java / PHP / .NET | `bundle install` / `mvn`-`gradle` / `composer install` / `dotnet restore` |

Never copy a virtualenv or equivalent from the old machine — they embed
absolute paths to the previous interpreter and fail confusingly. Build fresh.

Respect pinned versions (`.tool-versions`, `.nvmrc`, `.python-version`) and
any deliberate upper bounds in the dependency file; those bounds usually
exist because someone was bitten.

### 6. Verify against the baseline

Run the project's test suite (or its build), and compare to the manifest's
recorded baseline:

```bash
<test_cmd>
```

- **Matches the baseline** — done, the environment is genuinely complete.
- **Green but with FEWER tests than the baseline** — this is the trap the
  baseline exists to catch. Suites commonly skip whole files when an
  optional dependency is missing, producing a passing run over a smaller
  suite that is indistinguishable from success if you only look at the
  colour. Find the missing dependency and re-run the install step.
- **Failures** — investigate before declaring victory. Distinguish
  environment problems from real breakage: a machine-specific failure
  (timing thresholds, absolute paths, GPU assumptions) is not the same as
  broken code.

If the suite takes a long time, say so up front and use a single fast test
file for quick iteration while the full run happens.

Then actually launch the application or entry point once. Tests passing and
the thing running are different claims.

### 7. Restore state that lives outside the repository

Agent/assistant memory typically lives in a directory named after the
project's **absolute path** with separators replaced by dashes — so the
directory name changes when the project moves. Unpack into the path the
project now occupies, not the name it had on the old machine.

If the paths cannot match, put the folder anywhere and tell the assistant
directly: *"Read the notes in `<path>` before we start — this project has
history there."*

### 8. Optional: the release/build toolchain

Only if this machine will build or publish artifacts, not merely develop.
The manifest's `toolchain` list and the CI config are the best sources for
what is required. Expect this part to need the most human involvement —
GUI installers, VM or container images, signing certificates, and
authentication flows all tend to land here.

## Closing report

Tell the user plainly:

- What now works (verified how — baseline matched, app launched)
- What you installed
- What still needs them, with exact steps
- Anything you could not verify and why

Resist declaring success on the basis of a green test run alone if the
count was lower than the baseline; say what you actually observed.
