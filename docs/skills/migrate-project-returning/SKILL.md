---
name: migrate-project-returning
description: Resume work on a project on a machine that already has it - check for unpushed or uncommitted work BEFORE pulling, refresh a stale environment, verify the test count against the expected baseline, and re-sync assistant memory that git does not carry. Use this whenever the user returns to a second computer, switches back to another laptop or desktop, picks up a project they have not touched on this machine in a while, says a project here is out of date or behind, or asks what to do before continuing work on a machine they were away from. If the project has never been on this machine, use migrate-project-first-time instead.
---

# Resuming a project on a machine that has had it before

Run this on a machine that **already has** the project directory, its
untracked assets, and its reference data — you are returning to it after
working elsewhere.

If any of that is missing, use `migrate-project-first-time` instead.

The point of this path is what you *do not* do: you do not re-copy the
untracked assets. They do not change, copying them again wastes time, and
overwriting a newer local copy with a stale archived one is a real way to
lose work. Only two things actually need to move: **new commits** (via
git, not file copying) and **state that lives outside the repo**, which git
will never carry.

## Why the order matters

The instinct on returning is to pull immediately. Resist it. A machine you
left weeks ago may hold commits that were never pushed, edits never
committed, or a stash you forgot. Pulling first turns a simple situation
into a divergence, and divergences get "resolved" under time pressure in
ways that lose work.

So: **inspect, then reconcile, then pull.**

## Procedure

### 1. Look for work this machine holds that nowhere else does

```bash
cd <project>
git status --short | grep -v '^??'          # uncommitted tracked changes
git fetch origin
git log --oneline @{upstream}..HEAD         # local commits not pushed
git stash list                              # forgotten stashes
git branch --no-merged                      # unmerged branches
```

Interpret before acting:

- **Uncommitted changes** — commit or stash before pulling. Look at them
  first; on a machine you have been away from, they may be someone else's
  or a half-finished experiment you have since redone elsewhere.
- **Unpushed commits** — a real divergence. Push if wanted, or discard
  deliberately, but decide consciously. Do not reach for a force operation
  to make the symptom disappear.
- **Stashes** — easy to forget entirely and easy to destroy.
- **`??` untracked entries** — normally expected. These are the reference
  assets and sample data that are supposed to live here. Do not treat them
  as clutter without checking.

If anything is outstanding, surface it and let the user decide before
touching the working tree.

### 2. Pull

```bash
git status -sb        # "behind N" is the normal, healthy case
git pull
```

A fast-forward is what you want. If git reports divergence, go back to
step 1 — something here was never pushed.

Then check whether what arrived is ahead of the last release, if the
project tags releases:

```bash
git log --oneline $(git describe --tags --abbrev=0)..HEAD
```

### 3. Refresh the environment — the most likely thing to be stale

The dependency environment is not carried by git and does not update
itself. On a returning machine this is the single most common source of
confusing behaviour, because it fails *quietly*: the project still starts,
tests still pass, and only a subtle capability is missing.

Re-run the install step for the detected stack:

| Stack | Refresh |
|---|---|
| Python | venv's pip against `requirements*.txt` / `pyproject.toml` / `poetry install` |
| Node | `npm ci` (respects the lockfile) rather than `npm install` |
| Rust / Go | `cargo fetch` / `go mod download` |
| Ruby / PHP / .NET | `bundle install` / `composer install` / `dotnet restore` |

The specific failure worth naming: **a dependency added since you were last
here.** Test suites frequently skip whole files when an optional dependency
is absent, so the run stays green over a *smaller* suite. Nothing errors.
The only visible symptom is the test count.

If the environment is broken outright, or its base interpreter/runtime was
uninstalled or upgraded underneath it, delete and rebuild rather than
repairing — nothing of value lives in it.

### 4. Verify, and read the count rather than the colour

A single fast test file is usually enough to confirm the environment works:

```bash
<fast subset of the test command>
```

Run the full suite when you need the real guarantee — and if the project
has a recorded baseline (a `MIGRATION-MANIFEST.json`, a setup document, or
a note in the repo), **compare the number of tests, not just pass/fail**.
Fewer tests than expected means an incomplete environment, not a healthy
project. Go back to step 3.

If the suite is slow, say so before starting it rather than letting the user
wonder whether it has hung.

### 5. Re-sync state that lives outside the repository

Git does not carry assistant/agent memory, editor workspace state, or local
credentials. A machine you have not used for a while holds a stale copy —
and stale memory is worse than none, because it will confidently act on
conclusions that were superseded elsewhere.

Assistant memory usually lives in a directory named after the project's
absolute path with separators replaced by dashes. Copy the newer machine's
copy across. Where both machines have diverged, **merge rather than
overwrite**: individual notes are usually independent, so the union is
normally right, and only the index file needs the two versions combining.

### 6. Confirm the toolchain survived

Only relevant if this machine builds or publishes artifacts. Things rot
while you are away — VMs get removed, tokens expire, tools get upgraded:

```bash
# whatever the project needs, e.g.
gh auth status
docker ps
wsl --list --verbose
```

Re-authenticate where needed; device-code flows require the user.

Locally-pruned build artifacts (release directories and similar) are
normally expected to be absent — they are usually re-downloadable from
wherever releases are published. Check before assuming, though: an artifact
predating the project's use of a release host may exist *only* on one
machine.

## What not to do on a returning machine

- **Do not re-copy the untracked assets.** They are already here.
- **Do not `git add -A`** if this project keeps large untracked assets in
  the tree — it sweeps them into the index. `git add -u` stages only
  already-tracked files, which is what you almost always want.
- **Do not delete local-only artifacts** without confirming a copy exists
  elsewhere, asset by asset. "A release exists" is not proof that every one
  of its files was uploaded.
- **Do not assume the newest tag equals the tip of the branch.** Check.

## Closing report

Say what changed and what the user should know: how far behind the machine
was, what was pulled, what the environment refresh installed, the verified
test result *with its count*, and anything still needing them. If you found
unpushed work in step 1, lead with that — it is the most important thing
that happened.
