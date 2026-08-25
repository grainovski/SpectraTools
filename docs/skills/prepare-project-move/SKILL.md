---
name: prepare-project-move
description: Package a project for a move to another computer - find everything git will NOT carry (untracked assets, gitignored-but-irreplaceable files, agent memory, credentials), archive it, and write a manifest the receiving machine verifies against. Use this whenever the user is about to move, copy, migrate, transfer, back up, or hand off a project to a different machine, is setting up a second computer, mentions putting a project on a USB stick or cloud drive, says they are switching laptops, or asks what they need to bring along. Run this on the machine being LEFT, before anything is copied.
---

# Preparing a project to move to another computer

Run this on the machine you are **leaving**. It is the highest-stakes of the
three migration skills, because it is the only one where loss is
irreversible: on the receiving machine a missing file is an annoyance you
fix by going back, but if the source machine is wiped, reimaged or handed
on, anything you failed to package is gone.

The companion skills are `migrate-project-first-time` (bare receiving
machine) and `migrate-project-returning` (a machine that has had this
project before). Both read the manifest this skill writes.

## What you are actually looking for

Git is an excellent transport for everything it tracks, and useless for
everything else. The job here is finding the "everything else", which
falls into four categories that people forget in roughly this order:

1. **Untracked-but-precious files.** Sample data, fixtures too large or too
   private to commit, vendored reference sources, scratch notes. These sit
   in `git status` as `??` and are invisible to a clone.
2. **Gitignored-but-irreplaceable files.** The dangerous category, because
   `.gitignore` is usually a signal that something is *disposable* — build
   output, caches, virtualenvs. But the same file can hold artifacts that
   exist nowhere else: released binaries, signed installers, generated
   datasets that took hours to produce.
3. **State outside the repository entirely.** Agent/assistant memory,
   editor workspace settings, local database contents, credentials.
4. **Uncommitted or unpushed work.** Not "packaging" at all — just push it,
   and the problem disappears.

## Procedure

### 1. Confirm the scenario, then get git out of the way

Confirm this is the machine being left. If the user is actually *arriving*
somewhere, redirect them to `migrate-project-first-time` or
`migrate-project-returning`.

The cheapest win is making git carry as much as possible. Check for work
that would otherwise have to be hand-carried:

```bash
git status --short | grep -v '^??'          # uncommitted tracked changes
git log --oneline @{upstream}..HEAD         # committed but unpushed
git stash list                              # forgotten stashes
git branch --no-merged                      # unmerged local branches
git remote -v                               # is there even a remote?
```

Anything committed and pushed needs no archive. Offer to commit and push
what is outstanding — but do not push on the user's behalf without saying
so, and never force-push. If the project has **no remote at all**, say so
plainly: the entire repository including its history is then part of what
must be physically copied, which changes the plan.

### 2. Inventory what git will not carry

```bash
git status --short | grep '^??'                        # untracked
git status --short --ignored=traditional | grep '^!!'  # ignored, directory-level
```

**Get the list before you get the sizes.** Measuring every entry with
`du -sh` in the same pass is the obvious move and it will hang: ignored
trees routinely include a virtualenv or `node_modules` with tens of
thousands of files, and one `du` over that can take minutes. Size only the
handful of entries you are actually deciding about, and put a `timeout` on
it. This is a real failure, not a theoretical one — it wedged a five-minute
command the first time this skill was run for real.

Then classify. Do not guess — size and name are weak signals, and this is
the step where being wrong is expensive. Four questions resolve almost
everything:

- **Is there another copy anywhere?** A release binary that is also on a
  GitHub/GitLab release page is recoverable; the same file with no remote
  copy is not. Check with `gh release list` / `gh release view --json assets`
  or the equivalent, and verify per-asset (name, size, upload state) rather
  than trusting that a release exists.
- **Does the build regenerate it?** Before archiving anything ignored, grep
  the build scripts for it. A file that looks build-critical is often
  produced on demand — an icon, a generated header, a version stamp — and
  the script may quietly create it when absent. Read the code inside the
  guard, not just the guard: `if [ ! -f X ]` followed by a command that
  *makes* X means the project self-heals and there is nothing to carry.
- **Can it be regenerated, and at what cost?** A virtualenv rebuilds in
  minutes and should never be copied — it usually contains absolute paths to
  the old machine and will silently misbehave if moved. A dataset that took
  six hours of computation is a different matter.
- **Would losing it block work, or just annoy?** Vendored third-party
  sources that the code cites for correctness decisions block real work.

Present the classification and let the user correct it, using
AskUserQuestion with the actual paths and sizes filled in. They know which
files are precious; you know which are recoverable. Neither of you knows
both.

### 3. Check for secrets before archiving anything

Untracked files are exactly where credentials live. Scan before packaging:

```bash
git status --short | grep '^??'   # look for .env, *.pem, *.key, id_*, credentials*, *.p12
```

If any appear, surface them and let the user decide. Do not silently sweep
credentials into an archive that is about to cross a cloud drive or a USB
stick — and if they do want them moved, say plainly that an unencrypted
archive on shared storage is a poor place for them.

### 4. Detect the stack, so the receiving machine can be told what to install

Read, do not assume. The receiving skills need this, and CI config is often
the most honest description of what a project actually requires:

| Look for | Tells you |
|---|---|
| `requirements*.txt`, `pyproject.toml`, `Pipfile`, `poetry.lock`, `environment.yml` | Python, and which installer |
| `package.json` + `package-lock.json` / `yarn.lock` / `pnpm-lock.yaml` | Node, and which package manager |
| `Cargo.toml`, `go.mod`, `Gemfile`, `pom.xml`, `build.gradle*`, `composer.json`, `*.csproj`, `mix.exs`, `pubspec.yaml` | Rust / Go / Ruby / Java / PHP / .NET / Elixir / Dart |
| `.github/workflows/*.yml`, `.gitlab-ci.yml`, `Dockerfile`, `Makefile` | The real toolchain, versions, and test command |
| `.tool-versions`, `.nvmrc`, `.python-version`, `runtime.txt` | Pinned language versions |

CI config is the most honest description of a project's requirements when it
exists — but plenty of real projects have none, especially ones built and
released from a developer's own machine. When there is no CI, the same
information lives in the build and packaging scripts (`packaging/`,
`scripts/`, `build.*`), in any setup or contributing document, and in the
comments around pinned dependency versions, which usually explain *why* a
bound exists. Read those instead rather than reporting the toolchain as
unknown.

### 5. Record a baseline the receiving machine can check itself against

This is the step that makes the whole system work, so do not skip it even
though it is slow.

Run the project's test suite (or its build, if there are no tests) and
record the exact result — the count, not just pass/fail. On the receiving
machine, a suite that silently collects *fewer* tests than this baseline is
the classic symptom of a half-installed environment: optional dependencies
that tests skip on when missing produce a green run over a smaller suite,
which looks identical to success. A recorded baseline count turns that
invisible failure into an obvious one.

Note the wall-clock time too. If the suite takes hours, the receiving
machine needs to know that before assuming it has hung.

### 6. Create the archives and the manifest

Archive by category, so the receiving side can restore selectively:

```bash
tar -czf project-untracked-assets.tar.gz <the precious untracked paths>
tar -czf project-local-only.tar.gz <irreplaceable gitignored paths>
tar -czf agent-memory.tar.gz -C <memory parent dir> <memory dir>
```

Agent memory, if the project has any, usually lives outside the repo under a
directory named after the project's absolute path with separators replaced
by dashes. It does **not** travel by git, and a stale copy on the far end
will confidently act on superseded conclusions. Find it rather than assuming
the path.

Then write `MIGRATION-MANIFEST.json` next to the archives:

```json
{
  "project": "<name>",
  "source_machine": "<hostname>",
  "source_path": "<absolute path on this machine>",
  "created": "<ISO date>",
  "git": {
    "remote": "<url or null>",
    "commit": "<sha>",
    "branch": "<name>",
    "all_pushed": true,
    "newest_tag": "<tag or null>"
  },
  "stack": {"language": "...", "manifest": "...", "install_cmd": "...", "test_cmd": "..."},
  "baseline": {"test_cmd": "...", "result": "1129 passed, 0 failed", "duration": "~3h07m"},
  "archives": [
    {"file": "project-untracked-assets.tar.gz", "sha256": "...", "bytes": 0,
     "restores_to": "repo root", "contents": ["..."], "why_needed": "..."}
  ],
  "irreplaceable": ["paths with no remote copy anywhere - losing these is permanent"],
  "toolchain": ["external tools the project needs that are not language packages"],
  "known_gotchas": ["environment-specific traps worth carrying forward"]
}
```

Checksum every archive (`sha256sum`) so the far end can prove the transfer
was clean rather than hoping. Write them to a `SHA256SUMS.txt` in the same
format `sha256sum -c` expects, so verifying on arrival is one command
rather than a manual comparison.

Compression is worth a moment's thought on the big archive: installers,
media and other already-compressed payloads gain nothing from gzip and can
add minutes. Plain `tar` for those, and name the file `.tar` so it is
obvious.

### 7. Ship the skills themselves, and write a human entry point

Two things are easy to forget precisely because you have them and the
receiving machine does not.

**The skills do not travel.** They live in the user's profile
(`~/.claude/skills/`), not in the project, so a new computer has none of
them — including the one that would tell it how to receive this migration.
Copy the migration skills into the handoff folder alongside the archives.
Without this, the archives arrive with nothing that knows what to do
with them.

**The manifest is machine-readable, not human-readable.** Write a short
`START-HERE.md` next to it, addressed to a person sitting at the receiving
machine with no context. It should cover: what each file in the folder is
and how big; that installing the skills is step zero, with the exact target
path for their OS and a way to confirm it worked; how to tell which
scenario they are in; **the literal sentence to type**, including the path
to the handoff folder, since that is the one thing the assistant cannot
guess; what the assistant will do versus what it will ask of them; the
expected test result and what a smaller number means; and a by-hand
fallback if no assistant is available.

Both of these came out of running this skill for real: the archives were
correct and complete, and would still have arrived at a machine with no
skills installed and no plain-language instructions.

### 8. Hand off

State clearly what you did and what only the user can do. You cannot move
bytes between machines; they can.

Report: each archive with its size and one-line purpose, the total to
transfer, anything flagged as irreplaceable, and any credentials you
deliberately did not archive. Then give the concrete next step — copy this
folder to the USB stick / cloud drive, and on the other machine open
`START-HERE.md` and follow it.

Verify the copy at its destination rather than trusting that it worked:
re-run `sha256sum -c SHA256SUMS.txt` there, and open each archive
(`tar -tzf`) to confirm it is readable with the expected number of entries.
A truncated cloud-sync or USB write passes a size check and fails much
later, in a way that looks like a project problem rather than a transfer
one.

If the destination is a cloud folder, say plainly that the files exist on
local disk immediately but are not retrievable from another machine until
the sync client reports finished.

If a cloud-sync folder is present on the machine, offer to copy the archives
there rather than making them do it by hand.

## What good looks like

The user ends up with a small set of named archives, a manifest, and a clear
statement of what is irreplaceable. Nothing precious is left behind, no
credentials leave the machine unnoticed, and the receiving machine can
verify what it got instead of trusting it.
