# How to use the migration skills

Three skills that move a project between computers without losing anything.
They work on any project, not just this one.

---

## 1. Which one do I need?

```
Am I on the machine I'm LEAVING?
    └── yes → prepare-project-move
    └── no, I'm on the machine I'm ARRIVING at
            Has this project ever been on this machine?
                └── no, it's bare            → migrate-project-first-time
                └── yes, the folder is here  → migrate-project-returning
```

If you pick wrong, the skill notices and redirects you — each one opens by
confirming the scenario. So a wrong guess costs a sentence, not a mess.

| Skill | Run on | Takes | Does |
|---|---|---|---|
| `prepare-project-move` | machine you're leaving | 10–20 min (plus test baseline) | Finds what git won't carry, archives it, writes a manifest |
| `migrate-project-first-time` | bare machine | 30–90 min | Installs prerequisites, clones, restores, builds, verifies |
| `migrate-project-returning` | machine that had it | 5–15 min | Reconciles, pulls, refreshes stale env, re-syncs memory |

---

## 2. How to invoke them

**Just describe your situation.** The skills are written to trigger on
natural phrasing — you don't need to remember their names:

> "I'm moving this project to my laptop tomorrow, what do I need to bring?"

> "Setting up SpectraTools on the new machine, it's got nothing on it yet"

> "I'm back on the desktop, haven't touched it here in three weeks"

**Or name it explicitly** if you want to be certain:

> "Use prepare-project-move on this project"

Either works. If a skill doesn't fire when you expected it to, tell me the
phrasing you used — the trigger description can be tuned.

---

## 3. What I do vs. what you do

The skills are written so I do everything that can be automated and hand
you a precise list of what only a human can do.

**I handle:** all git inspection and reconciliation, finding untracked and
ignored files, checking what's recoverable from a release host, scanning for
credentials, detecting the stack, creating archives and checksums, writing
and verifying the manifest, installing packages non-interactively
(`winget`/`brew`/`apt`), building the environment, running the test suite
and comparing it to the baseline.

**You handle** (I'll give exact steps and tell you what "done" looks like):

- **Moving the bytes.** I can't put files on a USB stick or carry them
  between machines. If a cloud-sync folder exists, I'll offer to copy the
  archives there.
- **Installers needing admin/UAC or a GUI wizard.**
- **Anything with a login, licence key, or 2FA.**
- **Device-code auth** — `gh auth login` and similar print a code you must
  enter at a URL.
- **Deciding what's precious.** I can tell you what's recoverable; only you
  know which files matter. This comes as a question with real paths and
  sizes filled in.

---

## 4. A complete move, end to end

### On the old machine

> "I'm moving this project to another computer — prepare it"

I'll push anything outstanding (so git carries it rather than you), inventory
what git won't carry, ask you to confirm the precious/disposable split, scan
for credentials, run the test suite to record a baseline, then produce:

```
reference-sources.tar.gz     88 MB   ← needed for work, not for tests
sample-data.tar.gz           61 MB   ← manual testing
releases-local-only.tar.gz  389 MB   ← IRREPLACEABLE, no remote copy
agent-memory.tar.gz         490 KB   ← never travels by git
MIGRATION-MANIFEST.json       4 KB   ← the receiving machine verifies against this
```

Copy those to your drive/stick.

### On the new machine

> "Set this project up here, I've got the archives"

I'll install prerequisites, clone, verify the archive checksums, restore,
build the environment, and run the suite — then compare the result to the
manifest's baseline and tell you whether it genuinely matches.

### Coming back later

> "I'm back on the desktop, catch it up"

I'll check for unpushed work **before** pulling, pull, refresh the
environment, verify the test count, and re-sync memory.

---

## 5. The manifest, and why it's the important part

`prepare-project-move` writes `MIGRATION-MANIFEST.json` alongside the
archives. `docs/skills/EXAMPLE-MANIFEST.json` is a real one generated from
this project. It records the stack, toolchain, per-archive checksums, what's
irreplaceable, what must *not* be copied — and the **baseline test result**.

That baseline earns its place. The worst migration bug isn't a loud failure;
it's this:

> A test suite skips whole files when an optional dependency is missing.
> The run stays green — over a **smaller** suite. It looks exactly like
> success.

This project hit it precisely: `uproot` was missing, 30 ROOT tests silently
vanished, the suite collected **1099** instead of **1129** and reported all
passed. The Windows build was quietly broken too. Nothing errored.

Comparing the *count* against a recorded baseline turns that from invisible
into obvious. Which is why the receiving skills are told to read the number,
not the colour.

---

## 6. Traps the skills are built to avoid

| Trap | What the skills do |
|---|---|
| Pulling before checking for unpushed work | Returning skill inspects **first**, reconciles, then pulls |
| Copying a virtualenv | Never copied — it embeds absolute paths to the old machine. Rebuilt instead |
| Assuming "gitignored" = "disposable" | Ignored files are reviewed, not skipped; and checked for whether the build regenerates them |
| Trusting that a release exists | Verified per asset (name, size, upload state) — a release can be missing files |
| `git add -A` | Flagged; `git add -u` recommended for projects with large untracked assets |
| Credentials swept into a cloud-bound archive | Untracked files scanned for `.env`, keys, tokens before anything is packaged |
| A green-but-smaller test run | Count compared against the recorded baseline |
| Stale assistant memory | Merged rather than overwritten, and restored to the **new** path |

---

## 7. Installing them on another machine

The skills live in your personal skills directory, which is outside any
repo — so they don't travel by git on their own. The copies in
`docs/skills/` are the versioned master.

```bash
cp -r docs/skills/prepare-project-move        ~/.claude/skills/
cp -r docs/skills/migrate-project-first-time  ~/.claude/skills/
cp -r docs/skills/migrate-project-returning   ~/.claude/skills/
```

On Windows: `%USERPROFILE%\.claude\skills\`. Copy rather than symlink, and
re-copy after editing here.

To confirm they registered, ask: *"what skills do you have for moving a
project between computers?"*

---

## 8. What's actually been tested

Honest status, so you know how much to trust them:

**Tested for real** — `prepare-project-move` was run end-to-end against this
project. Git inspection, untracked/ignored inventory, credential scan, stack
detection and manifest generation all worked on real data and produced
`EXAMPLE-MANIFEST.json`. Three defects were found and fixed:

1. Sizing every ignored entry with `du` **hung for 5 minutes** on the 818 MB
   `.venv`. The skill now says to list first and size selectively.
2. The stack table leaned on CI config; this project has none. The skill now
   names the fallbacks (build scripts, setup docs, dependency comments).
3. Nothing told it to check whether the build *regenerates* an ignored file.
   It flagged `assets/` as potentially build-critical when the build scripts
   auto-create it. The skill now says to read the code inside the guard.

**Partially tested** — the two receiving skills' inspection commands were all
run and verified against this repo, but a genuine end-to-end run needs a
second machine. Their logic is sound; their execution is unproven.

**Not done** — no formal eval suite. These are well-grounded and
partly-exercised, not empirically tuned. If one misbehaves, say so and it
gets fixed.
