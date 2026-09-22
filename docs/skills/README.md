# Project-migration skills

Three Claude Code skills for moving a project between computers. They are
**general** — nothing in them is specific to SpectraTools — and are kept
here so they are versioned, backed up, and travel to another machine with
the repository. Skills themselves live outside any repo, so a personal-only
copy is exactly the kind of thing these skills exist to stop you losing.

| Skill | Run it on | Job |
|---|---|---|
| `prepare-project-move` | the machine you are **leaving** | Find everything git will not carry, archive it, write a manifest |
| `migrate-project-first-time` | a **bare** target machine | Install prerequisites, clone, restore, build, verify |
| `migrate-project-returning` | a target that **has had** the project | Reconcile, pull, refresh a stale environment, re-sync memory |

## How they fit together

`prepare-project-move` writes a `MIGRATION-MANIFEST.json` next to the
archives it creates, recording the detected stack, the toolchain, a
checksum per archive, what is irreplaceable, and — the part that does the
most work — the **baseline test result**.

Both receiving skills verify against that manifest. This is what turns the
worst class of migration bug into an automatic catch: a test suite that
skips whole files when an optional dependency is missing still reports
green, over a *smaller* suite. Comparing counts against a recorded baseline
makes that visible; looking at pass/fail alone does not.

See `HOWTO.md` for the full usage manual — which skill to use when, what
I do versus what you do, a complete worked move, and what has actually been
tested. `EXAMPLE-MANIFEST.json` is a real manifest generated from this
project.

## Installing

They are active when copied into the personal skills directory:

```bash
cp -r docs/skills/prepare-project-move        ~/.claude/skills/
cp -r docs/skills/migrate-project-first-time  ~/.claude/skills/
cp -r docs/skills/migrate-project-returning   ~/.claude/skills/
```

On Windows that is `%USERPROFILE%\.claude\skills\`.

Copy them rather than symlinking, and re-copy after editing here — the two
locations are independent, and this directory is the master.

## Keeping the copies in sync

There can be up to three copies, and nothing keeps them together
automatically:

| Copy | Where | Why it has to exist |
|---|---|---|
| repo | `docs/skills/` | the versioned master, backed up and travels with the project |
| personal | `~/.claude/skills/` | the copy Claude Code actually loads — edits here are what run |
| payload | the move folder's `skills/` | a **bare** machine has neither the repo nor any skills, including the one that would tell it how to receive the migration |

Check them with:

```bash
python scripts/check_skill_sync.py
```

It reports which copies exist here and whether they agree, and
`--sync-from repo` (or `personal`, or `drive`) overwrites the others from
one. Decide which copy is right before syncing — an edit made in the
personal copy to fix a misbehaving skill is as likely to be the master as
this one.

`tests/test_skill_sync.py` runs the same comparison in the suite, so drift
surfaces without anyone remembering to look. A machine with only one copy
skips rather than fails; a missing copy is not drift.

**Compare with line endings normalised.** This directory is CRLF because
git checks it out that way on Windows, while the other copies are LF. A
plain `diff`, `cmp` or `md5sum` reports *every line of every skill* as
changed, which reads as total drift and is purely an artefact — it has
already caused one false alarm. The script normalises; `--raw` exists only
to show what the artefact looks like.

## Editing

Keep the `name:` field identical to the directory name, and keep the
`description:` field specific about *when* to trigger, since that is the
only thing Claude sees when deciding whether to use a skill. Everything
about when to use a skill belongs in the description, not the body.
