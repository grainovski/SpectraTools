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

## Editing

Keep the `name:` field identical to the directory name, and keep the
`description:` field specific about *when* to trigger, since that is the
only thing Claude sees when deciding whether to use a skill. Everything
about when to use a skill belongs in the description, not the body.
