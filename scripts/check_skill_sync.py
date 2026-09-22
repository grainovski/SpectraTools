"""Compare the migration skills across the copies that must stay in sync.

The three skills live in up to three places, and they are independent files
with no automation between them:

  repo      docs/skills/<name>/SKILL.md   -- the versioned master
  personal  ~/.claude/skills/<name>/      -- the copy Claude Code actually
                                             loads; edits here are what run
  drive     the move payload's skills/    -- so a BARE machine, which has
                                             neither the repo nor any skills,
                                             can still be told how to receive
                                             a migration

All three are structurally necessary, so the copies cannot be collapsed.
What was missing was any way to notice they had diverged: drift was found
only when somebody remembered to look, by hand, using a comparison most
people would get wrong.

LINE ENDINGS ARE THE TRAP. `docs/skills/` is CRLF because git checks it out
that way on Windows, while the other two copies are LF. A plain `diff`,
`cmp` or `md5sum` therefore reports every line of every skill as changed,
which reads as total drift and is entirely an artefact -- it has already
caused one false alarm. Every comparison here normalises CRLF to LF first,
and `--raw` exists only to show what that artefact looks like.

Usage:
    python scripts/check_skill_sync.py                 report, exit 1 on drift
    python scripts/check_skill_sync.py --raw           show the CRLF artefact
    python scripts/check_skill_sync.py --sync-from repo
                                                       copy repo -> the others
"""

import argparse
import hashlib
import os
import shutil
import sys

SKILLS = (
    "prepare-project-move",
    "migrate-project-first-time",
    "migrate-project-returning",
)

#: The move payload's location on this machine. Absent on a machine that has
#: never prepared a move, which is normal and not drift.
DRIVE_SKILLS = r"G:/My Drive/spectratools-move/skills"


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def locations():
    """The copies that exist here, in precedence order.

    A missing copy is reported as absent rather than as a difference: not
    every machine prepares moves, and a fresh clone has nothing installed
    yet.
    """
    found = {}
    candidates = [
        ("repo", os.path.join(repo_root(), "docs", "skills")),
        ("personal", os.path.expanduser("~/.claude/skills")),
        ("drive", DRIVE_SKILLS),
    ]
    for name, path in candidates:
        if os.path.isdir(path):
            found[name] = path
    return found


def digest(path, normalise=True):
    """sha256 of a skill file, CRLF folded to LF unless asked otherwise."""
    if not os.path.isfile(path):
        return None
    data = open(path, "rb").read()
    if normalise:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def compare(normalise=True):
    """{skill: {location: digest-or-None}} over the copies present here."""
    where = locations()
    table = {}
    for skill in SKILLS:
        table[skill] = {
            name: digest(os.path.join(base, skill, "SKILL.md"), normalise)
            for name, base in where.items()
        }
    return where, table


def drifted(table):
    """Skills whose present copies disagree, or that one copy is missing."""
    out = []
    for skill, row in table.items():
        present = [d for d in row.values() if d is not None]
        if not present:
            continue
        if len(row) != len(present) or len(set(present)) != 1:
            out.append(skill)
    return out


def report(normalise=True):
    where, table = compare(normalise)
    kind = "normalised" if normalise else "RAW (shows the CRLF artefact)"
    print("Migration skill sync -- %s\n" % kind)
    for name in ("repo", "personal", "drive"):
        state = where.get(name, "absent on this machine")
        print("  %-9s %s" % (name, state))
    print()
    names = list(where)
    print("  %-28s %s" % ("skill", "  ".join("%-10s" % n for n in names)))
    for skill, row in table.items():
        cells = "  ".join("%-10s" % ((row[n] or "missing")[:10]) for n in names)
        mark = "" if skill not in drifted(table) else "   <-- DIFFERS"
        print("  %-28s %s%s" % (skill, cells, mark))

    bad = drifted(table)
    print()
    if not where:
        print("  no copies found at all")
        return 1
    if len(where) == 1:
        print("  only one copy present -- nothing to compare")
        return 0
    if bad:
        print("  DRIFT in: %s" % ", ".join(bad))
        print("  fix with: python scripts/check_skill_sync.py --sync-from repo")
        return 1
    print("  all %d copies agree on all %d skills" % (len(where), len(SKILLS)))
    return 0


def sync(source):
    """Overwrite every other copy from `source`.

    Deliberately not automatic. Which copy is right is a judgement -- an
    edit made in `personal` to fix a misbehaving skill is just as likely to
    be the correct master as `repo` is.
    """
    where = locations()
    if source not in where:
        print("source %r is not present here (have: %s)"
              % (source, ", ".join(where) or "none"))
        return 1
    targets = [n for n in where if n != source]
    if not targets:
        print("nothing to sync to")
        return 0
    for skill in SKILLS:
        src = os.path.join(where[source], skill, "SKILL.md")
        if not os.path.isfile(src):
            print("  skip %s (absent in %s)" % (skill, source))
            continue
        for name in targets:
            dst = os.path.join(where[name], skill, "SKILL.md")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            print("  %s -> %s/%s" % (source, name, skill))
    print()
    return report()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", action="store_true",
                    help="compare without normalising, to show the artefact")
    ap.add_argument("--sync-from", choices=("repo", "personal", "drive"),
                    help="overwrite the other copies from this one")
    args = ap.parse_args()
    if args.sync_from:
        return sync(args.sync_from)
    return report(normalise=not args.raw)


if __name__ == "__main__":
    sys.exit(main())
