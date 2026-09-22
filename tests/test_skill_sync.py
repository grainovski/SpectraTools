"""The migration skills must not drift between their copies.

They live in up to three independent places with no automation between them
-- the repo's `docs/skills/` master, the personal `~/.claude/skills/` copy
Claude Code actually loads, and the move payload's copy that a bare machine
receives. Editing one and forgetting the others is silent: nothing errors,
nothing looks wrong, and the next machine move follows whichever copy it
happens to find.

Until now the only defence was remembering to check by hand. This turns
that into something the suite says out loud.

Machines that have only the repo copy -- a fresh clone, or anything that has
never prepared a move -- skip rather than fail. A missing copy is not drift.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

check_skill_sync = pytest.importorskip("check_skill_sync")


def test_the_skill_copies_agree():
    """The check that matters. Compares CRLF-normalised, because a raw
    comparison reports every line of every skill as changed on Windows and
    has already produced one false alarm."""
    where, table = check_skill_sync.compare(normalise=True)
    if len(where) < 2:
        pytest.skip("only %d copy present (%s); nothing to compare"
                    % (len(where), ", ".join(where) or "none"))

    bad = check_skill_sync.drifted(table)
    assert not bad, (
        "the migration skills have drifted between copies: %s\n"
        "copies present: %s\n"
        "Fix with: python scripts/check_skill_sync.py --sync-from <copy>\n"
        "Decide WHICH copy is right first -- an edit made in the personal "
        "copy to fix a misbehaving skill is as likely to be the master as "
        "the repo one."
        % (", ".join(bad), ", ".join("%s=%s" % kv for kv in where.items())))


def test_every_skill_is_present_in_the_repo_master():
    """The repo copy is the versioned master, so it must be complete even
    when nothing else is installed. A skill that exists only in the personal
    copy is not backed up and does not travel."""
    root = check_skill_sync.repo_root()
    for skill in check_skill_sync.SKILLS:
        path = os.path.join(root, "docs", "skills", skill, "SKILL.md")
        assert os.path.isfile(path), "%s is missing from the repo master" % skill


def test_the_comparison_can_detect_a_difference(tmp_path):
    """Control. test_the_skill_copies_agree passes on every machine where
    the copies happen to match, which is most of them -- so it is only worth
    having if the comparison underneath it reacts to a real difference."""
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_bytes(b"one\ntwo\n")
    b.write_bytes(b"one\nTWO\n")
    assert check_skill_sync.digest(str(a)) != check_skill_sync.digest(str(b))

    # and drifted() must actually flag it
    table = {"x": {"repo": "aaa", "personal": "bbb"}}
    assert check_skill_sync.drifted(table) == ["x"]
    table = {"x": {"repo": "aaa", "personal": "aaa"}}
    assert check_skill_sync.drifted(table) == []


def test_normalisation_is_what_makes_the_check_usable(tmp_path):
    """The other half of the control, and the reason this file exists at all.

    A CRLF copy and an LF copy of the SAME skill must compare equal. Without
    this, the check above fails on every Windows checkout for a reason that
    has nothing to do with drift -- which is exactly the false alarm the
    2026-09-20 sync check produced.
    """
    crlf = tmp_path / "crlf.md"
    lf = tmp_path / "lf.md"
    crlf.write_bytes(b"---\r\nname: x\r\n---\r\nbody\r\n")
    lf.write_bytes(b"---\nname: x\n---\nbody\n")

    assert check_skill_sync.digest(str(crlf)) == check_skill_sync.digest(str(lf)), (
        "CRLF and LF copies of the same skill compare as different; the "
        "sync check would report drift on every Windows checkout")
    assert (check_skill_sync.digest(str(crlf), normalise=False)
            != check_skill_sync.digest(str(lf), normalise=False)), (
        "the raw comparison no longer distinguishes them, so this test is "
        "not demonstrating what normalisation is for")


def test_a_missing_copy_is_not_reported_as_drift():
    """A machine that has never prepared a move has no Drive copy, and a
    fresh clone has nothing installed. Neither is drift, and reporting it as
    such would train everyone to ignore the check."""
    assert check_skill_sync.drifted({"x": {"repo": "aaa"}}) == []
    assert check_skill_sync.drifted({"x": {}}) == []
    # but a copy that is present and EMPTY is drift, not absence
    assert check_skill_sync.drifted({"x": {"repo": "aaa", "personal": None}}) == ["x"]
