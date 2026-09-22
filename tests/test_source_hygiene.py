"""Source checks that nothing else in this project performs.

No linter is configured and pyflakes is not in the virtualenv, so the kind
of drift a linter would catch is invisible here: it breaks no test, changes
no behaviour, and accumulates. The 2026-09-21 audit found eight unused
imports that had built up exactly that way.

Written as a test rather than by adding a linter on purpose. The checks
below are the ones that have actually bitten this repository, they need no
new dependency, and they run in the suite everyone already runs.
"""

import ast
import io
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Directories that are not ours to tidy: the virtualenv, build output, and
#: the reference source trees (tv, hdtv, Radware, libmfile) kept beside the
#: project for parity work.
_SKIP = {".venv", ".git", "__pycache__", "build", "dist", "releases",
         "hdtv", "tv-1.9.13", "srcRW", "libmfile-1.0.7", "packaging",
         "node_modules"}


def _python_files():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP]
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(base, name)


def _unused_imports(source):
    """Imported names that appear nowhere else in `source`.

    Deliberately crude, and deliberately crude in the SAFE direction: a name
    counts as used the moment it appears anywhere else in the text, so a
    string, an annotation, a doctest or an ``__all__`` entry all exempt it.
    That under-reports rather than over-reports, which is what a guard in a
    test suite wants -- a false positive here would block a commit over
    nothing.

    A line carrying ``# noqa`` is exempt, for imports kept on purpose for
    their side effect. main.py imports auto_calibrate_dialog that way, to
    warm scipy on a background thread.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [(a.asname or a.name).split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [a.asname or a.name for a in node.names
                     if a.name != "*"]
        else:
            continue
        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        if "noqa" in line:
            continue
        for name in names:
            found.setdefault(name, node.lineno)
    return {name: line for name, line in found.items()
            if source.count(name) <= 1}


def test_no_unused_imports():
    """Eight of these had accumulated by the 2026-09-21 audit, across seven
    test files. Nothing failed because of them, which is the point: without
    a check they only ever grow."""
    offenders = []
    for path in _python_files():
        source = io.open(path, encoding="utf-8").read()
        try:
            unused = _unused_imports(source)
        except SyntaxError:                     # not ours to police
            continue
        for name, lineno in sorted(unused.items(), key=lambda kv: kv[1]):
            offenders.append("%s:%d imports %r and never uses it"
                             % (os.path.relpath(path, ROOT), lineno, name))
    assert not offenders, (
        "%d unused import(s):\n  %s\n\nRemove them, or add '# noqa' with a "
        "reason if the import is there for its side effect."
        % (len(offenders), "\n  ".join(offenders)))


def test_the_unused_import_check_can_fail():
    """Control. The scan above walks a few hundred files and passes; that
    is only meaningful if it would have caught the eight it was written
    for."""
    bad = "import math\nimport os\n\nprint(os.getcwd())\n"
    assert _unused_imports(bad) == {"math": 1}, (
        "the scan did not flag an obviously unused import")


def test_the_unused_import_check_respects_noqa():
    """Side-effect imports are legitimate and must stay exempt, or the check
    above forces someone to delete main.py's scipy warm-up."""
    kept = "import auto_calibrate_dialog  # noqa: F401 - pulls in scipy\n"
    assert _unused_imports(kept) == {}, (
        "a '# noqa' import was flagged; the warm-up import would have to go")


def test_the_unused_import_check_does_not_cry_wolf():
    """The other half of the control: a name used only in an annotation, a
    docstring or a string literal must NOT be reported. A guard that flags
    working code gets deleted rather than obeyed."""
    used_in_annotation = "import typing\n\n\ndef f(x: typing.Any):\n    return x\n"
    assert _unused_imports(used_in_annotation) == {}
    used_in_string = 'import json\n\nDOC = "json is used here"\n'
    assert _unused_imports(used_in_string) == {}


@pytest.mark.parametrize("path", ["efficiency.py", "main_window.py",
                                  "help_content.py"])
def test_the_scan_actually_reaches_the_app(path):
    """Control for the walk itself, not the parser. If _SKIP or the walk
    ever stopped reaching the application, test_no_unused_imports would
    pass over an empty set and look perfectly healthy."""
    reached = {os.path.relpath(p, ROOT).replace("\\", "/")
               for p in _python_files()}
    assert path in reached, (
        "%s is not being scanned; the walk covers %d files"
        % (path, len(reached)))
