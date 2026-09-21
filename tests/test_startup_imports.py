"""What the application is allowed to import before its window appears.

Startup is dominated by first-touch disk reads, not by computation. On a
cold launch -- the first one after a boot, which is the one a user
notices -- the installed application took **12.3 s** to show a window
against **1.2 s** warm, because it read its way through the bundle.

scipy is 88 files and 48 MB of a 237 MB bundle, and nothing needs it
until a peak is fitted, a spectrum is searched or an assignment is
restored. Importing it to draw an empty plot is pure cold-start cost.

This runs in a SUBPROCESS on purpose. By the time the rest of the suite
has run, scipy is in this process's sys.modules for perfectly good
reasons, so asking about the current process would answer a different
question and always pass.
"""

import subprocess
import sys
import os
import textwrap


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Packages that must NOT be loaded by importing the application.
#: uproot and awkward are here because they were already lazy before this
#: test existed, and nothing should quietly make them eager either.
FORBIDDEN_AT_STARTUP = ("scipy", "uproot", "awkward")


def _modules_after(statement):
    """Top-level package names loaded by `statement`, in a fresh process."""
    code = (
        "import sys\n"
        "%s\n"
        "print(' '.join(sorted({m.split('.')[0] for m in sys.modules})))\n"
        % statement
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO,
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return set(out.stdout.split())


def test_starting_the_app_does_not_import_scipy():
    loaded = _modules_after("import main")
    offenders = sorted(set(FORBIDDEN_AT_STARTUP) & loaded)
    assert not offenders, (
        "importing main pulled in %s. Something gained a module-level "
        "import of it; move that import inside the function that needs it."
        % offenders
    )


def test_the_check_can_see_a_package_that_IS_loaded():
    """Control. Without it the test above would pass just as happily if
    the subprocess never imported anything, or if the module-name
    collection were broken."""
    loaded = _modules_after("import main")
    assert "numpy" in loaded, "numpy should be loaded at startup"
    assert "matplotlib" in loaded, "matplotlib draws the empty plot"


def test_scipy_is_still_reachable_when_actually_needed():
    """Deferring must not mean losing. Each of the three call paths binds
    its scipy symbol on first use."""
    loaded = _modules_after(
        "import numpy as np\n"
        "import peak_fit\n"
        "peak_fit.hypermet_step(np.array([0.0, 1.0]), 0.5, 1.0, 0.1)"
    )
    assert "scipy" in loaded, "peak_fit never bound scipy.special"


def test_restoring_an_assignment_still_reaches_scipy():
    loaded = _modules_after(
        "from energy_assignments import EnergyAssignments, restore\n"
        "restore(EnergyAssignments(None, ((100.0, 121.783),)), [(100.0, 4.0)])"
    )
    assert "scipy" in loaded, "energy_assignments never bound scipy.optimize"


def _run(code, timeout=300):
    """Run `code` in a fresh process; return (returncode, stdout, stderr)."""
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                         capture_output=True, text=True, timeout=timeout)
    return out.returncode, out.stdout, out.stderr


def test_the_warm_up_returns_at_once_and_loads_scipy_behind_the_window():
    """Deferring scipy did not remove its cost, it moved it: the first
    action needing it paid the lot. Measured, opening the Automatic
    Calibration dialog for the first time spent 1.10 s importing and 0.01 s
    building the dialog, against 0.00 s every time after.

    The warm-up starts that import on a thread once the window is already
    up, so the cost lands where nobody is waiting on it. It must return
    immediately -- doing the import inline here would simply move the stall
    back into startup, which is what v5.2.5 removed.
    """
    code = (
        "import sys, time\n"
        "import main\n"
        "assert 'scipy' not in sys.modules, 'scipy loaded before the warm-up'\n"
        "t = time.perf_counter()\n"
        "main._warm_up_scipy_modules()\n"
        "elapsed = time.perf_counter() - t\n"
        "assert elapsed < 0.5, 'warm-up blocked for %.2f s' % elapsed\n"
        "deadline = time.time() + 120\n"
        "while 'scipy' not in sys.modules and time.time() < deadline:\n"
        "    time.sleep(0.05)\n"
        "print('loaded' if 'scipy' in sys.modules else 'NEVER LOADED')\n"
    )
    rc, out, err = _run(code)
    assert rc == 0, err
    assert out.strip() == "loaded", (
        "the warm-up never imported scipy, so the first calibration still "
        "pays for it: %s" % err)


def test_a_failing_warm_up_is_silent_and_harmless():
    """It runs for speed alone, so it must never take the app down or print
    a traceback nobody can act on. The real import happens again on first
    use and reports its failure there, where the user is waiting for
    something.

    The subprocess proves the import really is broken before calling the
    warm-up. Without that, the test would pass just as happily against a
    setup that did nothing, having exercised no failure at all.

    Written as one dedented block rather than concatenated "...\n" pieces:
    backslash escapes do not survive the way this file is edited, and a
    mangled one turns into a syntax error rather than a wrong test.
    """
    code = textwrap.dedent(
        """
        import sys, time
        import main
        # Binding a module name to None is what makes `import` raise.
        sys.modules['auto_calibrate_dialog'] = None
        try:
            import auto_calibrate_dialog
        except ImportError:
            pass
        else:
            print('SETUP INEFFECTIVE: the import did not raise')
            raise SystemExit(2)
        main._warm_up_scipy_modules()
        time.sleep(2.0)
        print('survived')
        """
    )
    rc, out, err = _run(code)
    assert rc != 2, "the test never broke the import: %s" % out
    assert rc == 0, "a warm-up failure took the process down: %s" % err
    assert out.strip() == "survived"
    assert "Traceback" not in err, (
        "the warm-up printed a traceback the user cannot act on: %s" % err)
