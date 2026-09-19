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
