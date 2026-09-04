import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from html.parser import HTMLParser

from PySide6.QtGui import QDesktopServices

import help_content
from help_content import (
    build_about_html,
    build_howto_html,
    build_knowledge_database_html,
    open_help_page,
)

_VOID_ELEMENTS = {"meta", "br", "img", "hr", "link", "input"}
_BASE64_IMAGE = re.compile(r"data:image/png;base64,[A-Za-z0-9+/=]+")


class _TagBalanceChecker(HTMLParser):
    """Stack-based well-formedness check -- every non-void start tag must
    be closed by the matching end tag, in order, with nothing left open
    at EOF. Standard-library only, no new test dependency."""

    def __init__(self):
        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag not in _VOID_ELEMENTS:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack, f"unexpected closing </{tag}> with nothing open"
        assert self.stack[-1] == tag, f"expected </{self.stack[-1]}>, got </{tag}>"
        self.stack.pop()


def _strip_base64_images(html):
    """Removes embedded PNG data URIs before doing substring checks on
    prose content -- a base64 blob is effectively random text, so a
    short/common search term (e.g. "net") can coincidentally appear
    inside one by chance, letting a check pass even if every real prose
    occurrence of that term was deleted."""
    return _BASE64_IMAGE.sub("", html)


def test_howto_html_contains_every_shortcut():
    html = build_howto_html()
    # Wrapped in <kbd>...</kbd> rather than a bare substring check: several
    # of these shortcuts are literal prefixes of others (e.g. "Ctrl+S" is
    # a prefix of "Ctrl+Shift+C"), so a bare `shortcut in html` check can
    # pass even if the shorter shortcut's own row was deleted entirely.
    for shortcut in [
        "Ctrl+O", "Ctrl+Shift+O", "Ctrl+S", "Ctrl+W", "Ctrl+Q",
        "Ctrl+G", "Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+D",
        "Ctrl+L", "Ctrl+T", "Ctrl+M", "Ctrl+R", "Ctrl+N", "Ctrl+A", "Ctrl+Shift+A",
        "Ctrl+=", "Ctrl+-", "Ctrl+0",
        "Ctrl+F", "Ctrl+B", "Ctrl+C", "Ctrl+Shift+C", "Ctrl+E", "Ctrl+I",
        "Ctrl+Alt+C",
        "B", "R", "P", "F1", "F5",
    ]:
        assert f"<kbd>{shortcut}</kbd>" in html, f"missing shortcut {shortcut!r}"


def test_howto_html_covers_every_operation():
    html = build_howto_html()
    for topic in [
        "Loading a spectrum",
        "Working with multiple spectra",
        "Saving a spectrum",
        "Calibrating the energy axis",
        "Multiply",
        "Rebin",
        "Normalize",
        "Add and Subtract Spectra",
        "Performing a fit",
        "Integration",
        "Saving and exporting",
        "Saving and reloading fits",
        "Calibrating from fitted peaks",
        "View options",
        "Knowledge Database",
    ]:
        assert topic in html, f"missing topic {topic!r}"


def test_howto_html_documents_export_formats():
    html = build_howto_html()
    for term in ["_fits.jsonl", "Export All Fits", "_fits_report.txt", "Export This Fit"]:
        assert term in html, f"missing term {term!r}"


def test_howto_integration_section_mentions_optional_background():
    html = build_howto_html()
    assert "Background regions are optional" in html


def test_howto_ctrl_c_description_says_hide_not_delete():
    html = build_howto_html()
    # Shortcuts table: each shortcut's own exact row text, so a partial
    # revert of just one row can't hide behind the other's wording.
    assert "Clear in-progress marks and hide committed fits (not delete)" in html
    assert "Permanently delete the active spectrum's committed fits (in-progress marks untouched)" in html
    # Prose paragraph: the same two claims, worded differently there --
    # both locations must independently say the right thing.
    assert "hides its already-committed fits" in html
    assert "Ctrl+Shift+C</kbd> permanently deletes the active spectrum's" in html


def test_howto_html_documents_ctrl_b_preview():
    html = build_howto_html()
    assert "Preview the background fit from the two background regions alone" in html
    assert "no fit region or peaks needed" in html


def test_howto_html_documents_fix_checkbox():
    html = build_howto_html()
    # "Fix" doesn't otherwise appear anywhere in the HowTo page, so these
    # don't risk false-passing on unrelated text -- specific multi-word
    # substrings, matching this file's established convention, so a
    # partial revert can't hide behind generic wording either.
    assert 'Check a row\'s "Fix" box to hold that parameter' in html
    assert "as that parameter's starting guess" in html


def test_howto_html_is_a_complete_html_document():
    html = build_howto_html()
    assert html.strip().startswith("<!doctype html>")
    assert "<title>SpectraTools -- HowTo</title>" in html
    for tag in ("<html>", "<head>", "<body>"):
        assert tag in html, f"missing {tag}"


def test_howto_html_has_balanced_tags():
    checker = _TagBalanceChecker()
    checker.feed(build_howto_html())
    assert checker.stack == [], f"unclosed tags at EOF: {checker.stack}"


def test_knowledge_database_html_contains_parameter_names():
    html = _strip_base64_images(build_knowledge_database_html())
    for term in [
        "position", "FWHM", "amplitude", "tail fraction",
        "tail beta", "background slope", "Volume", "full", "net",
    ]:
        assert term in html, f"missing term {term!r}"


def test_knowledge_database_html_documents_fix_checkbox_zero_uncertainty():
    html = _strip_base64_images(build_knowledge_database_html())
    assert 'checked "Fix" in the Fit Parameters panel' in html
    assert "uncertainty is reported as exactly zero" in html


def test_knowledge_database_html_embeds_seven_figures():
    html = build_knowledge_database_html()
    assert html.count("data:image/png;base64,") == 7


def test_knowledge_database_html_has_no_external_links():
    html = build_knowledge_database_html()
    assert "http://" not in html
    assert "https://" not in html


def test_knowledge_database_html_is_a_complete_html_document():
    html = build_knowledge_database_html()
    assert html.strip().startswith("<!doctype html>")
    assert "<title>SpectraTools -- Knowledge Database</title>" in html


def test_knowledge_database_html_has_balanced_tags():
    checker = _TagBalanceChecker()
    checker.feed(build_knowledge_database_html())
    assert checker.stack == [], f"unclosed tags at EOF: {checker.stack}"


def test_about_html_shows_dev_fallback_when_build_info_is_absent(monkeypatch):
    # Setting the sys.modules entry to None forces the next `import
    # build_info` to raise ImportError immediately, regardless of
    # whether a real build_info.py happens to exist on disk from a
    # prior local `packaging/windows/build.ps1` run -- this is the
    # standard way to deterministically simulate "module not
    # importable" without touching the filesystem.
    monkeypatch.setitem(sys.modules, "build_info", None)
    html = build_about_html()
    assert "dev" in html
    assert "development build" in html


def test_about_html_shows_stamped_version_and_date_when_build_info_exists(monkeypatch):
    fake_module = types.ModuleType("build_info")
    fake_module.VERSION = "1.2.3"
    fake_module.BUILD_DATE = "2026-08-01"
    monkeypatch.setitem(sys.modules, "build_info", fake_module)
    html = build_about_html()
    assert "1.2.3" in html
    assert "2026-08-01" in html


def test_about_html_shows_dev_fallback_when_build_info_is_missing_attributes(monkeypatch):
    # A module that imports fine but lacks VERSION/BUILD_DATE (e.g. a
    # stale build_info.py from an earlier schema) must fall back the
    # same as a missing module, not raise AttributeError.
    fake_module = types.ModuleType("build_info")
    monkeypatch.setitem(sys.modules, "build_info", fake_module)
    html = build_about_html()
    assert "dev" in html
    assert "development build" in html


def test_about_html_contains_program_name_and_copyright():
    html = build_about_html()
    assert "SpectraTools" in html
    assert "Georgi Rainovski" in html


def test_about_html_is_a_complete_html_document():
    html = build_about_html()
    assert html.strip().startswith("<!doctype html>")
    assert "<title>About SpectraTools</title>" in html


def test_open_help_page_writes_a_temp_file_and_opens_it(qapp, monkeypatch):
    # Windows/macOS path: QDesktopServices.openUrl() is used directly.
    # Must return True, not just record the call -- a falsy return (e.g.
    # bare list.append()'s implicit None) would be indistinguishable
    # from a real failure.
    monkeypatch.setattr(help_content.sys, "platform", "win32")
    opened = []

    def _open_url(url):
        opened.append(url)
        return True

    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(_open_url))
    open_help_page("<html><body>hello</body></html>")
    assert len(opened) == 1
    path = opened[0].toLocalFile()
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert f.read() == "<html><body>hello</body></html>"


def test_open_help_page_returns_true_on_success(qapp, monkeypatch):
    monkeypatch.setattr(help_content.sys, "platform", "win32")
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: True))
    assert open_help_page("<html></html>") is True


def test_open_help_page_returns_false_when_no_browser_handler(qapp, monkeypatch):
    # QDesktopServices.openUrl itself returns False (not an exception)
    # when there's no registered handler for the URL -- e.g. no default
    # browser configured on Windows/macOS. Must be surfaced, not
    # silently dropped.
    monkeypatch.setattr(help_content.sys, "platform", "win32")
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: False))
    assert open_help_page("<html></html>") is False


def test_open_help_page_never_uses_the_linux_fallback_off_linux(qapp, monkeypatch):
    # Mirror image of test_open_help_page_never_uses_qdesktopservices_on_linux
    # -- pins the other half of the branch just as loudly, so a future
    # edit can't accidentally make Windows/macOS fall through to the
    # Linux-only subprocess path either.
    monkeypatch.setattr(help_content.sys, "platform", "win32")
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: False))

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("the Linux browser fallback must not run off Linux")

    monkeypatch.setattr(help_content.shutil, "which", _unexpected_call)
    monkeypatch.setattr(help_content.subprocess, "Popen", _unexpected_call)
    assert open_help_page("<html></html>") is False


def test_open_help_page_returns_false_on_write_failure(qapp, monkeypatch):
    def _raise_oserror(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _raise_oserror)
    assert open_help_page("<html></html>") is False


def test_open_help_page_never_uses_qdesktopservices_on_linux(qapp, monkeypatch):
    # QDesktopServices.openUrl() shells out to xdg-open internally on
    # Linux, using this frozen process's own environment (including its
    # PyInstaller-set LD_LIBRARY_PATH) with no way for us to sanitize
    # it first -- confirmed to crash a spawned browser with GLIBCXX
    # version-mismatch errors even when xdg-open successfully finds
    # one. So on Linux we never call it at all, unconditionally --
    # always launch the browser ourselves instead, where we control
    # the environment. This pins that decision down explicitly.
    monkeypatch.setattr(help_content.sys, "platform", "linux")

    def _unexpected_open_url(*args, **kwargs):
        raise AssertionError("QDesktopServices.openUrl must never be called on Linux")

    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(_unexpected_open_url))
    monkeypatch.setattr(
        help_content.shutil, "which",
        lambda name: "/usr/bin/firefox" if name == "firefox" else None,
    )
    monkeypatch.setattr(help_content.subprocess, "Popen", lambda args, **kwargs: None)
    assert open_help_page("<html></html>") is True


def test_open_help_page_falls_back_to_a_browser_binary_on_linux(qapp, monkeypatch):
    # Many real Linux installs have no xdg-open and no default-browser
    # association configured at all, even when a real browser binary
    # is present -- launching a binary directly bypasses that OS-level
    # lookup entirely.
    monkeypatch.setattr(help_content.sys, "platform", "linux")
    monkeypatch.setattr(
        help_content.shutil, "which",
        lambda name: "/usr/bin/firefox" if name == "firefox" else None,
    )
    popen_calls = []
    monkeypatch.setattr(
        help_content.subprocess, "Popen",
        lambda args, **kwargs: popen_calls.append((args, kwargs)),
    )
    assert open_help_page("<html></html>") is True
    assert len(popen_calls) == 1
    args, kwargs = popen_calls[0]
    assert args[0] == "/usr/bin/firefox"
    assert os.path.exists(args[1])


def test_open_help_page_returns_false_when_no_linux_candidate_found(qapp, monkeypatch):
    monkeypatch.setattr(help_content.sys, "platform", "linux")
    monkeypatch.setattr(help_content.shutil, "which", lambda name: None)
    assert open_help_page("<html></html>") is False


def test_open_help_page_skips_missing_candidates_and_uses_first_found(qapp, monkeypatch):
    # "xdg-open" is checked first (see _FALLBACK_BROWSERS) but is
    # commonly absent on minimal installs -- confirm the loop moves on
    # to the next candidate instead of giving up.
    monkeypatch.setattr(help_content.sys, "platform", "linux")
    monkeypatch.setattr(
        help_content.shutil, "which",
        lambda name: "/usr/bin/chromium" if name == "chromium" else None,
    )
    popen_calls = []
    monkeypatch.setattr(
        help_content.subprocess, "Popen",
        lambda args, **kwargs: popen_calls.append((args, kwargs)),
    )
    assert open_help_page("<html></html>") is True
    assert popen_calls[0][0][0] == "/usr/bin/chromium"


def test_open_help_page_fallback_tries_the_next_candidate_if_popen_raises(qapp, monkeypatch):
    monkeypatch.setattr(help_content.sys, "platform", "linux")
    monkeypatch.setattr(
        help_content.shutil, "which",
        lambda name: {"xdg-open": "/usr/bin/xdg-open", "firefox": "/usr/bin/firefox"}.get(name),
    )
    popen_calls = []

    def _popen(args, **kwargs):
        popen_calls.append(args)
        if args[0] == "/usr/bin/xdg-open":
            raise OSError("exec format error")

    monkeypatch.setattr(help_content.subprocess, "Popen", _popen)
    assert open_help_page("<html></html>") is True
    assert [c[0] for c in popen_calls] == ["/usr/bin/xdg-open", "/usr/bin/firefox"]


def test_open_help_page_fallback_restores_saved_ld_library_path(qapp, monkeypatch):
    # PyInstaller's Linux bootloader points LD_LIBRARY_PATH at the
    # bundled library directory so the frozen app's own dependencies
    # resolve correctly, saving whatever the real original value was
    # (if any) under LD_LIBRARY_PATH_ORIG. A child process like a
    # system browser inheriting the bundle's path unmodified can end
    # up loading mismatched bundled libraries instead of its own --
    # confirmed for real: a spawned Epiphany crashes immediately with
    # `libstdc++.so.6: version 'GLIBCXX_3.4.30' not found` when this
    # isn't done.
    monkeypatch.setattr(help_content.sys, "platform", "linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIxxxxxx")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/usr/lib/original")
    monkeypatch.setattr(
        help_content.shutil, "which",
        lambda name: "/usr/bin/firefox" if name == "firefox" else None,
    )
    popen_calls = []
    monkeypatch.setattr(
        help_content.subprocess, "Popen",
        lambda args, **kwargs: popen_calls.append(kwargs),
    )
    open_help_page("<html></html>")
    assert popen_calls[0]["env"]["LD_LIBRARY_PATH"] == "/usr/lib/original"


def test_open_help_page_fallback_strips_ld_library_path_when_no_orig_saved(qapp, monkeypatch):
    monkeypatch.setattr(help_content.sys, "platform", "linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIxxxxxx")
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    monkeypatch.setattr(
        help_content.shutil, "which",
        lambda name: "/usr/bin/firefox" if name == "firefox" else None,
    )
    popen_calls = []
    monkeypatch.setattr(
        help_content.subprocess, "Popen",
        lambda args, **kwargs: popen_calls.append(kwargs),
    )
    open_help_page("<html></html>")
    assert "LD_LIBRARY_PATH" not in popen_calls[0]["env"]


def test_knowledge_database_html_explains_sigma_and_fwhm():
    html = _strip_base64_images(build_knowledge_database_html())
    assert "2.3548" in html
    assert "FWHM_err = 2.3548" in html


def test_knowledge_database_html_explains_position_volume_and_uncertainties():
    html = _strip_base64_images(build_knowledge_database_html())
    assert "area = amplitude &middot; &sigma; &middot; &radic;(2&pi;)" in html
    assert "full_area = area + (background_slope" in html
    # The tailed form too: the volume is the integral of the WHOLE fitted
    # shape now, not of its Gaussian core. The page previously published
    # the quadrature-of-relative-errors formula for area_err, which
    # assumed amplitude and sigma were uncorrelated -- it is gone because
    # the code no longer does that.
    assert "2r&beta; / erfcx(y)" in html
    assert "area_err = |area|" not in html
    text = _rendered_text(build_knowledge_database_html())
    assert "including the correlations" in text
    assert "anti-correlated" in text


def test_knowledge_database_explains_fitting_the_background_jointly():
    # F5: fit_peaks(fit_background=True), opt-in via the Fit Parameters
    # checkbox. The page must say WHY errors grow (that is the correction,
    # and a user seeing bigger numbers will otherwise think it broke) and be
    # honest that it is not always the better choice.
    text = _rendered_text(build_knowledge_database_html())
    assert "Fitting the background with the peaks" in text
    assert "Fit background" in text
    assert "claims information nobody has" in text
    assert "grow" in text
    # The honest caveat, and the actionable guidance.
    assert "cost two degrees of freedom" in text
    assert "steeply sloping" in text
    # And the non-obvious detail about what "level" means.
    assert "not at channel zero" in text


def test_knowledge_database_explains_why_widths_are_shared():
    # F4: shared width is the default. A user who does not know that will
    # not understand why one peak's width moved when a neighbour was added.
    text = _rendered_text(build_knowledge_database_html())
    assert "Why peaks in one fit share a width" in text
    assert "Independent widths" in text
    assert "same detector" in text


def test_knowledge_database_explains_the_background_uncertainty_band():
    # F6: background_error() / FitResult.background_level_error(), drawn by
    # fit_mode._draw_fit_curves as a shaded band. The page has to explain
    # both the formula and -- more useful to a reader -- why the band is
    # narrow between the regions and wide outside them.
    text = _rendered_text(build_knowledge_database_html())
    assert "How certain is the background?" in text
    assert "shaded band" in text
    assert "no cross term" in text
    # The actionable part: what a wide band means and what to do about it.
    assert "extrapolated" in text
    assert "moving the background regions closer" in text


def test_knowledge_database_html_explains_integration_moments():
    html = _strip_base64_images(build_knowledge_database_html())
    assert "How integration computes gross, background, and net" in html
    assert "centroid = &Sigma;(x&middot;y) / &Sigma;y" in html


def test_knowledge_database_html_figure_numbers_and_cross_references():
    html = _strip_base64_images(build_knowledge_database_html())
    assert "see Figure 4 -- unless you" in html
    assert "Figure 4. Three peaks fit together" in html
    assert "Figure 6. Linear vs. quadratic calibration" in html
    assert "area = &Sigma;y" in html
    assert "&sigma;&sup2; = &Sigma;((x&minus;centroid)" in html


def test_howto_loading_section_lists_every_readable_format():
    """The count in the prose is part of the claim: adding a reader
    without updating it leaves the page quietly wrong about what the
    app can open."""
    html = build_howto_html()
    assert "five supported formats" in html
    for extension in (".txt", ".spe", ".spk", ".n42", ".lzs"):
        assert f"<b>{extension}</b>" in html


def test_howto_saving_section_notes_n42_is_read_only():
    html = build_howto_html()
    # Scoped to the "3. Saving a spectrum" section specifically -- a bare
    # `"read-only" in html` check could pass from unrelated prose
    # elsewhere on the page.
    saving_section = html.split("<h3>3. Saving a spectrum</h3>")[1].split("<h3>")[0]
    assert "<b>.n42</b>" in saving_section
    assert "<b>.lzs</b>" in saving_section
    assert "read-only" in saving_section
    # Guards against the specific stale claim this test was added to fix:
    # saving no longer supports "the same" formats as reading now that
    # reading also accepts the (write-unsupported) .n42 format.
    assert "the same three formats" not in html


def test_howto_html_documents_matrix_analysis():
    html = build_howto_html()
    # Matched on the TITLE, not the number: pinning the number turns every
    # inserted section into a spurious failure here, and the number is
    # covered properly by test_howto_section_numbering_is_sound below.
    assert "Matrix analysis</h3>" in html
    assert "Open Matrix" in html
    assert "Ctrl+Shift+O" in html.replace("&#43;", "+")


def test_howto_html_documents_matrix_panel_fit_integrate_calibrate():
    html = build_howto_html()
    # Ctrl+F/Ctrl+I/Ctrl+L/Ctrl+= are also used elsewhere on the page (main
    # window fit/calibrate/zoom), so a bare substring check on those alone
    # would pass even if the matrix panel's own section were never written.
    # <kbd>C</kbd>/<kbd>G</kbd> and the "Matrix panel" table heading are
    # unique to the new content, so they're what actually prove it exists.
    assert "<h3>Matrix panel</h3>" in html
    assert "<kbd>C</kbd>" in html
    assert "<kbd>G</kbd>" in html
    assert "<kbd>Ctrl+F</kbd>" in html
    assert "<kbd>Ctrl+I</kbd>" in html
    assert "<kbd>Ctrl+L</kbd>" in html
    assert "<kbd>Ctrl+=</kbd>" in html


def test_howto_html_documents_clear_marks_button():
    html = build_howto_html()
    assert "Clear Marks" in html
    # Specific behavior, not just the button's name -- pins down that it
    # clears both mark types together and leaves fit marks/committed
    # fits/an already-activated cut alone, matching what
    # matrix_panel.py's _clear_marks -> MatrixCutController.clear ->
    # MatrixCutState.reset actually does (cut_region, bg_regions, and
    # both pending clicks; nothing on fit_controller's state).
    assert "resets the cut region and every" in html
    assert "fit marks, committed fits, and any" in html


def test_knowledge_database_explains_2d_matrix_concepts():
    html = _strip_base64_images(build_knowledge_database_html())
    assert "projection" in html.lower()
    assert "cut" in html.lower() or "gate" in html.lower()
    assert "coincidence" in html.lower()


def _rendered_text(html):
    """Visible text only. Asserting against raw HTML gives false
    positives -- an audit of this page found "n/a" and "progress"
    apparently present when neither appeared in anything a reader sees
    (one matched inside markup, the other only ever meant "in-progress
    marks")."""
    import re

    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", _strip_base64_images(html)))


def test_knowledge_database_explains_an_n_a_uncertainty():
    # Fits can report "± n/a" for a parameter the data does not
    # constrain (peak_fit._uncertainties_from -> fit_mode._err_text).
    # Before this the page never mentioned it, so a user meeting "n/a"
    # in the Fit Results panel had nothing to consult.
    text = _rendered_text(build_knowledge_database_html())
    assert "n/a" in text
    assert "did not constrain" in text
    # ... and the actionable part: which parameters, and what to do.
    assert "left tail" in text.lower()
    assert "unchecking" in text
    # ... and that the fit itself is still valid.
    assert "not a failed fit" in text


def test_knowledge_database_explains_the_degenerate_fit_refusal():
    # The other branch: peak parameters undetermined is still a hard
    # failure, and the message names peaks. The page should say why and
    # what to change.
    text = _rendered_text(build_knowledge_database_html())
    assert "cannot be separated" in text
    assert "same position" in text


def test_knowledge_database_documents_the_export_forms_of_an_unknown_error():
    # A text report writes "± n/a"; the .jsonl auto-log writes null so
    # the file stays valid JSON (fit_export._format_err / _json_safe).
    text = _rendered_text(build_knowledge_database_html())
    assert "_fits.jsonl" in text
    assert "null" in text


def test_howto_documents_the_matrix_loading_progress_window():
    # main_window._load_matrix_with_progress shows a QProgressDialog
    # labelled "Reading matrix..." and keeps the window responsive.
    text = _rendered_text(build_howto_html())
    assert "Reading matrix" in text
    assert "responsive" in text


def test_howto_documents_drag_panning():
    # main_window._pan_to / matrix_panel._pan_to: a drag keeps the X span
    # and rescales Y to what is visible. Since v4.0.1 either button does
    # it, so the page no longer says "right mouse button" in the passage
    # that describes the gesture -- it says so only where the right
    # button's extra behaviour (panning even with a marking key held)
    # differs from the left's.
    text = _rendered_text(build_howto_html())
    assert "keeps its width" in text
    assert "rescales" in text
    assert "The right button pans regardless" in text


def test_howto_says_ctrl_c_clears_cut_marks_in_the_matrix_panel():
    # matrix_panel._clear_everything now clears cut/background marks too.
    text = _rendered_text(build_howto_html())
    assert "Clear the cut and background marks" in text
    assert "Clear Marks" in text


def test_knowledge_database_explains_gate_width_weighting():
    # matrix_cut.compute_cut: net = cut - (N_cut / N_bg) * sum bg, where
    # the N terms are the channels actually summed, counted inclusively.
    text = _rendered_text(build_knowledge_database_html())
    assert "gate-width weighting" in text
    assert "net[ch]" in text
    # The worked example and the inclusive-counting rule, both of which a
    # reader needs to reproduce the number themselves.
    assert "40/20" in text
    assert "51 channels, not" in text


def test_knowledge_database_explains_background_regions_at_the_matrix_edge():
    # This paragraph used to WARN that an overhanging band under-subtracts,
    # because the width came from the marks while the counts came from the
    # data that exists. Both now come from the channels actually summed
    # (matrix_cut.compute_cut), so the page has to describe a rule that
    # holds rather than a trap to avoid -- otherwise it tells the reader to
    # fear something that no longer happens.
    text = _rendered_text(build_knowledge_database_html())
    assert "Both halves of the ratio count the same channels" in text
    assert "the subtraction stays correct" in text
    # The two edge behaviours the code actually implements.
    assert "contributes nothing at all" in text
    assert "no subtraction is performed" in text


def test_knowledge_database_explains_that_overlapping_bands_count_once():
    # matrix_cut.merge_regions merges overlapping regions, so a channel
    # covered twice is summed and counted once. Worth stating because the
    # accident that produces it -- a wide band with a narrower one inside
    # -- is easy and its effect on the result is not obvious.
    text = _rendered_text(build_knowledge_database_html())
    assert "Overlapping bands are counted once" in text
    assert "summed once and counted once" in text


def test_knowledge_database_records_both_intentional_tv_divergences():
    # The page must scope its TV-parity claim to where parity actually
    # holds -- fully-inside, non-overlapping bands -- and name both
    # deliberate differences, rather than claiming blanket parity or
    # mentioning only one of them.
    text = _rendered_text(build_knowledge_database_html())
    assert "do not\noverlap" in text or "do not overlap" in text
    assert "both are deliberate" in text
    assert "TV" in text
    # The rule's actual provenance, so a reader comparing against either
    # program knows which one this follows.
    assert "HDTV" in text


def test_knowledge_database_documents_root_file_support():
    # root_io / root_dialog: a ROOT file is a directory tree, so opening it
    # asks twice. The page must also state the two refusals, since a user
    # meeting them needs to know they are deliberate.
    text = _rendered_text(build_knowledge_database_html())
    assert "ROOT files" in text
    assert "directory tree" in text
    assert "asks twice" in text
    # Calibration comes from the axis when the file has one.
    assert "carries real coordinates" in text
    # The refusals, and that reading is one-way.
    assert "fractional" in text
    assert "non-uniform bin widths" in text
    assert "Nothing is written back" in text


def test_knowledge_database_documents_saving_and_reloading_work():
    # fit_persist: restore vs re-run is a real choice, and the reason it
    # matters is that v4.0.0 changed the numbers.
    text = _rendered_text(build_knowledge_database_html())
    assert "Saving and reloading your work" in text
    assert "Restore" in text and "Re-run" in text
    assert "changed how peak areas" in text
    # Schema versioning, so old files keep opening.
    assert "schema version" in text
    # Reload, and why a length change clears the marks.
    assert "Reload Spectrum" in text
    assert "anchored to channel numbers" in text


def test_knowledge_database_documents_the_export_formats():
    text = _rendered_text(build_knowledge_database_html())
    assert "Exporting results" in text
    assert ".csv" in text and ".tex" in text
    # The two details a user would otherwise be caught by.
    assert "empty cell" in text
    assert "Areas are never converted" in text


def test_knowledge_database_documents_calibrating_from_fitted_peaks():
    # X3. The two things a user would otherwise get wrong: why the list is
    # in channels, and that the residual is what says whether to trust it.
    text = _rendered_text(build_knowledge_database_html())
    assert "Calibrating from fitted peaks" in text
    assert "circular" in text
    assert "worst residual" in text
    assert "assigned to the wrong line" in text
    # And that the bare minimum is an interpolation, not a fit.
    assert "interpolation rather than a fit" in text


def test_howto_shortcut_table_lists_every_shortcut_the_app_registers():
    """The hand-maintained list above cannot catch a NEW shortcut going
    undocumented -- it only checks what somebody remembered to add. F5
    (Reload Spectrum) shipped in v4.0.0 and was missing from the HowTo
    entirely until this test was written.

    Compares the page against what main_window/matrix_panel/fit_mode
    actually call setShortcut() with, so a future shortcut cannot be
    added to the app and silently left out of the docs.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    registered = set()
    for name in ("main_window.py", "matrix_panel.py", "fit_mode.py"):
        registered |= set(re.findall(
            r'setShortcut\("([^"]+)"\)', (root / name).read_text(encoding="utf-8")
        ))
    assert registered, "found no shortcuts at all -- the pattern must have gone stale"

    html = build_howto_html()
    documented = set(re.findall(r"<kbd>([^<]+)</kbd>", html))
    missing = sorted(registered - documented)
    assert not missing, f"shortcuts registered but not documented in the HowTo: {missing}"


def test_howto_documents_the_v400_features():
    # The HowTo was never updated for v4.0.0 -- every one of these was
    # absent while the Knowledge Database covered them all, so a user
    # looking for "how do I open a ROOT file" found nothing.
    text = _rendered_text(build_howto_html())
    for term in ("Open ROOT File", "Reload Spectrum", "Save Fits",
                 "Load Fits", "Calibrate from Fitted Peaks", "Fit background"):
        assert term in text, f"HowTo does not mention {term!r}"
    # The export formats, and the two details that catch people out.
    assert ".csv" in text and ".tex" in text
    assert "empty cell" in text
    assert "worst residual" in text


def test_help_pages_document_multi_gate_cuts():
    """This shipped half-finished: compute_cut summed several gates but the
    panel could only ever mark one, so the CHANGELOG claimed a feature the
    user could not reach. Now that it works, both pages must say so -- the
    HowTo for how to mark them, the KB for what summing gates means."""
    howto = _rendered_text(build_howto_html())
    kb = _rendered_text(build_knowledge_database_html())

    assert "any number of regions allowed" in howto
    assert "as many gates as you want" in howto
    assert "Gating on more than one peak" in kb
    assert "totals across all of them" in kb
    # The caveat that decides whether it is a good idea.
    assert "contaminant" in kb
    assert "counted once" in kb


def test_howto_documents_left_drag_panning():
    """v4.0.1. The page previously said outright that "the left button is
    unaffected -- it is what places fit marks", which is now false in the
    common case. Both the main-window and matrix-panel passages have to
    move, or one of them keeps telling the user the old story."""
    text = _rendered_text(build_howto_html())

    assert "either mouse button" in text
    assert "left button is unaffected" not in text
    # The precedence rule, which is what stops it feeling unpredictable.
    assert "Marking always takes precedence" in text
    assert "stands aside while the plot toolbar" in text
    # And the matrix panel passage, not just the main-window one.
    assert text.count("either mouse button") >= 2


# ---------------------------------------------------------------------------
# v4.1.0: the audit fixes changed what several reported numbers mean, so the
# Knowledge Database has to say so. Rendered text only -- see _rendered_text.
# ---------------------------------------------------------------------------


def test_knowledge_database_no_longer_claims_the_net_total_is_quadrature():
    """The net total's uncertainty is propagated through the covariance
    now. The page used to state the old formula outright, and a stale
    formula in the physics documentation is worse than none."""
    text = _rendered_text(build_knowledge_database_html())
    assert "with their uncertainties combined in quadrature" not in text
    assert "whole covariance matrix in one step" in text
    assert "anti" in text and "correlated" in text


def test_knowledge_database_explains_why_a_derived_spectrum_is_not_sqrt_n():
    text = _rendered_text(build_knowledge_database_html())
    assert "var(net[ch])" in text
    # The three operations that transform it, each named.
    assert "scale it by the square of the factor" in text
    assert "adds together the variances" in text
    # And the counter-intuitive consequence, stated plainly.
    assert "larger" in text


def test_knowledge_database_cross_reference_resolves_to_a_real_section():
    """The integration section points the reader at a named section. A
    dangling pointer in help text is invisible until a user follows it."""
    import re

    html = build_knowledge_database_html()
    headings = {
        re.sub(r"<[^>]+>", "", h).strip()
        for h in re.findall(r"<h2>(.*?)</h2>", html, re.S)
    }
    assert "Why a cut&#39;s uncertainty is not &radic;N" in headings or \
           "Why a cut's uncertainty is not &radic;N" in headings, headings
    text = _rendered_text(html)
    assert "uncertainty is not" in text


def test_knowledge_database_says_integration_can_use_a_propagated_variance():
    text = _rendered_text(build_knowledge_database_html())
    assert "uses that real variance instead" in text
    assert "negative" in text


def test_knowledge_database_documents_the_weighted_calibration():
    text = _rendered_text(build_knowledge_database_html())
    assert "&plusmn; ch" in text or "± ch" in text
    assert "shows a dash instead of a number" in text
    assert "slope and its uncertainty" in text


def test_knowledge_database_explains_the_capped_tail_beta():
    """v4.1.0 S7: beta is bounded above now, and a user who sees it sitting
    exactly on the bound needs to know that means 'no tail here to measure'
    rather than a measurement."""
    text = _rendered_text(build_knowledge_database_html())
    assert "Why &beta; is capped" in text or "Why \u03b2 is capped" in text
    assert "no tail here to measure" in text
    # And the cross-reference it makes must resolve.
    import re
    headings = {
        re.sub(r"<[^>]+>", "", h).strip()
        for h in re.findall(r"<h2>(.*?)</h2>", build_knowledge_database_html(), re.S)
    }
    assert any("uncertainty reads" in h for h in headings), headings


def test_howto_section_numbering_is_sound():
    """Numbered sections run 1..N with no gaps or duplicates, and every
    cross-reference of the form "12. Some Title" names the section that
    actually carries that number.

    Both halves earned their place. Inserting Go To as section 12 left two
    sections numbered 13 until it was caught; and two references had long
    read "11. Matrix analysis" while that section had drifted to 13, which
    nothing checked.
    """
    import re

    html = build_howto_html()
    numbered = re.findall(r"<h3>(\d+)\.\s*([^<]+)</h3>", html)
    numbers = [int(n) for n, _ in numbered]
    assert numbers, "no numbered sections found"
    assert numbers == sorted(numbers), f"sections out of order: {numbers}"
    assert len(numbers) == len(set(numbers)), f"duplicate section numbers: {numbers}"
    assert numbers == list(range(1, len(numbers) + 1)), f"gap in numbering: {numbers}"

    by_number = {n: title.strip().lower() for n, title in numbered}
    for number, title in re.findall(r'"(\d+)\.\s*([^"]+)"', html):
        actual = by_number.get(number)
        assert actual is not None, f'reference to "{number}. {title}" but no such section'
        assert actual == title.strip().lower(), (
            f'reference says "{number}. {title.strip()}" but section {number} '
            f'is "{actual}"'
        )


# ---------------------------------------------------------------------------
# v4.1.1: how the background is determined, what Ctrl+B previews, and what a
# committed fit draws. Rendered text only -- see _rendered_text.
# ---------------------------------------------------------------------------


def test_kb_explains_how_the_background_line_is_determined():
    text = _rendered_text(build_knowledge_database_html())
    assert "drawn through those two points" in text
    # The distinction that makes its uncertainty exact rather than
    # approximate, and which a reader could easily assume the other way.
    assert "not a least-squares fit" in text


def test_kb_says_ctrl_b_is_a_preview_and_needs_only_the_bg_regions():
    text = _rendered_text(build_knowledge_database_html())
    assert "needs both background regions marked and nothing else" in text
    assert "nothing is fitted, computed or stored" in text


def test_kb_says_ctrl_b_ignores_the_fit_background_checkbox():
    """The user asked how Ctrl+B differs with "Fit background" on. It does
    not: toggle_background_preview never reads the checkbox, and the
    preview always draws the same two-point line. Documenting the
    non-interaction is the point."""
    text = _rendered_text(build_knowledge_database_html())
    assert "is not affected by the" in text
    assert "always shows this same two-point line" in text


def test_kb_describes_what_a_committed_fit_draws():
    text = _rendered_text(build_knowledge_database_html())
    for element in ("dashed background line", "faint shaded band",
                    "total model curve"):
        assert element in text, element
    # The band appears in BOTH modes -- only its source differs. Saying it
    # only appears with the checkbox on would be wrong.
    assert "band is drawn either way" in text


def test_kb_band_is_narrowest_between_the_regions_not_at_them():
    """Measured: for regions whose means are known to +-2.1 and +-2.4
    counts, the band closes to +-1.6 BETWEEN them. The page (and a code
    comment) previously claimed it was narrowest AT the two regions."""
    text = _rendered_text(build_knowledge_database_html())
    assert "narrowest at the two background regions" not in text
    assert "narrowest point is between the regions" in text


def test_kb_does_not_explain_the_fit_background_checkbox_twice():
    """One control, one explanation. A second section covering the same
    checkbox was written and dropped before commit -- two explanations of
    one control drift apart."""
    html = build_knowledge_database_html()
    import re

    headings = re.findall(r"<h2>(.*?)</h2>", html, re.S)
    about_checkbox = [h for h in headings if "Fit background" in h or "fit background" in h]
    assert len(about_checkbox) <= 1, f"more than one section on the checkbox: {about_checkbox}"



def test_kb_says_what_the_optimiser_is_fitted_against_in_each_mode():
    """The mechanical heart of the difference: unticked fits the residual
    left after subtracting the two-point line, ticked fits the raw counts
    with the line as two more free parameters."""
    text = _rendered_text(build_knowledge_database_html())
    assert "fitted against the residual" in text
    assert "fitted against the raw counts" in text
    assert "the background has no free parameters and cannot move" in text


def test_kb_says_the_two_point_line_seeds_both_modes():
    """Easy to assume the line is only computed when the box is unticked.
    It is computed either way -- it seeds the peak amplitudes and widths
    in both."""
    text = _rendered_text(build_knowledge_database_html())
    assert "it is what seeds the peak" in text


def test_kb_explains_why_the_background_level_is_quoted_at_the_region_middle():
    """The page already stated the fact. Without the reason it reads as an
    arbitrary display choice, when it is what keeps the two background
    parameters independently determinable."""
    text = _rendered_text(build_knowledge_database_html())
    assert "almost perfectly anti-correlated with the slope" in text
    assert "converted back to an ordinary line" in text


def test_howto_documents_source_file_assignment():
    """Specific multi-word substrings, matching this file's convention, so
    a partial revert cannot hide behind generic wording."""
    html = build_howto_html()
    assert "Load source..." in html
    assert "Suggest remaining" in html
    # The two-anchor requirement and the ambiguity rule are the two facts
    # a user cannot guess from the UI.
    assert "closer than half the distance to the runner-up" in html
    assert "stay blank rather than guessed" in html


def test_knowledge_database_documents_the_sou_format_and_matching_rule():
    html = build_knowledge_database_html()
    # The four columns, in order, as the file actually carries them.
    assert "energy in keV" in html
    assert "relative intensity" in html
    # The intensity scale is per-file: co56.sou peaks at 100000 and
    # na24.sou at 1000, so the page must not promise 10000 universally.
    assert "not comparable between" in html
    # Why two anchors are needed at all.
    assert "requires a channel-to-" in html
    # The rule, and the two consequences that follow from it.
    assert "closer than half the distance to the" in html
    assert "is never" in html and "suggested for a second peak" in html
    assert "neither receives it" in html


def test_knowledge_database_warns_that_suggestions_are_not_identifications():
    """The honest caveat: an unambiguous match can still be the wrong
    line if the anchors were misidentified."""
    html = build_knowledge_database_html()
    assert "not an identification" in html
    assert "wrong line if the provisional anchors" in html


def test_help_documents_that_switching_source_drops_old_guesses():
    """Behaviour a user cannot see coming, in both pages."""
    assert "loading a different source file" in build_howto_html()
    kb = build_knowledge_database_html()
    assert "Loading a <b>different source file</b> clears" in kb
    assert "calibrate\nagainst the source you had just rejected" in kb


def test_help_documents_the_calibration_workflow():
    """Fragments unique to the new sections.

    An earlier version asserted "strongest line" and "100", both of which
    a pre-existing sentence elsewhere already satisfied ("normalise the
    strongest line to 10000" contains "100"), so the whole CalEnEff
    section could be deleted with the test still green.
    """
    howto = build_howto_html()
    assert "Finish and save for CalEnEff" in howto
    assert "its own width comes back blank" in howto

    kb = build_knowledge_database_html()
    # The restore rule, the export, and the chi-squared caveat, each by a
    # phrase that appears nowhere else on the page.
    assert "within that peak" in kb and "FWHM" in kb
    assert "normalised to <b>100</b>" in kb
    assert "efficiency curve" in kb
    assert "unweighted" in kb
    assert "no degrees of freedom" in kb


def test_help_documents_the_new_results_columns():
    kb = build_knowledge_database_html()
    # Short single-line fragments: the HTML is wrapped, so an assertion
    # on a whole sentence silently matches nothing.
    assert "1332.49(12)" in kb          # the notation, explained
    assert "352.72" in kb               # the capped form, uncertainty dropped
    assert "661.66(3)" in kb            # the capped form, uncertainty kept
    assert "capped at two decimals" in kb
    # Not the bare word "tooltip" -- three unrelated pre-existing mentions
    # already satisfy that.
    assert "gross and net areas" in kb


def test_help_does_not_still_describe_the_dropped_column():
    """The '#' column and the old uncooked example are gone from the
    app; documentation that still describes them is worse than none."""
    kb = build_knowledge_database_html()
    assert "<b>#</b>" not in kb
    assert "352.7217(14)" not in kb


def test_help_says_the_calibration_plot_is_live():
    howto = build_howto_html()
    assert "redraws after every change" in howto
