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
        "Ctrl+O", "Ctrl+S", "Ctrl+W", "Ctrl+Q",
        "Ctrl+G", "Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+D",
        "Ctrl+L", "Ctrl+T", "Ctrl+M", "Ctrl+R", "Ctrl+N", "Ctrl+A", "Ctrl+Shift+A",
        "Ctrl+=", "Ctrl+-", "Ctrl+0",
        "Ctrl+F", "Ctrl+C", "Ctrl+Shift+C", "Ctrl+E", "Ctrl+I",
        "B", "R", "P", "F1",
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
        "View options",
        "Knowledge Database",
    ]:
        assert topic in html, f"missing topic {topic!r}"


def test_howto_integration_section_mentions_optional_background():
    html = build_howto_html()
    assert "Background regions are optional" in html


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


def test_knowledge_database_html_embeds_four_figures():
    html = build_knowledge_database_html()
    assert html.count("data:image/png;base64,") == 4


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
