import os
import re
import sys
import tempfile
import types
from html.parser import HTMLParser

from PySide6.QtGui import QDesktopServices

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
        "Ctrl+L", "Ctrl+T", "Ctrl+M", "Ctrl+R", "Ctrl+N",
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
        "Performing a fit",
        "Integration",
        "View options",
        "Knowledge Database",
    ]:
        assert topic in html, f"missing topic {topic!r}"


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
    opened = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url))
    )
    open_help_page("<html><body>hello</body></html>")
    assert len(opened) == 1
    path = opened[0].toLocalFile()
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert f.read() == "<html><body>hello</body></html>"


def test_open_help_page_returns_true_on_success(qapp, monkeypatch):
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: True))
    assert open_help_page("<html></html>") is True


def test_open_help_page_returns_false_when_no_browser_handler(qapp, monkeypatch):
    # QDesktopServices.openUrl itself returns False (not an exception)
    # when there's no registered handler for the URL -- e.g. no default
    # browser configured. Must be surfaced, not silently dropped.
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: False))
    assert open_help_page("<html></html>") is False


def test_open_help_page_returns_false_on_write_failure(qapp, monkeypatch):
    def _raise_oserror(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _raise_oserror)
    assert open_help_page("<html></html>") is False
