from html.parser import HTMLParser

from help_content import build_howto_html

_VOID_ELEMENTS = {"meta", "br", "img", "hr", "link", "input"}


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
        "B", "R", "P",
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


from help_content import build_knowledge_database_html


def test_knowledge_database_html_contains_parameter_names():
    html = build_knowledge_database_html()
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
