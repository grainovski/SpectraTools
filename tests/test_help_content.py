from help_content import build_howto_html


def test_howto_html_contains_every_shortcut():
    html = build_howto_html()
    for shortcut in [
        "Ctrl+O", "Ctrl+S", "Ctrl+W", "Ctrl+Q",
        "Ctrl+G", "Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+D",
        "Ctrl+L", "Ctrl+T", "Ctrl+M", "Ctrl+R", "Ctrl+N",
        "Ctrl+=", "Ctrl+-", "Ctrl+0",
        "Ctrl+F", "Ctrl+C", "Ctrl+Shift+C", "Ctrl+E", "Ctrl+I",
    ]:
        assert shortcut in html, f"missing shortcut {shortcut!r}"
    for bare_key_markup in (">B<", ">R<", ">P<"):
        assert bare_key_markup in html


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
