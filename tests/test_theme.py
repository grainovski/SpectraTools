import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from main_window import MainWindow
from theme import DARK_BG, qt_stylesheet, style_axes


def test_qt_stylesheet_is_empty_for_light_theme():
    assert qt_stylesheet("light") == ""


def test_qt_stylesheet_is_nonempty_and_dark_for_dark_theme():
    css = qt_stylesheet("dark")
    assert css != ""
    assert DARK_BG in css


def test_style_axes_applies_light_colors():
    axes = Figure().add_subplot(111)
    style_axes(axes, "light")
    assert axes.get_facecolor() == (1.0, 1.0, 1.0, 1.0)


def test_style_axes_applies_dark_colors():
    axes = Figure().add_subplot(111)
    style_axes(axes, "dark")
    # DARK_BG = "#1e1e1e" -> RGB fractions of 0x1e = 30/255.
    r, g, b, a = axes.get_facecolor()
    assert round(r, 3) == round(30 / 255, 3)
    assert round(g, 3) == round(30 / 255, 3)
    assert round(b, 3) == round(30 / 255, 3)


def test_toggling_dark_theme_action_applies_and_persists(qapp):
    main_window = MainWindow()
    original_theme = main_window.settings.theme()
    try:
        assert main_window.dark_theme_action.isChecked() == (original_theme == "dark")

        main_window.dark_theme_action.setChecked(True)
        assert qapp.styleSheet() != ""
        assert main_window.settings.theme() == "dark"
        assert main_window._theme == "dark"

        main_window.dark_theme_action.setChecked(False)
        assert qapp.styleSheet() == ""
        assert main_window.settings.theme() == "light"
        assert main_window._theme == "light"
    finally:
        # Restore whatever theme was persisted before this test ran, so
        # running the suite doesn't leave the real machine's QSettings
        # (Windows registry) toggled to whatever this test happened to
        # set last.
        main_window.settings.set_theme(original_theme)
        qapp.setStyleSheet(qt_stylesheet(original_theme))
