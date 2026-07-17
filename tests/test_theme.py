import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from main_window import MainWindow
from spectrum import DARK_COLOR_CYCLE, LIGHT_COLOR_CYCLE, LoadedSpectrum, next_color
from theme import DARK_BG, fit_drawing_colors, qt_stylesheet, style_axes


def test_fit_drawing_colors_dark_theme_matches_tv():
    # TV's own tv-1.9.13/etc/Xtv: fit-function.foreground0 = gold,
    # bg-function.foreground0 = green (X11 pure green, not CSS's darker
    # shade).
    fit_color, bg_color = fit_drawing_colors("dark")
    assert fit_color == "#FFD700"
    assert bg_color == "#00FF00"


def test_fit_drawing_colors_light_theme_is_the_dark_theme_complement():
    # Light theme uses the HSV (180 deg hue rotation) complements of
    # TV's gold/green, not an arbitrary color -- vivid blue and magenta.
    fit_color, bg_color = fit_drawing_colors("light")
    assert fit_color == "#0028FF"
    assert bg_color == "#FF00FF"


def test_fit_drawing_colors_differ_between_themes():
    assert fit_drawing_colors("light") != fit_drawing_colors("dark")


def test_qt_stylesheet_is_empty_for_light_theme():
    assert qt_stylesheet("light") == ""


def test_qt_stylesheet_is_nonempty_and_dark_for_dark_theme():
    css = qt_stylesheet("dark")
    assert css != ""
    assert DARK_BG in css


def test_qt_stylesheet_styles_checkbox_and_radio_indicators_for_dark_theme():
    """Regression guard: Qt's default checkbox/radio indicators can be
    effectively invisible against a dark stylesheet unless explicitly
    styled -- this is what the "Show"/"Active" spectrum-panel controls
    and the Fit Parameters checkboxes rely on."""
    css = qt_stylesheet("dark")
    assert "QCheckBox::indicator" in css
    assert "QRadioButton::indicator" in css
    assert "QCheckBox::indicator:checked" in css
    assert "QRadioButton::indicator:checked" in css


def test_next_color_light_theme_matches_existing_palette():
    assert next_color(0) == LIGHT_COLOR_CYCLE[0]
    assert next_color(0, "light") == "#1f77b4"


def test_next_color_dark_theme_uses_tv_palette():
    # TV's own "colored" X11 resource scheme (tv-1.9.13/etc/Xtv):
    # foreground0 is yellow against a black background.
    assert next_color(0, "dark") == "#FFFF00"
    assert next_color(0, "dark") == DARK_COLOR_CYCLE[0]
    assert next_color(0, "dark") != next_color(0, "light")


def test_next_color_wraps_and_differs_by_theme_at_every_index():
    for i in range(len(DARK_COLOR_CYCLE)):
        assert next_color(i, "dark") == DARK_COLOR_CYCLE[i]
    # Wraps past the palette length, same as the light-theme cycle.
    assert next_color(len(DARK_COLOR_CYCLE), "dark") == DARK_COLOR_CYCLE[0]


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


def test_toggling_dark_theme_recolors_already_loaded_spectra(qapp):
    """A spectrum's trace color should switch to the TV palette the
    moment dark theme is toggled on -- not just for newly-loaded files --
    using the color_index remembered at load time, and switch back when
    toggled off."""
    main_window = MainWindow()
    original_theme = main_window.settings.theme()
    try:
        main_window.dark_theme_action.setChecked(False)  # start from light
        spectrum = LoadedSpectrum("synthetic.txt", [1, 2, 3], next_color(0, "light"))
        spectrum.color_index = 0
        main_window.spectra.append(spectrum)

        main_window.dark_theme_action.setChecked(True)
        assert spectrum.color == "#FFFF00"

        main_window.dark_theme_action.setChecked(False)
        assert spectrum.color == LIGHT_COLOR_CYCLE[0]
    finally:
        main_window.settings.set_theme(original_theme)
        qapp.setStyleSheet(qt_stylesheet(original_theme))


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
