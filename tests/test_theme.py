import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from main_window import MainWindow
from spectrum import DARK_COLOR_CYCLE, LIGHT_COLOR_CYCLE, LoadedSpectrum, next_color
from theme import DARK_BG, fit_drawing_colors, qt_stylesheet, style_axes


def test_fit_drawing_colors_never_matches_the_spectrum_color():
    """Regression guard for the reported "blue fit on a blue spectrum"
    clash: whatever hue a spectrum happens to be drawn in, the computed
    fit/background-line colors must land on a distinctly different hue,
    for both themes."""
    import colorsys
    from theme import _hex_to_rgb01

    for spectrum_color in ("#1f77b4", "#FFFF00", "#FF0000", "#00FF00", "#7f7f7f"):
        spectrum_hue, _l, _s = colorsys.rgb_to_hls(*_hex_to_rgb01(spectrum_color))
        for theme in ("light", "dark"):
            fit_color, bg_color = fit_drawing_colors(spectrum_color, theme)
            for drawn_color in (fit_color, bg_color):
                drawn_hue, _l, _s = colorsys.rgb_to_hls(*_hex_to_rgb01(drawn_color))
                hue_distance = min(
                    abs(drawn_hue - spectrum_hue), 1 - abs(drawn_hue - spectrum_hue)
                )
                assert hue_distance > 0.15, (
                    f"{drawn_color} too close in hue to spectrum color {spectrum_color}"
                )


def test_fit_drawing_colors_differ_between_fit_and_background_line():
    fit_color, bg_color = fit_drawing_colors("#1f77b4", "dark")
    assert fit_color != bg_color


def test_fit_drawing_colors_differ_between_themes_for_the_same_spectrum():
    assert fit_drawing_colors("#1f77b4", "light") != fit_drawing_colors("#1f77b4", "dark")


def test_fit_drawing_colors_differ_by_spectrum_color():
    assert fit_drawing_colors("#1f77b4", "dark") != fit_drawing_colors("#FFFF00", "dark")


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


def _opaque_icon_colors(icon, size=24):
    """The set of distinct, non-transparent pixel colors an icon
    actually renders -- sampling a single fixed coordinate is
    unreliable since it can easily land on a transparent gap between
    glyph strokes."""
    image = icon.pixmap(size, size).toImage()
    colors = set()
    for x in range(size):
        for y in range(size):
            color = image.pixelColor(x, y)
            if color.alpha() > 10:
                colors.add(color.name())
    return colors


def test_zoom_icons_and_builtin_save_icon_match_color_in_both_themes(qapp):
    """The custom Zoom In X/Zoom Out X/Show Full Spectrum icons should
    render in exactly the same color as matplotlib's own built-in icons
    (e.g. Save) in both themes -- black in light theme, white in dark
    theme -- not a fixed color regardless of theme."""
    main_window = MainWindow()
    original_theme = main_window.settings.theme()
    try:
        save_action = next(
            a for a in main_window.nav_toolbar.actions() if a.text() == "Save"
        )
        for theme, expected in (("light", "#000000"), ("dark", "#ffffff")):
            main_window.dark_theme_action.setChecked(theme == "dark")
            assert _opaque_icon_colors(save_action.icon()) == {expected}
            assert _opaque_icon_colors(main_window.zoom_in_action.icon()) == {expected}
            assert _opaque_icon_colors(main_window.zoom_out_action.icon()) == {expected}
            assert _opaque_icon_colors(main_window.full_spectrum_action.icon()) == {expected}
            assert _opaque_icon_colors(main_window.calibration_load_action.icon()) == {expected}
            assert _opaque_icon_colors(main_window.calibration_toggle_action.icon()) == {expected}
    finally:
        main_window.settings.set_theme(original_theme)


def test_nav_toolbar_palette_background_reflects_theme(qapp):
    """Regression guard: matplotlib's own dark-mode icon detection
    (NavigationToolbar2QT's _IconEngine) reads the toolbar's *QPalette*,
    which Qt stylesheet background-color rules alone don't update --
    without explicitly setting the palette too, Home/Pan/Save would
    silently stay black (low-contrast) under dark theme regardless of
    this app's own QSS."""
    main_window = MainWindow()
    original_theme = main_window.settings.theme()
    try:
        main_window.dark_theme_action.setChecked(False)
        light_value = main_window.nav_toolbar.palette().color(
            main_window.nav_toolbar.backgroundRole()
        ).value()
        assert light_value >= 128

        main_window.dark_theme_action.setChecked(True)
        dark_value = main_window.nav_toolbar.palette().color(
            main_window.nav_toolbar.backgroundRole()
        ).value()
        assert dark_value < 128
    finally:
        main_window.settings.set_theme(original_theme)
        qapp.setStyleSheet(qt_stylesheet(original_theme))
