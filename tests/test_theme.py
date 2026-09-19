import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from main_window import MainWindow
from spectrum import DARK_COLOR_CYCLE, LoadedSpectrum, next_color
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


# --- the light palette, a ramp since 5.2.6 ---------------------------
#
# It used to be matplotlib's tab10 list. It is now a red-to-blue ramp: the
# first spectrum red, the second blue, every later one between them.


def test_light_theme_starts_red_then_blue():
    assert next_color(0, "light") == "#FF0000"
    assert next_color(1, "light") == "#0000FF"
    assert next_color(0) == next_color(0, "light"), "light is the default"


def test_light_theme_fills_in_by_halving_the_widest_gap():
    """Each spectrum after the second splits the largest gap left by the
    ones before it. The ORDER is the point, not just the endpoints: it is
    what keeps a handful of loaded spectra far apart from each other."""
    assert [next_color(i, "light") for i in range(2, 9)] == [
        "#800080",  # 1/2
        "#BF0040",  # 1/4
        "#4000BF",  # 3/4
        "#DF0020",  # 1/8
        "#9F0060",  # 3/8
        "#60009F",  # 5/8
        "#2000DF",  # 7/8
    ]


def test_an_even_march_would_fail_this():
    """Control for the test above. Walking evenly from red to blue is the
    obvious reading of "in between these two limits" and would satisfy any
    check that only asked whether the colours lie on the ramp -- but it
    puts the third spectrum a tenth of the way from the first, where the
    two are hardest to tell apart. Pinning the rejected value stops a later
    "simplification" from quietly reintroducing it."""
    assert next_color(2, "light") != "#E3001C"


def test_every_light_colour_lies_on_the_red_blue_ramp():
    """No green anywhere, and red plus blue always makes a full component:
    the ramp is a straight line between the two endpoints."""
    for i in range(64):
        color = next_color(i, "light")
        r, g, b = (int(color[j:j + 2], 16) for j in (1, 3, 5))
        assert g == 0, "%s has green in it; the ramp is red to blue" % color
        assert abs((r + b) - 255) <= 1, color


def test_the_light_ramp_never_repeats_a_colour():
    """Unlike the dark cycle there is no wrap -- index 10 must not come
    back as index 0, or two loaded spectra would be drawn identically."""
    seen = [next_color(i, "light") for i in range(64)]
    assert len(set(seen)) == len(seen)


def test_the_repeat_check_can_see_a_repeat():
    """Control: a wrapping palette has to fail the assertion above, or it
    proves nothing about the ramp."""
    wrapping = [DARK_COLOR_CYCLE[i % len(DARK_COLOR_CYCLE)] for i in range(64)]
    assert len(set(wrapping)) != len(wrapping)


def test_the_app_icon_keeps_its_own_palette():
    """The icon is the application's mark, on the taskbar and in the
    installer, and it used to be drawn from the light spectrum cycle. It
    must NOT have been dragged onto the ramp along with the traces."""
    from main_window import _APP_ICON_BAR_HEIGHTS, _APP_ICON_COLORS

    assert _APP_ICON_COLORS[:5] == (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    )
    assert len(_APP_ICON_COLORS) >= len(_APP_ICON_BAR_HEIGHTS)
    assert _APP_ICON_COLORS[0] != next_color(0, "light")


# --- the dark palette, which 5.2.6 had to leave exactly alone ---------


def test_next_color_dark_theme_uses_tv_palette():
    # TV's own "colored" X11 resource scheme (tv-1.9.13/etc/Xtv):
    # foreground0 is yellow against a black background.
    assert next_color(0, "dark") == "#FFFF00"
    assert next_color(0, "dark") == DARK_COLOR_CYCLE[0]
    assert next_color(0, "dark") != next_color(0, "light")


def test_the_dark_palette_is_unchanged_at_every_index():
    """Entry for entry, wrap included."""
    for i in range(len(DARK_COLOR_CYCLE)):
        assert next_color(i, "dark") == DARK_COLOR_CYCLE[i]
    assert next_color(len(DARK_COLOR_CYCLE), "dark") == DARK_COLOR_CYCLE[0]
    assert next_color(len(DARK_COLOR_CYCLE) + 3, "dark") == DARK_COLOR_CYCLE[3]


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


def test_the_first_two_loaded_spectra_really_draw_red_and_blue(qapp):
    """End to end, through the real load path and the real axes.

    Every other test here reads spectrum.color, which is the value the app
    intends to draw. This one reads the Line2D matplotlib actually drew, so
    a palette that is right in the model but never reaches the canvas
    cannot pass."""
    import os

    fixtures = os.path.join(os.path.dirname(__file__), "fixtures")
    main_window = MainWindow()
    original_theme = main_window.settings.theme()
    try:
        main_window.dark_theme_action.setChecked(False)
        main_window._load_files([
            os.path.join(fixtures, "test.txt"),
            os.path.join(fixtures, "test1.txt"),
        ])
        assert len(main_window.spectra) == 2

        drawn = [line.get_color() for line in main_window.axes.lines]
        assert drawn[:2] == ["#FF0000", "#0000FF"], drawn
    finally:
        main_window.settings.set_theme(original_theme)
        qapp.setStyleSheet(qt_stylesheet(original_theme))


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
        assert spectrum.color == "#FF0000"
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
