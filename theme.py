"""Light/dark theme support: a Qt stylesheet for the app chrome plus
matplotlib axes colors for the plot canvas, switched together by one
`theme` string ("light" or "dark")."""

import colorsys

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtGui import QColor, QPalette

DARK_BG = "#1e1e1e"
DARK_PANEL = "#2b2b2b"
DARK_TEXT = "#e0e0e0"
DARK_BORDER = "#3f3f3f"
DARK_HIGHLIGHT = "#3a6ea5"

DARK_QSS = f"""
QMainWindow, QDialog, QWidget {{
    background-color: {DARK_BG};
    color: {DARK_TEXT};
}}
QMenuBar {{
    background-color: {DARK_PANEL};
    color: {DARK_TEXT};
}}
QMenuBar::item:selected {{
    background-color: {DARK_HIGHLIGHT};
}}
QMenu {{
    background-color: {DARK_PANEL};
    color: {DARK_TEXT};
    border: 1px solid {DARK_BORDER};
}}
QMenu::item:selected {{
    background-color: {DARK_HIGHLIGHT};
}}
QDockWidget {{
    color: {DARK_TEXT};
    titlebar-close-icon: none;
}}
QDockWidget::title {{
    background-color: {DARK_PANEL};
    padding: 4px;
}}
QToolBar {{
    background-color: {DARK_PANEL};
    border: none;
    spacing: 4px;
}}
QStatusBar {{
    background-color: {DARK_PANEL};
    color: {DARK_TEXT};
}}
QTableWidget, QListWidget, QTreeWidget {{
    background-color: {DARK_PANEL};
    color: {DARK_TEXT};
    gridline-color: {DARK_BORDER};
    alternate-background-color: {DARK_BG};
}}
QTableWidget::item:selected, QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {DARK_HIGHLIGHT};
}}
/* Qt draws a selection in its INACTIVE palette when the widget does not
   hold focus, which is a pale grey that all but disappears against a dark
   panel -- the ROOT object list opens with its first row selected and it
   read as nothing being selected at all. Same highlight either way; the
   selection is real whether or not the widget happens to have focus. */
QTableWidget::item:selected:!active,
QListWidget::item:selected:!active,
QTreeWidget::item:selected:!active {{
    background-color: {DARK_HIGHLIGHT};
    color: {DARK_TEXT};
}}
QHeaderView::section {{
    background-color: {DARK_PANEL};
    color: {DARK_TEXT};
    border: 1px solid {DARK_BORDER};
}}
QPushButton {{
    background-color: {DARK_PANEL};
    color: {DARK_TEXT};
    border: 1px solid {DARK_BORDER};
    padding: 4px 8px;
}}
QPushButton:hover {{
    background-color: {DARK_HIGHLIGHT};
}}
QPushButton:disabled {{
    color: #808080;
}}
QLineEdit, QTextEdit, QAbstractSpinBox {{
    background-color: {DARK_PANEL};
    color: {DARK_TEXT};
    border: 1px solid {DARK_BORDER};
}}
QCheckBox, QRadioButton, QLabel {{
    color: {DARK_TEXT};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 13px;
    height: 13px;
    background-color: {DARK_PANEL};
    border: 1px solid {DARK_TEXT};
}}
QRadioButton::indicator {{
    border-radius: 7px;
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {DARK_HIGHLIGHT};
    border: 1px solid {DARK_HIGHLIGHT};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border: 1px solid {DARK_BORDER};
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background-color: {DARK_PANEL};
}}
QToolTip {{
    background-color: {DARK_PANEL};
    color: {DARK_TEXT};
    border: 1px solid {DARK_BORDER};
}}
"""

# matplotlib axes colors, keyed by theme name.
_AXES_COLORS = {
    "light": {"bg": "white", "fg": "black", "grid": "#b0b0b0"},
    "dark": {"bg": DARK_BG, "fg": DARK_TEXT, "grid": "#444444"},
}

# A single neutral color for plot elements that need to stay visible
# against either theme's axes background. No longer used by
# fit_drawing_colors() itself (light theme now gets its own pair below),
# kept for any other call site that still wants a theme-agnostic line.
NEUTRAL_LINE_COLOR = "gray"

# TV's own fit-drawing colors (tv-1.9.13/etc/Xtv): fit-function.foreground0
# is gold, bg-function.foreground0 is green (X11's pure #00FF00, not CSS's
# darker #008000). Documented here for reference, but no longer used
# directly by fit_drawing_colors() below -- a fixed color (TV's or its
# complement) can land arbitrarily close to whichever color a given
# spectrum happens to be drawn in (e.g. dark theme's own first spectrum
# color is yellow, right next to gold; light theme's default first
# spectrum color is tab:blue, right next to blue, this scheme's original
# light-theme fit color -- both clash). fit_drawing_colors() instead
# derives the fit/background-line colors from the specific spectrum's own
# color, guaranteeing separation regardless of which color that turns out
# to be.
TV_FIT_COLOR = "#FFD700"  # gold
TV_BACKGROUND_COLOR = "#00FF00"  # green (X11)


def _hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb01_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(
        *(round(max(0.0, min(1.0, c)) * 255) for c in rgb)
    )


def fit_drawing_colors(spectrum_color, theme):
    """(fit_color, background_line_color) for drawing a committed fit or
    integration result over a spectrum plotted in `spectrum_color`. Both
    are hues complementary to that specific spectrum's own color (a 180
    degree hue rotation, plus a further offset between the two so they
    don't match each other either) -- not a fixed per-theme color -- so
    they never blend into the trace they're drawn on top of, whichever
    color that trace happens to be. Rendered at full saturation and a
    lightness tuned per theme (bright for a dark background, a medium
    shade for a light one) rather than reusing the spectrum color's own
    saturation/lightness, since a desaturated or very light/dark spectrum
    color would otherwise produce a washed-out, still-hard-to-see result."""
    h, _l, _s = colorsys.rgb_to_hls(*_hex_to_rgb01(spectrum_color))
    fit_hue = (h + 0.5) % 1.0
    bg_hue = (fit_hue + 0.19) % 1.0
    lightness = 0.60 if theme == "dark" else 0.42
    fit_color = _rgb01_to_hex(colorsys.hls_to_rgb(fit_hue, lightness, 1.0))
    bg_color = _rgb01_to_hex(colorsys.hls_to_rgb(bg_hue, lightness, 1.0))
    return fit_color, bg_color


def qt_stylesheet(theme):
    """Returns the QApplication-wide stylesheet for `theme` ("light" or
    "dark") -- "" for light, since that's Qt's own unmodified default."""
    return DARK_QSS if theme == "dark" else ""


def style_axes(axes, theme):
    """Applies `theme`'s colors to a matplotlib Axes -- figure/axes
    background, tick/label/title color, spine color, and grid color.
    Must be called again after every axes.clear(), which resets all of
    this back to matplotlib's own defaults."""
    colors = _AXES_COLORS.get(theme, _AXES_COLORS["light"])
    axes.figure.set_facecolor(colors["bg"])
    axes.set_facecolor(colors["bg"])
    axes.xaxis.label.set_color(colors["fg"])
    axes.yaxis.label.set_color(colors["fg"])
    axes.title.set_color(colors["fg"])
    axes.tick_params(colors=colors["fg"], labelcolor=colors["fg"])
    for spine in axes.spines.values():
        spine.set_color(colors["fg"])
    axes.grid(True, color=colors["grid"])


def style_nav_toolbar_palette(nav_toolbar, theme):
    """Sets a matplotlib NavigationToolbar2QT's actual QPalette -- not
    just this app's own QSS. A QSS rule alone changes the toolbar's
    *paint* immediately, but a `.palette()` query only starts
    reflecting it once Qt has actually shown/polished that specific
    widget under the new stylesheet -- an as-yet-unshown toolbar (as
    in most of this app's own tests, which construct windows without
    ever calling .show()) keeps returning Qt's original light-mode
    colors regardless of the app-wide QSS. matplotlib's own icon
    rendering (NavigationToolbar2QT._icon, see refresh_builtin_toolbar_icons
    below) reads exactly this palette's background/foreground to decide
    whether and which color to recolor Home/Pan/Save for a dark
    background, so setting it explicitly here -- rather than waiting on
    Qt's own show/polish timing, which this app has no control over --
    is what guarantees the icons are correct immediately and
    deterministically, not just eventually/incidentally (and _icon()
    itself never re-runs on its own afterward regardless -- see
    refresh_builtin_toolbar_icons's own docstring). The foreground is
    set to plain white here
    (not this app's usual DARK_TEXT) so those re-rendered icons come out
    the same exact color as this app's own hand-drawn zoom/calibration
    icons, which are also plain white/black rather than DARK_TEXT --
    these toolbars have no visible text labels (icon-only buttons), so
    there's no legibility trade-off to plain white over DARK_TEXT here.

    Free function (not a method) because every top-level window that
    builds its own separate nav_toolbar -- MainWindow, MatrixPanel,
    MatrixHeatmapWindow -- needs to run exactly this same logic on its
    own toolbar; each one calls this directly rather than one
    borrowing a private method off another via an unrelated `self`."""
    palette = nav_toolbar.palette()
    if theme == "dark":
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Button):
            palette.setColor(role, QColor(DARK_PANEL))
        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
            palette.setColor(role, QColor("white"))
    else:
        palette = QPalette()
    nav_toolbar.setPalette(palette)


def refresh_builtin_toolbar_icons(nav_toolbar):
    """Re-renders a matplotlib NavigationToolbar2QT's own built-in
    Home/Pan/Save icons to match whatever theme style_nav_toolbar_palette
    (above) was just applied to it with.

    NavigationToolbar2QT._icon() bakes each icon into a static QPixmap
    exactly once, at whatever moment it's called -- confirmed against
    the installed matplotlib version that this never happens again on
    its own afterward, no matter how the toolbar's palette later
    changes (unlike the palette query itself, which Qt's own
    stylesheet cascade CAN update on a shown/polished widget with no
    code of ours involved -- see style_nav_toolbar_palette's own
    docstring). So without an explicit call like this one, right after
    every theme change, the icons silently stay whatever color matched
    the palette the last time _icon() happened to run -- which may be
    all the way back at toolbar construction."""
    image_files = {
        text: image_file
        for text, _tooltip, image_file, _callback in nav_toolbar.toolitems
        if text is not None
    }
    for action in nav_toolbar.actions():
        image_file = image_files.get(action.text())
        if image_file is not None:
            action.setIcon(NavigationToolbar2QT._icon(nav_toolbar, image_file + ".png"))
