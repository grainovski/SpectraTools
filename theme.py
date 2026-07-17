"""Light/dark theme support: a Qt stylesheet for the app chrome plus
matplotlib axes colors for the plot canvas, switched together by one
`theme` string ("light" or "dark")."""

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
QTableWidget, QListWidget {{
    background-color: {DARK_PANEL};
    color: {DARK_TEXT};
    gridline-color: {DARK_BORDER};
    alternate-background-color: {DARK_BG};
}}
QTableWidget::item:selected, QListWidget::item:selected {{
    background-color: {DARK_HIGHLIGHT};
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

# A single neutral color for plot elements (e.g. the background-level
# line) that need to stay visible against either theme's axes background,
# rather than threading theme state through every plotting call site.
NEUTRAL_LINE_COLOR = "gray"


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
