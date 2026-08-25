import types

import numpy as np
from matplotlib.colors import to_rgba

from matrix_heatmap import MatrixHeatmapWindow, _downsample_for_display
from theme import DARK_BG, DARK_TEXT


def _fake_panel():
    # MatrixHeatmapWindow only needs a `panel._heatmap_windows` list to
    # self-remove from on close (see closeEvent) -- a real MatrixPanel
    # would work too, but constructing one means decoding a real matrix
    # fixture, which these tests don't otherwise need.
    return types.SimpleNamespace(_heatmap_windows=[])


def test_matrix_heatmap_window_displays_matrix(qapp):
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "light", _fake_panel())

    assert window.windowTitle() == "Heatmap -- test.mtx"
    assert window.image is not None


def test_matrix_heatmap_window_has_navigation_toolbar_for_zoom(qapp):
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "light", _fake_panel())

    # NavigationToolbar2QT's stock "Zoom" tool is what provides
    # rectangle-select zoom in/out -- confirm it's present (this app's
    # own _TrimmedNavigationToolbar deliberately drops it for 1D
    # spectra, but a 2D image genuinely needs it, so the heatmap uses
    # the stock toolbar, not the trimmed one).
    action_texts = [a.text() for a in window.nav_toolbar.actions()]
    assert "Zoom" in action_texts


def test_matrix_heatmap_axes_are_labeled_in_original_channels_after_downsampling(qapp):
    """The heatmap downsamples for display, but its axes say "channel" --
    without an explicit extent, imshow ticks ran over the downsampled
    BLOCK indices, so "channel 500" on an 8192-channel matrix was really
    channel 4000. The extent pins the ticks to the original channel
    coordinates whatever the downsampling factor. Asymmetric shape so a
    rows/cols (Y/X) swap cannot cancel out."""
    matrix = np.zeros((2048, 1024), dtype=np.int64)  # rows=Y=2048, cols=X=1024
    window = MatrixHeatmapWindow(matrix, "big.mtx", "light", _fake_panel())

    assert _downsample_for_display(matrix).shape == (1024, 512)  # it DID downsample
    assert tuple(window.image.get_extent()) == (0, 1024, 0, 2048)  # (left, right, bottom, top)


def test_matrix_heatmap_extent_matches_channels_without_downsampling_too(qapp):
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "small.mtx", "light", _fake_panel())
    assert tuple(window.image.get_extent()) == (0, 10, 0, 10)


def test_matrix_heatmap_handles_negative_values_without_crashing(qapp):
    matrix = np.array([[-5, 10], [20, -1]], dtype=np.int64)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "light", _fake_panel())
    assert window.image is not None


def test_matrix_heatmap_window_applies_theme_to_plot_and_colorbar(qapp):
    # Regression guard: passing theme="dark" must actually change the
    # rendered colors, not just be accepted as an argument -- a test that
    # only checked "constructor doesn't crash" would pass even if the
    # style_axes() calls were silently removed from __init__, exactly the
    # class of "all tests green, feature dead in real use" bug this app's
    # own history has already hit twice (matrix_panel.py's focus-policy
    # and eventFilter bugs, caught in Task 4's review).
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "dark", _fake_panel())

    # style_axes(self.axes, theme) sets the *figure's* facecolor, shared
    # by every Axes drawn in it (including the colorbar's own).
    assert window.figure.get_facecolor() == to_rgba(DARK_BG)

    # The colorbar draws its tick labels on its own separate Axes
    # (self.colorbar.ax), distinct from self.axes -- style_axes(self.axes,
    # ...) alone can't reach them. Confirms the second, explicit
    # style_axes(self.colorbar.ax, theme) call is what keeps the colorbar's
    # tick numbers legible (not left at matplotlib's hardcoded black,
    # which would be illegible against the now-dark figure background).
    tick_colors = {t.get_color() for t in window.colorbar.ax.get_yticklabels()}
    assert tick_colors == {DARK_TEXT}


def test_matrix_heatmap_window_removes_itself_from_panel_on_close(qapp):
    # Regression guard: closing a heatmap window directly (its own
    # titlebar X, not via the owning panel) must prune it from
    # panel._heatmap_windows -- that list's Python-side reference is what
    # keeps this parentless top-level window alive, so without pruning, a
    # closed-but-still-referenced window would be retained for the rest of
    # the panel's lifetime, growing unboundedly across repeat open/close.
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    panel = _fake_panel()
    window = MatrixHeatmapWindow(matrix, "test.mtx", "light", panel)
    panel._heatmap_windows.append(window)

    window.close()

    assert window not in panel._heatmap_windows


def _opaque_icon_colors(icon, size=24):
    """Same technique as tests/test_theme.py's own helper of the same
    name -- the set of distinct, non-transparent pixel colors an icon
    actually renders. Sampling a single fixed coordinate is unreliable
    since it can easily land on a transparent gap between glyph
    strokes, and comparing QIcon objects for equality/identity would
    not reliably catch a stale-palette bug either (Qt icon objects can
    compare unequal for irrelevant reasons, or equal despite different
    rendered pixels)."""
    image = icon.pixmap(size, size).toImage()
    colors = set()
    for x in range(size):
        for y in range(size):
            color = image.pixelColor(x, y)
            if color.alpha() > 10:
                colors.add(color.name())
    return colors


def _save_icon(nav_toolbar):
    return next(a for a in nav_toolbar.actions() if a.text() == "Save").icon()


def test_heatmap_window_toolbar_icons_follow_theme_toggle(qapp):
    """Regression guard: matrix_heatmap.py builds its own separate
    nav_toolbar, entirely independent of MainWindow's -- before this
    fix, matplotlib's built-in Home/Pan/Save icons never re-rendered
    when the theme changed after the window was already open (there
    was no refresh hook of any kind, since this window "never redrew
    after construction")."""
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "light", _fake_panel())
    assert _opaque_icon_colors(_save_icon(window.nav_toolbar)) == {"#000000"}

    window._refresh_theme("dark")
    assert _opaque_icon_colors(_save_icon(window.nav_toolbar)) == {"#ffffff"}

    window._refresh_theme("light")
    assert _opaque_icon_colors(_save_icon(window.nav_toolbar)) == {"#000000"}


def test_heatmap_window_constructed_already_dark_gets_dark_toolbar_icons(qapp):
    """The other realistic case besides a live refresh: a heatmap
    window opened while the app is ALREADY in dark theme must build
    its toolbar with dark-correct icons from construction."""
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "dark", _fake_panel())
    assert _opaque_icon_colors(_save_icon(window.nav_toolbar)) == {"#ffffff"}


def test_heatmap_window_toolbar_palette_background_reflects_theme(qapp):
    """Same regression guard as tests/test_theme.py's own
    test_nav_toolbar_palette_background_reflects_theme, mirrored for
    the heatmap window's own separate nav_toolbar."""
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "light", _fake_panel())
    light_value = window.nav_toolbar.palette().color(window.nav_toolbar.backgroundRole()).value()
    assert light_value >= 128

    window._refresh_theme("dark")
    dark_value = window.nav_toolbar.palette().color(window.nav_toolbar.backgroundRole()).value()
    assert dark_value < 128


def test_heatmap_window_refresh_theme_recolors_axes_and_colorbar(qapp):
    """Companion regression guard, one level beyond the toolbar icons
    themselves: _refresh_theme must also re-apply style_axes to both
    the plot axes and the colorbar's own separate axes (exactly what
    test_matrix_heatmap_window_applies_theme_to_plot_and_colorbar
    already checks at construction time, mirrored here for a LATER
    call)."""
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)
    window = MatrixHeatmapWindow(matrix, "test.mtx", "light", _fake_panel())
    assert window.figure.get_facecolor() == to_rgba("white")

    window._refresh_theme("dark")
    assert window.figure.get_facecolor() == to_rgba(DARK_BG)
    tick_colors = {t.get_color() for t in window.colorbar.ax.get_yticklabels()}
    assert tick_colors == {DARK_TEXT}


def test_downsample_for_display_block_sums_correctly():
    matrix = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.int64,
    )

    result = _downsample_for_display(matrix, max_dim=2)

    # Hand-computed 2x2 block sums -- top-left block is rows 0-1/cols 0-1
    # (1+2+5+6=14), top-right is rows 0-1/cols 2-3 (3+4+7+8=22), bottom-left
    # is rows 2-3/cols 0-1 (9+10+13+14=46), bottom-right is rows 2-3/cols
    # 2-3 (11+12+15+16=54). Sums, not averages, matching this app's
    # existing rebin convention (spectrum_operations.py): counts add, they
    # don't average, when a histogram's binning is coarsened.
    expected = np.array([[14, 22], [46, 54]])
    np.testing.assert_array_equal(result, expected)


def test_downsample_for_display_passes_through_small_matrix_unchanged():
    # A matrix already at or under max_dim in both dimensions should take
    # the factor == 1 early-return path and come back unchanged -- the
    # default max_dim (1024) comfortably covers this 10x10 matrix.
    matrix = np.arange(100, dtype=np.int64).reshape(10, 10)

    result = _downsample_for_display(matrix)

    np.testing.assert_array_equal(result, matrix)
