LIGHT_COLOR_CYCLE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)

# TV's own "colored" X11 resource palette (tv-1.9.13/etc/Xtv,
# Xtv.*colored*foreground0 through foreground15), used against TV's own
# black background -- the dark-theme analog of LIGHT_COLOR_CYCLE. Values
# are the standard X11 rgb.txt colors for each named entry (X11 "green"
# is pure #00FF00, distinct from CSS's darker #008000), kept verbatim
# rather than adjusted for "nicer" contrast, to actually match TV's own
# look rather than just being inspired by it.
DARK_COLOR_CYCLE = (
    "#FFFF00",  # yellow
    "#FF00FF",  # magenta
    "#FF0000",  # red
    "#0000FF",  # blue
    "#FFFFFF",  # white
    "#F5DEB3",  # wheat
    "#00FFFF",  # cyan
    "#FFC0CB",  # pink
    "#BA55D3",  # medium orchid
    "#8B8682",  # seashell4
    "#B22222",  # firebrick
    "#8470FF",  # light slate blue
    "#7FFFD4",  # aquamarine
    "#ADD8E6",  # light blue
    "#00FF00",  # green (X11)
    "#D59027",
)

COLOR_CYCLE = LIGHT_COLOR_CYCLE  # kept as the default/light-theme cycle


class LoadedSpectrum:
    def __init__(self, path, data, color):
        self.path = path
        self.data = data
        self.color = color
        self.visible = True
        self.active = False
        self.fits = []


def active_spectrum(spectra):
    """The one spectrum flagged active, or None if there isn't one.

    Lives here rather than as a MainWindow method because both
    MainWindow and MatrixPanel own a `spectra` list and both are used
    interchangeably as FitModeController's `main_window` (the panel
    duck-types the same surface) -- a method on MainWindow alone would
    quietly become part of that contract. Callers pass the list itself,
    so neither class has to grow anything.

    'Active' is a radio-button selection: at most one spectrum has it
    set, and None is a normal, expected result (nothing loaded, or the
    active spectrum was just closed), never an error.
    """
    return next((s for s in spectra if s.active), None)


def panned_xlim(xlim, delta, bound_a, bound_b):
    """`xlim` shifted by `delta`, keeping its span, clamped so the view
    never leaves the data.

    Shared by the main window and the matrix panel, which both offer the
    same right-drag pan and would otherwise each need this arithmetic.
    Lives here, with next_color/active_spectrum, because it is pure and
    Qt-free.

    The span is preserved exactly rather than recomputed from the clamped
    ends: a pan that runs into an edge should stop, not squash the view.
    Order is preserved too -- a negative-b calibration makes
    channel_to_display decreasing, so xlim is legitimately descending and
    rewriting it ascending would flip the axis mid-drag. `bound_a`/
    `bound_b` are accepted in either order for the same reason.
    """
    lo_bound, hi_bound = min(bound_a, bound_b), max(bound_a, bound_b)
    start, end = xlim[0] + delta, xlim[1] + delta
    lo, hi = min(start, end), max(start, end)
    if hi - lo >= hi_bound - lo_bound:
        # The view is already at least as wide as the data: pinning it to
        # the full extent is the only position that keeps the span AND
        # respects both bounds, so there is nothing to pan.
        return xlim
    if lo < lo_bound:
        shift = lo_bound - lo
    elif hi > hi_bound:
        shift = hi_bound - hi
    else:
        shift = 0.0
    return (start + shift, end + shift)


def next_color(index, theme="light"):
    cycle = DARK_COLOR_CYCLE if theme == "dark" else LIGHT_COLOR_CYCLE
    return cycle[index % len(cycle)]
