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
    def __init__(self, path, data, color, variance=None):
        self.path = path
        self.data = data
        self.color = color
        self.visible = True
        self.active = False
        self.fits = []
        # Per-channel variance, or None for a spectrum read straight from
        # a file. None means "assume Poisson", which is correct for raw
        # counts and wrong for anything derived: a matrix cut or an
        # Add/Subtract result has variance larger than its own counts and
        # can go negative, where no Poisson error exists at all. Only the
        # operations that know the propagated variance set this; see
        # matrix_cut.compute_cut_with_variance and
        # spectrum_operations.combined_variance.
        self.variance = variance
        # What the calibration dialog was last told for this spectrum --
        # an EnergyAssignments, or None. Kept so refitting does not
        # discard the user's identifications; see energy_assignments.py.
        # Session-lived: never written to disk, never saved with fits.
        self.energy_assignments = None


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


#: The Go To mark. Distinct from every existing mark colour on purpose --
#: background regions are green, fit regions blue, matrix cuts red -- so a
#: jump target is never mistaken for something the fit will act on. Dotted
#: and thin for the same reason: it is a bookmark, not a region.
GOTO_MARKER_COLOR = "tab:purple"

#: How wide a view Go To leaves behind, in CHANNELS. Deliberately a
#: channel count rather than a keV span: it means the same thing whether or
#: not a calibration is active, and it stays sensible across coarse and fine
#: binning, where a fixed keV span would be far too wide on one and far too
#: narrow on the other. Ctrl+= / Ctrl+- adjust from there.
GOTO_WINDOW_CHANNELS = 100


def goto_channel_window(channel, max_channel, width=GOTO_WINDOW_CHANNELS):
    """(lo, hi) channel bounds centred on `channel`, clamped to
    [0, max_channel], keeping the full `width` wherever the data allows.

    Works in CHANNELS rather than display units on purpose. With a
    quadratic calibration a fixed keV span is not a fixed channel span, and
    centring in keV would put the target off-centre in the data; converting
    the two ends afterwards keeps the window exactly `width` channels wide
    and the target exactly in the middle of it.

    Clamping SHIFTS rather than truncates. A target near channel 0 would
    otherwise get half a window, with the peak pinned against the edge --
    the view slides inward instead, so a line at channel 5 is still shown
    with real context around it. The width is only reduced when the whole
    spectrum is narrower than the window, where there is nothing else to
    show.
    """
    if max_channel < 0:
        raise ValueError("max_channel must be >= 0")
    width = min(float(width), float(max_channel))
    half = width / 2.0
    lo = float(channel) - half
    hi = float(channel) + half
    if lo < 0.0:
        hi -= lo          # push the window right by however far it overhung
        lo = 0.0
    if hi > max_channel:
        lo -= hi - max_channel
        hi = float(max_channel)
    return max(0.0, lo), min(float(max_channel), hi)


def pan_button_is_active(event, axes, nav_toolbar, controllers):
    """True when this mouse press should start a drag-pan.

    Shared by the main window and the matrix panel, which offer the same
    gesture on the same kind of view. It lived in both files, and the
    copies had already drifted in the one way that matters: the panel
    carries BOTH cut and fit marking and so has two controllers to
    consult, where the main window has one. A copy that consulted only the
    first would let a drag nudge the view in the middle of placing a mark,
    and nothing in a diff of either file alone would show it. Passing the
    controllers in makes "which controllers can veto a pan" the caller's
    single visible decision.

    The RIGHT button always pans -- that is the v3.1.3 gesture, and it
    keeps working with a marking key held, which is useful for scrolling
    along while keeping a key down.

    The LEFT button pans only when nothing else wants it:

      * A held marking key takes precedence absolutely. Marking is precise
        work and must not depend on how steady the hand is.
      * The matplotlib toolbar's own Pan and Zoom tools drive the left
        button themselves. While either is armed, `nav_toolbar.mode` is
        non-empty and this stands down, or both would act on one drag.

    Outside those, a bare left press does nothing at all -- fit_mode's
    on_click returns early without a held key -- which is what made the
    gesture free to take in v4.0.1.
    """
    if event.inaxes != axes or event.x is None:
        return False
    if event.button == 3:
        return True
    if event.button != 1:
        return False
    if any(controller._held_key is not None for controller in controllers):
        return False
    return not str(nav_toolbar.mode)


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
