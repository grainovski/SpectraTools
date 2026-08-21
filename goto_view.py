"""Go To -- jump the x-axis to one energy or channel and mark it.

Shared by MainWindow and MatrixPanel as a mixin rather than written twice.
Both already expose the same view surface -- `spectra`, `axes`,
`channel_to_display`/`display_to_channel`, `_calibration`/
`_calibration_active`, `_plot_data(xlim_override=...)` and
`fit_controller` -- which is the same duck-typed contract
FitModeController relies on, so there is nothing to abstract beyond
inheriting.

Written as a mixin from the start deliberately: the drag-pan predicate was
copied into both classes and the copies drifted in the one way that
mattered (v4.1.0 audit, finding C1). A projection is a spectrum view, and
anything that works on one of these plots should work on the other by
construction rather than by two people remembering.
"""

import math

from PySide6.QtWidgets import QDialog

from calibration import CalibrationError
from spectrum import GOTO_MARKER_COLOR, goto_channel_window


class GoToMixin:
    #: Channel the last Go To jumped to, or None. A class-level default so
    #: a subclass that forgets to initialise it still behaves, and so the
    #: attribute exists before the first _plot_data call during __init__.
    _goto_marker_channel = None

    def _goto_display_bounds(self):
        """(lo, hi, max_channel) for what the axis can currently show, with
        lo/hi in display units and ascending. None when nothing is visible.

        Sorted because a negative calibration slope makes the display axis
        descend, and both callers want a range to test against rather than
        the axis direction.
        """
        visible = [s for s in self.spectra if s.visible]
        if not visible:
            return None
        max_channel = max(len(s.data) for s in visible) - 1
        if max_channel < 0:
            return None
        first = float(self.channel_to_display(0))
        last = float(self.channel_to_display(max_channel))
        return min(first, last), max(first, last), max_channel

    def open_goto_dialog(self):
        from goto_dialog import GoToDialog

        bounds = self._goto_display_bounds()
        if bounds is None:
            self.fit_controller._show_status_message(
                "Go To needs a visible spectrum.", 4000
            )
            return
        lo, hi, _max_channel = bounds
        # The unit follows what the AXIS shows, not merely whether a
        # calibration exists: one that is loaded but switched off leaves the
        # axis in channels, and asking for keV then would take a number the
        # view cannot use.
        calibrated = self._calibration_active and self._calibration is not None
        dialog = GoToDialog(self, calibrated, lo, hi)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.goto_display_value(dialog.result_value)

    def goto_display_value(self, display_value):
        """Centre the view on `display_value` -- keV or channel, whichever
        the axis is showing -- and leave a mark there.

        Public so a test, or any later caller such as a double-click in the
        results list, can jump without driving the dialog.
        """
        bounds = self._goto_display_bounds()
        if bounds is None:
            return
        _lo, _hi, max_channel = bounds
        try:
            channel = float(self.display_to_channel(display_value))
        except CalibrationError as exc:
            # invert() refuses at a quadratic's vertex, where an energy maps
            # to no single channel.
            self.fit_controller._show_status_message(f"Go To failed: {exc}", 6000)
            return
        if not math.isfinite(channel):
            self.fit_controller._show_status_message(
                "Go To failed: that value does not map to a channel.", 6000
            )
            return
        lo_channel, hi_channel = goto_channel_window(channel, max_channel)
        self._goto_marker_channel = channel
        # Direction preserved rather than sorted: a descending axis has to
        # stay descending, exactly as _plot_data's own full-range xlim does.
        self._plot_data(xlim_override=(
            self.channel_to_display(lo_channel),
            self.channel_to_display(hi_channel),
        ))

    def draw_goto_marker(self):
        """The vertical line Go To leaves at its target. Call from
        `_plot_data` after the spectra are drawn.

        Redrawn every replot from the stored CHANNEL rather than kept as an
        artist, so it survives `axes.clear()` and follows a calibration
        change to the same physical channel instead of stranding itself at a
        stale keV coordinate.
        """
        if self._goto_marker_channel is None:
            return
        self.axes.axvline(
            self.channel_to_display(self._goto_marker_channel),
            color=GOTO_MARKER_COLOR, linestyle=":", linewidth=1.2, zorder=1.5,
        )

    def clear_goto_marker(self, redraw=True):
        """Drops the Go To mark.

        Called by Clear alongside the fit marks: it is a mark on the plot,
        and the one control for removing marks should remove all of them.
        `redraw=False` lets Clear do its own single replot rather than
        painting a frame nobody sees.
        """
        if self._goto_marker_channel is None:
            return
        self._goto_marker_channel = None
        if redraw:
            self._plot_data(preserve_view=True)
