"""The calibration half of the view surface both spectrum windows share.

MainWindow and MatrixPanel each draw a spectrum against an x axis that is
either channels or keV, and each has to convert between the two in every
place that draws or reads an x-coordinate. That conversion, and the dialog
that changes which calibration is in force, were written twice -- once per
class, character for character.

They had not drifted when this was factored out, but the same pair had
already done exactly that once: the v4.1.0 audit found the drag-pan
predicate copied into both windows with the copies no longer agreeing.
`goto_view.GoToMixin` was the reaction to that finding, and this is the
same remedy applied to the calibration methods it left behind.

What deliberately stays per-class is `_apply_calibration_change`: the two
windows genuinely differ there, since the main window keeps the same
detector-channel region in view across a unit change while the panel has
its own replot path. The mixin calls it through `self`, so each class
keeps its own.

Requires of its host: `_calibration`, `_calibration_active`, and an
`_apply_calibration_change(calibration, active)` method.
"""

from PySide6.QtWidgets import QDialog

from calibration_dialog import CalibrationDialog


class CalibrationViewMixin:
    """Channel/display conversion and the calibration dialog."""

    def channel_to_display(self, channel):
        """Converts a channel number (or numpy array of channel numbers)
        to whatever's on the x-axis right now: the same value if no
        calibration is active, or its calibrated keV equivalent if one
        is. Every place that draws an x-coordinate routes through this."""
        if not self._calibration_active or self._calibration is None:
            return channel
        return self._calibration.apply(channel)

    def display_to_channel(self, display_x):
        """Inverse of channel_to_display -- converts an x-axis
        coordinate (channel or keV, whichever is currently displayed)
        back to a channel number. Every place that reads a click/hover
        x-coordinate routes through this."""
        if not self._calibration_active or self._calibration is None:
            return display_x
        return self._calibration.invert(display_x)

    def _open_calibration_dialog(self):
        dialog = CalibrationDialog(
            self, initial=self._calibration, initially_active=self._calibration_active
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_calibration_change(dialog.result_calibration, dialog.result_active)


def zoomed_limits(xlim, factor, center, display_lo, display_hi):
    """New (lo, hi) for a zoom step, anchored on `center`.

    The channel under `center` keeps its place in the view instead of
    jumping to the middle. Zooming into a peak with the wheel leaves that
    peak under the pointer, so the next tick goes on zooming into the
    same peak rather than into whatever the last one moved there.

    `center` of None means the middle of the current view, which is what
    the keyboard zoom passes. Anchoring on the midpoint is the same
    symmetric zoom as before, so that path is unchanged.

    `xlim` may be descending: a negative-b calibration makes
    channel_to_display decreasing and matplotlib honours the inversion,
    so the input's orientation is preserved on the way out rather than
    silently flipping the axis on every step.

    Clamped to [display_lo, display_hi] -- no zoom may show channels the
    data does not have. Clamping an end necessarily breaks the anchor
    there, which is the honest outcome: beyond the data there is nothing
    to hold still.

    Shared by the main window and the matrix panel's projection, which
    had their own copies of this arithmetic and would otherwise have to
    be corrected in step.
    """
    ascending = xlim[0] <= xlim[1]
    view_lo, view_hi = (xlim[0], xlim[1]) if ascending else (xlim[1], xlim[0])
    if center is None:
        center = (view_lo + view_hi) / 2.0
    new_lo = max(display_lo, center - (center - view_lo) * factor)
    new_hi = min(display_hi, center + (view_hi - center) * factor)
    if new_hi <= new_lo:
        new_hi = min(display_hi, new_lo + 1)
    return (new_lo, new_hi) if ascending else (new_hi, new_lo)
