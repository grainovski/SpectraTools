STEP_LEFT_BG = "left_bg"
STEP_RIGHT_BG = "right_bg"
STEP_FIT_REGION = "fit_region"
STEP_MARKING_PEAKS = "marking_peaks"


class FitModeState:
    """Tracks the in-progress region/peak marking sequence for one
    fit-mode session. Pure state -- no Qt/matplotlib dependency."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step = STEP_LEFT_BG
        self.left_bg_region = None
        self.right_bg_region = None
        self.fit_region = None
        self.peak_positions = []

    def add_region(self, lo, hi):
        region = (min(lo, hi), max(lo, hi))
        if self.step == STEP_LEFT_BG:
            self.left_bg_region = region
            self.step = STEP_RIGHT_BG
        elif self.step == STEP_RIGHT_BG:
            self.right_bg_region = region
            self.step = STEP_FIT_REGION
        elif self.step == STEP_FIT_REGION:
            self.fit_region = region
            self.step = STEP_MARKING_PEAKS
        else:
            raise ValueError("Not currently awaiting a region selection")

    def add_peak(self, position):
        if self.step != STEP_MARKING_PEAKS:
            raise ValueError("Not currently marking peaks")
        lo, hi = self.fit_region
        if not (lo <= position <= hi):
            return False
        self.peak_positions.append(position)
        return True

    def ready_to_fit(self):
        return (
            self.left_bg_region is not None
            and self.right_bg_region is not None
            and self.fit_region is not None
            and len(self.peak_positions) > 0
        )

    def ordered_bg_regions(self):
        """Returns (left, right) background regions ordered by mean
        x-coordinate, regardless of which was marked first."""
        a, b = self.left_bg_region, self.right_bg_region
        a_mid = (a[0] + a[1]) / 2
        b_mid = (b[0] + b[1]) / 2
        return (a, b) if a_mid <= b_mid else (b, a)
