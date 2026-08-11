from fit_mode import BG_REGION_CAP, FitModeState


def test_initial_state():
    state = FitModeState()
    assert state.bg_regions == []
    assert state.pending_bg_click is None
    assert state.fit_region is None
    assert state.pending_fit_click is None
    assert state.peak_positions == []


def test_bg_click_pairs_into_one_region():
    state = FitModeState()
    assert state.add_bg_click(70) is None
    assert state.pending_bg_click == 70
    completed = state.add_bg_click(85)
    assert completed == (70, 85)
    assert state.pending_bg_click is None
    assert state.bg_regions == [(70, 85)]


def test_bg_click_normalizes_reversed_order():
    state = FitModeState()
    state.add_bg_click(130)
    completed = state.add_bg_click(115)
    assert completed == (115, 130)


def test_bg_region_ring_buffer_evicts_oldest_after_two():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    assert state.bg_regions == [(70, 85), (115, 130)]

    state.add_bg_click(200)
    state.add_bg_click(210)
    assert state.bg_regions == [(115, 130), (200, 210)]


def test_fit_region_click_pairs_and_overwrites_on_new_pair():
    state = FitModeState()
    completed = state.add_fit_click(85)
    assert completed is None
    completed = state.add_fit_click(115)
    assert completed == (85, 115)
    assert state.fit_region == (85, 115)

    state.add_fit_click(90)
    state.add_fit_click(110)
    assert state.fit_region == (90, 110)


def test_completing_a_new_fit_region_drops_out_of_bounds_peaks():
    """Regression test: a peak marked for a since-abandoned fit region
    must not silently survive into a newly-marked, disjoint region --
    it isn't a valid initial guess there and previously caused the next
    fit_peaks() call to fail outright (the optimizer's starting point
    landed outside the new region)."""
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    state.toggle_peak(100.0, proximity=1.0)
    assert state.peak_positions == [100.0]

    # A brand new, disjoint region -- the old peak (100.0) is now
    # outside [135, 165] and must be dropped.
    state.add_fit_click(135)
    state.add_fit_click(165)
    assert state.peak_positions == []


def test_completing_a_new_fit_region_keeps_peaks_still_inside_it():
    """A peak that happens to still fall inside a re-marked/adjusted
    region (not a wholesale new one) is kept, not unconditionally
    wiped."""
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    state.toggle_peak(100.0, proximity=1.0)

    # Slightly widened region -- 100.0 is still inside it.
    state.add_fit_click(80)
    state.add_fit_click(120)
    assert state.peak_positions == [100.0]


def test_peak_toggle_requires_a_fit_region():
    state = FitModeState()
    assert state.toggle_peak(100.0, proximity=1.0) is None
    assert state.peak_positions == []


def test_peak_toggle_rejects_position_outside_fit_region():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.toggle_peak(200.0, proximity=1.0) is None
    assert state.peak_positions == []


def test_peak_toggle_adds_then_removes():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)

    added = state.toggle_peak(100.0, proximity=1.0)
    assert added == ("added", 100.0)
    assert state.peak_positions == [100.0]

    removed = state.toggle_peak(100.4, proximity=1.0)
    assert removed == ("removed", 100.0)
    assert state.peak_positions == []


def test_peak_toggle_respects_proximity_threshold():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    state.toggle_peak(100.0, proximity=1.0)

    added = state.toggle_peak(105.0, proximity=1.0)
    assert added == ("added", 105.0)
    assert state.peak_positions == [100.0, 105.0]


def test_ready_to_fit_requires_two_bg_regions_a_fit_region_and_a_peak():
    state = FitModeState()
    assert state.ready_to_fit() is False
    state.add_bg_click(70)
    state.add_bg_click(85)
    assert state.ready_to_fit() is False
    state.add_bg_click(115)
    state.add_bg_click(130)
    assert state.ready_to_fit() is False
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.ready_to_fit() is False
    state.toggle_peak(100.0, proximity=1.0)
    assert state.ready_to_fit() is True


def test_reset_clears_everything():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.add_fit_click(85)
    state.add_fit_click(115)
    state.toggle_peak(100.0, proximity=1.0)

    state.reset()

    assert state.bg_regions == []
    assert state.fit_region is None
    assert state.peak_positions == []


def test_ordered_bg_regions_regardless_of_marking_order():
    state = FitModeState()
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.add_bg_click(70)
    state.add_bg_click(85)

    left, right = state.ordered_bg_regions()
    assert left == (70, 85)
    assert right == (115, 130)


def test_ready_to_integrate_true_with_zero_bg_regions_and_a_fit_region():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.ready_to_integrate() is True


def test_ready_to_integrate_false_with_exactly_one_bg_region():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.ready_to_integrate() is False


def test_ordered_bg_regions_returns_none_pair_when_no_regions_marked():
    state = FitModeState()
    assert state.ordered_bg_regions() == (None, None)


def test_bg_region_cap_is_two():
    assert BG_REGION_CAP == 2


def test_pending_clicks_are_independent_across_bg_fit_and_peak():
    state = FitModeState()
    state.add_bg_click(70)          # pending bg click, not yet a region
    state.add_fit_click(85)
    state.add_fit_click(115)        # completes fit region
    state.toggle_peak(100.0, proximity=1.0)  # adds a peak
    state.add_bg_click(85)          # completes the bg pair from earlier

    assert state.bg_regions == [(70, 85)]
    assert state.fit_region == (85, 115)
    assert state.peak_positions == [100.0]


def test_fit_blocked_reason_when_no_fit_region():
    state = FitModeState()
    assert state.fit_blocked_reason() == "Mark the fit region (hold R and click twice) before fitting"


def test_fit_blocked_reason_when_bg_regions_incomplete():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.fit_blocked_reason() == "Mark two background regions (hold B and click twice per region) before fitting"


def test_fit_blocked_reason_when_no_peaks():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.fit_blocked_reason() == "Mark at least one peak (hold P and click) before fitting"


def test_fit_blocked_reason_is_none_when_ready():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.add_fit_click(85)
    state.add_fit_click(115)
    state.toggle_peak(100.0, proximity=1.0)
    assert state.ready_to_fit() is True
    assert state.fit_blocked_reason() is None


def test_integrate_blocked_reason_when_no_fit_region():
    state = FitModeState()
    assert state.integrate_blocked_reason() == "Mark the fit region (hold R and click twice) before integrating"


def test_integrate_blocked_reason_when_exactly_one_bg_region():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.integrate_blocked_reason() == "Mark zero or two background regions (not one) before integrating"


def test_integrate_blocked_reason_is_none_with_zero_bg_regions():
    state = FitModeState()
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.ready_to_integrate() is True
    assert state.integrate_blocked_reason() is None


def test_integrate_blocked_reason_is_none_with_two_bg_regions():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.add_fit_click(85)
    state.add_fit_click(115)
    assert state.ready_to_integrate() is True
    assert state.integrate_blocked_reason() is None


def test_background_blocked_reason_when_no_bg_regions():
    state = FitModeState()
    assert (
        state.background_blocked_reason()
        == "Mark two background regions (hold B and click twice per region) before previewing the background"
    )


def test_background_blocked_reason_when_one_bg_region():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    assert (
        state.background_blocked_reason()
        == "Mark two background regions (hold B and click twice per region) before previewing the background"
    )


def test_background_blocked_reason_is_none_when_two_bg_regions():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    assert state.background_blocked_reason() is None
    assert state.ready_to_preview_background() is True


def test_add_bg_click_resets_background_preview_flag_when_regions_change():
    state = FitModeState()
    state.add_bg_click(70)
    state.add_bg_click(85)
    state.add_bg_click(115)
    state.add_bg_click(130)
    state.show_background_preview = True

    # A third completed pair evicts the oldest region -- a preview
    # computed from the old regions is now stale and must clear.
    state.add_bg_click(200)
    state.add_bg_click(210)

    assert state.show_background_preview is False
