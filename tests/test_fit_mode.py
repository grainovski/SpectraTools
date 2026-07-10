import pytest

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


def test_bg_region_cap_is_two():
    assert BG_REGION_CAP == 2
