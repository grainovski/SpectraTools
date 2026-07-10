import pytest

from fit_mode import (
    STEP_FIT_REGION,
    STEP_LEFT_BG,
    STEP_MARKING_PEAKS,
    STEP_RIGHT_BG,
    FitModeState,
)


def test_initial_state():
    state = FitModeState()
    assert state.step == STEP_LEFT_BG
    assert state.left_bg_region is None
    assert state.right_bg_region is None
    assert state.fit_region is None
    assert state.peak_positions == []


def test_region_sequence_advances_steps():
    state = FitModeState()
    state.add_region(70, 85)
    assert state.step == STEP_RIGHT_BG
    assert state.left_bg_region == (70, 85)

    state.add_region(130, 115)  # reversed order gets normalized
    assert state.step == STEP_FIT_REGION
    assert state.right_bg_region == (115, 130)

    state.add_region(85, 115)
    assert state.step == STEP_MARKING_PEAKS
    assert state.fit_region == (85, 115)


def test_add_region_raises_when_not_awaiting_a_region():
    state = FitModeState()
    state.add_region(70, 85)
    state.add_region(115, 130)
    state.add_region(85, 115)
    with pytest.raises(ValueError):
        state.add_region(200, 210)


def test_add_peak_within_fit_region():
    state = FitModeState()
    state.add_region(70, 85)
    state.add_region(115, 130)
    state.add_region(85, 115)

    assert state.add_peak(100.0) is True
    assert state.peak_positions == [100.0]


def test_add_peak_outside_fit_region_is_ignored():
    state = FitModeState()
    state.add_region(70, 85)
    state.add_region(115, 130)
    state.add_region(85, 115)

    assert state.add_peak(200.0) is False
    assert state.peak_positions == []


def test_add_peak_before_fit_region_raises():
    state = FitModeState()
    state.add_region(70, 85)
    with pytest.raises(ValueError):
        state.add_peak(100.0)


def test_ready_to_fit_requires_all_regions_and_a_peak():
    state = FitModeState()
    assert state.ready_to_fit() is False
    state.add_region(70, 85)
    state.add_region(115, 130)
    state.add_region(85, 115)
    assert state.ready_to_fit() is False
    state.add_peak(100.0)
    assert state.ready_to_fit() is True


def test_reset_clears_everything():
    state = FitModeState()
    state.add_region(70, 85)
    state.add_region(115, 130)
    state.add_region(85, 115)
    state.add_peak(100.0)

    state.reset()

    assert state.step == STEP_LEFT_BG
    assert state.left_bg_region is None
    assert state.peak_positions == []


def test_ordered_bg_regions_regardless_of_marking_order():
    state = FitModeState()
    # mark the spatially-RIGHT region first
    state.add_region(115, 130)
    state.add_region(70, 85)
    state.add_region(85, 115)

    left, right = state.ordered_bg_regions()
    assert left == (70, 85)
    assert right == (115, 130)
