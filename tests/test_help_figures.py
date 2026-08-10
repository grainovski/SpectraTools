from help_figures import (
    anatomy_of_a_fit_figure,
    calibration_curve_figure,
    integration_background_figure,
    multiplet_figure,
    sigma_fwhm_figure,
    tail_effect_figure,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_anatomy_of_a_fit_figure_returns_valid_png():
    data = anatomy_of_a_fit_figure()
    assert data.startswith(_PNG_SIGNATURE)
    assert len(data) > 20000  # a blank same-size figure renders to ~10KB; real content is 49-58KB


def test_tail_effect_figure_returns_valid_png():
    data = tail_effect_figure()
    assert data.startswith(_PNG_SIGNATURE)
    assert len(data) > 20000  # a blank same-size figure renders to ~10KB; real content is 49-58KB


def test_multiplet_figure_returns_valid_png():
    data = multiplet_figure()
    assert data.startswith(_PNG_SIGNATURE)
    assert len(data) > 20000  # a blank same-size figure renders to ~10KB; real content is 49-58KB


def test_calibration_curve_figure_returns_valid_png():
    data = calibration_curve_figure()
    assert data.startswith(_PNG_SIGNATURE)
    assert len(data) > 20000  # a blank same-size figure renders to ~10KB; real content is 49-58KB


def test_sigma_fwhm_figure_returns_valid_png():
    data = sigma_fwhm_figure()
    assert data.startswith(_PNG_SIGNATURE)
    assert len(data) > 20000  # measured ~42KB


def test_integration_background_figure_returns_valid_png():
    data = integration_background_figure()
    assert data.startswith(_PNG_SIGNATURE)
    assert len(data) > 20000  # measured ~42KB
