import json

import pytest

from fit_export import (
    append_auto_log, auto_log_path, fit_result_to_json_record,
    fit_result_to_text_report, integration_result_to_json_record,
    integration_result_to_text_report, write_text_report,
)
from peak_fit import FitResult, IntegrationResult, PeakResult


def _make_result(timestamp="2026-07-16T12:00:00"):
    peak = PeakResult(
        position=100.0, position_err=0.1,
        fwhm=7.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0,
        amplitude=200.0, sigma=3.0,
        amplitude_err=5.0, sigma_err=0.05,
        full_area=1200.0, full_area_err=50.0,
    )
    return FitResult(
        left_bg_region=(70.0, 85.0),
        right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0),
        background_slope=0.1,
        background_intercept=20.0,
        peaks=[peak],
        fixed_params={"sigma": 3.0},
        timestamp=timestamp,
        gross_area=1200.0, gross_area_err=34.6,
        net_area=1000.0, net_area_err=50.0,
        reduced_chi2=1.15,
    )


def test_auto_log_path_derives_from_spectrum_stem(tmp_path):
    spectrum_path = str(tmp_path / "eu.spe")
    assert auto_log_path(spectrum_path) == str(tmp_path / "eu_fits.jsonl")


def test_fit_result_to_json_record_includes_every_field_and_uncertainty():
    result = _make_result()
    record = fit_result_to_json_record(result, "eu.spe")
    assert record["timestamp"] == "2026-07-16T12:00:00"
    assert record["spectrum_path"] == "eu.spe"
    assert record["left_bg_region"] == [70.0, 85.0]
    assert record["right_bg_region"] == [115.0, 130.0]
    assert record["fit_region"] == [85.0, 115.0]
    assert record["background_slope"] == 0.1
    assert record["background_intercept"] == 20.0
    assert record["link_widths"] is True
    assert record["fixed_params"] == {"sigma": 3.0}
    assert record["gross_area"] == 1200.0
    assert record["gross_area_err"] == 34.6
    assert record["net_area"] == 1000.0
    assert record["net_area_err"] == 50.0
    assert record["reduced_chi2"] == 1.15
    assert record["peaks"] == [
        {
            "position": 100.0, "position_err": 0.1,
            "fwhm": 7.0, "fwhm_err": 0.2,
            "amplitude": 200.0, "amplitude_err": 5.0,
            "sigma": 3.0, "sigma_err": 0.05,
            "area": 1000.0, "area_err": 50.0,
            "full_area": 1200.0, "full_area_err": 50.0,
        }
    ]


def test_append_auto_log_writes_one_valid_json_line(tmp_path):
    spectrum_path = str(tmp_path / "eu.spe")
    result = _make_result()

    append_auto_log(spectrum_path, result)

    log_path = auto_log_path(spectrum_path)
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["timestamp"] == "2026-07-16T12:00:00"


def test_append_auto_log_appends_without_disturbing_earlier_lines(tmp_path):
    spectrum_path = str(tmp_path / "eu.spe")
    first = _make_result(timestamp="2026-07-16T12:00:00")
    second = _make_result(timestamp="2026-07-16T12:05:00")

    append_auto_log(spectrum_path, first)
    append_auto_log(spectrum_path, second)

    log_path = auto_log_path(spectrum_path)
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["timestamp"] == "2026-07-16T12:00:00"
    assert json.loads(lines[1])["timestamp"] == "2026-07-16T12:05:00"


def test_fit_result_to_text_report_includes_every_parameter_and_uncertainty():
    result = _make_result()
    report = fit_result_to_text_report(result, "eu.spe", fit_number=1)
    assert "Fit 1" in report
    assert "eu.spe" in report
    assert "2026-07-16T12:00:00" in report
    assert "Position:" in report and "100" in report
    assert "FWHM:" in report and "7" in report
    assert "Amplitude:" in report and "200" in report
    assert "Sigma:" in report
    assert "Fixed parameters: sigma=3" in report
    assert "Reduced chi^2: 1.15" in report
    assert "Region full area (no background subtracted):" in report
    assert "Region net area (background subtracted):" in report
    # Both the region-level totals and this single peak's own full/net
    # area resolve to the same numbers here (one peak fills the region).
    assert report.count("1200") == 2  # region gross_area + peak full_area
    assert report.count("1000") == 2  # region net_area + peak area (net)


def test_write_text_report_joins_multiple_fit_blocks_in_order(tmp_path):
    results = [(1, _make_result("2026-07-16T12:00:00")), (2, _make_result("2026-07-16T12:05:00"))]
    out_path = tmp_path / "report.txt"

    write_text_report(str(out_path), results, "eu.spe")

    text = out_path.read_text(encoding="utf-8")
    assert text.index("Fit 1") < text.index("Fit 2")


def _make_integration_result(timestamp="2026-07-16T12:00:00"):
    return IntegrationResult(
        left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0), background_density=20.0,
        gross_area=1000.0, gross_area_err=30.0,
        gross_centroid=100.0, gross_centroid_err=0.5,
        gross_fwhm=8.0, gross_fwhm_err=0.4,
        gross_skewness=0.0, gross_skewness_err=0.1,
        background_area=200.0, background_area_err=10.0,
        background_centroid=100.0, background_centroid_err=1.0,
        background_fwhm=9.0, background_fwhm_err=0.5,
        background_skewness=0.0, background_skewness_err=0.1,
        net_area=800.0, net_area_err=32.0,
        net_centroid=100.0, net_centroid_err=0.6,
        net_fwhm=7.5, net_fwhm_err=0.4,
        net_skewness=0.0, net_skewness_err=0.1,
        timestamp=timestamp,
    )


def test_integration_result_to_json_record_includes_gross_background_net():
    result = _make_integration_result()
    record = integration_result_to_json_record(result, "eu.spe")
    assert record["type"] == "integration"
    assert record["timestamp"] == "2026-07-16T12:00:00"
    assert record["spectrum_path"] == "eu.spe"
    assert record["fit_region"] == [85.0, 115.0]
    assert record["background_density"] == 20.0
    assert record["gross"]["area"] == 1000.0
    assert record["background"]["area"] == 200.0
    assert record["net"]["area"] == 800.0
    assert record["net"]["centroid"] == 100.0


def test_append_auto_log_handles_an_integration_result(tmp_path):
    spectrum_path = str(tmp_path / "eu.spe")
    append_auto_log(spectrum_path, _make_integration_result())

    log_path = auto_log_path(spectrum_path)
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "integration"


def test_integration_result_to_text_report_includes_all_three_layers():
    result = _make_integration_result()
    report = integration_result_to_text_report(result, "eu.spe", fit_number=1)
    assert "Integration" in report
    assert "Gross:" in report
    assert "Background:" in report
    assert "Net:" in report
    assert "800" in report


def test_integration_result_json_fwhm_keV_uses_derivative_at_centroid_for_quadratic():
    """Regression guard for the Integration-result JSON path (mirrors
    the FitResult version above) -- fwhm_keV must use the calibration's
    derivative at the layer's own centroid, not at the fwhm's own
    numeric value."""
    from calibration import Calibration

    result = _make_integration_result()
    cal = Calibration(kind="quadratic", a=0.0, b=1.0, c=0.5)
    record = integration_result_to_json_record(result, "eu.spe", calibration=cal)
    correct_slope = abs(cal.derivative(100.0))  # net_centroid, NOT net_fwhm (7.5)
    assert record["net"]["fwhm_keV"] == pytest.approx(correct_slope * 7.5)
    assert record["net"]["fwhm_err_keV"] == pytest.approx(correct_slope * 0.4)


def test_integration_result_text_report_fwhm_keV_uses_derivative_at_centroid_for_quadratic():
    """Same regression guard for the Integration-result text-report path."""
    from calibration import Calibration

    result = _make_integration_result()
    cal = Calibration(kind="quadratic", a=0.0, b=1.0, c=0.5)
    report = integration_result_to_text_report(result, "eu.spe", calibration=cal)
    correct_slope = abs(cal.derivative(100.0))  # net_centroid, NOT net_fwhm (7.5)
    correct_fwhm_keV = correct_slope * 7.5
    assert f"{correct_fwhm_keV:.6g}" in report


def test_write_text_report_handles_a_mix_of_fit_and_integration_results(tmp_path):
    from peak_fit import FitResult, PeakResult

    peak = PeakResult(
        position=100.0, position_err=0.1, fwhm=7.0, fwhm_err=0.2,
        area=1000.0, area_err=50.0, amplitude=200.0, sigma=3.0,
    )
    fit = FitResult(
        left_bg_region=(70.0, 85.0), right_bg_region=(115.0, 130.0),
        fit_region=(85.0, 115.0), background_slope=0.0, background_intercept=20.0,
        peaks=[peak], timestamp="2026-07-16T12:00:00",
    )
    integration = _make_integration_result(timestamp="2026-07-16T12:05:00")
    out_path = tmp_path / "report.txt"

    write_text_report(str(out_path), [(1, fit), (2, integration)], "eu.spe")

    text = out_path.read_text(encoding="utf-8")
    assert text.index("Fit 1") < text.index("Fit 2 (Integration)")


def test_json_record_omits_kev_fields_without_calibration():
    result = _make_result()
    record = fit_result_to_json_record(result, "eu.spe")
    assert record["peaks"][0].get("position_keV") is None


def test_json_record_includes_kev_fields_with_calibration():
    from calibration import Calibration

    result = _make_result()
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    record = fit_result_to_json_record(result, "eu.spe", calibration=cal)
    peak_record = record["peaks"][0]
    # _make_result()'s peak: position=100.0, position_err=0.1, fwhm=7.0, fwhm_err=0.2
    assert peak_record["position_keV"] == pytest.approx(cal.apply(100.0))
    assert peak_record["position_err_keV"] == pytest.approx(abs(cal.derivative(100.0)) * 0.1)
    assert peak_record["fwhm_keV"] == pytest.approx(abs(cal.derivative(100.0)) * 7.0)
    assert peak_record["fwhm_err_keV"] == pytest.approx(abs(cal.derivative(100.0)) * 0.2)


def test_json_record_fwhm_keV_uses_derivative_at_position_for_quadratic():
    """Regression guard: fwhm_keV/fwhm_err_keV must use the calibration's
    derivative evaluated at the peak's POSITION (100.0), not at the
    FWHM's own numeric value (7.0) -- for a quadratic calibration these
    differ substantially, unlike linear where the derivative is
    constant everywhere and this distinction is invisible."""
    from calibration import Calibration

    result = _make_result()
    cal = Calibration(kind="quadratic", a=0.0, b=1.0, c=0.5)
    record = fit_result_to_json_record(result, "eu.spe", calibration=cal)
    peak_record = record["peaks"][0]
    correct_slope = abs(cal.derivative(100.0))  # NOT cal.derivative(7.0)
    assert peak_record["fwhm_keV"] == pytest.approx(correct_slope * 7.0)
    assert peak_record["fwhm_err_keV"] == pytest.approx(correct_slope * 0.2)


def test_text_report_includes_kev_parenthetical_with_calibration():
    from calibration import Calibration

    result = _make_result()
    cal = Calibration(kind="linear", a=10.0, b=0.5)
    report = fit_result_to_text_report(result, "eu.spe", calibration=cal)
    assert "keV" in report


def test_text_report_omits_kev_parenthetical_without_calibration():
    result = _make_result()
    report = fit_result_to_text_report(result, "eu.spe")
    assert "keV" not in report


def test_text_report_fwhm_keV_uses_derivative_at_position_for_quadratic():
    """Same regression guard as the JSON test above, for the text-report
    path (_format_dual) instead of the JSON path (_peak_record)."""
    from calibration import Calibration

    result = _make_result()
    cal = Calibration(kind="quadratic", a=0.0, b=1.0, c=0.5)
    report = fit_result_to_text_report(result, "eu.spe", calibration=cal)
    correct_slope = abs(cal.derivative(100.0))  # NOT cal.derivative(7.0)
    correct_fwhm_keV = correct_slope * 7.0
    assert f"{correct_fwhm_keV:.6g}" in report
