import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from data import mock_data
from utils import validation


def test_empty_and_short_reports_rejected():
    assert not validation.validate_report_text("")[0]
    assert not validation.validate_report_text("too short")[0]


def test_upload_validation():
    assert not validation.validate_upload("a.exe", b"x" * 100)[0]
    assert not validation.validate_upload("a.txt", b"")[0]
    assert validation.validate_upload("a.txt", mock_data.EXAMPLE_REPORT.encode())[0]


def test_example_analysis_shape():
    r = mock_data.mock_analyze(mock_data.EXAMPLE_REPORT)
    assert r["incident"]["type"] == "Loss of Containment"
    assert r["root_causes"] and r["root_causes"][0]["evidence"] in mock_data.EXAMPLE_REPORT
    h = mock_data.mock_history(r, mock_data.build_history())
    assert h["same_equipment_incidents"]
