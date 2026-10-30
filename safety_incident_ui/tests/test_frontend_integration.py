"""Integration test suite for the Streamlit UI service layer and FastAPI backend.

Verifies:
1. End-to-end incident submission, prediction, and persistence in PostgreSQL.
2. Retrieval of saved incident analysis (simulating page reload).
3. Incident listing with filters for user submissions vs historical dataset.
4. Document upload and metadata persistence without path exposure.
5. Truthful error reporting when backend is unavailable (NO silent mock fallback).
6. Clear separation between live database mode and demo mode.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UI_ROOT = PROJECT_ROOT / "safety_incident_ui"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UI_ROOT))

from services import api_client
from utils import validation


LIVE_BACKEND_URL = os.getenv("SAFETY_API_URL", "http://127.0.0.1:8502")


@pytest.fixture(autouse=True)
def ensure_live_mode():
    """Ensure tests run against the live backend by default."""
    api_client.API_URL = LIVE_BACKEND_URL
    api_client.IS_DEMO_MODE = False
    api_client.USE_MOCK = False
    yield
    api_client.API_URL = LIVE_BACKEND_URL
    api_client.IS_DEMO_MODE = False
    api_client.USE_MOCK = False


# =====================================================================
# 1. LIVE BACKEND CONNECTIVITY & INCIDENT LISTING
# =====================================================================

def test_list_incidents_live_shape():
    """Verify list_incidents connects to PostgreSQL and returns normalized DataFrame."""
    api_client.list_incidents.clear()
    df = api_client.list_incidents(limit=25)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {
        "id", "date", "type", "equipment", "equipment_tag", "severity",
        "root_cause", "risk", "unit", "location", "department", "summary",
        "source", "severity_score",
    }
    assert expected_cols.issubset(set(df.columns))
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_list_incidents_source_filtering():
    """Verify filtering by record_source correctly separates historical from user submissions."""
    api_client.list_incidents.clear()
    df_hist = api_client.list_incidents(source="historical_dataset", limit=20)
    assert not df_hist.empty
    assert (df_hist["source"] == "historical_dataset").all()

    api_client.list_incidents.clear()
    df_user = api_client.list_incidents(source="user_submission", limit=20)
    assert not df_user.empty
    assert (df_user["source"] == "user_submission").all()


# =====================================================================
# 2. SUBMIT INCIDENT, PERSISTENCE & RETRIEVAL (PAGE RELOAD WORKFLOW)
# =====================================================================

def test_submit_incident_and_retrieve_analysis():
    """Test full workflow: submit valid incident -> persist to DB -> retrieve saved analysis."""
    report_text = (
        "On 2026-10-29 at 14:15 in Chemical Processing, a major leak was detected "
        "at Chemical Pump P-204 due to seal degradation and vibration. "
        "The operator emergency-isolated the line. No injuries reported."
    )

    incident_payload = {
        "incident_year": 2026,
        "incident_month": 10,
        "incident_dayofweek": 3,
        "incident_hour": 14,
        "incident_minute": 15,
        "department": "Chemical Processing",
        "location": "Unit 2",
        "shift": "Morning",
        "employee_role": "Chemical Plant Operator",
        "experience_level": "3-7 years",
        "equipment_type": "Chemical Pump",
        "equipment_status": "Malfunctioning",
        "incident_type": "Loss of Containment",
        "hazard_type": "Chemical Exposure",
        "incident_description": report_text,
        "reported_by_role": "Safety Officer",
    }

    # 1. Submit prediction request
    pred_result = api_client.predict_incident(
        incident_payload,
        original_report_text=report_text,
        submission_method="paste",
        submission_payload={"source": "pytest_integration"},
    )

    assert "incident_id" in pred_result
    assert "predictions" in pred_result
    incident_id = pred_result["incident_id"]
    preds = pred_result["predictions"]

    assert preds["root_cause_category"]
    assert preds["severity_level"]
    assert preds["risk_level"]

    # 2. Retrieve the saved analysis (as views/analysis.py does)
    analysis = api_client.get_incident_analysis(incident_id)

    assert analysis["incident"]["id"] == incident_id
    assert analysis["incident"]["summary"] == report_text[:360]
    assert "Chemical Pump" in analysis["extracted_information"]["equipment"]
    assert len(analysis["root_causes"]) >= 1
    assert analysis["root_causes"][0]["cause"] == preds["root_cause_category"]
    assert analysis["severity"]["label"] == preds["severity_level"]

    # 3. Simulate browser reload: clear cache and re-query
    api_client.list_incidents.clear()
    analysis_reloaded = api_client.get_incident_analysis(incident_id)
    assert analysis_reloaded["incident"]["id"] == incident_id
    assert analysis_reloaded["severity"]["label"] == preds["severity_level"]


# =====================================================================
# 3. DOCUMENT UPLOAD & PRIVATE METADATA WORKFLOW
# =====================================================================

def test_document_upload_and_linkage():
    """Test document upload: verifies private storage and linkage to incident prediction."""
    content = b"Safety Investigation Report: Valve Stem Breakage on Reactor Feed Line 3."
    filename = "reactor_valve_investigation.txt"

    # Upload document
    metadata = api_client.upload_incident_file(
        file_bytes=content,
        original_filename=filename,
        mime_type="text/plain",
    )

    assert "storage_key" in metadata
    assert metadata["original_filename"] == filename
    assert metadata.get("size_bytes") == len(content) or metadata.get("file_size_bytes") == len(content)
    assert "sha256_hash" in metadata
    # Verify no local filesystem paths are exposed
    assert "D:\\" not in metadata["storage_key"]
    assert "C:\\" not in metadata["storage_key"]

    # Submit incident referencing the uploaded file key
    incident_payload = {
        "incident_year": 2026,
        "incident_month": 10,
        "incident_dayofweek": 3,
        "department": "Chemical Processing",
        "equipment_type": "Reactor",
        "incident_description": "Valve stem failure on reactor feed line resulting in minor upset.",
    }

    result = api_client.predict_incident(
        incident_payload,
        original_report_text="Valve stem failure on reactor feed line",
        submission_method="upload",
        uploaded_file_key=metadata["storage_key"],
    )

    assert result["incident_id"]
    analysis = api_client.get_incident_analysis(result["incident_id"])
    assert analysis["incident"]["id"] == result["incident_id"]


# =====================================================================
# 4. TRUTHFUL ERROR STATES (NO SILENT MOCK FALLBACK)
# =====================================================================

def test_truthful_error_when_backend_unreachable():
    """Verify that if the backend is down, an ApiError is raised and NO mock data is returned."""
    # Point to an unallocated port where no server is running
    api_client.API_URL = "http://127.0.0.1:59998"
    api_client.list_incidents.clear()

    with pytest.raises(api_client.ApiError) as exc_info:
        api_client.list_incidents()
    assert "Could not connect to backend" in str(exc_info.value)

    with pytest.raises(api_client.ApiError) as exc_info:
        api_client.get_incident_analysis("00000000-0000-0000-0000-000000000000")
    assert "Could not connect to backend" in str(exc_info.value)


def test_nonexistent_incident_error():
    """Verify 404 for a non-existent incident raises clear ApiError."""
    api_client.API_URL = LIVE_BACKEND_URL
    non_existent_id = "00000000-0000-0000-0000-000000000000"

    with pytest.raises(api_client.ApiError) as exc_info:
        api_client.get_incident_analysis(non_existent_id)
    assert "was not found" in str(exc_info.value)


# =====================================================================
# 5. DEMO MODE CLEAR SEPARATION
# =====================================================================

def test_demo_mode_isolation():
    """Verify demo mode returns synthetic data and is explicitly marked."""
    api_client.IS_DEMO_MODE = True
    api_client.USE_MOCK = True

    df_demo = api_client.list_incidents()
    assert not df_demo.empty
    # In mock data, IDs start with INC-
    assert df_demo.iloc[0]["id"].startswith("INC-")

    # In live mode, IDs are UUIDs
    api_client.IS_DEMO_MODE = False
    api_client.USE_MOCK = False
    api_client.list_incidents.clear()
    df_live = api_client.list_incidents(limit=5)
    # Check that live IDs are standard 36-character UUID strings
    assert len(df_live.iloc[0]["id"]) == 36


# =====================================================================
# 6. SQL AGGREGATIONS & VISUALIZATION DATA CONTRACTS
# =====================================================================

def test_kpis_and_populations_accounting():
    """Verify get_kpis returns real database populations and statistics."""
    kpis = api_client.get_kpis()
    assert kpis["total_incidents"] >= 8049
    assert kpis["historical_count"] == 8000
    assert kpis["user_submission_count"] >= 49
    assert kpis["ground_truth_count"] >= 8000
    assert kpis["predicted_count"] >= 49
    assert kpis["high_severity_count"] > 0
    assert kpis["high_severity_pct"] > 0.0


def test_trends_and_distributions():
    """Verify trends, distributions, and heatmap return populated aggregates."""
    trends = api_client.get_trends()
    assert len(trends) > 0
    assert "month" in trends[0]
    assert "total" in trends[0]

    depts = api_client.get_distributions(dim="department", limit=5)
    assert len(depts) <= 5
    assert all("name" in d and "count" in d for d in depts)

    shifts = api_client.get_distributions(dim="shift", limit=3)
    assert len(shifts) <= 3

    preds = api_client.get_distributions(dim="predicted_root_cause", limit=5)
    assert len(preds) > 0


def test_heatmap_and_eval_metrics():
    """Verify heatmap matrix and model evaluation accounting."""
    hm = api_client.get_heatmap(top_equipment=5, top_causes=4)
    assert len(hm["equipment"]) <= 5
    assert len(hm["root_causes"]) <= 4
    assert len(hm["matrix"]) == len(hm["equipment"])

    eval_m = api_client.get_eval_metrics()
    assert eval_m["total_historical_with_ground_truth"] == 8000
    assert eval_m["total_prediction_runs"] >= 49
    assert "methodology_note" in eval_m
    assert "Training dataset records are excluded" in eval_m["methodology_note"]


def test_chart_builders():
    """Verify chart helper functions produce valid Plotly Figure objects from SQL data."""
    from components import charts

    trends = [{"month": "2026-01", "total": 100, "high_severity": 10}]
    fig_trend = charts.trend_line_chart(trends)
    assert fig_trend is not None

    dist = [{"name": "Assembly", "count": 150}, {"name": "Production", "count": 120}]
    fig_dist = charts.distribution_bar_chart(dist, label="Department")
    assert fig_dist is not None

    hm = {"equipment": ["Pump"], "root_causes": ["Human Error"], "matrix": [[5]]}
    fig_hm = charts.matrix_heatmap(hm)
    assert fig_hm is not None

