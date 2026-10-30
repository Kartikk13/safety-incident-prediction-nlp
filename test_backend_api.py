"""
test_backend_api.py

Automated test suite for FastAPI backend endpoints:
- GET /health
- POST /upload
- POST /predict
- POST /nlp/extract
- GET /incidents
- GET /incidents/{id}
- GET /incidents/{id}/analysis
- GET /analytics/kpis
- GET /analytics/trends
- GET /analytics/distributions
- GET /analytics/heatmap
- GET /analytics/similar
- GET /analytics/dataset-summary
- GET /models
"""

import os
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"
import sys
from pathlib import Path
import uuid
from decimal import Decimal
try:
    import pytest
    fixture = pytest.fixture
except ImportError:
    def fixture(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from database import SessionLocal
from models import Incident, PredictionRun, Prediction, AnalysisResult, ModelVersion
from sqlalchemy import select, func


@fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# =====================================================================
# 1. HEALTH & READINESS
# =====================================================================

def test_health_endpoint(client):
    """Verify /health reports healthy status, database connection, and 3 loaded models."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert set(data["models_loaded"]) == {
        "root_cause_category",
        "severity_level",
        "risk_level",
    }
    assert "timestamp" in data


# =====================================================================
# 2. DOCUMENT UPLOAD
# =====================================================================

def test_upload_valid_text_file(client):
    """Verify /upload saves a valid .txt file and returns sanitized metadata."""
    file_content = b"Sample incident report for upload test."
    files = {"file": ("test_report.txt", file_content, "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "file" in data
    assert data["file"]["original_filename"] == "test_report.txt"
    assert "storage_key" in data["file"]
    assert "storage_path" not in data["file"]  # Private path must never be exposed


def test_upload_unsupported_file_type(client):
    """Verify /upload rejects unsupported extensions like .exe with 400."""
    files = {"file": ("malicious.exe", b"MZ...", "application/octet-stream")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_empty_file(client):
    """Verify /upload rejects empty files with 400."""
    files = {"file": ("empty.txt", b"", "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# =====================================================================
# 3. NLP EXTRACTION
# =====================================================================

def test_nlp_extract_valid_report(client):
    """Verify /nlp/extract parses date, time, equipment, and hazard."""
    payload = {
        "text": "On 2026-09-12 at 14:30 a hydrocarbon leak was detected at the seal of feed pump P-101A in Process Unit A.",
        "context": {"department": "Production", "shift": "Morning"},
    }
    response = client.post("/nlp/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    features = data["features"]
    assert features["incident_year"] == 2026
    assert features["incident_month"] == 9
    assert features["incident_dayofweek"] == 5
    assert features["incident_hour"] == 14
    assert features["incident_minute"] == 30
    assert features["equipment_type"] == "Pump"
    assert features["department"] == "Production"
    assert features["shift"] == "Morning"
    assert data["nlp_method"] == "hybrid_regex_rules"


def test_nlp_extract_empty_text(client):
    """Verify /nlp/extract rejects blank input with 400."""
    response = client.post("/nlp/extract", json={"text": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


# =====================================================================
# 4. PREDICTION & ATOMIC PERSISTENCE
# =====================================================================

def test_predict_and_persistence(client):
    """
    Verify /predict:
    - Runs all 3 ML models.
    - Persists Incident, PredictionRun, Predictions (with confidence scores), and AnalysisResult.
    - Preserves backward compatibility and returns rich analysis object.
    """
    unique_desc = f"Integration test report {uuid.uuid4().hex}: Operator noticed pressure drop on Compressor K-301."
    payload = {
        "incident_description": unique_desc,
        "equipment_type": "Air Compressor",
        "department": "Utilities",
        "submission_method": "form",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # 1. Response contract
    assert "predictions" in data
    assert "incident_id" in data
    assert "prediction_run_id" in data
    assert "analysis" in data
    preds = data["predictions"]
    assert "root_cause_category" in preds
    assert "severity_level" in preds
    assert "risk_level" in preds

    inc_id = uuid.UUID(data["incident_id"])
    run_id = uuid.UUID(data["prediction_run_id"])

    # 2. Database verification
    with SessionLocal() as db:
        # Verify incident record
        inc = db.scalar(select(Incident).where(Incident.id == inc_id))
        assert inc is not None
        assert inc.record_source == "user_submission"
        assert inc.submission_method == "form"
        assert inc.equipment_type == "Air Compressor"
        # Actual labels must NOT be overwritten by predictions!
        assert inc.actual_root_cause_category is None
        assert inc.actual_severity_level is None
        assert inc.actual_risk_level is None

        # Verify prediction run
        run = db.scalar(select(PredictionRun).where(PredictionRun.id == run_id))
        assert run is not None
        assert run.incident_id == inc_id
        assert run.status == "success"
        assert run.duration_ms is not None

        # Verify individual predictions
        db_preds = db.scalars(select(Prediction).where(Prediction.prediction_run_id == run_id)).all()
        assert len(db_preds) == 3
        for p in db_preds:
            assert p.predicted_value in [preds["root_cause_category"], preds["severity_level"], preds["risk_level"]]
            assert p.confidence_score is not None
            assert p.confidence_score > Decimal("0.0")

        # Verify AnalysisResult was created
        ar = db.scalar(select(AnalysisResult).where(AnalysisResult.prediction_run_id == run_id))
        assert ar is not None
        assert ar.extracted_entities["equipment"] == ["Air Compressor"]
        assert ar.summary is not None


# =====================================================================
# 5. INCIDENTS LISTING & PAGINATION
# =====================================================================

def test_list_incidents_default_pagination(client):
    """Verify /incidents default pagination returns page 1 of size 25."""
    response = client.get("/incidents")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["size"] == 25
    assert len(data["items"]) == 25
    assert data["total"] >= 8000
    assert data["pages"] >= 320


def test_list_incidents_source_filter(client):
    """Verify filtering by record_source works cleanly."""
    # Historical dataset filter
    resp_hist = client.get("/incidents?record_source=historical_dataset")
    assert resp_hist.status_code == 200
    assert resp_hist.json()["total"] == 8000
    for item in resp_hist.json()["items"]:
        assert item["record_source"] == "historical_dataset"

    # User submission filter
    resp_user = client.get("/incidents?record_source=user_submission")
    assert resp_user.status_code == 200
    assert resp_user.json()["total"] >= 1
    for item in resp_user.json()["items"]:
        assert item["record_source"] == "user_submission"


def test_list_incidents_filters(client):
    """Verify department and equipment filters."""
    response = client.get("/incidents?department=Production&equipment_type=Pump")
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["department"] == "Production"
        assert item["equipment_type"] == "Pump"


def test_list_incidents_pagination_limit(client):
    """Verify /incidents rejects size > 100 with 422."""
    response = client.get("/incidents?size=150")
    assert response.status_code == 422


# =====================================================================
# 6. INCIDENT DETAIL & ANALYSIS RETRIEVAL
# =====================================================================

def test_get_incident_detail_historical(client):
    """Verify /incidents/{id} returns historical incident with no predictions."""
    with SessionLocal() as db:
        hist_inc = db.scalar(select(Incident).where(Incident.record_source == "historical_dataset").limit(1))

    response = client.get(f"/incidents/{hist_inc.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["incident"]["id"] == str(hist_inc.id)
    assert data["incident"]["record_source"] == "historical_dataset"
    assert data["incident"]["actual_root_cause_category"] is not None
    assert data["predictions"] is None  # Historical training records do not have synthetic predictions


def test_get_incident_detail_not_found(client):
    """Verify /incidents/{id} returns 404 for non-existent UUID."""
    random_uuid = uuid.uuid4()
    response = client.get(f"/incidents/{random_uuid}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_incident_analysis_view(client):
    """Verify /incidents/{id}/analysis formats data matching Streamlit analysis view."""
    with SessionLocal() as db:
        hist_inc = db.scalar(select(Incident).where(Incident.record_source == "historical_dataset").limit(1))

    response = client.get(f"/incidents/{hist_inc.id}/analysis")
    assert response.status_code == 200
    data = response.json()
    assert "incident" in data
    assert "extracted_information" in data
    assert "root_causes" in data
    assert "severity" in data
    assert "historical_analysis" in data
    assert len(data["root_causes"]) >= 1


# =====================================================================
# 7. ANALYTICS & DASHBOARD ENDPOINTS
# =====================================================================

def test_analytics_kpis(client):
    """Verify /analytics/kpis returns accurate totals and source breakdown."""
    response = client.get("/analytics/kpis")
    assert response.status_code == 200
    data = response.json()
    assert data["total_incidents"] >= 8000
    assert data["high_severity_count"] > 0
    assert data["high_severity_pct"] > 0.0
    assert "by_record_source" in data
    assert data["by_record_source"]["historical_dataset"] == 8000
    assert data["total_incidents"] == sum(data["by_record_source"].values())


def test_analytics_trends(client):
    """Verify /analytics/trends returns monthly timeline data."""
    response = client.get("/analytics/trends?period_months=12")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "month" in data[0]
    assert "total" in data[0]
    assert "high_severity" in data[0]


def test_analytics_distributions(client):
    """Verify /analytics/distributions returns aggregated counts."""
    # Department distribution
    resp = client.get("/analytics/distributions?dimension=department&top=5")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) <= 5
    assert data[0]["count"] >= data[-1]["count"]  # Descending order

    # Predictions distribution
    resp_pred = client.get("/analytics/distributions?dimension=predicted_root_cause")
    assert resp_pred.status_code == 200
    assert isinstance(resp_pred.json(), list)


def test_analytics_distributions_invalid_dim(client):
    """Verify /analytics/distributions rejects invalid dimensions with 400."""
    resp = client.get("/analytics/distributions?dimension=invalid_dimension")
    assert resp.status_code == 400
    assert "Unsupported dimension" in resp.json()["detail"]


def test_analytics_heatmap(client):
    """Verify /analytics/heatmap returns valid cross-tabulation matrix."""
    response = client.get("/analytics/heatmap?top_equipment=5&top_causes=4")
    assert response.status_code == 200
    data = response.json()
    assert len(data["equipment"]) <= 5
    assert len(data["root_causes"]) <= 4
    assert len(data["matrix"]) == len(data["equipment"])
    assert len(data["matrix"][0]) == len(data["root_causes"])


def test_analytics_similar(client):
    """Verify /analytics/similar finds matching historical incidents."""
    response = client.get("/analytics/similar?equipment=Pump&limit=3")
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 3
    for item in data:
        assert item["similarity"] > 0.0


def test_analytics_similar_missing_params(client):
    """Verify /analytics/similar requires at least one parameter."""
    response = client.get("/analytics/similar")
    assert response.status_code == 400


def test_analytics_dataset_summary(client):
    """Verify /analytics/dataset-summary returns verified training stats."""
    response = client.get("/analytics/dataset-summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 8000
    assert data["date_range"]["min"] == "2021-01-01"
    assert data["date_range"]["max"] == "2026-08-31"
    assert "target_distributions" in data
    assert "root_cause_category" in data["target_distributions"]
    assert "null_counts" in data
    assert data["null_counts"]["injury_type"] == 1135
    assert "latest_import" in data


def test_models_inventory(client):
    """Verify /models returns registered model versions and loaded status."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 3
    loaded_targets = {m["target_name"] for m in data if m["is_loaded"]}
    assert set(loaded_targets) == {"root_cause_category", "severity_level", "risk_level"}


def test_eval_metrics_endpoint(client):
    """Verify /analytics/eval-metrics reports evaluation population and methodology note."""
    response = client.get("/analytics/eval-metrics")
    assert response.status_code == 200
    data = response.json()
    assert "evaluated_records" in data
    assert "total_historical_with_ground_truth" in data
    assert data["total_historical_with_ground_truth"] == 8000
    assert "methodology_note" in data
    assert "Training dataset records are excluded" in data["methodology_note"]



if __name__ == "__main__":
    import traceback
    with TestClient(app) as test_client:
        # Get all test functions in definition order
        test_funcs = [
            (name, obj) for name, obj in globals().items()
            if name.startswith("test_") and callable(obj)
        ]
        passed = 0
        failed = 0
        print(f"Running {len(test_funcs)} backend API tests...\n" + "=" * 60)
        for name, fn in test_funcs:
            try:
                fn(test_client)
                print(f"  PASS: {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL: {name}: {e}")
                traceback.print_exc()
                failed += 1
        print("=" * 60)
        print(f"Results: {passed} passed, {failed} failed out of {len(test_funcs)} total.")
        if failed > 0:
            sys.exit(1)

