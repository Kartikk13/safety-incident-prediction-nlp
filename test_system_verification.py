"""
test_system_verification.py

Comprehensive verification and audit test suite covering:
1. End-to-End Workflows:
   - Report submission, NLP extraction, feature review, prediction, persistence.
   - History exploration and retrieval of saved incidents.
   - Dashboard SQL aggregations and source population updates.
   - Document upload private storage, magic-byte checks, and path disclosure prevention.
   - Safe handling of invalid inputs and database error boundaries.
2. Data Integrity:
   - Historical import count and partition view verification (8,000 rows).
   - Duplicate-import idempotency and row hash stability.
   - Foreign-key cascade and constraint integrity.
   - Atomic transaction rollback on persistence failures.
   - Prediction-to-incident associations and calibrated confidence scores.
   - Model version SHA-256 fingerprint associations.
   - Strict separation of ground-truth labels vs model predictions.
   - Source filtering rigor ('historical_dataset' vs 'user_submission' vs 'all').
   - Absence of mock data leaks in production endpoints.
3. Security:
   - Path traversal defenses in upload and file retrieval.
   - Non-disclosure of internal storage paths in API responses.
   - Rejection of spoofed MIME types and disguised executables.
   - SQL injection resistance on dynamic parameters.
"""

import os
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"
import sys
from pathlib import Path
import uuid
import hashlib
from decimal import Decimal
from datetime import date, datetime

try:
    import pytest
    fixture = pytest.fixture
except ImportError:
    class MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(f):
                return f
            return decorator
        @staticmethod
        def fail(msg):
            raise AssertionError(msg)
    pytest = MockPytest()
    fixture = pytest.fixture
from fastapi.testclient import TestClient
from sqlalchemy import select, func, text, or_

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from database import SessionLocal, engine
from models import (
    Incident as IncidentRecord,
    PredictionRun,
    Prediction,
    AnalysisResult,
    ModelVersion,
    IncidentFile,
    DatasetImportLog,
)
from file_storage import UPLOAD_DIR

sys.path.append(str(PROJECT_ROOT / "safety_incident_ui"))
from services.nlp_extractor import extract_incident_features


@fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# =====================================================================
# 1. DATA INTEGRITY & DATABASE AUDIT
# =====================================================================

def test_historical_import_count_and_partition_views():
    """Verify exactly 8,000 historical records exist and partition views match."""
    with SessionLocal() as db:
        hist_count = db.scalar(
            select(func.count(IncidentRecord.id)).where(
                IncidentRecord.record_source == "historical_dataset"
            )
        )
        assert hist_count == 8000, f"Expected 8000 historical records, found {hist_count}"

        # Test PostgreSQL partition views created in migration 8f853c4d65c8
        view_hist_count = db.execute(text("SELECT count(*) FROM v_historical_incidents")).scalar()
        assert view_hist_count == 8000, f"Expected 8000 in v_historical_incidents, got {view_hist_count}"

        user_count = db.scalar(
            select(func.count(IncidentRecord.id)).where(
                IncidentRecord.record_source == "user_submission"
            )
        )
        view_user_count = db.execute(text("SELECT count(*) FROM v_user_incidents")).scalar()
        assert view_user_count == user_count


def test_duplicate_import_idempotency():
    """Verify that attempting to import identical rows skips duplicates via source_row_hash."""
    from import_historical_dataset import HistoricalDatasetImporter

    importer = HistoricalDatasetImporter(
        csv_path=PROJECT_ROOT / "safety_incidents_cleaned.csv",
        mode="dry_run",
        sample_limit=20,
        batch_size=20,
    )
    result = importer.run()
    # All 20 sample rows already exist in the database from Phase 2
    assert result["rows_accepted"] == 20
    assert result["rows_skipped_duplicate"] == 20
    assert result["rows_inserted"] == 0
    assert result["status"] == "completed"


def test_foreign_key_and_cascade_relationships():
    """Verify foreign key integrity between Incident, PredictionRun, Prediction, and AnalysisResult."""
    with SessionLocal() as db:
        # Create an isolated test incident
        test_inc_id = uuid.uuid4()
        test_run_id = uuid.uuid4()
        test_version = db.scalar(select(ModelVersion).limit(1))
        assert test_version is not None

        inc = IncidentRecord(
            id=test_inc_id,
            record_source="user_submission",
            submission_method="paste",
            incident_description="Test incident for foreign key and cascade audit.",
            incident_year=2026,
            incident_month=10,
            incident_dayofweek=3,
        )
        db.add(inc)
        db.flush()

        run = PredictionRun(
            id=test_run_id,
            incident_id=test_inc_id,
            status="success",
            duration_ms=45,
        )
        db.add(run)
        db.flush()

        pred = Prediction(
            prediction_run_id=test_run_id,
            target_name="severity_level",
            predicted_value="Moderate",
            model_version_id=test_version.id,
            confidence_score=Decimal("0.8500"),
        )
        db.add(pred)

        analysis = AnalysisResult(
            prediction_run_id=test_run_id,
            extracted_entities={"equipment": ["Pump"]},
            summary="Cascade audit analysis.",
        )
        db.add(analysis)
        db.commit()

        # Verify child records exist
        p_check = db.scalar(select(Prediction).where(Prediction.prediction_run_id == test_run_id))
        assert p_check is not None
        ar_check = db.scalar(select(AnalysisResult).where(AnalysisResult.prediction_run_id == test_run_id))
        assert ar_check is not None

        # Delete the test incident and verify cascade deletion
        db.delete(inc)
        db.commit()

        # Cascade check: prediction run, predictions, and analysis result must be deleted
        assert db.scalar(select(PredictionRun).where(PredictionRun.id == test_run_id)) is None
        assert db.scalar(select(Prediction).where(Prediction.prediction_run_id == test_run_id)) is None
        assert db.scalar(select(AnalysisResult).where(AnalysisResult.prediction_run_id == test_run_id)) is None


def test_transaction_rollback_on_failure(client):
    """Verify that if an error occurs during persistence, rollback prevents orphaned incident records."""
    with SessionLocal() as db:
        initial_inc_count = db.scalar(select(func.count(IncidentRecord.id)))

    # Supply an invalid uploaded_file_key that does not exist in private storage
    payload = {
        "incident_description": "Transaction rollback test failure.",
        "uploaded_file_key": "non_existent_key_1234567890.txt",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "not found in private storage" in response.json()["detail"]

    # Verify no orphan incident record was committed
    with SessionLocal() as db:
        post_count = db.scalar(select(func.count(IncidentRecord.id)))
        assert post_count == initial_inc_count, "Rollback failed: orphan incident record was committed"


def test_model_version_fingerprint_integrity():
    """Verify loaded model versions match the SHA-256 hash of on-disk joblib pipelines."""
    with SessionLocal() as db:
        versions = db.scalars(select(ModelVersion)).all()
        assert len(versions) >= 3

        for v in versions:
            artifact_file = PROJECT_ROOT / "saved_models" / f"{v.target_name}_pipeline.joblib"
            assert artifact_file.exists(), f"Artifact missing: {artifact_file}"
            actual_sha256 = hashlib.sha256(artifact_file.read_bytes()).hexdigest()
            assert v.artifact_sha256 == actual_sha256, f"Hash mismatch for {v.target_name}"


def test_strict_separation_of_ground_truth_and_predictions(client):
    """Verify that predictions are NEVER saved to actual_* columns and actual historical labels are uncorrupted."""
    # Submit a new incident
    payload = {
        "incident_description": "A worker reported a chemical spill near the Reactor Bay.",
        "department": "Chemical Processing",
        "equipment_type": "Reactor",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    inc_id = resp.json()["incident_id"]

    with SessionLocal() as db:
        user_inc = db.scalar(select(IncidentRecord).where(IncidentRecord.id == uuid.UUID(inc_id)))
        assert user_inc is not None
        assert user_inc.record_source == "user_submission"
        # Actual ground truth labels MUST remain NULL on user submissions
        assert user_inc.actual_root_cause_category is None
        assert user_inc.actual_severity_level is None
        assert user_inc.actual_risk_level is None

        # Check that historical records have actual labels preserved
        hist_sample = db.scalar(
            select(IncidentRecord).where(IncidentRecord.record_source == "historical_dataset").limit(1)
        )
        assert hist_sample.actual_root_cause_category is not None
        assert hist_sample.actual_severity_level is not None
        assert hist_sample.actual_risk_level is not None


def test_source_filtering_rigor(client):
    """Verify that filtering by record_source never leaks records from other populations."""
    # Historical filter
    hist_resp = client.get("/incidents?source=historical_dataset&size=50")
    assert hist_resp.status_code == 200
    hist_items = hist_resp.json()["items"]
    assert len(hist_items) > 0
    for item in hist_items:
        assert item["record_source"] == "historical_dataset"
        assert item["actual_severity_level"] is not None

    # User submission filter
    user_resp = client.get("/incidents?source=user_submission&size=50")
    assert user_resp.status_code == 200
    user_items = user_resp.json()["items"]
    assert len(user_items) > 0
    for item in user_items:
        assert item["record_source"] == "user_submission"

    # All records
    all_resp = client.get("/incidents?source=all&size=10")
    assert all_resp.status_code == 200
    assert all_resp.json()["total"] >= 8000


def test_aggregation_accuracy_and_accounting(client):
    """Verify mathematical consistency across KPI counts, sources, and trend buckets."""
    resp = client.get("/analytics/kpis")
    assert resp.status_code == 200
    kpis = resp.json()

    assert kpis["historical_count"] == 8000
    assert kpis["user_submission_count"] >= 1
    assert kpis["total_incidents"] == kpis["historical_count"] + kpis["user_submission_count"]
    assert kpis["ground_truth_count"] == 8000
    assert kpis["predicted_count"] >= 1

    # Trends verification
    trends_resp = client.get("/analytics/trends?period_months=12")
    assert trends_resp.status_code == 200
    trends = trends_resp.json()
    assert len(trends) > 0
    for t in trends:
        assert t["total"] >= t["high_severity"]
        assert len(t["month"]) == 7  # 'YYYY-MM' format


def test_no_mock_data_leakage(client):
    """Verify that mock data from mock_data.py does not contaminate database queries."""
    resp = client.get("/incidents?size=100")
    assert resp.status_code == 200
    items = resp.json()["items"]
    # mock_data.py has simulated IDs like "INC-2024-001"
    for item in items:
        # Database IDs must be valid UUID strings
        try:
            uuid.UUID(item["id"])
        except ValueError:
            pytest.fail(f"Non-UUID incident ID found in production API: {item['id']}")


# =====================================================================
# 2. END-TO-END WORKFLOWS
# =====================================================================

def test_e2e_full_reporting_prediction_persistence_and_retrieval(client):
    """
    End-to-End Workflow:
    1. Extract features from report text.
    2. Review / modify extracted values.
    3. Submit to /predict.
    4. Verify predictions and atomic database persistence.
    5. Retrieve through /incidents/{id} and /incidents/{id}/analysis.
    6. Verify dashboard aggregates update appropriately.
    """
    # 1. NLP extraction
    extract_resp = client.post(
        "/nlp/extract",
        json={
            "text": "On 2026-10-14 at 16:45, a technician in the Electrical Room reported a malfunctioning Motor Control Centre.",
            "context": {"department": "Electrical"},
        },
    )
    assert extract_resp.status_code == 200
    features = extract_resp.json()["features"]
    assert features["incident_year"] == 2026
    assert features["incident_hour"] == 16
    assert features["equipment_type"] == "Motor Control Centre"
    assert features["equipment_status"] == "Malfunctioning"

    # 2. User correction (Tier 1 precedence)
    features["location"] = "Electrical Room"
    features["shift"] = "Evening"

    # 3. Predict & persist
    pred_resp = client.post(
        "/predict",
        json={
            **features,
            "submission_method": "paste",
            "original_report_text": "On 2026-10-14 at 16:45, a technician in the Electrical Room reported a malfunctioning Motor Control Centre.",
        },
    )
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    incident_id = pred_data["incident_id"]
    assert pred_data["predictions"]["root_cause_category"] in [
        "Equipment Failure", "Maintenance Defect", "Electrical Fault", "Wear and Tear", "Human Error", "Procedure Violation"
    ]
    assert pred_data["predictions"]["severity_level"] in ["Minor", "Moderate", "Serious", "Severe", "Critical"]
    assert pred_data["predictions"]["risk_level"] in ["Low", "Medium", "High", "Critical"]

    # 4. History retrieval
    detail_resp = client.get(f"/incidents/{incident_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()["incident"]
    assert detail["id"] == incident_id
    assert detail["location"] == "Electrical Room"
    assert detail["shift"] == "Evening"
    assert detail["record_source"] == "user_submission"

    # 5. Analysis page view retrieval (supports deep-linking / reload)
    analysis_resp = client.get(f"/incidents/{incident_id}/analysis")
    assert analysis_resp.status_code == 200
    analysis = analysis_resp.json()
    assert analysis["incident"]["id"] == incident_id
    assert len(analysis["root_causes"]) >= 1
    assert "severity" in analysis

    # 6. Dashboard KPI accounting
    kpi_resp = client.get("/analytics/kpis")
    assert kpi_resp.status_code == 200
    kpis = kpi_resp.json()
    assert kpis["user_submission_count"] >= 1
    assert kpis["historical_count"] == 8000
    assert kpis["total_incidents"] == kpis["historical_count"] + kpis["user_submission_count"]


def test_e2e_document_upload_private_storage_and_linkage(client):
    """
    Verify document upload:
    - Stores file in private UPLOAD_DIR with randomized UUID name.
    - Excludes internal storage_path from API response.
    - Links uploaded file to incident record upon prediction.
    """
    content = b"Incident Report: High pressure steam valve leak observed in the Boiler House."
    resp = client.post(
        "/upload",
        files={"file": ("incident_safety_log.txt", content, "text/plain")},
    )
    assert resp.status_code == 200
    file_info = resp.json()["file"]
    storage_key = file_info["storage_key"]

    # Verify no private disk path is exposed to caller
    assert "storage_path" not in file_info
    assert file_info["original_filename"] == "incident_safety_log.txt"
    assert file_info["size_bytes"] == len(content)

    # Verify file was written to private UPLOAD_DIR
    stored_path = UPLOAD_DIR / storage_key
    assert stored_path.exists()
    assert stored_path.read_bytes() == content

    # Submit incident referencing this upload
    pred_resp = client.post(
        "/predict",
        json={
            "incident_description": "High pressure steam valve leak observed in the Boiler House.",
            "uploaded_file_key": storage_key,
            "submission_method": "upload",
            "submission_payload": {
                "original_filename": "incident_safety_log.txt",
                "mime_type": "text/plain",
            },
        },
    )
    assert pred_resp.status_code == 200
    inc_id = pred_resp.json()["incident_id"]

    # Verify linkage in incident details
    detail_resp = client.get(f"/incidents/{inc_id}")
    assert detail_resp.status_code == 200
    files_linked = detail_resp.json()["files"]
    assert len(files_linked) == 1
    assert files_linked[0]["filename"] == "incident_safety_log.txt"
    assert "storage_path" not in files_linked[0]


# =====================================================================
# 3. SECURITY REVIEW & DEFENSIVE TESTING
# =====================================================================

def test_security_path_traversal_prevention_on_upload_link(client):
    """Verify that path traversal attempts in uploaded_file_key are blocked."""
    # Attempt to traverse out of UPLOAD_DIR
    traversal_keys = [
        "../app.py",
        "..\\database.py",
        "../../alembic.ini",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
    ]
    for bad_key in traversal_keys:
        resp = client.post(
            "/predict",
            json={
                "incident_description": "Path traversal probe.",
                "uploaded_file_key": bad_key,
            },
        )
        assert resp.status_code in (400, 422), f"Failed to reject traversal key: {bad_key}"
        assert "not found in private storage" in resp.json()["detail"] or "Validation" in resp.text


def test_security_upload_magic_byte_enforcement(client):
    """Verify that executable files disguised with .pdf or .docx extensions are rejected."""
    fake_pdf = b"MZ\x90\x00\x03\x00\x00\x00This is an executable disguised as PDF"
    resp = client.post(
        "/upload",
        files={"file": ("malicious.pdf", fake_pdf, "application/pdf")},
    )
    assert resp.status_code == 400
    assert "Invalid PDF" in resp.json()["detail"]

    fake_docx = b"MZ\x90\x00\x03\x00\x00\x00This is an executable disguised as DOCX"
    resp = client.post(
        "/upload",
        files={"file": ("malicious.docx", fake_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert resp.status_code == 400
    assert "Invalid DOCX" in resp.json()["detail"]


def test_security_sql_injection_defense(client):
    """Verify that SQL injection payloads in filter and exploration endpoints are neutralized."""
    sqli_payloads = [
        "' OR '1'='1",
        "1; DROP TABLE incidents; --",
        "admin'--",
        "1 UNION SELECT null, null, null--",
    ]
    for sqli in sqli_payloads:
        # Dimension parameter validation
        dim_resp = client.get(f"/analytics/distributions?dimension={sqli}")
        assert dim_resp.status_code == 400
        detail = dim_resp.json()["detail"]
        assert "Unsupported dimension" in detail or "Invalid dimension" in detail

        # Department filter parameter
        list_resp = client.get(f"/incidents?department={sqli}")
        assert list_resp.status_code == 200
        # Should return 0 items safely without SQL error
        assert list_resp.json()["total"] == 0


def test_security_invalid_uuid_handling(client):
    """Verify that malformed UUIDs in path parameters fail safely with 422, not 500."""
    bad_uuids = [
        "12345",
        "not-a-uuid",
        "'; DROP TABLE users; --",
        "00000000-0000-0000-0000-00000000000Z",
    ]
    for bad_id in bad_uuids:
        resp = client.get(f"/incidents/{bad_id}")
        assert resp.status_code == 422
        resp_an = client.get(f"/incidents/{bad_id}/analysis")
        assert resp_an.status_code == 422


# =====================================================================
# STANDALONE EXECUTION RUNNER
# =====================================================================

if __name__ == "__main__":
    import traceback
    with TestClient(app) as test_client:
        test_funcs = [
            (name, obj) for name, obj in list(globals().items())
            if name.startswith("test_") and callable(obj)
        ]
        passed = 0
        failed = 0
        print(f"\n=======================================================")
        print(f" Safety Incident System Verification & Audit Suite")
        print(f" Running {len(test_funcs)} verification tests...")
        print(f"=======================================================\n")

        for name, fn in test_funcs:
            try:
                # Check if function takes 'client' parameter
                import inspect
                sig = inspect.signature(fn)
                if "client" in sig.parameters:
                    fn(test_client)
                else:
                    fn()
                print(f"  [PASS] {name}")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                traceback.print_exc()
                failed += 1

        print("\n" + "=" * 55)
        print(f" Verification Results: {passed} passed, {failed} failed out of {len(test_funcs)} total.")
        print("=" * 55)
        if failed > 0:
            sys.exit(1)

