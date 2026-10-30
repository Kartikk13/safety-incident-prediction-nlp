#!/usr/bin/env python3
"""
test_production_smoke.py

Production smoke test suite to run against any deployed environment.
Verifies critical health, ML inference, persistence, and UI connectivity.

Usage:
    python test_production_smoke.py --api-url http://localhost:8502 --frontend-url http://localhost:8501
"""

import argparse
import sys
import uuid
import requests


def run_smoke_tests(api_url: str, frontend_url: str | None = None) -> bool:
    api_url = api_url.rstrip("/")
    if frontend_url:
        frontend_url = frontend_url.rstrip("/")

    print("\n" + "=" * 65)
    print(f" Production Deployment Smoke Test Suite")
    print(f" Target API: {api_url}")
    if frontend_url:
        print(f" Target Frontend: {frontend_url}")
    print("=" * 65 + "\n")

    passed = 0
    total = 0

    def check(name: str, fn):
        nonlocal passed, total
        total += 1
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    # 1. Health endpoint
    def test_health():
        r = requests.get(f"{api_url}/health", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert data["status"] == "ok", f"Expected status 'ok', got {data.get('status')}"
        assert data["database"] == "connected", f"Expected db 'connected', got {data.get('database')}"
        assert set(data["models_loaded"]) == {"root_cause_category", "severity_level", "risk_level"}
    check("1. GET /health - Service, Database, and 3 ML Models Loaded", test_health)

    # 2. Models inventory
    def test_models():
        r = requests.get(f"{api_url}/models", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert len(data) >= 3
        for m in data:
            assert m["is_loaded"] is True
            assert m["artifact_sha256"]
    check("2. GET /models - Verify Registered Model Checksums", test_models)

    # 3. NLP extraction
    def test_nlp_extract():
        payload = {
            "text": "On 2026-10-30 at 14:15 a technician in Maintenance observed a broken Conveyor.",
            "context": {"department": "Maintenance"},
        }
        r = requests.post(f"{api_url}/nlp/extract", json=payload, timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        f = r.json()["features"]
        assert f["equipment_type"] == "Conveyor"
        assert f["equipment_status"] == "Damaged"
        assert f["department"] == "Maintenance"
    check("3. POST /nlp/extract - Rule-Based & Aliased Feature Extraction", test_nlp_extract)

    # 4. Upload document
    created_storage_key = None
    def test_upload():
        nonlocal created_storage_key
        content = b"Smoke test incident upload: oil leakage detected on floor."
        files = {"file": ("smoke_report.txt", content, "text/plain")}
        r = requests.post(f"{api_url}/upload", files=files, timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        file_meta = r.json()["file"]
        assert "storage_path" not in file_meta, "Internal storage path must not be exposed"
        assert file_meta["original_filename"] == "smoke_report.txt"
        created_storage_key = file_meta["storage_key"]
    check("4. POST /upload - Private Storage & Path Disclosure Prevention", test_upload)

    # 5. Predict & persist
    created_incident_id = None
    def test_predict():
        nonlocal created_incident_id
        payload = {
            "incident_description": "Smoke test production prediction: oil leakage on pump base.",
            "department": "Production",
            "equipment_type": "Pump",
            "equipment_status": "Damaged",
            "hazard_type": "Oil Leakage",
            "submission_method": "upload" if created_storage_key else "paste",
            "uploaded_file_key": created_storage_key,
        }
        r = requests.post(f"{api_url}/predict", json=payload, timeout=15)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        res = r.json()
        assert "incident_id" in res
        created_incident_id = res["incident_id"]
        preds = res["predictions"]
        assert "root_cause_category" in preds
        assert "severity_level" in preds
        assert "risk_level" in preds
    check("5. POST /predict - Multi-Target Inference & Database Persistence", test_predict)

    # 6. Retrieve incident detail
    def test_get_incident():
        assert created_incident_id is not None
        r = requests.get(f"{api_url}/incidents/{created_incident_id}", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        inc = r.json()["incident"]
        assert inc["id"] == created_incident_id
        assert inc["record_source"] == "user_submission"
    check("6. GET /incidents/{id} - Persisted Incident Detail Retrieval", test_get_incident)

    # 7. Retrieve analysis view
    def test_get_analysis():
        assert created_incident_id is not None
        r = requests.get(f"{api_url}/incidents/{created_incident_id}/analysis", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        an = r.json()
        assert "root_causes" in an
        assert "severity" in an
        assert "historical_analysis" in an
    check("7. GET /incidents/{id}/analysis - UI Analysis Payload Verification", test_get_analysis)

    # 8. Dashboard KPIs
    def test_kpis():
        r = requests.get(f"{api_url}/analytics/kpis", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        kpis = r.json()
        assert kpis["historical_count"] == 8000
        assert kpis["total_incidents"] >= 8000
        assert kpis["user_submission_count"] >= 1
    check("8. GET /analytics/kpis - SQL Aggregated Population Accounting", test_kpis)

    # 9. Frontend healthcheck
    if frontend_url:
        def test_frontend_health():
            r = requests.get(f"{frontend_url}/_stcore/health", timeout=10)
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            assert r.text.strip().lower() == "ok"
        check("9. GET /_stcore/health - Streamlit Frontend Healthcheck", test_frontend_health)

    print("\n" + "=" * 65)
    print(f" Smoke Test Results: {passed}/{total} checks passed.")
    print("=" * 65 + "\n")
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Deployment Smoke Test Suite")
    parser.add_argument("--api-url", default="http://127.0.0.1:8502", help="FastAPI backend URL")
    parser.add_argument("--frontend-url", default=None, help="Streamlit frontend URL (optional)")
    args = parser.parse_args()

    success = run_smoke_tests(api_url=args.api_url, frontend_url=args.frontend_url)
    if not success:
        sys.exit(1)
