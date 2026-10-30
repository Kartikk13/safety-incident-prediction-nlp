"""Single integration point between the Streamlit UI and the FastAPI backend.

In live mode (default), all queries and commands connect to the FastAPI backend
and PostgreSQL database. If the backend is unreachable or returns an error,
it raises an ApiError explaining the failure to the user; it NEVER silently
falls back to synthetic mock data.

In demo mode (enabled by setting SAFETY_DEMO_MODE=1 or SAFETY_API_URL=demo),
it serves synthetic demonstration fixtures from data/mock_data.py with clear
UI warnings.
"""

from __future__ import annotations

import os
from typing import Any
import pandas as pd
import requests
import streamlit as st

from data import mock_data

API_URL = os.getenv("SAFETY_API_URL", "http://127.0.0.1:8502").rstrip("/")
IS_DEMO_MODE = os.getenv("SAFETY_DEMO_MODE", "0").lower() in ("1", "true", "yes") or API_URL == "demo"
USE_MOCK = IS_DEMO_MODE

# Pipeline stages for UI progress display
ANALYSIS_STAGES = [
    "Report received",
    "Text preprocessing",
    "Extracting incident information",
    "Root cause analysis",
    "Severity classification",
    "Historical comparison",
]


class ApiError(Exception):
    """Raised with a safe, informative message for Streamlit users."""


_REQUIRED = ("incident", "extracted_information", "root_causes", "severity", "historical_analysis")


def _normalize(result: dict) -> dict:
    if not isinstance(result, dict) or any(k not in result for k in _REQUIRED):
        raise ApiError("The analysis service returned an unexpected response structure. Please try again.")
    result["extracted_information"] = {
        k: list(result["extracted_information"].get(k) or [])
        for k in (
            "equipment", "hazards", "materials", "human_factors",
            "maintenance_factors", "operational_factors", "environmental_factors",
        )
    }
    result["root_causes"] = result.get("root_causes") or []
    for h in ("similar_incidents", "same_equipment_incidents", "recurring_root_causes"):
        result["historical_analysis"].setdefault(h, [])
    return result


def _map_severity_score(sev: str | None) -> float:
    mapping = {
        "Critical": 0.95,
        "Severe": 0.80,
        "Serious": 0.60,
        "Moderate": 0.40,
        "Minor": 0.20,
        "High": 0.85,
        "Medium": 0.50,
        "Low": 0.25,
    }
    return mapping.get(sev or "", 0.35)


@st.cache_data(show_spinner=False, ttl=60)
def list_incidents(
    source: str | None = None,
    limit: int = 200,
    page: int = 1,
    department: str | None = None,
    location: str | None = None,
    equipment_type: str | None = None,
    incident_type: str | None = None,
    severity_level: str | None = None,
    root_cause: str | None = None,
    q: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Retrieve incidents from PostgreSQL via FastAPI and format for Streamlit views."""
    if IS_DEMO_MODE:
        df = mock_data.build_history()
        if source:
            df = df[df.get("source", "user_submission") == source]
        return df

    params: dict[str, Any] = {"page": page, "size": min(limit, 100)}
    if source:
        params["record_source"] = source
    if department:
        params["department"] = department
    if location:
        params["location"] = location
    if equipment_type:
        params["equipment_type"] = equipment_type
    if incident_type:
        params["incident_type"] = incident_type
    if severity_level:
        params["severity_level"] = severity_level
    if root_cause:
        params["root_cause"] = root_cause
    if q:
        params["q"] = q
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    try:
        response = requests.get(f"{API_URL}/incidents", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])

        # If a larger limit was requested and there are more pages, fetch remaining items
        if limit > 100 and data.get("pages", 1) > 1:
            total_pages = min(data.get("pages", 1), (limit + 99) // 100)
            for next_page in range(2, total_pages + 1):
                p_params = dict(params, page=next_page, size=100)
                next_resp = requests.get(f"{API_URL}/incidents", params=p_params, timeout=15)
                if next_resp.status_code == 200:
                    items.extend(next_resp.json().get("items", []))
                if len(items) >= limit:
                    items = items[:limit]
                    break

        if not items:
            return pd.DataFrame(columns=[
                "id", "date", "type", "equipment", "equipment_tag", "severity",
                "root_cause", "risk", "unit", "location", "department", "summary",
                "source", "severity_score"
            ])

        records = []
        for item in items:
            preds = item.get("predictions") or {}
            sev = item.get("actual_severity_level") or preds.get("severity_level") or "Unknown"
            rc = item.get("actual_root_cause_category") or preds.get("root_cause_category") or "Unknown"
            rk = item.get("actual_risk_level") or preds.get("risk_level") or "Unknown"
            dt_str = item.get("incident_date") or item.get("created_at")

            records.append({
                "id": str(item["id"]),
                "date": pd.to_datetime(dt_str) if dt_str else pd.Timestamp.now(),
                "type": item.get("incident_type") or "General Incident",
                "equipment": item.get("equipment_type") or "Not Applicable",
                "equipment_tag": item.get("equipment_type") or "Not Applicable",
                "severity": sev,
                "root_cause": rc,
                "risk": rk,
                "unit": item.get("department") or item.get("location") or "General Facility",
                "location": item.get("location") or "General Facility",
                "department": item.get("department") or "General",
                "summary": item.get("incident_description") or "",
                "source": item.get("record_source") or "user_submission",
                "severity_score": _map_severity_score(sev),
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df

    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Could not connect to backend at {API_URL}. Please ensure the FastAPI server is running."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError("The request to the backend service timed out. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Incident retrieval failed: {exc}") from exc


def get_incident_analysis(incident_id: str) -> dict:
    """Retrieve full analysis dictionary for a specific incident."""
    if IS_DEMO_MODE:
        df = list_incidents()
        row = df[df["id"] == incident_id]
        if row.empty:
            raise ApiError(f"Incident {incident_id} was not found.")
        r = row.iloc[0]
        result = mock_data.mock_analyze(r["summary"])
        result["incident"].update(id=r["id"], date=r["date"].strftime("%Y-%m-%d"), type=r["type"], location=r["unit"])
        result["severity"].update(label=r["severity"], score=float(r["severity_score"]))
        result["historical_analysis"] = mock_data.mock_history(result, df[df["id"] != incident_id])
        return _normalize(result)

    try:
        response = requests.get(
            f"{API_URL}/incidents/{incident_id}/analysis",
            timeout=20,
        )
        if response.status_code == 404:
            raise ApiError(f"Incident {incident_id} was not found in the database.")
        response.raise_for_status()
        result = response.json()
        return _normalize(result)
    except ApiError:
        raise
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Could not connect to backend at {API_URL}. Please ensure FastAPI is running."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Failed to fetch analysis for incident {incident_id}: {exc}") from exc


def analyze_incident(text: str, on_stage=None) -> dict:
    """Direct end-to-end report analysis helper."""
    if IS_DEMO_MODE:
        result = mock_data.mock_analyze(text)
        result["historical_analysis"] = mock_data.mock_history(result, list_incidents())
        return _normalize(result)

    # In live mode, run through real NLP extraction and prediction
    payload = {
        "incident_description": text.strip(),
        "original_report_text": text.strip(),
        "submission_method": "paste",
    }
    result = predict_incident(payload)
    incident_id = result.get("incident_id")
    if not incident_id:
        raise ApiError("The prediction service did not return an incident identifier.")
    return get_incident_analysis(incident_id)


def upload_incident_file(
    file_bytes: bytes,
    original_filename: str,
    mime_type: str | None = None,
) -> dict:
    """Upload an incident document to private backend storage."""
    if IS_DEMO_MODE:
        return {
            "storage_key": f"demo-{original_filename}",
            "original_filename": original_filename,
            "size_bytes": len(file_bytes),
            "file_size_bytes": len(file_bytes),
            "sha256_hash": "demo_hash",
        }

    try:
        response = requests.post(
            f"{API_URL}/upload",
            files={
                "file": (
                    original_filename,
                    file_bytes,
                    mime_type or "application/octet-stream",
                )
            },
            timeout=60,
        )
        if response.status_code == 400:
            detail = response.json().get("detail", "Invalid upload payload.")
            raise ApiError(detail)
        response.raise_for_status()

        result = response.json()
        metadata = result.get("file")

        if not isinstance(metadata, dict) or not metadata.get("storage_key"):
            raise ApiError("The upload service returned an unexpected response.")

        return metadata

    except ApiError:
        raise
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Could not connect to backend at {API_URL}. Please ensure FastAPI is running."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError(
            "Could not upload document. Check that FastAPI is running."
        ) from exc


def predict_incident(
    incident: dict,
    *,
    original_report_text: str | None = None,
    submission_method: str | None = None,
    submission_payload: dict | None = None,
    uploaded_file_key: str | None = None,
) -> dict:
    """Send model features and submission metadata to the prediction API."""
    if IS_DEMO_MODE:
        mock_pred = {
            "root_cause_category": "Procedure Not Followed",
            "severity_level": "Medium",
            "risk_level": "Medium",
            "incident_id": "DEMO-001",
            "prediction_run_id": "DEMO-RUN-001",
            "status": "success",
            "predictions": {
                "root_cause_category": "Procedure Not Followed",
                "severity_level": "Medium",
                "risk_level": "Medium",
            },
        }
        return mock_pred

    payload = dict(incident)
    if original_report_text is not None:
        payload["original_report_text"] = original_report_text
    if submission_method is not None:
        payload["submission_method"] = submission_method
    if submission_payload is not None:
        payload["submission_payload"] = submission_payload
    if uploaded_file_key is not None:
        payload["uploaded_file_key"] = uploaded_file_key

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=120,
        )
        if response.status_code == 400:
            detail = response.json().get("detail", "Invalid prediction request.")
            raise ApiError(f"Validation error: {detail}")
        response.raise_for_status()

        result = response.json()
        predictions = result.get("predictions")
        expected = {"root_cause_category", "severity_level", "risk_level"}

        if not isinstance(predictions, dict) or not expected.issubset(predictions):
            raise ApiError("The prediction API returned an unexpected response.")

        # Provide top-level convenience keys for backward compatibility
        result["root_cause_category"] = predictions["root_cause_category"]
        result["severity_level"] = predictions["severity_level"]
        result["risk_level"] = predictions["risk_level"]

        # Invalidate cache so history and dashboard reflect new incident
        list_incidents.clear()

        return result

    except ApiError:
        raise
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Could not connect to backend at {API_URL}. Please ensure FastAPI is running."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError(
            "Could not connect to the prediction API. Check that FastAPI is running on the configured API port."
        ) from exc


def get_kpis(source: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:
    """Fetch dashboard KPI metrics computed via SQL aggregation."""
    if IS_DEMO_MODE:
        df = list_incidents()
        high = df[df["severity"].isin(["High", "Critical", "Severe"])]
        return {
            "total_incidents": len(df),
            "high_severity_count": len(high),
            "high_severity_pct": round(len(high) / max(len(df), 1) * 100, 1),
            "last_90_days_count": len(df),
            "prev_90_days_count": len(df),
            "last_90_days_delta": 0,
            "top_high_severity_equipment_past_year": "P-101A",
            "top_equipment_high_severity_count": len(high),
        }
    params: dict[str, Any] = {}
    if source:
        params["source"] = source
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    resp = requests.get(f"{API_URL}/analytics/kpis", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_trends(source: str | None = None, freq: str = "monthly") -> list[dict]:
    """Fetch time series trends computed via SQL."""
    if IS_DEMO_MODE:
        return []
    params: dict[str, Any] = {"freq": freq}
    if source:
        params["source"] = source
    resp = requests.get(f"{API_URL}/analytics/trends", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_distributions(dim: str, source: str | None = None, limit: int = 10) -> list[dict]:
    """Fetch category distribution counts computed via SQL."""
    if IS_DEMO_MODE:
        return []
    params: dict[str, Any] = {"dimension": dim, "dim": dim, "top": limit, "limit": limit}
    if source:
        params["source"] = source
        params["record_source"] = source
    resp = requests.get(f"{API_URL}/analytics/distributions", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_heatmap(source: str | None = None, top_equipment: int = 10, top_causes: int = 8) -> dict:
    """Fetch cross-tabulation matrix computed via SQL."""
    if IS_DEMO_MODE:
        return {"equipment": [], "root_causes": [], "matrix": []}
    params: dict[str, Any] = {"top_equipment": top_equipment, "top_causes": top_causes}
    if source:
        params["source"] = source
    resp = requests.get(f"{API_URL}/analytics/heatmap", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_dataset_summary() -> dict:
    """Fetch historical training dataset statistics."""
    if IS_DEMO_MODE:
        return {}
    resp = requests.get(f"{API_URL}/analytics/dataset-summary", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_eval_metrics() -> dict:
    """Fetch model evaluation metrics and population coverage from backend."""
    if IS_DEMO_MODE:
        return {
            "evaluated_records": 0,
            "total_historical_with_ground_truth": 8000,
            "total_user_submissions": 0,
            "total_prediction_runs": 0,
            "overlap_records_with_both": 0,
            "accuracy": {},
            "methodology_note": "Demo mode: No evaluation metrics available.",
        }
    resp = requests.get(f"{API_URL}/analytics/eval-metrics", timeout=15)
    resp.raise_for_status()
    return resp.json()

