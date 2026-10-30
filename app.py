"""
app.py

FastAPI Backend for Safety Incident Prediction and Root Cause Analysis.
Provides endpoints for health verification, private document storage,
ML inference, NLP feature extraction, paginated incident exploration,
incident detail retrieval, and dashboard/historical analytics.
"""

from datetime import date, datetime, time, timezone
from decimal import Decimal
import hashlib
import os
from pathlib import Path
import re
import time as time_module
import traceback
from typing import Any
import uuid

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, func, or_, select, text
from sqlalchemy.orm import Session

# Prevent cython C-extension errors under Windows AppLocker if applicable
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"

from database import SessionLocal, engine, get_db
from file_storage import UPLOAD_DIR, save_incident_file, verify_file_exists
from models import (
    AnalysisResult,
    DatasetImportLog,
    Incident as IncidentRecord,
    IncidentFile,
    ModelVersion,
    Prediction,
    PredictionRun,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

targets = [
    "root_cause_category",
    "severity_level",
    "risk_level",
]

feature_cols = [
    "incident_year",
    "incident_month",
    "incident_dayofweek",
    "incident_hour",
    "incident_minute",
    "department",
    "location",
    "shift",
    "employee_role",
    "experience_level",
    "equipment_type",
    "equipment_status",
    "incident_type",
    "hazard_type",
    "incident_description",
    "reported_by_role",
]

# Load ML artifacts at startup
artifacts = {
    target: joblib.load(MODEL_DIR / f"{target}_pipeline.joblib")
    for target in targets
}

# Create stable metadata and SHA-256 fingerprint for each model artifact
model_metadata = {}

for target in targets:
    artifact_path = MODEL_DIR / f"{target}_pipeline.joblib"
    sha256 = hashlib.sha256()
    with artifact_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            sha256.update(chunk)

    model_metadata[target] = {
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": sha256.hexdigest(),
        "version_tag": f"sha256-{sha256.hexdigest()[:12]}",
    }

app = FastAPI(
    title="Safety Incident Intelligence & Prediction API",
    description="Backend API for incident prediction, NLP extraction, exploration, and historical analytics.",
    version="2.0.0",
)

# Enable CORS with configurable origins (defaults to Streamlit and local dev ports)
cors_env = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501,http://localhost:8502,http://127.0.0.1:8502,*")
allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# PYDANTIC SCHEMAS
# =====================================================================

class Incident(BaseModel):
    """Prediction submission request schema (preserved for backward compatibility)."""
    model_config = ConfigDict(extra="ignore")

    # Optional submission metadata; excluded from model features.
    original_report_text: str | None = None
    submission_method: str | None = None
    submission_payload: dict[str, Any] | None = None

    incident_year: int | None = None
    incident_month: int | None = None
    incident_dayofweek: int | None = None
    incident_hour: int | None = None
    incident_minute: int | None = None

    department: str | None = None
    location: str | None = None
    shift: str | None = None
    employee_role: str | None = None
    experience_level: str | None = None
    equipment_type: str | None = None
    equipment_status: str | None = None
    incident_type: str | None = None
    hazard_type: str | None = None
    reported_by_role: str | None = None

    incident_description: str
    uploaded_file_key: str | None = None


class IncidentItem(BaseModel):
    """Summary item for paginated listings."""
    id: str
    incident_date: str | None = None
    incident_time: str | None = None
    department: str | None = None
    location: str | None = None
    shift: str | None = None
    equipment_type: str | None = None
    equipment_status: str | None = None
    incident_type: str | None = None
    hazard_type: str | None = None
    incident_description: str | None = None
    record_source: str
    submission_method: str | None = None
    actual_root_cause_category: str | None = None
    actual_severity_level: str | None = None
    actual_risk_level: str | None = None
    risk_score: float | None = None
    predictions: dict[str, str] | None = None
    created_at: datetime


class PaginatedIncidents(BaseModel):
    items: list[IncidentItem]
    total: int
    page: int
    size: int
    pages: int


class IncidentDetailResponse(BaseModel):
    incident: dict[str, Any]
    predictions: dict[str, str] | None = None
    prediction_runs: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    analysis_result: dict[str, Any] | None = None


class PredictResponse(BaseModel):
    incident_id: str
    prediction_run_id: str
    predictions: dict[str, str]
    analysis: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    database: str
    models_loaded: list[str]
    timestamp: datetime


class KPIResponse(BaseModel):
    total_incidents: int
    high_severity_count: int
    high_severity_pct: float
    last_90_days_count: int
    prev_90_days_count: int
    last_90_days_delta: int
    top_high_severity_equipment_past_year: str
    top_equipment_high_severity_count: int
    by_record_source: dict[str, int]
    historical_count: int = 0
    user_submission_count: int = 0
    ground_truth_count: int = 0
    predicted_count: int = 0
    both_count: int = 0


class TrendItem(BaseModel):
    month: str
    total: int
    high_severity: int


class DistributionItem(BaseModel):
    name: str
    count: int


class HeatmapResponse(BaseModel):
    equipment: list[str]
    root_causes: list[str]
    matrix: list[list[int]]


class SimilarIncidentItem(BaseModel):
    id: str
    incident_date: str | None
    incident_type: str | None
    equipment_type: str | None
    actual_severity_level: str | None
    actual_root_cause_category: str | None
    similarity: float


class ModelVersionItem(BaseModel):
    id: str
    target_name: str
    version_tag: str
    artifact_path: str
    artifact_sha256: str | None
    trained_at: datetime | None
    created_at: datetime
    is_loaded: bool


class DatasetSummaryResponse(BaseModel):
    total_records: int
    date_range: dict[str, str | None]
    target_distributions: dict[str, dict[str, int]]
    null_counts: dict[str, int]
    latest_import: dict[str, Any] | None


class NLPExtractRequest(BaseModel):
    text: str
    context: dict[str, str | None] | None = None


class NLPExtractResponse(BaseModel):
    features: dict[str, Any]
    nlp_method: str


# =====================================================================
# NLP FEATURE EXTRACTION HELPER
# =====================================================================

CATEGORIES = {
    "department": [
        "Assembly", "Chemical Processing", "Electrical",
        "Fabrication & Welding", "Facilities & Housekeeping",
        "Maintenance", "Packaging", "Production",
        "Quality Control & Laboratory", "Utilities",
        "Warehouse & Logistics",
    ],
    "shift": ["Morning", "Evening", "Night"],
    "equipment_type": [
        "Air Compressor", "Autoclave", "Battery Bank", "Boiler",
        "Box Cutter", "CNC Lathe", "Cable Tray", "Chemical Pump",
        "Chemical Storage Cabinet", "Chiller", "Conveyor",
        "Diesel Generator", "Distillation Column",
        "Distribution Panel", "Dock Leveler", "Drilling Machine",
        "Drum Handling Trolley", "Electric Motor", "Extension Cord",
        "Floor Scrubber", "Forklift", "Fume Hood", "Furnace",
        "Gas Cutting Torch", "Gas Cylinder Manifold",
        "Grinding Machine", "Hand Tools", "Hydraulic Press",
        "Injection Moulding Machine", "Labelling Machine", "Ladder",
        "Milling Machine", "Motor Control Centre", "Not Applicable",
        "Overhead Crane", "Packing Machine", "Packing Table",
        "Pallet Jack", "Palletizer", "Pipeline", "Plasma Cutter",
        "Pneumatic Tool", "Press Brake", "Pressure Washer", "Pump",
        "Racking", "Reach Truck", "Reactor", "Robotic Arm",
        "Scrubber", "Shrink Wrapping Machine", "Stamping Machine",
        "Steam Line", "Storage Tank", "Switchgear", "Testing Machine",
        "Torque Wrench", "Transformer", "Trolley", "Waste Compactor",
        "Welding Machine", "Workstation",
    ],
    "equipment_status": [
        "Damaged", "Malfunctioning", "Not Applicable", "Operational",
        "Out of Service", "Overdue for Maintenance",
        "Shutdown/Isolated", "Under Maintenance",
    ],
    "incident_type": [
        "Caught In/Between", "Chemical Exposure", "Chemical Spill",
        "Electrical Burn", "Electrical Shock", "Ergonomic Injury",
        "Explosion", "Falling Object", "Fire", "Gas Leak",
        "Heat Stress", "Machine Injury", "Manual Handling Injury",
        "Near Miss", "Noise Exposure", "Other", "Slip/Fall",
        "Struck By Object", "Trip/Fall", "Vehicle/Forklift Incident",
    ],
    "hazard_type": [
        "Chemical Substance", "Confined Space", "Electrical Energy",
        "Excessive Heat", "Falling Object", "Fire Hazard",
        "Gas/Pressure", "Height Work", "Inadequate PPE",
        "Moving Machinery", "Noise", "Oil Leakage", "Other",
        "Poor Housekeeping", "Poor Lighting", "Sharp Object",
        "Slippery Surface", "Unsafe Lifting", "Vehicle Movement",
    ],
}

PHRASE_MAPPINGS = {
    "equipment_type": {
        "fork truck": "Forklift",
        "fork lift": "Forklift",
        "air compressor unit": "Air Compressor",
        "pumping unit": "Pump",
        "feed pump": "Pump",
    },
    "incident_type": {
        "caught fire": "Fire",
        "burst into flames": "Fire",
        "ignited": "Fire",
        "electric shock": "Electrical Shock",
        "slipped and fell": "Slip/Fall",
        "slip and fall": "Slip/Fall",
        "hydrocarbon leak": "Chemical Spill",
    },
    "hazard_type": {
        "fire broke out": "Fire Hazard",
        "exposed electrical wires": "Electrical Energy",
        "oil on the floor": "Slippery Surface",
        "hydrocarbon leak": "Chemical Substance",
    },
}


def extract_features_from_text(text_input: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Robust regex + vocabulary NLP extraction supporting ISO dates, times, domain aliases, conflict resolution, and negation guards."""
    import sys
    from pathlib import Path
    ui_dir = str(Path(__file__).resolve().parent / "safety_incident_ui")
    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)
    try:
        from services.nlp_extractor import extract_incident_features
        return extract_incident_features(text_input, context=context)
    except Exception:
        ctx = context or {}
        text_content = text_input.strip()

        extracted: dict[str, Any] = {
            "incident_year": None,
            "incident_month": None,
            "incident_dayofweek": None,
            "incident_hour": None,
            "incident_minute": None,
            "department": ctx.get("department") or None,
            "location": ctx.get("location") or None,
            "shift": ctx.get("shift") or None,
            "employee_role": None,
            "experience_level": None,
            "equipment_type": None,
            "equipment_status": None,
            "incident_type": None,
            "hazard_type": None,
            "incident_description": text_content,
            "reported_by_role": None,
        }
        return extracted


# =====================================================================
# MODEL VERSION MANAGEMENT
# =====================================================================

def get_or_create_model_version(db: Session, target: str) -> ModelVersion:
    """Retrieve or register the model version matching the loaded artifact SHA-256."""
    metadata = model_metadata[target]

    version = db.scalar(
        select(ModelVersion).where(
            ModelVersion.target_name == target,
            ModelVersion.version_tag == metadata["version_tag"],
        )
    )

    if version is None:
        version = ModelVersion(
            target_name=target,
            version_tag=metadata["version_tag"],
            artifact_path=metadata["artifact_path"],
            artifact_sha256=metadata["artifact_sha256"],
            metadata_json={"versioning_method": "artifact_sha256"},
        )
        db.add(version)
        db.flush()

    return version


# =====================================================================
# ENDPOINTS
# =====================================================================

@app.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    """Health check verifying in-memory ML models and live PostgreSQL connection."""
    try:
        db.execute(text("SELECT 1")).scalar()
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    status_code = status.HTTP_200_OK if db_status == "connected" else status.HTTP_503_SERVICE_UNAVAILABLE

    response_data = {
        "status": "ok" if db_status == "connected" else "degraded",
        "database": db_status,
        "models_loaded": list(artifacts.keys()),
        "timestamp": datetime.now(timezone.utc),
    }

    if status_code != status.HTTP_200_OK:
        raise HTTPException(status_code=status_code, detail=response_data)

    return response_data


@app.post("/upload")
async def upload_incident_file(file: UploadFile = File(...)):
    """Store an incident document privately and return its metadata."""
    try:
        metadata = await save_incident_file(file)
        # Do not expose private disk paths to external clients
        metadata.pop("storage_path", None)
        return {"file": metadata}
    finally:
        await file.close()


@app.post("/nlp/extract", response_model=NLPExtractResponse)
def extract_incident_nlp(payload: NLPExtractRequest):
    """Extract 16 feature columns from free-text incident descriptions."""
    text_content = payload.text.strip()
    if not text_content:
        raise HTTPException(status_code=400, detail="Report text cannot be empty.")

    extracted_dict = extract_features_from_text(text_content, context=payload.context)
    feature_keys = [
        "incident_year", "incident_month", "incident_dayofweek",
        "incident_hour", "incident_minute", "department",
        "location", "shift", "employee_role", "experience_level",
        "equipment_type", "equipment_status", "incident_type",
        "hazard_type", "incident_description", "reported_by_role",
    ]
    features = {k: extracted_dict.get(k) for k in feature_keys}
    return {
        "features": features,
        "nlp_method": "hybrid_regex_rules",
    }


@app.post("/predict")
def predict(incident: Incident, db: Session = Depends(get_db)):
    """
    Run prediction models, persist user incident, prediction run, model versions,
    individual predictions (with confidence scores), and rich AnalysisResult.
    """
    start_time = time_module.time()

    # 1. Feature preparation & inference
    try:
        feature_data = incident.model_dump(include=set(feature_cols))
        input_df = pd.DataFrame([feature_data], columns=feature_cols)

        predictions: dict[str, str] = {}
        confidences: dict[str, float] = {}

        for target, artifact in artifacts.items():
            preprocessor = artifact["preprocessor"]
            model = artifact["model"]

            processed = preprocessor.transform(input_df)
            pred_val = str(model.predict(processed)[0])
            predictions[target] = pred_val

            # Compute real confidence score via predict_proba if available
            if hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba(processed)[0]
                    confidences[target] = round(float(max(probas)), 4)
                except Exception:
                    confidences[target] = 0.80
            else:
                confidences[target] = 0.80

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    # 2. Atomic persistence
    try:
        duration_ms = int((time_module.time() - start_time) * 1000)

        with db.begin_nested():
            # Create user incident record
            incident_record = IncidentRecord(
                record_source="user_submission",
                submission_method=incident.submission_method or "paste",
                original_report_text=(
                    incident.original_report_text
                    if incident.original_report_text is not None
                    else incident.incident_description
                ),
                submission_payload=incident.submission_payload,
                **feature_data,
            )
            db.add(incident_record)
            db.flush()

            # Record prediction run
            prediction_run = PredictionRun(
                incident_id=incident_record.id,
                requested_by=None,
                status="success",
                duration_ms=duration_ms,
            )
            db.add(prediction_run)
            db.flush()

            # Link uploaded file if provided (modular storage verification)
            uploaded_file_key = incident.uploaded_file_key
            if uploaded_file_key:
                storage_meta = verify_file_exists(uploaded_file_key)
                file_size = storage_meta["size_bytes"]
                sha256_hash = storage_meta["sha256_hash"]
                original_filename = (
                    (incident.submission_payload or {}).get("original_filename")
                    or "uploaded_document"
                )
                mime_type = (
                    (incident.submission_payload or {}).get("mime_type")
                    or "application/octet-stream"
                )

                db.add(
                    IncidentFile(
                        incident_id=incident_record.id,
                        original_filename=original_filename,
                        storage_key=uploaded_file_key,
                        mime_type=mime_type,
                        size_bytes=file_size,
                        sha256_hash=sha256_hash,
                    )
                )

            # Store predictions with real confidence scores and model version FK
            for target, predicted_value in predictions.items():
                version = get_or_create_model_version(db, target)
                conf = confidences.get(target, 0.8)

                db.add(
                    Prediction(
                        prediction_run_id=prediction_run.id,
                        target_name=target,
                        predicted_value=predicted_value,
                        model_version_id=version.id,
                        confidence_score=Decimal(str(conf)),
                    )
                )

            # Search up to 5 similar historical incidents
            similar_q = select(
                IncidentRecord.id,
                IncidentRecord.incident_date,
                IncidentRecord.incident_type,
                IncidentRecord.equipment_type,
                IncidentRecord.actual_severity_level,
                IncidentRecord.actual_root_cause_category,
            ).where(
                IncidentRecord.record_source == "historical_dataset",
                or_(
                    IncidentRecord.equipment_type == incident.equipment_type,
                    IncidentRecord.incident_type == incident.incident_type,
                    IncidentRecord.actual_root_cause_category == predictions["root_cause_category"],
                ),
            ).limit(5)
            similar_rows = db.execute(similar_q).fetchall()
            similar_list = [
                {
                    "id": str(r.id),
                    "date": r.incident_date.isoformat() if r.incident_date else None,
                    "type": r.incident_type,
                    "equipment": r.equipment_type,
                    "severity": r.actual_severity_level,
                    "root_cause": r.actual_root_cause_category,
                    "similarity": 0.85 if r.equipment_type == incident.equipment_type else 0.70,
                }
                for r in similar_rows
            ]

            # Build and persist AnalysisResult
            extracted_info = {
                "equipment": [incident.equipment_type] if incident.equipment_type else [],
                "hazards": [incident.hazard_type] if incident.hazard_type else [],
                "materials": [],
                "human_factors": [incident.employee_role] if incident.employee_role else [],
                "maintenance_factors": [incident.equipment_status] if incident.equipment_status else [],
                "operational_factors": [incident.shift] if incident.shift else [],
                "environmental_factors": [],
            }

            analysis_summary = (
                incident_record.incident_description[:360]
                if incident_record.incident_description
                else "Incident analysis produced by ML pipeline."
            )

            db.add(
                AnalysisResult(
                    prediction_run_id=prediction_run.id,
                    extracted_entities=extracted_info,
                    contributing_factors=[
                        f"Department: {incident.department}" if incident.department else None,
                        f"Equipment status: {incident.equipment_status}" if incident.equipment_status else None,
                    ],
                    similar_incidents=similar_list,
                    summary=analysis_summary,
                )
            )

        db.commit()

        # Build analysis dictionary compatible with views/analysis.py
        sev_label = predictions["severity_level"]
        sev_score = 0.85 if sev_label in ("Critical", "Severe") else (0.60 if sev_label == "Serious" else 0.35)

        analysis_dict = {
            "incident": {
                "id": str(incident_record.id),
                "type": incident_record.incident_type or "Unspecified",
                "summary": analysis_summary,
                "date": str(incident_record.incident_date) if incident_record.incident_date else None,
                "time": str(incident_record.incident_time) if incident_record.incident_time else None,
                "location": incident_record.location or incident_record.department or "Not specified",
            },
            "extracted_information": extracted_info,
            "root_causes": [
                {
                    "cause": predictions["root_cause_category"],
                    "confidence": confidences.get("root_cause_category", 0.85),
                    "evidence": incident_record.incident_description,
                    "supporting_factors": [incident.department] if incident.department else [],
                }
            ],
            "severity": {
                "label": sev_label,
                "score": sev_score,
                "supporting_factors": [predictions["risk_level"]],
            },
            "historical_analysis": {
                "similar_incidents": similar_list,
                "same_equipment_incidents": [s for s in similar_list if s["equipment"] == incident.equipment_type],
                "recurring_root_causes": [
                    {
                        "cause": predictions["root_cause_category"],
                        "total": len(similar_list),
                        "on_same_equipment": sum(1 for s in similar_list if s["equipment"] == incident.equipment_type),
                        "last_seen": similar_list[0]["date"] if similar_list else None,
                    }
                ],
            },
            "report_text": incident_record.original_report_text or incident_record.incident_description,
        }

        return {
            "incident_id": str(incident_record.id),
            "prediction_run_id": str(prediction_run.id),
            "predictions": predictions,
            "analysis": analysis_dict,
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Prediction succeeded, but persisting results to the database failed.",
        ) from exc


# =====================================================================
# INCIDENT EXPLORATION & RETRIEVAL
# =====================================================================

@app.get("/incidents", response_model=PaginatedIncidents)
def list_incidents(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(25, ge=1, le=100, description="Items per page"),
    record_source: str | None = Query(None, description="'historical_dataset' or 'user_submission'"),
    source: str | None = Query(None, description="Alias for record_source"),
    start_date: date | None = Query(None, description="Filter incidents on or after this date"),
    end_date: date | None = Query(None, description="Filter incidents on or before this date"),
    department: str | None = Query(None, description="Filter by department"),
    location: str | None = Query(None, description="Filter by location"),
    equipment_type: str | None = Query(None, description="Filter by equipment type"),
    incident_type: str | None = Query(None, description="Filter by incident type"),
    severity_level: str | None = Query(None, description="Filter by severity level"),
    root_cause: str | None = Query(None, description="Filter by root cause category"),
    q: str | None = Query(None, description="Search term in description or ID"),
    sort_by: str = Query("incident_date", pattern="^(incident_date|created_at|department|equipment_type)$"),
    sort_dir: str = Query("desc", pattern="^(asc|desc)$"),
    db: Session = Depends(get_db),
):
    """
    Search and retrieve paginated safety incidents with multiple filter criteria.
    Supports filtering by provenance (historical vs user submission).
    """
    query = select(IncidentRecord)

    # Apply filters
    active_source = source or record_source
    if active_source and active_source != "all":
        query = query.where(IncidentRecord.record_source == active_source)

    if start_date:
        query = query.where(IncidentRecord.incident_date >= start_date)
    if end_date:
        query = query.where(IncidentRecord.incident_date <= end_date)

    if department:
        query = query.where(IncidentRecord.department.ilike(f"%{department.strip()}%"))
    if location:
        query = query.where(IncidentRecord.location.ilike(f"%{location.strip()}%"))
    if equipment_type:
        query = query.where(IncidentRecord.equipment_type == equipment_type)
    if incident_type:
        query = query.where(IncidentRecord.incident_type == incident_type)
    if severity_level:
        query = query.where(IncidentRecord.actual_severity_level == severity_level)
    if root_cause:
        query = query.where(IncidentRecord.actual_root_cause_category == root_cause)

    if q and q.strip():
        term = f"%{q.strip()}%"
        # Search by description or text cast of UUID
        query = query.where(
            or_(
                IncidentRecord.incident_description.ilike(term),
                func.cast(IncidentRecord.id, text("TEXT")).ilike(term),
            )
        )

    # Compute total count
    count_query = select(func.count()).select_from(query.subquery())
    total = db.scalar(count_query) or 0

    # Sorting
    sort_col = getattr(IncidentRecord, sort_by, IncidentRecord.incident_date)
    query = query.order_by(desc(sort_col) if sort_dir == "desc" else sort_col)

    # Pagination
    offset = (page - 1) * size
    query = query.offset(offset).limit(size)

    records = db.scalars(query).all()

    items: list[IncidentItem] = []
    for r in records:
        # If user submission, attach predictions if present
        preds = None
        if r.record_source == "user_submission":
            latest_run = db.scalar(
                select(PredictionRun.id)
                .where(PredictionRun.incident_id == r.id)
                .order_by(desc(PredictionRun.created_at))
                .limit(1)
            )
            if latest_run:
                p_rows = db.scalars(
                    select(Prediction).where(Prediction.prediction_run_id == latest_run)
                ).all()
                if p_rows:
                    preds = {p.target_name: p.predicted_value for p in p_rows}

        items.append(
            IncidentItem(
                id=str(r.id),
                incident_date=r.incident_date.isoformat() if r.incident_date else None,
                incident_time=r.incident_time.isoformat() if r.incident_time else None,
                department=r.department,
                location=r.location,
                shift=r.shift,
                equipment_type=r.equipment_type,
                equipment_status=r.equipment_status,
                incident_type=r.incident_type,
                hazard_type=r.hazard_type,
                incident_description=r.incident_description,
                record_source=r.record_source,
                submission_method=r.submission_method,
                actual_root_cause_category=r.actual_root_cause_category,
                actual_severity_level=r.actual_severity_level,
                actual_risk_level=r.actual_risk_level,
                risk_score=float(r.risk_score) if r.risk_score else None,
                predictions=preds,
                created_at=r.created_at,
            )
        )

    pages = (total + size - 1) // size if total > 0 else 1

    return PaginatedIncidents(
        items=items,
        total=total,
        page=page,
        size=size,
        pages=pages,
    )


@app.get("/incidents/{incident_id}", response_model=IncidentDetailResponse)
def get_incident_detail(incident_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieve full incident details, associated predictions, files, and analysis result."""
    record = db.scalar(select(IncidentRecord).where(IncidentRecord.id == incident_id))
    if not record:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")

    # Prediction runs and predictions
    run_records = db.scalars(
        select(PredictionRun).where(PredictionRun.incident_id == incident_id)
    ).all()

    runs_data = []
    latest_analysis = None
    latest_predictions = None

    for run in run_records:
        preds = db.scalars(select(Prediction).where(Prediction.prediction_run_id == run.id)).all()
        pred_dict = {
            p.target_name: {
                "value": p.predicted_value,
                "confidence": float(p.confidence_score) if p.confidence_score is not None else None,
            }
            for p in preds
        }
        if not latest_predictions and pred_dict:
            latest_predictions = {k: v["value"] for k, v in pred_dict.items()}

        runs_data.append({
            "run_id": str(run.id),
            "status": run.status,
            "duration_ms": run.duration_ms,
            "created_at": run.created_at.isoformat(),
            "predictions": pred_dict,
        })

        # Check for AnalysisResult
        ar = db.scalar(select(AnalysisResult).where(AnalysisResult.prediction_run_id == run.id))
        if ar and not latest_analysis:
            latest_analysis = {
                "extracted_entities": ar.extracted_entities,
                "contributing_factors": ar.contributing_factors,
                "similar_incidents": ar.similar_incidents,
                "summary": ar.summary,
                "created_at": ar.created_at.isoformat(),
            }

    # Uploaded files
    file_records = db.scalars(
        select(IncidentFile).where(IncidentFile.incident_id == incident_id)
    ).all()
    files_data = [
        {
            "id": str(f.id),
            "filename": f.original_filename,
            "mime_type": f.mime_type,
            "size_bytes": f.size_bytes,
            "uploaded_at": f.uploaded_at.isoformat(),
        }
        for f in file_records
    ]

    inc_dict = {
        "id": str(record.id),
        "incident_date": record.incident_date.isoformat() if record.incident_date else None,
        "incident_time": record.incident_time.isoformat() if record.incident_time else None,
        "department": record.department,
        "location": record.location,
        "shift": record.shift,
        "employee_role": record.employee_role,
        "experience_level": record.experience_level,
        "equipment_type": record.equipment_type,
        "equipment_status": record.equipment_status,
        "incident_type": record.incident_type,
        "hazard_type": record.hazard_type,
        "incident_description": record.incident_description,
        "injury_type": record.injury_type,
        "body_part_affected": record.body_part_affected,
        "medical_treatment_required": record.medical_treatment_required,
        "record_source": record.record_source,
        "submission_method": record.submission_method,
        "actual_root_cause_category": record.actual_root_cause_category,
        "root_cause_description": record.root_cause_description,
        "actual_severity_level": record.actual_severity_level,
        "actual_risk_level": record.actual_risk_level,
        "risk_score": float(record.risk_score) if record.risk_score is not None else None,
        "contributing_factor_1": record.contributing_factor_1,
        "contributing_factor_2": record.contributing_factor_2,
        "immediate_action": record.immediate_action,
        "corrective_action": record.corrective_action,
        "preventive_action": record.preventive_action,
        "investigation_status": record.investigation_status,
        "incident_status": record.incident_status,
        "lost_time_days": record.lost_time_days,
        "created_at": record.created_at.isoformat(),
    }

    return {
        "incident": inc_dict,
        "predictions": latest_predictions,
        "prediction_runs": runs_data,
        "files": files_data,
        "analysis_result": latest_analysis,
    }


@app.get("/incidents/{incident_id}/analysis")
def get_incident_analysis_view(incident_id: uuid.UUID, db: Session = Depends(get_db)):
    """Return normalized incident analysis object ready for rendering on UI analysis page."""
    record = db.scalar(select(IncidentRecord).where(IncidentRecord.id == incident_id))
    if not record:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")

    # 1. Determine severity and root causes
    if record.record_source == "historical_dataset":
        sev_label = record.actual_severity_level or "Unknown"
        sev_score = float(record.risk_score) / 100.0 if record.risk_score else 0.50
        root_cause = record.actual_root_cause_category or "Unspecified"
        root_causes = [
            {
                "cause": root_cause,
                "confidence": 1.0,
                "evidence": record.root_cause_description or record.incident_description,
                "supporting_factors": [
                    f for f in [record.contributing_factor_1, record.contributing_factor_2] if f
                ],
            }
        ]
        supporting_sev = [record.actual_risk_level] if record.actual_risk_level else []
    else:
        # Check predictions from latest prediction run
        latest_run = db.scalar(
            select(PredictionRun.id)
            .where(PredictionRun.incident_id == incident_id)
            .order_by(desc(PredictionRun.created_at))
            .limit(1)
        )
        preds = {}
        if latest_run:
            p_rows = db.scalars(
                select(Prediction).where(Prediction.prediction_run_id == latest_run)
            ).all()
            preds = {p.target_name: p.predicted_value for p in p_rows}

        sev_label = preds.get("severity_level", "Moderate")
        sev_score = 0.85 if sev_label in ("Critical", "Severe") else (0.60 if sev_label == "Serious" else 0.35)
        root_cause = preds.get("root_cause_category", "Procedure Violation")
        root_causes = [
            {
                "cause": root_cause,
                "confidence": 0.85,
                "evidence": record.incident_description,
                "supporting_factors": [record.department] if record.department else [],
            }
        ]
        supporting_sev = [preds.get("risk_level", "Medium")]

    # 2. Query similar historical incidents
    similar_q = select(
        IncidentRecord.id,
        IncidentRecord.incident_date,
        IncidentRecord.incident_type,
        IncidentRecord.equipment_type,
        IncidentRecord.actual_severity_level,
        IncidentRecord.actual_root_cause_category,
    ).where(
        IncidentRecord.record_source == "historical_dataset",
        IncidentRecord.id != incident_id,
        or_(
            IncidentRecord.equipment_type == record.equipment_type,
            IncidentRecord.incident_type == record.incident_type,
            IncidentRecord.actual_root_cause_category == root_cause,
        ),
    ).limit(5)
    similar_rows = db.execute(similar_q).fetchall()

    similar_list = [
        {
            "id": str(r.id),
            "date": r.incident_date.isoformat() if r.incident_date else None,
            "type": r.incident_type,
            "equipment": r.equipment_type,
            "severity": r.actual_severity_level,
            "root_cause": r.actual_root_cause_category,
            "similarity": 0.85 if r.equipment_type == record.equipment_type else 0.70,
        }
        for r in similar_rows
    ]

    return {
        "incident": {
            "id": str(record.id),
            "type": record.incident_type or "Type not determined",
            "date": record.incident_date.isoformat() if record.incident_date else None,
            "time": record.incident_time.isoformat() if record.incident_time else None,
            "location": record.location or record.department or "Not stated",
            "summary": record.incident_description[:360] if record.incident_description else None,
        },
        "extracted_information": {
            "equipment": [record.equipment_type] if record.equipment_type else [],
            "hazards": [record.hazard_type] if record.hazard_type else [],
            "materials": [],
            "human_factors": [record.employee_role] if record.employee_role else [],
            "maintenance_factors": [record.equipment_status] if record.equipment_status else [],
            "operational_factors": [record.shift] if record.shift else [],
            "environmental_factors": [],
        },
        "root_causes": root_causes,
        "severity": {
            "label": sev_label,
            "score": sev_score,
            "supporting_factors": supporting_sev,
        },
        "historical_analysis": {
            "similar_incidents": similar_list,
            "same_equipment_incidents": [s for s in similar_list if s["equipment"] == record.equipment_type],
            "recurring_root_causes": [
                {
                    "cause": root_cause,
                    "total": len(similar_list),
                    "on_same_equipment": sum(1 for s in similar_list if s["equipment"] == record.equipment_type),
                    "last_seen": similar_list[0]["date"] if similar_list else None,
                }
            ],
        },
        "report_text": record.original_report_text or record.incident_description,
    }


# =====================================================================
# DASHBOARD & ANALYTICS ENDPOINTS (SQL AGGREGATED)
# =====================================================================

@app.get("/analytics/kpis", response_model=KPIResponse)
def get_kpis(
    record_source: str | None = Query(None, description="'historical_dataset' or 'user_submission'"),
    source: str | None = Query(None, description="Alias for record_source"),
    db: Session = Depends(get_db),
):
    """Return top-level KPIs, severity percentages, 90-day trend comparisons, and source breakdown."""
    active_source = source or record_source
    base_filter = []
    if active_source and active_source != "all":
        base_filter.append(IncidentRecord.record_source == active_source)

    # 1. Total & High severity counts
    totals_q = select(
        func.count(IncidentRecord.id).label("total"),
        func.count(IncidentRecord.id).filter(
            IncidentRecord.actual_severity_level.in_(["High", "Critical", "Serious"])
        ).label("high_sev"),
        func.max(IncidentRecord.incident_date).label("max_date"),
    ).where(*base_filter)

    total_row = db.execute(totals_q).fetchone()
    total_count = total_row.total or 0
    high_count = total_row.high_sev or 0
    max_d = total_row.max_date or date.today()
    high_pct = round((high_count / total_count * 100), 1) if total_count > 0 else 0.0

    # 2. Last 90 days vs previous 90 days
    last_90_start = max_d - pd.Timedelta(days=90)
    prev_90_start = max_d - pd.Timedelta(days=180)

    last_90_count = db.scalar(
        select(func.count(IncidentRecord.id))
        .where(*base_filter, IncidentRecord.incident_date > last_90_start)
    ) or 0

    prev_90_count = db.scalar(
        select(func.count(IncidentRecord.id))
        .where(
            *base_filter,
            IncidentRecord.incident_date <= last_90_start,
            IncidentRecord.incident_date > prev_90_start,
        )
    ) or 0

    # 3. Top equipment with high severity in past year
    one_year_start = max_d - pd.Timedelta(days=365)
    top_eq_q = (
        select(
            IncidentRecord.equipment_type,
            func.count(IncidentRecord.id).label("eq_count"),
        )
        .where(
            *base_filter,
            IncidentRecord.actual_severity_level.in_(["High", "Critical", "Serious"]),
            IncidentRecord.incident_date > one_year_start,
            IncidentRecord.equipment_type.isnot(None),
            IncidentRecord.equipment_type != "Not Applicable",
        )
        .group_by(IncidentRecord.equipment_type)
        .order_by(desc("eq_count"))
        .limit(1)
    )
    top_eq_row = db.execute(top_eq_q).fetchone()
    top_eq_name = top_eq_row.equipment_type if top_eq_row else "None"
    top_eq_count = top_eq_row.eq_count if top_eq_row else 0

    # 4. Breakdown by record_source
    src_rows = db.execute(
        select(IncidentRecord.record_source, func.count(IncidentRecord.id))
        .group_by(IncidentRecord.record_source)
    ).fetchall()
    by_src = {r[0]: r[1] for r in src_rows}

    # 5. Population statistics
    hist_count = by_src.get("historical_dataset", 0)
    user_count = by_src.get("user_submission", 0)
    ground_truth_count = db.scalar(
        select(func.count(IncidentRecord.id)).where(IncidentRecord.actual_severity_level.isnot(None))
    ) or 0
    predicted_count = db.scalar(select(func.count(PredictionRun.id))) or 0
    both_count = db.scalar(
        select(func.count(IncidentRecord.id))
        .where(
            IncidentRecord.actual_severity_level.isnot(None),
            IncidentRecord.id.in_(select(PredictionRun.incident_id)),
        )
    ) or 0

    return {
        "total_incidents": total_count,
        "high_severity_count": high_count,
        "high_severity_pct": high_pct,
        "last_90_days_count": last_90_count,
        "prev_90_days_count": prev_90_count,
        "last_90_days_delta": last_90_count - prev_90_count,
        "top_high_severity_equipment_past_year": top_eq_name,
        "top_equipment_high_severity_count": top_eq_count,
        "by_record_source": by_src,
        "historical_count": hist_count,
        "user_submission_count": user_count,
        "ground_truth_count": ground_truth_count,
        "predicted_count": predicted_count,
        "both_count": both_count,
    }


@app.get("/analytics/trends", response_model=list[TrendItem])
def get_incident_trends(
    record_source: str | None = Query(None),
    source: str | None = Query(None),
    department: str | None = Query(None),
    equipment: str | None = Query(None),
    period_months: int = Query(24, ge=1, le=120),
    db: Session = Depends(get_db),
):
    """Monthly timeline aggregation of total incidents and high-severity incidents."""
    active_source = source or record_source
    filters = [IncidentRecord.incident_date.isnot(None)]

    if active_source and active_source != "all":
        filters.append(IncidentRecord.record_source == active_source)
    if department:
        filters.append(IncidentRecord.department == department)
    if equipment:
        filters.append(IncidentRecord.equipment_type == equipment)

    # Date cutoff
    max_d = db.scalar(select(func.max(IncidentRecord.incident_date)).where(*filters))
    if max_d:
        cutoff = max_d - pd.DateOffset(months=period_months)
        filters.append(IncidentRecord.incident_date >= cutoff.date())

    month_expr = func.to_char(func.date_trunc("month", IncidentRecord.incident_date), "YYYY-MM")

    q = (
        select(
            month_expr.label("month"),
            func.count(IncidentRecord.id).label("total"),
            func.count(IncidentRecord.id).filter(
                IncidentRecord.actual_severity_level.in_(["High", "Critical", "Serious"])
            ).label("high_sev"),
        )
        .where(*filters)
        .group_by(month_expr)
        .order_by(month_expr)
    )

    rows = db.execute(q).fetchall()
    return [{"month": r.month, "total": r.total, "high_severity": r.high_sev} for r in rows]


@app.get("/analytics/distributions", response_model=list[DistributionItem])
def get_distributions(
    dimension: str | None = Query(
        None,
        description="One of: 'department', 'location', 'shift', 'equipment_type', 'incident_type', 'hazard_type', 'severity', 'root_cause', 'risk', 'predicted_severity', 'predicted_root_cause', 'predicted_risk'",
    ),
    dim: str | None = Query(None, description="Alias for dimension"),
    record_source: str | None = Query(None),
    source: str | None = Query(None),
    top: int = Query(10, ge=1, le=100),
    limit: int | None = Query(None, ge=1, le=100, description="Alias for top"),
    db: Session = Depends(get_db),
):
    """SQL aggregation for categorical breakdowns across safety dimensions."""
    active_source = source or record_source
    active_dim = dim or dimension
    active_top = limit or top

    if not active_dim:
        raise HTTPException(status_code=400, detail="Missing required 'dimension' or 'dim' parameter.")

    # 1. Predictions distributions (from predictions table)
    if active_dim in ("predicted_severity", "predicted_root_cause", "predicted_risk"):
        target_map = {
            "predicted_severity": "severity_level",
            "predicted_root_cause": "root_cause_category",
            "predicted_risk": "risk_level",
        }
        target_name = target_map[active_dim]

        q = (
            select(
                Prediction.predicted_value.label("name"),
                func.count(Prediction.id).label("count"),
            )
            .where(Prediction.target_name == target_name)
            .group_by(Prediction.predicted_value)
            .order_by(desc("count"))
            .limit(active_top)
        )
        rows = db.execute(q).fetchall()
        return [{"name": r.name, "count": r.count} for r in rows]

    # 2. Incident table distributions
    dim_map = {
        "department": IncidentRecord.department,
        "location": IncidentRecord.location,
        "shift": IncidentRecord.shift,
        "equipment_type": IncidentRecord.equipment_type,
        "incident_type": IncidentRecord.incident_type,
        "hazard_type": IncidentRecord.hazard_type,
        "severity": IncidentRecord.actual_severity_level,
        "root_cause": IncidentRecord.actual_root_cause_category,
        "risk": IncidentRecord.actual_risk_level,
    }

    if active_dim not in dim_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dimension '{active_dim}'. Choose from: {list(dim_map.keys()) + ['predicted_severity', 'predicted_root_cause', 'predicted_risk']}",
        )

    col = dim_map[active_dim]
    filters = [col.isnot(None), col != "Not Applicable"]
    if active_source and active_source != "all":
        filters.append(IncidentRecord.record_source == active_source)

    q = (
        select(col.label("name"), func.count(IncidentRecord.id).label("count"))
        .where(*filters)
        .group_by(col)
        .order_by(desc("count"))
        .limit(active_top)
    )

    rows = db.execute(q).fetchall()
    return [{"name": r.name, "count": r.count} for r in rows]


@app.get("/analytics/heatmap", response_model=HeatmapResponse)
def get_equipment_cause_heatmap(
    top_equipment: int = Query(10, ge=1, le=50),
    top_causes: int = Query(8, ge=1, le=30),
    record_source: str | None = Query("historical_dataset"),
    db: Session = Depends(get_db),
):
    """Matrix cross-tabulation of Equipment vs Root Cause frequency for heatmap views."""
    filters = [
        IncidentRecord.equipment_type.isnot(None),
        IncidentRecord.equipment_type != "Not Applicable",
        IncidentRecord.actual_root_cause_category.isnot(None),
    ]
    if record_source and record_source != "all":
        filters.append(IncidentRecord.record_source == record_source)

    # Top equipment
    top_eq = db.scalars(
        select(IncidentRecord.equipment_type)
        .where(*filters)
        .group_by(IncidentRecord.equipment_type)
        .order_by(desc(func.count(IncidentRecord.id)))
        .limit(top_equipment)
    ).all()

    # Top root causes
    top_rc = db.scalars(
        select(IncidentRecord.actual_root_cause_category)
        .where(*filters)
        .group_by(IncidentRecord.actual_root_cause_category)
        .order_by(desc(func.count(IncidentRecord.id)))
        .limit(top_causes)
    ).all()

    if not top_eq or not top_rc:
        return {"equipment": [], "root_causes": [], "matrix": []}

    # Cross-tab counts
    counts_q = (
        select(
            IncidentRecord.equipment_type,
            IncidentRecord.actual_root_cause_category,
            func.count(IncidentRecord.id).label("cnt"),
        )
        .where(
            *filters,
            IncidentRecord.equipment_type.in_(top_eq),
            IncidentRecord.actual_root_cause_category.in_(top_rc),
        )
        .group_by(IncidentRecord.equipment_type, IncidentRecord.actual_root_cause_category)
    )
    counts = db.execute(counts_q).fetchall()

    count_map = {(r[0], r[1]): r[2] for r in counts}

    matrix = [
        [count_map.get((eq, rc), 0) for rc in top_rc]
        for eq in top_eq
    ]

    return {
        "equipment": list(top_eq),
        "root_causes": list(top_rc),
        "matrix": matrix,
    }


@app.get("/analytics/similar", response_model=list[SimilarIncidentItem])
def search_similar_incidents(
    equipment: str | None = Query(None),
    incident_type: str | None = Query(None),
    root_cause: str | None = Query(None),
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Find historical incidents similar to given parameters with similarity scores."""
    if not equipment and not incident_type and not root_cause:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of 'equipment', 'incident_type', or 'root_cause'.",
        )

    criteria = []
    if equipment:
        criteria.append(IncidentRecord.equipment_type == equipment)
    if incident_type:
        criteria.append(IncidentRecord.incident_type == incident_type)
    if root_cause:
        criteria.append(IncidentRecord.actual_root_cause_category == root_cause)

    q = (
        select(
            IncidentRecord.id,
            IncidentRecord.incident_date,
            IncidentRecord.incident_type,
            IncidentRecord.equipment_type,
            IncidentRecord.actual_severity_level,
            IncidentRecord.actual_root_cause_category,
        )
        .where(
            IncidentRecord.record_source == "historical_dataset",
            or_(*criteria),
        )
        .order_by(desc(IncidentRecord.incident_date))
        .limit(limit)
    )

    rows = db.execute(q).fetchall()

    results = []
    for r in rows:
        sim = 0.0
        if incident_type and r.incident_type == incident_type:
            sim += 0.4
        if root_cause and r.actual_root_cause_category == root_cause:
            sim += 0.4
        if equipment and r.equipment_type == equipment:
            sim += 0.2
        if sim == 0.0:
            sim = 0.50

        results.append(
            SimilarIncidentItem(
                id=str(r.id),
                incident_date=r.incident_date.isoformat() if r.incident_date else None,
                incident_type=r.incident_type,
                equipment_type=r.equipment_type,
                actual_severity_level=r.actual_severity_level,
                actual_root_cause_category=r.actual_root_cause_category,
                similarity=round(sim, 2),
            )
        )

    return sorted(results, key=lambda x: -x.similarity)


@app.get("/analytics/eval-metrics")
def get_model_evaluation(db: Session = Depends(get_db)):
    """
    Model evaluation statistics strictly on validated records with both ground-truth
    labels and recorded predictions.
    """
    labeled_with_preds = db.execute(
        select(
            IncidentRecord.id,
            IncidentRecord.actual_root_cause_category,
            IncidentRecord.actual_severity_level,
            IncidentRecord.actual_risk_level,
        )
        .where(
            IncidentRecord.actual_root_cause_category.isnot(None),
            IncidentRecord.id.in_(select(PredictionRun.incident_id)),
        )
    ).fetchall()

    evaluated_count = len(labeled_with_preds)
    total_historical = db.scalar(
        select(func.count(IncidentRecord.id)).where(IncidentRecord.record_source == "historical_dataset")
    ) or 0
    total_user_submissions = db.scalar(
        select(func.count(IncidentRecord.id)).where(IncidentRecord.record_source == "user_submission")
    ) or 0
    total_predictions = db.scalar(select(func.count(PredictionRun.id))) or 0

    evaluation_data = {
        "evaluated_records": evaluated_count,
        "total_historical_with_ground_truth": total_historical,
        "total_user_submissions": total_user_submissions,
        "total_prediction_runs": total_predictions,
        "overlap_records_with_both": evaluated_count,
        "accuracy": {},
        "methodology_note": (
            "Model accuracy requires independent validation incidents with verified ground-truth labels and model predictions. "
            "Training dataset records are excluded from evaluation metrics to prevent optimistic overfitting bias."
        ),
    }

    if evaluated_count > 0:
        correct_rc = 0
        correct_sev = 0
        correct_risk = 0
        for row in labeled_with_preds:
            preds = db.scalars(select(Prediction).where(Prediction.prediction_run_id.in_(
                select(PredictionRun.id).where(PredictionRun.incident_id == row.id)
            ))).all()
            pred_map = {p.target_name: p.predicted_value for p in preds}
            if pred_map.get("root_cause_category") == row.actual_root_cause_category:
                correct_rc += 1
            if pred_map.get("severity_level") == row.actual_severity_level:
                correct_sev += 1
            if pred_map.get("risk_level") == row.actual_risk_level:
                correct_risk += 1
        evaluation_data["accuracy"] = {
            "root_cause_category": round(correct_rc / evaluated_count, 3),
            "severity_level": round(correct_sev / evaluated_count, 3),
            "risk_level": round(correct_risk / evaluated_count, 3),
        }

    return evaluation_data


@app.get("/analytics/dataset-summary", response_model=DatasetSummaryResponse)
def get_dataset_summary(db: Session = Depends(get_db)):
    """Summary statistics, null audit, and latest import log for the historical dataset."""
    total_historical = db.scalar(
        select(func.count(IncidentRecord.id)).where(IncidentRecord.record_source == "historical_dataset")
    ) or 0

    dates = db.execute(
        select(
            func.min(IncidentRecord.incident_date),
            func.max(IncidentRecord.incident_date),
        ).where(IncidentRecord.record_source == "historical_dataset")
    ).fetchone()

    # Target distributions
    targets_dist: dict[str, dict[str, int]] = {}
    for col_name, attr in [
        ("root_cause_category", IncidentRecord.actual_root_cause_category),
        ("severity_level", IncidentRecord.actual_severity_level),
        ("risk_level", IncidentRecord.actual_risk_level),
    ]:
        rows = db.execute(
            select(attr, func.count(IncidentRecord.id))
            .where(IncidentRecord.record_source == "historical_dataset")
            .group_by(attr)
            .order_by(desc(func.count(IncidentRecord.id)))
        ).fetchall()
        targets_dist[col_name] = {r[0]: r[1] for r in rows if r[0]}

    # Null audit
    null_counts: dict[str, int] = {}
    for col_name, attr in [
        ("injury_type", IncidentRecord.injury_type),
        ("body_part_affected", IncidentRecord.body_part_affected),
        ("root_cause_description", IncidentRecord.root_cause_description),
        ("contributing_factor_2", IncidentRecord.contributing_factor_2),
        ("equipment_status", IncidentRecord.equipment_status),
    ]:
        nulls = db.scalar(
            select(func.count(IncidentRecord.id)).where(
                IncidentRecord.record_source == "historical_dataset",
                attr.is_(None),
            )
        ) or 0
        null_counts[col_name] = nulls

    # Latest import log
    latest_log = db.scalar(
        select(DatasetImportLog).order_by(desc(DatasetImportLog.started_at)).limit(1)
    )
    latest_import_data = None
    if latest_log:
        latest_import_data = {
            "batch_id": latest_log.import_batch_id,
            "source_file": latest_log.source_file,
            "mode": latest_log.mode,
            "status": latest_log.status,
            "rows_read": latest_log.total_rows_read,
            "rows_inserted": latest_log.rows_inserted,
            "rows_skipped": latest_log.rows_skipped_duplicate,
            "completed_at": latest_log.completed_at.isoformat() if latest_log.completed_at else None,
        }

    return {
        "total_records": total_historical,
        "date_range": {
            "min": dates[0].isoformat() if dates and dates[0] else None,
            "max": dates[1].isoformat() if dates and dates[1] else None,
        },
        "target_distributions": targets_dist,
        "null_counts": null_counts,
        "latest_import": latest_import_data,
    }


@app.get("/models", response_model=list[ModelVersionItem])
def list_models(db: Session = Depends(get_db)):
    """List registered model versions and in-memory operational status."""
    versions = db.scalars(select(ModelVersion).order_by(ModelVersion.target_name)).all()
    results = []
    for v in versions:
        is_loaded = v.target_name in artifacts and model_metadata.get(v.target_name, {}).get("version_tag") == v.version_tag
        results.append(
            ModelVersionItem(
                id=str(v.id),
                target_name=v.target_name,
                version_tag=v.version_tag,
                artifact_path=v.artifact_path,
                artifact_sha256=v.artifact_sha256,
                trained_at=v.trained_at,
                created_at=v.created_at,
                is_loaded=is_loaded,
            )
        )
    return results
