
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    text,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Time,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=new_uuid
    )
    full_name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True
    )
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(
        String(30), default="user"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'safety_analyst', 'admin')",
            name="ck_users_role",
        ),
    )


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=new_uuid
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    record_source: Mapped[str] = mapped_column(
        String(30), default="user_submission", index=True
    )
    submission_method: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )

    original_report_text: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    submission_payload: Mapped[dict[str, Any] | None] = (
        mapped_column(JSONB, nullable=True)
    )

    incident_date: Mapped[date | None] = mapped_column(
        Date, nullable=True
    )
    incident_time: Mapped[time | None] = mapped_column(
        Time, nullable=True
    )

    incident_year: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    incident_month: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    incident_dayofweek: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    incident_hour: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    incident_minute: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    department: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    location: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    shift: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )
    employee_role: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    experience_level: Mapped[str | None] = mapped_column(
        String(80), nullable=True
    )
    equipment_type: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    equipment_status: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    incident_type: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    hazard_type: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    incident_description: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    reported_by_role: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )

    # Historical ground-truth labels; never overwrite with predictions.
    actual_root_cause_category: Mapped[str | None] = (
        mapped_column(String(120), nullable=True)
    )
    root_cause_description: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    actual_severity_level: Mapped[str | None] = (
        mapped_column(String(80), nullable=True)
    )
    actual_risk_level: Mapped[str | None] = (
        mapped_column(String(80), nullable=True)
    )

    # Ingestion provenance and idempotency tracking
    source_row_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )

    risk_score: Mapped[Decimal | None] = mapped_column(
        Numeric, nullable=True
    )
    injury_type: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    body_part_affected: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    medical_treatment_required: Mapped[bool | None] = (
        mapped_column(Boolean, nullable=True)
    )
    contributing_factor_1: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    contributing_factor_2: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    immediate_action: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    corrective_action: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    preventive_action: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    investigation_status: Mapped[str | None] = mapped_column(
        String(80), nullable=True
    )
    incident_status: Mapped[str | None] = mapped_column(
        String(80), nullable=True
    )
    lost_time_days: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "record_source IN ('user_submission', 'historical_dataset')",
            name="ck_incidents_record_source",
        ),
        CheckConstraint(
            "submission_method IS NULL OR submission_method IN "
            "('paste', 'upload', 'form', 'dataset_import')",
            name="ck_incidents_submission_method",
        ),
        Index("ix_incidents_created_at", "created_at"),
        Index("ix_incidents_incident_date", "incident_date"),
        Index("ix_incidents_department", "department"),
        Index("ix_incidents_actual_severity", "actual_severity_level"),
        Index("ix_incidents_actual_risk", "actual_risk_level"),
        Index(
            "ix_incidents_actual_root_cause",
            "actual_root_cause_category",
        ),
        Index("ix_incidents_equipment_type", "equipment_type"),
        Index("ix_incidents_incident_type", "incident_type"),
        Index("ix_incidents_source_date", "record_source", "incident_date"),
        Index(
            "uq_incidents_historical_row_hash",
            "source_row_hash",
            unique=True,
            postgresql_where=text("record_source = 'historical_dataset'"),
        ),
    )


class IncidentFile(Base):
    __tablename__ = "incident_files"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=new_uuid
    )
    incident_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"),
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(
        String(255)
    )
    storage_key: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256_hash: Mapped[str] = mapped_column(String(64))
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=new_uuid
    )
    target_name: Mapped[str] = mapped_column(String(50))
    version_tag: Mapped[str] = mapped_column(String(100))
    artifact_path: Mapped[str] = mapped_column(Text)
    artifact_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    trained_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    metadata_json: Mapped[dict[str, Any] | None] = (
        mapped_column(JSONB, nullable=True)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "target_name", "version_tag",
            name="uq_model_versions_target_version",
        ),
        CheckConstraint(
            "target_name IN "
            "('root_cause_category', 'severity_level', 'risk_level')",
            name="ck_model_versions_target",
        ),
    )


class PredictionRun(Base):
    __tablename__ = "prediction_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=new_uuid
    )
    incident_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"),
        index=True,
    )
    requested_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(30), default="success"
    )
    duration_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('success', 'failed')",
            name="ck_prediction_runs_status",
        ),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=new_uuid
    )
    prediction_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prediction_runs.id", ondelete="CASCADE"),
        index=True,
    )
    target_name: Mapped[str] = mapped_column(String(50))
    predicted_value: Mapped[str] = mapped_column(String(150))
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT")
    )
    confidence_score: Mapped[Decimal | None] = mapped_column(
        Numeric, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "prediction_run_id", "target_name",
            name="uq_predictions_run_target",
        ),
        CheckConstraint(
            "target_name IN "
            "('root_cause_category', 'severity_level', 'risk_level')",
            name="ck_predictions_target",
        ),
    )


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=new_uuid
    )
    prediction_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prediction_runs.id", ondelete="CASCADE"),
        unique=True,
    )
    extracted_entities: Mapped[dict[str, Any] | None] = (
        mapped_column(JSONB, nullable=True)
    )
    contributing_factors: Mapped[list[Any] | None] = (
        mapped_column(JSONB, nullable=True)
    )
    similar_incidents: Mapped[list[Any] | None] = (
        mapped_column(JSONB, nullable=True)
    )
    summary: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DatasetImportLog(Base):
    __tablename__ = "dataset_import_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=new_uuid
    )
    import_batch_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True
    )
    source_file: Mapped[str] = mapped_column(String(255))
    file_sha256: Mapped[str] = mapped_column(String(64))
    file_size_bytes: Mapped[int] = mapped_column(BigInteger)
    mode: Mapped[str] = mapped_column(
        String(30), default="full_import"
    )
    status: Mapped[str] = mapped_column(
        String(30), default="started"
    )
    total_rows_read: Mapped[int] = mapped_column(Integer, default=0)
    rows_accepted: Mapped[int] = mapped_column(Integer, default=0)
    rows_inserted: Mapped[int] = mapped_column(Integer, default=0)
    rows_skipped_duplicate: Mapped[int] = mapped_column(
        Integer, default=0
    )
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0)
    validation_summary: Mapped[dict[str, Any] | None] = (
        mapped_column(JSONB, nullable=True)
    )
    error_details: Mapped[list[Any] | None] = (
        mapped_column(JSONB, nullable=True)
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "mode IN ('dry_run', 'sample', 'full_import')",
            name="ck_dataset_import_logs_mode",
        ),
        CheckConstraint(
            "status IN ('started', 'completed', 'failed', 'completed_with_warnings')",
            name="ck_dataset_import_logs_status",
        ),
    )
