#!/usr/bin/env python3
"""
import_historical_dataset.py

Auditable, chunked, idempotent ingestion of the historical training dataset
into PostgreSQL for the Safety Incident Prediction project.

Usage:
    # Dry-run validation (non-destructive; does not commit incident rows)
    python import_historical_dataset.py --dry-run

    # Smoke-test sample of 10 rows
    python import_historical_dataset.py --sample 10 --yes

    # Full production import with confirmation prompt
    python import_historical_dataset.py --execute

Author: Antigravity Automated Engineering
Date: 2026-10-29
"""

import argparse
import csv
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import sys
import time as time_module
from typing import Any
import uuid

# Avoid cython C-extension block under Windows AppLocker if applicable
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select, text
from database import SessionLocal, engine
from models import Incident, DatasetImportLog

# Deterministic UUIDv5 namespace for historical records
HISTORICAL_NAMESPACE = uuid.UUID("a6e2e580-0a2a-4dfb-bb6e-8ff462e76f57")

EXPECTED_COLUMNS = [
    "incident_date",
    "incident_time",
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
    "injury_type",
    "body_part_affected",
    "medical_treatment_required",
    "root_cause_category",
    "root_cause_description",
    "contributing_factor_1",
    "contributing_factor_2",
    "severity_level",
    "risk_level",
    "risk_score",
    "immediate_action",
    "corrective_action",
    "preventive_action",
    "reported_by_role",
    "investigation_status",
    "incident_status",
    "lost_time_days",
]


def compute_file_sha256(filepath: Path) -> tuple[str, int]:
    """Compute SHA-256 hash and byte size of a file in streaming chunks."""
    h = hashlib.sha256()
    size = 0
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def parse_date(date_str: str) -> date | None:
    """Safely parse YYYY-MM-DD or standard ISO date string."""
    if not date_str or not date_str.strip():
        return None
    s = date_str.strip()
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        try:
            return datetime.fromisoformat(s).date()
        except ValueError:
            return None


def parse_time(time_str: str) -> time | None:
    """
    Parse incident_time string.
    Supports '1900-01-01 HH:MM:SS', 'HH:MM:SS', or 'HH:MM'.
    """
    if not time_str or not time_str.strip():
        return None
    s = time_str.strip()
    if " " in s:
        # e.g., '1900-01-01 00:09:00'
        parts = s.split(" ", 1)
        s = parts[1]
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            pass
    return None


def parse_bool(val: Any) -> bool | None:
    """Parse boolean values like 'Yes', 'No', 'True', 'False'."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("yes", "true", "t", "1"):
        return True
    if s in ("no", "false", "f", "0"):
        return False
    return None


def clean_str(val: Any) -> str | None:
    """Strip whitespace and convert empty string or 'nan' to None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def compute_row_hash(row: dict[str, Any]) -> str:
    """
    Generate a canonical SHA-256 hash of key identifying fields for idempotency.
    Never relies on CSV line numbering.
    """
    key_components = [
        str(row.get("incident_date") or "").strip(),
        str(row.get("incident_time") or "").strip(),
        str(row.get("department") or "").strip(),
        str(row.get("location") or "").strip(),
        str(row.get("incident_description") or "").strip(),
        str(row.get("root_cause_category") or "").strip(),
        str(row.get("severity_level") or "").strip(),
        str(row.get("risk_level") or "").strip(),
    ]
    raw = "|".join(key_components)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class HistoricalDatasetImporter:
    def __init__(
        self,
        csv_path: Path,
        mode: str = "dry_run",
        batch_size: int = 500,
        sample_limit: int | None = None,
    ):
        self.csv_path = csv_path
        self.mode = mode
        self.batch_size = batch_size
        self.sample_limit = sample_limit
        self.batch_id = f"import-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        self.total_rows_read = 0
        self.rows_accepted = 0
        self.rows_inserted = 0
        self.rows_skipped_duplicate = 0
        self.rows_rejected = 0

        self.target_distributions: dict[str, dict[str, int]] = {
            "root_cause_category": {},
            "severity_level": {},
            "risk_level": {},
        }
        self.null_counts: dict[str, int] = {}
        self.errors: list[dict[str, Any]] = []
        self.date_min: date | None = None
        self.date_max: date | None = None

    def validate_row(self, row: dict[str, str], row_idx: int) -> tuple[dict[str, Any] | None, str | None]:
        """
        Validate and transform a single CSV row into Incident attributes.
        Returns (transformed_dict, None) if valid, or (None, rejection_reason).
        """
        # 1. Essential required fields check
        desc = clean_str(row.get("incident_description"))
        if not desc:
            return None, "MISSING_INCIDENT_DESCRIPTION"

        raw_date = row.get("incident_date")
        inc_date = parse_date(raw_date) if raw_date else None
        if not inc_date:
            return None, f"INVALID_INCIDENT_DATE: {raw_date}"

        raw_time = row.get("incident_time")
        inc_time = parse_time(raw_time) if raw_time else None
        if not inc_time:
            return None, f"INVALID_INCIDENT_TIME: {raw_time}"

        # 2. Ground-truth target labels
        root_cause = clean_str(row.get("root_cause_category"))
        if not root_cause:
            return None, "MISSING_ROOT_CAUSE_CATEGORY"

        severity = clean_str(row.get("severity_level"))
        if not severity:
            return None, "MISSING_SEVERITY_LEVEL"

        risk = clean_str(row.get("risk_level"))
        if not risk:
            return None, "MISSING_RISK_LEVEL"

        # 3. Numeric fields
        raw_score = clean_str(row.get("risk_score"))
        risk_score = None
        if raw_score:
            try:
                risk_score = Decimal(raw_score)
            except InvalidOperation:
                return None, f"INVALID_RISK_SCORE: {raw_score}"

        raw_ltd = clean_str(row.get("lost_time_days"))
        lost_time_days = None
        if raw_ltd:
            try:
                lost_time_days = int(float(raw_ltd))
            except (ValueError, TypeError):
                return None, f"INVALID_LOST_TIME_DAYS: {raw_ltd}"

        # 4. Canonical row hash and deterministic UUID
        row_hash = compute_row_hash(row)
        deterministic_id = uuid.uuid5(HISTORICAL_NAMESPACE, row_hash)

        # 5. Derived temporal features
        incident_year = inc_date.year
        incident_month = inc_date.month
        incident_dayofweek = inc_date.weekday()
        incident_hour = inc_time.hour
        incident_minute = inc_time.minute

        # 6. Build the complete dictionary for the Incident model
        transformed: dict[str, Any] = {
            "id": deterministic_id,
            "record_source": "historical_dataset",
            "submission_method": "dataset_import",
            "original_report_text": desc,
            "submission_payload": {
                "source_file": self.csv_path.name,
                "source_row_index": row_idx,
                "import_batch_id": self.batch_id,
            },
            "source_row_hash": row_hash,
            "incident_date": inc_date,
            "incident_time": inc_time,
            "incident_year": incident_year,
            "incident_month": incident_month,
            "incident_dayofweek": incident_dayofweek,
            "incident_hour": incident_hour,
            "incident_minute": incident_minute,
            "department": clean_str(row.get("department")),
            "location": clean_str(row.get("location")),
            "shift": clean_str(row.get("shift")),
            "employee_role": clean_str(row.get("employee_role")),
            "experience_level": clean_str(row.get("experience_level")),
            "equipment_type": clean_str(row.get("equipment_type")),
            "equipment_status": clean_str(row.get("equipment_status")),
            "incident_type": clean_str(row.get("incident_type")),
            "hazard_type": clean_str(row.get("hazard_type")),
            "incident_description": desc,
            "reported_by_role": clean_str(row.get("reported_by_role")),
            "actual_root_cause_category": root_cause,
            "root_cause_description": clean_str(row.get("root_cause_description")),
            "actual_severity_level": severity,
            "actual_risk_level": risk,
            "risk_score": risk_score,
            "injury_type": clean_str(row.get("injury_type")),
            "body_part_affected": clean_str(row.get("body_part_affected")),
            "medical_treatment_required": parse_bool(row.get("medical_treatment_required")),
            "contributing_factor_1": clean_str(row.get("contributing_factor_1")),
            "contributing_factor_2": clean_str(row.get("contributing_factor_2")),
            "immediate_action": clean_str(row.get("immediate_action")),
            "corrective_action": clean_str(row.get("corrective_action")),
            "preventive_action": clean_str(row.get("preventive_action")),
            "investigation_status": clean_str(row.get("investigation_status")),
            "incident_status": clean_str(row.get("incident_status")),
            "lost_time_days": lost_time_days,
        }

        # Track distributions and metadata
        self.target_distributions["root_cause_category"][root_cause] = (
            self.target_distributions["root_cause_category"].get(root_cause, 0) + 1
        )
        self.target_distributions["severity_level"][severity] = (
            self.target_distributions["severity_level"].get(severity, 0) + 1
        )
        self.target_distributions["risk_level"][risk] = (
            self.target_distributions["risk_level"].get(risk, 0) + 1
        )

        for col, val in transformed.items():
            if val is None:
                self.null_counts[col] = self.null_counts.get(col, 0) + 1

        if self.date_min is None or inc_date < self.date_min:
            self.date_min = inc_date
        if self.date_max is None or inc_date > self.date_max:
            self.date_max = inc_date

        return transformed, None

    def run(self) -> dict[str, Any]:
        """Execute the ingestion pipeline with chunking and idempotency."""
        start_time = time_module.time()

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {self.csv_path}")

        file_sha256, file_size = compute_file_sha256(self.csv_path)

        print(f"\n=======================================================")
        print(f" Safety Incident Dataset Ingestion Pipeline")
        print(f"=======================================================")
        print(f" Batch ID     : {self.batch_id}")
        print(f" File         : {self.csv_path.name}")
        print(f" File Size    : {file_size / (1024 * 1024):.2f} MB ({file_size:,} bytes)")
        print(f" File SHA-256 : {file_sha256}")
        print(f" Execution Mode: {self.mode.upper()}")
        print(f" Batch Size   : {self.batch_size}")
        if self.sample_limit:
            print(f" Sample Limit : {self.sample_limit} rows")
        print(f"=======================================================\n")

        # 1. Initialize audit log in database
        import_log_id = uuid.uuid4()
        with SessionLocal() as db:
            log_record = DatasetImportLog(
                id=import_log_id,
                import_batch_id=self.batch_id,
                source_file=self.csv_path.name,
                file_sha256=file_sha256,
                file_size_bytes=file_size,
                mode=self.mode,
                status="started",
                total_rows_read=0,
                rows_accepted=0,
                rows_inserted=0,
                rows_skipped_duplicate=0,
                rows_rejected=0,
                validation_summary=None,
                error_details=None,
            )
            db.add(log_record)
            db.commit()

        # 2. Process CSV file line by line
        current_batch: list[dict[str, Any]] = []

        try:
            with self.csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []

                missing_cols = set(EXPECTED_COLUMNS) - set(header)
                if missing_cols:
                    raise ValueError(f"CSV header is missing expected columns: {sorted(missing_cols)}")

                for row_idx, row in enumerate(reader, start=1):
                    if self.sample_limit and self.total_rows_read >= self.sample_limit:
                        break

                    self.total_rows_read += 1
                    transformed, rejection_reason = self.validate_row(row, row_idx)

                    if rejection_reason:
                        self.rows_rejected += 1
                        # Record sanitized error (no raw report description logged)
                        self.errors.append({
                            "row_index": row_idx,
                            "reason": rejection_reason,
                        })
                        continue

                    self.rows_accepted += 1
                    current_batch.append(transformed)

                    if len(current_batch) >= self.batch_size:
                        self._process_batch(current_batch)
                        current_batch = []

                if current_batch:
                    self._process_batch(current_batch)
                    current_batch = []

            status = "completed" if not self.errors else "completed_with_warnings"

        except Exception as exc:
            status = "failed"
            self.errors.append({"general_error": str(exc)})
            print(f"\n[FATAL ERROR] Ingestion aborted: {exc}")
            raise

        finally:
            elapsed = round(time_module.time() - start_time, 2)
            summary = {
                "file_sha256": file_sha256,
                "file_size_bytes": file_size,
                "date_min": self.date_min.isoformat() if self.date_min else None,
                "date_max": self.date_max.isoformat() if self.date_max else None,
                "elapsed_seconds": elapsed,
                "target_distributions": self.target_distributions,
                "null_counts": self.null_counts,
            }

            with SessionLocal() as db:
                log_to_update = db.scalar(
                    select(DatasetImportLog).where(DatasetImportLog.id == import_log_id)
                )
                if log_to_update:
                    log_to_update.status = status
                    log_to_update.total_rows_read = self.total_rows_read
                    log_to_update.rows_accepted = self.rows_accepted
                    log_to_update.rows_inserted = self.rows_inserted
                    log_to_update.rows_skipped_duplicate = self.rows_skipped_duplicate
                    log_to_update.rows_rejected = self.rows_rejected
                    log_to_update.validation_summary = summary
                    log_to_update.error_details = self.errors[:100]  # Store up to 100 error details
                    log_to_update.completed_at = datetime.now(timezone.utc)
                    db.commit()

        self._print_summary(elapsed, status)
        return {
            "batch_id": self.batch_id,
            "status": status,
            "total_rows_read": self.total_rows_read,
            "rows_accepted": self.rows_accepted,
            "rows_inserted": self.rows_inserted,
            "rows_skipped_duplicate": self.rows_skipped_duplicate,
            "rows_rejected": self.rows_rejected,
            "summary": summary,
        }

    def _process_batch(self, batch: list[dict[str, Any]]) -> None:
        """
        Check existing records in the database by source_row_hash and insert new ones.
        If self.mode == 'dry_run', checks duplicates but does not commit inserts.
        """
        if not batch:
            return

        batch_hashes = [r["source_row_hash"] for r in batch]

        with SessionLocal() as db:
            # Query existing historical records matching any of these hashes
            existing_hashes = set(
                db.scalars(
                    select(Incident.source_row_hash).where(
                        Incident.record_source == "historical_dataset",
                        Incident.source_row_hash.in_(batch_hashes),
                    )
                ).all()
            )

            to_insert = [r for r in batch if r["source_row_hash"] not in existing_hashes]
            duplicates_count = len(batch) - len(to_insert)
            self.rows_skipped_duplicate += duplicates_count

            if self.mode != "dry_run" and to_insert:
                incident_objects = [Incident(**data) for data in to_insert]
                db.add_all(incident_objects)
                db.commit()
                self.rows_inserted += len(to_insert)

            print(
                f"  [Chunk {self.total_rows_read:,} rows processed] "
                f"New: {len(to_insert)}, Skipped duplicates: {duplicates_count}"
            )

    def _print_summary(self, elapsed: float, status: str) -> None:
        print(f"\n=======================================================")
        print(f" Ingestion Summary ({status.upper()})")
        print(f"=======================================================")
        print(f" Total Rows Read       : {self.total_rows_read:,}")
        print(f" Rows Accepted         : {self.rows_accepted:,}")
        print(f" Rows Inserted         : {self.rows_inserted:,} (Mode: {self.mode})")
        print(f" Rows Skipped (Dupes)  : {self.rows_skipped_duplicate:,}")
        print(f" Rows Rejected         : {self.rows_rejected:,}")
        print(f" Date Range            : {self.date_min} to {self.date_max}")
        print(f" Elapsed Time          : {elapsed} seconds")
        print(f"=======================================================")
        print("\nTarget Label Distributions in Ingestion:")
        for target, dist in self.target_distributions.items():
            print(f"\n  --- {target} ({len(dist)} classes) ---")
            for label, count in sorted(dist.items(), key=lambda x: -x[1]):
                print(f"    {label:30} : {count:5,} ({count/max(1, self.rows_accepted)*100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest historical safety incident dataset into PostgreSQL."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=PROJECT_ROOT / "safety_incidents_cleaned.csv",
        help="Path to the cleaned dataset CSV (default: safety_incidents_cleaned.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all rows and check DB duplicates without inserting into incidents.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only first N rows for testing.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform actual database insertion.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of records per transaction chunk (default: 500).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation when --execute is specified.",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.sample and not args.execute:
        print("Error: Specify either --dry-run (safe test), --sample N, or --execute.")
        sys.exit(1)

    mode = "dry_run"
    if args.execute:
        mode = "full_import"
    elif args.sample:
        mode = "sample"

    if mode in ("full_import", "sample") and not args.yes:
        confirm = input(
            f"Are you sure you want to run {mode.upper()} against the database? [y/N]: "
        )
        if confirm.strip().lower() not in ("y", "yes"):
            print("Import cancelled by user.")
            sys.exit(0)

    importer = HistoricalDatasetImporter(
        csv_path=args.csv_path,
        mode=mode,
        batch_size=args.chunk_size,
        sample_limit=args.sample,
    )
    importer.run()


if __name__ == "__main__":
    main()

