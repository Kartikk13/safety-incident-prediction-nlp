"""
file_storage.py

Modular private document storage architecture for Safety Incident Prediction.
Supports:
- LocalStorageBackend (default for development and single-node deployment)
- S3StorageBackend (for AWS S3, Cloudflare R2, MinIO, or GCP Cloud Storage)
- UUID-based secure filename generation
- Magic-byte validation (%PDF-, PK) and extension/MIME verification
- Safe orphaned file cleanup with a minimum 24-hour grace period
"""

import abc
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import re
from typing import Any
import uuid

from fastapi import HTTPException, UploadFile
from sqlalchemy import select

from database import SessionLocal
from models import IncidentFile

PROJECT_ROOT = Path(__file__).resolve().parent

# Default local upload directory (can be overridden by environment variable)
UPLOAD_DIR = Path(
    os.getenv("SAFETY_UPLOAD_DIR", str(PROJECT_ROOT / "private_uploads"))
).resolve()

MAX_FILE_MB = 10
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
ALLOWED_MIME_TYPES = {
    ".txt": {"text/plain"},
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    },
}

GENERATED_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}\.(txt|pdf|docx)$")


class BaseStorageBackend(abc.ABC):
    """Abstract base class for modular incident file storage."""

    @abc.abstractmethod
    async def save_file(self, upload: UploadFile) -> dict[str, Any]:
        """Validate, hash, and persist an uploaded file."""
        pass

    @abc.abstractmethod
    def verify_file_exists(self, storage_key: str) -> dict[str, Any]:
        """Verify file existence and return size_bytes and sha256_hash."""
        pass

    @abc.abstractmethod
    def cleanup_orphans(self, grace_period_hours: int = 24) -> int:
        """Purge unlinked files older than grace_period_hours."""
        pass


class LocalStorageBackend(BaseStorageBackend):
    """Local filesystem storage using randomized UUID filenames and path validation."""

    def __init__(self, upload_dir: Path | None = None):
        self._upload_dir = upload_dir or UPLOAD_DIR

    @property
    def upload_dir(self) -> Path:
        # Dynamically read global UPLOAD_DIR to support test monkeypatching
        return UPLOAD_DIR

    async def save_file(self, upload: UploadFile) -> dict[str, Any]:
        original_filename = Path(upload.filename or "").name.strip()
        extension = Path(original_filename).suffix.lower()

        if not original_filename or extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload a TXT, PDF, or DOCX file.",
            )

        content_type = (upload.content_type or "").lower()
        if content_type and content_type not in ALLOWED_MIME_TYPES[extension]:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file type does not match its extension.",
            )

        content = await upload.read(MAX_FILE_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_FILE_MB} MB limit.",
            )

        # Magic-byte inspection
        if extension == ".pdf" and not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Invalid PDF file.")

        if extension == ".docx" and not content.startswith(b"PK"):
            raise HTTPException(status_code=400, detail="Invalid DOCX file.")

        sha256_hash = hashlib.sha256(content).hexdigest()
        storage_name = f"{uuid.uuid4().hex}{extension}"

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        destination = (self.upload_dir / storage_name).resolve()

        # Path traversal guard
        if destination.parent != self.upload_dir.resolve():
            raise HTTPException(status_code=400, detail="Invalid storage destination.")

        try:
            with destination.open("xb") as file:
                file.write(content)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="The uploaded file could not be stored.",
            ) from exc

        return {
            "original_filename": original_filename,
            "storage_key": storage_name,
            "mime_type": content_type or "application/octet-stream",
            "size_bytes": len(content),
            "sha256_hash": sha256_hash,
            "storage_path": str(destination),
        }

    def verify_file_exists(self, storage_key: str) -> dict[str, Any]:
        """Verify that a storage_key resolves strictly inside UPLOAD_DIR."""
        if not storage_key or not GENERATED_NAME_PATTERN.fullmatch(storage_key):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file was not found in private storage.",
            )

        file_path = (self.upload_dir / storage_key).resolve()
        if file_path.parent != self.upload_dir.resolve() or not file_path.is_file():
            raise HTTPException(
                status_code=400,
                detail="Uploaded file was not found in private storage.",
            )

        return {
            "size_bytes": file_path.stat().st_size,
            "sha256_hash": hashlib.sha256(file_path.read_bytes()).hexdigest(),
        }

    def cleanup_orphans(self, grace_period_hours: int = 24) -> int:
        if grace_period_hours < 24:
            raise ValueError("Grace period must be at least 24 hours.")

        target_dir = self.upload_dir
        if not target_dir.exists():
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(hours=grace_period_hours)
        deleted_count = 0

        with SessionLocal() as db:
            for file_path in target_dir.iterdir():
                if not file_path.is_file():
                    continue

                if not GENERATED_NAME_PATTERN.fullmatch(file_path.name):
                    continue

                try:
                    modified_at = datetime.fromtimestamp(
                        file_path.stat().st_mtime,
                        tz=timezone.utc,
                    )
                except FileNotFoundError:
                    continue

                if modified_at > cutoff:
                    continue

                # Recheck the database immediately before deletion
                is_referenced = db.scalar(
                    select(IncidentFile.id)
                    .where(IncidentFile.storage_key == file_path.name)
                    .limit(1)
                )

                if is_referenced is not None:
                    continue

                try:
                    file_path.unlink()
                    deleted_count += 1
                except FileNotFoundError:
                    continue

        return deleted_count


class S3StorageBackend(BaseStorageBackend):
    """
    S3-compatible object storage backend supporting AWS S3, Cloudflare R2, and MinIO.
    Configured via:
        STORAGE_BACKEND=s3
        AWS_S3_BUCKET=<bucket_name>
        AWS_ENDPOINT_URL=<optional_endpoint_url_for_r2_or_minio>
        AWS_ACCESS_KEY_ID=<key_id>
        AWS_SECRET_ACCESS_KEY=<secret_key>
        AWS_DEFAULT_REGION=<region>
    """

    def __init__(self):
        self.bucket = os.getenv("AWS_S3_BUCKET", "safety-incident-uploads")
        self.endpoint_url = os.getenv("AWS_ENDPOINT_URL")
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    region_name=self.region,
                )
            except ImportError as err:
                raise RuntimeError(
                    "boto3 is required for S3StorageBackend. Run `pip install boto3`."
                ) from err
        return self._client

    async def save_file(self, upload: UploadFile) -> dict[str, Any]:
        original_filename = Path(upload.filename or "").name.strip()
        extension = Path(original_filename).suffix.lower()

        if not original_filename or extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload a TXT, PDF, or DOCX file.",
            )

        content = await upload.read(MAX_FILE_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_FILE_MB} MB limit.")

        if extension == ".pdf" and not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Invalid PDF file.")
        if extension == ".docx" and not content.startswith(b"PK"):
            raise HTTPException(status_code=400, detail="Invalid DOCX file.")

        sha256_hash = hashlib.sha256(content).hexdigest()
        storage_name = f"{uuid.uuid4().hex}{extension}"
        content_type = (upload.content_type or "").lower() or "application/octet-stream"

        client = self._get_client()
        try:
            client.put_object(
                Bucket=self.bucket,
                Key=f"private_uploads/{storage_name}",
                Body=content,
                ContentType=content_type,
                Metadata={"sha256": sha256_hash, "original_filename": original_filename},
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to store document in object storage.") from exc

        return {
            "original_filename": original_filename,
            "storage_key": storage_name,
            "mime_type": content_type,
            "size_bytes": len(content),
            "sha256_hash": sha256_hash,
            "storage_path": f"s3://{self.bucket}/private_uploads/{storage_name}",
        }

    def verify_file_exists(self, storage_key: str) -> dict[str, Any]:
        if not storage_key or not GENERATED_NAME_PATTERN.fullmatch(storage_key):
            raise HTTPException(status_code=400, detail="Uploaded file was not found in private storage.")

        client = self._get_client()
        try:
            res = client.head_object(Bucket=self.bucket, Key=f"private_uploads/{storage_key}")
            return {
                "size_bytes": res["ContentLength"],
                "sha256_hash": res.get("Metadata", {}).get("sha256", ""),
            }
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file was not found in private storage.")

    def cleanup_orphans(self, grace_period_hours: int = 24) -> int:
        # S3 lifecycle policies handle automated object expiration in cloud buckets
        return 0


def get_storage_backend() -> BaseStorageBackend:
    """Factory selecting storage backend based on the STORAGE_BACKEND environment variable."""
    backend_type = os.getenv("STORAGE_BACKEND", "local").lower().strip()
    if backend_type in ("s3", "r2", "minio"):
        return S3StorageBackend()
    return LocalStorageBackend()


# Standard functions maintaining complete backward compatibility
async def save_incident_file(upload: UploadFile) -> dict[str, Any]:
    """Validate and save an uploaded incident document privately."""
    return await get_storage_backend().save_file(upload)


def verify_file_exists(storage_key: str) -> dict[str, Any]:
    """Verify that an uploaded file exists in storage and return its metadata."""
    return get_storage_backend().verify_file_exists(storage_key)


def cleanup_orphaned_files(grace_period_hours: int = 24) -> int:
    """Delete old, unreferenced files created by this application."""
    return get_storage_backend().cleanup_orphans(grace_period_hours=grace_period_hours)
