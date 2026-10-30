
import os
import tempfile
import time
import uuid
from pathlib import Path

from sqlalchemy import select

import file_storage
from database import SessionLocal
from models import IncidentFile


# Get a real storage key from the database, but do not touch its real file.
with SessionLocal() as db:
    referenced_key = db.scalar(
        select(IncidentFile.storage_key).limit(1)
    )

if not referenced_key:
    raise RuntimeError(
        "No incident_files records found. Cannot test referenced-file preservation."
    )

if Path(referenced_key).name != referenced_key:
    raise RuntimeError("Unexpected storage key format; aborting test.")

# Use an isolated temporary directory instead of real private storage.
with tempfile.TemporaryDirectory(prefix="cleanup_test_") as temp_dir:
    test_dir = Path(temp_dir)

    referenced_file = test_dir / referenced_key
    orphan_file = test_dir / f"{uuid.uuid4().hex}.txt"

    referenced_file.write_text("Referenced test file", encoding="utf-8")
    orphan_file.write_text("Orphan test file", encoding="utf-8")

    # Make both files older than the 24-hour grace period.
    old_timestamp = time.time() - (48 * 60 * 60)
    os.utime(referenced_file, (old_timestamp, old_timestamp))
    os.utime(orphan_file, (old_timestamp, old_timestamp))

    # Temporarily redirect cleanup to the isolated test directory.
    original_upload_dir = file_storage.UPLOAD_DIR

    try:
        file_storage.UPLOAD_DIR = test_dir

        deleted_count = file_storage.cleanup_orphaned_files(
            grace_period_hours=24
        )

        assert deleted_count == 1, (
            f"Expected 1 deletion, got {deleted_count}"
        )
        assert referenced_file.exists(), (
            "FAIL: Referenced file was deleted."
        )
        assert not orphan_file.exists(), (
            "FAIL: Orphaned file was not deleted."
        )

        print("PASS: Referenced file was preserved.")
        print("PASS: Old orphaned file was deleted.")
        print("PASS: Exactly one file was deleted.")

    finally:
        file_storage.UPLOAD_DIR = original_upload_dir
