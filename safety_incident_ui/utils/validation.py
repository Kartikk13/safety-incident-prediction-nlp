"""Input validation for incident reports. Pure functions, no Streamlit."""
from __future__ import annotations

# Supported extensions match the backend file storage capabilities
SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")
MAX_FILE_MB = 10
MIN_REPORT_CHARS = 40


def validate_report_text(text: str | None) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "The report is empty. Paste the incident description to continue."
    if len(text.strip()) < MIN_REPORT_CHARS:
        return False, f"The report is too short to analyze. Describe what happened in at least {MIN_REPORT_CHARS} characters."
    return True, ""


def validate_upload(name: str, data: bytes) -> tuple[bool, str, str]:
    """Return (ok, message, decoded_text)."""
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}.", ""
    if not data:
        return False, "The uploaded file is empty.", ""
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return False, f"The file is larger than {MAX_FILE_MB} MB.", ""

    # Binary document formats are supported for upload but not direct text parsing without verified OCR/parser
    if ext in (".pdf", ".docx"):
        return True, "", ""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False, "The text file could not be read as UTF-8 text.", ""

    ok, msg = validate_report_text(text)
    return (ok, msg, text if ok else "")

