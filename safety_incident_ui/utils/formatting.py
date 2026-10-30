from __future__ import annotations


def pct(value) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "Not provided"


def or_missing(value, missing: str = "Not stated in report") -> str:
    return str(value) if value not in (None, "", []) else missing
