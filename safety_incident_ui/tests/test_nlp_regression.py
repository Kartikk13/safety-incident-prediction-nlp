"""
test_nlp_regression.py

Automated regression suite verifying:
- Canonical and paraphrased entity extraction
- Negation guards and resolved hazard handling
- Absence of false-positive equipment status on generic or unfamiliar phrasing
- Contrastive clause and multi-equipment conflict resolution
- 16-field schema preservation and None fallbacks
- Strict isolation of NLP diagnostics from ML feature columns
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from services.nlp_extractor import (
    extract_incident_features,
    has_negation,
    hazard_is_resolved,
    has_explicit_status_evidence,
    CATEGORIES,
)


FEATURE_COLUMNS_16 = [
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


def test_schema_completeness_and_conservative_fallbacks():
    """Verify that all 16 schema columns exist and unknown values remain None."""
    res = extract_incident_features("A short note with no discernible details.")
    for col in FEATURE_COLUMNS_16:
        assert col in res, f"Missing required column: {col}"

    assert res["equipment_type"] is None
    assert res["equipment_status"] is None
    assert res["hazard_type"] is None
    assert res["incident_year"] is None
    assert res["incident_hour"] is None
    assert res["nlp_method"] == "production_hybrid_rules"
    assert res["nlp_review_required"] is True
    assert "nlp_diagnostics" in res


def test_canonical_extraction():
    """Verify standard canonical mentions are matched accurately."""
    report = "The pump was damaged in the Compressor Room during the Morning shift."
    res = extract_incident_features(report)
    assert res["equipment_type"] == "Pump"
    assert res["equipment_status"] == "Damaged"
    assert res["location"] == "Compressor Room"
    assert res["shift"] == "Morning"


def test_equipment_and_status_paraphrases():
    """Verify domain aliases for equipment and status."""
    report = "The pumping unit was badly impaired."
    res = extract_incident_features(report)
    assert res["equipment_type"] == "Pump"
    assert res["equipment_status"] == "Malfunctioning"

    report_forklift = "The fork truck was broken."
    res_fork = extract_incident_features(report_forklift)
    assert res_fork["equipment_type"] == "Forklift"
    assert res_fork["equipment_status"] == "Damaged"


def test_unfamiliar_equipment_conservative_fallback():
    """Ensure unknown equipment is not forced into similarity candidates."""
    report = "The hydraulic actuator failed during operation."
    res = extract_incident_features(report)
    assert res["equipment_type"] is None
    # No recognized equipment type -> status must remain None
    assert res["equipment_status"] is None


def test_negated_hazard_handling():
    """Verify explicit negations suppress hazard false positives."""
    report1 = "No oil leakage was observed near the pump."
    res1 = extract_incident_features(report1)
    assert res1["equipment_type"] == "Pump"
    assert res1["hazard_type"] is None

    report2 = "There was no oil leak near the pump."
    res2 = extract_incident_features(report2)
    assert res2["hazard_type"] is None

    report3 = "Oil leakage was absent from the work area."
    res3 = extract_incident_features(report3)
    assert res3["hazard_type"] is None


def test_resolved_hazard_handling():
    """Verify historically repaired or resolved hazards are excluded."""
    report1 = "The oil leakage was repaired yesterday."
    res1 = extract_incident_features(report1)
    assert res1["hazard_type"] is None

    report2 = "The oil leak had been repaired before this report and was not part of the incident."
    res2 = extract_incident_features(report2)
    assert res2["hazard_type"] is None


def test_department_paraphrase_vs_ambiguous_wording():
    """Ensure clear department phrases match while ambiguous roles do not hallucinate departments."""
    report_valid = "The maintenance crew inspected the unit."
    res_valid = extract_incident_features(report_valid)
    assert res_valid["department"] == "Maintenance"

    report_ambiguous = "The mechanics checked the unit after the incident."
    res_ambiguous = extract_incident_features(report_ambiguous)
    assert res_ambiguous["department"] is None


def test_multiple_equipment_conflict_resolution():
    """Verify contrastive clauses resolve equipment negation and assign status to active equipment."""
    report = "The pump was not damaged, but a nearby conveyor was malfunctioning."
    res = extract_incident_features(report)
    assert res["equipment_type"] == "Conveyor"
    assert res["equipment_status"] == "Malfunctioning"

    report2 = "The pump was not damaged, but the conveyor was damaged."
    res2 = extract_incident_features(report2)
    assert res2["equipment_type"] == "Conveyor"
    assert res2["equipment_status"] == "Damaged"


def test_iso_and_text_dates_times():
    """Verify date and time extraction across formats."""
    # ISO date + 24h time
    res_iso = extract_incident_features("On 2026-09-12 at 14:30 an incident occurred.")
    assert res_iso["incident_year"] == 2026
    assert res_iso["incident_month"] == 9
    assert res_iso["incident_dayofweek"] == 5
    assert res_iso["incident_hour"] == 14
    assert res_iso["incident_minute"] == 30

    # Natural text date + 12h AM/PM time
    res_nat = extract_incident_features("On 12 October 2024 at 5:25 PM an incident occurred.")
    assert res_nat["incident_year"] == 2024
    assert res_nat["incident_month"] == 10
    assert res_nat["incident_dayofweek"] == 5
    assert res_nat["incident_hour"] == 17
    assert res_nat["incident_minute"] == 25


def test_diagnostic_metadata_isolation():
    """Verify that diagnostics are separated and never injected into model feature vectors."""
    report = "The pumping unit was badly impaired."
    res = extract_incident_features(report)

    # Diagnostic metadata must be present under its own key
    assert "nlp_diagnostics" in res
    assert res["nlp_diagnostics"]["equipment_type"]["method"] in ("explicit_alias", "explicit_match")

    # Feature vector for CatBoost pipeline must strictly contain the 16 features
    feature_payload = {k: v for k, v in res.items() if k in FEATURE_COLUMNS_16}
    assert len(feature_payload) == 16
    assert "nlp_diagnostics" not in feature_payload
    assert "nlp_method" not in feature_payload
    assert "nlp_review_required" not in feature_payload
