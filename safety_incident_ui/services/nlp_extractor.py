import re
from datetime import datetime
from typing import Any

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
    "location": [
        "Admin Block", "Assembly Line A", "Assembly Line B",
        "Blending Area", "Boiler House", "Cable Gallery",
        "Canteen", "Chemical Store", "Compressor Room",
        "Cooling Tower Area", "Corridors and Stairways",
        "Cutting Area", "DG Room", "Despatch Area",
        "Dispatch Packing Line", "Effluent Treatment Plant",
        "Electrical Room", "Fabrication Shop",
        "Final Inspection Bay", "Finished Goods Store",
        "Heavy Fabrication Yard", "Incoming Inspection",
        "Labelling Section", "Loading Dock", "MCC Room",
        "Machining Bay", "Main Warehouse", "Maintenance Workshop",
        "Motor Repair Bay", "Moulding Area", "Packing Hall",
        "Paint Booth Area", "Palletizing Area", "Parking Area",
        "Plant Floor", "Press Shop", "Process Control Area",
        "Production Line 1", "Production Line 2",
        "Quality Laboratory", "Raw Material Store", "Reactor Bay",
        "Sample Store", "Stamping Area", "Sub-assembly Area",
        "Substation", "Tank Farm", "Testing Room", "Tool Crib",
        "Utility Yard", "Washrooms", "Water Treatment Plant",
        "Welding Bay", "Yard",
    ],
    "employee_role": [
        "Assembly Operator", "Boiler Operator", "CNC Operator",
        "Chemical Operator", "Chemist", "Contractor", "Driver",
        "Electrical Supervisor", "Electrical Technician",
        "Electrician", "Fabrication Supervisor", "Fabricator",
        "Facilities Supervisor", "Facilities Technician", "Fitter",
        "Forklift Operator", "Helper", "Housekeeping Staff",
        "Lab Supervisor", "Lab Technician", "Line Supervisor",
        "Loader", "Machine Operator", "Maintenance Supervisor",
        "Maintenance Technician", "Mechanical Fitter",
        "Packing Helper", "Packing Operator", "Process Supervisor",
        "Process Technician", "Production Helper",
        "Production Technician", "Quality Inspector",
        "Security Guard", "Setter", "Shift Chemist",
        "Stores Supervisor", "Utility Operator", "Utility Supervisor",
        "Utility Technician", "Warehouse Associate", "Welder",
    ],
    "reported_by_role": [
        "Co-worker", "Employee (Self)", "Medical Staff",
        "Safety Officer", "Shift Manager", "Supervisor",
        "Team Leader",
    ],
}

EQUIPMENT_ALIASES: dict[str, str] = {
    "pumping unit": "Pump",
    "pump unit": "Pump",
    "feed pump": "Pump",
    "fork truck": "Forklift",
    "fork lift": "Forklift",
    "air compressor unit": "Air Compressor",
    "compressor unit": "Air Compressor",
    "diesel generator set": "Diesel Generator",
    "diesel gen": "Diesel Generator",
    "dg set": "Diesel Generator",
    "welding rig": "Welding Machine",
    "welder machine": "Welding Machine",
    "overhead travelling crane": "Overhead Crane",
    "eot crane": "Overhead Crane",
    "pallet truck": "Pallet Jack",
}

STATUS_ALIASES: dict[str, str] = {
    "badly impaired": "Malfunctioning",
    "impaired": "Malfunctioning",
    "broken": "Damaged",
    "faulty": "Malfunctioning",
    "defective": "Malfunctioning",
    "out of order": "Malfunctioning",
    "not working": "Malfunctioning",
    "inoperative": "Malfunctioning",
    "failed": "Malfunctioning",
    "failure": "Malfunctioning",
    "shut down": "Shutdown/Isolated",
    "isolated": "Shutdown/Isolated",
    "needs maintenance": "Overdue for Maintenance",
    "being serviced": "Under Maintenance",
    "in service": "Operational",
    "running normally": "Operational",
}

DEPARTMENT_ALIASES: dict[str, str] = {
    "maintenance crew": "Maintenance",
    "maintenance team": "Maintenance",
    "maint crew": "Maintenance",
    "electrical team": "Electrical",
    "fabrication crew": "Fabrication & Welding",
    "warehouse staff": "Warehouse & Logistics",
    "logistics team": "Warehouse & Logistics",
    "qc lab": "Quality Control & Laboratory",
    "quality control": "Quality Control & Laboratory",
    "production line crew": "Production",
    "housekeeping team": "Facilities & Housekeeping",
}

HAZARD_ALIASES: dict[str, str] = {
    "hydrocarbon leak": "Chemical Substance",
    "oil leak": "Oil Leakage",
    "oil spillage": "Oil Leakage",
    "oil on the floor": "Slippery Surface",
    "oil on floor": "Slippery Surface",
    "wet floor": "Slippery Surface",
    "water on the floor": "Slippery Surface",
    "water on floor": "Slippery Surface",
    "exposed electrical wires": "Electrical Energy",
    "exposed wires": "Electrical Energy",
    "bare wires": "Electrical Energy",
    "exposed wire": "Electrical Energy",
    "fire broke out": "Fire Hazard",
    "flames": "Fire Hazard",
    "gas leakage": "Gas/Pressure",
    "high pressure": "Gas/Pressure",
    "heavy lifting": "Unsafe Lifting",
    "no ppe": "Inadequate PPE",
    "missing ppe": "Inadequate PPE",
}

INCIDENT_TYPE_ALIASES: dict[str, str] = {
    "caught fire": "Fire",
    "burst into flames": "Fire",
    "ignited": "Fire",
    "received an electric shock": "Electrical Shock",
    "got an electric shock": "Electrical Shock",
    "slipped and fell": "Slip/Fall",
    "slip and fall": "Slip/Fall",
    "tripped and fell": "Trip/Fall",
    "trip and fall": "Trip/Fall",
    "struck by object": "Struck By Object",
    "hit by object": "Struck By Object",
    "forklift collision": "Vehicle/Forklift Incident",
    "forklift crash": "Vehicle/Forklift Incident",
}

PHRASE_MAPPINGS: dict[str, dict[str, str]] = {
    "equipment_type": EQUIPMENT_ALIASES,
    "equipment_status": STATUS_ALIASES,
    "incident_type": INCIDENT_TYPE_ALIASES,
    "hazard_type": HAZARD_ALIASES,
    "department": DEPARTMENT_ALIASES,
}

EXPLICIT_STATUS_TERMS = [
    "damaged",
    "malfunctioning",
    "broken",
    "faulty",
    "defective",
    "failed",
    "failure",
    "impaired",
    "overdue for maintenance",
    "under maintenance",
    "operational",
    "working",
    "not working",
    "out of order",
    "shutdown",
    "isolated",
    "shut down",
    "badly impaired",
    "inoperative",
]


def normalize(text: str) -> str:
    """Normalize whitespace and lowercase text."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def find_category(text: str, categories: list[str]) -> str | None:
    """Exact word-boundary matching against known category labels."""
    normalized = normalize(text)
    for category in sorted(categories, key=len, reverse=True):
        pattern = (
            r"(?<!\w)"
            + re.escape(category.casefold()).replace(r"\ ", r"\s+")
            + r"(?!\w)"
        )
        if re.search(pattern, normalized):
            return category
    return None


def has_negation(text: str, phrase: str) -> bool:
    """Conservative check for explicit negation patterns around phrase."""
    text_norm = normalize(text)
    phrase_norm = normalize(phrase)

    # Generate variants (e.g. 'oil leakage' -> 'oil leak')
    variants = [phrase_norm]
    if "leakage" in phrase_norm:
        variants.append(phrase_norm.replace("leakage", "leak"))

    for p in variants:
        patterns = [
            rf"\bno\s+(?:evidence\s+of\s+)?(?:any\s+)?{re.escape(p)}\b",
            rf"\b(?:no|without|absence\s+of|zero)\s+{re.escape(p)}\b",
            rf"\b{re.escape(p)}\s+(?:was|were|is|are)\s+(?:not\s+present|absent|not\s+observed|not\s+found|non-existent|not\s+detected)\b",
            rf"\b(?:not|never)\s+(?:observed|found|detected|reported|seen)\b.{{0,35}}\b{re.escape(p)}\b",
            rf"\b{re.escape(p)}\s+(?:was|were)\s+absent\b",
            rf"\bno\s+{re.escape(p)}\b",
        ]
        if any(re.search(pat, text_norm) for pat in patterns):
            return True
    return False


def hazard_is_resolved(text: str, phrase: str) -> bool:
    """Detect explicit resolution patterns for hazards."""
    text_norm = normalize(text)
    phrase_norm = normalize(phrase)

    variants = [phrase_norm]
    if "leakage" in phrase_norm:
        variants.append(phrase_norm.replace("leakage", "leak"))

    for p in variants:
        patterns = [
            rf"\b{re.escape(p)}\b.{{0,45}}\b(?:repaired|fixed|resolved|eliminated|cleaned up|rectified|contained)\b",
            rf"\b(?:repaired|fixed|resolved|eliminated|cleaned up|rectified|contained)\b.{{0,45}}\b{re.escape(p)}\b",
            rf"\b{re.escape(p)}\b.{{0,45}}\b(?:no longer present|no longer exists|had been repaired|had been fixed)\b",
            rf"\b{re.escape(p)}\b.{{0,65}}\b(?:before this report|prior to this report|not part of the incident)\b",
        ]
        if any(re.search(pat, text_norm) for pat in patterns):
            return True
    return False


def has_explicit_status_evidence(text: str) -> bool:
    """Require clear status wording before extracting equipment status."""
    text_norm = normalize(text)
    return any(
        re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text_norm)
        for term in EXPLICIT_STATUS_TERMS
    )


def is_status_negated(text: str, status_term: str) -> bool:
    """Check if all occurrences of status_term in text are explicitly negated."""
    pattern = rf"(?<!\w){re.escape(status_term.casefold())}(?!\w)"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return False

    negated_count = 0
    for m in matches:
        start_idx = max(0, m.start() - 35)
        prefix = text[start_idx:m.start()].casefold()
        if re.search(r"\b(?:not|no|never|without)\s+(?:being\s+|found\s+|observed\s+)?$", prefix) or \
           re.search(r"\b(?:was|were|is|are)\s+not\s*$", prefix):
            negated_count += 1

    return negated_count == len(matches)


def is_equipment_negated_in_clause(clause: str, eq_term: str) -> bool:
    """Check if eq_term is explicitly negated within a single clause."""
    norm = normalize(clause)
    p = re.escape(eq_term.casefold())
    patterns = [
        rf"\b{p}\s+(?:was|were|is|are)?\s*not\s+(?:damaged|broken|faulty|involved|affected)\b",
        rf"\bnot\s+(?:damaged|broken|faulty|involved|affected)\b.{{0,25}}\b{p}\b",
        rf"\bno\s+(?:damage|issue|problem|defect)\s+(?:to|on|with|in)\s+(?:the\s+)?{p}\b",
        rf"\b(?:no|neither)\s+{p}\b",
        rf"\b{p}\s+(?:was|were|is|are)?\s*(?:undamaged|unaffected|ok|intact)\b",
    ]
    return any(re.search(pat, norm) for pat in patterns)


def resolve_equipment_and_status(
    text: str,
    search_text: str,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """
    Extract equipment and status with conflict awareness and negation guards.
    Handles contrastive clauses and multiple equipment mentions.
    """
    diagnostics: dict[str, Any] = {}
    text_norm = normalize(text)

    # Split text into contrastive / structural clauses
    clauses = [
        c.strip()
        for c in re.split(r"\b(?:but|however|whereas|while|although|instead\s+of)\b|[;.]", text_norm)
        if c.strip()
    ]
    if not clauses:
        clauses = [text_norm]

    all_equipment_mentions: list[dict[str, Any]] = []

    for clause_idx, clause in enumerate(clauses):
        # 1. Check aliases in this clause
        for alias, canonical in sorted(EQUIPMENT_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", clause):
                negated = is_equipment_negated_in_clause(clause, alias)
                all_equipment_mentions.append({
                    "label": canonical,
                    "raw_match": alias,
                    "clause_idx": clause_idx,
                    "clause_text": clause,
                    "is_alias": True,
                    "negated": negated,
                })

        # 2. Check canonical categories in this clause
        for cat in sorted(CATEGORIES["equipment_type"], key=len, reverse=True):
            pat = r"(?<!\w)" + re.escape(cat.casefold()).replace(r"\ ", r"\s+") + r"(?!\w)"
            if re.search(pat, clause):
                if not any(e["label"] == cat and e["clause_idx"] == clause_idx for e in all_equipment_mentions):
                    negated = is_equipment_negated_in_clause(clause, cat)
                    all_equipment_mentions.append({
                        "label": cat,
                        "raw_match": cat,
                        "clause_idx": clause_idx,
                        "clause_text": clause,
                        "is_alias": False,
                        "negated": negated,
                    })

    active_mentions = [e for e in all_equipment_mentions if not e["negated"]]

    selected_equipment: str | None = None
    target_clause_text = text_norm

    if active_mentions:
        # If any active mention has explicit status words in its clause, prioritize it
        prioritized = [
            e for e in active_mentions
            if has_explicit_status_evidence(e["clause_text"])
        ]
        chosen = prioritized[0] if prioritized else active_mentions[0]
        selected_equipment = chosen["label"]
        target_clause_text = chosen["clause_text"]
        diagnostics["equipment_type"] = {
            "method": "explicit_alias" if chosen["is_alias"] else "explicit_match",
            "candidate": selected_equipment,
            "all_active": [e["label"] for e in active_mentions],
        }
    elif all_equipment_mentions:
        diagnostics["equipment_type"] = {
            "method": "rule_guard",
            "reason": "all_equipment_candidates_negated",
        }
    else:
        diagnostics["equipment_type"] = {
            "method": "conservative_fallback",
            "reason": "no_explicit_equipment_evidence",
        }

    # 3. Extract equipment status
    selected_status: str | None = None

    if selected_equipment is None:
        diagnostics["equipment_status"] = {
            "method": "rule_guard",
            "reason": "no_valid_equipment_identified",
        }
    elif not has_explicit_status_evidence(text):
        diagnostics["equipment_status"] = {
            "method": "rejected",
            "reason": "no_explicit_status_evidence",
        }
    else:
        # Check status candidates first in the target clause (or full text if no target clause)
        search_status_texts = [target_clause_text, text_norm] if target_clause_text != text_norm else [text_norm]

        status_candidates: list[dict[str, Any]] = []
        for st_text in search_status_texts:
            for alias, canonical in sorted(STATUS_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
                if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", st_text):
                    if not is_status_negated(st_text, alias):
                        status_candidates.append({
                            "label": canonical,
                            "raw_match": alias,
                            "is_alias": True,
                        })

            for cat in sorted(CATEGORIES["equipment_status"], key=len, reverse=True):
                if re.search(r"(?<!\w)" + re.escape(cat.casefold()).replace(r"\ ", r"\s+") + r"(?!\w)", st_text):
                    if not is_status_negated(st_text, cat.casefold()):
                        if not any(sc["label"] == cat for sc in status_candidates):
                            status_candidates.append({
                                "label": cat,
                                "raw_match": cat,
                                "is_alias": False,
                            })
            if status_candidates:
                break

        if status_candidates:
            selected_status = status_candidates[0]["label"]
            diagnostics["equipment_status"] = {
                "method": "explicit_alias" if status_candidates[0]["is_alias"] else "explicit_match",
                "candidate": selected_status,
                "all_candidates": [sc["label"] for sc in status_candidates],
            }
        else:
            diagnostics["equipment_status"] = {
                "method": "rule_guard",
                "reason": "status_terms_negated_or_unmatched",
            }

    return selected_equipment, selected_status, diagnostics


def extract_incident_features(
    report_text: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Extract explicit incident details adhering to the 16-field schema.
    Unknown or unevidenced features remain None (conservative fallback).
    Includes nlp_diagnostics separate from feature columns.
    """
    context = context or {}
    text_content = report_text.strip()

    extracted: dict[str, Any] = {
        "incident_year": None,
        "incident_month": None,
        "incident_dayofweek": None,
        "incident_hour": None,
        "incident_minute": None,
        "department": context.get("department") or None,
        "location": context.get("location") or None,
        "shift": context.get("shift") or None,
        "employee_role": None,
        "experience_level": None,
        "equipment_type": None,
        "equipment_status": None,
        "incident_type": None,
        "hazard_type": None,
        "incident_description": text_content,
        "reported_by_role": None,
    }

    diagnostics: dict[str, Any] = {}

    # 1. Date extraction: ISO ('YYYY-MM-DD') or Natural text ('12 October 2024')
    iso_date = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text_content)
    if iso_date:
        try:
            d = datetime.strptime(iso_date.group(0), "%Y-%m-%d").date()
            extracted["incident_year"] = d.year
            extracted["incident_month"] = d.month
            extracted["incident_dayofweek"] = d.weekday()
            diagnostics["date"] = {"method": "regex_iso_date"}
        except ValueError:
            pass
    else:
        date_match = re.search(
            r"\b(\d{1,2})\s+"
            r"(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+"
            r"(\d{4})\b",
            text_content,
            re.IGNORECASE,
        )
        if date_match:
            try:
                incident_date = datetime.strptime(date_match.group(0), "%d %B %Y")
                extracted["incident_year"] = incident_date.year
                extracted["incident_month"] = incident_date.month
                extracted["incident_dayofweek"] = incident_date.weekday()
                diagnostics["date"] = {"method": "regex_text_date"}
            except ValueError:
                pass

    # 2. Time extraction: 24h ('17:25', '14:30', '09:15') or 12h ('5:25 PM')
    time_match = re.search(
        r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*(AM|PM))?\b",
        text_content,
        re.IGNORECASE,
    )
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        suffix = (time_match.group(3) or "").upper()
        if suffix == "PM" and hour < 12:
            hour += 12
        elif suffix == "AM" and hour == 12:
            hour = 0
        extracted["incident_hour"] = hour
        extracted["incident_minute"] = minute
        diagnostics["time"] = {"method": "regex_time"}

    # 3. Shift extraction
    if not extracted["shift"]:
        for s in CATEGORIES["shift"]:
            if re.search(rf"\b{s}\s+shift\b", text_content, re.IGNORECASE):
                extracted["shift"] = s
                diagnostics["shift"] = {"method": "explicit_shift_pattern"}
                break

    # Prepare search text with location names masked out to prevent cross-contamination
    search_text = text_content
    for loc in CATEGORIES["location"]:
        search_text = re.sub(
            r"(?<!\w)" + re.escape(loc).replace(r"\ ", r"\s+") + r"(?!\w)",
            " ",
            search_text,
            flags=re.IGNORECASE,
        )

    # 4. Department extraction (check context -> aliases -> canonical)
    if not extracted["department"]:
        for alias, dept in DEPARTMENT_ALIASES.items():
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text_content, re.IGNORECASE):
                extracted["department"] = dept
                diagnostics["department"] = {"method": "department_alias", "alias": alias}
                break
        if not extracted["department"]:
            dept_match = find_category(search_text, CATEGORIES["department"])
            if dept_match:
                extracted["department"] = dept_match
                diagnostics["department"] = {"method": "explicit_match"}

    # 5. Location extraction
    if not extracted["location"]:
        loc_match = find_category(text_content, CATEGORIES["location"])
        if loc_match:
            extracted["location"] = loc_match
            diagnostics["location"] = {"method": "explicit_match"}

    # 6. Equipment type and Equipment status (with conflict-resolution & negation guards)
    eq_type, eq_status, eq_diag = resolve_equipment_and_status(text_content, search_text)
    extracted["equipment_type"] = eq_type
    extracted["equipment_status"] = eq_status
    diagnostics.update(eq_diag)

    # 7. Hazard type extraction with negation & resolution guards
    hazard_candidate = None
    hazard_method = None

    # Check hazard aliases first (e.g. 'oil on the floor' -> 'Slippery Surface', 'oil leak' -> 'Oil Leakage')
    for alias, canonical_hazard in sorted(HAZARD_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text_content, re.IGNORECASE):
            hazard_candidate = canonical_hazard
            hazard_method = "hazard_alias"
            break

    # If no alias match, check canonical hazard categories
    if not hazard_candidate:
        can_hazard = find_category(text_content, CATEGORIES["hazard_type"])
        if can_hazard:
            hazard_candidate = can_hazard
            hazard_method = "explicit_match"

    # Apply negation and resolution guards to candidate hazard
    if hazard_candidate:
        if has_negation(text_content, hazard_candidate) or hazard_is_resolved(text_content, hazard_candidate):
            extracted["hazard_type"] = None
            diagnostics["hazard_type"] = {
                "method": "rule_guard",
                "candidate": hazard_candidate,
                "reason": "negated_or_resolved_mention",
            }
        else:
            extracted["hazard_type"] = hazard_candidate
            diagnostics["hazard_type"] = {
                "method": hazard_method,
                "candidate": hazard_candidate,
            }
    else:
        diagnostics["hazard_type"] = {
            "method": "conservative_fallback",
            "reason": "no_hazard_evidence",
        }

    # 8. Incident type extraction with phrase aliases
    for alias, canonical_incident in sorted(INCIDENT_TYPE_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text_content, re.IGNORECASE):
            extracted["incident_type"] = canonical_incident
            diagnostics["incident_type"] = {"method": "incident_type_alias", "alias": alias}
            break

    if not extracted["incident_type"]:
        inc_match = find_category(text_content, CATEGORIES["incident_type"])
        if inc_match:
            extracted["incident_type"] = inc_match
            diagnostics["incident_type"] = {"method": "explicit_match"}

    # 9. Employee role & Reported by role
    for role in sorted(CATEGORIES["employee_role"], key=len, reverse=True):
        if re.search(r"(?<!\w)" + re.escape(role.casefold()).replace(r"\ ", r"\s+") + r"(?!\w)", text_content, re.IGNORECASE):
            extracted["employee_role"] = role
            diagnostics["employee_role"] = {"method": "explicit_match"}
            break

    for role in sorted(CATEGORIES["reported_by_role"], key=len, reverse=True):
        if re.search(r"(?<!\w)" + re.escape(role.casefold()).replace(r"\ ", r"\s+") + r"(?!\w)", text_content, re.IGNORECASE):
            extracted["reported_by_role"] = role
            diagnostics["reported_by_role"] = {"method": "explicit_match"}
            break

    # 10. Experience level
    experience_patterns = [
        (r"\bless than 1 year\b", "Less than 1 year"),
        (r"\b1\s*[-–]\s*3 years?\b", "1-3 years"),
        (r"\b3\s*[-–]\s*7 years?\b", "3-7 years"),
        (r"\bmore than 7 years\b", "More than 7 years"),
    ]
    for pattern, label in experience_patterns:
        if re.search(pattern, text_content, re.IGNORECASE):
            extracted["experience_level"] = label
            diagnostics["experience_level"] = {"method": "regex_experience"}
            break

    # Attach diagnostics and metadata
    extracted["nlp_diagnostics"] = diagnostics
    extracted["nlp_method"] = "production_hybrid_rules"
    extracted["nlp_review_required"] = True

    return extracted
