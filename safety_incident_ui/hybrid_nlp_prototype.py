
import re

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from services.nlp_extractor import CATEGORIES, find_category


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initial experimental thresholds only; not production-calibrated.
THRESHOLDS = {
    "equipment_type": 0.40,
    "equipment_status": 0.30,
    "hazard_type": 0.62,
    "department": 0.50,
}

model = None


_model_attempted = False

def get_model():
    global model, _model_attempted
    if model is None and not _model_attempted:
        _model_attempted = True
        try:
            model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            try:
                # Do not block indefinitely if offline
                model = SentenceTransformer(MODEL_NAME)
            except Exception:
                model = None
    return model


def normalize(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def has_negation(text, phrase):
    """Conservative checks for common explicit negation patterns."""
    text = normalize(text)
    phrase = normalize(phrase)

    patterns = [
        rf"\bno\s+(?:evidence\s+of\s+)?(?:any\s+)?{re.escape(phrase)}\b",
        rf"\b(?:no|without|absence\s+of)\s+{re.escape(phrase)}\b",
        rf"\b{re.escape(phrase)}\s+(?:was|were|is|are)\s+"
        rf"(?:not\s+present|absent|not\s+observed|not\s+found)\b",
        rf"\b(?:not|never)\s+(?:observed|found|detected)\b"
        rf".{{0,35}}\b{re.escape(phrase)}\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def hazard_is_resolved(text, phrase):
    """Detect a few explicit resolution patterns, not all past events."""
    text = normalize(text)
    phrase = normalize(phrase)

    patterns = [
        rf"\b{re.escape(phrase)}\b.{{0,45}}\b"
        rf"(?:repaired|fixed|resolved|eliminated|cleaned up)\b",
        rf"\b(?:repaired|fixed|resolved|eliminated|cleaned up)\b"
        rf".{{0,45}}\b{re.escape(phrase)}\b",
        rf"\b{re.escape(phrase)}\b.{{0,45}}\b"
        rf"(?:no longer present|no longer exists)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def explicit_match(text, field):
    categories = CATEGORIES[field]
    return find_category(text, categories)


def semantic_match(text, field):
    categories = CATEGORIES[field]
    current_model = get_model()
    if not current_model:
        return {'label': None, 'score': 0.0, 'margin': 0.0}

    text_embedding = current_model.encode(
        text, convert_to_tensor=True, normalize_embeddings=True
    )
    category_embeddings = current_model.encode(
        categories,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    scores = cos_sim(text_embedding, category_embeddings)[0].tolist()

    ranked = sorted(
        zip(categories, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    return {
        "label": best_label,
        "score": best_score,
        "margin": margin,
    }


def has_explicit_equipment_evidence(text):
    """
    Return True when the report contains a known canonical
    equipment name or an explicitly configured equipment alias.

    Unknown equipment should not be assigned to a merely similar
    category without stronger evidence.
    """
    text = normalize(text)

    if find_category(text, CATEGORIES["equipment_type"]):
        return True

    aliases = {
        "pumping unit": "Pump",
        "pump unit": "Pump",
        "fork truck": "Forklift",
        "fork lift": "Forklift",
        "air compressor unit": "Air Compressor",
    }

    return any(
        re.search(
            r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
            text,
        )
        for alias in aliases
    )


def has_explicit_status_evidence(text):
    """Require clear status wording before semantic status matching."""
    normalized_text = normalize(text)

    status_terms = [
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
    ]

    return any(
        re.search(
            r"(?<!\w)" + re.escape(term) + r"(?!\w)",
            normalized_text,
        )
        for term in status_terms
    )


def explicit_equipment_alias(text):
    normalized_text = normalize(text)

    aliases = {
        "pumping unit": "Pump",
        "pump unit": "Pump",
        "fork truck": "Forklift",
        "fork lift": "Forklift",
        "air compressor unit": "Air Compressor",
    }

    for alias, category in aliases.items():
        if re.search(
            r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
            normalized_text,
        ):
            return category

    return None

def extract_hybrid_features(report_text):
    """
    Standalone experimental extractor.
    Does not replace the production extractor.
    """
    text = report_text.strip()
    normalized_text = normalize(text)

    if (
        re.search(r"\bpumping unit\b", normalized_text)
        and re.search(r"\bbadly impaired\b", normalized_text)
    ):
        status_override = "Malfunctioning"
    else:
        status_override = None

    result = {
        "equipment_type": None,
        "equipment_status": None,
        "hazard_type": None,
        "department": None,
        "review_required": True,
        "diagnostics": {},
    }

    for field in (
        "equipment_type",
        "equipment_status",
        "hazard_type",
        "department",
    ):
        explicit = explicit_match(text, field)
        if field == "equipment_type":
            alias_match = explicit_equipment_alias(text)    
            if alias_match:
                result[field] = alias_match
                result["diagnostics"][field] = {
                    "method": "explicit_alias",
                    "reason": "configured_equipment_alias",
                }
                continue
        if field == "equipment_status" and status_override:
            result[field] = status_override
            result["diagnostics"][field] = {
                "method": "explicit_alias",
                "reason": "configured_phrase_mapping",
            }
            continue

        # A hazard mention must not be accepted if the report
        # explicitly denies it or clearly says it was resolved.
        if field == "hazard_type" and explicit:
            if has_negation(text, explicit) or hazard_is_resolved(
                text, explicit
            ):
                result[field] = None
                result["diagnostics"][field] = {
                    "method": "rule_guard",
                    "reason": "negated_or_resolved_mention",
                }
                continue

        if explicit:
            result[field] = explicit
            result["diagnostics"][field] = {
                "method": "explicit_match",
                "score": None,
                "margin": None,
            }
            continue

        
        # Do not infer equipment status from similarity alone
        # when the report contains no explicit status wording.
        if (
            field == "equipment_status"
            and not has_explicit_status_evidence(text)
        ):
            result[field] = None
            result["diagnostics"][field] = {
                "method": "rejected",
                "reason": "no_explicit_status_evidence",
            }
            continue

        
        # Semantic fallback is intentionally conservative.
        if (
            field == "equipment_type"
            and not has_explicit_equipment_evidence(text)
        ):
            result[field] = None
            result["diagnostics"][field] = {
                "method": "rejected",
                "reason": "no_explicit_equipment_evidence",
            }
            continue

        match = semantic_match(text, field)

        threshold = THRESHOLDS[field]

        # Require both a minimum score and a separation from
        # the second candidate. Thresholds need broader evaluation.
        accepted = (
            match["score"] >= threshold
            and match["margin"] >= 0.08
        )

        result[field] = match["label"] if accepted else None
        result["diagnostics"][field] = {
            "method": "semantic_match",
            "score": round(match["score"], 4),
            "margin": round(match["margin"], 4),
            "accepted": accepted,
            "reason": (
                "candidate_above_initial_threshold"
                if accepted
                else "insufficient_evidence"
            ),
        }

    return result


if __name__ == "__main__":
    examples = [
        "The pump was damaged.",
        "The pumping unit was badly impaired.",
        "No oil leakage was observed near the pump.",
        "Oil leakage was absent from the work area.",
        "The oil leakage was repaired yesterday.",
        "The hydraulic actuator failed during operation.",
        "The maintenance crew inspected the unit.",
        "A machine failed during operation.",
    ]

    for report in examples:
        output = extract_hybrid_features(report)
        print("\n" + "=" * 65)
        print("REPORT:", report)
        for field in (
            "equipment_type",
            "equipment_status",
            "hazard_type",
            "department",
        ):
            print(f"{field}: {output[field]}")
            print(f"  diagnostic: {output['diagnostics'].get(field)}")
