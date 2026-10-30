"""DEMONSTRATION DATA ONLY.

Everything here is synthetic and exists so the UI can run before the real
model/backend exists. Nothing in this file represents real incidents or
industry statistics. The UI must not import this module directly; it goes
through services/api_client.py, which is the swap point for the FastAPI/ML
responses. Severity labels and cause names below are placeholders.
"""
from __future__ import annotations

import hashlib
import random
import re
from datetime import date, timedelta

import pandas as pd

SEVERITY_LEVELS = ["Low", "Medium", "High"]  # replace with the model's real labels
INCIDENT_TYPES = ["Loss of Containment", "Equipment Failure", "Fire / Explosion",
                  "Process Upset", "Near Miss", "Personnel Injury"]

EQUIPMENT = {  # tag -> (name, unit)
    "P-101A": ("Feed pump", "Process Unit A"), "R-201": ("Reactor", "Process Unit A"),
    "PSV-212": ("Relief valve", "Process Unit A"), "K-301": ("Recycle gas compressor", "Process Unit B"),
    "E-140": ("Heat exchanger", "Process Unit B"), "H-101": ("Fired heater", "Process Unit B"),
    "LT-315": ("Level transmitter", "Process Unit B"), "T-410": ("Storage tank", "Tank Farm"),
    "L-2204": ("Transfer pipeline", "Tank Farm"),
}

ROOT_CAUSES = {
    "Inadequate Maintenance": dict(factor="Maintenance", kw=["overdue", "maintenance", "deferred"],
        equipment=["P-101A", "K-301", "E-140"], types=["Loss of Containment", "Equipment Failure"], sev=0.50,
        template="The seal on {name} {tag} leaked during routine operation. Preventive maintenance was overdue and wear was not identified at the last inspection."),
    "Procedure Not Followed": dict(factor="Human", kw=["procedure", "skipped", "bypassed"],
        equipment=["R-201", "H-101", "T-410"], types=["Process Upset", "Near Miss", "Personnel Injury"], sev=0.55,
        template="During startup of {name} {tag} an operator skipped a step in the written procedure and a valve was left in the wrong position."),
    "Excessive Operating Pressure": dict(factor="Operational", kw=["pressure exceeded", "overpressure", "high pressure"],
        equipment=["R-201", "PSV-212", "K-301"], types=["Fire / Explosion", "Loss of Containment", "Process Upset"], sev=0.68,
        template="Operating pressure exceeded the design limit on {name} {tag} and the relief device lifted."),
    "Corrosion / Material Degradation": dict(factor="Maintenance", kw=["corrosion", "wall thinning", "erosion"],
        equipment=["L-2204", "E-140", "T-410"], types=["Loss of Containment", "Equipment Failure"], sev=0.50,
        template="Wall thinning from corrosion was found on {name} {tag} after a small hydrocarbon leak."),
    "Instrument Failure": dict(factor="Operational", kw=["sensor", "instrument", "transmitter", "alarm did not"],
        equipment=["LT-315", "R-201", "H-101"], types=["Process Upset", "Near Miss"], sev=0.40,
        template="A sensor fault on {name} {tag} gave an incorrect reading and the high-level alarm did not activate."),
    "Inadequate Training": dict(factor="Human", kw=["inexperienced", "unfamiliar", "training"],
        equipment=["P-101A", "H-101", "T-410"], types=["Near Miss", "Process Upset", "Personnel Injury"], sev=0.38,
        template="An inexperienced operator unfamiliar with {name} {tag} shutdown steps triggered a process upset."),
    "Management of Change Gap": dict(factor="Operational", kw=["temporary modification", "management of change"],
        equipment=["R-201", "L-2204", "P-101A"], types=["Process Upset", "Loss of Containment"], sev=0.52,
        template="A temporary modification on {name} {tag} was made without management of change review and caused abnormal flow."),
}

EXAMPLE_REPORT = (
    "On 2026-09-12 at 14:30 a hydrocarbon leak was detected at the seal of feed pump P-101A in Process Unit A. "
    "The operator isolated the pump and no personnel were hurt. Preventive maintenance on the pump seal was overdue "
    "and wear was not identified at the last inspection. Operating pressure exceeded the normal range for about "
    "ten minutes before the leak, and heavy rain reduced visibility during the response."
)

_FACTORS = {
    "human_factors": ["operator", "skipped", "inexperienced", "unfamiliar", "fatigue", "procedure"],
    "maintenance_factors": ["overdue", "maintenance", "inspection", "wear", "corrosion", "wall thinning"],
    "operational_factors": ["pressure", "startup", "shutdown", "alarm", "flow", "temperature"],
    "environmental_factors": ["rain", "wind", "heat", "humidity", "lightning", "storm"],
}
_HAZARDS = ["fire", "explosion", "toxic release", "leak", "release", "overpressure", "flammable", "hydrogen sulfide"]
_MATERIALS = ["hydrocarbon", "hydrogen sulfide", "ethylene", "propylene", "benzene", "naphtha", "ammonia", "methanol", "steam", "caustic"]
_EQUIP_WORDS = ["pump", "compressor", "reactor", "heat exchanger", "relief valve", "heater", "tank", "pipeline", "transmitter"]
_TAG_RE = re.compile(r"\b[A-Z]{1,3}-\d{2,4}[A-Z]?\b")


def severity_label(score: float) -> str:
    return "High" if score >= 0.65 else "Medium" if score >= 0.4 else "Low"


def build_history(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    end = date(2026, 9, 30)
    causes = list(ROOT_CAUSES)
    rows = []
    for _ in range(n):
        cause = rng.choices(causes, [26, 18, 12, 16, 12, 8, 8])[0]
        cfg = ROOT_CAUSES[cause]
        tag = rng.choice(cfg["equipment"]) if rng.random() < 0.7 else rng.choice(list(EQUIPMENT))
        name, unit = EQUIPMENT[tag]
        score = min(0.98, max(0.05, rng.gauss(cfg["sev"], 0.17)))
        rows.append(dict(date=end - timedelta(days=rng.randrange(730)), type=rng.choice(cfg["types"]),
                         equipment_tag=tag, equipment=f"{tag} ({name})", unit=unit,
                         severity=severity_label(score), severity_score=round(score, 2), root_cause=cause,
                         summary=cfg["template"].format(name=name, tag=tag)))
    rows.sort(key=lambda r: r["date"])
    seq: dict[int, int] = {}
    for r in rows:
        seq[r["date"].year] = seq.get(r["date"].year, 0) + 1
        r["id"] = f"INC-{r['date'].year}-{seq[r['date'].year]:04d}"
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _hits(low: str, words: list[str]) -> list[str]:
    return [w for w in words if w in low]


def mock_analyze(text: str) -> dict:
    """Keyword heuristics standing in for the NLP/ML pipeline.
    Evidence strings are real sentences copied from the submitted report;
    confidence values are demo heuristics, not model output."""
    low, sents = text.lower(), _sentences(text)
    tags = list(dict.fromkeys(_TAG_RE.findall(text)))
    words = _hits(low, _EQUIP_WORDS)
    equipment = [f"{t} ({EQUIPMENT[t][0]})" if t in EQUIPMENT else t for t in tags] or [w.capitalize() for w in words]
    causes = []
    for cause, cfg in ROOT_CAUSES.items():
        hit = _hits(low, cfg["kw"])
        if hit:
            ev = next((s for s in sents if any(h in s.lower() for h in hit)), None)
            causes.append(dict(cause=cause, confidence=min(0.92, 0.5 + 0.12 * len(hit)), evidence=ev,
                               supporting_factors=[cfg["factor"]] + (["Equipment"] if equipment else [])))
    causes = sorted(causes, key=lambda c: -c["confidence"])[:3]
    if "fire" in low or "explosion" in low: itype = "Fire / Explosion"
    elif _hits(low, ["injured", "injury", "fatal", "burn"]): itype = "Personnel Injury"
    elif "near miss" in low: itype = "Near Miss"
    elif _hits(low, ["leak", "release", "spill"]): itype = "Loss of Containment"
    elif _hits(low, ["upset", "trip", "abnormal"]): itype = "Process Upset"
    else: itype = "Equipment Failure"
    sev_hits = _hits(low, ["fire", "explosion", "injured", "injury", "fatal", "leak", "release", "pressure exceeded"])
    score = min(0.95, 0.25 + 0.3 * bool(_hits(low, ["fire", "explosion"])) + 0.25 * bool(_hits(low, ["injured", "injury", "fatal"]))
                + 0.12 * bool(_hits(low, ["leak", "release"])) + 0.1 * ("pressure exceeded" in low))
    d = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    t = re.search(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", text)
    loc = re.search(r"\b(Process Unit [A-Z0-9]+|Tank Farm|Utilities)\b", text)
    return {
        "incident": dict(id="DRAFT-" + hashlib.sha1(text.encode()).hexdigest()[:6].upper(), type=itype,
                         date=d.group(1) if d else None, time=t.group(0) if t else None,
                         location=loc.group(1) if loc else None, summary=" ".join(sents[:2])[:360]),
        "extracted_information": dict(equipment=equipment, hazards=[h.capitalize() for h in _hits(low, _HAZARDS)],
                                      materials=[m.capitalize() for m in _hits(low, _MATERIALS)],
                                      **{k: [w.capitalize() for w in _hits(low, v)] for k, v in _FACTORS.items()}),
        "root_causes": causes,
        "severity": dict(label=severity_label(score), score=round(score, 2), supporting_factors=[h.capitalize() for h in sev_hits]),
        "historical_analysis": {"similar_incidents": [], "same_equipment_incidents": [], "recurring_root_causes": []},
        "report_text": text,
    }


def _records(d: pd.DataFrame, cols: list[str]) -> list[dict]:
    d = d[cols].copy()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    return d.to_dict("records")


def mock_history(result: dict, df: pd.DataFrame) -> dict:
    tags = [e.split(" ")[0] for e in result["extracted_information"]["equipment"] if e.split(" ")[0] in EQUIPMENT]
    causes = [c["cause"] for c in result["root_causes"]]
    d = df.copy()
    d["similarity"] = 0.4 * (d["type"] == result["incident"]["type"]) + 0.4 * d["root_cause"].isin(causes) + 0.2 * d["equipment_tag"].isin(tags)
    cols = ["id", "date", "type", "equipment", "severity", "root_cause"]
    sim = d[d["similarity"] >= 0.6].sort_values(["similarity", "date"], ascending=False).head(5)
    recurring = []
    for c in causes:
        sub = df[df["root_cause"] == c]
        recurring.append(dict(cause=c, total=len(sub), on_same_equipment=int(sub["equipment_tag"].isin(tags).sum()),
                              last_seen=sub["date"].max().strftime("%Y-%m-%d") if len(sub) else None))
    return {"similar_incidents": [dict(r, similarity=round(s, 2)) for r, s in zip(_records(sim, cols), sim["similarity"])],
            "same_equipment_incidents": _records(df[df["equipment_tag"].isin(tags)].head(8), cols) if tags else [],
            "recurring_root_causes": recurring}
