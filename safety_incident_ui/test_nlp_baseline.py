
from services.nlp_extractor import extract_incident_features

TEST_CASES = [
    {
        "name": "Known canonical phrases",
        "report": "The pump was damaged.",
    },
    {
        "name": "Paraphrased equipment and status",
        "report": "The pumping unit was badly impaired.",
    },
    {
        "name": "Unfamiliar equipment",
        "report": "The hydraulic actuator failed during operation.",
    },
    {
        "name": "Negated hazard",
        "report": "No oil leakage was observed near the pump.",
    },
    {
        "name": "Historical hazard",
        "report": (
            "The operator inspected the pump after the "
            "oil leakage had been repaired."
        ),
    },
    {
        "name": "Department paraphrase",
        "report": "The mechanics checked the unit after the incident.",
    },
    {
        "name": "Conflicting equipment and status",
        "report": (
            "The pump was not damaged, but a nearby conveyor "
            "was malfunctioning."
        ),
    },
]

FIELDS = [
    "department",
    "equipment_type",
    "equipment_status",
    "incident_type",
    "hazard_type",
]

for index, case in enumerate(TEST_CASES, start=1):
    result = extract_incident_features(case["report"])

    print(f"\n{'=' * 60}")
    print(f"TEST {index}: {case['name']}")
    print(f"Report: {case['report']}")

    for field in FIELDS:
        print(f"{field}: {result.get(field)}")
    