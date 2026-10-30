
from hybrid_nlp_prototype import extract_hybrid_features

TEST_CASES = [
    {
        "name": "Canonical pump",
        "report": "The pump was damaged.",
        "expected": {
            "equipment_type": "Pump",
            "equipment_status": "Damaged",
        },
    },
    {
        "name": "Negated oil leakage",
        "report": "No oil leakage was observed near the pump.",
        "expected": {"hazard_type": None},
    },
    {
        "name": "Absent oil leakage",
        "report": "Oil leakage was absent from the work area.",
        "expected": {"hazard_type": None},
    },
    {
        "name": "Repaired oil leakage",
        "report": "The oil leakage was repaired yesterday.",
        "expected": {"hazard_type": None},
    },
    {
        "name": "Unknown equipment",
        "report": "The hydraulic actuator failed during operation.",
        "expected": {"equipment_type": None},
    },
    {
        "name": "Ambiguous equipment",
        "report": "A machine failed during operation.",
        "expected": {"equipment_type": None},
    },
    {
        "name": "Equipment paraphrase",
        "report": "The pumping unit was badly impaired.",
        "expected": {
            "equipment_type": "Pump",
            "equipment_status": "Malfunctioning",
        },
    },
]

checks = 0
passed = 0

for case in TEST_CASES:
    result = extract_hybrid_features(case["report"])

    print(f"\nTEST: {case['name']}")
    print(f"Report: {case['report']}")

    for field, expected in case["expected"].items():
        actual = result.get(field)
        ok = actual == expected
        checks += 1
        passed += int(ok)

        print(
            f"[{'PASS' if ok else 'FAIL'}] {field}: "
            f"expected={expected!r}, actual={actual!r}"
        )

print(f"\nSUMMARY: {passed}/{checks} checks passed")
