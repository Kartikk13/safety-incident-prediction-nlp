
from services.nlp_extractor import extract_incident_features

# Expected outputs represent the intended behavior of the improved extractor.
# This script does not modify the extractor or any saved models.

TEST_CASES = [
    {
        "name": "Known canonical phrases",
        "report": "The pump was damaged.",
        "expected": {
            "equipment_type": "Pump",
            "equipment_status": "Damaged",
        },
    },
    {
        "name": "Paraphrased equipment and status",
        "report": "The pumping unit was badly impaired.",
        "expected": {
            "equipment_type": "Pump",
            "equipment_status": "Malfunctioning",
        },
    },
    {
        "name": "Unfamiliar equipment",
        "report": "The hydraulic actuator failed during operation.",
        "expected": {
            "equipment_type": None,
        },
    },
    {
        "name": "Explicitly negated hazard",
        "report": "No oil leakage was observed near the pump.",
        "expected": {
            "equipment_type": "Pump",
            "hazard_type": None,
        },
    },
    {
        "name": "Resolved issue explicitly excluded from incident",
        "report": (
            "The oil leak had been repaired before this report "
            "and was not part of the incident."
        ),
        "expected": {
            "hazard_type": None,
        },
    },
    {
        "name": "Clear department paraphrase",
        "report": "The maintenance crew inspected the unit.",
        "expected": {
            "department": "Maintenance",
        },
    },
    {
        "name": "Ambiguous department wording",
        "report": "The mechanics checked the unit after the incident.",
        "expected": {
            "department": None,
        },
    },
    {
        "name": "Multiple equipment items and conflicting statuses",
        "report": (
            "The pump was not damaged, but a nearby conveyor "
            "was malfunctioning."
        ),
        "expected": {
            "equipment_type": "Conveyor",
            "equipment_status": "Malfunctioning",
        },
    },
    
    {
        "name": "Negation variation: no oil leak",
        "report": "There was no oil leak near the pump.",
        "expected": {
            "hazard_type": None,
        },
    },
    {
        "name": "Negation variation: leakage absent",
        "report": "Oil leakage was absent from the work area.",
        "expected": {
            "hazard_type": None,
        },
    },
    {
        "name": "Historical hazard with exact category wording",
        "report": "The oil leakage was repaired yesterday.",
        "expected": {
            "hazard_type": None,
        },
    },
    {
        "name": "Positive equipment paraphrase",
        "report": "The pumping unit was badly impaired.",
        "expected": {
            "equipment_type": "Pump",
            "equipment_status": "Malfunctioning",
        },
    },
    {
        "name": "Negated status with another damaged item",
        "report": (
            "The pump was not damaged, but the conveyor "
            "was damaged."
        ),
        "expected": {
            "equipment_type": "Conveyor",
            "equipment_status": "Damaged",
        },
    },
]

total_checks = 0
passed_checks = 0
failed_checks = []

for index, case in enumerate(TEST_CASES, start=1):
    actual = extract_incident_features(case["report"])

    print(f"\n{'=' * 65}")
    print(f"TEST {index}: {case['name']}")
    print(f"Report: {case['report']}")

    for field, expected_value in case["expected"].items():
        actual_value = actual.get(field)
        passed = actual_value == expected_value

        total_checks += 1
        if passed:
            passed_checks += 1
        else:
            failed_checks.append(
                {
                    "test": case["name"],
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )

        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {field}: "
            f"expected={expected_value!r}, actual={actual_value!r}"
        )

print(f"\n{'=' * 65}")
print("EVALUATION SUMMARY")
print(f"Passed: {passed_checks}/{total_checks}")
print(f"Failed: {len(failed_checks)}/{total_checks}")

if failed_checks:
    print("\nFAILURES TO INVESTIGATE:")
    for failure in failed_checks:
        print(
            f"- {failure['test']} | {failure['field']} | "
            f"expected={failure['expected']!r} | "
            f"actual={failure['actual']!r}"
        )
