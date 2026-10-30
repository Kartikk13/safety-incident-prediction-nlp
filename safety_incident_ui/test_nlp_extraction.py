from services.nlp_extractor import extract_incident_features

reports = [
    "On 12 October 2024 at 5:25 PM, a maintenance technician "
    "on the night shift observed oil on the floor near the pump.",

    "A worker observed oil on the floor near the pump.",

    "On 12 October 2024, a worker observed oil on the floor "
    "near the pump at 09:15.",
]

for i, report in enumerate(reports, start=1):
    result = extract_incident_features(report)
    print(f"\nTest {i}")
    for field in (
        "incident_year",
        "incident_month",
        "incident_dayofweek",
        "incident_hour",
        "incident_minute",
        "shift",
        "hazard_type",
    ):
        print(f"{field}: {result.get(field)}")