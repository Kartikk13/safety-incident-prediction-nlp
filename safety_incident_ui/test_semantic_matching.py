
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Diagnostic cases only. Similarity is not a probability.
TEST_CASES = [
    {
        "name": "Equipment paraphrase",
        "text": "The pumping unit was badly impaired.",
        "candidates": ["Pump", "Conveyor", "Forklift"],
        "expected": "Pump",
    },
    {
        "name": "Status paraphrase",
        "text": "The pumping unit was badly impaired.",
        "candidates": [
            "Malfunctioning", "Damaged",
            "Operational", "Under Maintenance",
        ],
        "expected": "Malfunctioning",
    },
    {
        "name": "Explicit positive hazard",
        "text": "Oil was leaking onto the floor near the pump.",
        "candidates": [
            "Oil Leakage", "Slippery Surface",
            "Chemical Substance",
        ],
        "expected": "Oil Leakage",
    },
    {
        "name": "Negated hazard",
        "text": "No oil leakage was observed near the pump.",
        "candidates": [
            "Oil Leakage", "Slippery Surface",
            "Chemical Substance",
        ],
        "expected": None,
    },
    {
        "name": "Hazard explicitly absent",
        "text": "Oil leakage was absent from the work area.",
        "candidates": [
            "Oil Leakage", "Slippery Surface",
            "Chemical Substance",
        ],
        "expected": None,
    },
    {
        "name": "Historical hazard repaired",
        "text": "The oil leakage was repaired yesterday.",
        "candidates": [
            "Oil Leakage", "Slippery Surface",
            "Chemical Substance",
        ],
        "expected": None,
    },
    {
        "name": "Unfamiliar equipment",
        "text": "The hydraulic actuator failed during operation.",
        "candidates": ["Pump", "Conveyor", "Forklift"],
        "expected": None,
    },
    {
        "name": "Canonical equipment",
        "text": "The conveyor was damaged.",
        "candidates": ["Pump", "Conveyor", "Forklift"],
        "expected": "Conveyor",
    },
    {
        "name": "Ambiguous equipment",
        "text": "A machine failed during operation.",
        "candidates": ["Pump", "Conveyor", "Forklift"],
        "expected": None,
    },
    {
        "name": "Clear department paraphrase",
        "text": "The maintenance crew inspected the unit.",
        "candidates": [
            "Maintenance", "Production",
            "Electrical", "Warehouse & Logistics",
        ],
        "expected": "Maintenance",
    },
]


def main():
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    for case in TEST_CASES:
        text_embedding = model.encode(
            case["text"],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        candidate_embeddings = model.encode(
            case["candidates"],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        scores = cos_sim(
            text_embedding, candidate_embeddings
        )[0].tolist()

        ranked = sorted(
            zip(case["candidates"], scores),
            key=lambda item: item[1],
            reverse=True,
        )

        top_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score

        print("\n" + "=" * 60)
        print(f"Test: {case['name']}")
        print(f"Text: {case['text']}")
        print(f"Expected: {case['expected']}")
        print(f"Top candidate: {top_label}")
        print(f"Top similarity: {top_score:.4f}")
        print(f"Top-two margin: {margin:.4f}")

        for label, score in ranked:
            print(f"  {label}: {score:.4f}")

        if case["expected"] is None:
            print(
                "Semantic-only decision: SHOULD REMAIN UNKNOWN "
                "(requires rejection logic)"
            )
        else:
            print(
                "Top candidate matches expected:",
                top_label == case["expected"],
            )


if __name__ == "__main__":
    main()
