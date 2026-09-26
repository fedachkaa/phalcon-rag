from pathlib import Path

from phalcon_rag.pipeline_factory import create_pipeline
from phalcon_rag.utils import load_json

questions = load_json(Path("data/evaluation/questions.json"))
pipeline = create_pipeline()

hit_at_5_total = {}
expected_source_recall_at_5 = {}

for case in questions:
    chunks = pipeline.retrieve(case["question"])
    found_sources = set()

    print(f"\nQuestion: {case['question']}")

    for chunk in chunks:
        for expected_source in case["expected_sources"]:
            if chunk.id == expected_source or chunk.id.startswith(
                expected_source + "::"
            ):
                found_sources.add(expected_source)

        print(chunk.id)

    hit_at_5_total[case["id"]] = int(bool(found_sources))

    expected_source_recall_at_5[case["id"]] = {
        "found": len(found_sources),
        "total": len(case["expected_sources"]),
    }

    print(f"Hit@5 = {hit_at_5_total[case['id']]}")

# Hit@5

hits = sum(hit_at_5_total.values())
hit_at_5 = hits / len(questions)

print(f"\nHit@5: {hits}/{len(questions)} ({hit_at_5 * 100:.1f}%)")

# Expected Source Recall@5

recalls = []

print("\nExpected Source Recall@5:")

for case_id, result in expected_source_recall_at_5.items():
    recall = result["found"] / result["total"]
    recalls.append(recall)

    print(f"{case_id}: {result['found']}/{result['total']} ({recall * 100:.1f}%)")

mean_recall = sum(recalls) / len(recalls)

print(f"\nMean Expected Source Recall@5: {mean_recall * 100:.1f}%")
