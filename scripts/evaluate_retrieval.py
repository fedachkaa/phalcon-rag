from pathlib import Path

from phalcon_rag.pipeline_factory import create_pipeline
from phalcon_rag.utils import load_json

questions = load_json(Path("data/evaluation/questions.json"))
pipeline = create_pipeline()

hit_total = {}
expected_source_recall = {}

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

    hit_total[case["id"]] = int(bool(found_sources))

    expected_source_recall[case["id"]] = {
        "found": len(found_sources),
        "total": len(case["expected_sources"]),
    }

    print(f"Hit@7 = {hit_total[case['id']]}")

hits = sum(hit_total.values())
hit_rate = hits / len(questions)

print(f"Hit@7: {hits}/{len(questions)} ({hit_rate * 100:.1f}%)")


recalls = []

for case_id, result in expected_source_recall.items():
    recall = result["found"] / result["total"]
    recalls.append(recall)

    print(f"{case_id}: {result['found']}/{result['total']} ({recall * 100:.1f}%)")

mean_recall = sum(recalls) / len(recalls)

print(f"Mean Expected Source Recall@7: {mean_recall * 100:.1f}%")
