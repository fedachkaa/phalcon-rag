from pathlib import Path

from phalcon_rag.utils import load_json

bm25_results_path = Path("data/evaluation/results/bm25.json")
qwen_results_path = Path("data/evaluation/results/qwen.json")

bm25_results = load_json(bm25_results_path)
qwen_results = load_json(qwen_results_path)

bm25_results_prep = {item["id"]: item["rank"] for item in bm25_results["results"]}

qwen_results_prep = {item["id"]: item["rank"] for item in qwen_results["results"]}

metrics = {
    "qwen_better": 0,
    "bm25_better": 0,
    "equal": 0,
}

for question_id, value in bm25_results_prep.items():
    qwen_rank = qwen_results_prep[question_id]
    bm25_rank = value

    qwen_compare_rank = qwen_rank if qwen_rank is not None else float("inf")
    bm25_compare_rank = bm25_rank if bm25_rank is not None else float("inf")

    if qwen_compare_rank < bm25_compare_rank:
        metrics["qwen_better"] += 1
    elif qwen_compare_rank > bm25_compare_rank:
        metrics["bm25_better"] += 1
    else:
        metrics["equal"] += 1

    print(f"{question_id} qwen={qwen_compare_rank} bm25={bm25_compare_rank}")


print(metrics)
