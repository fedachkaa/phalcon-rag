from phalcon_rag.retrieval.bm25 import BM25Retriever
from pathlib import Path
from phalcon_rag.utils import load_chunks, load_json, save_result


processed_data_path = Path("data/processed/phalcon_docs_5.20.jsonl")
retrieval_questions_path = Path("data/evaluation/retrieval_questions.json")

processed_data = load_chunks(processed_data_path)
retrieval_questions = load_json(retrieval_questions_path)

retriever = BM25Retriever(processed_data)

results = []
recall_at_5 = 0
recall_at_10 = 0
reciprocal_rank_sum = 0

for question in retrieval_questions:
    search_results = retriever.search(question['query'], top_k=len(processed_data))

    relevant_rank = None

    for rank, (chunk, score) in enumerate(search_results, start=1):
        if chunk.id in question["relevant_chunk_ids"]:
            relevant_rank = rank
            break

    if relevant_rank is not None:
        if relevant_rank <= 5:
            recall_at_5 += 1

        if relevant_rank <= 10:
            recall_at_10 += 1

        reciprocal_rank_sum += 1 / relevant_rank

    results.append({
        "id": question["id"],
        "query": question["query"],
        "category": question["category"],
        "relevant_chunk_ids": question["relevant_chunk_ids"],
        "rank": relevant_rank,
    })


questions_count = len(retrieval_questions)

metrics = {
    "recall_at_5": recall_at_5/ questions_count,
    "recall_at_10": recall_at_10 / questions_count,
    "mrr": reciprocal_rank_sum / questions_count,
}

evaluation_result = {
    "model": "bm25",
    "questions_count": questions_count,
    "metrics": metrics,
    "results": results,
}

save_result("bm25", evaluation_result)