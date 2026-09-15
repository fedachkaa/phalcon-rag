from pathlib import Path

from phalcon_rag.models import SOURCE_PHALCON_DOCS
from phalcon_rag.retrieval.bm25 import BM25Retriever
from phalcon_rag.retrieval.dense import DenseRetriever
from phalcon_rag.retrieval.hybrid import HybridRetriever
from phalcon_rag.utils import load_json, load_retrieval_data, save_result

retrieval_questions = load_json(Path("data/evaluation/retrieval_questions.json"))


chunks, embeddings = load_retrieval_data(
    chunks_path=Path("data/processed/phalcon_docs_5.20.jsonl"),
    embeddings_path=Path(f"data/embeddings/{SOURCE_PHALCON_DOCS}/qwen_embeddings.npy"),
    chunk_ids_path=Path(f"data/embeddings/{SOURCE_PHALCON_DOCS}/chunk_ids.json"),
)

dense_retriever = DenseRetriever(chunks, embeddings)
bm25_retriever = BM25Retriever(chunks)
hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)

results = []

recall_at_5 = 0
recall_at_10 = 0
reciprocal_rank_sum = 0

for question in retrieval_questions:
    fused_results = hybrid_retriever.search(question["query"])

    relevant_rank = None

    for rank, (chunk, score) in enumerate(fused_results, start=1):
        if chunk.id in question["relevant_chunk_ids"]:
            relevant_rank = rank
            break

    if relevant_rank is not None:
        if relevant_rank <= 5:
            recall_at_5 += 1

        if relevant_rank <= 10:
            recall_at_10 += 1

        reciprocal_rank_sum += 1 / relevant_rank

    results.append(
        {
            "id": question["id"],
            "query": question["query"],
            "category": question["category"],
            "relevant_chunk_ids": question["relevant_chunk_ids"],
            "rank": relevant_rank,
        }
    )

questions_count = len(retrieval_questions)

metrics = {
    "recall_at_5": recall_at_5 / questions_count,
    "recall_at_10": recall_at_10 / questions_count,
    "mrr": reciprocal_rank_sum / questions_count,
}

evaluation_result = {
    "model": "qwen_bm25_rrf",
    "questions_count": questions_count,
    "metrics": metrics,
    "results": results,
}

save_result("qwen_bm25_rrf", evaluation_result)
