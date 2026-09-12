from pathlib import Path
from phalcon_rag.retrieval.hybrid import HybridRetriever
from phalcon_rag.retrieval.dense import DenseRetriever
from phalcon_rag.retrieval.bm25 import BM25Retriever
from phalcon_rag.reranking.cross_encoder import CrossEncoderReranker
from phalcon_rag.utils import load_json, load_retrieval_data, save_result


retrieval_questions = load_json(Path("data/evaluation/retrieval_questions.json"))

chunks, embeddings = load_retrieval_data(
    chunks_path=Path("data/processed/phalcon_docs_5.20.jsonl"),
    embeddings_path=Path("data/embeddings/qwen_embeddings.npy"),
    chunk_ids_path=Path("data/embeddings/chunk_ids.json"),
)

reranker_model_name = "BAAI/bge-reranker-v2-m3"

dense_retriever = DenseRetriever(chunks, embeddings)
bm25_retriever = BM25Retriever(chunks)
hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
reranker = CrossEncoderReranker(reranker_model_name)

results = []
recall_at_1 = 0
recall_at_3 = 0
recall_at_5 = 0
recall_at_10 = 0
reciprocal_rank_sum = 0

for question in retrieval_questions:
    candidates = hybrid_retriever.search(
        question["query"],
        top_k=50,
    )

    candidates = candidates[:10]

    reranked = reranker.rerank(
        question["query"],
        candidates,
    )

    relevant_rank = None

    for rank, (chunk, score) in enumerate(reranked, start=1):
        if chunk.id in question["relevant_chunk_ids"]:
            relevant_rank = rank
            break
    
    if relevant_rank is not None:
        if relevant_rank == 1:
            recall_at_1 += 1
    
        if relevant_rank <= 3:
            recall_at_3 += 1

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
    "recall_at_1": recall_at_1 / questions_count,
    "recall_at_3": recall_at_3 / questions_count,
    "recall_at_5": recall_at_5 / questions_count,
    "recall_at_10": recall_at_10 / questions_count,
    "mrr": reciprocal_rank_sum / questions_count,
}

evaluation_result = {
    "model": reranker_model_name,
    "retrieval": "qwen_bm25_rrf_4_1",
    "questions_count": questions_count,
    "metrics": metrics,
    "results": results,
}

save_result("reranker", evaluation_result)
