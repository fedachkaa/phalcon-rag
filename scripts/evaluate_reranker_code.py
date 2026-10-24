from pathlib import Path

from phalcon_rag.config import SOURCE_CODE_QUERY_INSTRUCTION
from phalcon_rag.models import SOURCE_PHALCON_SOURCE_CODE, Chunk
from phalcon_rag.reranking.cross_encoder import CrossEncoderReranker
from phalcon_rag.retrieval.bm25 import BM25Retriever
from phalcon_rag.retrieval.dense import DenseRetriever
from phalcon_rag.retrieval.hybrid import HybridRetriever
from phalcon_rag.utils import load_json, load_retrieval_data, save_result


retrieval_questions = load_json(
    Path("data/evaluation/questions.json")
)

chunks, embeddings = load_retrieval_data(
    chunks_path=Path("data/processed/phalcon_source_code_5.20.jsonl"),
    embeddings_path=Path(
        f"data/embeddings/{SOURCE_PHALCON_SOURCE_CODE}/qwen_embeddings.npy"
    ),
    chunk_ids_path=Path(
        f"data/embeddings/{SOURCE_PHALCON_SOURCE_CODE}/chunk_ids.json"
    ),
)

reranker_model_name = "BAAI/bge-reranker-base"

dense_retriever = DenseRetriever(
    chunks,
    embeddings,
    SOURCE_CODE_QUERY_INSTRUCTION,
)
bm25_retriever = BM25Retriever(chunks)
hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
reranker = CrossEncoderReranker(reranker_model_name)


def get_method_id(chunk: Chunk) -> str:
    return f"{chunk.metadata['file']}::{chunk.metadata['method']}"


results = []

hit_at_1 = 0
hit_at_3 = 0
hit_at_5 = 0
hit_at_10 = 0

expected_source_recall_at_1_sum = 0
expected_source_recall_at_3_sum = 0
expected_source_recall_at_5_sum = 0
expected_source_recall_at_10_sum = 0

reciprocal_rank_sum = 0
evaluated_questions_count = 0


for question in retrieval_questions:
    expected_sources = {
        source
        for source in question["expected_sources"]
        if ".zep::" in source
    }

    if not expected_sources:
        continue

    evaluated_questions_count += 1

    query = question["question"]

    candidates = hybrid_retriever.search(
        query,
        candidate_k=50,
        top_k=10,
    )

    reranked = reranker.rerank(
        query,
        candidates,
    )

    source_ranks = {}

    for rank, (chunk, _) in enumerate(reranked, start=1):
        method_id = get_method_id(chunk)

        if method_id in expected_sources and method_id not in source_ranks:
            source_ranks[method_id] = rank

    relevant_ranks = sorted(source_ranks.values())
    first_relevant_rank = relevant_ranks[0] if relevant_ranks else None

    if first_relevant_rank is not None:
        if first_relevant_rank <= 1:
            hit_at_1 += 1

        if first_relevant_rank <= 3:
            hit_at_3 += 1

        if first_relevant_rank <= 5:
            hit_at_5 += 1

        if first_relevant_rank <= 10:
            hit_at_10 += 1

        reciprocal_rank_sum += 1 / first_relevant_rank

    top_1_sources = {
        get_method_id(chunk)
        for chunk, _ in reranked[:1]
    }

    top_3_sources = {
        get_method_id(chunk)
        for chunk, _ in reranked[:3]
    }

    top_5_sources = {
        get_method_id(chunk)
        for chunk, _ in reranked[:5]
    }

    top_10_sources = {
        get_method_id(chunk)
        for chunk, _ in reranked[:10]
    }

    expected_source_recall_at_1 = (
        len(expected_sources & top_1_sources) / len(expected_sources)
    )
    expected_source_recall_at_3 = (
        len(expected_sources & top_3_sources) / len(expected_sources)
    )
    expected_source_recall_at_5 = (
        len(expected_sources & top_5_sources) / len(expected_sources)
    )
    expected_source_recall_at_10 = (
        len(expected_sources & top_10_sources) / len(expected_sources)
    )

    expected_source_recall_at_1_sum += expected_source_recall_at_1
    expected_source_recall_at_3_sum += expected_source_recall_at_3
    expected_source_recall_at_5_sum += expected_source_recall_at_5
    expected_source_recall_at_10_sum += expected_source_recall_at_10

    results.append(
        {
            "id": question["id"],
            "question": query,
            "category": question["category"],
            "expected_sources": sorted(expected_sources),
            "found_sources": source_ranks,
            "first_relevant_rank": first_relevant_rank,
            "relevant_ranks": relevant_ranks,
            "expected_source_recall_at_1": expected_source_recall_at_1,
            "expected_source_recall_at_3": expected_source_recall_at_3,
            "expected_source_recall_at_5": expected_source_recall_at_5,
            "expected_source_recall_at_10": expected_source_recall_at_10,
            "top_10": [
                {
                    "id": chunk.id,
                    "method_id": get_method_id(chunk),
                    "score": score,
                }
                for chunk, score in reranked[:10]
            ],
        }
    )

if evaluated_questions_count == 0:
    raise ValueError("No source-code evaluation questions found")

metrics = {
    "hit_at_1": hit_at_1 / evaluated_questions_count,
    "hit_at_3": hit_at_3 / evaluated_questions_count,
    "hit_at_5": hit_at_5 / evaluated_questions_count,
    "hit_at_10": hit_at_10 / evaluated_questions_count,
    "mean_expected_source_recall_at_1":
        expected_source_recall_at_1_sum / evaluated_questions_count,
    "mean_expected_source_recall_at_3":
        expected_source_recall_at_3_sum / evaluated_questions_count,
    "mean_expected_source_recall_at_5":
        expected_source_recall_at_5_sum / evaluated_questions_count,
    "mean_expected_source_recall_at_10":
        expected_source_recall_at_10_sum / evaluated_questions_count,
    "mrr": reciprocal_rank_sum / evaluated_questions_count,
}

evaluation_result = {
    "model": reranker_model_name,
    "retrieval": "qwen_bm25_rrf_4_1",
    "source": SOURCE_PHALCON_SOURCE_CODE,
    "questions_count": evaluated_questions_count,
    "skipped_questions_count": (
        len(retrieval_questions) - evaluated_questions_count
    ),
    "metrics": metrics,
    "results": results,
}

save_result("source_reranker", evaluation_result)