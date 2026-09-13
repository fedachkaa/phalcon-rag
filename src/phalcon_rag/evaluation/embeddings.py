import numpy as np
from sentence_transformers import SentenceTransformer

from phalcon_rag.embeddings import MODELS, format_query
from phalcon_rag.models import Chunk


def evaluate_model(
    model_key: str,
    retrieval_questions: list,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    model: SentenceTransformer,
) -> dict:
    results = []

    recall_at_5 = 0
    recall_at_10 = 0
    reciprocal_rank_sum = 0

    for question in retrieval_questions:
        query = question["query"]
        formatted_query = format_query(model_key, query)
        expected_ids = question["relevant_chunk_ids"]

        query_embedding = model.encode(
            formatted_query,
            normalize_embeddings=True,
        )

        scores = embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1]

        relevant_rank = None

        for rank, index in enumerate(ranked_indices, start=1):
            if chunks[index].id in expected_ids:
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
                "query": query,
                "category": question["category"],
                "relevant_chunk_ids": expected_ids,
                "rank": relevant_rank,
            }
        )

    questions_count = len(retrieval_questions)

    metrics = {
        "recall_at_5": recall_at_5 / questions_count,
        "recall_at_10": recall_at_10 / questions_count,
        "mrr": reciprocal_rank_sum / questions_count,
    }

    return {
        "model": model_key,
        "model_name": MODELS[model_key]["model_name"],
        "questions_count": questions_count,
        "metrics": metrics,
        "results": results,
    }
