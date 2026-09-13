import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from phalcon_rag.models import Chunk
from phalcon_rag.utils import load_json, load_retrieval_data, save_result

MODELS = {
    "qwen": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "embeddings_path": "data/embeddings/qwen_embeddings.npy",
        "query_instruction": (
            "Retrieve relevant Phalcon documentation passages "
            "that answer the technical question."
        ),
    },
    "bge": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "embeddings_path": "data/embeddings/bge_embeddings.npy",
        "query_prefix": ("Represent this sentence for searching relevant passages: "),
    },
    "gte": {
        "model_name": "Alibaba-NLP/gte-modernbert-base",
        "embeddings_path": "data/embeddings/gte_embeddings.npy",
    },
    "e5": {
        "model_name": "intfloat/multilingual-e5-large-instruct",
        "embeddings_path": "data/embeddings/e5_embeddings.npy",
        "query_instruction": (
            "Retrieve relevant technical documentation that answers "
            "the user's question."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=MODELS.keys(),
        help="Embedding model to evaluate",
    )

    return parser.parse_args()


def format_query(model_key: str, query: str) -> str:
    config = MODELS[model_key]

    if model_key == "bge":
        return config["query_prefix"] + query

    if model_key in {"e5", "qwen"}:
        return f"Instruct: {config['query_instruction']}\nQuery: {query}"

    return query


def debug_question(
    question_id: str,
    model_key: str,
    retrieval_questions: list,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    model: SentenceTransformer,
) -> None:
    question = next(item for item in retrieval_questions if item["id"] == question_id)

    query = format_query(model_key, question["query"])
    expected_ids = question["relevant_chunk_ids"]

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    scores = embeddings @ query_embedding
    ranked_indices = np.argsort(scores)[::-1]

    print("QUERY:", question["query"])
    print("EXPECTED:", expected_ids)
    print()

    for rank, index in enumerate(ranked_indices[:10], start=1):
        chunk = chunks[index]

        print(f"#{rank} score={scores[index]:.4f}")
        print("ID:", chunk.id)
        print(chunk.content[:500])
        print()

    for expected_id in expected_ids:
        expected_chunk = next(chunk for chunk in chunks if chunk.id == expected_id)

        print("EXPECTED CHUNK:")
        print(expected_chunk.id)
        print(expected_chunk.content[:1000])


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


def main() -> None:
    args = parse_args()
    model_key = args.model

    retrieval_questions = load_json(Path("data/evaluation/retrieval_questions.json"))

    chunks, embeddings = load_retrieval_data(
        chunks_path=Path("data/processed/phalcon_docs_5.20.jsonl"),
        embeddings_path=Path(MODELS[model_key]["embeddings_path"]),
        chunk_ids_path=Path("data/embeddings/chunk_ids.json"),
    )

    print("Model:", MODELS[model_key]["model_name"])
    print("Embeddings shape:", embeddings.shape)
    print("Chunks:", len(chunks))

    model = SentenceTransformer(MODELS[model_key]["model_name"])

    result = evaluate_model(
        model_key,
        retrieval_questions,
        chunks,
        embeddings,
        model,
    )

    print()
    print("Metrics:")
    print(result["metrics"])

    save_result(model_key, result)


if __name__ == "__main__":
    main()
