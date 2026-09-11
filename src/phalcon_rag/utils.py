from pathlib import Path
from phalcon_rag.models import Chunk
import json
import numpy as np


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []

    with path.open(mode="r", encoding="utf-8") as file:
        for line in file:
            chunks.append(Chunk(**json.loads(line)))

    return chunks


def load_json(path: Path):
    with path.open(mode="r", encoding="utf-8") as file:
        return json.load(file)


def save_result(model_key: str, result: dict) -> None:
    results_dir = Path("data/evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{model_key}.json"

    with results_path.open(mode="w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved:", results_path)


def is_retrieval_noise(chunk: Chunk) -> bool:
    content = chunk.content.strip()

    return content == (
        ":::info[NOTE]\n"
        "All classes are prefixed with `Phalcon`\n"
        ":::"
    )


def load_retrieval_data(
    chunks_path: Path,
    embeddings_path: Path,
    chunk_ids_path: Path,
) -> tuple[list[Chunk], np.ndarray]:
    chunks = load_chunks(chunks_path)
    embedding_chunk_ids = load_json(chunk_ids_path)

    current_chunk_ids = [
        chunk.id
        for chunk in chunks
    ]

    assert embedding_chunk_ids == current_chunk_ids

    embeddings = np.load(embeddings_path)

    assert len(embeddings) == len(chunks)

    filtered_indices = [
        index
        for index, chunk in enumerate(chunks)
        if not is_retrieval_noise(chunk)
    ]

    filtered_chunks = [
        chunks[index]
        for index in filtered_indices
    ]

    filtered_embeddings = embeddings[filtered_indices]

    return filtered_chunks, filtered_embeddings
