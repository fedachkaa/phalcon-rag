import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from phalcon_rag.embeddings import MODELS
from phalcon_rag.models import Chunk, SOURCE_PHALCON_SOURCE_CODE, SOURCE_PHALCON_DOCS
from phalcon_rag.utils import load_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODELS.keys(), help="Embedding model to use")
    parser.add_argument("--source", type=str, required=True, choices=[SOURCE_PHALCON_DOCS, SOURCE_PHALCON_SOURCE_CODE], help="Chunk source")

    return parser.parse_args()

def get_embedding_text(chunk: Chunk) -> str:
    if chunk.source == SOURCE_PHALCON_SOURCE_CODE:
        return (
            f"File: {chunk.metadata['file']}\n"
            f"Method: {chunk.metadata['method']}\n\n"
            f"{chunk.content}"
        )

    return chunk.content


def generate_embeddings(
    chunks: list[Chunk],
    model: SentenceTransformer,
) -> np.ndarray:
    return model.encode(
        [get_embedding_text(chunk) for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def save_embeddings(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    embeddings_path: Path,
) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_path, embeddings)

    chunk_ids_path = embeddings_path.parent / "chunk_ids.json"

    with chunk_ids_path.open("w", encoding="utf-8") as file:
        json.dump(
            [chunk.id for chunk in chunks],
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved embeddings:", embeddings_path)
    print("Saved chunk IDs:", chunk_ids_path)


def main() -> None:
    args = parse_args()
    config = MODELS[args.model]

    chunks = load_chunks(Path(f"data/processed/{args.source}_5.20.jsonl"))

    model = SentenceTransformer(config["model_name"])

    embeddings = generate_embeddings(chunks, model)

    save_embeddings(
        chunks,
        embeddings,
        Path(config["embeddings_path"].format(source=args.source))
    )


if __name__ == "__main__":
    main()
