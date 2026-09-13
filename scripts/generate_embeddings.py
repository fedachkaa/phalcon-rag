import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from phalcon_rag.embeddings import MODELS
from phalcon_rag.models import Chunk
from phalcon_rag.utils import load_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=MODELS.keys(),
        help="Embedding model to use",
    )

    return parser.parse_args()


def generate_embeddings(
    chunks: list[Chunk],
    model: SentenceTransformer,
) -> np.ndarray:
    return model.encode(
        [chunk.content for chunk in chunks],
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

    chunks = load_chunks(Path("data/processed/phalcon_docs_5.20.jsonl"))

    model = SentenceTransformer(config["model_name"])

    embeddings = generate_embeddings(chunks, model)

    save_embeddings(
        chunks,
        embeddings,
        Path(config["embeddings_path"]),
    )


if __name__ == "__main__":
    main()
