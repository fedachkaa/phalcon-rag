import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from phalcon_rag.embeddings import MODELS
from phalcon_rag.evaluation.embeddings import evaluate_model
from phalcon_rag.models import SOURCE_PHALCON_DOCS
from phalcon_rag.utils import load_json, load_retrieval_data, save_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=MODELS.keys(),
        help="Embedding model to evaluate",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MODELS[args.model]

    retrieval_questions = load_json(Path("data/evaluation/retrieval_questions.json"))

    chunks, embeddings = load_retrieval_data(
        chunks_path=Path("data/processed/phalcon_docs_5.20.jsonl"),
        embeddings_path=Path(
            config["embeddings_path"].format(source=SOURCE_PHALCON_DOCS)
        ),
        chunk_ids_path=Path(f"data/embeddings/{SOURCE_PHALCON_DOCS}/chunk_ids.json"),
    )

    model = SentenceTransformer(config["model_name"])

    result = evaluate_model(
        args.model,
        retrieval_questions,
        chunks,
        embeddings,
        model,
    )

    save_result(args.model, result)


if __name__ == "__main__":
    main()
