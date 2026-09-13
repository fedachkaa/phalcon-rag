import argparse
from pathlib import Path
from dotenv import load_dotenv

from phalcon_rag.config import GENERATION_MODEL
from phalcon_rag.generation.answer_generator import AnswerGenerator
from phalcon_rag.pipeline import RagPipeline
from phalcon_rag.reranking.cross_encoder import CrossEncoderReranker
from phalcon_rag.retrieval.bm25 import BM25Retriever
from phalcon_rag.retrieval.dense import DenseRetriever
from phalcon_rag.retrieval.hybrid import HybridRetriever
from phalcon_rag.utils import load_retrieval_data

load_dotenv()


def answer_question(question: str) -> str:
    chunks, embeddings = load_retrieval_data(
        chunks_path=Path("data/processed/phalcon_docs_5.20.jsonl"),
        embeddings_path=Path("data/embeddings/qwen_embeddings.npy"),
        chunk_ids_path=Path("data/embeddings/chunk_ids.json"),
    )

    pipeline = RagPipeline(
        hybrid_retriever=HybridRetriever(
            DenseRetriever(chunks, embeddings), BM25Retriever(chunks)
        ),
        reranker=CrossEncoderReranker(),
        answer_generator=AnswerGenerator(GENERATION_MODEL),
    )

    return pipeline.answer(question)


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question about Phalcon documentation."
    )
    parser.add_argument("question", help="Question to answer.")
    args = parser.parse_args()

    result = answer_question(args.question)

    print(result)


if __name__ == "__main__":
    main()
