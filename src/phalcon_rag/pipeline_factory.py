from pathlib import Path

from phalcon_rag.config import (
    EMBEDDING_QUERY_INSTRUCTION,
    GENERATION_MODEL,
    SOURCE_CODE_QUERY_INSTRUCTION,
)
from phalcon_rag.generation.answer_generator import AnswerGenerator
from phalcon_rag.generation.query_rewriter import QueryRewriter
from phalcon_rag.models import SOURCE_PHALCON_DOCS, SOURCE_PHALCON_SOURCE_CODE
from phalcon_rag.pipeline import RagPipeline
from phalcon_rag.reranking.cross_encoder import CrossEncoderReranker
from phalcon_rag.retrieval.bm25 import BM25Retriever
from phalcon_rag.retrieval.dense import DenseRetriever
from phalcon_rag.retrieval.hybrid import HybridRetriever
from phalcon_rag.retrieval.multi_source import MultiSourceRetriever
from phalcon_rag.utils import load_retrieval_data


def create_pipeline() -> RagPipeline:
    instructions = {
        SOURCE_PHALCON_DOCS: EMBEDDING_QUERY_INSTRUCTION,
        SOURCE_PHALCON_SOURCE_CODE: SOURCE_CODE_QUERY_INSTRUCTION,
    }

    retrievers = {}
    for source in [SOURCE_PHALCON_DOCS, SOURCE_PHALCON_SOURCE_CODE]:
        chunks, embeddings = load_retrieval_data(
            chunks_path=Path(f"data/processed/{source}_5.20.jsonl"),
            embeddings_path=Path(f"data/embeddings/{source}/qwen_embeddings.npy"),
            chunk_ids_path=Path(f"data/embeddings/{source}/chunk_ids.json"),
        )

        retrievers[source] = HybridRetriever(
            DenseRetriever(chunks, embeddings, instructions[source]),
            BM25Retriever(chunks),
        )

    answer_generator = AnswerGenerator(GENERATION_MODEL)

    return RagPipeline(
        retriever=MultiSourceRetriever(
            retrievers[SOURCE_PHALCON_DOCS],
            retrievers[SOURCE_PHALCON_SOURCE_CODE],
        ),
        reranker=CrossEncoderReranker(),
        answer_generator=answer_generator,
        query_rewriter=QueryRewriter(GENERATION_MODEL)
    )
