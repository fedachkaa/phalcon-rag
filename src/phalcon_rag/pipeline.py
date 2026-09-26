from phalcon_rag.config import (
    CANDIDATE_K,
    HYBRID_TOP_K,
    RERANK_DOCS_TOP_K,
    RERANK_SOURCE_TOP_K,
)
from phalcon_rag.generation.answer_generator import AnswerGenerator
from phalcon_rag.models import SOURCE_PHALCON_DOCS, SOURCE_PHALCON_SOURCE_CODE, Chunk
from phalcon_rag.reranking.cross_encoder import CrossEncoderReranker
from phalcon_rag.retrieval.base import Retriever


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: CrossEncoderReranker,
        answer_generator: AnswerGenerator,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.answer_generator = answer_generator

    def retrieve(self, query: str) -> list[Chunk]:
        candidates = self.retriever.search(
            query,
            candidate_k=CANDIDATE_K,
            top_k=HYBRID_TOP_K,
        )

        docs_candidates = [
            candidate
            for candidate in candidates
            if candidate[0].source == SOURCE_PHALCON_DOCS
        ]

        source_candidates = [
            candidate
            for candidate in candidates
            if candidate[0].source == SOURCE_PHALCON_SOURCE_CODE
        ]

        docs_reranked = self.reranker.rerank(query, docs_candidates)
        source_reranked = self.reranker.rerank(query, source_candidates)

        selected = (
            docs_reranked[:RERANK_DOCS_TOP_K] + source_reranked[:RERANK_SOURCE_TOP_K]
        )

        return [chunk for chunk, _ in selected]

    def answer(self, query: str) -> str:
        chunks = self.retrieve(query)

        return self.answer_generator.generate(query, chunks)
