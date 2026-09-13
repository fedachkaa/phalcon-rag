from phalcon_rag.models import Chunk

from .config import CANDIDATE_K, HYBRID_TOP_K, RERANK_TOP_K
from .generation.answer_generator import AnswerGenerator
from .reranking.cross_encoder import CrossEncoderReranker
from .retrieval.hybrid import HybridRetriever


class RagPipeline:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        answer_generator: AnswerGenerator,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.answer_generator = answer_generator

    def retrieve(self, query: str) -> list[Chunk]:
        candidates = self.hybrid_retriever.search(
            query,
            candidate_k=CANDIDATE_K,
            top_k=HYBRID_TOP_K,
        )

        reranked = self.reranker.rerank(query, candidates)

        return [chunk for chunk, _ in reranked[:RERANK_TOP_K]]

    def answer(self, query: str) -> str:
        chunks = self.retrieve(query)

        return self.answer_generator.generate(query, chunks)
