from phalcon_rag.models import Chunk

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
            candidate_k=50,
            top_k=10,
        )

        reranked = self.reranker.rerank(query, candidates)

        return [chunk for chunk, _ in reranked[:5]]

    def answer(self, query: str) -> str:
        chunks = self.retrieve(query)

        return self.answer_generator.generate(query, chunks)
