from phalcon_rag.config import CANDIDATE_K, HYBRID_TOP_K, RERANK_TOP_K
from phalcon_rag.generation.answer_generator import AnswerGenerator
from phalcon_rag.models import Chunk
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

        reranked = self.reranker.rerank(query, candidates)

        return [chunk for chunk, _ in reranked[:RERANK_TOP_K]]

    def answer(self, query: str) -> str:
        chunks = self.retrieve(query)

        return self.answer_generator.generate(query, chunks)
