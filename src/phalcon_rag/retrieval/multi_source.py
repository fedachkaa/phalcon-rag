from phalcon_rag.config import RRF_K
from phalcon_rag.models import Chunk
from phalcon_rag.retrieval.hybrid import HybridRetriever
from phalcon_rag.retrieval.rrf import reciprocal_rank_fusion


class MultiSourceRetriever:
    def __init__(
        self,
        docs_retriever: HybridRetriever,
        source_retriever: HybridRetriever,
    ):
        self.docs_retriever = docs_retriever
        self.source_retriever = source_retriever

    def search(
        self,
        query: str,
        candidate_k: int = 50,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        docs_results = self.docs_retriever.search(
            query, candidate_k=candidate_k, top_k=top_k
        )
        source_results = self.source_retriever.search(
            query, candidate_k=candidate_k, top_k=top_k
        )

        fused = reciprocal_rank_fusion(
            [
                (docs_results, 1.0),
                (source_results, 1.0),
            ],
            k=RRF_K,
        )

        return fused[:top_k]
