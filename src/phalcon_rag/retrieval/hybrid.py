from phalcon_rag.config import BM25_WEIGHT, DENSE_WEIGHT, RRF_K
from phalcon_rag.retrieval.bm25 import BM25Retriever
from phalcon_rag.retrieval.dense import DenseRetriever
from phalcon_rag.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    def search(
        self,
        query: str,
        candidate_k: int = 50,
        top_k: int = 10,
    ):
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)

        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)

        fused = reciprocal_rank_fusion(
            [(dense_results, DENSE_WEIGHT), (bm25_results, BM25_WEIGHT)], k=RRF_K
        )

        return fused[:top_k]
