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
        top_k: int = 50,
    ):
        qwen_results = self.dense_retriever.search(query, top_k=top_k)

        bm25_results = self.bm25_retriever.search(query, top_k=top_k,)

        return reciprocal_rank_fusion([
            (qwen_results, 4.0),
            (bm25_results, 1.0)
        ])
    