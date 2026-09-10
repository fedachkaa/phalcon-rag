from phalcon_rag.ingestion.models import Chunk
import bm25s

class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

        corpus = [
            chunk.content
            for chunk in chunks
        ]

        corpus_tokens = bm25s.tokenize(corpus)

        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)


    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        query_tokens = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(query_tokens, k = top_k)

        return [
            (self.chunks[index], float(score))
            for index, score in zip(results[0], scores[0])
        ]
