from sentence_transformers import CrossEncoder
from phalcon_rag.models import Chunk


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        
        pairs = [
            (query, chunk.content)
            for chunk, _ in candidates
        ]

        scores = self.model.predict(pairs)

        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(candidates, scores)
        ]

        return sorted(
            reranked,
            key=lambda item: item[1],
            reverse=True,
        )