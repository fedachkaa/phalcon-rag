from sentence_transformers import CrossEncoder

from phalcon_rag.config import RERANKER_MODEL
from phalcon_rag.models import SOURCE_PHALCON_DOCS, SOURCE_PHALCON_SOURCE_CODE, Chunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []

        pairs = [(query, self.get_reranking_text(chunk)) for chunk, _ in candidates]

        scores = self.model.predict(pairs)

        reranked = [
            (chunk, float(score)) for (chunk, _), score in zip(candidates, scores)
        ]

        return sorted(
            reranked,
            key=lambda item: item[1],
            reverse=True,
        )

    def get_reranking_text(self, chunk: Chunk) -> str:
        if chunk.source == SOURCE_PHALCON_DOCS:
            return chunk.content
        elif chunk.source == SOURCE_PHALCON_SOURCE_CODE:
            return (
                f"File: {chunk.metadata['file']}\n"
                f"Method: {chunk.metadata['method']}\n\n"
                f"{chunk.content}"
            )

        return ""
