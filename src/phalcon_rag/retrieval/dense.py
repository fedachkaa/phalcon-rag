import numpy as np
from sentence_transformers import SentenceTransformer

from phalcon_rag.config import EMBEDDING_MODEL, EMBEDDING_QUERY_INSTRUCTION
from phalcon_rag.models import Chunk


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ):
        self.chunks = chunks
        self.embeddings = embeddings

        self.model_config = {
            "model_name": EMBEDDING_MODEL,
            "query_instruction": EMBEDDING_QUERY_INSTRUCTION,
        }

        self.model = SentenceTransformer(self.model_config["model_name"])

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        formatted_query = (
            f"Instruct: {self.model_config['query_instruction']}\nQuery: {query}"
        )

        query_embedding = self.model.encode(formatted_query, normalize_embeddings=True)

        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[index], float(scores[index])) for index in ranked_indices]
