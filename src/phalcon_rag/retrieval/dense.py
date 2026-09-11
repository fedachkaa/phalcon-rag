from sentence_transformers import SentenceTransformer
from phalcon_rag.models import Chunk
import numpy as np


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ):
        self.chunks = chunks
        self.embeddings = embeddings

        self.model_config = {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "query_instruction": "Retrieve relevant Phalcon documentation passages that answer the technical question."
        }

        self.model = SentenceTransformer(
            self.model_config["model_name"]
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        formatted_query = (
            f'Instruct: {self.model_config["query_instruction"]}\n'
            f"Query: {query}"
        )

        query_embedding = self.model.encode(formatted_query, normalize_embeddings=True)

        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self.chunks[index], float(scores[index]))
            for index in ranked_indices
        ]