from typing import Protocol

from phalcon_rag.models import Chunk


class Retriever(Protocol):
    def search(
        self,
        query: str,
        candidate_k: int = 50,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]: ...
