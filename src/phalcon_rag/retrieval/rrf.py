
from phalcon_rag.models import Chunk


def reciprocal_rank_fusion(
    rankings: list[tuple[list[tuple[Chunk, float]], float]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    scores = {}
    chunks_by_id = {}

    for ranking, weight in rankings:
        for rank, (chunk, _) in enumerate(ranking, start=1):
            chunks_by_id[chunk.id] = chunk

            if chunk.id not in scores:
                scores[chunk.id] = 0

            scores[chunk.id] += weight / (k + rank)

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    return [
        (chunks_by_id[chunk_id], score)
        for chunk_id, score in sorted_scores
    ]
