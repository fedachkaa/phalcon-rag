import pytest

from phalcon_rag.config import RRF_K
from phalcon_rag.models import Chunk
from phalcon_rag.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_returns_results_sorted_by_score():
    chunk_a = Chunk(id="a", content="A", source="docs")
    chunk_b = Chunk(id="b", content="B", source="docs")

    rankings = [
        (
            [
                (chunk_a, 0.9),
                (chunk_b, 0.8),
            ],
            1.0,
        )
    ]

    result = reciprocal_rank_fusion(rankings, RRF_K)

    assert result[0][0].id == "a"
    assert result[1][0].id == "b"


def test_rrf_merges_same_chunk_from_multiple_rankings():
    chunk_a = Chunk(id="a", content="A", source="docs")

    rankings = [
        ([(chunk_a, 0.9)], 1.0),
        ([(chunk_a, 0.8)], 1.0),
    ]

    result = reciprocal_rank_fusion(rankings, RRF_K)

    assert len(result) == 1
    assert result[0][0].id == "a"
    assert result[0][1] == pytest.approx(2 / 61)


def test_rrf_respects_ranking_weights():
    chunk_a = Chunk(id="a", content="A", source="docs")
    chunk_b = Chunk(id="b", content="B", source="docs")

    rankings = [
        ([(chunk_a, 0.9)], 1.0),
        ([(chunk_b, 0.9)], 4.0),
    ]

    result = reciprocal_rank_fusion(rankings, RRF_K)

    assert result[0][0].id == "b"
    assert result[1][0].id == "a"
    assert result[0][1] == pytest.approx(4 / 61)
    assert result[1][1] == pytest.approx(1 / 61)


def test_rrf_returns_empty_list_for_empty_rankings():
    rankings = [([], 1.0), ([], 3.0)]

    result = reciprocal_rank_fusion(rankings, RRF_K)

    assert result == []
