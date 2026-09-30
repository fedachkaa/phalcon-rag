from unittest.mock import Mock

import pytest

from phalcon_rag.pipeline import RagPipeline


@pytest.fixture
def pipeline():
    return RagPipeline(
        retriever=Mock(),
        reranker=Mock(),
        answer_generator=Mock(),
        query_rewriter=Mock(),
    )


def test_run_without_context_does_not_rewrite_query(pipeline):
    pipeline.retrieve = Mock(return_value=[])
    pipeline.query_rewriter.rewrite = Mock()
    pipeline.answer_generator.generate = Mock(return_value="answer")

    pipeline.run("How does findFirst work?")

    pipeline.query_rewriter.rewrite.assert_not_called()
    pipeline.retrieve.assert_called_once_with("How does findFirst work?")


def test_run_with_context_uses_rewritten_query(pipeline):
    context = "$booking = Booking::findFirst($id);"

    pipeline.query_rewriter.rewrite = Mock(
        return_value="Phalcon Model findFirst internal implementation"
    )
    pipeline.retrieve = Mock(return_value=[])
    pipeline.answer_generator.generate = Mock(return_value="answer")

    pipeline.run(
        query="What happens here?",
        context=context,
    )

    pipeline.query_rewriter.rewrite.assert_called_once_with(
        query="What happens here?",
        context=context,
    )

    pipeline.retrieve.assert_called_once_with(
        "Phalcon Model findFirst internal implementation"
    )


def test_run_passes_original_query_and_context_to_generator(pipeline):
    context = "$booking = Booking::findFirst($id);"

    pipeline.query_rewriter.rewrite = Mock(
        return_value="Phalcon Model findFirst internal implementation"
    )
    pipeline.retrieve = Mock(return_value=[])

    pipeline.answer_generator.generate = Mock(return_value="answer")

    pipeline.run(
        query="What happens here?",
        context=context,
    )

    pipeline.answer_generator.generate.assert_called_once_with(
        "What happens here?",
        [],
        context=context,
    )
