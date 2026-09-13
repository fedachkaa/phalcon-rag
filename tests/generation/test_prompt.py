from phalcon_rag.generation.prompt import build_context, build_prompt
from phalcon_rag.models import Chunk


def test_build_context_numbers_chunks_and_includes_sources():
    chunks = [
        Chunk(
            id="chunk-1",
            content="First chunk content.",
            source="models.md",
        ),
        Chunk(
            id="chunk-2",
            content="Second chunk content.",
            source="transactions.md",
        ),
    ]

    context = build_context(chunks)

    assert "[1]" in context
    assert "Source: models.md" in context
    assert "First chunk content." in context

    assert "[2]" in context
    assert "Source: transactions.md" in context
    assert "Second chunk content." in context


def test_build_prompt_includes_question():
    query = "How do I create a transaction in Phalcon?"

    prompt = build_prompt(query, [])

    assert query in prompt


def test_build_prompt_includes_retrieved_context():
    chunk = Chunk(
        id="chunk-1",
        content="Transactions are created using the transaction manager.",
        source="transactions.md",
    )

    prompt = build_prompt(
        "How do transactions work?",
        [chunk],
    )

    assert "[1]" in prompt
    assert "Source: transactions.md" in prompt
    assert "Transactions are created using the transaction manager." in prompt
