from phalcon_rag.generation.prompt import build_context, build_prompt
from phalcon_rag.models import SOURCE_PHALCON_DOCS, Chunk


def test_build_context_numbers_chunks_and_includes_sources():
    chunks = [
        Chunk(
            id="chunk-1",
            content="First chunk content.",
            source=SOURCE_PHALCON_DOCS,
            metadata={
                "file_path": "models.mdx",
                "section": "Models",
            },
        ),
        Chunk(
            id="chunk-2",
            content="Second chunk content.",
            source=SOURCE_PHALCON_DOCS,
            metadata={
                "file_path": "transactions.mdx",
                "section": "Transactions",
            },
        ),
    ]

    context = build_context(chunks)

    assert "[1]" in context
    assert "Type: DOCUMENTATION" in context
    assert "File: models.mdx" in context
    assert "Section: Models" in context
    assert "First chunk content." in context

    assert "[2]" in context
    assert "File: transactions.mdx" in context
    assert "Section: Transactions" in context
    assert "Second chunk content." in context


def test_build_prompt_includes_question():
    query = "How do I create a transaction in Phalcon?"

    prompt = build_prompt(query, [])

    assert query in prompt


def test_build_prompt_includes_retrieved_context():
    chunk = Chunk(
        id="chunk-1",
        content="Transactions are created using the transaction manager.",
        source=SOURCE_PHALCON_DOCS,
        metadata={
            "file_path": "transactions.mdx",
            "section": "Transactions",
        },
    )

    prompt = build_prompt(
        "How do transactions work?",
        [chunk],
    )

    assert "[1]" in prompt
    assert "Type: DOCUMENTATION" in prompt
    assert "File: transactions.mdx" in prompt
    assert "Section: Transactions" in prompt
    assert "Transactions are created using the transaction manager." in prompt


def test_build_prompt_includes_selected_code():
    context = "$booking = Booking::findFirst($id);"

    prompt = build_prompt(
        query="What happens here?",
        chunks=[],
        context=context,
    )

    assert "Selected user code:" in prompt
    assert context in prompt
    assert "Treat selected user code as the code being analyzed" in prompt
