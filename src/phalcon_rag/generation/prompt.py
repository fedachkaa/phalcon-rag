from phalcon_rag.models import SOURCE_PHALCON_DOCS, SOURCE_PHALCON_SOURCE_CODE, Chunk


def build_context(chunks: list[Chunk]) -> str:
    context = ""

    for pos, chunk in enumerate(chunks, start=1):
        context += f"[{pos}]\n{prepare_chunk(chunk)}\n\n"

    return context


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    return f"""You are a technical assistant for the Phalcon framework. Answer the user's question using only the provided context.

There are two types of evidence:
    - DOCUMENTATION: public API / documented behavior / usage
    - SOURCE CODE: internal implementation / execution flow

Rules:
    - Do not use outside knowledge.
    - If the context is insufficient, say that you cannot answer from the provided context.
    - Cite supporting sources using [1], [2], etc.
    - Prefer concise technical answers.
    - Include code examples only if supported by the context.
    - When the question asks about public API, documented behavior, or usage, prioritize relevant DOCUMENTATION context.
    - When the question asks how something works internally, prioritize relevant SOURCE CODE context and use it to explain the implementation flow.
    - When both documentation and source code are relevant, combine them to explain both documented behavior and internal implementation.

    Question:
    {query}

    Context:
    {build_context(chunks)}"""


def prepare_chunk(chunk: Chunk) -> str:
    if chunk.source == SOURCE_PHALCON_DOCS:
        return (
            f"Type: DOCUMENTATION\n"
            f"File: {chunk.metadata['file_path']}\n"
            f"Section: {chunk.metadata['section']}\n"
            f"Content: {chunk.content}"
        )
    elif chunk.source == SOURCE_PHALCON_SOURCE_CODE:
        return (
            f"Type: SOURCE CODE\n"
            f"File: {chunk.metadata['file']}\n"
            f"Method: {chunk.metadata['method']}\n"
            f"Lines: {chunk.metadata['start_line']}-{chunk.metadata['end_line']}\n"
            f"Content:\n{chunk.content}"
        )
    return ""
