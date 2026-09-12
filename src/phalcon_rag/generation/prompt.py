from phalcon_rag.models import Chunk


def build_context(chunks: list[Chunk]) -> str:
    context = ""

    for pos, chunk in enumerate(chunks, start=1):
        context += (
            f"[{pos}]\n"
            f"Source: {chunk.source}\n"
            f"Content:\n{chunk.content}\n\n"
        )

    return context


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    return f"""You are a Phalcon documentation assistant. Answer the user's question using only the provided context.

Rules:
    - Do not use outside knowledge.
    - If the context is insufficient, say that you cannot answer from the provided documentation.
    - Cite supporting sources using [1], [2], etc.
    - Prefer concise technical answers.
    - Include code examples only if supported by the context.

    Question:
    {query}

    Context:
    {build_context(chunks)}"""