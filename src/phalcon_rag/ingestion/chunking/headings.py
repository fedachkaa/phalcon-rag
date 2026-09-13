import re

from phalcon_rag.models import Chunk, Document

from .utils import _make_unique_id, _slugify

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


def _split_by_headings(
    document: Document,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    id_counts: dict[str, int] = {}

    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_heading_path: list[str] = []

    for line in document.content.splitlines():
        heading_match = HEADING_PATTERN.match(line)

        if heading_match:
            if current_lines:
                chunks.append(
                    _create_chunk(
                        document=document,
                        lines=current_lines,
                        heading_path=current_heading_path,
                        id_counts=id_counts,
                    )
                )

            heading_level = len(heading_match.group(1))
            heading_title = heading_match.group(2).strip()

            while heading_stack and heading_stack[-1][0] >= heading_level:
                heading_stack.pop()

            heading_stack.append((heading_level, heading_title))

            current_heading_path = [title for _, title in heading_stack]

            current_lines = [line]

            continue

        current_lines.append(line)

    if current_lines:
        chunks.append(
            _create_chunk(
                document=document,
                lines=current_lines,
                heading_path=current_heading_path,
                id_counts=id_counts,
            )
        )

    return chunks


def _create_chunk(
    document: Document,
    lines: list[str],
    heading_path: list[str],
    id_counts: dict[str, int],
) -> Chunk:
    content = "\n".join(lines).strip()

    section = heading_path[-1] if heading_path else None

    base_id = _create_chunk_id(
        document_id=document.id,
        heading_path=heading_path,
    )

    chunk_id = _make_unique_id(
        base_id=base_id,
        id_counts=id_counts,
    )

    return Chunk(
        id=chunk_id,
        content=content,
        source=document.source,
        metadata={
            **document.metadata,
            "section": section,
            "heading_path": heading_path.copy(),
        },
    )


def _create_chunk_id(
    document_id: str,
    heading_path: list[str],
) -> str:
    if not heading_path:
        return f"{document_id}::root"

    path = "::".join(_slugify(heading) for heading in heading_path)

    return f"{document_id}::{path}"
