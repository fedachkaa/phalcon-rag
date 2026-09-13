from phalcon_rag.models import Chunk

from .utils import _slugify


def _create_method_chunk(
    parent_chunk: Chunk,
    method_name: str,
    lines: list[str],
) -> Chunk:
    content = "\n".join(lines).strip()

    heading_path = [
        *parent_chunk.metadata.get(
            "heading_path",
            [],
        ),
        method_name,
    ]

    return Chunk(
        id=(f"{parent_chunk.id}::{_slugify(method_name)}"),
        content=content,
        source=parent_chunk.source,
        metadata={
            **parent_chunk.metadata,
            "section": method_name,
            "heading_path": heading_path,
            "method": method_name,
        },
    )


def _create_parent_chunk(
    parent_chunk: Chunk,
    lines: list[str],
) -> Chunk:
    return Chunk(
        id=parent_chunk.id,
        content="\n".join(lines).strip(),
        source=parent_chunk.source,
        metadata={
            **parent_chunk.metadata,
        },
    )
