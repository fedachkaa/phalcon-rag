import re
from phalcon_rag.ingestion.models import Chunk
from .utils import _word_count, MAX_CHUNK_WORDS
from .factories import _create_method_chunk, _create_parent_chunk

API_METHOD_PATTERN = re.compile(r'^<h4 id="[^"]+"><code>(.+?)</code></h4>$')
MAX_API_ITEMS_PER_CHUNK = 10

def _contains_api_methods(
    content: str,
) -> bool:
    return any(
        API_METHOD_PATTERN.match(line)
        for line in content.splitlines()
    )

def _split_by_api_methods(
    chunk: Chunk,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    prefix_lines: list[str] = []
    current_lines: list[str] = []
    current_method: str | None = None

    for line in chunk.content.splitlines():
        method_match = API_METHOD_PATTERN.match(line)

        if method_match:
            if current_method is not None:
                chunks.append(
                    _create_method_chunk(
                        parent_chunk=chunk,
                        method_name=current_method,
                        lines=current_lines,
                    )
                )

            elif prefix_lines:
                prefix_content = "\n".join(prefix_lines).strip()

                if prefix_content:
                    chunks.append(
                        _create_parent_chunk(
                            parent_chunk=chunk,
                            lines=prefix_lines,
                        )
                    )

            current_method = method_match.group(1).strip()
            current_lines = [line]

            continue

        if current_method is None:
            prefix_lines.append(line)
        else:
            current_lines.append(line)

    if current_method is not None:
        chunks.append(
            _create_method_chunk(
                parent_chunk=chunk,
                method_name=current_method,
                lines=current_lines,
            )
        )

    return chunks

def _contains_api_items(
    content: str,
) -> bool:
    return (
        "<ApiList>" in content
        and "<ApiItem " in content
        and "</ApiItem>" in content
    )

def _split_by_api_items(
    chunk: Chunk,
) -> list[Chunk]:
    matches = list(
        re.finditer(
            r"<ApiItem\b.*?</ApiItem>",
            chunk.content,
            flags=re.DOTALL,
        )
    )

    if not matches:
        return [chunk]

    prefix = chunk.content[:matches[0].start()]
    api_items = [
        match.group(0).strip()
        for match in matches
    ]

    chunks: list[Chunk] = []
    current_items: list[str] = []
    part = 1

    for item in api_items:
        candidate = "\n\n".join(current_items + [item])

        candidate_content = (
            f"{prefix.strip()}\n\n"
            f"{candidate}\n\n"
            "</ApiList>"
        )

        if current_items and (
            len(current_items) >= MAX_API_ITEMS_PER_CHUNK
            or _word_count(candidate_content) > MAX_CHUNK_WORDS
        ):
            chunks.append(
                _create_api_items_chunk(
                    parent_chunk=chunk,
                    prefix=prefix,
                    items=current_items,
                    part=part,
                )
            )

            part += 1
            current_items = [item]
        else:
            current_items.append(item)
            
    if current_items:
        chunks.append(
            _create_api_items_chunk(
                parent_chunk=chunk,
                prefix=prefix,
                items=current_items,
                part=part,
            )
        )

    return chunks

def _create_api_items_chunk(
    parent_chunk: Chunk,
    prefix: str,
    items: list[str],
    part: int,
) -> Chunk:
    items_content = "\n\n".join(items)

    content = (
        f"{prefix.strip()}\n\n"
        f"{items_content}\n\n"
        "</ApiList>"
    )

    return Chunk(
        id=f"{parent_chunk.id}::api-items-{part}",
        content=content,
        source=parent_chunk.source,
        metadata={
            **parent_chunk.metadata,
            "api_items_part": part,
        },
    )


def _contains_method_annotations(
    content: str,
) -> bool:
    return sum(
        1
        for line in content.splitlines()
        if line.strip().startswith("@method ")
    ) > 1

def _split_method_annotations(
    chunk: Chunk,
) -> list[Chunk]:
    lines = [
        line
        for line in chunk.content.splitlines()
        if line.strip()
    ]

    method_lines = [
        line
        for line in lines
        if line.strip().startswith("@method ")
    ]

    if len(method_lines) <= 1:
        return [chunk]

    chunks: list[Chunk] = []
    current_methods: list[str] = []
    part = 1

    for method_line in method_lines:
        candidate = "\n".join([*current_methods, method_line])

        if (current_methods and _word_count(candidate) > MAX_CHUNK_WORDS):
            chunks.append(
                _create_method_annotations_chunk(
                    parent_chunk=chunk,
                    method_lines=current_methods,
                    part=part,
                )
            )

            part += 1
            current_methods = [method_line]

        else:
            current_methods.append(method_line)

    if current_methods:
        chunks.append(
            _create_method_annotations_chunk(
                parent_chunk=chunk,
                method_lines=current_methods,
                part=part,
            )
        )

    return chunks

def _create_method_annotations_chunk(
    parent_chunk: Chunk,
    method_lines: list[str],
    part: int,
) -> Chunk:
    return Chunk(
        id=(
            f"{parent_chunk.id}::"
            f"method-annotations-{part}"
        ),
        content="\n".join(method_lines),
        source=parent_chunk.source,
        metadata={
            **parent_chunk.metadata,
             "method_annotations_part": part,
        },
    )