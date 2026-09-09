import re
from phalcon_rag.ingestion.models import Chunk
from .utils import _word_count, MAX_CHUNK_WORDS
from .factories import _create_method_chunk, _create_parent_chunk

METHOD_SIGNATURE_PATTERN = re.compile(
    r"^(?:final\s+)?"
    r"(?:public|protected|private)\s+"
    r"(?:static\s+)?function\s+"
    r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
)


def _split_by_method_signatures(
    chunk: Chunk,
) -> list[Chunk]:
    method_blocks = _find_method_signature_blocks(chunk.content)

    if not method_blocks:
        return [chunk]

    lines = chunk.content.splitlines()

    chunks: list[Chunk] = []

    first_method_start = method_blocks[0][0]

    if first_method_start > 0:
        prefix_lines = lines[
            :first_method_start
        ]

        prefix_content = "\n".join(prefix_lines).strip()

        if prefix_content:
            chunks.append(
                _create_parent_chunk(
                    parent_chunk=chunk,
                    lines=prefix_lines,
                )
            )

    for index, method_block in enumerate(
        method_blocks
    ):
        method_start, _, method_name = method_block

        if index + 1 < len(method_blocks):
            method_end = method_blocks[index + 1][0]
        else:
            method_end = len(lines)

        method_lines = lines[
            method_start:method_end
        ]

        chunks.append(
            _create_method_chunk(
                parent_chunk=chunk,
                method_name=method_name,
                lines=method_lines,
            )
        )

    return chunks

def _contains_method_signatures(
    content: str,
) -> bool:
    return bool(_find_method_signature_blocks(content))

def _find_method_signature_blocks(
    content: str,
) -> list[tuple[int, int, str]]:
    lines = content.splitlines()

    method_blocks: list[tuple[int, int, str]] = []

    in_php_block = False
    block_start = 0
    block_lines: list[str] = []

    for index, line in enumerate(lines):
        stripped_line = line.strip()

        if (not in_php_block and stripped_line == "```php"):
            in_php_block = True
            block_start = index
            block_lines = []
            continue

        if (in_php_block and stripped_line == "```"):
            method_name = _extract_method_name(block_lines)

            if method_name:
                method_blocks.append((block_start, index, method_name))

            in_php_block = False
            block_lines = []

            continue

        if in_php_block:
            block_lines.append(line)

    return method_blocks

def _extract_method_name(
    lines: list[str],
) -> str | None:
    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            continue

        if stripped_line == "<?php":
            return None

        match = METHOD_SIGNATURE_PATTERN.match(stripped_line)

        if match:
            return match.group(1)

        return None
    
    return None

def _contains_multiple_method_signatures(content: str,) -> bool:
    lines = content.splitlines()

    method_count = 0

    for line in lines:
        if METHOD_SIGNATURE_PATTERN.match(line.strip()):
            method_count += 1

    return method_count > 1

def _split_method_signature_list(
    chunk: Chunk,
) -> list[Chunk]:
    lines = chunk.content.splitlines()

    method_blocks: list[str] = []
    current_lines: list[str] = []

    inside_php_block = False

    for line in lines:
        stripped = line.strip()

        if stripped == "```php":
            inside_php_block = True
            continue

        if stripped == "```":
            inside_php_block = False
            continue

        if not inside_php_block:
            continue

        if (METHOD_SIGNATURE_PATTERN.match(stripped) and current_lines):
            method_blocks.append("\n".join(current_lines).strip())
            current_lines = []

        current_lines.append(line)

    if current_lines:
        method_blocks.append("\n".join(current_lines).strip())

    if len(method_blocks) <= 1:
        return [chunk]

    chunks: list[Chunk] = []
    current_methods: list[str] = []
    part = 1

    for method_block in method_blocks:
        candidate_methods = [
            *current_methods,
            method_block,
        ]

        candidate_content = (
            "```php\n"
            + "\n\n".join(candidate_methods)
            + "\n```"
        )

        if (current_methods and _word_count(candidate_content) > MAX_CHUNK_WORDS):
            chunks.append(
                _create_method_list_chunk(
                    parent_chunk=chunk,
                    method_blocks=current_methods,
                    part=part,
                )
            )

            part += 1
            current_methods = [method_block]

        else:
            current_methods.append(method_block)

    if current_methods:
        chunks.append(
            _create_method_list_chunk(
                parent_chunk=chunk,
                method_blocks=current_methods,
                part=part,
            )
        )

    return chunks

def _create_method_list_chunk(
    parent_chunk: Chunk,
    method_blocks: list[str],
    part: int,
) -> Chunk:
    methods_content = "\n\n".join(method_blocks)

    content = (
        "```php\n"
        f"{methods_content}\n"
        "```"
    )

    return Chunk(
        id=(
            f"{parent_chunk.id}::"
            f"method-signatures-{part}"
        ),
        content=content,
        source=parent_chunk.source,
        metadata={
            **parent_chunk.metadata,
            "method_signatures_part": part,
        },
    )
