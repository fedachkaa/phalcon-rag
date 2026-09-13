from phalcon_rag.models import Chunk

from .headings import HEADING_PATTERN
from .utils import MAX_CHUNK_WORDS, _word_count


def _split_oversized_chunk(
    chunk: Chunk,
) -> list[Chunk]:
    blocks = _split_into_blocks(chunk.content)
    blocks = _merge_heading_blocks(blocks)

    chunks: list[Chunk] = []

    current_blocks: list[str] = []
    current_word_count = 0
    part_number = 1

    for block in blocks:
        block_word_count = _word_count(block)

        if current_blocks and (current_word_count + block_word_count > MAX_CHUNK_WORDS):
            chunks.append(
                _create_fallback_chunk(
                    parent_chunk=chunk,
                    blocks=current_blocks,
                    part_number=part_number,
                )
            )

            part_number += 1
            current_blocks = []
            current_word_count = 0

        current_blocks.append(block)
        current_word_count += block_word_count

    if current_blocks:
        chunks.append(
            _create_fallback_chunk(
                parent_chunk=chunk,
                blocks=current_blocks,
                part_number=part_number,
            )
        )

    return chunks


def _merge_heading_blocks(
    blocks: list[str],
) -> list[str]:
    merged: list[str] = []
    pending_heading: str | None = None

    for block in blocks:
        stripped = block.strip()

        if HEADING_PATTERN.fullmatch(stripped):
            if pending_heading:
                pending_heading += "\n\n" + stripped
            else:
                pending_heading = stripped
            continue

        if pending_heading:
            block = f"{pending_heading}\n\n{block}"
            pending_heading = None

        merged.append(block)

    if pending_heading:
        merged.append(pending_heading)

    return merged


def _create_fallback_chunk(
    parent_chunk: Chunk,
    blocks: list[str],
    part_number: int,
) -> Chunk:
    content = "\n\n".join(blocks).strip()

    return Chunk(
        id=(f"{parent_chunk.id}::part-{part_number}"),
        content=content,
        source=parent_chunk.source,
        metadata={
            **parent_chunk.metadata,
            "part": part_number,
        },
    )


def _split_into_blocks(
    content: str,
) -> list[str]:
    lines = content.splitlines()

    blocks: list[str] = []
    current_lines: list[str] = []

    in_code_block = False

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("```"):
            current_lines.append(line)

            if in_code_block:
                in_code_block = False

                block = "\n".join(current_lines).strip()

                if block:
                    blocks.append(block)

                current_lines = []
            else:
                in_code_block = True

            continue

        if in_code_block:
            current_lines.append(line)
            continue

        if not stripped_line:
            if current_lines:
                block = "\n".join(current_lines).strip()

                if block:
                    blocks.append(block)

                current_lines = []

            continue

        current_lines.append(line)

    if current_lines:
        block = "\n".join(current_lines).strip()

        if block:
            blocks.append(block)

    return blocks
