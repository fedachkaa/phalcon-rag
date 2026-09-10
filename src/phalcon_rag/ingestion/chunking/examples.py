from phalcon_rag.models import Chunk
from .utils import _word_count, MAX_CHUNK_WORDS

def _split_query_builder_examples(
    chunk: Chunk,
) -> list[Chunk]:
    lines = chunk.content.splitlines()

    example_blocks: list[str] = []
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

        if (stripped.startswith("// SELECT") and current_lines):
            example_blocks.append("\n".join(current_lines).strip())
            current_lines = []

        current_lines.append(line)

    if current_lines:
        example_blocks.append("\n".join(current_lines).strip())

    if len(example_blocks) <= 1:
        return [chunk]

    return _group_example_blocks(
        parent_chunk=chunk,
        example_blocks=example_blocks,
    )

def _group_example_blocks(
    parent_chunk: Chunk,
    example_blocks: list[str],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_examples: list[str] = []
    part = 1

    for example in example_blocks:
        candidate_examples = [
            *current_examples,
            example,
        ]

        candidate_content = (
            "### Examples\n\n"
            "```php\n"
            + "\n\n".join(candidate_examples)
            + "\n```"
        )

        if (current_examples and _word_count(candidate_content) > MAX_CHUNK_WORDS):
            chunks.append(
                _create_examples_chunk(
                    parent_chunk=parent_chunk,
                    examples=current_examples,
                    part=part,
                )
            )

            part += 1
            current_examples = [example]

        else:
            current_examples.append(example)

    if current_examples:
        chunks.append(
            _create_examples_chunk(
                parent_chunk=parent_chunk,
                examples=current_examples,
                part=part,
            )
        )

    return chunks

def _create_examples_chunk(
    parent_chunk: Chunk,
    examples: list[str],
    part: int,
) -> Chunk:
    examples_content = "\n\n".join(examples)

    content = (
        "### Examples\n\n"
        "```php\n"
        f"{examples_content}\n"
        "```"
    )

    return Chunk(
        id=f"{parent_chunk.id}::examples-{part}",
        content=content,
        source=parent_chunk.source,
        metadata={
            **parent_chunk.metadata,
            "examples_part": part,
        },
    )

def _is_query_builder_examples(
    chunk: Chunk,
) -> bool:
    return (
        chunk.metadata.get("section") == "Examples"
        and chunk.content.count("// SELECT") > 1
        and "$builder" in chunk.content
    )