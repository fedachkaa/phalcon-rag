import re

from phalcon_rag.models import Chunk

from .factories import _create_parent_chunk
from .utils import MAX_CHUNK_WORDS, _slugify, _word_count


def _split_large_markdown_table(
    chunk: Chunk,
) -> list[Chunk]:
    lines = chunk.content.splitlines()

    if len(lines) < 3:
        return [chunk]

    header = lines[0]
    separator = lines[1]
    rows = lines[2:]

    if not (header.strip().startswith("|") and separator.strip().startswith("|")):
        return [chunk]

    chunks: list[Chunk] = []
    current_rows: list[str] = []
    part = 1

    for row in rows:
        candidate = "\n".join(
            [
                header,
                separator,
                *current_rows,
                row,
            ]
        )

        if current_rows and _word_count(candidate) > MAX_CHUNK_WORDS:
            chunks.append(
                _create_table_chunk(
                    parent_chunk=chunk,
                    header=header,
                    separator=separator,
                    rows=current_rows,
                    part=part,
                )
            )

            part += 1
            current_rows = [row]

        else:
            current_rows.append(row)

    if current_rows:
        chunks.append(
            _create_table_chunk(
                parent_chunk=chunk,
                header=header,
                separator=separator,
                rows=current_rows,
                part=part,
            )
        )

    return chunks


def _create_table_chunk(
    parent_chunk: Chunk,
    header: str,
    separator: str,
    rows: list[str],
    part: int,
) -> Chunk:
    content = "\n".join(
        [
            header,
            separator,
            *rows,
        ]
    ).strip()

    return Chunk(
        id=f"{parent_chunk.id}::table-{part}",
        content=content,
        source=parent_chunk.source,
        metadata={
            **parent_chunk.metadata,
            "table_part": part,
        },
    )


def _is_markdown_table(
    content: str,
) -> bool:
    lines = [line for line in content.splitlines() if line.strip()]

    if len(lines) < 3:
        return False

    return lines[0].strip().startswith("|") and lines[1].strip().startswith("|")


def _contains_events_table(
    chunk: Chunk,
) -> bool:
    section = chunk.metadata.get("section")

    if section != "List of Events":
        return False

    return (
        "| Component" in chunk.content
        and "| Event" in chunk.content
        and "| Parameters" in chunk.content
    )


def _split_events_table(
    chunk: Chunk,
) -> list[Chunk]:
    lines = chunk.content.splitlines()

    table_rows: list[str] = []
    prefix_lines: list[str] = []

    table_started = False

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("|") and "|" in stripped_line[1:]:
            table_started = True
            table_rows.append(line)
            continue

        if not table_started:
            prefix_lines.append(line)

    if len(table_rows) < 3:
        return [chunk]

    header = table_rows[0]
    separator = table_rows[1]
    data_rows = table_rows[2:]

    grouped_rows: dict[str, list[str]] = {}

    for row in data_rows:
        columns = [column.strip() for column in row.strip("|").split("|")]

        if len(columns) < 3:
            continue

        component = columns[0]

        grouped_rows.setdefault(component, []).append(row)

    chunks: list[Chunk] = []

    prefix_content = "\n".join(prefix_lines).strip()

    if prefix_content:
        chunks.append(
            _create_parent_chunk(
                parent_chunk=chunk,
                lines=prefix_lines,
            )
        )

    for component, rows in grouped_rows.items():
        chunks.append(
            _create_events_chunk(
                parent_chunk=chunk,
                component=component,
                header=header,
                separator=separator,
                rows=rows,
            )
        )

    return chunks


def _create_events_chunk(
    parent_chunk: Chunk,
    component: str,
    header: str,
    separator: str,
    rows: list[str],
) -> Chunk:
    content = "\n".join(
        [
            header,
            separator,
            *rows,
        ]
    ).strip()

    clean_component = _extract_link_text(component)

    heading_path = [
        *parent_chunk.metadata.get(
            "heading_path",
            [],
        ),
        clean_component,
    ]

    return Chunk(
        id=(f"{parent_chunk.id}::{_slugify(clean_component)}"),
        content=content,
        source=parent_chunk.source,
        metadata={
            **parent_chunk.metadata,
            "section": clean_component,
            "heading_path": heading_path,
            "event_component": clean_component,
        },
    )


def _extract_link_text(
    value: str,
) -> str:
    match = re.match(r"\[([^\]]+)\]", value)

    if match:
        return match.group(1)

    return value
