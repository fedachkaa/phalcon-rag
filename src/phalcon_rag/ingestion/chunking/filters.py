import re

from phalcon_rag.models import Chunk

from .headings import HEADING_PATTERN


def _filter_chunks(
    chunks: list[Chunk],
) -> list[Chunk]:
    return [chunk for chunk in chunks if not _should_skip_chunk(chunk)]


def _should_skip_chunk(
    chunk: Chunk,
) -> bool:
    section = chunk.metadata.get("section")

    if section == "Method Summary":
        return True

    if _is_heading_only_chunk(chunk):
        return True

    return bool(not _has_substantive_content(chunk))


def _is_heading_only_chunk(
    chunk: Chunk,
) -> bool:
    lines = [line.strip() for line in chunk.content.splitlines() if line.strip()]

    if len(lines) != 1:
        return False

    return bool(HEADING_PATTERN.match(lines[0]))


def _has_substantive_content(
    chunk: Chunk,
) -> bool:
    content = chunk.content
    content = re.sub(r"^#{1,6}\s+.*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)
    content = re.sub(r"\[\s*\]\[[^\]]+\]", "", content)
    content = re.sub(r"<[A-Za-z][^>]*/>", "", content)
    content = re.sub(r"^\s*---+\s*$", "", content, flags=re.MULTILINE)

    return bool(content.strip())
