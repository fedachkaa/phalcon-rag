import re

MAX_CHUNK_WORDS = 500


def _word_count(
    content: str,
) -> int:
    return len(content.split())


def _slugify(value: str) -> str:
    value = value.lower().strip()

    value = re.sub(r"[^a-z0-9_]+", "-", value)

    return value.strip("-")


def _make_unique_id(
    base_id: str,
    id_counts: dict[str, int],
) -> str:
    count = id_counts.get(base_id, 0) + 1
    id_counts[base_id] = count

    if count == 1:
        return base_id

    return f"{base_id}-{count}"
