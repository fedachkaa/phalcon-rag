from dataclasses import dataclass

from phalcon_rag.ingestion.source_parser import mask_comments
from phalcon_rag.models import Chunk, Method

MAX_METHOD_SIZE = 1500
MIN_CHUNK_SIZE = 300


@dataclass
class SplitPoint:
    position: int
    depth: int


def chunk_method(method: Method) -> list[Chunk]:
    if len(method.content) <= MAX_METHOD_SIZE:
        return [method_to_chunk(method)]

    return split_large_method(method)


def method_to_chunk(method: Method) -> Chunk:
    chunk_id = f"{method.id}::0"

    return Chunk(
        id=chunk_id,
        content=method.content,
        source=method.source,
        metadata={
            **method.metadata,
            "method": method.name,
            "parent_id": method.id,
            "chunk_index": 0,
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "start_line": method.start_line,
            "end_line": method.end_line,
        },
    )


def split_large_method(method: Method) -> list[Chunk]:
    split_points = find_split_points(method.content)

    ranges = split_structurally(
        method.content,
        split_points,
    )

    ranges = merge_small_ranges(ranges)

    chunks = []

    for index, (start, end) in enumerate(ranges):
        chunk_id = f"{method.id}::{index}"

        chunk_start_line = method.start_line + method.content.count("\n", 0, start)

        chunk_end_line = method.start_line + method.content.count("\n", 0, end)

        chunks.append(
            Chunk(
                id=chunk_id,
                content=method.content[start:end],
                source=method.source,
                metadata={
                    **method.metadata,
                    "method": method.name,
                    "parent_id": method.id,
                    "chunk_index": index,
                    "prev_chunk_id": (
                        f"{method.id}::{index - 1}" if index > 0 else None
                    ),
                    "next_chunk_id": (
                        f"{method.id}::{index + 1}" if index < len(ranges) - 1 else None
                    ),
                    "start_line": chunk_start_line,
                    "end_line": chunk_end_line,
                },
            )
        )

    return chunks


def find_split_points(content: str) -> list[SplitPoint]:
    parsed_content = mask_comments(content)

    split_points = []

    depth = 0
    in_string = None
    escaped = False

    for position, char in enumerate(parsed_content):
        if in_string:
            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == in_string:
                in_string = None

            continue

        if char in ('"', "'"):
            in_string = char
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth >= 1:
                split_points.append(
                    SplitPoint(
                        position=position + 1,
                        depth=depth,
                    )
                )

        elif char == ";" and depth >= 1:
            split_points.append(
                SplitPoint(
                    position=position + 1,
                    depth=depth,
                )
            )

    return split_points


def split_structurally(
    content: str,
    split_points: list[SplitPoint],
    start: int = 0,
    end: int | None = None,
    depth: int = 1,
) -> list[tuple[int, int]]:
    if end is None:
        end = len(content)

    if end - start <= MAX_METHOD_SIZE:
        return [(start, end)]

    points = [
        point.position
        for point in split_points
        if point.depth == depth and start < point.position < end
    ]

    if not points:
        deeper_points_exist = any(
            point.depth > depth and start < point.position < end
            for point in split_points
        )

        if deeper_points_exist:
            return split_structurally(
                content=content,
                split_points=split_points,
                start=start,
                end=end,
                depth=depth + 1,
            )

        return hard_split(start, end)

    ranges = pack_segments(
        start=start,
        end=end,
        points=points,
    )

    result = []

    for range_start, range_end in ranges:
        if range_end - range_start <= MAX_METHOD_SIZE:
            result.append((range_start, range_end))
            continue

        deeper_points_exist = any(
            point.depth > depth and range_start < point.position < range_end
            for point in split_points
        )

        if deeper_points_exist:
            result.extend(
                split_structurally(
                    content=content,
                    split_points=split_points,
                    start=range_start,
                    end=range_end,
                    depth=depth + 1,
                )
            )
        else:
            result.extend(hard_split(range_start, range_end))

    return result


def pack_segments(
    start: int,
    end: int,
    points: list[int],
) -> list[tuple[int, int]]:
    ranges = []
    current_start = start

    while end - current_start > MAX_METHOD_SIZE:
        max_end = current_start + MAX_METHOD_SIZE

        available_points = [
            point for point in points if current_start < point <= max_end
        ]

        if not available_points:
            break

        current_end = max(available_points)

        ranges.append((current_start, current_end))
        current_start = current_end

    ranges.append((current_start, end))

    return ranges


def hard_split(
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    ranges = []

    while end - start > MAX_METHOD_SIZE:
        ranges.append((start, start + MAX_METHOD_SIZE))

        start += MAX_METHOD_SIZE

    if start < end:
        ranges.append((start, end))

    return ranges


def merge_small_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if len(ranges) <= 1:
        return ranges

    result = []
    index = 0

    while index < len(ranges):
        start, end = ranges[index]
        size = end - start

        if size >= MIN_CHUNK_SIZE:
            result.append((start, end))
            index += 1
            continue

        # Try merging with previous range
        if result:
            prev_start, _ = result[-1]

            if end - prev_start <= MAX_METHOD_SIZE:
                result[-1] = (prev_start, end)
                index += 1
                continue

        # Try merging with next range
        if index + 1 < len(ranges):
            _, next_end = ranges[index + 1]

            if next_end - start <= MAX_METHOD_SIZE:
                result.append((start, next_end))
                index += 2
                continue

        result.append((start, end))
        index += 1

    return result
