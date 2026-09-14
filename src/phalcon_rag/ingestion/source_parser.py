import re
from pathlib import Path

from phalcon_rag.models import Method


METHOD_PATTERN = re.compile(
    r'\b'
    r'(?:(?:public|protected|private)\s+)?'
    r'(?:(?:static)\s+)?'
    r'function\s+'
    r'([a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*)'
    r'\s*\('
    r'.*?'
    r'\)'
    r'(?:\s*->\s*[^{]+)?'
    r'\s*\{',
    re.DOTALL | re.IGNORECASE
)

def parse_methods(source_dir: Path) -> list[Method]:
    methods = []

    for entry in source_dir.rglob("*.zep"):
        if entry.is_file():
            raw_content = entry.read_text(encoding="utf-8")
            parsed_content = mask_comments(raw_content)

        for match in METHOD_PATTERN.finditer(parsed_content):
            method_name = match.group(1)

            start_pos = match.start()
            opening_brace_pos = match.end() - 1

            start_line = parsed_content.count("\n", 0, start_pos) + 1

            try:
                closing_brace_pos = find_closing_brace(parsed_content, opening_brace_pos)
            except ValueError as error:
                print(
                    f"[WARNING] {entry} :: {method_name} "
                    f"at line {start_line} | {error}"
                )
                continue

            start_line = parsed_content.count("\n", 0, start_pos) + 1
            end_line = parsed_content.count("\n", 0, closing_brace_pos) + 1

            method_content = raw_content[start_pos:closing_brace_pos + 1]

            relative_path = entry.relative_to(source_dir).as_posix()

            methods.append(
                Method(
                    id=f"{relative_path}::{method_name}",
                    name=method_name,
                    content=method_content,
                    source="source_code",
                    start_line=start_line,
                    end_line=end_line,
                    metadata={
                        "file": relative_path,
                        "language": "zephir",
                    },
                )
            )

    return methods


def mask_comments(content: str) -> str:
    result = list(content)

    in_string = None
    escaped = False
    i = 0

    while i < len(content):
        char = content[i]

        if in_string:
            if escaped:
                escaped = False

            elif char == "\\":
                escaped = True

            elif char == in_string:
                in_string = None

            i += 1
            continue

        if char in ('"', "'"):
            in_string = char
            i += 1
            continue

        if (
            char == "/"
            and i + 1 < len(content)
            and content[i + 1] == "/"
        ):
            while i < len(content) and content[i] != "\n":
                result[i] = " "
                i += 1

            continue

        if (
            char == "/"
            and i + 1 < len(content)
            and content[i + 1] == "*"
        ):
            result[i] = " "
            result[i + 1] = " "
            i += 2

            while i < len(content):
                if (
                    content[i] == "*"
                    and i + 1 < len(content)
                    and content[i + 1] == "/"
                ):
                    result[i] = " "
                    result[i + 1] = " "
                    i += 2
                    break

                if content[i] != "\n":
                    result[i] = " "

                i += 1

            continue

        i += 1

    return "".join(result)


def find_closing_brace(content: str, opening_brace_pos: int) -> int:
    if content[opening_brace_pos] != "{":
        raise ValueError("opening_brace_pos must point to '{'")

    depth = 0
    in_string = None
    escaped = False

    for position in range(opening_brace_pos, len(content)):
        char = content[position]

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

            if depth == 0:
                return position

    raise ValueError("Matching closing brace not found")
