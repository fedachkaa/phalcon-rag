from typing import Any
import yaml


def parse_frontmatter(content: str) -> tuple[dict[str: Any], str]:
    if not content.startswith("---"):
        return {}, content
    
    parts = content.split("---", 2)

    if len(parts) < 3:
        return {}, content
    
    raw_frontmatter = parts[1]
    body = parts[2].lstrip()

    metadata = yaml.safe_load(raw_frontmatter) or {}

    return metadata, body