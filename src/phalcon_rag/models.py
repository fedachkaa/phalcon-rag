from dataclasses import dataclass, field
from typing import Any

SOURCE_PHALCON_SOURCE_CODE = "phalcon_source_code"
SOURCE_PHALCON_DOCS = "phalcon_docs"


@dataclass
class Document:
    id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Method:
    id: str
    name: str
    content: str
    source: str
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)
