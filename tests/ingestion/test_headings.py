from phalcon_rag.ingestion.chunking.headings import (
    _create_chunk_id,
    _split_by_headings,
)
from phalcon_rag.models import Document


def test_create_chunk_id_uses_heading_path():
    chunk_id = _create_chunk_id(
        document_id="models",
        heading_path=["Models", "Relationships"],
    )

    assert chunk_id == "models::models::relationships"


def test_create_chunk_id_uses_root_when_heading_path_is_empty():
    chunk_id = _create_chunk_id(
        document_id="models",
        heading_path=[],
    )

    assert chunk_id == "models::root"


def test_split_by_headings_preserves_heading_hierarchy():
    document = Document(
        id="models",
        source="models.md",
        content="""# Models
Models introduction.

## Relationships
Relationships documentation.

### One-to-Many
One-to-many documentation.
""",
    )

    chunks = _split_by_headings(document)

    assert len(chunks) == 3

    assert chunks[0].metadata["heading_path"] == ["Models"]
    assert chunks[0].metadata["section"] == "Models"

    assert chunks[1].metadata["heading_path"] == [
        "Models",
        "Relationships",
    ]
    assert chunks[1].metadata["section"] == "Relationships"

    assert chunks[2].metadata["heading_path"] == [
        "Models",
        "Relationships",
        "One-to-Many",
    ]
    assert chunks[2].metadata["section"] == "One-to-Many"
