from phalcon_rag.ingestion.frontmatter import parse_frontmatter


def test_parse_frontmatter_returns_metadata_and_body():
    content = """---
title: Models
description: Working with Phalcon models
---
# Models

Some documentation.
"""

    metadata, body = parse_frontmatter(content)

    assert metadata == {
        "title": "Models",
        "description": "Working with Phalcon models",
    }
    assert body == "# Models\n\nSome documentation.\n"


def test_parse_frontmatter_returns_original_content_when_frontmatter_is_missing():
    content = "# Models\n\nSome documentation."

    metadata, body = parse_frontmatter(content)

    assert metadata == {}
    assert body == content


def test_parse_frontmatter_returns_original_content_when_closing_delimiter_is_missing():
    content = """---
title: Models
# Models
"""

    metadata, body = parse_frontmatter(content)

    assert metadata == {}
    assert body == content
