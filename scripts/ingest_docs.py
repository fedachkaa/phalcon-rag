from pathlib import Path
from phalcon_rag.ingestion.docs_loader import DocsLoader


docs_path = Path(
    "data/raw/phalcon-docs/src/content/docs-5.20"
)

loader = DocsLoader(docs_path)
documents = loader.load()

print(f"Loaded documents: {len(documents)}")

if documents:
    first_document = documents[0]

    print(f"ID: {first_document.id}")
    print(f"Source: {first_document.source}")
    print(f"Metadata: {first_document.metadata}")
    print()
    print(first_document.content[:500])
