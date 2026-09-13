from pathlib import Path

from phalcon_rag.ingestion.docs_chunker import DocsChunker
from phalcon_rag.ingestion.docs_loader import DocsLoader
from phalcon_rag.ingestion.exporter import JsonlExporter

docs_path = Path("data/raw/phalcon-docs/src/content/docs-5.20")
output_path = Path("data/processed/phalcon_docs_5.20.jsonl")

loader = DocsLoader(docs_path)
chunker = DocsChunker()
exporter = JsonlExporter()

documents = loader.load()
chunks = []

for document in documents:
    chunks.extend(chunker.chunk(document))

exporter.export(chunks=chunks, output_path=output_path)
