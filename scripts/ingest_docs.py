from pathlib import Path
from phalcon_rag.ingestion.docs_loader import DocsLoader
from phalcon_rag.ingestion.docs_chunker import DocsChunker

docs_path = Path(
    "data/raw/phalcon-docs/src/content/docs-5.20"
)

loader = DocsLoader(docs_path)
chunker = DocsChunker()

documents = loader.load()
chunks = []

for document in documents:
    chunks.extend(chunker.chunk(document))

chunk_sizes = [len(chunk.content.split()) for chunk in chunks]

print()
print(f"Min chunk size: {min(chunk_sizes)} words")
print(f"Max chunk size: {max(chunk_sizes)} words")
print(f"Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.1f} words")
