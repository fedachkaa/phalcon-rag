from pathlib import Path
from ..models import Document
from .frontmatter import parse_frontmatter
from .cleaning import clean_content

class DocsLoader:
    def __init__(self, docs_path: Path):
        self.docs_path = docs_path

    def load(self) -> list[Document]:
        documents = []
        for file_path in self.docs_path.rglob("*.mdx"):
            raw_content = file_path.read_text(encoding="utf-8")

            frontmatter, content = parse_frontmatter(raw_content)
            content = clean_content(content)
            relative_path = file_path.relative_to(self.docs_path).as_posix()

            document = Document(
                id=str(relative_path),
                content=content,
                source="phalcon_docs",
                metadata={
                    "file_path": str(relative_path),
                    "version": "5.20",
                    **frontmatter
                }
            )

            documents.append(document)
        
        return documents
