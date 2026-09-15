import json
from dataclasses import asdict
from pathlib import Path

from phalcon_rag.ingestion.source_chunker import chunk_method
from phalcon_rag.ingestion.source_parser import parse_methods
from phalcon_rag.models import SOURCE_PHALCON_SOURCE_CODE

SOURCE_DIR = Path("data/raw/phalcon-source-code/phalcon")
OUTPUT_FILE = Path(f"data/processed/{SOURCE_PHALCON_SOURCE_CODE}_5.20.jsonl")


def main():
    methods = parse_methods(SOURCE_DIR)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for method in methods:
            chunks = chunk_method(method)

            for chunk in chunks:
                file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
