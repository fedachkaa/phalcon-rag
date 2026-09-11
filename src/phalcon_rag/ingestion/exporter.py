from dataclasses import asdict
import json
from pathlib import Path
from ..models import Chunk


class JsonlExporter:
    def export(self, chunks: list[Chunk], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open(mode="w", encoding="UTF-8") as file:
            for chunk in chunks:
                record = asdict(chunk)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
