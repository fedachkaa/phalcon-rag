from pathlib import Path
from phalcon_rag.models import Chunk
import json

def load_chunks(path: Path) -> list[Chunk]:
    chunks = []

    with path.open(mode="r", encoding="utf-8") as file:
        for line in file:
            chunks.append(Chunk(**json.loads(line)))

    return chunks


def load_json(path: Path):
    with path.open(mode="r", encoding="utf-8") as file:
        return json.load(file)


def save_result(model_key: str, result: dict) -> None:
    results_dir = Path("data/evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{model_key}.json"

    with results_path.open(mode="w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved:", results_path)
