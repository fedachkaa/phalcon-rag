import json
from pathlib import Path

from phalcon_rag.evaluation.token_analysis import analyze_model_tokens

MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": 32768,
    "BAAI/bge-large-en-v1.5": 512,
    "Alibaba-NLP/gte-modernbert-base": 8192,
    "intfloat/multilingual-e5-large-instruct": 512,
}


def main():
    processed_data_path = Path("data/processed/phalcon_docs_5.20.jsonl")

    with processed_data_path.open(encoding="utf-8") as file:
        processed_data = [json.loads(line) for line in file]

    results = []

    for model, max_tokens in MODELS.items():
        results.append(
            analyze_model_tokens(
                model_name=model,
                max_tokens=max_tokens,
                processed_data=processed_data,
            )
        )

    output_path = Path("data/evaluation/results/token_analysis.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
