from transformers import AutoTokenizer
from pathlib import Path
import json
import numpy as np

MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": 32768,
    "BAAI/bge-large-en-v1.5": 512,
    "Alibaba-NLP/gte-modernbert-base": 8192,
    "intfloat/multilingual-e5-large-instruct": 512,
}

processed_data_path = Path("data/processed/phalcon_docs_5.20.jsonl")
processed_data = []
with processed_data_path.open(mode="r", encoding="UTF-8") as file:
    for line in file:
        processed_data.append(json.loads(line))

for model, max_tokens in MODELS.items():
    print(f'-----{model}-----')

    tokenizer = AutoTokenizer.from_pretrained(model)
    token_lengths = []

    for chunk_data in processed_data:
        tokens = tokenizer.encode(chunk_data['content'], add_special_tokens=True)
        token_lengths.append(len(tokens))

    mean = np.mean(token_lengths)
    median = np.median(token_lengths)
    p95 = np.percentile(token_lengths, 95)
    p99 = np.percentile(token_lengths, 99)
    min_tokens = min(token_lengths)
    max_chunk_tokens = max(token_lengths)

    oversized = [
        length
        for length in token_lengths
        if length > max_tokens
    ]

    percentage = len(oversized) / len(token_lengths) * 100

    print(f"Chunks: {len(token_lengths)}")
    print(f"Mean: {mean:.1f}")
    print(f"Median: {median:.1f}")
    print(f"P95: {p95:.1f}")
    print(f"P99: {p99:.1f}")
    print(f"Min: {min_tokens}")
    print(f"Max: {max_chunk_tokens}")
    print(f"Model limit: {max_tokens}")
    print(f"Over limit: {len(oversized)} ({percentage:.2f}%)")

    top_indices = np.argsort(token_lengths)[-10:][::-1]

    output_path = Path("data/processed/token_analysis.txt")

    oversized_indices = [
        index
        for index, length in enumerate(token_lengths)
        if length > max_tokens
    ]

    oversized_indices.sort(key=lambda index: token_lengths[index])

    with output_path.open("w", encoding="utf-8") as file:
        file.write(model + "\n" + "-" * 80 + "\n")

        for index in oversized_indices[:30]:
            chunk = processed_data[index]

            file.write(f"ID: {chunk['id']}\n")
            file.write(f"Tokens: {token_lengths[index]}\n")
            file.write(chunk["content"])
            file.write("\n\n" + "-" * 80 + "\n")

    print()