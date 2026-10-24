import statistics
import time

import torch
from sentence_transformers import SentenceTransformer

from phalcon_rag.config import (
    EMBEDDING_MODEL,
    EMBEDDING_QUERY_INSTRUCTION,
    SOURCE_CODE_QUERY_INSTRUCTION,
)


QUERY = "How does Phalcon execute Model::findFirst()?"
RUNS = 10
WARMUP_RUNS = 5


def main() -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)

    formatted_queries = [
        f"Instruct: {instruction}\nQuery: {QUERY}"
        for instruction in (
            EMBEDDING_QUERY_INSTRUCTION,
            SOURCE_CODE_QUERY_INSTRUCTION,
        )
    ]

    print(f"Model: {EMBEDDING_MODEL}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Runs: {RUNS} ({WARMUP_RUNS} warm-up)\n")

    timings = []

    for i in range(RUNS):
        start = time.perf_counter()

        for query in formatted_queries:
            model.encode(query, normalize_embeddings=True)

        elapsed = time.perf_counter() - start
        timings.append(elapsed)

        print(f"Run {i + 1}: {elapsed:.3f}s")

    measured = timings[WARMUP_RUNS:]

    print("\nResults:")
    print(f"Mean:   {statistics.mean(measured):.3f}s")
    print(f"Median: {statistics.median(measured):.3f}s")
    print(f"Min:    {min(measured):.3f}s")
    print(f"Max:    {max(measured):.3f}s")


if __name__ == "__main__":
    main()
