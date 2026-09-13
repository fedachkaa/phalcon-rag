from typing import Any

import numpy as np
from transformers import AutoTokenizer


def analyze_model_tokens(
    model_name: str,
    max_tokens: int,
    processed_data: list[dict[str, Any]],
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    token_lengths = [
        len(tokenizer.encode(chunk["content"], add_special_tokens=True))
        for chunk in processed_data
    ]

    oversized_indices = [
        index for index, length in enumerate(token_lengths) if length > max_tokens
    ]

    oversized_indices.sort(
        key=lambda index: token_lengths[index],
        reverse=True,
    )

    oversized = [
        {
            "id": processed_data[index]["id"],
            "tokens": token_lengths[index],
        }
        for index in oversized_indices[:30]
    ]

    return {
        "model_name": model_name,
        "chunks": len(token_lengths),
        "mean": float(np.mean(token_lengths)),
        "median": float(np.median(token_lengths)),
        "p95": float(np.percentile(token_lengths, 95)),
        "p99": float(np.percentile(token_lengths, 99)),
        "min": min(token_lengths),
        "max": max(token_lengths),
        "model_limit": max_tokens,
        "over_limit": len(oversized_indices),
        "percentage_over_limit": len(oversized_indices) / len(token_lengths) * 100,
        "oversized": oversized,
    }
