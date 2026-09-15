MODELS = {
    "qwen": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "embeddings_path": "data/embeddings/{source}/qwen_embeddings.npy",
        "query_instruction": (
            "Retrieve relevant Phalcon documentation passages "
            "that answer the technical question."
        ),
    },
    "bge": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "embeddings_path": "data/embeddings/{source}/bge_embeddings.npy",
        "query_prefix": ("Represent this sentence for searching relevant passages: "),
    },
    "gte": {
        "model_name": "Alibaba-NLP/gte-modernbert-base",
        "embeddings_path": "data/embeddings/{source}/gte_embeddings.npy",
    },
    "e5": {
        "model_name": "intfloat/multilingual-e5-large-instruct",
        "embeddings_path": "data/embeddings/{source}/e5_embeddings.npy",
        "query_instruction": (
            "Retrieve relevant technical documentation that answers "
            "the user's question."
        ),
    },
}


def format_query(model_key: str, query: str) -> str:
    config = MODELS[model_key]

    if model_key == "bge":
        return config["query_prefix"] + query

    if model_key in {"e5", "qwen"}:
        return f"Instruct: {config['query_instruction']}\nQuery: {query}"

    return query
