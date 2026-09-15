# src/phalcon_rag/config.py

# Embedding
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_QUERY_INSTRUCTION = "Given a Phalcon documentation question, retrieve relevant passages that answer the question."
SOURCE_CODE_QUERY_INSTRUCTION = (
    "Given a Phalcon technical question, retrieve relevant Phalcon source code."
)

# Hybrid retrieval
DENSE_WEIGHT = 4.0
BM25_WEIGHT = 1.0
RRF_K = 60

# Retrieval pipeline
CANDIDATE_K = 50
HYBRID_TOP_K = 10
RERANK_TOP_K = 5

# Reranking
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Generation
GENERATION_MODEL = "gpt-5.6-luna"
