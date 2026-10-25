import os

from dotenv import load_dotenv

load_dotenv()

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

# Reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_DOCS_TOP_K = 2
RERANK_SOURCE_TOP_K = 5

# Generation
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-5.6-luna")

