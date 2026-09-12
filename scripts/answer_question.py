from pathlib import Path
from phalcon_rag.retrieval.hybrid import HybridRetriever
from phalcon_rag.retrieval.dense import DenseRetriever
from phalcon_rag.retrieval.bm25 import BM25Retriever
from phalcon_rag.reranking.cross_encoder import CrossEncoderReranker
from phalcon_rag.generation.answer_generator import AnswerGenerator
from phalcon_rag.utils import load_retrieval_data

question = "How to see changed fields in the model?"

chunks, embeddings = load_retrieval_data(
    chunks_path=Path("data/processed/phalcon_docs_5.20.jsonl"),
    embeddings_path=Path("data/embeddings/qwen_embeddings.npy"),
    chunk_ids_path=Path("data/embeddings/chunk_ids.json"),
)

bm25_retriever = BM25Retriever(chunks)
dense_retriever = DenseRetriever(chunks, embeddings)
hybrid_retriever = HybridRetriever(dense_retriever, bm25_retriever)
reranker = CrossEncoderReranker()
answer_generator = AnswerGenerator("gpt-5.6-luna")

candidates = hybrid_retriever.search(
    question,
    top_k=10,
)

reranked = reranker.rerank(question, candidates)

top_chunks = [
    chunk
    for chunk, _ in reranked[:5]
]

result = answer_generator.generate(question, top_chunks)

print(result)