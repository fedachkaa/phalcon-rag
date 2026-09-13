from pathlib import Path

from phalcon_rag.config import GENERATION_MODEL
from phalcon_rag.generation.answer_generator import AnswerGenerator
from phalcon_rag.pipeline import RagPipeline
from phalcon_rag.reranking.cross_encoder import CrossEncoderReranker
from phalcon_rag.retrieval.bm25 import BM25Retriever
from phalcon_rag.retrieval.dense import DenseRetriever
from phalcon_rag.retrieval.hybrid import HybridRetriever
from phalcon_rag.utils import load_json, load_retrieval_data, save_result

chunks, embeddings = load_retrieval_data(
    chunks_path=Path("data/processed/phalcon_docs_5.20.jsonl"),
    embeddings_path=Path("data/embeddings/qwen_embeddings.npy"),
    chunk_ids_path=Path("data/embeddings/chunk_ids.json"),
)

retrieval_questions = load_json(Path("data/evaluation/retrieval_questions.json"))
answer_generator = AnswerGenerator(GENERATION_MODEL)
pipeline = RagPipeline(
    hybrid_retriever=HybridRetriever(
        DenseRetriever(chunks, embeddings), BM25Retriever(chunks)
    ),
    reranker=CrossEncoderReranker(),
    answer_generator=answer_generator,
)

results = []
for question in retrieval_questions:
    top_chunks = pipeline.retrieve(question["query"])

    answer = answer_generator.generate(
        question["query"],
        top_chunks,
    )

    results.append(
        {
            "id": question["id"],
            "query": question["query"],
            "expected_answer_points": question["expected_answer_points"],
            "retrieved_chunks": [
                {
                    "position": position,
                    "id": chunk.id,
                    "source": chunk.source,
                    "content": chunk.content,
                }
                for position, chunk in enumerate(top_chunks, start=1)
            ],
            "generated_answer": answer,
        }
    )

save_result("generation", results)
