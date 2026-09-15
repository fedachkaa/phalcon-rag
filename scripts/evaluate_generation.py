from pathlib import Path

from dotenv import load_dotenv

from phalcon_rag.config import (
    EMBEDDING_QUERY_INSTRUCTION,
    GENERATION_MODEL,
    SOURCE_CODE_QUERY_INSTRUCTION,
)
from phalcon_rag.generation.answer_generator import AnswerGenerator
from phalcon_rag.models import SOURCE_PHALCON_DOCS, SOURCE_PHALCON_SOURCE_CODE
from phalcon_rag.pipeline import RagPipeline
from phalcon_rag.reranking.cross_encoder import CrossEncoderReranker
from phalcon_rag.retrieval.bm25 import BM25Retriever
from phalcon_rag.retrieval.dense import DenseRetriever
from phalcon_rag.retrieval.hybrid import HybridRetriever
from phalcon_rag.retrieval.multi_source import MultiSourceRetriever
from phalcon_rag.utils import load_json, load_retrieval_data, save_result

load_dotenv()


retrieval_questions = load_json(Path("data/evaluation/retrieval_questions.json"))

instructions = {
    SOURCE_PHALCON_DOCS: EMBEDDING_QUERY_INSTRUCTION,
    SOURCE_PHALCON_SOURCE_CODE: SOURCE_CODE_QUERY_INSTRUCTION,
}

retrievers = {}
for source in [SOURCE_PHALCON_DOCS, SOURCE_PHALCON_SOURCE_CODE]:
    chunks, embeddings = load_retrieval_data(
        chunks_path=Path(f"data/processed/{source}_5.20.jsonl"),
        embeddings_path=Path(f"data/embeddings/{source}/qwen_embeddings.npy"),
        chunk_ids_path=Path(f"data/embeddings/{source}/chunk_ids.json"),
    )

    retrievers[source] = HybridRetriever(
        DenseRetriever(chunks, embeddings, instructions[source]), BM25Retriever(chunks)
    )

answer_generator = AnswerGenerator(GENERATION_MODEL)

pipeline = RagPipeline(
    retriever=MultiSourceRetriever(
        retrievers[SOURCE_PHALCON_DOCS],
        retrievers[SOURCE_PHALCON_SOURCE_CODE],
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
