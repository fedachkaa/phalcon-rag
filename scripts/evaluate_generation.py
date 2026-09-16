from pathlib import Path

from dotenv import load_dotenv

from phalcon_rag.pipeline_factory import create_pipeline
from phalcon_rag.utils import load_json, save_result

load_dotenv()


retrieval_questions = load_json(Path("data/evaluation/retrieval_questions.json"))

pipeline = create_pipeline()

results = []
for question in retrieval_questions:
    top_chunks = pipeline.retrieve(question["query"])

    answer = pipeline.answer_generator.generate(
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
