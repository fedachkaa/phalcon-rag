from pathlib import Path

from dotenv import load_dotenv

from phalcon_rag.pipeline_factory import create_pipeline
from phalcon_rag.utils import load_json, save_result

load_dotenv()


retrieval_questions = load_json(Path("data/evaluation/questions.json"))

pipeline = create_pipeline()

results = []
for question in retrieval_questions:
    top_chunks = pipeline.retrieve(question["question"])

    answer = pipeline.answer_generator.generate(
        question["question"],
        top_chunks,
    )

    results.append(
        {
            "id": question["id"],
            "question": question["question"],
            "category": question["category"],
            "expected_answerable": question["expected_answerable"],
            "expected_sources": question["expected_sources"],
            "expected_concepts": question["expected_concepts"],
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
            "evaluation": {
                "correctness": None,
                "completeness": None,
                "groundedness": None,
                "notes": "",
            },
        }
    )

save_result("generation", results)
