import argparse

from dotenv import load_dotenv

from phalcon_rag.pipeline_factory import create_pipeline

load_dotenv()


def answer_question(question: str) -> str:
    pipeline = create_pipeline()

    return pipeline.answer(question)


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question about Phalcon documentation or source code"
    )
    parser.add_argument("question", help="Question to answer.")
    args = parser.parse_args()

    result = answer_question(args.question)

    print(result)


if __name__ == "__main__":
    main()
