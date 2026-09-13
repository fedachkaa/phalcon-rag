from openai import OpenAI

from phalcon_rag.models import Chunk

from .prompt import build_prompt


class AnswerGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.client = OpenAI()

    def generate(
        self,
        query: str,
        chunks: list[Chunk],
    ) -> str:
        response = self.client.responses.create(
            model=self.model_name, input=build_prompt(query, chunks)
        )

        return response.output_text
