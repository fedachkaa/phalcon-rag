from openai import OpenAI

from phalcon_rag.generation.prompt import build_prompt
from phalcon_rag.models import Chunk


class AnswerGenerator:
    def __init__(self, client: OpenAI, model_name: str):
        self.model_name = model_name
        self.client = client

    def generate(
        self, query: str, chunks: list[Chunk], context: str | None = None
    ) -> str:
        response = self.client.responses.create(
            model=self.model_name, input=build_prompt(query, chunks, context)
        )

        return response.output_text
