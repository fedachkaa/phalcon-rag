from openai import OpenAI


class QueryRewriter:
    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def rewrite(
        self,
        query: str,
        context: str,
    ) -> str:
        prompt = self.build_rewrite_prompt(
            query=query,
            context=context,
        )

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )

        return response.output_text.strip()

    @staticmethod
    def build_rewrite_prompt(
        query: str,
        context: str,
    ) -> str:
        return f"""You generate semantic search queries for a Phalcon framework RAG system.

Your task is to rewrite the user's question and selected PHP code into a concise search query for retrieving relevant Phalcon documentation and source code.

Rules:
- Do not answer the user's question.
- Focus on Phalcon classes, methods, APIs, relations, and framework behavior visible in the selected code.
- Preserve relevant method and class names from the code.
- Include concepts implied by the user's question when useful for retrieval.
- Do not invent APIs or implementation details.
- Return only the search query.

Question:
{query}

Selected PHP code:
```php
{context}
```"""