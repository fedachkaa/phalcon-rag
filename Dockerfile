FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

COPY data/processed ./data/processed
COPY data/embeddings ./data/embeddings

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "phalcon_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]