# Phalcon RAG Assistant

A retrieval-augmented AI assistant for the **Phalcon PHP framework**, grounded in official **Phalcon 5.20 documentation** and **Zephir source code**. Ask questions about framework APIs and internals, or provide selected PHP code for a context-aware explanation.

Built as an end-to-end RAG engineering project: source ingestion, structure-aware chunking, benchmark-driven retrieval selection, hybrid search, cross-encoder reranking, grounded answer generation, a FastAPI interface, and containerized AWS deployment.

## Features

- **Documentation + implementation evidence:** Answers draw on both official documentation and Phalcon source code.
- **Context-aware questions:** Optionally send a PHP snippet alongside your question, including through the PhpStorm integration.
- **Hybrid retrieval:** Qwen3-Embedding-0.6B dense retrieval plus BM25, combined using weighted Reciprocal Rank Fusion (4:1 dense to lexical).
- **Source-aware reranking:** MiniLM cross-encoder reranks documentation and source-code candidates separately; the generation context includes up to **2 documentation + 5 source-code chunks**.
- **Traceable answers:** The API returns an answer and structured source metadata, including source file, method, and line range where available.
- **CPU deployment:** Dockerized FastAPI service tested on an AWS EC2 `t3.large` instance with Hugging Face models cached on a persistent Docker volume.

## Architecture

```text
Phalcon 5.20 Documentation       Phalcon 5.20 Zephir Source
             |                                |
    Structure-aware chunks             Method-level chunks
             |                                |
       Dense + BM25                      Dense + BM25
             |                                |
        Weighted RRF                     Weighted RRF
             |                                |
       Docs candidates                  Source candidates
             |                                |
       MiniLM reranking                 MiniLM reranking
             |                                |
         Top 2 docs                       Top 5 source
             +---------------+----------------+
                             |
              Question + optional PHP context
                             |
                  Grounded LLM generation
                             |
                  Answer + source metadata
```

The corpus includes approximately **10.8k documentation chunks from 151 files**, plus method-level source-code chunks with metadata.

Documentation and source code are retrieved independently, and candidate passages are reranked separately by source type. The application initializes the RAG pipeline at startup and reuses it across requests.

## API

The FastAPI application is defined in `src/phalcon_rag/api/app.py`.

### `GET /health`

Returns:

```json
{
  "status": "ok"
}
```

This endpoint reports a healthy response after the application has successfully started and initialized its RAG pipeline.

### `POST /api/v1/ask`

Accepts a required question and an optional PHP code snippet.

Both fields allow **1–2000 characters after trimming leading and trailing whitespace**, when provided.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the difference between DI get() and getShared()?"}'
```

With selected PHP code:

```json
{
  "question": "Could this code trigger N+1 queries?",
  "context": "foreach ($users as $user) { Orders::find(['conditions' => 'user_id = :id:', 'bind' => ['id' => $user->id]]); }"
}
```

Response shape (illustrative values, not a recorded model answer):

```json
{
  "answer": "The loop may issue a separate orders query for each user...",
  "sources": [
    {
      "id": "example-chunk-id",
      "type": "source_code",
      "file": "phalcon/Mvc/Model.zep",
      "section": null,
      "method": "find",
      "start_line": 100,
      "end_line": 150
    }
  ]
}
```

The `sources` array may include documentation and source-code references. The `section`, `method`, `start_line`, and `end_line` fields are nullable.

Interactive API documentation is available at `/docs` while the service is running.

## Run Locally

### Requirements

- Python 3.11 or newer (Python 3.12 is used in the Docker image).
- Processed Phalcon 5.20 documentation and source-code corpora.
- Qwen embeddings corresponding to the processed corpora.
- An OpenAI API key.
- A supported OpenAI generation model (defaults to `gpt-5.6-luna`; configurable via `GENERATION_MODEL`).
- Internet access for initial embedding and reranker model downloads.

### Installation

Clone the repository:

```bash
git clone https://github.com/fedachkaa/phalcon-rag.git
cd phalcon-rag
```

Create and activate a virtual environment:

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project:

```bash
pip install -e .
```

Create an environment configuration:

**Linux / macOS:**

```bash
cp .env.example .env
```

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
```

Set your `OPENAI_API_KEY` in `.env`.

Optionally, set `GENERATION_MODEL` to override the default generation model (`gpt-5.6-luna`).

```dotenv
OPENAI_API_KEY=your-api-key
GENERATION_MODEL=gpt-5.6-luna
```

**Never commit `.env`, API keys, or other credentials.**

### Prepare the Knowledge Base

Before starting the service, download and process the Phalcon 5.20 documentation and Zephir source code, then generate the corresponding Qwen embeddings.

See [Setup and Evaluation Commands](docs/setup-and-evaluation-commands.md) for the preparation workflow.

The runtime expects corpus and embedding artifacts in the project directory, including:

```text
data/
├── processed/
│   ├── phalcon_docs_5.20.jsonl
│   └── phalcon_source_code_5.20.jsonl
└── embeddings/
    ├── phalcon_docs/
    │   ├── qwen_embeddings.npy
    │   └── chunk_ids.json
    └── phalcon_source_code/
        ├── qwen_embeddings.npy
        └── chunk_ids.json
```

The RAG pipeline loads these artifacts during initialization. Missing or incompatible files will prevent the service from starting successfully.

### Start the API

Run the FastAPI application from the repository root:

```bash
uvicorn phalcon_rag.api.app:app --host 127.0.0.1 --port 8000
```

Check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{ "status": "ok" }
```

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

### Command-Line Interface

You can also ask a question directly from the command line:

```bash
python scripts/answer_question.py "How do I create a transaction in Phalcon?"
```

The CLI loads the RAG pipeline and generates an answer using the prepared knowledge base.

### API Security

The current API is intended for local development or private deployment. It does not implement authentication or request-level authorization.

Do not expose it directly to the public internet without appropriate access controls, authentication, rate limiting, and transport security.

## PhpStorm Integration

The [Phalcon RAG Assistant PhpStorm plugin](https://github.com/fedachkaa/phalcon-rag-plugin) integrates this backend directly into the IDE.

It supports questions with optional selected PHP code, Markdown-formatted answers, and clickable references to Phalcon documentation and Zephir source code.

**[Download the PhpStorm plugin (v1.0.0)](https://github.com/fedachkaa/phalcon-rag-plugin/releases/tag/v1.0.0)**

The plugin requires a running backend API. Configure the API base URL under **PhpStorm → Settings → Tools → Phalcon RAG**.

### Demo

https://github.com/user-attachments/assets/6614303c-7cce-4686-b5a4-38b705e1b134

## Evaluation Highlights

The model and retrieval configuration were chosen through comparative evaluation rather than by model size alone.

| Experiment                                           | Reported result                                                        |
| ---------------------------------------------------- | ---------------------------------------------------------------------- |
| Qwen3-Embedding-0.6B dense retrieval                 | Recall@10 **0.900** (30-question documentation benchmark)              |
| Hybrid retrieval (4:1 dense:BM25)                    | Recall@10 **0.933** (same benchmark)                                   |
| Generation evaluation, latest reported configuration | **171/180 (95.0%)** across correctness, completeness, and groundedness |

The generation scores are manual benchmark assessments, **not an independently verified accuracy guarantee**.

Earlier evaluation runs reported **164/180 (91.1%)**.

The final generation evaluation reported:

| Metric       |               Score |
| ------------ | ------------------: |
| Correctness  |               60/60 |
| Completeness |               51/60 |
| Groundedness |               60/60 |
| **Overall**  | **171/180 (95.0%)** |

See [Detailed Evaluation Results](docs/evaluation.md) for methodology, comparisons, and limitations.

## Deployment

The application has been built into a Docker image, pushed to **Amazon ECR**, and deployed to an **AWS EC2 `t3.large`** instance.

The FastAPI service was accessible through an **SSH tunnel**, with the API bound to `127.0.0.1:8000` on the EC2 host rather than exposed publicly.

Hugging Face model downloads were persisted in a Docker volume to avoid repeated downloads after container restarts.

The EC2 instance is not intended to remain running continuously. Stopping an instance avoids ongoing instance compute charges, although EBS storage and other allocated resources may still incur costs.

For more information, see the [AWS Deployment Guide](docs/deployment.md).

## Project Layout

```text
src/phalcon_rag/
├── api/             # FastAPI routes and request/response schemas
├── ingestion/       # Documentation and Zephir source parsing/chunking
├── retrieval/       # Dense, BM25, hybrid, multi-source, RRF
├── reranking/       # Cross-encoder reranking
├── generation/      # Prompting, query rewriting, answer generation
├── evaluation/      # Evaluation utilities
├── config.py
├── pipeline.py
└── pipeline_factory.py

scripts/             # Ingestion, embedding, benchmarking, CLI
tests/               # Automated tests

docs/
├── deployment.md
├── evaluation.md
└── setup-and-evaluation-commands.md

Dockerfile
pyproject.toml
```

## Documentation

- [Setup and Evaluation Commands](docs/setup-and-evaluation-commands.md)
- [Detailed Evaluation Results](docs/evaluation.md)
- [AWS Deployment Guide](docs/deployment.md)
- [PhpStorm Plugin Repository](https://github.com/fedachkaa/phalcon-rag-plugin)

## Limitations and Roadmap

- **Indirect execution paths:** Retrieving related implementation details spread across multiple Zephir methods remains challenging.
- **Framework version:** The knowledge base is scoped to Phalcon 5.20.
- **Private API:** The current service does not include authentication, rate limiting, or production-grade API access controls.
- **Single-turn interactions:** Chat history, streaming responses, and multi-turn conversations are not implemented.

Potential future improvements include framework-version filtering, improved citation presentation, streaming, and conversational history.

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run linting and formatting checks:

```bash
ruff check .
ruff format --check .
```

Run automated tests:

```bash
pytest
```

To format the code automatically:

```bash
ruff format .
```
