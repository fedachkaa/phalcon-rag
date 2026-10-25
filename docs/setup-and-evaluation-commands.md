# Local Setup and Evaluation Commands

This guide covers local installation, preparation of the Phalcon 5.20 knowledge base, retrieval and generation evaluations, and running the RAG assistant.

Run all commands from the repository root unless stated otherwise.

## 1. Installation

### Requirements

- Python 3.11 or newer.
- Git.
- Sufficient disk space and memory for the embedding and reranking models.
- An OpenAI API key for answer generation.
- Internet access for cloning source repositories and downloading models.

Clone the repository:

```bash
git clone https://github.com/fedachkaa/phalcon-rag.git
cd phalcon-rag
```

Create and activate a virtual environment.

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project:

```bash
pip install -e .
```

For development tools such as Ruff and pytest:

```bash
pip install -e ".[dev]"
```

## 2. Configuration

Create a local `.env` file from the provided example.

**Linux / macOS:**

```bash
cp .env.example .env
```

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

Add your OpenAI API key:

```dotenv
OPENAI_API_KEY=your-api-key
```

The `.env` file must not be committed to Git.

The CLI and evaluation scripts that use OpenAI load this file through `python-dotenv`. The FastAPI application also needs the environment variable to be available when it starts.

## 3. Prepare Phalcon Knowledge Sources

The RAG pipeline uses official Phalcon 5.20 documentation and the matching Phalcon source code.

Create the raw data directory if necessary:

```bash
mkdir -p data/raw
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data/raw
```

### Documentation

Clone the official documentation repository:

```bash
git clone https://github.com/phalcon/documentation.git data/raw/phalcon-docs
```

The ingestion script expects the Phalcon 5.20 documentation at:

```text
data/raw/phalcon-docs/src/content/docs-5.20
```

### Zephir Source Code

Clone the Phalcon 5.20.0 source-code release:

```bash
git clone --branch v5.20.0 --depth 1 https://github.com/phalcon/cphalcon.git data/raw/phalcon-source-code
```

The source-code ingestion script expects Zephir files under:

```text
data/raw/phalcon-source-code/phalcon
```

Using matching framework versions helps keep documentation and implementation details aligned.

Before continuing, verify that both expected directories exist.

## 4. Ingest Knowledge Sources

### Documentation Ingestion

Run:

```bash
python scripts/ingest_docs.py
```

Expected output:

```text
data/processed/phalcon_docs_5.20.jsonl
```

The ingestion pipeline creates structure-aware documentation chunks and preserves metadata for source attribution.

### Source-Code Ingestion

Run:

```bash
python scripts/ingest_source_code.py
```

Expected output:

```text
data/processed/phalcon_source_code_5.20.jsonl
```

The source-code ingestion pipeline parses Zephir methods into chunks and preserves metadata including source file, method name, and line range.

## 5. Analyze Token Distribution

Optionally analyze the token lengths of the documentation chunks:

```bash
python scripts/analyze_tokens.py
```

The script reads:

```text
data/processed/phalcon_docs_5.20.jsonl
```

And writes:

```text
data/evaluation/results/token_analysis.json
```

The analysis covers the embedding models evaluated during development.

Ensure the output directory exists before running this script:

```bash
mkdir -p data/evaluation/results
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data/evaluation/results
```

## 6. Generate Embeddings

The runtime uses Qwen3-Embedding-0.6B for both documentation and source-code retrieval.

Generate documentation embeddings:

```bash
python scripts/generate_embeddings.py --model=qwen --source=phalcon_docs
```

Generate source-code embeddings:

```bash
python scripts/generate_embeddings.py --model=qwen --source=phalcon_source_code
```

Expected output files:

```text
data/embeddings/
├── phalcon_docs/
│   ├── qwen_embeddings.npy
│   └── chunk_ids.json
└── phalcon_source_code/
    ├── qwen_embeddings.npy
    └── chunk_ids.json
```

The `chunk_ids.json` files preserve the ordering of the corresponding embeddings.

The runtime checks that embedding IDs match the loaded corpus. If the corpus is regenerated or changed, regenerate its embeddings as well.

### Alternative Embedding Models

The following models were also evaluated during development:

```bash
python scripts/generate_embeddings.py --model=bge --source=phalcon_docs
python scripts/generate_embeddings.py --model=gte --source=phalcon_docs
python scripts/generate_embeddings.py --model=e5 --source=phalcon_docs
```

These alternative embeddings are used for comparative experiments. The final runtime configuration uses Qwen embeddings.

## 7. Run Retrieval Evaluations

The documentation retrieval benchmark is loaded from:

```text
data/evaluation/retrieval_questions.json
```

The multi-source benchmark is loaded from:

```text
data/evaluation/questions.json
```

Ensure the corresponding benchmark files exist before running evaluations.

### Embedding Model Evaluation

Evaluate each model after generating its documentation embeddings:

```bash
python scripts/evaluate_embeddings.py qwen
python scripts/evaluate_embeddings.py bge
python scripts/evaluate_embeddings.py gte
python scripts/evaluate_embeddings.py e5
```

These experiments evaluate retrieval quality using metrics such as Recall@5, Recall@10, and Mean Reciprocal Rank (MRR).

### BM25 Evaluation

Run the lexical retrieval baseline:

```bash
python scripts/evaluate_bm25.py
```

### Hybrid Retrieval Evaluation

Run the Qwen + BM25 hybrid retrieval evaluation:

```bash
python scripts/evaluate_hybrid.py
```

The hybrid retriever combines dense and lexical rankings using weighted Reciprocal Rank Fusion.

The current configuration uses a 4:1 dense-to-BM25 weighting.

### Cross-Encoder Reranking Evaluation

Run:

```bash
python scripts/evaluate_reranker.py
```

This script evaluates the configured MiniLM cross-encoder on hybrid retrieval candidates.

The script uses:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

### Multi-Source Retrieval Evaluation

Run:

```bash
python scripts/evaluate_retrieval.py
```

This script loads the complete RAG retrieval pipeline, retrieves documentation and source-code chunks, and compares the returned chunk IDs with expected sources.

**Retrieval evaluation configuration:** The current script evaluates all chunks returned by `pipeline.retrieve()`, using up to 2 documentation chunks and 5 source-code chunks per question. Metrics are reported as `Hit@7` and `Mean Expected Source Recall@7`. These reflect the current source-aware allocation rather than a globally ranked top-7.

Because the current pipeline may return up to **2 documentation + 5 source-code chunks**, those printed metrics are not necessarily strict Top-5 metrics.

To reproduce a strict historical Top-5 evaluation, the script must explicitly limit the evaluated chunks to five and use the corresponding historical retrieval configuration.

See [Evaluation Results](evaluation.md) for recorded experimental results and their limitations.

## 8. Run Generation Evaluation

Generation evaluation requires:

- Prepared documentation and source-code corpora.
- Qwen embeddings for both sources.
- The multi-source evaluation questions.
- A valid OpenAI API key.

Run:

```bash
python scripts/evaluate_generation.py
```

The script executes the retrieval and answer-generation pipeline for each evaluation question.

The current retrieval and generation workflow is:

```text
Documentation                  Source Code
      |                             |
 Dense + BM25                  Dense + BM25
      |                             |
 Weighted RRF                  Weighted RRF
      |                             |
      +-------------+---------------+
                    |
           Source-aware reranking
                    |
        Up to 2 docs + 5 source chunks
                    |
             LLM generation
                    |
             Generated answer
```

The script saves generated answers, retrieved chunks, expected concepts, and empty evaluation fields to:

```text
data/evaluation/results/generation.json
```

### Manual Scoring

The generation evaluation script **does not automatically score answers**.

Each generated answer must be reviewed manually against three criteria:

| Criterion    | Score range | Description                                       |
| ------------ | ----------- | ------------------------------------------------- |
| Correctness  | 0–2         | Are the claims correct?                           |
| Completeness | 0–2         | Does the answer cover the expected concepts?      |
| Groundedness | 0–2         | Is the answer supported by the retrieved context? |

For 30 questions, the maximum combined score is:

```text
30 questions × 3 criteria × 2 points = 180 points
```

The reported historical and later evaluation results are documented in [Evaluation Results](evaluation.md).

Running `evaluate_generation.py` creates a new set of generated answers; it does not automatically reproduce the previously reported manual scores.

## 9. Ask a Question

To query the RAG pipeline directly:

```bash
python scripts/answer_question.py "How do I create a transaction in Phalcon?"
```

Additional example questions:

```text
How can I bind parameters in a Phalcon query?
How do I register an event listener?
What is the difference between DI get() and getShared()?
```

The CLI initializes the RAG pipeline, retrieves and reranks relevant Phalcon chunks, and generates an answer.

## 10. Run the FastAPI Server

Start the server from the repository root.

**Linux / macOS:**

```bash
set -a
source .env
set +a

uvicorn phalcon_rag.api.app:app --host 127.0.0.1 --port 8000
```

**Windows PowerShell:**

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*OPENAI_API_KEY\s*=\s*(.+)\s*$') {
        $env:OPENAI_API_KEY = $Matches[1].Trim().Trim('"', "'")
    }
}

uvicorn phalcon_rag.api.app:app --host 127.0.0.1 --port 8000
```

The application initializes the RAG pipeline during startup. The first startup may take additional time while embedding and reranking models are downloaded or loaded.

Check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

The API is intended for local or private use. It does not currently provide authentication or rate limiting.

## 11. Required Files

After corpus preparation, the following files are required by the full runtime pipeline:

```text
data/processed/phalcon_docs_5.20.jsonl
data/processed/phalcon_source_code_5.20.jsonl

data/embeddings/phalcon_docs/qwen_embeddings.npy
data/embeddings/phalcon_docs/chunk_ids.json

data/embeddings/phalcon_source_code/qwen_embeddings.npy
data/embeddings/phalcon_source_code/chunk_ids.json
```

The retrieval and generation evaluation scripts additionally require the relevant benchmark JSON files under `data/evaluation/`.

Without the processed corpora and matching embeddings, the full RAG pipeline cannot initialize successfully.

## 12. Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Check linting:

```bash
ruff check .
```

Automatically fix supported lint issues:

```bash
ruff check . --fix
```

Check formatting without changing files:

```bash
ruff format --check .
```

Apply formatting:

```bash
ruff format .
```

Run automated tests:

```bash
pytest
```

## Related Documentation

- [Project README](../README.md)
- [Retrieval and Generation Evaluation](evaluation.md)
- [AWS Deployment](deployment.md)
