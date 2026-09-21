# Phalcon RAG

A Retrieval-Augmented Generation (RAG) system for answering technical questions about the Phalcon PHP framework using its official documentation and source code.

The project was built as a practical exploration of RAG architecture: document ingestion and domain-aware chunking, dense and keyword retrieval, hybrid search, reranking, answer generation, and evaluation of each stage.

The current knowledge base combines the official Phalcon 5.20 documentation with the corresponding Phalcon 5.20 source code.

## How It Works

The current RAG pipeline retrieves from two knowledge sources:

- official Phalcon 5.20 documentation;
- Phalcon 5.20 source code.

```text
Phalcon Documentation        Phalcon Source Code
        ↓                            ↓
Documentation Ingestion      Source-Code Ingestion
        ↓                            ↓
Structured Chunks            Method-Level Chunks
        ↓                            ↓
Dense Retrieval + BM25       Dense Retrieval + BM25
        ↓                            ↓
Weighted RRF                 Weighted RRF
        └──────────────┬─────────────┘
                       ↓
              Multi-Source Retrieval
                       ↓
          Source-Aware Cross-Encoder
                       ↓
        2 Documentation + 3 Source Chunks
                       ↓
              LLM Answer Generation
                       ↓
          Answer with Source Citations
```

Documentation and source code are retrieved independently and fused at the multi-source layer. Reranking is performed separately per source type so that documentation and implementation evidence both remain represented in the final context.

## Project Stages and Results

### 1. Documentation Ingestion and Chunking

The Phalcon 5.20 documentation was loaded from the official documentation repository and converted into a structured corpus.

Instead of splitting documents using a fixed character or token window, the ingestion pipeline uses domain-aware chunking based on documentation structure.

It handles, among other things:

- headings and sections;
- API definitions;
- method signatures;
- code examples;
- tables;
- annotations;
- query builder examples;
- oversized sections.

The resulting corpus contains approximately **10.8k chunks from 151 documentation files**.

Each chunk preserves metadata such as its source file and documentation section so retrieved information can later be traced back to its source.

### 2. Embedding Model Evaluation

Several embedding models were evaluated on a custom retrieval benchmark containing 30 Phalcon questions across multiple categories:

- how-to;
- conceptual;
- problem-oriented;
- API questions.

Evaluated models:

| Model                          |  Recall@5 | Recall@10 |       MRR |
| ------------------------------ | --------: | --------: | --------: |
| Qwen3-Embedding-0.6B           | **0.833** | **0.900** |     0.587 |
| BGE Large EN v1.5              |     0.733 |     0.833 | **0.661** |
| GTE ModernBERT Base            |     0.600 |     0.733 |     0.354 |
| Multilingual E5 Large Instruct |     0.733 |     0.867 |     0.468 |

`Qwen/Qwen3-Embedding-0.6B` was selected for dense retrieval because the first retrieval stage prioritizes recall: missing a relevant document at this stage prevents later reranking from recovering it.

### 3. BM25 Retrieval

A BM25 retriever was implemented as a lexical retrieval baseline.

Results:

| Retriever  | Recall@5 | Recall@10 |   MRR |
| ---------- | -------: | --------: | ----: |
| BM25       |    0.467 |     0.567 | 0.347 |

BM25 performs worse than dense retrieval on its own, but it retrieves some relevant chunks that dense retrieval misses. This makes it useful as a complementary signal in hybrid retrieval.

### 4. Hybrid Retrieval

Dense Qwen retrieval and BM25 retrieval were combined using weighted Reciprocal Rank Fusion.

Several weighting strategies were evaluated:

| Qwen : BM25 | Recall@5 | Recall@10 |       MRR |
| ----------- | -------: | --------: | --------: |
| Qwen only   |    0.833 |     0.900 |     0.587 |
| 1 : 1       |    0.667 |     0.800 |     0.458 |
| 2 : 1       |    0.800 |     0.867 |     0.545 |
| 3 : 1       |    0.733 | **0.933** |     0.581 |
| **4 : 1**   |    0.800 | **0.933** | **0.610** |

The final pipeline therefore uses a **4:1 weighting in favor of dense retrieval**.

The hybrid retriever collects the top 50 candidates from both retrieval methods, fuses their rankings, and passes the top 10 hybrid candidates to the reranker.

### 5. Cross-Encoder Reranking

Several cross-encoder reranking models were evaluated on top of the hybrid retrieval pipeline rather than selecting a reranker without comparison.

For each benchmark question, the retrieval pipeline produced candidates using:

```text
Dense Top 50 + BM25 Top 50
              ↓
         Weighted RRF
              ↓
        Hybrid Top 10
```

Each reranking model then reordered the same 10 candidates, allowing their impact on retrieval quality to be compared under identical conditions.

Based on the evaluation results, `BAAI/bge-reranker-v2-m3` was selected for the final pipeline.

Unlike the first-stage retrievers, the cross-encoder evaluates the query and each candidate passage together. This is more computationally expensive, but allows for more precise relevance scoring over the small candidate set.

The final retrieval pipeline is therefore:

```text
Dense Top 50 + BM25 Top 50
              ↓
         Weighted RRF
              ↓
        Hybrid Top 10
              ↓
   BGE Cross-Encoder Reranker
              ↓
            Top 5
```

### 6. Source-Code Ingestion and Multi-Source Retrieval

The knowledge base was extended with the Phalcon 5.20 source code.

Zephir source files are parsed into method-level chunks while preserving metadata such as:

- source file;
- class/method name;
- start and end lines;
- chunk source type.

Documentation and source code use separate hybrid retrievers and query instructions.

The two ranked result lists are then combined using Reciprocal Rank Fusion before source-aware reranking.

The final context currently keeps:

- **2 documentation chunks**;
- **3 source-code chunks**.

This balance was selected after retrieval evaluation on a 30-question benchmark containing documentation, implementation, and mixed questions.

### 7. Answer Generation

The final five chunks are provided to the LLM together with the user's question.

The generation prompt distinguishes between:

- **DOCUMENTATION** — public API, documented behavior, and usage;
- **SOURCE CODE** — internal implementation and execution flow.

The model is instructed to:

- answer using only the supplied context;
- avoid relying on outside knowledge;
- cite supporting chunks;
- prioritize source-code evidence when the question asks how something works internally;
- avoid inventing details when the retrieved context is insufficient.

### 8. Multi-Source Retrieval Evaluation

The multi-source retrieval pipeline was evaluated on a 30-question benchmark covering documentation, internal implementation, and mixed questions.

Two metrics are reported:

- Hit@5 — whether at least one expected source appears in the final five chunks;
- Expected Source Recall@5 — the proportion of expected sources recovered in the final five chunks.

Several documentation/source allocations were compared:
| Final Context | Hit@5 | Mean Expected Source Recall@5 |
| --------------------- | --------: | ----------------------------: |
| 3 docs + 2 source | 80.0% | 65.0% |
| **2 docs + 3 source** | **86.7%** | 70.8% |
| 1 doc + 4 source | 86.7% | **71.9%** |

The final pipeline currently uses 2 documentation chunks and 3 source-code chunks as a balance between documentation coverage and implementation detail.

The evaluation also exposed difficult cases involving indirect source-code execution flows, where relevant methods cannot be identified from method names alone. These cases are useful targets for future retrieval improvements.

---

## Project Structure

```text
src/phalcon_rag/
├── ingestion/
├── retrieval/
│   ├── dense.py
│   ├── bm25.py
│   ├── hybrid.py
│   ├── multi_source.py
│   └── rrf.py
├── reranking/
├── generation/
├── evaluation/
├── config.py
├── models.py
├── pipeline.py
└── pipeline_factory.py
```

---

## Installation

The project requires **Python 3.11+**.

Clone the repository:

```bash
git clone https://github.com/fedachkaa/phalcon-rag.git
cd phalcon-rag
```

Create and activate a virtual environment.

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project:

```bash
pip install -e .
```

For development tools such as Ruff and pytest:

```bash
pip install -e ".[dev]"
```

---

## Configuration

Create a local `.env` file from the provided example:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Add your OpenAI API key:

```dotenv
OPENAI_API_KEY=your-api-key
```

The `.env` file is excluded from Git and must never be committed.

## Usage

The commands below assume the project has been installed and the virtual environment is activated.

### 1. Prepare Phalcon Knowledge Sources

The RAG pipeline uses both the official Phalcon 5.20 documentation and the Phalcon 5.20 source code.

Clone the documentation repository:

```bash
git clone https://github.com/phalcon/documentation.git data/raw/phalcon-docs
```

The Phalcon 5.20 documentation should then be available at:

```text
data/raw/phalcon-docs/src/content/docs-5.20
```

Clone the matching Phalcon source-code version:

```bash
git clone --branch v5.20.0 --depth 1 https://github.com/phalcon/cphalcon.git data/raw/phalcon-source-code
```

The Zephir source files used by the ingestion pipeline will then be available at:

```text
data/raw/phalcon-source-code/phalcon
```

Using the same Phalcon version for both sources keeps documentation and implementation details aligned.

### 2. Ingest Knowledge Sources

Process the documentation:

```bash
python scripts/ingest_docs.py
```

This creates:

```text
data/processed/phalcon_docs_5.20.jsonl
```

Process the Phalcon source code:

```bash
python scripts/ingest_source_code.py
```

This creates:

```text
data/processed/phalcon_source_code_5.20.jsonl
```

The source-code ingestion pipeline parses Zephir files into method-level chunks and preserves metadata such as the source file, method name, line range, and neighboring chunks.

### 3. Analyze Token Distribution

Optionally inspect chunk token lengths for the supported embedding models:

```bash
python scripts/analyze_tokens.py
```

This step requires the processed corpus created during ingestion.

### 4. Generate Embeddings

Generate document embeddings before running dense or hybrid retrieval:

```bash
python scripts/generate_embeddings.py --model=qwen --source=phalcon_docs

python scripts/generate_embeddings.py --model=qwen --source=phalcon_source_code
```

The script creates:

```text
data/embeddings/phalcon_docs/qwen_embeddings.npy
data/embeddings/phalcon_docs/chunk_ids.json

data/embeddings/phalcon_source_code/qwen_embeddings.npy
data/embeddings/phalcon_source_code/chunk_ids.json
```

To generate embeddings for the other evaluated models:

```bash
python scripts/generate_embeddings.py --model=bge --source=phalcon_docs

python scripts/generate_embeddings.py --model=gte --source=phalcon_docs

python scripts/generate_embeddings.py --model=e5 --source=phalcon_docs
```

### 5. Run Retrieval Evaluations

Embedding models can be evaluated after their embeddings have been generated:

```bash
python scripts/evaluate_embeddings.py qwen
python scripts/evaluate_embeddings.py bge
python scripts/evaluate_embeddings.py gte
python scripts/evaluate_embeddings.py e5
```

The final RAG pipeline requires Qwen embeddings for both documentation and source-code corpora.

Run the BM25 retrieval evaluation:

```bash
python scripts/evaluate_bm25.py
```

Run the hybrid dense + BM25 retrieval evaluation:

```bash
python scripts/evaluate_hybrid.py
```

Run the cross-encoder reranking evaluation:

```bash
python scripts/evaluate_reranker.py
```

Run the multi-source retrieval evaluation:

```bash
python scripts/evaluate_retrieval.py
```

This evaluates the final retrieved context using Hit@5 and Expected Source Recall@5 over the multi-source benchmark.

The hybrid and reranking stages require the generated Qwen embeddings.

### 6. Run Generation Evaluation

Before running generation, create a local `.env` file and provide an OpenAI API key:

```dotenv
OPENAI_API_KEY=your-api-key
```

Then run:

```bash
python scripts/evaluate_generation.py
```

This evaluates the complete pipeline:

```text
Dense Retrieval
    +
BM25 Retrieval
    ↓
Weighted RRF
    ↓
Cross-Encoder Reranking
    ↓
LLM Answer Generation
```

### 7. Ask a Question

To query the RAG pipeline directly:

```bash
python scripts/answer_question.py "How do I create a transaction in Phalcon?"
```

Example questions:

```text
How do I create a transaction in Phalcon?
How can I bind parameters in a Phalcon query?
How do I register an event listener?
```

The system retrieves relevant documentation and source-code chunks, reranks them separately by source type, and generates an answer grounded in the retrieved Phalcon context.

### Required Files for the Full Pipeline

After the preparation steps, the following files are required:

```text
data/processed/phalcon_docs_5.20.jsonl
data/processed/phalcon_source_code_5.20.jsonl

data/embeddings/phalcon_docs/qwen_embeddings.npy
data/embeddings/phalcon_docs/chunk_ids.json

data/embeddings/phalcon_source_code/qwen_embeddings.npy
data/embeddings/phalcon_source_code/chunk_ids.json
```

Without these files, hybrid retrieval, reranking, generation evaluation, and interactive question answering cannot run.

## Development

Run Ruff:

```bash
ruff check .
```

Automatically fix supported lint issues:

```bash
ruff check . --fix
```

Format the project:

```bash
ruff format .
```

Run tests:

```bash
pytest
```

---

## Planned Improvements

The current version supports retrieval over both Phalcon documentation and source code.

Planned work includes:

- generation evaluation over the expanded multi-source benchmark;
- improved retrieval for indirect source-code execution flows;
- API layer for external clients;
- IDE integration, starting with a PhpStorm plugin;
- improved citation presentation;
- Phalcon version filtering.

Possible later improvements include:

- conversation history;
- streaming responses;
- support for multiple framework versions;
- additional repositories or project-specific knowledge sources.
