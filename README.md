# Phalcon RAG

A Retrieval-Augmented Generation (RAG) system for answering technical questions about the Phalcon PHP framework using the official Phalcon documentation.

The project was built as a practical exploration of RAG architecture: document ingestion and domain-aware chunking, dense and keyword retrieval, hybrid search, reranking, answer generation, and evaluation of each stage.

The current knowledge base is built from the official Phalcon 5.20 documentation.

## How It Works

The current RAG pipeline is:

```text
Phalcon Documentation
        ↓
Document Ingestion & Cleaning
        ↓
Domain-Aware Chunking
        ↓
Dense Retrieval (Qwen3 Embedding)
        +
BM25 Keyword Retrieval
        ↓
Weighted Reciprocal Rank Fusion
        ↓
Top 10 Hybrid Candidates
        ↓
BGE Cross-Encoder Reranker
        ↓
Top 5 Chunks
        ↓
LLM Answer Generation
        ↓
Answer with Source Citations
```

The system intentionally combines semantic and keyword retrieval. Dense retrieval provides strong semantic recall, while BM25 can recover relevant documentation based on exact technical terminology. Their rankings are combined using weighted Reciprocal Rank Fusion (RRF) before reranking.

---

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

| Retriever | Recall@5 | Recall@10 |   MRR |
| --------- | -------: | --------: | ----: |
| BM25      |    0.467 |     0.567 | 0.347 |

BM25 performs worse than dense retrieval on its own, but it retrieves some relevant chunks that dense retrieval misses. This makes it useful as a complementary signal in hybrid retrieval.

### 4. Hybrid Retrieval

Dense Qwen retrieval and BM25 retrieval were combined using weighted Reciprocal Rank Fusion.

Several weighting strategies were evaluated:

| Qwen : BM25 | Recall@5 | Recall@10 |       MRR |
| ----------- | -------: | --------: | --------: |
| Qwen only   |    0.833 |     0.900 |     0.587 |
| 1 : 1       |    0.667 |     0.800 |     0.458 |
| 2 : 1       |    0.800 |     0.867 |     0.545 |
| 3 : 1       |    0.733 | **0.933** |     0.581 |
| **4 : 1**   |    0.800 | **0.933** | **0.610** |

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

| Reranker                    | Recall@5 | Recall@10 |     MRR |
| --------------------------- | -------: | --------: | ------: |
| **BAAI/bge-reranker-v2-m3** |  **...** |   **...** | **...** |

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

### 6. Answer Generation

The five highest-ranked chunks are provided to an LLM together with the user's question.

The generation prompt instructs the model to:

- answer using only the supplied context;
- avoid relying on outside knowledge;
- cite the documentation sources supporting the answer;
- avoid inventing information when the retrieved context is insufficient.

This means retrieval failure should preferably result in an incomplete or cautious answer rather than a hallucinated technical answer.

### 7. End-to-End RAG Evaluation

The complete RAG pipeline was manually evaluated using the same 30-question benchmark.

Four dimensions were scored from 0 to 2:

| Metric             | What it measures                                             |
| ------------------ | ------------------------------------------------------------ |
| Correctness        | Whether the technical claims are correct                     |
| Completeness       | Whether the important parts of the question are answered     |
| Groundedness       | Whether claims are supported by retrieved context            |
| Citation Precision | Whether citations directly support the claims they accompany |

The evaluation produced a score of approximately **98% of the maximum possible score**.

The evaluation also exposed useful failure cases. In particular, some questions demonstrated that when retrieval does not provide enough information, the generation stage correctly avoids fabricating unsupported details.

Evaluation datasets and results are available under:

```text
data/evaluation/
├── retrieval_questions.json
└── results/
```

---

## Project Structure

```text
src/phalcon_rag/
├── ingestion/       # Documentation loading, cleaning and chunking
├── retrieval/       # Dense, BM25 and hybrid retrieval
├── reranking/       # Cross-encoder reranking
├── generation/      # Prompt construction and answer generation
├── evaluation/      # Reusable evaluation utilities
├── config.py        # RAG pipeline configuration
├── models.py        # Core data models
└── pipeline.py      # End-to-end RAG pipeline

scripts/
├── ingest_docs.py
├── analyze_tokens.py
├── evaluate_embeddings.py
├── evaluate_bm25.py
├── evaluate_hybrid.py
├── evaluate_reranker.py
├── evaluate_generation.py
└── answer_question.py
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

## The `.env` file is excluded from Git and must never be committed.

## Usage

### Ingest Documentation

Run the documentation ingestion pipeline:

```bash
python scripts/ingest_docs.py
```

This processes the Phalcon documentation and creates the chunked corpus used by the retrieval pipeline.

### Analyze Token Distribution

```bash
python scripts/analyze_tokens.py
```

The script evaluates chunk token lengths against the context limits of the embedding models.

### Run Retrieval Evaluations

Individual retrieval stages can be evaluated using:

```bash
python scripts/evaluate_embeddings.py
python scripts/evaluate_bm25.py
python scripts/evaluate_hybrid.py
python scripts/evaluate_reranker.py
```

### Run Generation Evaluation

```bash
python scripts/evaluate_generation.py
```

This runs the complete retrieval → reranking → generation pipeline against the evaluation dataset.

### Ask a Question

To use the RAG pipeline interactively:

```bash
python scripts/answer_question.py
```

Example questions:

```text
How do I create a transaction in Phalcon?

How can I bind parameters in a Phalcon query?

How do I register an event listener?
```

The system retrieves relevant documentation, reranks the results, and generates an answer grounded in the retrieved Phalcon sources.

---

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

The current version focuses on building and evaluating a complete RAG pipeline over the Phalcon documentation.

The next stage is to expand the knowledge base with the **Phalcon source code**.

Planned work includes:

- source code ingestion;
- source-aware chunking designed specifically for PHP code;
- retrieval across both documentation and source code;
- distinguishing documentation and implementation sources in generated answers;
- additional evaluation for source-code questions;
- a simple web interface for asking questions.

Possible later improvements include:

- conversation history;
- streaming responses;
- Phalcon version filtering;
- support for multiple repositories or framework versions;
- IDE integration;
- improved citation handling.

The goal is to evolve the project from a documentation RAG prototype into a practical **Phalcon development assistant** capable of reasoning over both documentation and implementation details.
