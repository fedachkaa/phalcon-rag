# Retrieval and Generation Evaluation

> **Historical results:** This document preserves earlier experiments, including five-chunk context allocations (2 documentation + 3 source-code chunks). The current application selects up to **2 documentation + 5 source-code chunks**. The reported generation scores of **164/180 (91.1%)** and **171/180 (95.0%)** come from separate manual evaluations. Their exact run configurations should be verified against saved artifacts before making a controlled comparison.

## 1. Documentation Ingestion and Chunking

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

## 2. Embedding Model Evaluation

Several embedding models were evaluated on a custom retrieval benchmark containing 30 Phalcon questions across multiple categories:

- how-to;
- conceptual;
- problem-oriented;
- API questions.

Evaluated models:

| Model | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| Qwen3-Embedding-0.6B | **0.833** | **0.900** | 0.587 |
| BGE Large EN v1.5 | 0.733 | 0.833 | **0.661** |
| GTE ModernBERT Base | 0.600 | 0.733 | 0.354 |
| Multilingual E5 Large Instruct | 0.733 | 0.867 | 0.468 |

`Qwen/Qwen3-Embedding-0.6B` was selected for dense retrieval because the first retrieval stage prioritizes recall: missing a relevant document at this stage prevents later reranking from recovering it.

## 3. BM25 Retrieval

A BM25 retriever was implemented as a lexical retrieval baseline.

| Retriever | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| BM25 | 0.467 | 0.567 | 0.347 |

BM25 performs worse than dense retrieval on its own, but it retrieves some relevant chunks that dense retrieval misses. This makes it useful as a complementary signal in hybrid retrieval.

## 4. Hybrid Retrieval

Dense Qwen retrieval and BM25 retrieval were combined using weighted Reciprocal Rank Fusion (RRF).

Several weighting strategies were evaluated:

| Qwen : BM25 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| Qwen only | 0.833 | 0.900 | 0.587 |
| 1 : 1 | 0.667 | 0.800 | 0.458 |
| 2 : 1 | 0.800 | 0.867 | 0.545 |
| 3 : 1 | 0.733 | **0.933** | 0.581 |
| **4 : 1** | 0.800 | **0.933** | **0.610** |

The final pipeline therefore uses a **4:1 weighting in favor of dense retrieval**.

The hybrid retriever collects the top 50 candidates from both retrieval methods, fuses their rankings, and passes the top 10 hybrid candidates to the reranker.

## 5. Cross-Encoder Reranking

Several cross-encoder reranking models were evaluated on top of the hybrid retrieval pipeline rather than selecting a reranker without comparison.

For each benchmark question, the retrieval pipeline produced candidates using:

```text
Dense Top 50 + BM25 Top 50
              ↓
         Weighted RRF
              ↓
        Hybrid Top 10
```

Each reranking model then reordered the same 10 candidates, allowing their impact on retrieval quality and inference latency to be compared under identical conditions.

The following rerankers were evaluated:

- `BAAI/bge-reranker-v2-m3`
- `BAAI/bge-reranker-base`
- `cross-encoder/ms-marco-MiniLM-L6-v2`

`BAAI/bge-reranker-v2-m3` achieved the strongest source-code retrieval quality, but its CPU inference cost was too high for an interactive RAG pipeline. In profiling, reranking alone could account for most of the request latency.

`cross-encoder/ms-marco-MiniLM-L6-v2` provided a much better latency-quality trade-off. It also performed particularly well on documentation retrieval at the evaluated cutoff.

To address lower source-code recall at Top 3, the source-code context allocation was subsequently increased from 3 to 5 chunks. A reported generation evaluation using the expanded allocation achieved:

| Metric | Score |
|---|---:|
| Correctness | 60/60 |
| Completeness | 51/60 |
| Groundedness | 60/60 |
| **Overall** | **171/180 (95.0%)** |

The current pipeline reranks documentation and source-code candidates separately and selects up to two documentation and five source-code chunks:

```text
Dense Top 50 + BM25 Top 50
              ↓
         Weighted RRF
              ↓
        Hybrid Top 10
              ↓
     MiniLM Cross-Encoder
          ↙         ↘
     Docs Top 2   Source Top 5
          ↘         ↙
     Up to 7 context chunks
              ↓
        Answer Generation
```

Unlike the first-stage retrievers, the cross-encoder evaluates the query and each candidate passage together. Since it is applied only to the small candidate set produced by hybrid retrieval, MiniLM provides a practical relevance/latency trade-off for CPU inference.

## 6. Source-Code Ingestion and Multi-Source Retrieval

The knowledge base was extended with the Phalcon 5.20 source code.

Zephir source files are parsed into method-level chunks while preserving metadata such as:

- source file;
- class/method name;
- start and end lines;
- chunk source type.

Documentation and source code use separate hybrid retrievers and query instructions.

The two ranked result lists are combined using Reciprocal Rank Fusion before source-aware reranking.

The **current application** selects up to:

- **2 documentation chunks**;
- **5 source-code chunks**.

An earlier five-chunk allocation (2 documentation + 3 source-code chunks) was chosen following retrieval evaluation on a 30-question benchmark containing documentation, implementation, and mixed questions. Those historical results are preserved in Section 8.

## 7. Answer Generation

The selected chunks (up to **7 total**) are provided to the LLM together with the user's question and, when supplied through the API, optional PHP code context. With code context, the application may rewrite the retrieval query before retrieving evidence.

The generation prompt distinguishes between:

- **DOCUMENTATION** — public API, documented behavior, and usage;
- **SOURCE CODE** — internal implementation and execution flow.

The model is instructed to:

- answer using only the supplied context;
- avoid relying on outside knowledge;
- cite supporting chunks;
- prioritize source-code evidence when the question asks how something works internally;
- avoid inventing details when the retrieved context is insufficient.

## 8. Multi-Source Retrieval Evaluation (Historical Five-Chunk Experiment)

The multi-source retrieval pipeline was evaluated on a 30-question benchmark covering documentation, internal implementation, and mixed questions.

Two metrics were reported for the **five-chunk configurations**:

- **Hit@5** — whether at least one expected source appeared in the final five chunks;
- **Expected Source Recall@5** — the proportion of expected sources recovered in the final five chunks.

Several documentation/source allocations were compared:

| Final context | Hit@5 | Mean Expected Source Recall@5 |
|---|---:|---:|
| 3 docs + 2 source | 80.0% | 65.0% |
| **2 docs + 3 source** | **86.7%** | 70.8% |
| 1 doc + 4 source | 86.7% | **71.9%** |

At that stage, **2 documentation + 3 source-code chunks** were selected as a balance between documentation coverage and implementation detail.

Subsequent experiments increased the source-code allocation to five chunks. The current application uses up to **2 documentation + 5 source-code chunks**. The Hit@5 and Expected Source Recall@5 values above **do not measure the current seven-chunk configuration**.

The evaluation also exposed difficult cases involving indirect source-code execution flows, where relevant methods cannot be identified from method names alone. These cases are useful targets for future retrieval improvements.

## 9. Initial Generation Evaluation (Historical)

Answer generation was manually evaluated on the same 30-question benchmark used for the historical multi-source retrieval evaluation.

Each generated answer was scored from 0 to 2 on three dimensions:

- **Correctness** — whether the claims made in the answer are correct;
- **Completeness** — whether the answer covers the expected concepts needed to answer the question;
- **Groundedness** — whether the answer is supported by the retrieved context without introducing unsupported claims.

| Metric | Score | Mean |
|---|---:|---:|
| Correctness | 60/60 | 2.00/2 |
| Completeness | 44/60 | 1.47/2 |
| Groundedness | 60/60 | 2.00/2 |
| **Overall** | **164/180** | **91.1%** |

In this evaluation, answers generally remained grounded in the supplied Phalcon documentation and source code, including cases where the retrieved context was incomplete.

The main limitation was **completeness**. Many incomplete answers were associated with missing or partial retrieval context rather than unsupported generation. For example, indirect source-code flows such as relationship resolution may require multiple connected methods that are difficult to retrieve from the query alone.

The evaluation also showed that strict expected-source retrieval metrics can underestimate final answer quality: in some cases, the exact expected chunk was not retrieved, while alternative retrieved context was still sufficient to produce a correct and complete answer.

Detailed per-question generation results and manual scores were recorded in the generation evaluation results file.

## 10. Subsequent Generation Evaluation

A later reported manual evaluation using an expanded source-code context produced:

| Metric | Score |
|---|---:|
| Correctness | 60/60 |
| Completeness | 51/60 |
| Groundedness | 60/60 |
| **Overall** | **171/180 (95.0%)** |

These results were manually assessed on the reported benchmark. The precise run settings and saved evaluation artifacts should be checked before interpreting the difference from the earlier **164/180 (91.1%)** score as a controlled improvement caused solely by increasing source-code context.

Neither score is an independent accuracy guarantee. The benchmark contains only 30 questions, and manual grading may introduce evaluator bias.
