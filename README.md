# PDF RAG – Enterprise Retrieval-Augmented Generation for PDF Documents on Databricks

> **An end-to-end, production-ready RAG pipeline** that ingests PDF documents,
> stores embeddings in Databricks Vector Search (or local FAISS/Chroma), and
> answers natural-language questions with full source citations.
> Designed for the Databricks free-tier _and_ for commercial enterprise deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Quick Start – Local Development](#quick-start--local-development)
5. [Databricks Deployment](#databricks-deployment)
6. [Configuration Reference](#configuration-reference)
7. [LLM & Embedding Backends](#llm--embedding-backends)
8. [Hybrid Reranking](#hybrid-reranking)
9. [Notebooks](#notebooks)
10. [Running Tests](#running-tests)
11. [Evaluation](#evaluation)
12. [Enterprise Considerations](#enterprise-considerations)
13. [Contributing](#contributing)

---

## Overview

PDF RAG solves three problems:

| Problem | Solution |
|---------|---------|
| How do I ask questions about a large collection of PDFs? | Ingest them into a vector store; retrieve the most relevant passages at query time. |
| How do I run this for free on Databricks? | HuggingFace embeddings + local FAISS on a single-node cluster, or Databricks Community Edition. |
| How do I scale this commercially? | Switch to Databricks Vector Search + Foundation Model APIs with one config-file change. |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                  │
│  PDF Files         Text         Chunks      Embeddings  Vector   │
│  (DBFS /  ──────► Extraction ──► Chunking ──► Model   ──► Store  │
│   Volume)         pdfplumber    LangChain   (HF / DB   (FAISS /  │
│                                  Splitter    / OpenAI)  DB VS)   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                           │
│                                                                  │
│  Question ──► Embeddings ──► Dense Search ──┐                   │
│                              (Vector Store) │                    │
│                                             ├──► Hybrid Merge   │
│             BM25 (optional) ───────────────┘   + Re-rank        │
│                                                      │           │
│                                              Top-K Chunks        │
│                                                      │           │
│                                          ┌───────────▼────────┐ │
│                                          │   LLM (DBRX /      │ │
│                                          │   GPT-4o / HF)     │ │
│                                          └───────────┬────────┘ │
│                                                      │           │
│                                           Answer + Citations     │
│                                           (MLflow logged)        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
PDF_RAG/
├── config/
│   ├── config.yaml              # Default (local/HuggingFace) settings
│   └── databricks_config.yaml  # Databricks cluster settings
├── notebooks/
│   └── PDF - RAG - Retrieval Augmented Generation.py  # All-in-one Databricks notebook
├── src/pdf_rag/
│   ├── ingestion/
│   │   ├── pdf_loader.py       # PDF → LangChain Documents (local/S3/DBFS/Volume)
│   │   └── text_chunker.py     # Recursive character splitting + metadata
│   ├── embeddings/
│   │   └── embedder.py         # Databricks / OpenAI / HuggingFace factory
│   ├── vector_store/
│   │   ├── databricks_vs.py    # Databricks Vector Search wrapper
│   │   └── local_vs.py         # FAISS / Chroma for local dev
│   ├── retrieval/
│   │   └── retriever.py        # Hybrid BM25 + dense + optional re-ranker
│   ├── generation/
│   │   └── llm_chain.py        # RAG chain, prompt, LLM factory
│   ├── pipeline/
│   │   ├── ingestion_pipeline.py  # Orchestrates ingestion end-to-end
│   │   └── rag_pipeline.py        # Orchestrates query end-to-end
│   └── utils/
│       ├── config.py           # YAML config + env-var overrides
│       └── logger.py           # Structured logging
├── tests/                      # pytest test suite
├── pyproject.toml
└── requirements.txt
```

---

## Quick Start – Local Development

### Prerequisites

* Python 3.10+

### Install

```bash
git clone https://github.com/kingoffrisco/PDF_RAG.git
cd PDF_RAG

# Core + local dev extras (FAISS + HuggingFace embeddings, no API keys needed)
pip install -e ".[local,dev]"
```

### Ingest PDFs

```python
from pdf_rag.pipeline.ingestion_pipeline import IngestionPipeline

# Uses config/config.yaml by default:
#   embedding.backend = huggingface
#   vector_store.local_backend = faiss
pipeline = IngestionPipeline.from_config()
chunks = pipeline.run("/path/to/your/pdfs/")
print(f"Indexed {len(chunks)} chunks.")
```

### Query

```python
from pdf_rag.pipeline.rag_pipeline import RAGPipeline

# Point at a persisted FAISS index
rag = RAGPipeline.from_ingestion_pipeline(pipeline, corpus_documents=chunks)

answer = rag.query("What are the key findings in the report?")
print(answer)

# With source documents
result = rag.query_with_sources("What risks are identified?")
print(result["answer"])
for doc in result["sources"]:
    print(f"  → {doc.metadata['file_name']}, page {doc.metadata['page_number']}")
```

---

## Databricks Deployment

### Step 1 – Clone the repo into Databricks Repos

In your Databricks workspace:
```
Repos → Add Repo → https://github.com/kingoffrisco/PDF_RAG.git
```

### Step 2 – Open the notebook

Open **`notebooks/PDF - RAG - Retrieval Augmented Generation.py`**.

### Step 3 – Configure (Sections 2 & 6)

Edit the variables at the top of the configuration cells:

```python
CATALOG        = "main"            # Unity Catalog catalog
SCHEMA         = "pdf_rag"         # Schema (created if missing)
TABLE_NAME     = "document_chunks" # Delta table for chunk storage
VS_ENDPOINT    = "pdf_rag_endpoint"# Vector Search endpoint name
PDF_SOURCE_DIR = "s3://your-bucket/your-pdfs/"  # S3, Volume, or local path
REPO_PATH      = "/Workspace/Repos/your_email@example.com/PDF_RAG"
```

PDF sources supported:
- **S3**: `s3://your-bucket/path/` (files are staged to a Unity Catalog Volume automatically)
- **Unity Catalog Volume**: `/Volumes/catalog/schema/volume/`
- **Local/DBFS path** (non-serverless only)

### Step 4 – Run sections 1–10 (Ingestion)

Run cells in order through **Section 10** to:
1. Install dependencies.
2. Create the Unity Catalog schema and Vector Search endpoint.
3. Set the MLflow experiment.
4. Load and chunk your PDFs.
5. Generate embeddings and write chunks to a Delta table.
6. Create (or refresh) the Vector Search index.

### Step 5 – Run sections 11–15 (Q&A)

Run through **Section 15** to build the hybrid reranking retriever and RAG chain, then ask questions interactively or in batch.

### Step 6 – Run sections 16–20 (Evaluation, optional)

Run through **Section 20** to compute RAGAS metrics and log results to MLflow.

### Step 7 – Deploy Gradio app (optional, Section 21)

Follow the deployment instructions in **Section 21** to launch a Gradio chat interface as a Databricks App or locally in the notebook.

### Switching backends with one config change

| Scenario | `config/config.yaml` change |
|----------|---------------------------|
| Local dev (free) | `embedding.backend: huggingface`, `llm.backend: huggingface` |
| Databricks free tier | `embedding.backend: databricks`, `llm.backend: databricks` |
| OpenAI | `embedding.backend: openai`, `llm.backend: openai` |

Or use environment variables (no config file edits needed):
```bash
export PDF_RAG__LLM__BACKEND=openai
export PDF_RAG__EMBEDDING__BACKEND=openai
export OPENAI_API_KEY=sk-...
```

---

## Configuration Reference

All settings live in `config/config.yaml`.  
Override any value with an environment variable:  
`PDF_RAG__<SECTION>__<KEY>=value`

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `embedding` | `backend` | `huggingface` | `databricks` \| `openai` \| `huggingface` |
| `embedding` | `model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Model / endpoint name |
| `llm` | `backend` | `databricks` | `databricks` \| `openai` \| `huggingface` |
| `llm` | `model_name` | `databricks-dbrx-instruct` | Model / endpoint name |
| `llm` | `temperature` | `0.0` | Sampling temperature |
| `llm` | `max_tokens` | `1024` | Max response tokens |
| `chunking` | `chunk_size` | `1000` | Characters per chunk |
| `chunking` | `chunk_overlap` | `200` | Overlap between chunks |
| `retrieval` | `dense_k` | `10` | Dense search candidates |
| `retrieval` | `bm25_k` | `10` | BM25 candidates |
| `retrieval` | `final_k` | `5` | Docs fed to LLM |
| `retrieval` | `reranker_model` | `null` | Cross-encoder model (optional) |
| `vector_store` | `type` | `local` | `local` \| `databricks` |
| `vector_store` | `local_backend` | `faiss` | `faiss` \| `chroma` |
| `vector_store.databricks` | `catalog` | `main` | UC catalog |
| `vector_store.databricks` | `schema` | `pdf_rag` | UC schema |
| `vector_store.databricks` | `vector_search_endpoint` | `pdf_rag_endpoint` | VS endpoint name |
| `mlflow` | `enabled` | `true` | Enable LangChain autologging |

---

## LLM & Embedding Backends

| Backend | Embedding model | LLM | API key required | Cost |
|---------|----------------|-----|-----------------|------|
| `huggingface` | `all-MiniLM-L6-v2` (local) | `zephyr-7b-beta` (local) | No | Free |
| `databricks` | `databricks-bge-large-en` | `databricks-meta-llama-3-3-70b-instruct` | Databricks PAT | Free on DBX Community / pay-as-you-go |
| `openai` | `text-embedding-3-small` | `gpt-4o-mini` | `OPENAI_API_KEY` | Paid |

---

## Hybrid Reranking

The notebook implements a two-stage retrieval strategy that improves answer quality:

1. **Dense retrieval** — Databricks Vector Search returns the top 15 candidate chunks.
2. **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (query, chunk) pair and reranks them.
3. **Adaptive fallback** — Reranking is applied when the top cross-encoder score exceeds 0.3 or score variance is high; otherwise the original vector search order is kept.

The top 10 chunks from the reranker are passed to the LLM.

---

## Notebooks

The repository contains a single all-in-one Databricks notebook:

**`notebooks/PDF - RAG - Retrieval Augmented Generation.py`**

| Section | Purpose |
|---------|---------|
| 1 – Install dependencies | `%pip install` all required packages and restart Python |
| 2 – Configuration | Set catalog, schema, endpoint, PDF source, LLM/embedding backend |
| 3 – Unity Catalog schema | Create catalog and schema if missing |
| 4 – Verify Vector Search endpoint | Create or confirm VS endpoint existence |
| 5 – Set MLflow experiment | Configure experiment path for run tracking |
| 6 – Parameters (ingestion) | Chunk size, overlap, S3/Volume source path, repo path |
| 7 – Load & chunk PDFs | Load from S3, Volume, or local path; split into chunks |
| 8 – Generate embeddings | Embed chunks with `databricks-bge-large-en` |
| 9 – Write to Delta table | Persist chunks + vectors to Unity Catalog Delta table |
| 10 – Create / refresh Vector Search index | Build or sync the Vector Search index |
| 11 – Parameters (Q&A) | LLM model, retrieval k, repo path |
| 12 – Build retriever | Connect to Databricks Vector Search |
| 13 – Hybrid reranking retriever | Cross-encoder reranking layer (15 → 10 chunks) |
| 14 – Build RAG chain | Assemble LLM + reranked retriever + MLflow logging |
| 15 – Interactive Q&A | Single question with source citations |
| 16 – Batch questions | Loop over multiple questions |
| 17 – Parameters (evaluation) | Evaluation LLM and embedding config |
| 18 – Define evaluation questions | 10 domain-specific Q&A pairs |
| 19 – Run RAG chain on eval set | Generate answers for all eval questions |
| 20 – Compute RAGAS metrics | Faithfulness, relevancy, precision, recall |
| 21 – Log to MLflow | Persist metric scores and raw results |
| 22 – Deploy Gradio app | Launch a chat UI as a Databricks App or in-notebook |

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[local,dev]"
pip install pdfplumber reportlab rank-bm25

# Run the full test suite
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/pdf_rag --cov-report=term-missing
```

---

## Evaluation

Quality is measured with [RAGAS](https://github.com/explodinggradients/ragas):

| Metric | Target | Description |
|--------|--------|-------------|
| `faithfulness` | > 0.85 | Answer grounded in retrieved context |
| `answer_relevancy` | > 0.80 | Answer relevant to question |
| `context_precision` | > 0.75 | Retrieved chunks are precise |
| `context_recall` | > 0.75 | Retrieved chunks are complete |

All evaluation runs are tracked in MLflow for trend analysis.

---

## Enterprise Considerations

| Concern | How it's addressed |
|---------|-------------------|
| **Data governance** | PDFs and chunks stored in Unity Catalog with column-level permissions |
| **Security** | No secrets in code; all credentials via env vars or Databricks Secrets |
| **Scalability** | Databricks Vector Search scales horizontally; Delta table handles millions of chunks |
| **Reproducibility** | All experiments tracked in MLflow; model serving via MLflow endpoints |
| **Observability** | Structured logging + MLflow LangChain autologging on every query |
| **Cost control** | Free-tier: HuggingFace models, no API calls; Commercial: Foundation Model APIs |
| **Extensibility** | Swap any component (embedder / LLM / vector store) with one config change |

---

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes with tests.
3. Run `pytest tests/` – all tests must pass.
4. Open a pull request.

---

## License

MIT – see [LICENSE](LICENSE).
