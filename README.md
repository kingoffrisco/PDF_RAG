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
8. [Notebooks](#notebooks)
9. [Running Tests](#running-tests)
10. [Evaluation](#evaluation)
11. [Enterprise Considerations](#enterprise-considerations)
12. [Contributing](#contributing)

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
│   ├── 01_setup_and_config.py  # Install deps, create UC schema & VS endpoint
│   ├── 02_pdf_ingestion.py     # Load PDFs → Delta table → Vector index
│   ├── 03_rag_query_interface.py  # Interactive Q&A
│   └── 04_evaluation.py        # RAGAS metrics + MLflow logging
├── src/pdf_rag/
│   ├── ingestion/
│   │   ├── pdf_loader.py       # PDF → LangChain Documents (local/DBFS/Volume)
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
├── tests/                      # pytest test suite (53 tests)
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

### Step 2 – Run the setup notebook

Open **`notebooks/01_setup_and_config.py`** and edit the variables at the top:

```python
CATALOG        = "main"
SCHEMA         = "pdf_rag"
VS_ENDPOINT    = "pdf_rag_endpoint"
PDF_SOURCE_DIR = "dbfs:/mnt/raw_pdfs"
```

Run all cells. This creates the Unity Catalog schema and the Vector Search endpoint.

### Step 3 – Ingest your PDFs

Run **`notebooks/02_pdf_ingestion.py`**.  
The notebook:
1. Loads PDFs from `PDF_SOURCE_DIR`.
2. Chunks, embeds, and writes them to a Delta table.
3. Creates (or refreshes) the Vector Search index.

### Step 4 – Ask questions

Run **`notebooks/03_rag_query_interface.py`** with your questions.

### Step 5 – Evaluate quality (optional)

Run **`notebooks/04_evaluation.py`** with your ground-truth Q&A pairs to get
RAGAS faithfulness / relevancy / precision / recall scores, logged to MLflow.

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
| `databricks` | `databricks-bge-large-en` | `databricks-dbrx-instruct` | Databricks PAT | Free on DBX Community / pay-as-you-go |
| `openai` | `text-embedding-3-small` | `gpt-4o-mini` | `OPENAI_API_KEY` | Paid |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_setup_and_config.py` | One-time cluster setup: install libs, create UC schema & VS endpoint |
| `02_pdf_ingestion.py` | Ingest PDFs → Delta table → Vector Search index |
| `03_rag_query_interface.py` | Interactive Q&A with source citations |
| `04_evaluation.py` | RAGAS metrics evaluation + MLflow logging |

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
