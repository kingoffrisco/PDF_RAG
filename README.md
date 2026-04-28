# Energy Plan PDF RAG – Question Answering System on Databricks

> **A production-ready RAG system** for asking natural language questions about electricity plans and energy documents. Built with Databricks Vector Search, Foundation Model APIs, and LangChain. Features both an interactive Streamlit app and notebook interface.

---

## Table of Contents

1. [Overview](#overview)
2. [System Status](#system-status)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Repository Structure](#repository-structure)
6. [Deployment Options](#deployment-options)
7. [Configuration](#configuration)
8. [How It Works](#how-it-works)
9. [Development](#development)

---

## Overview

This system solves the problem of quickly finding information across a large collection of PDF documents about electricity plans, energy rates, and solar buyback programs.

**What it does:**
* Ingests PDF documents and extracts text
* Chunks documents and generates embeddings
* Stores embeddings in Databricks Vector Search for fast retrieval
* Answers natural language questions with source citations
* Provides both web interface (Streamlit) and notebook interface

**Sample questions:**
* "What types of energy plans are available?"
* "How do solar buyback programs work?"
* "What are common fees in electricity plans?"
* "Are there plans with no monthly charge?"

---

## System Status

### ✅ Working Infrastructure

| Component | Status | Details |
|-----------|--------|---------|
| **Data Table** | ✅ Live | `main.pdf_rag.document_chunks` |
| **Vector Search Index** | ✅ Online | `main.pdf_rag.document_chunks_index` |
| **Vector Search Endpoint** | ✅ Active | `pdf_rag_endpoint` |
| **Embedding Model** | ✅ Active | `databricks-bge-large-en` (1024-dim) |
| **LLM Model** | ✅ Active | `databricks-meta-llama-3-3-70b-instruct` |
| **Streamlit App** | ⚠️ Deployed* | `pdf-rag-chat` |
| **Notebook Interface** | ✅ Working | `Energy_Plan_QA_Interactive.py` |

\* **Streamlit App Note:** The app is deployed and running successfully, but requires workspace admins to enable "On-Behalf-Of User Authorization" in workspace security settings for user authentication. [See deployment details](#databricks-app-streamlit).

### Current Deployment

* **App Name:** `pdf-rag-chat`
* **App URL:** `https://pdf-rag-chat-2479810620852778.aws.databricksapps.com`
* **Working Notebook:** [Energy_Plan_QA_Interactive](#notebook-1963340446073469)
* **Workspace:** `https://dbc-60fb4a1c-8bce.cloud.databricks.com`

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                  │
│  PDF Files         Text         Chunks      Embeddings  Delta    │
│  (Volume)  ──────► Extraction ──► Chunking ──► Model   ──► Table │
│                    pdfplumber    LangChain   databricks-  main.  │
│                                  Splitter    bge-large   pdf_rag │
│                                              -en         .doc... │
│                                                     │             │
│                                                     ▼             │
│                                            Vector Search Index    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                           │
│                                                                  │
│  Question ──► Embeddings ──► Vector Search ──► Top-K Chunks     │
│               (bge-large)    (Similarity)            │           │
│                                                      │           │
│                                          ┌───────────▼────────┐ │
│                                          │   LLM (Llama 3.3   │ │
│                                          │   70B Instruct)    │ │
│                                          └───────────┬────────┘ │
│                                                      │           │
│                                           Answer + Citations     │
│                                           (with page numbers)    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1: Interactive Notebook (Recommended - No Setup Needed)

The notebook interface is already working and ready to use:

1. Open [Energy_Plan_QA_Interactive](#notebook-1963340446073469)
2. Run all cells (packages install automatically)
3. Ask questions in the final cell:

```python
my_question = "How do solar buyback programs work?"
ask_question(my_question)
```

### Option 2: Streamlit Web App (Requires Admin Setup)

The Streamlit app is deployed but requires a one-time workspace setting:

1. **Admin must enable On-Behalf-Of authorization:**
   * Admin Console → Security → On-Behalf-Of
   * Enable "On-Behalf-Of User Authorization"

2. **Access the app:**
   * URL: https://pdf-rag-chat-2479810620852778.aws.databricksapps.com
   * Works immediately after authorization is enabled (no redeployment needed)

---

## Repository Structure

```
PDF_RAG/
├── app.py                          # Streamlit web application
├── app.yaml                        # Streamlit server configuration
├── databricks.yml                  # Databricks Apps deployment config
├── requirements.txt                # Python dependencies for app
├── DEPLOYMENT.md                   # Deployment troubleshooting guide
├── README.md                       # This file
├── config/
│   ├── config.yaml                 # Default RAG settings
│   └── databricks_config.yaml     # Databricks-specific settings
├── notebooks/
│   ├── Energy_Plan_QA_Interactive.py  # 🟢 Working notebook interface
│   ├── 01_setup_and_config.py     # Initial setup (already run)
│   ├── 02_pdf_ingestion.py        # PDF ingestion (already run)
│   └── 03_rag_query_interface.py  # Original query notebook
├── src/pdf_rag/                   # Core RAG package
│   ├── ingestion/
│   │   ├── pdf_loader.py          # PDF loading utilities
│   │   └── text_chunker.py        # Text chunking
│   ├── embeddings/
│   │   └── embedder.py            # Embeddings factory (Databricks/OpenAI/HF)
│   ├── vector_store/
│   │   ├── databricks_vs.py       # Databricks Vector Search wrapper
│   │   └── local_vs.py            # FAISS/Chroma for local dev
│   ├── retrieval/
│   │   └── retriever.py           # Hybrid retrieval + reranking
│   ├── generation/
│   │   └── llm_chain.py           # RAG chain + LLM factory
│   ├── pipeline/
│   │   ├── ingestion_pipeline.py  # End-to-end ingestion
│   │   └── rag_pipeline.py        # End-to-end query
│   └── utils/
│       ├── config.py              # Config management
│       └── logger.py              # Structured logging
└── tests/                         # pytest test suite
    ├── test_embedder.py
    ├── test_retriever.py
    └── ...
```

---

## Deployment Options

### Notebook Interface (✅ Currently Working)

**File:** `Energy_Plan_QA_Interactive.py`

**Features:**
* Uses your personal credentials (no app authentication needed)
* Same RAG functionality as the Streamlit app
* Cell-by-cell execution for debugging
* Instant feedback and results

**Usage:**
```python
# After running initialization cells:
my_question = "What types of energy plans are available?"
ask_question(my_question)
```

**Output includes:**
* Full answer from LLM
* Source citations with file names and page numbers
* Retrieval diagnostics

---

### Databricks App (Streamlit)

**Status:** ⚠️ Deployed but blocked by authentication

**App Details:**
* **Name:** `pdf-rag-chat`
* **URL:** https://pdf-rag-chat-2479810620852778.aws.databricksapps.com
* **Compute:** Medium (2 vCPUs, 6 GB)
* **Deployment Status:** `RUNNING` ✅
* **Authentication:** Requires On-Behalf-Of authorization ⚠️

**Required One-Time Setup (Admin Only):**

The app is fully deployed and running, but Databricks Apps require a workspace security setting for user authentication:

1. **Admin Access Required:**
   * Click your profile icon (top right)
   * Select "Admin Console" (only visible to workspace admins)

2. **Enable On-Behalf-Of Authorization:**
   * Navigate to: **Security** → **On-Behalf-Of**
   * Toggle **"On-Behalf-Of User Authorization"** to **ON**
   * Save changes

3. **Test the App:**
   * Wait 30 seconds
   * Access the app URL (no redeployment needed)
   * App will authenticate users automatically

**If you're not a workspace admin:**
* Contact your admin with this request:
  > "Please enable 'On-Behalf-Of User Authorization' in Admin Console → Security → On-Behalf-Of. This is required for the PDF RAG Databricks App to authenticate users."

**App Resources (All Configured ✅):**
* SQL Warehouse
* Embeddings Endpoint (`databricks-bge-large-en`)
* LLM Endpoint (`databricks-meta-llama-3-3-70b-instruct`)
* Vector Search Index (`main.pdf_rag.document_chunks_index`)
* UC Table (`main.pdf_rag.document_chunks`)
* UC Volume (`main.pdf_rag.temp_pdfs_3cfabbfa`)

**Service Principal Permissions (All Granted ✅):**
* CAN_QUERY on both model endpoints
* SELECT on document_chunks table
* READ on vector search index
* WRITE_VOLUME on temp_pdfs volume

---

## Configuration

### Current Settings

The system is currently configured for production use on Databricks:

| Setting | Value | Description |
|---------|-------|-------------|
| **Catalog** | `main` | Unity Catalog catalog |
| **Schema** | `pdf_rag` | Schema for tables and indexes |
| **Table** | `document_chunks` | Delta table with PDF chunks |
| **Vector Search Endpoint** | `pdf_rag_endpoint` | VS endpoint name |
| **Embedding Model** | `databricks-bge-large-en` | 1024-dimensional embeddings |
| **LLM Model** | `databricks-meta-llama-3-3-70b-instruct` | Chat-optimized LLM |
| **Temperature** | `0.0` | Deterministic responses |
| **Max Tokens** | `1024` | Maximum response length |
| **Retrieval K** | `10` | Number of chunks to retrieve |
| **Chunk Size** | `1000` | Characters per chunk |
| **Chunk Overlap** | `200` | Overlap between chunks |

### Modifying Configuration

**For notebook interface:**
Edit values directly in cell 4 of `Energy_Plan_QA_Interactive.py`:

```python
CATALOG = "main"
SCHEMA = "pdf_rag"
EMBEDDING_MODEL = "databricks-bge-large-en"
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"
```

**For Streamlit app:**
Edit values in `app.py` and redeploy:

```python
CATALOG = "main"
SCHEMA = "pdf_rag"
EMBEDDING_MODEL = "databricks-bge-large-en"
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"
```

**For general settings:**
Edit `config/databricks_config.yaml`:

```yaml
embedding:
  backend: databricks
  model_name: databricks-bge-large-en

llm:
  backend: databricks
  model_name: databricks-meta-llama-3-3-70b-instruct
  temperature: 0.0
  max_tokens: 1024
```

---

## How It Works

### VectorSearchRetriever

Custom LangChain retriever that queries Databricks Vector Search:

```python
class VectorSearchRetriever(BaseRetriever):
    """Retrieves relevant document chunks from Databricks Vector Search."""
    
    vsc_client: VectorSearchClient
    index: VectorSearchIndex
    embeddings: Embeddings
    k: int = 10  # Number of results to return
    text_column: str = "content"
    columns: List[str] = ["chunk_id", "content", "source", "file_name", "page_number"]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1. Embed the query
        query_vector = self.embeddings.embed_query(query)
        
        # 2. Search vector index
        response = self.index.similarity_search(
            query_vector=query_vector,
            columns=self.columns,
            num_results=self.k
        )
        
        # 3. Convert to LangChain Documents
        documents = []
        for row in response['result']['data_array']:
            col_map = {col: row[i] for i, col in enumerate(self.columns)}
            content = col_map.get(self.text_column, "")
            metadata = {k: v for k, v in col_map.items() if k != self.text_column}
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
```

### RAG Chain

The system uses LangChain to orchestrate retrieval and generation:

```python
# 1. Initialize components
embeddings = get_embeddings(backend="databricks", model_name="databricks-bge-large-en")
vsc = VectorSearchClient(workspace_url=host, personal_access_token=token)
vs_index = vsc.get_index(endpoint_name="pdf_rag_endpoint", index_name="main.pdf_rag.document_chunks_index")

# 2. Create retriever
retriever = VectorSearchRetriever(
    vsc_client=vsc,
    index=vs_index,
    embeddings=embeddings,
    k=10
)

# 3. Create LLM
llm = get_llm(backend="databricks", model_name="databricks-meta-llama-3-3-70b-instruct")

# 4. Create RAG chain
rag_chain = RAGChain(llm=llm, retriever=retriever)

# 5. Query
result = rag_chain.query_with_sources("How do solar buyback programs work?")
print(result["answer"])
for doc in result["sources"]:
    print(f"  → {doc.metadata['file_name']}, page {doc.metadata['page_number']}")
```

### Data Flow

1. **User asks a question** → "What are common fees in electricity plans?"
2. **Embeddings model** converts question to 1024-dim vector
3. **Vector Search** finds 10 most similar document chunks
4. **LLM** receives question + retrieved chunks as context
5. **LLM generates** answer grounded in retrieved content
6. **System returns** answer + source citations (file names, page numbers)

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/pdf_rag --cov-report=term-missing
```

### Adding New Documents

To add more PDFs to the system:

1. Upload PDFs to a Unity Catalog volume or DBFS location
2. Run the ingestion pipeline (see `notebooks/02_pdf_ingestion.py`)
3. The vector index updates automatically
4. New content is immediately available for queries

### Local Development

For local development without Databricks infrastructure:

```bash
# Install with local extras (FAISS + HuggingFace)
pip install -e ".[local,dev]"

# Configure for local mode
export PDF_RAG__EMBEDDING__BACKEND=huggingface
export PDF_RAG__LLM__BACKEND=huggingface
export PDF_RAG__VECTOR_STORE__TYPE=local
export PDF_RAG__VECTOR_STORE__LOCAL_BACKEND=faiss
```

### Project Structure Notes

* **`src/pdf_rag/`** - Core reusable package for ingestion, retrieval, and generation
* **`app.py`** - Streamlit interface (web deployment)
* **`Energy_Plan_QA_Interactive.py`** - Notebook interface (current working version)
* **`notebooks/`** - Setup, ingestion, and development notebooks
* **`config/`** - YAML configuration files
* **`tests/`** - pytest test suite

---

## Troubleshooting

### "Login error" on Streamlit App

**Symptom:** App shows "Login error - Sorry, there was an error while trying to authenticate to app"

**Cause:** On-Behalf-Of User Authorization is not enabled in workspace settings

**Solution:**
1. Contact workspace admin
2. Ask them to enable: Admin Console → Security → On-Behalf-Of → "On-Behalf-Of User Authorization"
3. Wait 30 seconds, then access app URL (no redeployment needed)

### Notebook Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'pdf_rag'`

**Solution:**
```python
# Add src to Python path
import sys
sys.path.insert(0, '/Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG/src')
```

### Vector Search Index Not Found

**Symptom:** `IndexNotFoundException` or `404 Not Found`

**Solution:**
* Verify index exists: `main.pdf_rag.document_chunks_index`
* Check endpoint is ONLINE: `pdf_rag_endpoint`
* Confirm you have READ permission on the index

### Slow Query Performance

**Cause:** LLM inference + vector search typically takes 3-5 seconds

**Optimization options:**
* Reduce `k` (number of retrieved chunks) from 10 to 5
* Reduce `MAX_TOKENS` from 1024 to 512
* Use a smaller/faster LLM model (trade-off: lower quality)

---

## Next Steps

* **✅ Complete:** Infrastructure, data ingestion, notebook interface
* **⚠️ Pending:** Streamlit app authentication (admin action required)
* **Future enhancements:**
  * Add reranking with CrossEncoder for better retrieval
  * Implement chat history for multi-turn conversations
  * Add evaluation metrics (RAGAS) for answer quality
  * Enable MLflow logging for query analytics
  * Add user feedback collection

---

## License

MIT – see [LICENSE](LICENSE).
