# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# ///
# MAGIC %md
# MAGIC # PDF Retrieval Augmented Generation Notebook
# MAGIC

# COMMAND ----------

# MAGIC %md ## 1. Install dependencies

# COMMAND ----------

# MAGIC %pip install pdfplumber langchain langchain-core langchain-text-splitters \
# MAGIC              langchain-community langchain-databricks langchain-huggingface \
# MAGIC              databricks-vectorsearch sentence-transformers faiss-cpu \
# MAGIC              rank-bm25 mlflow pyyaml
# MAGIC
# MAGIC # Restart Python to pick up new packages
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 2. Configuration

# COMMAND ----------

import os

# ── Edit these variables ──────────────────────────────────────────────────────
CATALOG        = "main"            # Unity Catalog catalog
SCHEMA         = "pdf_rag"         # Schema (will be created if missing)
TABLE_NAME     = "document_chunks" # Delta table for chunk storage
VS_ENDPOINT    = "pdf_rag_endpoint"# Vector Search endpoint name
PDF_SOURCE_DIR = "dbfs:/mnt/raw_pdfs"  # Where your PDFs live

# LLM / Embedding backend ("databricks" | "openai" | "huggingface")
EMBEDDING_BACKEND = "databricks"
EMBEDDING_MODEL   = "databricks-bge-large-en"
LLM_BACKEND       = "databricks"
LLM_MODEL         = "databricks-dbrx-instruct"

# MLflow experiment
MLFLOW_EXPERIMENT = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/pdf_rag"
# ─────────────────────────────────────────────────────────────────────────────

print(f"Catalog:            {CATALOG}")
print(f"Schema:             {SCHEMA}")
print(f"Table:              {TABLE_NAME}")
print(f"VS Endpoint:        {VS_ENDPOINT}")
print(f"PDF Source:         {PDF_SOURCE_DIR}")
print(f"Embedding backend:  {EMBEDDING_BACKEND} / {EMBEDDING_MODEL}")
print(f"LLM backend:        {LLM_BACKEND} / {LLM_MODEL}")
print(f"MLflow experiment:  {MLFLOW_EXPERIMENT}")

# COMMAND ----------

# MAGIC %md ## 3. Create Unity Catalog schema

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema {CATALOG}.{SCHEMA} ready.")

# COMMAND ----------

# MAGIC %md ## 4. Verify Vector Search endpoint

# COMMAND ----------

# DBTITLE 1,Cell 9
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

endpoints = [ep["name"] for ep in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT in endpoints:
    print(f"✅  Vector Search endpoint '{VS_ENDPOINT}' already exists.")
else:
    print(f"⏳  Creating Vector Search endpoint '{VS_ENDPOINT}' …")
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print(f"✅  Endpoint '{VS_ENDPOINT}' created.  It may take ~5 minutes to become ONLINE.")

# COMMAND ----------

# MAGIC %md ## 5. Set MLflow experiment

# COMMAND ----------

import mlflow

mlflow.set_experiment(MLFLOW_EXPERIMENT)
print(f"MLflow experiment set to: {MLFLOW_EXPERIMENT}")

# COMMAND ----------

# MAGIC %md ## 6. Parameters
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 15
# ── Edit to match your environment ────────────────────────────────────────────
CATALOG        = "main"
SCHEMA         = "pdf_rag"
TABLE_NAME     = "document_chunks"
VS_ENDPOINT    = "pdf_rag_endpoint"
PDF_SOURCE_DIR = "s3://aws-kingoffisco-s3-bucket/RAG/"   # single file OR directory
RECURSIVE_SCAN = False                  # set True to scan sub-directories

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

EMBEDDING_BACKEND = "databricks"
EMBEDDING_MODEL   = "databricks-bge-large-en"

# Path to the cloned PDF_RAG repository inside Databricks Repos.
# e.g. "/Workspace/Repos/your_email@example.com/PDF_RAG"
REPO_PATH = "/Workspace/Repos/kingoffrisco@yahoo.com/PDF_RAG"
# ─────────────────────────────────────────────────────────────────────────────

FULL_TABLE    = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
INDEX_NAME    = f"{FULL_TABLE}_index"

# COMMAND ----------

# MAGIC %md ## 7. Load & chunk PDFs

# COMMAND ----------

# DBTITLE 1,Cell 17
import sys
sys.path.insert(0, f"{REPO_PATH}/src")

from pdf_rag.ingestion.pdf_loader import load_pdf
from pdf_rag.ingestion.text_chunker import chunk_documents

import os

# Note: DBFS mounts (/dbfs/mnt/) are not available on serverless compute
if PDF_SOURCE_DIR.startswith("dbfs:/mnt/"):
    print("⚠️  DBFS mounts are not available on serverless compute.")
    print("\nPlease update PDF_SOURCE_DIR in Cell 15 to use one of these options:")
    print("  1. S3 path: s3://aws-kingoffisco-s3-bucket/your-pdf-folder/")
    print("  2. Volume path: /Volumes/main/pdf_rag/pdfs/")
    raise ValueError("Invalid PDF_SOURCE_DIR: DBFS mounts not supported on serverless")

# For S3 paths, copy files to a temporary Unity Catalog Volume
if PDF_SOURCE_DIR.startswith("s3://"):
    try:
        files = dbutils.fs.ls(PDF_SOURCE_DIR)
        pdf_paths = [f.path for f in files if f.path.lower().endswith('.pdf')]
        
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in {PDF_SOURCE_DIR}")
        
        print(f"Found {len(pdf_paths)} PDF file(s) in {PDF_SOURCE_DIR}")
        
        # Create temporary Volume for staging
        import uuid
        temp_volume_name = f"temp_pdfs_{uuid.uuid4().hex[:8]}"
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{temp_volume_name}")
        temp_volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{temp_volume_name}"
        
        print(f"Using temporary volume: {temp_volume_path}")
        
        raw_docs = []
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            temp_pdf_path = f"{temp_volume_path}/{filename}"
            
            print(f"  Copying: {filename}")
            dbutils.fs.cp(pdf_path, temp_pdf_path)
            
            # Volume paths are accessible as local filesystem paths
            print(f"  Loading: {filename}")
            raw_docs.extend(load_pdf(temp_pdf_path, source_id=filename))
        
        # Clean up: drop the temporary volume
        spark.sql(f"DROP VOLUME IF EXISTS {CATALOG}.{SCHEMA}.{temp_volume_name}")
        print(f"Cleaned up temporary volume")
        
    except Exception as e:
        print(f"⚠️  Error loading PDFs from S3: {e}")
        # Clean up on error
        try:
            spark.sql(f"DROP VOLUME IF EXISTS {CATALOG}.{SCHEMA}.{temp_volume_name}")
        except:
            pass
        raise
else:
    # Local paths or Volume paths
    from pdf_rag.ingestion.pdf_loader import load_pdfs_from_directory
    import pathlib
    
    src = PDF_SOURCE_DIR.replace("dbfs:/", "/dbfs/")
    
    if pathlib.Path(src).is_dir():
        raw_docs = load_pdfs_from_directory(src, recursive=RECURSIVE_SCAN)
    else:
        raw_docs = load_pdf(src)

print(f"Loaded {len(raw_docs)} page(s).")

chunks = chunk_documents(raw_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
print(f"Created {len(chunks)} chunk(s).")

# COMMAND ----------

# MAGIC %md ## 8. Generate embeddings
# MAGIC

# COMMAND ----------

from pdf_rag.embeddings.embedder import get_embeddings

embeddings_model = get_embeddings(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)

texts = [c.page_content for c in chunks]
vectors = embeddings_model.embed_documents(texts)
print(f"Generated {len(vectors)} embedding vectors (dim={len(vectors[0])}).")

# COMMAND ----------

# MAGIC %md ## 9. Write to Delta table
# MAGIC

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F

rows = []
for chunk, vec in zip(chunks, vectors):
    row = {
        "chunk_id":    chunk.metadata.get("chunk_id", ""),
        "content":     chunk.page_content,
        "source":      chunk.metadata.get("source", ""),
        "file_name":   chunk.metadata.get("file_name", ""),
        "page_number": int(chunk.metadata.get("page_number", 0)),
        "chunk_index": int(chunk.metadata.get("chunk_index", 0)),
        "embedding":   vec,
    }
    rows.append(row)

pdf_data = pd.DataFrame(rows)
df = spark.createDataFrame(pdf_data)

(
    df.write
      .format("delta")
      .mode("append")
      .option("mergeSchema", "true")
      .saveAsTable(FULL_TABLE)
)

spark.sql(f"ALTER TABLE {FULL_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

count = spark.table(FULL_TABLE).count()
print(f"✅  {FULL_TABLE} now has {count} row(s).")

# COMMAND ----------

# MAGIC %md ## 10. Create / refresh Vector Search index
# MAGIC

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing_indexes = [
    idx["name"]
    for idx in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])
]

if INDEX_NAME in existing_indexes:
    print(f"Index '{INDEX_NAME}' already exists – triggering sync …")
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
    index.sync()
else:
    print(f"Creating index '{INDEX_NAME}' …")
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=INDEX_NAME,
        source_table_name=FULL_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_dimension=len(vectors[0]),
        embedding_vector_column="embedding",
    )

print("✅  Vector index ready.  Proceed to Notebook 03.")

# COMMAND ----------

# MAGIC %md ## 11. Parameters
# MAGIC

# COMMAND ----------

# ── Edit to match your environment ────────────────────────────────────────────
CATALOG     = "main"
SCHEMA      = "pdf_rag"
TABLE_NAME  = "document_chunks"
VS_ENDPOINT = "pdf_rag_endpoint"

EMBEDDING_BACKEND = "databricks"
EMBEDDING_MODEL   = "databricks-bge-large-en"

LLM_BACKEND  = "databricks"
LLM_MODEL    = "databricks-meta-llama-3-3-70b-instruct"  # Updated to valid endpoint
TEMPERATURE  = 0.0
MAX_TOKENS   = 1024

RETRIEVAL_K     = 8    # Increased from 5 to 8 for more context
DENSE_K         = 10   # candidates from vector search

# Path to the cloned PDF_RAG repository inside Databricks Repos.
REPO_PATH = "/Workspace/Repos/kingoffrisco@yahoo.com/PDF_RAG"  # Updated to your actual path
# ─────────────────────────────────────────────────────────────────────────────

FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
INDEX_NAME = f"{FULL_TABLE}_index"

# COMMAND ----------

# MAGIC %md ## 12. Build retriever from Databricks Vector Search
# MAGIC

# COMMAND ----------

import sys
sys.path.insert(0, f"{REPO_PATH}/src")

from langchain_databricks.vectorstores import DatabricksVectorSearch
from pdf_rag.embeddings.embedder import get_embeddings

embeddings_model = get_embeddings(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)

vs = DatabricksVectorSearch(
    endpoint=VS_ENDPOINT,
    index_name=INDEX_NAME,
    embedding=embeddings_model,
    text_column="content",
    columns=["chunk_id", "content", "source", "file_name", "page_number"],
)

retriever = vs.as_retriever(search_kwargs={"k": RETRIEVAL_K})  # Using RETRIEVAL_K = 8
print("✅  Databricks Vector Search retriever ready.")

# COMMAND ----------

# DBTITLE 1,Add Reranking Layer
# %pip install sentence-transformers
# dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Implement Reranking
import sys
sys.path.insert(0, f"{REPO_PATH}/src")

from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from langchain_databricks.vectorstores import DatabricksVectorSearch
from pdf_rag.embeddings.embedder import get_embeddings
import numpy as np

class RerankedRetriever(BaseRetriever):
    """Custom retriever with hybrid reranking - only reranks when confidence is high."""
    
    base_retriever: BaseRetriever
    reranker: CrossEncoder
    initial_k: int = 15
    final_k: int = 10
    confidence_threshold: float = 0.3  # Minimum score to use reranking
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Step 1: Retrieve initial candidates
        docs = self.base_retriever.invoke(query)
        
        if len(docs) == 0:
            return []
        
        # Step 2: Score all candidates with cross-encoder
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        # Step 3: Hybrid approach - check confidence
        max_score = float(np.max(scores))
        score_std = float(np.std(scores))
        
        # Use reranking only if:
        # - Max score is above threshold (confident match)
        # - OR there's significant variance (clear ranking signal)
        use_reranking = max_score > self.confidence_threshold or score_std > 0.5
        
        if use_reranking:
            # Sort by reranker score and take top final_k
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            result = [doc for doc, score in scored_docs[:self.final_k]]
            print(f"🎯 Reranking applied (max_score={max_score:.3f}, std={score_std:.3f})")
        else:
            # Fall back to original vector search order
            result = docs[:self.final_k]
            print(f"⚡ Vector search fallback (max_score={max_score:.3f}, std={score_std:.3f})")
        
        return result

# Recreate embeddings model and vector search (needed after Python restart)
embeddings_model = get_embeddings(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)

vs = DatabricksVectorSearch(
    endpoint=VS_ENDPOINT,
    index_name=INDEX_NAME,
    embedding=embeddings_model,
    text_column="content",
    columns=["chunk_id", "content", "source", "file_name", "page_number"],
)

# Initialize the reranker model
print("Loading cross-encoder reranker...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Create base retriever with k=15 for initial retrieval
base_retriever = vs.as_retriever(search_kwargs={"k": 15})

# Wrap with hybrid reranking layer (retrieve 15, rerank to top 10)
reranked_retriever = RerankedRetriever(
    base_retriever=base_retriever,
    reranker=reranker_model,
    initial_k=15,
    final_k=10,
    confidence_threshold=0.3
)

print("✅  Hybrid reranked retriever ready (retrieve 15, rerank to top 10)")
print(f"   Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
print(f"   Strategy: Rerank when confidence > 0.3, else use vector search order")

# COMMAND ----------

# MAGIC %md ## 13. Build the RAG chain

# COMMAND ----------

# DBTITLE 1,Build RAG Chain with Reranking
import mlflow
from pdf_rag.generation.llm_chain import RAGChain, get_llm

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/pdf_rag")

llm = get_llm(
    backend=LLM_BACKEND,
    model_name=LLM_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

# Use reranked retriever instead of base retriever
rag_chain = RAGChain(llm=llm, retriever=reranked_retriever, enable_mlflow=True)
print("✅  RAG chain ready with reranking.")
print(f"   Retrieval strategy: Fetch 10 candidates → Rerank to top 5")

# COMMAND ----------

# MAGIC %md ## 14. Interactive Q&A
# MAGIC
# MAGIC Run the cell below with your question.

# COMMAND ----------

# DBTITLE 1,Cell 32
# Change the question here:
QUESTION = "Summarise the key findings in the document."

# Verify the index is ready before querying
import time
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)
index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
status = index.describe()
state = status.get('status', {}).get('detailed_state', '')

if not state.startswith('ONLINE'):
    print(f"⚠️  Vector search index is not yet ONLINE (current state: {state})")
    print("The index was just created and may take a few minutes to provision.")
    print("Please wait and re-run this cell once the index is ready.")
    raise RuntimeError(f"Vector search index not ready (state: {state})")

result = rag_chain.query_with_sources(QUESTION)

print("=" * 70)
print("ANSWER:")
print(result["answer"])
print()
print("SOURCES:")
for doc in result["sources"]:
    print(f"  • {doc.metadata.get('file_name')} – Page {doc.metadata.get('page_number')}")

# COMMAND ----------

# MAGIC %md ## 15. Batch questions example

# COMMAND ----------

questions = [
    "What are the main risks identified?",
    "What recommendations are made?",
    "Who are the key stakeholders?",
]

for q in questions:
    answer = rag_chain.query(q)
    print(f"Q: {q}")
    print(f"A: {answer[:300]}…\n")

# COMMAND ----------

# MAGIC %md
# MAGIC # RAG Evaluation
# MAGIC
# MAGIC Evaluates the quality of the RAG pipeline using **RAGAS** metrics:
# MAGIC
# MAGIC | Metric | What it measures |
# MAGIC |--------|-----------------|
# MAGIC | `faithfulness` | Is the answer grounded in the retrieved context? |
# MAGIC | `answer_relevancy` | Is the answer relevant to the question? |
# MAGIC | `context_precision` | Are the retrieved chunks precise for the question? |
# MAGIC | `context_recall` | Are the retrieved chunks complete (all relevant info present)? |
# MAGIC
# MAGIC Results are logged to MLflow for tracking over time.
# MAGIC

# COMMAND ----------

# MAGIC %pip install ragas datasets

# COMMAND ----------

# MAGIC %md ## 16. Parameters

# COMMAND ----------

# ── Edit to match your environment ────────────────────────────────────────────
CATALOG     = "main"
SCHEMA      = "pdf_rag"
TABLE_NAME  = "document_chunks"
VS_ENDPOINT = "pdf_rag_endpoint"

EMBEDDING_BACKEND = "databricks"
EMBEDDING_MODEL   = "databricks-bge-large-en"
LLM_BACKEND       = "databricks"
LLM_MODEL         = "databricks-meta-llama-3-3-70b-instruct"  # Updated to valid endpoint

# Path to the cloned PDF_RAG repository inside Databricks Repos.
REPO_PATH = "/Workspace/Repos/kingoffrisco@yahoo.com/PDF_RAG"  # Updated to your actual path
# ─────────────────────────────────────────────────────────────────────────────

FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
INDEX_NAME = f"{FULL_TABLE}_index"

# COMMAND ----------

# MAGIC %md ## 17. Define evaluation questions
# MAGIC
# MAGIC For a rigorous evaluation, provide ground-truth answers.
# MAGIC Here we use a minimal set; in production supply 50–100 Q&A pairs.

# COMMAND ----------

# DBTITLE 1,Define evaluation questions
eval_dataset = [
    {
        "question": "What types of energy plans are described in the documents?",
        "ground_truth": "The documents describe Electricity Facts Labels (EFLs) for various residential energy plans, including solar buyback plans, fixed-rate plans, and variable-rate plans from multiple providers.",
    },
    {
        "question": "What are the common fees associated with these energy plans?",
        "ground_truth": "Common fees include monthly base charges, early termination or cancellation fees, and TDSP (Transmission and Distribution Service Provider) delivery charges. Some plans have no termination fees.",
    },
    {
        "question": "How do solar buyback or solar credit programs work?",
        "ground_truth": "Solar buyback programs credit customers for excess energy their solar panels send to the grid. Credits are calculated by multiplying the buyback rate by the excess energy quantity and can offset energy charges, though typically not base charges, TDSP charges, or taxes.",
    },
    {
        "question": "What contract term lengths are available?",
        "ground_truth": "Contract terms vary by plan and provider, ranging from month-to-month plans to fixed-term contracts of 12, 24, or 36 months.",
    },
    {
        "question": "What factors affect the average price per kWh?",
        "ground_truth": "The average price per kWh depends on usage level (500, 1000, or 2000 kWh), base charges, energy charges, TDSP delivery charges, and any applicable credits like solar buyback credits.",
    },
    {
        "question": "What are the renewable energy content percentages?",
        "ground_truth": "Renewable energy content varies by plan, ranging from 0% to 100% renewable content. Some plans specifically highlight their renewable energy composition.",
    },
    {
        "question": "What happens if a customer cancels their contract early?",
        "ground_truth": "Early cancellation typically results in a termination or cancellation fee, which varies by provider and plan. Some plans offer no early termination fees, while others may charge fees ranging from $150 to $395 or more.",
    },
    {
        "question": "How are energy charges typically structured?",
        "ground_truth": "Energy charges are typically a per-kWh rate that may be fixed or variable. The total bill includes energy charges plus base charges, TDSP delivery charges, and applicable taxes and fees.",
    },
    {
        "question": "What disclosure information is required in these documents?",
        "ground_truth": "Electricity Facts Labels must include average pricing at different usage levels, contract terms, renewable energy content, fees, provider contact information, and details about how charges are calculated.",
    },
    {
        "question": "Are there plans with no base charge or monthly fee?",
        "ground_truth": "Some plans have no monthly base charge, while others have monthly base charges that typically range from around $5 to $15 per month, depending on the provider and plan type.",
    },
]

# COMMAND ----------

# MAGIC %md ## 18. Run RAG chain over eval set

# COMMAND ----------

# DBTITLE 1,Run RAG Chain with Reranking on Eval Set
import sys
sys.path.insert(0, f"{REPO_PATH}/src")

from langchain_databricks.vectorstores import DatabricksVectorSearch
from pdf_rag.embeddings.embedder import get_embeddings
from pdf_rag.generation.llm_chain import RAGChain, get_llm
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import numpy as np

# Define the RerankedRetriever class with hybrid approach
class RerankedRetriever(BaseRetriever):
    """Custom retriever with hybrid reranking - only reranks when confidence is high."""
    
    base_retriever: BaseRetriever
    reranker: CrossEncoder
    initial_k: int = 15
    final_k: int = 10
    confidence_threshold: float = 0.3
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if len(docs) == 0:
            return []
        
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        max_score = float(np.max(scores))
        score_std = float(np.std(scores))
        
        use_reranking = max_score > self.confidence_threshold or score_std > 0.5
        
        if use_reranking:
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:self.final_k]]
        else:
            return docs[:self.final_k]

# Setup embeddings and vector search
embeddings_model = get_embeddings(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)

vs = DatabricksVectorSearch(
    endpoint=VS_ENDPOINT,
    index_name=INDEX_NAME,
    embedding=embeddings_model,
    text_column="content",
    columns=["chunk_id", "content", "source", "file_name", "page_number"],
)

# Create base retriever with k=15
base_retriever = vs.as_retriever(search_kwargs={"k": 15})

# Initialize reranker and wrap retriever
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
eval_reranked_retriever = RerankedRetriever(
    base_retriever=base_retriever,
    reranker=reranker_model,
    initial_k=15,
    final_k=10,
    confidence_threshold=0.3
)

# Build RAG chain with hybrid reranking
llm = get_llm(backend=LLM_BACKEND, model_name=LLM_MODEL, temperature=0.0, max_tokens=1024)
rag_chain = RAGChain(llm=llm, retriever=eval_reranked_retriever, enable_mlflow=False)

print("Running evaluation with hybrid reranking (retrieve 15, rerank to top 10)...")
results = []
for item in eval_dataset:
    resp = rag_chain.query_with_sources(item["question"])
    results.append({
        "question":   item["question"],
        "answer":     resp["answer"],
        "contexts":   [d.page_content for d in resp["sources"]],
        "ground_truth": item["ground_truth"],
    })

print(f"✅  Evaluated {len(results)} questions with hybrid reranking")

# COMMAND ----------

# MAGIC %md ## 19. Compute RAGAS metrics

# COMMAND ----------

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_databricks import ChatDatabricks, DatabricksEmbeddings

# Configure RAGAS to use Databricks LLM and embeddings instead of OpenAI
llm_ragas = ChatDatabricks(
    endpoint=LLM_MODEL,
    temperature=0.0,
    max_tokens=1024,
)

embeddings_ragas = DatabricksEmbeddings(
    endpoint=EMBEDDING_MODEL
)

ds = Dataset.from_list(results)
scores = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm_ragas,
    embeddings=embeddings_ragas,
)
print(scores)

# COMMAND ----------

# MAGIC %md ## 20. Log results to MLflow

# COMMAND ----------

# DBTITLE 1,Deploy Gradio App
# MAGIC %md
# MAGIC ## 21. Deploy Gradio App
# MAGIC
# MAGIC Your chat interface is ready to deploy! The app provides a user-friendly Q&A interface.

# COMMAND ----------

# DBTITLE 1,Deployment Instructions
# Deployment Instructions for Gradio App

print("📦 Gradio App Files:")
print("  ✅ app.py - Main application")
print("  ✅ requirements.txt - Dependencies")
print("  ✅ DEPLOYMENT.md - Full guide")
print("\n" + "="*70)
print("🚀 DEPLOYMENT OPTIONS")
print("="*70)

print("\n1️⃣  OPTION 1: Deploy as Databricks App (Recommended)")
print("   Run this command from your LOCAL terminal (not in notebook):")
print("")
print("   databricks apps create pdf-rag-chat \\")
print("       --source-code-path /Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG")
print("")
print("   Then access your app at the URL provided by the CLI.")
print("")

print("\n2️⃣  OPTION 2: Test Locally from This Notebook")
print("   Run this in a new cell:")
print("")
print("   %pip install gradio")
print("   %run /Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG/app.py")
print("")
print("   The app will start on your cluster and you can test it.")

print("\n" + "="*70)
print("📚 WHAT THE APP DOES")
print("="*70)
print("  • Provides a chat interface for asking questions")
print("  • Uses your 15→10 hybrid reranking system")
print("  • Shows source citations (PDF names and page numbers)")
print("  • Includes example questions to get started")
print("  • Modern purple/blue themed UI")

print("\n💡 TIP: For production API access, we'll set up Model Serving later.")

# COMMAND ----------

import mlflow

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/pdf_rag")

with mlflow.start_run(run_name="ragas_evaluation"):
    # Convert EvaluationResult to pandas DataFrame to access metrics
    scores_df = scores.to_pandas()
    
    # Log each metric from the DataFrame columns
    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        if metric in scores_df.columns:
            # Take the mean across all samples for this metric
            value = scores_df[metric].mean()
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric, value)
    
    mlflow.log_dict(results, "eval_results.json")
    print("✅  Evaluation results logged to MLflow.")

scores.to_pandas()

# COMMAND ----------

# DBTITLE 1,Quick Q&A Test (No Gradio)
# Quick Interactive Test - Ask Your Question Here

import sys
sys.path.insert(0, f"{REPO_PATH}/src")

from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_databricks.vectorstores import DatabricksVectorSearch
from pdf_rag.embeddings.embedder import get_embeddings
from pdf_rag.generation.llm_chain import RAGChain, get_llm
from typing import List
import numpy as np

class RerankedRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    reranker: CrossEncoder
    initial_k: int = 15
    final_k: int = 10
    confidence_threshold: float = 0.3
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if len(docs) == 0:
            return []
        
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        max_score = float(np.max(scores))
        score_std = float(np.std(scores))
        use_reranking = max_score > self.confidence_threshold or score_std > 0.5
        
        if use_reranking:
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:self.final_k]]
        else:
            return docs[:self.final_k]

print("Initializing test Q&A system...")

embeddings_model = get_embeddings(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)
vs = DatabricksVectorSearch(
    endpoint=VS_ENDPOINT,
    index_name=INDEX_NAME,
    embedding=embeddings_model,
    text_column="content",
    columns=["chunk_id", "content", "source", "file_name", "page_number"],
)

base_retriever = vs.as_retriever(search_kwargs={"k": 15})
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
test_retriever = RerankedRetriever(
    base_retriever=base_retriever,
    reranker=reranker_model,
    initial_k=15,
    final_k=10,
    confidence_threshold=0.3
)

llm = get_llm(backend=LLM_BACKEND, model_name=LLM_MODEL, temperature=0.0, max_tokens=1024)
test_rag_chain = RAGChain(llm=llm, retriever=test_retriever, enable_mlflow=False)

print("\u2705 Ready! Change the question below and re-run this cell.\n")

# CHANGE YOUR QUESTION HERE:
question = "What types of energy plans are available?"

print("="*70)
print(f"Q: {question}")
print("="*70)

result = test_rag_chain.query_with_sources(question)

print(f"\nA: {result['answer']}")
print("\n" + "-"*70)
print("\nSources:")
for doc in result['sources']:
    print(f"  • {doc.metadata['file_name']} (Page {doc.metadata['page_number']})")

# COMMAND ----------

# DBTITLE 1,Install Gradio
# MAGIC %pip install gradio --quiet

# COMMAND ----------

# DBTITLE 1,Launch Gradio App
# Run the Gradio app in this notebook
print("🚀 Starting Gradio chat interface...")
print("⏱️  This will take 30-60 seconds to initialize...")
print("")

%run /Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG/app.py

# COMMAND ----------

# DBTITLE 1,Prepare Bundle Deployment Files
# Download app files for local bundle deployment
import os
import shutil

print("📦 Preparing files for local bundle deployment...\n")

# Files to include in bundle
files_to_show = [
    "/Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG/app.py",
    "/Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG/requirements.txt",
]

for file_path in files_to_show:
    if os.path.exists(file_path):
        print(f"✅ Found: {file_path}")
        # Show first few lines
        with open(file_path, 'r') as f:
            lines = f.readlines()[:5]
            print(f"   Preview: {lines[0][:60]}...")
    else:
        print(f"❌ Missing: {file_path}")

print("\n" + "="*70)
print("📋 TO DEPLOY VIA BUNDLE:")
print("="*70)
print("\n1. Download these files to your local D:\\project\\RAG folder:")
print("   - app.py")
print("   - requirements.txt")
print("   - src/ folder (your entire RAG package)")
print("\n2. You can download from Databricks Repos UI:")
print("   - Navigate to /Repos/kingoffrisco@yahoo.com/PDF_RAG")
print("   - Click the '...' menu → Export → Download as ZIP")
print("   - Extract to D:\\project\\RAG")
print("\n3. Or use Databricks CLI sync:")
print("   databricks sync /Workspace/Repos/kingoffrisco@yahoo.com/PDF_RAG D:\\project\\RAG")
