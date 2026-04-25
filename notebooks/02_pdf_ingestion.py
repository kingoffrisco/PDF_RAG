# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02 – PDF Ingestion
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads PDF files from DBFS / Unity Catalog Volumes.
# MAGIC 2. Extracts and chunks the text.
# MAGIC 3. Generates embeddings using the Databricks Foundation Model API.
# MAGIC 4. Writes chunks (with embeddings) to a Delta table in Unity Catalog.
# MAGIC 5. Creates (or refreshes) a Databricks Vector Search index on that table.
# MAGIC
# MAGIC **Prerequisites**: Run Notebook 01 first and ensure the variables below
# MAGIC match your environment.

# COMMAND ----------
# MAGIC %md ## 0. Parameters (edit these)

# COMMAND ----------

# ── Edit to match your environment ────────────────────────────────────────────
CATALOG        = "main"
SCHEMA         = "pdf_rag"
TABLE_NAME     = "document_chunks"
VS_ENDPOINT    = "pdf_rag_endpoint"
PDF_SOURCE_DIR = "dbfs:/mnt/raw_pdfs"   # single file OR directory
RECURSIVE_SCAN = False                  # set True to scan sub-directories

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

EMBEDDING_BACKEND = "databricks"
EMBEDDING_MODEL   = "databricks-bge-large-en"
# ─────────────────────────────────────────────────────────────────────────────

FULL_TABLE    = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
INDEX_NAME    = f"{FULL_TABLE}_index"

# COMMAND ----------
# MAGIC %md ## 1. Load & chunk PDFs

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/<your_repo>/PDF_RAG/src")

from pdf_rag.ingestion.pdf_loader import load_pdf, load_pdfs_from_directory
from pdf_rag.ingestion.text_chunker import chunk_documents

import os, pathlib

src = PDF_SOURCE_DIR.replace("dbfs:/", "/dbfs/")
if pathlib.Path(src).is_dir():
    raw_docs = load_pdfs_from_directory(PDF_SOURCE_DIR, recursive=RECURSIVE_SCAN)
else:
    raw_docs = load_pdf(PDF_SOURCE_DIR)

print(f"Loaded {len(raw_docs)} page(s).")

chunks = chunk_documents(raw_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
print(f"Created {len(chunks)} chunk(s).")

# COMMAND ----------
# MAGIC %md ## 2. Generate embeddings

# COMMAND ----------

from pdf_rag.embeddings.embedder import get_embeddings

embeddings_model = get_embeddings(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)

texts = [c.page_content for c in chunks]
vectors = embeddings_model.embed_documents(texts)
print(f"Generated {len(vectors)} embedding vectors (dim={len(vectors[0])}).")

# COMMAND ----------
# MAGIC %md ## 3. Write to Delta table

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
# MAGIC %md ## 4. Create / refresh Vector Search index

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing_indexes = [
    idx["name"]
    for idx in vsc.list_indexes(endpoint_name=VS_ENDPOINT).get("vector_indexes", [])
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
