# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 01 – Setup & Configuration
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Installs the `pdf-rag` package and its dependencies into the cluster.
# MAGIC 2. Validates the Databricks environment (Unity Catalog, Vector Search endpoint).
# MAGIC 3. Creates the target schema and tables in Unity Catalog.
# MAGIC 4. Writes a verified configuration object for use in later notebooks.

# COMMAND ----------
# MAGIC %md ## 1. Install dependencies

# COMMAND ----------

# MAGIC %pip install pdfplumber langchain langchain-core langchain-text-splitters \
# MAGIC              langchain-community langchain-databricks langchain-huggingface \
# MAGIC              databricks-vectorsearch sentence-transformers faiss-cpu \
# MAGIC              rank-bm25 mlflow pyyaml

# Restart Python to pick up new packages
dbutils.library.restartPython()

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
# MAGIC %md
# MAGIC ### ✅ Setup complete
# MAGIC Proceed to **Notebook 02 – PDF Ingestion**.
