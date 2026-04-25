# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03 – RAG Query Interface
# MAGIC
# MAGIC This notebook demonstrates interactive querying of the indexed PDF corpus
# MAGIC using the Databricks Foundation Model API for both embeddings and the LLM.
# MAGIC
# MAGIC **Prerequisites**: Notebooks 01 and 02 must have been run successfully.

# COMMAND ----------
# MAGIC %md ## 0. Parameters

# COMMAND ----------

# ── Edit to match your environment ────────────────────────────────────────────
CATALOG     = "main"
SCHEMA      = "pdf_rag"
TABLE_NAME  = "document_chunks"
VS_ENDPOINT = "pdf_rag_endpoint"

EMBEDDING_BACKEND = "databricks"
EMBEDDING_MODEL   = "databricks-bge-large-en"

LLM_BACKEND  = "databricks"
LLM_MODEL    = "databricks-dbrx-instruct"
TEMPERATURE  = 0.0
MAX_TOKENS   = 1024

RETRIEVAL_K     = 5    # final docs fed to the LLM
DENSE_K         = 10   # candidates from vector search
# ─────────────────────────────────────────────────────────────────────────────

FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
INDEX_NAME = f"{FULL_TABLE}_index"

# COMMAND ----------
# MAGIC %md ## 1. Build retriever from Databricks Vector Search

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/<your_repo>/PDF_RAG/src")

from databricks.vector_search.client import VectorSearchClient
from langchain_databricks.vectorstores import DatabricksVectorSearch
from pdf_rag.embeddings.embedder import get_embeddings

embeddings_model = get_embeddings(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)

vsc = VectorSearchClient(disable_notice=True)
index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)

vs = DatabricksVectorSearch(
    index=index,
    embedding=embeddings_model,
    text_column="content",
    columns=["chunk_id", "content", "source", "file_name", "page_number"],
)

retriever = vs.as_retriever(search_kwargs={"k": DENSE_K})
print("✅  Databricks Vector Search retriever ready.")

# COMMAND ----------
# MAGIC %md ## 2. Build the RAG chain

# COMMAND ----------

import mlflow
from pdf_rag.generation.llm_chain import RAGChain, get_llm

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/pdf_rag")

llm = get_llm(
    backend=LLM_BACKEND,
    model_name=LLM_MODEL,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

rag_chain = RAGChain(llm=llm, retriever=retriever, enable_mlflow=True)
print("✅  RAG chain ready.")

# COMMAND ----------
# MAGIC %md ## 3. Interactive Q&A
# MAGIC
# MAGIC Run the cell below with your question.

# COMMAND ----------

# Change the question here:
QUESTION = "Summarise the key findings in the document."

result = rag_chain.query_with_sources(QUESTION)

print("=" * 70)
print("ANSWER:")
print(result["answer"])
print()
print("SOURCES:")
for doc in result["sources"]:
    print(f"  • {doc.metadata.get('file_name')} – Page {doc.metadata.get('page_number')}")

# COMMAND ----------
# MAGIC %md ## 4. Batch questions example

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
