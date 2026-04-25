# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04 – RAG Evaluation
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
# MAGIC **Prerequisites**: Notebooks 01–03 must have run.

# COMMAND ----------
# MAGIC %pip install ragas datasets

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
LLM_BACKEND       = "databricks"
LLM_MODEL         = "databricks-dbrx-instruct"

# Path to the cloned PDF_RAG repository inside Databricks Repos.
# e.g. "/Workspace/Repos/your_email@example.com/PDF_RAG"
REPO_PATH = "/Workspace/Repos/your_email@example.com/PDF_RAG"
# ─────────────────────────────────────────────────────────────────────────────

FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
INDEX_NAME = f"{FULL_TABLE}_index"

# COMMAND ----------
# MAGIC %md ## 1. Define evaluation questions
# MAGIC
# MAGIC For a rigorous evaluation, provide ground-truth answers.
# MAGIC Here we use a minimal set; in production supply 50–100 Q&A pairs.

# COMMAND ----------

eval_dataset = [
    {
        "question": "What is the main topic of the document?",
        "ground_truth": "Provide the expected answer here.",
    },
    {
        "question": "What are the key recommendations?",
        "ground_truth": "Provide the expected answer here.",
    },
]

# COMMAND ----------
# MAGIC %md ## 2. Run RAG chain over eval set

# COMMAND ----------

import sys
sys.path.insert(0, f"{REPO_PATH}/src")

from databricks.vector_search.client import VectorSearchClient
from langchain_databricks.vectorstores import DatabricksVectorSearch
from pdf_rag.embeddings.embedder import get_embeddings
from pdf_rag.generation.llm_chain import RAGChain, get_llm

embeddings_model = get_embeddings(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)
vsc = VectorSearchClient(disable_notice=True)
index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)

from langchain_databricks.vectorstores import DatabricksVectorSearch
vs = DatabricksVectorSearch(
    index=index,
    embedding=embeddings_model,
    text_column="content",
    columns=["chunk_id", "content", "source", "file_name", "page_number"],
)
retriever = vs.as_retriever(search_kwargs={"k": 5})

llm = get_llm(backend=LLM_BACKEND, model_name=LLM_MODEL, temperature=0.0, max_tokens=1024)
rag_chain = RAGChain(llm=llm, retriever=retriever, enable_mlflow=False)

results = []
for item in eval_dataset:
    resp = rag_chain.query_with_sources(item["question"])
    results.append({
        "question":   item["question"],
        "answer":     resp["answer"],
        "contexts":   [d.page_content for d in resp["sources"]],
        "ground_truth": item["ground_truth"],
    })

# COMMAND ----------
# MAGIC %md ## 3. Compute RAGAS metrics

# COMMAND ----------

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

ds = Dataset.from_list(results)
scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(scores)

# COMMAND ----------
# MAGIC %md ## 4. Log results to MLflow

# COMMAND ----------

import mlflow

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/pdf_rag")

with mlflow.start_run(run_name="ragas_evaluation"):
    for metric, value in scores.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(metric, value)
    mlflow.log_dict(results, "eval_results.json")
    print("✅  Evaluation results logged to MLflow.")

scores.to_pandas()
