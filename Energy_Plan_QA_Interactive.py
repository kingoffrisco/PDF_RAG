# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# ///
# MAGIC %md
# MAGIC # ⚡ Energy Plan Q&A - Interactive Notebook
# MAGIC
# MAGIC This notebook provides the same RAG functionality as the Streamlit app,
# MAGIC but runs with your personal credentials (no app authentication needed).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Initialize RAG System

# COMMAND ----------

# MAGIC %pip install langchain>=0.1.0 langchain-core>=0.1.0 langchain-databricks>=0.1.0 databricks-vectorsearch>=0.22 sentence-transformers>=2.2.0 numpy>=1.24.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries and Configure
import sys
import os
from typing import List

# Add src to path
sys.path.insert(0, '/Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG/src')

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from pdf_rag.embeddings.embedder import get_embeddings
from pdf_rag.generation.llm_chain import RAGChain, get_llm
import numpy as np

# Configuration
CATALOG = "main"
SCHEMA = "pdf_rag"
TABLE_NAME = "document_chunks"
VS_ENDPOINT = "pdf_rag_endpoint"
EMBEDDING_BACKEND = "databricks"
EMBEDDING_MODEL = "databricks-bge-large-en"
LLM_BACKEND = "databricks"
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
INDEX_NAME = f"{FULL_TABLE}_index"

print("✅ Configuration loaded")

# COMMAND ----------

class VectorSearchRetriever(BaseRetriever):
    vsc_client: object
    index: object
    embeddings: object
    k: int = 10
    text_column: str = "content"
    columns: List[str] = []
    class Config:
        arbitrary_types_allowed = True
    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            query_vector = self.embeddings.embed_query(query)
            response = self.index.similarity_search(
                query_vector=query_vector, columns=self.columns, num_results=self.k
            )
            documents = []
            if response and 'result' in response and 'data_array' in response['result']:
                for row in response['result']['data_array']:
                    col_map = {col: row[i] for i, col in enumerate(self.columns)}
                    content = col_map.get(self.text_column, "")
                    metadata = {k: v for k, v in col_map.items() if k != self.text_column}
                    documents.append(Document(page_content=content, metadata=metadata))
            return documents
        except Exception as e:
            print(f"❌ Retriever error: {e}")
            return []

print("✅ VectorSearchRetriever defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Components

# COMMAND ----------

print("🔄 Initializing RAG components...")

# Get embeddings
embeddings_model = get_embeddings(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)
print("✅ Embeddings model loaded")

# Initialize Vector Search
w = WorkspaceClient()
vsc = VectorSearchClient(workspace_url=w.config.host, personal_access_token=w.config.token, disable_notice=True)
vs_index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
print("✅ Vector Search connected")

# Create retriever
retriever = VectorSearchRetriever(
    vsc_client=vsc, 
    index=vs_index, 
    embeddings=embeddings_model, 
    k=10,
    text_column="content", 
    columns=["chunk_id", "content", "source", "file_name", "page_number"]
)
print("✅ Retriever created")

# Get LLM
llm = get_llm(backend=LLM_BACKEND, model_name=LLM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
print("✅ LLM initialized")

# Create RAG chain
rag_chain = RAGChain(llm=llm, retriever=retriever, enable_mlflow=False)
print("✅ RAG chain ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ask Questions

# COMMAND ----------

def ask_question(question: str):
    """Ask a question and get an answer with sources."""
    print(f"\n❓ Question: {question}\n")
    print("🔄 Thinking...\n")
    
    result = rag_chain.query_with_sources(question)
    
    print("💬 Answer:")
    print("=" * 70)
    print(result["answer"])
    print("=" * 70)
    
    print("\n📚 Sources:")
    if result["sources"]:
        seen = set()
        for doc in result["sources"]:
            fn = doc.metadata.get('file_name', 'Unknown')
            pg = doc.metadata.get('page_number', '?')
            key = f"{fn}_{pg}"
            if key not in seen:
                print(f"   📄 {fn} (Page {pg})")
                seen.add(key)
    else:
        print("   No sources found")
    
    return result

# Example questions
print("💡 Try these example questions:\n")
print('   ask_question("What types of energy plans are available?")')
print('   ask_question("How do solar buyback programs work?")')
print('   ask_question("What are common fees in electricity plans?")')
print('   ask_question("Are there plans with no monthly charge?")')

# COMMAND ----------

# Try an example question
ask_question("What types of energy plans are available?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Interactive Q&A
# MAGIC
# MAGIC Run this cell and modify the question to ask your own questions:

# COMMAND ----------

# Your question here:
my_question = "How do solar buyback programs work?"

ask_question(my_question)

# COMMAND ----------

# Your question here:
my_question = "What is the top 10 things that can be learned?"

ask_question(my_question)
