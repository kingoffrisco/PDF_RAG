import streamlit as st
import sys
import os
from typing import List
import traceback

print("="*70)
print("🚀 Starting PDF RAG Chat App (Streamlit) - SIMPLIFIED VERSION")
print("="*70)

# Setup Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(app_dir, "src")
if os.path.exists(src_path):
    sys.path.insert(0, src_path)
    print(f"✅ Added {src_path} to Python path")

try:
    print("\n📦 Importing dependencies...")
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from databricks.vector_search.client import VectorSearchClient
    from databricks.sdk import WorkspaceClient
    from pdf_rag.embeddings.embedder import get_embeddings
    from pdf_rag.generation.llm_chain import RAGChain, get_llm
    import numpy as np
    print("  ✅ All dependencies loaded")
except Exception as e:
    print(f"❌ Import Error: {e}")
    st.error(f"Failed to import dependencies: {e}")
    st.stop()

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

@st.cache_resource
def initialize_rag():
    """Initialize RAG system once and cache it - WITHOUT RERANKER."""
    try:
        print("\n🔄 Initializing RAG (simplified - no reranking)...")
        embeddings_model = get_embeddings(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)
        w = WorkspaceClient()
        vsc_kwargs = {"workspace_url": w.config.host, "disable_notice": True}
        if w.config.token:
            vsc_kwargs["personal_access_token"] = w.config.token
        elif w.config.client_id and w.config.client_secret:
            vsc_kwargs["service_principal_client_id"] = w.config.client_id
            vsc_kwargs["service_principal_client_secret"] = w.config.client_secret
        vsc = VectorSearchClient(**vsc_kwargs)
        vs_index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
        
        # Use base retriever directly (no reranking)
        retriever = VectorSearchRetriever(
            vsc_client=vsc, index=vs_index, embeddings=embeddings_model, k=10,
            text_column="content", columns=["chunk_id", "content", "source", "file_name", "page_number"]
        )
        
        llm = get_llm(backend=LLM_BACKEND, model_name=LLM_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        rag_chain = RAGChain(llm=llm, retriever=retriever, enable_mlflow=False)
        print("✅ RAG initialized (simplified version)")
        return rag_chain, None
    except Exception as e:
        print(f"❌ Init error: {e}")
        print(traceback.format_exc())
        return None, str(e)

def format_sources(sources: List[Document]) -> str:
    if not sources:
        return "\n\n*No sources found*"
    text = "\n\n---\n\n**📚 Sources:**\n\n"
    seen = set()
    for doc in sources:
        fn = doc.metadata.get('file_name', 'Unknown')
        pg = doc.metadata.get('page_number', '?')
        key = f"{fn}_{pg}"
        if key not in seen:
            text += f"* 📄 {fn} (Page {pg})\n"
            seen.add(key)
    return text

# Page config
st.set_page_config(
    page_title="⚡ Energy Plan Q&A",
    page_icon="⚡",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("⚡ Energy Plan Q&A Assistant")
st.info("🔧 **Simplified Mode** - Reranking temporarily disabled for testing")
st.markdown("""
Ask me anything about electricity plans! I can help you understand:
* Plan types and pricing
* Fees and contract terms
* Solar buyback programs
* Renewable energy content
* And more!
""")

# Initialize RAG system
rag_chain, init_error = initialize_rag()

if init_error:
    st.error(f"❌ System initialization failed: {init_error}")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Example questions
with st.expander("💡 Example Questions", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What types of energy plans are available?"):
            st.session_state.example_question = "What types of energy plans are available?"
        if st.button("How do solar buyback programs work?"):
            st.session_state.example_question = "How do solar buyback programs work?"
    with col2:
        if st.button("What are common fees?"):
            st.session_state.example_question = "What are common fees?"
        if st.button("Plans with no monthly charge?"):
            st.session_state.example_question = "Plans with no monthly charge?"

# Handle example question
if "example_question" in st.session_state:
    prompt = st.session_state.example_question
    del st.session_state.example_question
else:
    prompt = st.chat_input("Ask a question about energy plans...")

# Process user input
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag_chain.query_with_sources(prompt)
                response = result["answer"] + format_sources(result["sources"])
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
