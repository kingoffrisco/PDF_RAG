import streamlit as st
import sys
import os
import traceback

st.set_page_config(page_title="Diagnostic", page_icon="🔍")

st.title("🔍 Diagnostic Check")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.write("✅ Streamlit is running")
st.write(f"✅ Python version: {sys.version}")
st.write(f"✅ Working directory: {os.getcwd()}")

# Test 1: Check imports
st.header("1. Testing Package Imports")

tests = {
    "databricks.sdk": lambda: __import__("databricks.sdk"),
    "databricks.vector_search": lambda: __import__("databricks.vector_search.client"),
    "sentence_transformers": lambda: __import__("sentence_transformers"),
    "langchain_core": lambda: __import__("langchain_core"),
    "langchain_databricks": lambda: __import__("langchain_databricks"),
}

for name, import_fn in tests.items():
    try:
        import_fn()
        st.write(f"✅ {name}")
    except Exception as e:
        st.error(f"❌ {name}: {e}")

# Test 2: Check local package
st.header("2. Testing Local Package (pdf_rag)")
app_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(app_dir, "src")
st.write(f"App dir: {app_dir}")
st.write(f"Src path exists: {os.path.exists(src_path)}")

if os.path.exists(src_path):
    sys.path.insert(0, src_path)
    st.write(f"✅ Added to sys.path")
    
    try:
        from pdf_rag.embeddings.embedder import get_embeddings
        st.write("✅ pdf_rag.embeddings.embedder imported")
    except Exception as e:
        st.error(f"❌ pdf_rag.embeddings.embedder: {e}")
        st.code(traceback.format_exc())
    
    try:
        from pdf_rag.generation.llm_chain import get_llm
        st.write("✅ pdf_rag.generation.llm_chain imported")
    except Exception as e:
        st.error(f"❌ pdf_rag.generation.llm_chain: {e}")
        st.code(traceback.format_exc())

# Test 3: Check authentication
st.header("3. Testing Authentication")
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    st.write(f"✅ WorkspaceClient created")
    st.write(f"Host: {w.config.host}")
    st.write(f"Has token: {bool(w.config.token)}")
    st.write(f"Has client_id: {bool(w.config.client_id)}")
except Exception as e:
    st.error(f"❌ WorkspaceClient: {e}")

# Test 4: Check Vector Search
st.header("4. Testing Vector Search")
try:
    from databricks.vector_search.client import VectorSearchClient
    from databricks.sdk import WorkspaceClient
    
    w = WorkspaceClient()
    vsc_kwargs = {"workspace_url": w.config.host, "disable_notice": True}
    if w.config.client_id and w.config.client_secret:
        vsc_kwargs["service_principal_client_id"] = w.config.client_id
        vsc_kwargs["service_principal_client_secret"] = w.config.client_secret
    
    vsc = VectorSearchClient(**vsc_kwargs)
    st.write("✅ VectorSearchClient created")
    
    # Try to get index
    vs_index = vsc.get_index(
        endpoint_name="pdf_rag_endpoint",
        index_name="main.pdf_rag.document_chunks_index"
    )
    st.write("✅ Vector Search index accessed")
    
except Exception as e:
    st.error(f"❌ Vector Search: {e}")
    st.code(traceback.format_exc())

st.success("Diagnostic complete!")
