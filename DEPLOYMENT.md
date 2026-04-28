# Energy Plan Q&A Gradio App - Deployment Guide

## Overview
This Gradio app provides a chat interface for your PDF RAG system. Users can ask questions about electricity plans and get answers with source citations.

## Files
* `app.py` - Main Gradio application
* `requirements.txt` - Python dependencies
* `src/` - Your existing RAG package

## Deployment Options

### Option 1: Databricks Apps (Recommended for Internal Use)

1. **Create the app:**
   ```bash
   databricks apps create pdf-rag-chat \
       --source-code-path /Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG
   ```

2. **Access the app:**
   * The CLI will return a URL like: `https://<workspace>.databricks.com/apps/<app-id>`
   * Share this URL with your team

3. **Update the app:**
   ```bash
   databricks apps update pdf-rag-chat \
       --source-code-path /Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG
   ```

### Option 2: Run Locally from Notebook

For quick testing before deployment:

```python
%pip install gradio sentence-transformers
import sys
sys.path.insert(0, "/Workspace/Repos/kingoffrisco@yahoo.com/PDF_RAG/src")

# Run the app
%run /Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG/app.py
```

### Option 3: Run from Compute Cluster

If you want to test the app on a running cluster:

```bash
# SSH into your cluster or use a notebook cell
python /Workspace/Users/kingoffrisco@yahoo.com/PDF_RAG/app.py
```

The app will be available at: `http://<cluster-driver-ip>:8080`

## App Features

* ⚡ **Chat Interface** - Clean conversational UI with chat history
* 📚 **Source Citations** - Shows which PDF pages answered each question
* 🎯 **Reranking** - Uses your hybrid retrieval system (15→10 docs)
* 💡 **Example Questions** - Pre-loaded questions to get started
* 🎨 **Modern Theme** - Purple/blue gradient styling

## Configuration

To modify settings, edit these variables in `app.py`:

```python
# LLM Settings
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"
TEMPERATURE = 0.0
MAX_TOKENS = 1024

# Retrieval Settings
initial_k = 15  # Candidates to retrieve
final_k = 10    # After reranking
confidence_threshold = 0.3
```

## Troubleshooting

**Issue:** App won't start
* **Solution:** Ensure your vector search endpoint (`pdf_rag_endpoint`) is ONLINE
* Check that the index `main.pdf_rag.document_chunks_index` exists

**Issue:** Import errors
* **Solution:** Verify the REPO_PATH points to your repo location
* Ensure all dependencies are installed: `%pip install -r requirements.txt`

**Issue:** Slow responses
* **Solution:** This is normal - LLM inference + reranking takes 3-5 seconds
* Consider reducing `MAX_TOKENS` or `initial_k` for faster responses

## Next Steps

After testing the Gradio app:
1. Collect user feedback on answer quality
2. Monitor which questions work well vs. struggle
3. When ready for production API access, move to **Model Serving** (Option 2 from earlier)

## Support

For issues, check:
* Vector Search endpoint status in the UI
* Notebook `01_setup_and_config` for RAG chain testing
* Databricks Apps logs: `databricks apps logs pdf-rag-chat`
