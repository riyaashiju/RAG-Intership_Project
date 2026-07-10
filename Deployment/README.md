# Networking RAG Chatbot

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that answers networking-related questions using Hybrid Search and Large Language Models.

## Live Demo

https://rag-intershipproject-cn9xkspxodzwahfwuukgdv.streamlit.app/

## Features

- Better Chunking
- Hybrid Search (ChromaDB + BM25)
- Gemini Embeddings
- LangGraph Workflow
- Groq Llama 3.3 70B
- Streamlit Interface

## Technologies

- Python
- Streamlit
- ChromaDB
- LangGraph
- Google Gemini API
- Groq API
- BM25

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Files

- app.py – Streamlit UI
- rag_pipeline.py – RAG pipeline
- requirements.txt – Python dependencies
- networking_chromadb_phase4/ – Vector database
- improved_chunks.pkl – Chunked documents
- improved_embeddings.pkl – Embeddings

## Author

**Riya Shiju**
