"""
Networking RAG - Streamlit Application
"""

import streamlit as st
from rag_pipeline import ask_network_question

# Page Configuration

st.set_page_config(
    page_title="Networking RAG Chatbot",
    page_icon="🌐",
    layout="wide"
)

# Header

st.title("🌐 Networking RAG Chatbot")

st.markdown("""
This chatbot answers networking-related questions using:

- ✅ Better Chunking
- ✅ Hybrid Search (ChromaDB + BM25)
- ✅ LangGraph Workflow
- ✅ Groq Llama 3.3 70B
""")

st.divider()

# Question Input

question = st.text_input(
    "Enter your networking question:",
    placeholder="Example: What is the purpose of DNS?"
)

# Button

if st.button("🚀 Get Answer", use_container_width=True):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching the knowledge base..."):

            result = ask_network_question(question)

        st.success("Answer generated successfully!")

        st.subheader("📖 Answer")

        st.write(result["answer"])

        with st.expander("📄 Retrieved Chunks"):

            for i, chunk in enumerate(result["sources"], start=1):

                st.markdown(f"### Chunk {i}")

                st.write(chunk)

                st.divider()

# Footer

st.markdown("---")

st.caption(
    "Networking RAG | Phase 4 | Better Chunking + Hybrid Search + LangGraph"
)