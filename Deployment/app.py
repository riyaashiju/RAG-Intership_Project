"""
Networking RAG - Streamlit Chat Application
"""
import streamlit as st
from rag_pipeline import ask_network_question

st.set_page_config(page_title="Networking AI Assistant",page_icon="🌐",layout="wide")

with st.sidebar:
    st.title("🌐 Networking AI")
    st.write("Ask networking-related questions and receive context-aware answers from the networking knowledge base.")
    st.divider()
    st.subheader("💡 Example Questions")
    st.markdown("- What is DNS?\n- Difference between TCP and UDP?\n- Explain the OSI Model.\n- What is HTTP?\n- What is a router?\n- What is subnetting?")
    st.divider()
    if st.button("🗑️ New Chat",use_container_width=True):
        st.session_state.messages=[]
        st.rerun()
    st.caption("Responses are generated only from the networking knowledge base.")

st.title("🌐 Networking AI Assistant")
st.caption("Ask any networking-related question.")

if "messages" not in st.session_state:
    st.session_state.messages=[{"role":"assistant","content":"👋 Hello! I'm your Networking AI Assistant.\n\nAsk me anything about computer networking."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"],avatar="🤖" if m["role"]=="assistant" else "👤"):
        st.markdown(m["content"])

prompt=st.chat_input("Type your networking question...")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user",avatar="👤"):
        st.markdown(prompt)
    with st.chat_message("assistant",avatar="🤖"):
        with st.spinner("Generating response..."):
            result=ask_network_question(prompt)
            answer=result["answer"]
            st.markdown(answer)
            if result.get("sources"):
                with st.expander("🔍 View Retrieved Context"):
                    for i,chunk in enumerate(result["sources"],1):
                        st.markdown(f"**Context {i}**")
                        st.write(chunk)
                        st.divider()
    st.session_state.messages.append({"role":"assistant","content":answer})
