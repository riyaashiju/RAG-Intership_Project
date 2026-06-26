"""
Networking RAG - Phase 4 Deployment Pipeline
Generated from Phase 4 notebook.
"""

import os
import pickle
import chromadb
from typing import TypedDict
from google import genai
from groq import Groq
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, START, END
import streamlit as st

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
gemini_client=genai.Client(api_key=GEMINI_API_KEY)
groq_client=Groq(api_key=GROQ_API_KEY)

with open("improved_chunks.pkl","rb") as f:
    improved_chunks=pickle.load(f)

with open("improved_embeddings.pkl","rb") as f:
    improved_embeddings=pickle.load(f)

chroma_client=chromadb.PersistentClient(path="./networking_chromadb_phase4")
collection=chroma_client.get_or_create_collection(name="networking_docs_phase4")

tokenized_chunks=[c.lower().split() for c in improved_chunks]
bm25=BM25Okapi(tokenized_chunks)

def hybrid_retrieve(question, top_k=5):
    response=gemini_client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=question
    )
    query_embedding=response.embeddings[0].values
    vector_results=collection.query(query_embeddings=[query_embedding], n_results=top_k)
    vector_chunks=vector_results["documents"][0]
    tokenized_query=question.lower().split()
    bm25_scores=bm25.get_scores(tokenized_query)
    bm25_indices=sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_chunks=[improved_chunks[i] for i in bm25_indices]
    combined=[]
    for chunk in vector_chunks+bm25_chunks:
        if chunk not in combined:
            combined.append(chunk)
    return combined[:top_k]

class GraphState(TypedDict):
    question:str
    retrieved_chunks:list
    answer:str

def retrieve_chunks(state):
    state["retrieved_chunks"]=hybrid_retrieve(state["question"])
    return state

def generate_answer(state):
    context="\n\n".join(state["retrieved_chunks"])
    prompt=f"""You are a networking assistant.

Answer only using the provided context.

If the answer is not available in the context, respond with:
I could not find that information in the knowledge base.

Context:
{context}

Question:
{state['question']}
"""
    response=groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )
    state["answer"]=response.choices[0].message.content
    return state

builder=StateGraph(GraphState)
builder.add_node("retrieve_chunks", retrieve_chunks)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START,"retrieve_chunks")
builder.add_edge("retrieve_chunks","generate_answer")
builder.add_edge("generate_answer",END)
graph=builder.compile()

def ask_network_question(question):
    result=graph.invoke({"question":question})
    return {
        "answer":result["answer"],
        "sources":result["retrieved_chunks"]
    }
