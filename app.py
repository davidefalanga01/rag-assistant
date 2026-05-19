import streamlit as st

from config import (
    COLLECTION_NAME,
    ENHANCED_CHUNKING,
    NEW_EMBED_MODEL,
    USE_ADAPTIVE_K,
    USE_HYBRID_SEARCH,
    USE_METADATA_FILTERING,
    USE_RERANKER,
)
from generation import generation_model
from rag import build_query_engine


@st.cache_resource
def load_query_engine():
    return build_query_engine(llm=generation_model())


query_engine = load_query_engine()

st.title("My RAG Search Engine")
st.write("Ask any question about my documents.")

with st.sidebar:
    st.subheader("Active RAG configuration")
    st.write(f"Collection: `{COLLECTION_NAME}`")
    st.checkbox("New embed model", value=NEW_EMBED_MODEL, disabled=True)
    st.checkbox("Enhanced chunking", value=ENHANCED_CHUNKING, disabled=True)
    st.checkbox("Reranker", value=USE_RERANKER, disabled=True)
    st.checkbox("Adaptive-k", value=USE_ADAPTIVE_K, disabled=True)
    st.checkbox("Metadata filtering", value=USE_METADATA_FILTERING, disabled=True)
    st.checkbox("Hybrid search", value=USE_HYBRID_SEARCH, disabled=True)

user_query = st.text_input("Enter your question:")

if user_query:
    with st.spinner("Searching documents..."):
        response = query_engine.query(user_query)

    st.subheader("Answer:")
    st.write(response.response)

    with st.expander("View Source Documents"):
        for node in response.source_nodes:
            st.write({"score": node.score, "metadata": node.node.metadata})
            st.write(node.node.get_content())
