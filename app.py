import streamlit as st
from llama_index.core.query_engine import RetrieverQueryEngine

from vector_database import build_hybrid_retriever, load_vector_db
from generation import generation_model

# 1. Load your LlamaIndex data
index = load_vector_db()
llm = generation_model()
hybrid_retriever = build_hybrid_retriever(
    index=index,
    dense_top_k=20,
    sparse_top_k=20,
    fusion_top_k=10,
)
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    llm=llm,
)

# 2. Build the UI
st.title("My RAG Search Engine 🔍")
st.write("Ask any question about my documents.")

# 3. Create the search bar
user_query = st.text_input("Enter your question:")

# 4. Handle the query
if user_query:
    with st.spinner("Searching documents..."):
        response = query_engine.query(user_query)
        
        st.subheader("Answer:")
        st.write(response.response)
        
        # Optional: Show sources/references
        with st.expander("View Source Documents"):
            for node in response.source_nodes:
                st.write(node.text)
