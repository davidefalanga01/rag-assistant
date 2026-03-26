import streamlit as st
from vector_database import load_vector_db
from generation import generation_model

# 1. Load your LlamaIndex data
index = load_vector_db()
llm = generation_model()
query_engine = index.as_query_engine(llm=llm)

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