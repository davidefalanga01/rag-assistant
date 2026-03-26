from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage

from dotenv import load_dotenv
import chromadb
import os

from vector_database import load_vector_db
from generation import generation_model

load_dotenv()

if __name__ == "__main__":
    index = load_vector_db()
    generator = generation_model()

    # Test the naive rag
    rag = index.as_query_engine(llm=generator)
    question = "How many stocks of Apple Inc are free on the market?"
    response = rag.query(question)
    print("Question:", question)
    print("Response:", response)

    for node in response.__dict__['source_nodes']:
        print(node)
    
    