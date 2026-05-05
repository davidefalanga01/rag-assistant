from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.postprocessor import SimilarityPostprocessor


from dotenv import load_dotenv
import chromadb
import os

from vector_database import load_vector_db
from generation import generation_model

load_dotenv()

USE_RERANKER = True
USE_ADAPTIVE_K = True

if __name__ == "__main__":
    index = load_vector_db()
    generator = generation_model("extra")

    node_postprocessors = []

    if USE_RERANKER:
        reranker = FlagEmbeddingReranker(
            model="BAAI/bge-reranker-base",
            top_n=5
        )
        node_postprocessors.append(reranker)
    
    if USE_ADAPTIVE_K:
        node_postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=0.60)
        )
    
    top_k = 5
    if node_postprocessors:
        top_k = 20

    # Test the rag
    rag = index.as_query_engine(llm=generator, 
                                similarity_top_k=top_k,
                                node_postprocessors=node_postprocessors)


    question = "How many stocks of Apple Inc are free on the market?"
    response = rag.query(question)

    print("Question:", question)
    print("Response:", response)

    for i, node in enumerate(response.source_nodes):
        print(f"Rank {i+1}: Node ID: {node.node.node_id} | Score: {node.score}")
