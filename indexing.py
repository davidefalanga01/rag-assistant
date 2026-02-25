from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

def create_vector_db():
    # Load documents
    documents = SimpleDirectoryReader("data/").load_data()

    parser = SentenceSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    # Split data in nodes
    nodes = parser.get_nodes_from_documents(documents)

    # Define the embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma_client = chromadb.EphemeralClient()
    # chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.create_collection("quickstart")

    # Build the Chroma vector database
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        vector_store=vector_store
    )
    return index



if __name__ == "__main__":
    index = create_vector_db()

    # Test the retriever part
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve("Who is the CEO of APPLE? ")
    print(len(nodes))