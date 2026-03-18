from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

def create_vector_db(
    documents=None,
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    db_path="./chroma_db",
    collection_name="rag_collection"
):
    parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} nodes from documents.")

    embed_model = HuggingFaceEmbedding(model_name=embed_model)

    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        storage_context=storage_context,
    )

    # Debug: verify persistence immediately after indexing
    count = chroma_collection.count()
    print(f"Chroma collection count right after indexing: {count}")

    return index


def load_vector_db(
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    db_path="./chroma_db",
    collection_name="rag_collection"
) -> VectorStoreIndex:
    """Load an existing Chroma collection into a VectorStoreIndex."""
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_collection(collection_name)

    if chroma_collection.count() == 0:
        raise ValueError("Collection exists but is empty.")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = HuggingFaceEmbedding(model_name=embed_model)  

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    return index


if __name__ == "__main__":
    documents = SimpleDirectoryReader("data/", required_exts=[".pdf"]).load_data()
    create_vector_db(documents=documents)
    print("Vector DB created.")