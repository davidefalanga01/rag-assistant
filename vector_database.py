from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode

from typing import List
import chromadb
import hashlib

def create_vector_db(
    documents=None,
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    db_path="./chroma_db",
    collection_name="rag_collection"
):
    parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)
    raw_nodes = parser.get_nodes_from_documents(documents)
    
    # 1. Intra-batch Deduplication (Fixes the Chroma DuplicateIDError)
    unique_nodes_dict = {}
    for node in raw_nodes:
        node_id = hashlib.md5(node.get_content().encode("utf-8")).hexdigest()
        node.id_ = node_id
        # By using a dictionary, duplicate IDs simply overwrite each other, 
        # leaving only one node per unique text.
        unique_nodes_dict[node_id] = node 
        
    unique_nodes = list(unique_nodes_dict.values())

    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    # 2. DB Deduplication (Your logic: prevents re-embedding existing chunks)
    if chroma_collection.count() > 0:
        # Ask Chroma which of our unique IDs it already has
        existing_ids = set(chroma_collection.get(ids=[n.id_ for n in unique_nodes])["ids"])
        final_nodes = [n for n in unique_nodes if n.id_ not in existing_ids]
    else:
        final_nodes = unique_nodes

    print(f"Adding {len(final_nodes)} new nodes (duplicates skipped).")

    if not final_nodes:
        # All nodes already indexed — just load and return
        print("No new nodes to add. Loading existing index.")
        # Ensure you return the existing nodes as well for your eval dataset!
        return load_vector_db(embed_model=embed_model, db_path=db_path, collection_name=collection_name), unique_nodes

    embed_model = HuggingFaceEmbedding(model_name=embed_model)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        final_nodes, # <-- Pass the fully deduplicated list!
        embed_model=embed_model,
        storage_context=storage_context,
    )

    print(f"Chroma collection count: {chroma_collection.count()}")
    
    # Returning unique_nodes allows your eval script to use the exact same hashed nodes
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


def load_raw_nodes(db_path: str, collection_name: str) -> List[TextNode]:
    """
    Reconstruct TextNode objects directly from the Chroma collection.

    Chroma stores: ids, embeddings, documents (raw text), metadatas.
    We rebuild TextNode objects so that LlamaIndex eval utilities work
    without needing the original document files.
    """

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(collection_name)

    if collection.count() == 0:
        return []

    # Fetch everything (you may batch this if very large)
    results = collection.get(include=["documents", "metadatas"])

    ids = results["ids"]
    documents = results["documents"]
    metadatas = results["metadatas"]

    nodes = []
    for i in range(len(ids)):
        text = documents[i]
        metadata = metadatas[i] if metadatas else {}

        node = TextNode(
            text=text,
            metadata=metadata or {}
        )

        # Restore original ID (important for eval consistency)
        node.id_ = ids[i]

        nodes.append(node)

    return nodes


if __name__ == "__main__":
    documents = SimpleDirectoryReader("data/", required_exts=[".pdf"]).load_data()
    create_vector_db(documents=documents)
    print("Vector DB created.")