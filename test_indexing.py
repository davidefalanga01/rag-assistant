from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os

from config import COLLECTION_NAME, DB_PATH
from vector_database import create_vector_db, load_raw_nodes, load_vector_db


def test_vector_db():

    DATA_DIR = "data/"

    # --- Test 1: create_vector_db ---
    print("=== Test 1: Creating vector DB ===")
    documents = SimpleDirectoryReader(DATA_DIR, required_exts=[".pdf"]).load_data()
    assert len(documents) > 0, "No documents loaded."
    print(f"Loaded {len(documents)} documents.")

    index = create_vector_db(
        documents=documents,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )
    assert index is not None, "Index was not created."
    print("Index created.")

    # --- Test 2: Check embeddings are persisted in Chroma ---
    print("\n=== Test 2: Checking Chroma collection ===")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    count = collection.count()
    assert count > 0, "Collection is empty — embeddings were not persisted."
    print(f"Collection contains {count} embedded nodes. ✓")

    # --- Test 3: Check embeddings are actually stored (not None) ---
    print("\n=== Test 3: Checking embedding vectors ===")
    result = collection.get(limit=1, include=["embeddings"])
    embedding = result["embeddings"][0]
    assert embedding is not None, "Embedding is None."
    assert len(embedding) > 0, "Embedding vector is empty."
    print(f"Sample embedding dim: {len(embedding)} ✓")

    # --- Test 4: load_vector_db ---
    print("\n=== Test 4: Loading vector DB ===")
    loaded_index = load_vector_db(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )
    assert loaded_index is not None, "Loaded index is None."
    print("Index loaded successfully. ✓")

    # --- Test 5: Query sanity check ---
    print("\n=== Test 5: Query sanity check ===")
    query_engine = loaded_index.as_retriever()
    response = query_engine.retrieve("What is this document about?")
    assert response is not None, "Query returned None."
    assert len(response) > 0, "Query returned empty response."
    print(f"Query response: {str(response[0])[:200]}... ✓")

    print("\n All tests passed.")

    # --- Test 6: load_raw_nodes basic reconstruction ---
    print("\n=== Test 6: Loading nodes from Chroma ===")
    nodes = load_raw_nodes(DB_PATH, COLLECTION_NAME)

    assert nodes is not None, "Nodes list is None."
    assert len(nodes) > 0, "No nodes reconstructed."
    assert isinstance(nodes[0], TextNode), "Returned objects are not TextNode."
    assert nodes[0].text is not None and len(nodes[0].text) > 0, "Node text is empty."

    print(f"Loaded {len(nodes)} nodes. ✓")

    # --- Test 7: ID consistency check ---
    print("\n=== Test 7: ID consistency check ===")

    # Get IDs directly from Chroma
    chroma_ids = set(collection.get(include=[])["ids"])

    # Get IDs from reconstructed nodes
    node_ids = set(n.id_ for n in nodes)

    assert chroma_ids == node_ids, (
        "Mismatch between Chroma IDs and reconstructed node IDs.\n"
        f"Missing in nodes: {chroma_ids - node_ids}\n"
        f"Extra in nodes: {node_ids - chroma_ids}"
    )

    print("Node IDs perfectly match Chroma IDs. ")


if __name__ == "__main__":
    test_vector_db()
