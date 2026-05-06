from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.mock import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from typing import List, Optional
import chromadb
import hashlib
import re

NEW_EMBED_MODEL = True  # Set to False to use default
ENHANCED_CHUNKING = True  # Set to False to use simple sentence splitting

_ITEM_PATTERN = re.compile(
    r"(?i)^\s*item\s+\d+[a-z]?\b\.?\s",
    re.MULTILINE,
)


def _split_by_sec_items(text: str) -> List[str]:
    """
    Hard-split a 10-K document on SEC Item headers.
    Returns a list of section strings, each starting with its Item header.
    """
    boundaries = [m.start() for m in _ITEM_PATTERN.finditer(text)]
    if not boundaries:
        return [text]

    sections = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        sections.append(text[start:end].strip())
    return sections


def _is_table_block(text: str) -> bool:
    """Heuristic: a block is a table if more than 30% of lines look tabular."""
    lines = text.splitlines()
    if not lines:
        return False

    tabular = sum(1 for line in lines if len(re.split(r"\s{3,}", line.strip())) > 2)
    return (tabular / len(lines)) > 0.30


def parse_10k_nodes(
    documents,
    embed_model_instance: HuggingFaceEmbedding,
    semantic_threshold: float = 0.20,
    fallback_chunk_size: int = 512,
    fallback_chunk_overlap: int = 64,
) -> List[TextNode]:
    """
    Hybrid document-aware + semantic chunking for 10-K filings.

    Pipeline per document:
      1. Hard-split on SEC Item boundaries
      2. Tables detected are kept as atomic nodes
      3. Long narrative sections use SemanticSplitter
      4. Short sections are kept as-is
    """
    fallback_splitter = SentenceSplitter(
        chunk_size=fallback_chunk_size,
        chunk_overlap=fallback_chunk_overlap,
    )
    semantic_splitter = SemanticSplitterNodeParser(
        embed_model=embed_model_instance,
        breakpoint_percentile_threshold=int((1 - semantic_threshold) * 100),
    )

    all_nodes: List[TextNode] = []

    for doc in documents:
        full_text = doc.get_content()
        base_meta = doc.metadata or {}

        sections = _split_by_sec_items(full_text)

        for section in sections:
            item_match = re.match(r"(?i)(item\s+\d+[a-z]?)", section)
            item_label = item_match.group(1).upper() if item_match else "PREAMBLE"

            section_meta = {
                **base_meta,
                "sec_item": item_label,
                "chunk_type": "narrative",
            }

            if _is_table_block(section):
                node = TextNode(
                    text=section,
                    metadata={**section_meta, "chunk_type": "table"},
                )
                all_nodes.append(node)
                continue

            token_estimate = len(section.split())
            if token_estimate <= fallback_chunk_size:
                node = TextNode(text=section, metadata=section_meta)
                all_nodes.append(node)
                continue

            from llama_index.core.schema import Document as LIDocument

            section_doc = LIDocument(text=section, metadata=section_meta)
            try:
                sub_nodes = semantic_splitter.get_nodes_from_documents([section_doc])
            except Exception:
                sub_nodes = fallback_splitter.get_nodes_from_documents([section_doc])

            for node in sub_nodes:
                node.metadata.setdefault("sec_item", item_label)
                node.metadata.setdefault("chunk_type", "narrative")
            all_nodes.extend(sub_nodes)

    return all_nodes


def create_vector_db(
    documents=None,
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    db_path="./chroma_db",
    collection_name="rag_collection",
):
    if ENHANCED_CHUNKING:
        print("Using enhanced 10-K chunking strategy.")
        embed_model_instance = HuggingFaceEmbedding(model_name=embed_model)
        raw_nodes = parse_10k_nodes(documents, embed_model_instance)
    else:
        print("Using simple sentence splitting for chunking.")
        parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        raw_nodes = parser.get_nodes_from_documents(documents)

    unique_nodes_dict = {}
    for node in raw_nodes:
        node_id = hashlib.md5(node.get_content().encode("utf-8")).hexdigest()
        node.id_ = node_id
        unique_nodes_dict[node_id] = node

    unique_nodes = list(unique_nodes_dict.values())

    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    if chroma_collection.count() > 0:
        existing_ids = set(chroma_collection.get(ids=[n.id_ for n in unique_nodes])["ids"])
        final_nodes = [n for n in unique_nodes if n.id_ not in existing_ids]
    else:
        final_nodes = unique_nodes

    print(f"Adding {len(final_nodes)} new nodes (duplicates skipped).")

    if not final_nodes:
        print("No new nodes to add. Loading existing index.")
        return (
            load_vector_db(
                embed_model=embed_model,
                db_path=db_path,
                collection_name=collection_name,
            ),
            unique_nodes,
        )

    if not ENHANCED_CHUNKING:
        embed_model_instance = HuggingFaceEmbedding(model_name=embed_model)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        final_nodes,
        embed_model=embed_model_instance,
        storage_context=storage_context,
    )

    print(f"Chroma collection count: {chroma_collection.count()}")

    return index


def load_vector_db(
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    db_path="./chroma_db",
    collection_name="rag_collection",
) -> VectorStoreIndex:
    """Load an existing Chroma collection into a VectorStoreIndex."""
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_collection = chroma_client.get_collection(collection_name)

    if chroma_collection.count() == 0:
        raise ValueError("Collection exists but is empty.")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model_instance = HuggingFaceEmbedding(model_name=embed_model)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model_instance,
    )

    return index


def load_raw_nodes(db_path: str, collection_name: str) -> List[TextNode]:
    """
    Reconstruct TextNode objects directly from the Chroma collection.

    Chroma stores ids, embeddings, documents, and metadatas. We rebuild TextNode
    objects so that LlamaIndex eval utilities can work without the source files.
    """
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(collection_name)

    if collection.count() == 0:
        return []

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
            metadata=metadata or {},
        )
        node.id_ = ids[i]

        nodes.append(node)

    return nodes


class DeduplicatingRetriever(BaseRetriever):
    """Remove duplicate node ids after fusion while preserving the best score."""

    def __init__(self, retriever: BaseRetriever, top_k: int):
        super().__init__()
        self._retriever = retriever
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        results = self._retriever.retrieve(query_bundle)
        best_by_node_id = {}

        for result in results:
            node_id = result.node.node_id
            current = best_by_node_id.get(node_id)
            if current is None or (result.score or 0.0) > (current.score or 0.0):
                best_by_node_id[node_id] = result

        deduped_results = sorted(
            best_by_node_id.values(),
            key=lambda result: result.score or 0.0,
            reverse=True,
        )
        return deduped_results[: self._top_k]


def build_hybrid_retriever(
    index: VectorStoreIndex,
    nodes: Optional[List[TextNode]] = None,
    db_path: str = "./chroma_db",
    collection_name: str = "rag_collection",
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    fusion_top_k: int = 10,
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
) -> BaseRetriever:
    """
    Build a dense + sparse hybrid retriever over the same Chroma collection.

    Dense retrieval uses the vector index. Sparse retrieval uses BM25 over the
    raw TextNodes reconstructed from Chroma, then LlamaIndex fuses the rankings.
    """

    if nodes is None:
        nodes = load_raw_nodes(db_path=db_path, collection_name=collection_name)

    if not nodes:
        raise ValueError("Cannot build a hybrid retriever without indexed nodes.")

    dense_retriever = index.as_retriever(similarity_top_k=dense_top_k)
    sparse_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=sparse_top_k,
    )

    fusion_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        llm=MockLLM(),
        mode=FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k=fusion_top_k * 2,
        num_queries=1,
        use_async=False,
        retriever_weights=[dense_weight, sparse_weight],
    )

    return DeduplicatingRetriever(fusion_retriever, top_k=fusion_top_k)


if __name__ == "__main__":
    documents = SimpleDirectoryReader("data/", required_exts=[".pdf"]).load_data()
    if NEW_EMBED_MODEL:
        create_vector_db(
            documents=documents,
            embed_model="BAAI/bge-large-en-v1.5",
            collection_name="rag_collection_v2",
        )
    else:
        create_vector_db(documents=documents)
    print("Vector DB created.")
