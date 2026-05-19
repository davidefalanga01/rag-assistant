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

from config import COLLECTION_NAME, DB_PATH, EMBED_MODEL, ENHANCED_CHUNKING
from retrieval_filters import RetrievalFilter, infer_retrieval_filter

_ITEM_PATTERN = re.compile(
    r"(?i)^\s*item\s+\d+[a-z]?\b\.?\s",
    re.MULTILINE,
)

_SEC_ITEM_INFO = {
    "PREAMBLE": ("Preamble", "preamble"),
    "ITEM 1": ("Business", "business"),
    "ITEM 1A": ("Risk Factors", "risk_factors"),
    "ITEM 1B": ("Unresolved Staff Comments", "staff_comments"),
    "ITEM 1C": ("Cybersecurity", "cybersecurity"),
    "ITEM 2": ("Properties", "properties"),
    "ITEM 3": ("Legal Proceedings", "legal_proceedings"),
    "ITEM 4": ("Mine Safety Disclosures", "mine_safety"),
    "ITEM 5": ("Market for Registrant's Common Equity", "market_equity"),
    "ITEM 6": ("Reserved", "reserved"),
    "ITEM 7": ("Management's Discussion and Analysis", "mda"),
    "ITEM 7A": ("Quantitative and Qualitative Disclosures About Market Risk", "market_risk"),
    "ITEM 8": ("Financial Statements and Supplementary Data", "financial_statements"),
    "ITEM 9": ("Changes in and Disagreements with Accountants", "accounting_disagreements"),
    "ITEM 9A": ("Controls and Procedures", "controls_procedures"),
    "ITEM 9B": ("Other Information", "other_information"),
    "ITEM 9C": ("Disclosure Regarding Foreign Jurisdictions", "foreign_jurisdictions"),
    "ITEM 10": ("Directors, Executive Officers and Corporate Governance", "governance"),
    "ITEM 11": ("Executive Compensation", "executive_compensation"),
    "ITEM 12": ("Security Ownership", "security_ownership"),
    "ITEM 13": ("Certain Relationships and Related Transactions", "related_transactions"),
    "ITEM 14": ("Principal Accountant Fees and Services", "accountant_fees"),
    "ITEM 15": ("Exhibits and Financial Statement Schedules", "exhibits"),
    "ITEM 16": ("Form 10-K Summary", "form_10k_summary"),
}


def _split_by_sec_items(text: str) -> List[str]:
    """
    Hard-split a 10-K document on SEC Item headers.
    Returns a list of section strings, each starting with its Item header.
    """
    boundaries = [m.start() for m in _ITEM_PATTERN.finditer(text)]
    if not boundaries:
        return [text.strip()] if text.strip() else []

    sections = []

    if boundaries[0] > 0:
        prefix = text[: boundaries[0]].strip()
        if prefix:
            sections.append(prefix)

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _normalize_sec_item(text: str) -> Optional[str]:
    """Return normalized SEC item labels such as ITEM 7A."""
    match = re.match(r"(?i)\s*item\s+(\d+[a-z]?)", text)
    if not match:
        return None
    return f"ITEM {match.group(1).upper()}"


def _is_probable_table_of_contents(text: str) -> bool:
    """Avoid letting a contents page poison section propagation."""
    item_count = len(_ITEM_PATTERN.findall(text))
    if item_count < 4:
        return False

    lower = text.lower()
    toc_markers = ("table of contents", "form 10-k", "page")
    return any(marker in lower for marker in toc_markers)


def _section_metadata(sec_item: str) -> dict:
    title, group = _SEC_ITEM_INFO.get(sec_item, ("Unknown", "unknown"))
    return {
        "sec_item": sec_item,
        "section_title": title,
        "section_group": group,
    }


def _document_metadata(base_meta: dict) -> dict:
    file_name = str(base_meta.get("file_name", "unknown"))
    lower_file_name = file_name.lower()

    metadata = {
        "company_name": "Unknown",
        "ticker": "UNKNOWN",
        "form_type": "10-K",
        "fiscal_year": 0,
        "period_end_date": "unknown",
        "filing_date": "unknown",
        "source_file": file_name,
        "source_page": str(base_meta.get("page_label", "unknown")),
    }

    if "apple" in lower_file_name and "2024" in lower_file_name:
        metadata.update(
            {
                "company_name": "Apple Inc.",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "period_end_date": "2024-09-28",
                "filing_date": "2024-11-01",
            }
        )

    return metadata


def _infer_statement_type(text: str, sec_item: str) -> str:
    if sec_item != "ITEM 8":
        return "not_applicable"

    lower = text.lower()

    if "consolidated statements of operations" in lower:
        return "income_statement"
    if "consolidated balance sheets" in lower:
        return "balance_sheet"
    if "consolidated statements of cash flows" in lower:
        return "cash_flow_statement"
    if "consolidated statements of shareholders" in lower:
        return "shareholders_equity"
    if "consolidated statements of comprehensive income" in lower:
        return "comprehensive_income"
    if "notes to consolidated financial statements" in lower:
        return "notes"
    if "report of independent registered public accounting firm" in lower:
        return "auditor_report"

    return "financial_statement_unknown"


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
    current_sec_item = "PREAMBLE"
    current_statement_type = "not_applicable"
    chunk_index = 0

    def add_node(node: TextNode, metadata: dict) -> None:
        nonlocal chunk_index
        node.metadata = {
            **(node.metadata or {}),
            **metadata,
            "chunk_index": chunk_index,
        }
        all_nodes.append(node)
        chunk_index += 1

    for doc_index, doc in enumerate(documents):
        full_text = doc.get_content()
        base_meta = doc.metadata or {}
        document_meta = {
            **base_meta,
            **_document_metadata(base_meta),
            "document_index": doc_index,
        }
        is_toc_page = _is_probable_table_of_contents(full_text)

        sections = _split_by_sec_items(full_text)

        for section_index, section in enumerate(sections):
            detected_item = _normalize_sec_item(section)

            if detected_item and not is_toc_page:
                current_sec_item = detected_item
                current_statement_type = (
                    "financial_statement_unknown"
                    if detected_item == "ITEM 8"
                    else "not_applicable"
                )

            item_label = detected_item or current_sec_item
            statement_type = _infer_statement_type(section, item_label)

            if item_label == "ITEM 8":
                if statement_type != "financial_statement_unknown":
                    current_statement_type = statement_type
                else:
                    statement_type = current_statement_type
            else:
                statement_type = "not_applicable"

            section_meta = {
                **document_meta,
                **_section_metadata(item_label),
                "chunk_type": "narrative",
                "statement_type": statement_type,
                "section_index": section_index,
            }

            if _is_table_block(section):
                node = TextNode(
                    text=section,
                    metadata={**section_meta, "chunk_type": "table"},
                )
                add_node(node, {**section_meta, "chunk_type": "table"})
                continue

            token_estimate = len(section.split())
            if token_estimate <= fallback_chunk_size:
                node = TextNode(text=section, metadata=section_meta)
                add_node(node, section_meta)
                continue

            from llama_index.core.schema import Document as LIDocument

            section_doc = LIDocument(text=section, metadata=section_meta)
            try:
                sub_nodes = semantic_splitter.get_nodes_from_documents([section_doc])
            except Exception:
                sub_nodes = fallback_splitter.get_nodes_from_documents([section_doc])

            for node in sub_nodes:
                add_node(node, section_meta)

    return all_nodes


def create_vector_db(
    documents=None,
    embed_model=EMBED_MODEL,
    db_path=DB_PATH,
    collection_name=COLLECTION_NAME,
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
    embed_model=EMBED_MODEL,
    db_path=DB_PATH,
    collection_name=COLLECTION_NAME,
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


class AtLeastOneRetriever(BaseRetriever):
    """Fallback to an unfiltered retriever if the primary retriever returns no nodes."""

    def __init__(
        self,
        retriever: BaseRetriever,
        fallback_retriever: BaseRetriever,
        fallback_top_k: int = 1,
    ):
        super().__init__()
        self._retriever = retriever
        self._fallback_retriever = fallback_retriever
        self._fallback_top_k = fallback_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        results = self._retriever.retrieve(query_bundle)
        if results:
            return results

        return self._fallback_retriever.retrieve(query_bundle)[: self._fallback_top_k]

    @property
    def last_filter(self):
        return getattr(self._retriever, "last_filter", None)


def _build_static_hybrid_retriever(
    index: VectorStoreIndex,
    nodes: List[TextNode],
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    fusion_top_k: int = 10,
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
    retrieval_filter: Optional[RetrievalFilter] = None,
) -> BaseRetriever:
    metadata_filters = None
    sparse_nodes = nodes

    if retrieval_filter is not None and not retrieval_filter.is_empty():
        filtered_nodes = [node for node in nodes if retrieval_filter.matches_node(node)]
        if filtered_nodes:
            sparse_nodes = filtered_nodes
            metadata_filters = retrieval_filter.to_llama_filters()

    dense_retriever = index.as_retriever(
        similarity_top_k=dense_top_k,
        filters=metadata_filters,
    )
    sparse_retriever = BM25Retriever.from_defaults(
        nodes=sparse_nodes,
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

    retriever = DeduplicatingRetriever(fusion_retriever, top_k=fusion_top_k)
    fallback_retriever = index.as_retriever(similarity_top_k=1)

    return AtLeastOneRetriever(
        retriever=retriever,
        fallback_retriever=fallback_retriever,
    )


class QueryDerivedFilterRetriever(BaseRetriever):
    """Infer metadata filters per query, then run hybrid retrieval on that scope."""

    def __init__(
        self,
        index: VectorStoreIndex,
        nodes: List[TextNode],
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        fusion_top_k: int = 10,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        base_filter: Optional[RetrievalFilter] = None,
    ):
        super().__init__()
        self._index = index
        self._nodes = nodes
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._fusion_top_k = fusion_top_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._base_filter = base_filter
        self._retriever_cache = {}
        self._fallback_retriever = index.as_retriever(similarity_top_k=1)
        self.last_filter: Optional[RetrievalFilter] = None

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        inferred_filter = infer_retrieval_filter(query_bundle.query_str)
        active_filter = (
            self._base_filter.merge(inferred_filter)
            if self._base_filter is not None
            else inferred_filter
        )

        if active_filter.is_empty():
            active_filter = None

        self.last_filter = active_filter
        cache_key = active_filter or RetrievalFilter()

        if cache_key not in self._retriever_cache:
            self._retriever_cache[cache_key] = _build_static_hybrid_retriever(
                index=self._index,
                nodes=self._nodes,
                dense_top_k=self._dense_top_k,
                sparse_top_k=self._sparse_top_k,
                fusion_top_k=self._fusion_top_k,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
                retrieval_filter=active_filter,
            )

        results = self._retriever_cache[cache_key].retrieve(query_bundle)
        if results:
            return results

        return self._fallback_retriever.retrieve(query_bundle)[:1]


class QueryDerivedDenseFilterRetriever(BaseRetriever):
    """Infer metadata filters per query, then run dense retrieval on that scope."""

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 10,
        base_filter: Optional[RetrievalFilter] = None,
    ):
        super().__init__()
        self._index = index
        self._similarity_top_k = similarity_top_k
        self._base_filter = base_filter
        self._retriever_cache = {}
        self._fallback_retriever = index.as_retriever(similarity_top_k=1)
        self.last_filter: Optional[RetrievalFilter] = None

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        inferred_filter = infer_retrieval_filter(query_bundle.query_str)
        active_filter = (
            self._base_filter.merge(inferred_filter)
            if self._base_filter is not None
            else inferred_filter
        )

        if active_filter.is_empty():
            active_filter = None

        self.last_filter = active_filter
        cache_key = active_filter or RetrievalFilter()

        if cache_key not in self._retriever_cache:
            self._retriever_cache[cache_key] = self._index.as_retriever(
                similarity_top_k=self._similarity_top_k,
                filters=active_filter.to_llama_filters() if active_filter else None,
            )

        results = self._retriever_cache[cache_key].retrieve(query_bundle)
        if results:
            return results

        return self._fallback_retriever.retrieve(query_bundle)[:1]


def build_dense_retriever(
    index: VectorStoreIndex,
    similarity_top_k: int = 10,
    retrieval_filter: Optional[RetrievalFilter] = None,
    infer_filters_from_query: bool = False,
) -> BaseRetriever:
    """Build a vector-only retriever, optionally scoped by metadata filters."""

    if infer_filters_from_query:
        return QueryDerivedDenseFilterRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            base_filter=retrieval_filter,
        )

    retriever = index.as_retriever(
        similarity_top_k=similarity_top_k,
        filters=(
            retrieval_filter.to_llama_filters()
            if retrieval_filter is not None and not retrieval_filter.is_empty()
            else None
        ),
    )
    fallback_retriever = index.as_retriever(similarity_top_k=1)

    return AtLeastOneRetriever(
        retriever=retriever,
        fallback_retriever=fallback_retriever,
    )


def build_hybrid_retriever(
    index: VectorStoreIndex,
    nodes: Optional[List[TextNode]] = None,
    db_path: str = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    dense_top_k: int = 20,
    sparse_top_k: int = 20,
    fusion_top_k: int = 10,
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
    retrieval_filter: Optional[RetrievalFilter] = None,
    infer_filters_from_query: bool = False,
) -> BaseRetriever:
    """
    Build a dense + sparse hybrid retriever over the same Chroma collection.

    Dense retrieval uses the vector index. Sparse retrieval uses BM25 over the
    raw TextNodes reconstructed from Chroma, then LlamaIndex fuses the rankings.
    Query-derived filtering is optional and scopes both dense and sparse sides.
    """

    if nodes is None:
        nodes = load_raw_nodes(db_path=db_path, collection_name=collection_name)

    if not nodes:
        raise ValueError("Cannot build a hybrid retriever without indexed nodes.")

    if infer_filters_from_query:
        return QueryDerivedFilterRetriever(
            index=index,
            nodes=nodes,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            fusion_top_k=fusion_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            base_filter=retrieval_filter,
        )

    return _build_static_hybrid_retriever(
        index=index,
        nodes=nodes,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        fusion_top_k=fusion_top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        retrieval_filter=retrieval_filter,
    )


if __name__ == "__main__":
    documents = SimpleDirectoryReader("data/", required_exts=[".pdf"]).load_data()
    create_vector_db(
        documents=documents,
        embed_model=EMBED_MODEL,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )
    print("Vector DB created.")
