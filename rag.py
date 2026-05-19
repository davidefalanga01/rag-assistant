from typing import List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from config import (
    CANDIDATE_TOP_K,
    COLLECTION_NAME,
    DB_PATH,
    EMBED_MODEL,
    FINAL_TOP_K,
    SIMILARITY_CUTOFF,
    USE_ADAPTIVE_K,
    USE_HYBRID_SEARCH,
    USE_METADATA_FILTERING,
    USE_RERANKER,
)
from generation import generation_model
from vector_database import build_dense_retriever, build_hybrid_retriever, load_vector_db


class PostprocessedRetriever(BaseRetriever):
    """Apply query-engine node postprocessors inside retrieval evaluation."""

    def __init__(self, retriever: BaseRetriever, node_postprocessors: list):
        super().__init__()
        self._retriever = retriever
        self._node_postprocessors = node_postprocessors

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)
        fallback_nodes = nodes[:1]

        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes,
                query_bundle=query_bundle,
            )
            if not nodes and fallback_nodes:
                return fallback_nodes

        return nodes

    @property
    def last_filter(self):
        return getattr(self._retriever, "last_filter", None)


class SafeSimilarityPostprocessor(SimilarityPostprocessor):
    """Keep the best candidate when adaptive-k would remove every node."""

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> List[NodeWithScore]:
        filtered_nodes = super()._postprocess_nodes(
            nodes,
            query_bundle=query_bundle,
        )

        if filtered_nodes or not nodes:
            return filtered_nodes

        return nodes[:1]


def build_node_postprocessors():
    node_postprocessors = []

    if USE_RERANKER:
        node_postprocessors.append(
            FlagEmbeddingReranker(
                model="BAAI/bge-reranker-base",
                top_n=FINAL_TOP_K,
            )
        )

    if USE_ADAPTIVE_K:
        node_postprocessors.append(
            SafeSimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
        )

    return node_postprocessors


def configured_top_k() -> int:
    if USE_RERANKER or USE_ADAPTIVE_K:
        return CANDIDATE_TOP_K
    return FINAL_TOP_K


def build_configured_retriever(index, top_k: int | None = None):
    top_k = top_k or configured_top_k()

    if USE_HYBRID_SEARCH:
        return build_hybrid_retriever(
            index=index,
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            dense_top_k=top_k,
            sparse_top_k=top_k,
            fusion_top_k=top_k,
            infer_filters_from_query=USE_METADATA_FILTERING,
        )

    return build_dense_retriever(
        index=index,
        similarity_top_k=top_k,
        infer_filters_from_query=USE_METADATA_FILTERING,
    )


def build_retriever_for_eval(index):
    retriever = build_configured_retriever(index)
    node_postprocessors = build_node_postprocessors()

    if not node_postprocessors:
        return retriever

    return PostprocessedRetriever(
        retriever=retriever,
        node_postprocessors=node_postprocessors,
    )


def build_query_engine(llm=None):
    index = load_vector_db(
        embed_model=EMBED_MODEL,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )
    retriever = build_configured_retriever(index)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm or generation_model("extra"),
        node_postprocessors=build_node_postprocessors(),
    )


if __name__ == "__main__":
    rag = build_query_engine()

    question = "How many stocks of Apple Inc are free on the market?"
    response = rag.query(question)

    print("Question:", question)
    print("Response:", response)

    for i, node in enumerate(response.source_nodes):
        print(f"Rank {i + 1}: Node ID: {node.node.node_id} | Score: {node.score}")
