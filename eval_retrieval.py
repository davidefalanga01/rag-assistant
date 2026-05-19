'''Evaluate the Retrieval capabilities of the RAG assistant on a set of test queries.'''
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import numpy as np
import pandas as pd
from config import COLLECTION_NAME, DB_PATH, EMBED_MODEL
from rag import build_retriever_for_eval
from vector_database import load_vector_db

SOFT_SIMILARITY_THRESHOLD = 0.70
RUN_LLM_JUDGE = False
LLM_JUDGE_TOP_N = 5


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def get_node_text(node_with_score):
    node = getattr(node_with_score, "node", node_with_score)

    if hasattr(node, "get_content"):
        return node.get_content(metadata_mode=MetadataMode.NONE)

    return getattr(node, "text", str(node))


def get_node_id(node_with_score):
    node = getattr(node_with_score, "node", node_with_score)
    return getattr(node, "node_id", None) or getattr(node, "id_", None)


def compute_hard_retrieval_metrics(retrieved_nodes, expected_ids):
    retrieved_ids = [node_id for node_id in map(get_node_id, retrieved_nodes) if node_id]
    expected_set = set(expected_ids or [])

    if not retrieved_ids or not expected_set:
        return {
            "hit_rate": 0.0,
            "mrr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "ap": 0.0,
            "ndcg": 0.0,
        }

    seen_relevant_ids = set()
    hits = []
    for node_id in retrieved_ids:
        is_new_relevant = node_id in expected_set and node_id not in seen_relevant_ids
        hits.append(1 if is_new_relevant else 0)
        if is_new_relevant:
            seen_relevant_ids.add(node_id)
    hit_count = sum(hits)

    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break

    precision = hit_count / len(retrieved_ids)
    recall = hit_count / len(expected_set)

    precision_sum = 0.0
    relevant_seen = 0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
    ap = precision_sum / min(len(expected_set), len(retrieved_ids))

    dcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = [1] * min(len(expected_set), len(retrieved_ids))
    idcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(ideal_hits, start=1))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "hit_rate": float(hit_count > 0),
        "mrr": float(mrr),
        "precision": float(precision),
        "recall": float(recall),
        "ap": float(ap),
        "ndcg": float(ndcg),
    }


def get_cached_embedding(text, embed_model, embedding_cache):
    if text not in embedding_cache:
        embedding_cache[text] = embed_model.get_text_embedding(text)
    return embedding_cache[text]


def compute_similarity_matrix(retrieved_texts, reference_texts, embed_model, embedding_cache):
    matrix = np.zeros((len(retrieved_texts), len(reference_texts)))

    for i, retrieved_text in enumerate(retrieved_texts):
        retrieved_embedding = get_cached_embedding(
            retrieved_text,
            embed_model,
            embedding_cache,
        )

        for j, reference_text in enumerate(reference_texts):
            reference_embedding = get_cached_embedding(
                reference_text,
                embed_model,
                embedding_cache,
            )
            matrix[i, j] = cosine_similarity(retrieved_embedding, reference_embedding)

    return matrix


def compute_soft_retrieval_metrics(
    retrieved_nodes,
    reference_texts,
    embed_model,
    embedding_cache,
    similarity_threshold=SOFT_SIMILARITY_THRESHOLD,
):
    retrieved_texts = [get_node_text(node) for node in retrieved_nodes]

    if not retrieved_texts or not reference_texts:
        return {
            "soft_hit_rate": 0.0,
            "soft_precision": 0.0,
            "soft_recall": 0.0,
            "soft_mrr": 0.0,
            "soft_avg_retrieved_max_similarity": 0.0,
            "soft_avg_reference_max_similarity": 0.0,
            "soft_relevant_retrieved_count": 0,
            "soft_covered_reference_count": 0,
        }

    similarity_matrix = compute_similarity_matrix(
        retrieved_texts=retrieved_texts,
        reference_texts=reference_texts,
        embed_model=embed_model,
        embedding_cache=embedding_cache,
    )

    retrieved_best_scores = similarity_matrix.max(axis=1)
    reference_best_scores = similarity_matrix.max(axis=0)

    retrieved_relevance = retrieved_best_scores >= similarity_threshold
    reference_coverage = reference_best_scores >= similarity_threshold

    relevant_ranks = np.where(retrieved_relevance)[0]
    soft_mrr = 0.0
    if len(relevant_ranks) > 0:
        soft_mrr = 1.0 / float(relevant_ranks[0] + 1)

    return {
        "soft_hit_rate": float(retrieved_relevance.any()),
        "soft_precision": float(retrieved_relevance.mean()),
        "soft_recall": float(reference_coverage.mean()),
        "soft_mrr": soft_mrr,
        "soft_avg_retrieved_max_similarity": float(retrieved_best_scores.mean()),
        "soft_avg_reference_max_similarity": float(reference_best_scores.mean()),
        "soft_relevant_retrieved_count": int(retrieved_relevance.sum()),
        "soft_covered_reference_count": int(reference_coverage.sum()),
    }


def parse_llm_judge_response(response_text):
    result = {
        "llm_retrieval_score": None,
        "llm_retrieval_hit": None,
        "llm_retrieval_feedback": "",
    }

    for line in response_text.splitlines():
        if line.startswith("SCORE:"):
            try:
                result["llm_retrieval_score"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                result["llm_retrieval_score"] = None
        elif line.startswith("HIT:"):
            hit_value = line.split(":", 1)[1].strip().lower()
            if hit_value in {"yes", "y", "true", "1"}:
                result["llm_retrieval_hit"] = 1.0
            elif hit_value in {"no", "n", "false", "0"}:
                result["llm_retrieval_hit"] = 0.0
        elif line.startswith("FEEDBACK:"):
            result["llm_retrieval_feedback"] = line.split(":", 1)[1].strip()

    return result


def llm_judge_retrieval(query, retrieved_nodes, reference_texts, llm, top_n=LLM_JUDGE_TOP_N):
    retrieved_texts = [get_node_text(node) for node in retrieved_nodes[:top_n]]
    retrieved_context = "\n\n---\n\n".join(retrieved_texts)
    reference_context = "\n\n---\n\n".join(reference_texts)

    prompt = f"""
You are evaluating retrieval quality for a RAG system.

Question:
{query}

Retrieved contexts:
{retrieved_context}

Reference contexts:
{reference_context}

Evaluate whether the retrieved contexts contain the information needed to answer the question.

Return exactly:
SCORE: <0-5>
HIT: <yes/no>
FEEDBACK: <short explanation>
"""

    response = llm.complete(prompt)
    return parse_llm_judge_response(str(response))


def evaluate_retriever_variant(
    retriever,
    qa_dataset,
    variant_name,
    hard_metrics,
    embed_model,
    embedding_cache,
    llm_judge=None,
):
    results = []
    for query_id, query in qa_dataset.queries.items():
        # The expected node IDs for this question
        expected_ids = qa_dataset.relevant_docs[query_id]
        reference_texts = [
            qa_dataset.corpus[doc_id]
            for doc_id in expected_ids
            if doc_id in qa_dataset.corpus
        ]

        retrieved_nodes = retriever.retrieve(query)
        hard_metric_values = compute_hard_retrieval_metrics(
            retrieved_nodes=retrieved_nodes,
            expected_ids=expected_ids,
        )
        soft_metrics = compute_soft_retrieval_metrics(
            retrieved_nodes=retrieved_nodes,
            reference_texts=reference_texts,
            embed_model=embed_model,
            embedding_cache=embedding_cache,
        )
        
        print(f"Query: {query}")
        print(f"Metrics: {hard_metric_values}")
        row = {
            "retriever": variant_name,
            "query_id": query_id,
            "query": query,
            **{
                f"hard_{metric_name}": metric_value
                for metric_name, metric_value in hard_metric_values.items()
            },
            **soft_metrics,
        }

        if hasattr(retriever, "last_filter"):
            last_filter = retriever.last_filter
            row["inferred_filter"] = str(last_filter) if last_filter else ""

        if llm_judge is not None:
            row.update(
                llm_judge_retrieval(
                    query=query,
                    retrieved_nodes=retrieved_nodes,
                    reference_texts=reference_texts,
                    llm=llm_judge,
                )
            )

        results.append(row)

    return results


def run_retrieval_eval(index, qa_dataset, similarity_top_k=10):
    print("Running retrieval evaluation...")

    hard_metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
    soft_metrics = [
        "soft_hit_rate",
        "soft_precision",
        "soft_recall",
        "soft_mrr",
        "soft_avg_retrieved_max_similarity",
        "soft_avg_reference_max_similarity",
        "soft_relevant_retrieved_count",
        "soft_covered_reference_count",
    ]
    summary_metrics = [f"hard_{metric}" for metric in hard_metrics] + soft_metrics
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    embedding_cache = {}
    llm_judge = None

    if RUN_LLM_JUDGE:
        from generation import generation_model

        llm_judge = generation_model("judge")
        summary_metrics.extend(["llm_retrieval_score", "llm_retrieval_hit"])

    retriever = build_retriever_for_eval(index)
    all_results = evaluate_retriever_variant(
        retriever=retriever,
        qa_dataset=qa_dataset,
        variant_name="configured_rag",
        hard_metrics=hard_metrics,
        embed_model=embed_model,
        embedding_cache=embedding_cache,
        llm_judge=llm_judge,
    )

    full_df = pd.DataFrame(all_results)

    metric_df = (
        full_df.groupby("retriever", as_index=False)[summary_metrics]
        .mean()
        .sort_values("retriever")
    )

    print("\nAverage Metrics:")
    print(metric_df)

    metric_df.to_csv("data/retrieval_eval_results.csv", index=False)
    full_df.to_csv("data/retrieval_eval_results_per_query.csv", index=False)
        

def main():
    print("Loading vector database...")
    index = load_vector_db(
        embed_model=EMBED_MODEL,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    print("Loading cleaned evaluation dataset...")
    qa_dataset = EmbeddingQAFinetuneDataset.from_json("data/eval_rag_dataset_cleaned.json")

    print("Evaluating retrieval...")
    run_retrieval_eval(index, qa_dataset)

if __name__ == "__main__":
    main()
