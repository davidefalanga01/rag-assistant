"""Filter the GT evaluation dataset using perfect evaluation results."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "eval_rag_dataset_with_refs.json"
RESULTS_PATH = BASE_DIR / "gt_eval_results.json"
OUTPUT_PATH = BASE_DIR / "data" / "eval_rag_dataset_with_perfect_refs.json"

def build_smaller_eval_dataset(dataset: dict) -> dict:
    easy_query_ids = ["0c536442-b095-49d4-affc-cd6e1d49b1c5",
                      "f4797921-19a0-4542-800d-5fbe32ec6055",
                      "b398b6b8-24a8-4fa0-843d-5bcdadfccc5c",
                      "452df78b-66e6-4351-ad3d-bcb297910baf",
                      "d530c7d8-85ea-4eca-ab8c-d7fbcf26ea3a"]
    
    medium_query_ids = ["bfc971ba-cdd5-484f-bb70-4b168c1fe5ee",
                        "9ed56ee7-88e0-4207-a39c-dc76caac78ff",
                        "f8132a63-f218-42e1-a2c6-543aa5d891ff",
                        "9a06328c-0f7e-410b-a06b-4409cff03947",
                        "53303dc0-392c-4c21-8980-edf14c7e6eda"]
    
    hard_query_ids = ["2ef98597-17a8-4c7e-b00e-316d2fbe6225",
                      "dd913ed3-26bf-4037-abd6-3e4c6329a4e5",
                      "39dfd884-141d-4fbf-b1ea-8e99512b1190",
                      "be22b423-ff4c-4891-b016-4bf15814b98e",
                      "0665cba7-1f96-40a8-b8a6-6b487e1b29b8"]
    
    selected_query_ids = set(easy_query_ids + medium_query_ids + hard_query_ids)

    smaller_queries = {
        query_id: dataset["queries"][query_id]
        for query_id in selected_query_ids
    }

    smaller_relevant_docs = {
        query_id: dataset["relevant_docs"][query_id]
        for query_id in selected_query_ids
        if query_id in dataset["relevant_docs"]
    }

    smaller_corpus = {
        doc_id: dataset["corpus"][doc_id]
        for doc_id in set().union(*smaller_relevant_docs.values())
    }

    smaller_responses = {
        query_id: dataset["responses"][query_id]
        for query_id in selected_query_ids
        if query_id in dataset["responses"]
    }

    return {
        "queries": smaller_queries,
        "corpus": smaller_corpus,
        "relevant_docs": smaller_relevant_docs,
        "responses": smaller_responses,
    }


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_perfect_result(result: dict) -> bool:
    return (
        result.get("faithfulness_score") == 1
        and result.get("relevancy_score") == 1
        and result.get("llm_judge_score") == 5
    )


def build_filtered_dataset(dataset: dict, evaluation_results: dict) -> tuple[dict, dict]:
    perfect_query_counts = Counter(
        result["query"]
        for result in evaluation_results.get("results", [])
        if is_perfect_result(result)
    )

    filtered_queries = {}
    for query_id, query in dataset["queries"].items():
        if perfect_query_counts[query] <= 0:
            continue

        filtered_queries[query_id] = query
        perfect_query_counts[query] -= 1

    filtered_responses = {
        query_id: dataset["responses"][query_id]
        for query_id in filtered_queries
        if query_id in dataset["responses"]
    }

    filtered_relevant_docs = {
        query_id: dataset["relevant_docs"][query_id]
        for query_id in filtered_queries
        if query_id in dataset["relevant_docs"]
    }

    referenced_doc_ids = {
        doc_id
        for doc_ids in filtered_relevant_docs.values()
        for doc_id in doc_ids
    }
    filtered_corpus = {
        doc_id: doc_text
        for doc_id, doc_text in dataset["corpus"].items()
        if doc_id in referenced_doc_ids
    }

    filtered_dataset = {
        "queries": filtered_queries,
        "corpus": filtered_corpus,
        "relevant_docs": filtered_relevant_docs,
        "responses": filtered_responses,
    }

    stats = {
        "source_queries": len(dataset.get("queries", {})),
        "evaluated_queries": len(evaluation_results.get("results", [])),
        "perfect_results": sum(
            1
            for result in evaluation_results.get("results", [])
            if is_perfect_result(result)
        ),
        "retained_queries": len(filtered_queries),
        "source_corpus_docs": len(dataset.get("corpus", {})),
        "filtered_corpus_docs": len(filtered_corpus),
        "excluded_queries": len(dataset.get("queries", {})) - len(filtered_queries),
    }
    return filtered_dataset, stats


def main() -> None:
    dataset = load_json(DATASET_PATH)
    evaluation_results = load_json(RESULTS_PATH)

    filtered_dataset, stats = build_filtered_dataset(dataset, evaluation_results)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(filtered_dataset, file, indent=2, ensure_ascii=False)

    print(f"Saved filtered dataset to {OUTPUT_PATH}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
