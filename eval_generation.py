'''Evaluate the Generation capabilities of the RAG assistant on a set of test queries.'''
import json
from pathlib import Path

import pandas as pd
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    SemanticSimilarityEvaluator,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import EMBED_MODEL
from filter_gt import build_smaller_eval_dataset
from generation import generation_model
from rag import build_query_engine

SHORT_EVAL = True  # Set to True to run a shorter evaluation with fewer queries
DEFAULT_OUTPUT_FILE = "data/generation_eval_results.csv"

llm = generation_model()
llm_gt = generation_model("big")
llm_judge = generation_model("judge")


def metric_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "nan"}:
            return None
        if normalized in {"true", "yes", "y"}:
            return 1.0
        if normalized in {"false", "no", "n"}:
            return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_summary(results):
    if not results:
        return {}

    def avg(key):
        values = [
            value
            for value in (metric_value(result.get(key)) for result in results)
            if value is not None
        ]
        return sum(values) / len(values) if values else None

    return {
        "total_queries": len(results),
        "faithfulness_avg_score": avg("faithfulness_score"),
        "faithfulness_pass_rate": avg("faithfulness_passing"),
        "relevancy_avg_score": avg("relevancy_score"),
        "relevancy_pass_rate": avg("relevancy_passing"),
        "correctness_avg_score": avg("correctness_score"),
        "correctness_pass_rate": avg("correctness_passing"),
        "llm_judge_avg_score": avg("llm_judge_score"),
        "semantic_similarity_avg_score": avg("semantic_similarity_score"),
        "semantic_similarity_pass_rate": avg("semantic_similarity_passing"),
    }


def resolve_output_files(output_file=None):
    summary_file = Path(output_file or DEFAULT_OUTPUT_FILE)

    if summary_file.suffix.lower() == ".json":
        legacy_json_file = summary_file
        summary_file = summary_file.with_suffix(".csv")
    else:
        legacy_json_file = summary_file.with_suffix(".json")
        if summary_file.suffix.lower() != ".csv":
            summary_file = summary_file.with_suffix(".csv")

    per_query_file = summary_file.with_name(
        f"{summary_file.stem}_per_query{summary_file.suffix}"
    )

    return summary_file, per_query_file, legacy_json_file


def normalize_loaded_result(row):
    result = {}
    for key, value in row.items():
        try:
            is_missing = pd.isna(value)
        except (TypeError, ValueError):
            is_missing = False
        result[key] = None if is_missing else value

    # Backward compatibility with the previous JSON format, where this field
    # contained the boolean pass/fail value despite the "feedback" suffix.
    if "semantic_similarity_passing" not in result:
        semantic_feedback = result.get("semantic_similarity_feedback")
        semantic_passing = metric_value(semantic_feedback)
        if semantic_passing is not None and semantic_passing in {0.0, 1.0}:
            result["semantic_similarity_passing"] = bool(semantic_passing)
            result["semantic_similarity_feedback"] = None

    return result


def load_previous_results(per_query_file, legacy_json_file):
    if per_query_file.exists():
        print(f"Trovato file '{per_query_file}'. Ripristino sessione precedente...")
        saved_df = pd.read_csv(per_query_file)
        return [
            normalize_loaded_result(row)
            for row in saved_df.to_dict(orient="records")
        ]

    if legacy_json_file.exists():
        print(f"Trovato file legacy '{legacy_json_file}'. Ripristino sessione precedente...")
        with open(legacy_json_file, "r") as f:
            saved_data = json.load(f)
        return [
            normalize_loaded_result(row)
            for row in saved_data.get("results", [])
        ]

    return []


def save_generation_results(results, summary_file, per_query_file):
    summary = compute_summary(results)

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    per_query_file.parent.mkdir(parents=True, exist_ok=True)

    metric_df = pd.DataFrame([summary])
    full_df = pd.DataFrame(results)

    metric_df.to_csv(summary_file, index=False)
    full_df.to_csv(per_query_file, index=False)

    return {"results": results, "summary": summary}


def llm_as_a_judge(query, answer, reference):
    prompt = f"""
    Evaluate the answer.

    Query: {query}
    Answer: {answer}
    Reference: {reference}

    Score from 1 to 5 for correctness and explain.
    Respond in this exact format:
    SCORE: <number>
    FEEDBACK: <explanation>
    """

    response = llm_judge.complete(prompt, max_tokens=100)
    response_text = str(response)

    # Extract score and feedback
    score, feedback = None, None
    for line in response_text.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                print(f"Invalid score format for query {query}.")
        elif line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return {"llm_judge_score": score, "llm_judge_feedback": feedback}


def run_generation_eval(qa_dataset, output_file=None):
    print("Running generation evaluation...")
    summary_file, per_query_file, legacy_json_file = resolve_output_files(output_file)

    query_engine = build_query_engine(llm=llm)

    # Embedding model for semantic similarity evaluation
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Evaluators
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm_gt)
    relevancy_evaluator = RelevancyEvaluator(llm=llm_gt)
    correctness_evaluator = CorrectnessEvaluator(llm=llm_gt)
    semantic_similarity_evaluator = SemanticSimilarityEvaluator(embed_model=embed_model)

    results = []
    processed_queries = set()

    # Load previous results if available to allow resuming
    try:
        results = load_previous_results(per_query_file, legacy_json_file)
        processed_queries = {result["query"] for result in results if result.get("query")}
        if processed_queries:
            print(f"Resumed {len(processed_queries)} query gia valutate.")
    except (json.JSONDecodeError, pd.errors.ParserError):
        print("Il file esistente e corrotto. Inizio da zero.")

    for query_id, query in qa_dataset["queries"].items():
        if query in processed_queries:
            print(f"Skipping query: {query}")
            continue

        print(f"\nQuery: {query}")
        reference_answer = qa_dataset["responses"].get(query_id, None)

        try:
            # Generate response from RAG
            response = query_engine.query(query)
            generated_answer = str(response)

            print(f"Generated Answer: {generated_answer}")
            print(f"Reference Answer: {reference_answer}")

            # Evaluate
            faithfulness = faithfulness_evaluator.evaluate_response(
                response=response,
            )

            relevancy = relevancy_evaluator.evaluate_response(
                query=query,
                response=response,
            )

            correctness = None
            if reference_answer:
                correctness = correctness_evaluator.evaluate(
                    query=query,
                    response=generated_answer,
                    reference=reference_answer,
                )

            llm_judge_result = llm_as_a_judge(query, generated_answer, reference_answer)
            print(f"LLM Judge Result: {llm_judge_result}")

            semantic_similarity = semantic_similarity_evaluator.evaluate(
                response=generated_answer,
                reference=reference_answer,
            )

            result = {
                "query_id": query_id,
                "query": query,
                "generated_answer": generated_answer,
                "reference_answer": reference_answer,
                "faithfulness_passing": faithfulness.passing,
                "faithfulness_score": faithfulness.score,
                "faithfulness_feedback": faithfulness.feedback,
                "relevancy_passing": relevancy.passing,
                "relevancy_score": relevancy.score,
                "relevancy_feedback": relevancy.feedback,
                "correctness_passing": correctness.passing if correctness else None,
                "correctness_score": correctness.score if correctness else None,
                "correctness_feedback": correctness.feedback if correctness else None,
                "llm_judge_score": llm_judge_result.get("llm_judge_score"),
                "llm_judge_feedback": llm_judge_result.get("llm_judge_feedback"),
                "semantic_similarity_score": semantic_similarity.score,
                "semantic_similarity_passing": semantic_similarity.passing,
                "semantic_similarity_feedback": semantic_similarity.feedback,
            }

            results.append(result)
            save_generation_results(results, summary_file, per_query_file)

        except Exception as e:
            print(f"\n[ERRORE] Esecuzione interrotta sulla query: '{query}'.")
            print(f"Motivo: {e}")
            print("Salvataggio dei risultati parziali in corso...")

            partial_data = save_generation_results(results, summary_file, per_query_file)

            print(f"Salvataggio completato in {summary_file} e {per_query_file}.")
            return partial_data

    print("\nTutte le query completate con successo!")
    final_data = save_generation_results(results, summary_file, per_query_file)
    print(f"Risultati complessivi salvati in {summary_file}.")
    print(f"Risultati per query salvati in {per_query_file}.")

    return final_data


def main():
    print("Loading cleaned evaluation dataset...")
    with open("data/eval_rag_dataset_with_refs.json", "r") as f:
        qa_dataset = json.load(f)

    if SHORT_EVAL:
        print("Building smaller dataset for short evaluation...")
        qa_dataset = build_smaller_eval_dataset(qa_dataset)

    print("Evaluating generation...")
    result = run_generation_eval(qa_dataset, output_file=DEFAULT_OUTPUT_FILE)

    print("Process completed. Summary of results:")
    for key, value in result["summary"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
