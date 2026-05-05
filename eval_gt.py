'''Evaluate the Generation capabilities of the RAG assistant on a set of test queries.'''
import os

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

import json
from generation import generation_model

llm_gt = generation_model("big") 
llm_judge = generation_model("judge")

def compute_summary(results):
    if not results:
        return {}
        
    def avg(key):
        values = [r[key] for r in results if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "total_queries": len(results),
        "faithfulness_avg_score": avg("faithfulness_score"),
        "faithfulness_pass_rate": avg("faithfulness_passing"),
        "relevancy_avg_score": avg("relevancy_score"),
        "relevancy_pass_rate": avg("relevancy_passing"),
        "llm_judge_avg_score": avg("llm_judge_score"),}

def llm_as_a_judge(query, answer, context):
    prompt = f"""
    Evaluate the answer.

    Query: {query}
    Answer: {answer}
    Context: {context}

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

    # Evaluators
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm_gt)
    relevancy_evaluator = RelevancyEvaluator(llm=llm_gt)

    results = []
    processed_queries = set()

    # Load previous results if available to allow resuming
    if os.path.exists(output_file):
        print(f"Trovato file '{output_file}'. Ripristino sessione precedente...")
        try:
            with open(output_file, "r") as f:
                saved_data = json.load(f)
                results = saved_data.get("results", [])
                
                processed_queries = {r["query"] for r in results}
                print(f"Resumed {len(processed_queries)} query già valutate.")
        except json.JSONDecodeError:
            print("Il file esistente è corrotto. Inizio da zero.")

    relevant_docs = qa_dataset.get("relevant_docs", {})
    corpus = qa_dataset.get("corpus", {})

    for query_id, query in qa_dataset["queries"].items():
        if query in processed_queries:
            print(f"Skipping query: {query}")
            continue

        print(f"\nQuery: {query}")
        reference_answer = qa_dataset["responses"].get(query_id, None)
        
        rel_doc_ids = relevant_docs.get(query_id, [])
        context_list = [corpus[d_id] for d_id in rel_doc_ids if d_id in corpus]

        try:
            # Generate response from RAG
            print(f"Reference Answer: {reference_answer}")

            # Evaluate
            faithfulness = faithfulness_evaluator.evaluate(
                query=query,
                response=reference_answer,
                contexts=context_list
            )

            relevancy = relevancy_evaluator.evaluate(
                query=query,  
                response=reference_answer,
                contexts=context_list
            )

            llm_judge_result = llm_as_a_judge(query, reference_answer, context_list)
            print(f"LLM Judge Result: {llm_judge_result}")

            result = {
                "query": query,
                "faithfulness_passing": faithfulness.passing,
                "faithfulness_score": faithfulness.score,
                "relevancy_passing": relevancy.passing,
                "relevancy_score": relevancy.score,
                "llm_judge_score": llm_judge_result.get("llm_judge_score"),
                "llm_judge_feedback": llm_judge_result.get("llm_judge_feedback"),
            }

            results.append(result)

        except Exception as e:
            print(f"\n[ERRORE] Esecuzione interrotta sulla query: '{query}'.")
            print(f"Motivo: {e}")
            print("Salvataggio dei risultati parziali in corso per permettere il riavvio domani...")
            
            # Calcola il summary parziale e salva
            partial_summary = compute_summary(results)
            with open(output_file, "w") as f:
                json.dump({"results": results, "summary": partial_summary}, f, indent=2)
                
            print(f"Salvataggio completato in {output_file}. Uscita in corso.")
            
            # Restituisce i parziali in modo che il blocco `main()` funzioni senza crashare
            return {"results": results, "summary": partial_summary}
    
    # Se il ciclo finisce regolarmente su tutte le domande
    print("\nTutte le query completate con successo!")
    final_summary = compute_summary(results)
    final_data = {"results": results, "summary": final_summary}
    
    with open(output_file, "w") as f:
        json.dump(final_data, f, indent=2)

    return final_data


def main():

    print("Loading cleaned evaluation dataset...")
    with open("data/eval_rag_dataset_with_refs.json", "r") as f:
        qa_dataset = json.load(f)

    print("Evaluating generation...")
    result = run_generation_eval(qa_dataset, output_file="gt_eval_results.json")

    print("Process completed. Summary of results:")
    for key, value in result["summary"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

