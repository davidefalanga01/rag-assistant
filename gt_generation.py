import json
import os

from llama_index.core import PromptTemplate
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from generation import generation_model 

llm = generation_model("judge")  

def load_partial():
    if os.path.exists("data/eval_rag_dataset_with_refs.json"):
        with open("data/eval_rag_dataset_with_refs.json", "r") as f:
            dataset = json.load(f)
        responses = dataset.get("responses", {})
        print(f"[Resume] Loaded {len(responses)} responses from checkpoint")
        return responses
    return {}

def generate_reference_answers(qa_dataset):
    print("Generating reference answers for the dataset...")
    
    # Simple prompt to generate the truth from the context
    qa_prompt_tmpl = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    responses = load_partial()
    
    for query_id, query_str in qa_dataset.queries.items():
        if (query_id in responses) and (responses.get(query_id) != ""):
            continue

        # 1. Fetch the relevant documents (context) for this specific query
        relevant_doc_ids = qa_dataset.relevant_docs.get(query_id, [])
        contexts = [qa_dataset.corpus[doc_id] for doc_id in relevant_doc_ids]

        if not contexts:
            print(f"Skipping {query_id} (no context)")
            continue

        context_str = "\n\n".join(contexts)
        
        # 2. Format the prompt and ask the LLM to generate the answer
        prompt = qa_prompt_tmpl.format(context_str=context_str, query_str=query_str)
        
        try: 
            response = llm.complete(prompt, max_tokens=300)
        except Exception as e:
            print(f"Error generating answer for {query_id}: {e}")
            break
        
        answer = str(response).strip()

        # Retry if empty answer
        if not answer:
            print(f"Empty answer for {query_id}, retrying...")
            # retry semplice
            response = llm.complete(prompt)
            answer = str(response).strip()

        # 3. Store it
        responses[query_id] = answer
        print(f"Generated ref answer for {query_id}")
        print(f"Response: {response}")
        
        

    # Inject the responses back into the dataset object
    # (EmbeddingQAFinetuneDataset doesn't natively use a "responses" dict
    # but we can add it for our purposes)
    dataset_dict = {
    "queries": qa_dataset.queries,
    "corpus": qa_dataset.corpus,
    "relevant_docs": qa_dataset.relevant_docs,
    }

    dataset_dict["responses"] = responses 
    
    return dataset_dict

if __name__ == "__main__":
    # Load your dataset (make sure to use the same one as in eval_generation)
    qa_dataset = EmbeddingQAFinetuneDataset.from_json("data/eval_rag_dataset_cleaned.json")
    
    # Generate reference answers
    qa_dataset_with_refs = generate_reference_answers(qa_dataset)
    
    # Optionally, save the updated dataset with generated answers for later use
    with open("data/eval_rag_dataset_with_refs.json", "w") as f:
        json.dump(qa_dataset_with_refs, f, indent=2)

    print("Saved dataset with generated reference answers to data/eval_rag_dataset_with_refs.json")