from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import numpy as np

'''Clean the generated RAG evaluation dataset by removing question-context pairs with low-quality contexts.'''
import json

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Perform dataset cleaning when creating the vector store index
def clean_dataset(
    dataset_path="data/eval_rag_dataset.json",
    output_path="data/eval_rag_dataset_cleaned.json",
    quality_threshold=0.5,
    embed_model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)

    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    cleaned_queries = {}
    cleaned_corpus = dataset.corpus.copy()
    cleaned_relevant_docs = {}

    removed = 0

    for q_id, question in dataset.queries.items():
        relevant_doc_ids = dataset.relevant_docs.get(q_id, [])

        if not relevant_doc_ids:
            removed += 1
            continue

        # Compute embedding for question
        q_emb = embed_model.get_text_embedding(question)

        # Compute similarity with ALL relevant docs (usually 1, but safe)
        sims = []
        for doc_id in relevant_doc_ids:
            doc_text = dataset.corpus.get(doc_id, "")
            if not doc_text:
                continue

            d_emb = embed_model.get_text_embedding(doc_text)
            sim = cosine_similarity(q_emb, d_emb)
            sims.append(sim)

        # Skip if no valid similarities
        if not sims:
            removed += 1
            continue

        avg_sim = sum(sims) / len(sims)

        # Keep only high-quality pairs
        if avg_sim >= quality_threshold:
            cleaned_queries[q_id] = question
            cleaned_relevant_docs[q_id] = relevant_doc_ids
        else:
            removed += 1

    cleaned_dataset = EmbeddingQAFinetuneDataset(
        queries=cleaned_queries,
        corpus=cleaned_corpus,
        relevant_docs=cleaned_relevant_docs,
    )

    cleaned_dataset.save_json(output_path)

    print(f"Original queries: {len(dataset.queries)}")
    print(f"Kept queries: {len(cleaned_queries)}")
    print(f"Removed queries: {removed}")
    print(f"Saved cleaned dataset to: {output_path}")

if __name__ == "__main__":
    clean_dataset()
