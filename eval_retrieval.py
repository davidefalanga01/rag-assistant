'''Evaluate the Retrieval capabilities of the RAG assistant on a set of test queries.'''
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

from vector_database import load_vector_db

def run_retrieval_eval(index, qa_dataset, SIMILARITY_TOP_K=10):
    print("Running retrieval evaluation...")

    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]

    retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        metric_names=metrics,
        retriever=retriever,
    )
 
    results = []
    for query_id, query in qa_dataset.queries.items():
        # The expected node IDs for this question
        expected_ids = qa_dataset.relevant_docs[query_id]
        eval_result = retriever_evaluator.evaluate(
            query=query,
            expected_ids=expected_ids,
        )
        
        print(eval_result)
        break

def main():
    print("Loading vector database...")
    index = load_vector_db(
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        db_path="./chroma_db",
        collection_name="rag_collection",
    )

    print("Loading cleaned evaluation dataset...")
    qa_dataset = EmbeddingQAFinetuneDataset.from_json("data/eval_rag_dataset_cleaned.json")

    print("Evaluating retrieval...")
    run_retrieval_eval(index, qa_dataset)

if __name__ == "__main__":
    main()